// The page golden test (Step 3 brief §4.10): the page's own pose code, unmodified, in Node.
//
//   node tools/page_golden/run.mjs <input.json> <output.json> <mesquite-mocap folder at 87d7c3b>
//
// It loads three.js r157 and its GLTFLoader from build-static/, the avatar glbs/scene2.glb (prepared as
// js/threejsscene.js:433-470 does), the tree trees/<id>/meta.json and the slerpDict literal of index.html. Then it
// runs js/custom_icm.js and js/bvh_converter.js UNMODIFIED in a vm context, with inert stubs for the DOM, jQuery,
// Materialize's toasts, moment, KalmanFilter and the timers, and a scripted clock (Date and performance.now). It
// replays the input's events in order: handleWSMessage(msg), calibrate(), and the recorder of index.html:373-431
// (startRecording, one setInterval tick, stopRecording), which this file mirrors because it lives in the page's
// inline script. After each message it writes the drawn local quaternion of the message's bone (and, for the
// phone's bone Hips, the hips position); at the end the BVH text and the drawn state of every pod bone.
//
// mesqsim/twin/page.py must reproduce these numbers (tests/test_page_twin.py).

import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { pathToFileURL } from 'node:url';

const [inputPath, outputPath, mocapDir] = process.argv.slice(2);
if (!mocapDir) {
  console.error('usage: node run.mjs <input.json> <output.json> <mesquite-mocap folder>');
  process.exit(2);
}
const lib = (f) => pathToFileURL(path.join(mocapDir, 'build-static', f)).href;
const THREE = await import(lib('three.module.js'));
const { GLTFLoader } = await import(lib('GLTFLoader.js'));
const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));

// --- the avatar, as js/threejsscene.js:433-470 (materials and shadows left out: they do not move bones)
const glb = fs.readFileSync(path.join(mocapDir, 'glbs', 'scene2.glb'));
const gltf = await new Promise((resolve, reject) =>
  new GLTFLoader().parse(glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength), '', resolve, reject));
const model = gltf.scene;
model.scale.set(0.00001, 0.00001, 0.00001);
model.traverse((child) => {
  if (child.name === 'mmHead') child.traverse((c) => c.scale.set(0.9, 0.85, 0.67));
});

// --- the scripted clock
const clock = { t: 0 };
const EPOCH = input.epoch_ms;
const RealDate = Date;
class FakeDate extends RealDate {
  constructor(...a) { if (a.length === 0) super(EPOCH + clock.t); else super(...a); }
  static now() { return new RealDate(EPOCH + clock.t).getTime(); }
}
const perf = { now: () => clock.t };

// --- inert stubs
const element = () => ({ innerHTML: '', textContent: '', dataset: {}, style: {}, value: '',
  classList: { add() {}, remove() {}, contains() { return false; } }, querySelector() { return element(); },
  appendChild() {}, addEventListener() {} });
const bodyClasses = new Set();
const jq = (selector) => {
  const o = {
    length: 0, each() { return o; }, attr() { return o; }, html() { return o; }, text() { return o; },
    css() { return o; }, fadeIn() { return o; }, fadeOut() { return o; }, remove() { return o; }, show() { return o; },
    hide() { return o; }, removeClass() { return o; },
    addClass(c) { if (selector === 'body') bodyClasses.add(c); return o; },
    hasClass(c) { return selector === 'body' && bodyClasses.has(c); },
  };
  return o;
};
const trees = { [input.tree]: JSON.parse(fs.readFileSync(path.join(mocapDir, 'trees', input.tree, 'meta.json'), 'utf8')) };
const ctx = {
  THREE, model, console, Date: FakeDate, performance: perf,
  document: { getElementById: element, querySelector: element, querySelectorAll: () => [], createElement: element,
    body: element() },
  $: jq, jQuery: jq,
  M: { toast() {}, Toast: { dismissAll() {} }, FormSelect: { init() {} }, Modal: { init() { return []; } } },
  moment: () => ({ fromNow: () => 'a few seconds' }),
  KalmanFilter: class { filter(v) { return v; } },
  localStorage: { getItem: () => null, setItem() {} },
  setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
  statsObjs: new Proxy({}, { get: () => ({ update() {} }) }),
  trees, treeType: input.tree, mac2Bones: {},
  line_tracker: [], trackingLine: new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial()),
  facemesh: { scale: { set() {} } }, circle: { scale: { set() {} } }, circle2: { scale: { set() {} } },
  recording: false, recordedMotionData: [], firstFrameJointData: null,
};
ctx.window = ctx;
ctx.self = ctx;
vm.createContext(ctx);

const html = fs.readFileSync(path.join(mocapDir, 'index.html'), 'utf8');
const slerpSource = html.match(/var slerpDict = \{[\s\S]*?\};/)[0];
vm.runInContext(slerpSource, ctx, { filename: 'index.html (slerpDict)' });
for (const f of ['js/custom_icm.js', 'js/bvh_converter.js']) {
  vm.runInContext(fs.readFileSync(path.join(mocapDir, f), 'utf8'), ctx, { filename: f });
}

// --- the recorder of index.html:373-431
function startRecording() {
  ctx.recording = true;
  ctx.recordedMotionData = [];
  ctx.mesqFrameTiming = { frames: 0, startMs: perf.now(), endMs: perf.now(), startISO: new FakeDate().toISOString(),
    endISO: null };
  ctx.firstFrameJointData = ctx.updateMotionData();
}
function tick() {
  if (ctx.recording) {
    ctx.updateMotionData();
    ctx.mesqFrameTiming.frames++;
    ctx.mesqFrameTiming.endMs = perf.now();
  }
}
function stopRecording() {
  ctx.recording = false;
  ctx.mesqFrameTiming.endISO = new FakeDate().toISOString();
  ctx.mesqFrameTiming.frames = ctx.recordedMotionData.length;
  return ctx.generateBVH(ctx.firstFrameJointData, ctx.recordedMotionData);
}

// --- replay
const out = { per_event: [], threw: [], bvh: null };
const logged = console.log;
console.log = () => {};  // generateBVH logs the joints and the frame count
for (const [i, ev] of input.events.entries()) {
  clock.t = ev.t_ms;
  if (ev.type === 'msg') {
    try {
      ctx.handleWSMessage(Object.assign({}, ev.msg));
    } catch (e) {
      out.threw.push([i, String(e)]);
    }
    const b = model.getObjectByName('mm' + ev.msg.bone.replace('Alt', ''));
    const rec = [b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w];
    if (ev.msg.bone === 'Hips') {
      const h = model.getObjectByName('mmHips').position;
      rec.push(h.x, h.y, h.z);
    }
    out.per_event.push(rec);
  } else {
    if (ev.type === 'calibrate') ctx.calibrate();
    else if (ev.type === 'record_start') startRecording();
    else if (ev.type === 'record_tick') tick();
    else if (ev.type === 'record_stop') out.bvh = stopRecording();
    out.per_event.push(null);
  }
}
console.log = logged;
const finalState = {};
for (const name of ['Hips', 'Spine', 'Head', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm',
  'RightHand', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot']) {
  const q = model.getObjectByName('mm' + name).quaternion;
  finalState[name] = [q.x, q.y, q.z, q.w];
}
out.final_state = finalState;
out.final_hips_position = model.getObjectByName('mmHips').position.toArray();
out.provenance = { node: process.version, three_revision: THREE.REVISION, slerp: ctx.slerpDict, tree: trees[input.tree] };
fs.writeFileSync(outputPath, JSON.stringify(out));
