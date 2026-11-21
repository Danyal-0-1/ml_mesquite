// Open a BVH file with the page's own loader (Step 3 brief, S3-M5: "a twin BVH opens in the page's own BVHLoader").
//
//   node tools/page_golden/load_bvh.mjs <file.bvh> <mesquite-mocap dir>
//
// Imports BVHLoader from the clone's build-static/ (as js/threejssceneplayer.js:6 does; it imports
// ./three.module.js, r157) and reports three things as one JSON line:
//
// 1. `new BVHLoader()`, as the player does (:332). At 87d7c3b it throws whatever the file: BVHLoader.js is a
//    pre-class loader calling `Loader.call(this, manager)`, and r157's Loader is a class. The script then uses an
//    object holding exactly what the constructor sets (parse reads nothing else), so the parser is the page's own.
// 2. `parse(text)` on the file as it is. The page writes three `;` provenance lines after `Frame Time:`
//    (js/bvh_converter.js), which BVH has no syntax for: the loader reads them as frames and runs out of numbers.
// 3. `parse(text)` with the lines that start with `;` removed: the bones, the clip, the keyframes, all finite?
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const [file, mocapDir] = process.argv.slice(2);
const { BVHLoader } = await import(pathToFileURL(path.join(mocapDir, 'build-static', 'BVHLoader.js')).href);
const text = fs.readFileSync(file, 'utf8');
const out = { constructed: true, construct_error: null, parsed_as_is: true, parse_error: null };
let loader;
try {
  loader = new BVHLoader();
} catch (e) {
  out.constructed = false;
  out.construct_error = String(e && e.message);
  loader = Object.create(BVHLoader.prototype);
  loader.animateBonePositions = true; // what the constructor sets (BVHLoader.js:23-24)
  loader.animateBoneRotations = true;
}
try {
  loader.parse(text);
} catch (e) {
  out.parsed_as_is = false;
  out.parse_error = String(e && e.message);
}
const stripped = text.split('\n').filter((l) => !l.startsWith(';')).join('\n');
const result = loader.parse(stripped);
const tracks = result.clip.tracks;
let finite = true;
for (const tr of tracks) {
  for (const v of tr.values) if (!Number.isFinite(v)) { finite = false; break; }
  for (const t of tr.times) if (!Number.isFinite(t)) { finite = false; break; }
}
Object.assign(out, {
  comment_lines: text.split('\n').length - stripped.split('\n').length,
  bones: result.skeleton.bones.length,
  names: result.skeleton.bones.map((b) => b.name),
  duration_s: result.clip.duration,
  tracks: tracks.length,
  keyframes: tracks.length ? tracks[0].times.length : 0,
  finite,
});
console.log(JSON.stringify(out));
