// =========================================================================
//  The phone app's golden harness (Step 3 brief §4.7).
//
//  Compiled with the app's own Core sources, unmodified, from mesquite-mocap
//  87d7c3b (new_device_code/phone_app/MesquiteHips/Sources/MesquiteHips/Core),
//  by tools/app_golden/make_fixtures.py. It reads one JSON file of scripted
//  inputs and prints what the real code does with them. Every double crosses
//  the boundary as its IEEE bit pattern in hexadecimal, so nothing is lost.
//
//    cases:   PosePipeline runs. Each frame is
//             [t, px, py, pz, qx, qy, qz, qw, tracking, zero request 0/1,
//              link accepts 0/1]; the harness calls requestZero() before
//             process() when asked, and markSent(t) after a message the link
//             accepts, as App/PhoneController.swift does.
//    encode:  HubMessage.encode on [qx, qy, qz, qw, px, py, pz, has position].
//    columns: Quat(columns:) on nine numbers, column by column.
//    lever:   LeverArmFit runs: a prior and [px, py, pz, qx, qy, qz, qw] samples.
// =========================================================================
import Foundation

func d(_ a: Any) -> Double { Double(bitPattern: UInt64(a as! String, radix: 16)!) }
func h(_ v: Double) -> String { String(v.bitPattern, radix: 16) }
func hv(_ v: Vec3) -> [String] { [h(v.x), h(v.y), h(v.z)] }
func hq(_ q: Quat) -> [String] { [h(q.x), h(q.y), h(q.z), h(q.w)] }

func tracking(_ s: String) -> Tracking {
    switch s {
    case "normal": return .normal
    case "not_available": return .notAvailable
    case "limited:initializing": return .limited(.initializing)
    case "limited:excessiveMotion": return .limited(.excessiveMotion)
    case "limited:insufficientFeatures": return .limited(.insufficientFeatures)
    case "limited:relocalizing": return .limited(.relocalizing)
    default: return .limited(.other)
    }
}

let input = try! Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
let root = try! JSONSerialization.jsonObject(with: input) as! [String: Any]
var result: [String: Any] = [:]

var casesOut: [[String: Any]] = []
for c in (root["cases"] as? [[String: Any]] ?? []) {
    let s = c["settings"] as! [String: Any]
    let lv = s["lever"] as! [Any]
    let frame = HipsFrame(lever: Vec3(d(lv[0]), d(lv[1]), d(lv[2])), hipHeight: d(s["hipHeight"]!),
                          facing: CameraFacing(rawValue: s["facing"] as! String)!)
    var p = PosePipeline(frame: frame, tree: PageTree(rawValue: s["tree"] as! String)!, rateHz: d(s["rateHz"]!))
    var rows: [Any] = []
    for fr in c["frames"] as! [[Any]] {
        let f = FrameInput(time: d(fr[0]), camPosition: Vec3(d(fr[1]), d(fr[2]), d(fr[3])),
                           camRotation: Quat(d(fr[4]), d(fr[5]), d(fr[6]), d(fr[7])),
                           tracking: tracking(fr[8] as! String))
        if (fr[9] as! Int) == 1 { p.requestZero() }
        let o = p.process(f)
        var row: [Any] = []
        if let r = o.reported { row.append(hv(r.position) + hq(r.orientation)) } else { row.append(NSNull()) }
        if let m = o.message { row.append(m.text); row.append(m.level) } else { row.append(NSNull()); row.append(NSNull()) }
        if let z = o.zero { row.append([z.epoch, h(z.heading), o.zeroWasRequested ? 1 : 0]) } else { row.append(NSNull()) }
        if let e = o.zeroError { row.append("\(e)") } else { row.append(NSNull()) }
        if o.message != nil && (fr[10] as! Int) == 1 { p.markSent(f.time) }
        rows.append(row)
    }
    casesOut.append(["name": c["name"]!, "out": rows])
}
result["cases"] = casesOut

var encOut: [Any] = []
for e in (root["encode"] as? [[Any]] ?? []) {
    let q = Quat(d(e[0]), d(e[1]), d(e[2]), d(e[3]))
    let pos: Vec3? = (e[7] as! Int) == 1 ? Vec3(d(e[4]), d(e[5]), d(e[6])) : nil
    if let m = HubMessage.encode(orientation: q, position: pos) { encOut.append([m.text, m.level]) } else { encOut.append(NSNull()) }
}
result["encode"] = encOut

var colOut: [Any] = []
for m in (root["columns"] as? [[Any]] ?? []) {
    let q = Quat(columns: Vec3(d(m[0]), d(m[1]), d(m[2])), Vec3(d(m[3]), d(m[4]), d(m[5])), Vec3(d(m[6]), d(m[7]), d(m[8])))
    colOut.append(hq(q))
}
result["columns"] = colOut

var leverOut: [Any] = []
for l in (root["lever"] as? [[String: Any]] ?? []) {
    let pr = l["prior"] as! [Any]
    var fit = LeverArmFit(prior: Vec3(d(pr[0]), d(pr[1]), d(pr[2])))
    for s in l["samples"] as! [[Any]] {
        fit.add(camPosition: Vec3(d(s[0]), d(s[1]), d(s[2])), camRotation: Quat(d(s[3]), d(s[4]), d(s[5]), d(s[6])))
    }
    switch fit.solve() {
    case .success(let r):
        leverOut.append(["ok": 1, "lever": hv(r.lever), "pivot": hv(r.pivot), "rms": h(r.rms),
                         "headingRange": h(r.headingRange), "samples": r.samples])
    case .failure(let f):
        switch f {
        case .tooFewSamples(let n): leverOut.append(["ok": 0, "failure": "tooFewSamples", "value": h(Double(n))])
        case .notEnoughTurn(let deg): leverOut.append(["ok": 0, "failure": "notEnoughTurn", "value": h(deg)])
        case .implausibleLever(let m): leverOut.append(["ok": 0, "failure": "implausibleLever", "value": h(m)])
        case .poorFit(let rms): leverOut.append(["ok": 0, "failure": "poorFit", "value": h(rms)])
        }
    }
}
result["lever"] = leverOut

let out = try! JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
FileHandle.standardOutput.write(out)
