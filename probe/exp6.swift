// Experiment 6: per-call cost of every CoreWLAN accessor.
// transmitRate() appeared to block in a tight loop; measure each accessor
// individually so we know which are actually fast-pollable.
import CoreWLAN
import Foundation
setvbuf(stdout, nil, _IOLBF, 0)
@inline(__always) func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
let ifaceName = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "en0"
let iface = CWWiFiClient.shared().interface(withName: ifaceName)!

func bench(_ label: String, _ n: Int, _ f: () -> Int) {
    var v: [Int] = []
    var worst = 0.0
    let t = now()
    for _ in 0..<n {
        let a = now()
        v.append(f())
        let d = now() - a
        if d > worst { worst = d }
    }
    let wall = now() - t
    var tr = 0
    for i in 1..<v.count where v[i] != v[i - 1] { tr += 1 }
    print(label + String(format: " n=%4d total=%6.3fs  %7.1f calls/s  mean=%6.2fms worst=%7.2fms  distinct=%2d changes=%3d",
                 n, wall, Double(n)/wall, wall/Double(n)*1000,
                 worst*1000, Set(v).count, tr))
    print("      values: \(Set(v).sorted().prefix(16).map(String.init).joined(separator: ","))")
}

print("=== per-call cost, 300 iterations each ===")
bench("rssiValue     ", 300) { iface.rssiValue() }
bench("noiseMeasure  ", 300) { iface.noiseMeasurement() }
print("\n--- transmitRate (Double, first call then timed) ---")
let first = iface.transmitRate()
print("  first value: \(first)")
var worstTR = 0.0
var vals: [Double] = []
let t0 = now()
for i in 0..<300 {
    let a = now()
    let x = iface.transmitRate()
    let d = now() - a
    if d > worstTR { worstTR = d }
    vals.append(x)
    if i % 50 == 0 { print(String(format: "   call %3d: %8.1f  %.2fms", i, x, d*1000)) }
}
let wallTR = now() - t0
print(String(format: "  transmitRate n=300 total=%.3fs  %.1f calls/s  mean=%.2fms worst=%.2fms",
             wallTR, 300/wallTR, wallTR/300*1000, worstTR*1000))
print("  distinct rate values: \(Set(vals).sorted().map { String(Int($0)) }.joined(separator: ","))")
