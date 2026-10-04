// Experiment 4: what is the TRUE refresh rate of the multi-AP vector?
//
// Polling scanForNetworks() 5x/sec and hashing the vector tells us when macOS
// actually performs a new sweep vs. replays its cache. This number is the
// time resolution of the whole sensing system, so it must be measured, not
// assumed.
//
// Usage: exp4 [secs] [iface]
import CoreWLAN
import Foundation

@inline(__always)
func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

let secs = CommandLine.arguments.count > 1 ? (Double(CommandLine.arguments[1]) ?? 90.0) : 90.0
let ifaceName = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "en0"
let iface = CWWiFiClient.shared().interface(withName: ifaceName)!

func vector() -> ([String], Double) {
    let t = now()
    guard let nets = try? iface.scanForNetworks(withName: nil, includeHidden: false) else {
        return ([], now() - t)
    }
    // Stable per-AP key: BSSID is TCC-redacted, so use (channel,width) which is
    // stable for a given AP within a sweep.
    let rows = nets.map { n -> String in
        let ch = n.wlanChannel
        return "\(ch?.channelNumber ?? -1)/\(ch?.channelWidth.rawValue ?? -1)=\(n.rssiValue)"
    }
    return (rows, now() - t)
}

print("polling scanForNetworks() every 200ms for \(Int(secs))s\n")

var lastKey = ""
var changeTimes: [Double] = []
var callTimes: [Double] = []
var counts: [Int] = []
var printed = 0
let t0 = now()

while now() - t0 < secs {
    let (rows, ct) = vector()
    callTimes.append(ct)
    counts.append(rows.count)
    if rows.isEmpty {
        usleep(200_000); continue
    }
    // Sort so AP ordering noise doesn't register as a change.
    let key = rows.sorted().joined(separator: ",")
    if key != lastKey {
        let t = now() - t0
        if printed < 3 || key != lastKey {
            print(String(format: "  t=%6.2fs  change  n=%2d  scanCost=%5.1fms  %@",
                         t, rows.count, ct * 1000,
                         rows.sorted().prefix(6).joined(separator: " ")))
        }
        changeTimes.append(t)
        lastKey = key
        printed += 1
    }
    usleep(200_000)
}

print("\n--- SUMMARY ---")
print("polls                : \(callTimes.count)")
print("distinct vectors     : \(changeTimes.count)")
var deltas: [Double] = []
for i in 1..<max(changeTimes.count, 1) { deltas.append(changeTimes[i] - changeTimes[i - 1]) }
if !deltas.isEmpty {
    let sorted = deltas.sorted()
    let mean = deltas.reduce(0, +) / Double(deltas.count)
    print("refresh interval mean: \(String(format: "%.2f", mean))s")
    print("refresh interval min : \(String(format: "%.2f", sorted.first!))s")
    print("refresh interval max : \(String(format: "%.2f", sorted.last!))s")
    print("=> effective multi-AP sampling rate: \(String(format: "%.2f Hz", 1.0 / mean)) Hz")
}
let avgCost = callTimes.reduce(0, +) / Double(callTimes.count)
let cnts = Set(counts).sorted()
print("avg scan call cost   : \(String(format: "%.1f", avgCost * 1000))ms")
print("AP count seen        : \(cnts.map(String.init).joined(separator: ","))")