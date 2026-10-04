// Experiment 7: can we beat the measured 12.1 s scan-cache refresh?
//
//   S1 baseline (includeHidden:false)
//   S2 includeHidden:true  -> forces active probe requests, may yield fresher
//                             and more APs (hidden SSID networks respond)
//   S3 three concurrent clients -> is the refresh limiter per-process or global?
//
// Uses only string interpolation: String(format:) with %@ and bridged objects
// crashed repeatedly on this SDK.
import CoreWLAN
import Foundation
setvbuf(stdout, nil, _IOLBF, 0)

@inline(__always) func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

let iface = CWWiFiClient.shared().interface(withName: "en0")!

/// Canonical, order-independent identity for a scan result set.
/// BSSID is TCC-redacted so we key on (channel,width,rssi) triples.
func vectorKey(_ hidden: Bool) -> String {
    guard let nets = try? iface.scanForNetworks(withName: nil, includeHidden: hidden) else {
        return ""
    }
    var rows: [String] = []
    for n in nets {
        let ch = n.wlanChannel
        rows.append("\(ch?.channelNumber ?? -1)/\(ch?.channelWidth.rawValue ?? -1)=\(n.rssiValue)")
    }
    rows.sort()
    return rows.joined(separator: ",")
}

func measure(_ label: String, _ dur: Double, _ hidden: Bool) {
    var last = ""
    var n = 0
    var maxAPs = 0
    var intervals: [Double] = []
    var lastChange = now()
    var callCosts: [Double] = []
    let t0 = now()
    while now() - t0 < dur {
        let c0 = now()
        let k = vectorKey(hidden)
        callCosts.append(now() - c0)
        if !k.isEmpty {
            maxAPs = max(maxAPs, k.split(separator: ",").count)
            if k != last {
                if n > 0 { intervals.append(now() - lastChange) }
                last = k
                n += 1
                lastChange = now()
            }
        }
        usleep(100_000)
    }
    let mean = intervals.isEmpty ? 0 : intervals.reduce(0, +) / Double(intervals.count)
    let avgCost = callCosts.isEmpty ? 0 : callCosts.reduce(0, +) / Double(callCosts.count)
    let ivMin = intervals.min() ?? 0
    let ivMax = intervals.max() ?? 0
    let hzMilli = mean > 0 ? Int(1000.0 / mean) : 0
    print("\(label) fresh=\(n)  maxAPs=\(maxAPs)  avgCall=\(Int(avgCost * 1000))ms")
    print("   interval mean=\(mean)  min=\(ivMin)  max=\(ivMax)  => \(hzMilli)/1000 Hz")
}

print("=== S1: baseline includeHidden:false, 30s ===")
measure("  S1", 30, false)

print("\n=== S2: includeHidden:true (active probes), 30s ===")
measure("  S2", 30, true)

print("\n=== S3: 3 concurrent clients polling, 20s ===")
let t0 = now()
let lock = NSLock()
var totals = [Int](repeating: 0, count: 3)
let group = DispatchGroup()
for idx in 0..<3 {
    DispatchQueue.global().async(group: group) {
        var last = ""
        var n = 0
        while now() - t0 < 20 {
            let k = vectorKey(false)
            if !k.isEmpty && k != last { n += 1; last = k }
            usleep(100_000)
        }
        lock.lock(); totals[idx] = n; lock.unlock()
    }
}
group.wait()
print("  fresh vectors per client: \(totals)  total=\(totals.reduce(0, +)) in 20s")
print("  (if totals are equal -> the limiter is GLOBAL, not per-process)")