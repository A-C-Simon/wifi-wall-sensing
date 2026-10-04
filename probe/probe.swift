// Decisive feasibility probe: how fast can we actually poll RSSI on macOS?
// Q1: is CWInterface.rssiValue readable at all without location permission?
// Q2: what is the real achievable poll rate?
// Q3: does it return FRESH values each call, or a frozen/cached constant?
import CoreWLAN
import Foundation

// Monotonic seconds, immune to Swift 6 Duration API churn.
@inline(__always)
func now() -> Double {
    Double(DispatchTime.now().uptimeNanoseconds) / 1e9
}

let ifaceName = "en0"
let client = CWWiFiClient.shared()
guard let iface = client.interface(withName: ifaceName) else {
    FileHandle.standardError.write("FATAL: no interface \(ifaceName)\n".data(using: .utf8)!)
    exit(2)
}

func qAt(_ a: [Double], _ p: Double) -> Double {
    a[min(a.count - 1, Int(p * Double(a.count - 1)))]
}

print("== Q1: single-shot capability ==")
print("rssiValue     : \(iface.rssiValue()) dBm")
print("noiseMeas     : \(iface.noiseMeasurement()) dBm")
print("transmitRate  : \(iface.transmitRate()) Mbps")
if let ch = iface.wlanChannel() {
    print("channel       : \(ch.channelNumber) band=\(ch.channelBand) width=\(ch.channelWidth)")
}
print("ssid          : \(iface.ssid() ?? "<nil: TCC redaction>")")
print("bssid         : \(iface.bssid() ?? "<nil: TCC redaction>")")

print("\n== Q2/Q3: tight-loop poll, 5000 samples, no sleep ==")
var samples: [Int] = []
var stamps: [Double] = []
let t0 = now()
for i in 0..<5000 {
    samples.append(iface.rssiValue())
    stamps.append(now() - t0)
    if i % 1000 == 0 {
        print(String(format: "  t=%7.3fs iter=%5d rssi=%3d rate=%.1f Hz",
                     stamps[i], i, samples[i], Double(i + 1) / max(stamps[i], 1e-9)))
    }
}
let wall = now() - t0
print(String(format: "\nRAW RATE      : %.1f samples/sec (%d samples in %.3fs)", 5000.0 / wall, 5000, wall))

let uniq = Set(samples).sorted()
var sum = 0.0
for v in samples { sum += Double(v) }
let mean = sum / Double(samples.count)
var acc = 0.0
for v in samples { let d = Double(v) - mean; acc += d * d }
let sd = sqrt(acc / Double(samples.count))

print("DISTINCT      : \(uniq.count) values -> \(uniq.map { String($0) }.joined(separator: ","))")
print(String(format: "RANGE         : min=%d max=%d mean=%.3f sd=%.4f dB",
             samples.min()!, samples.max()!, mean, sd))

var transitions = 0
for i in 1..<samples.count where samples[i] != samples[i - 1] { transitions += 1 }
print(String(format: "TRANSITIONS   : %d/%d (%.1f%%)  <- >0%% = live register, 0%% = frozen",
             transitions, samples.count - 1,
             100.0 * Double(transitions) / Double(samples.count - 1)))
print(String(format: "LATENCY  p50/p99: %.3f / %.3f ms", qAt(stamps.sorted(), 0.50), qAt(stamps.sorted(), 0.99)))

print("\n== Q4: multi-AP scan rate ==")
for attempt in 1...3 {
    let s = now()
    do {
        let nets: Set<CWNetwork> = try iface.scanForNetworks(withName: nil, includeHidden: false)
        print(String(format: "  scan %d: %d nets in %.3fs", attempt, nets.count, now() - s))
        if attempt == 1 {
            let sorted = nets.sorted { (a: CWNetwork, b: CWNetwork) -> Bool in
                a.rssiValue > b.rssiValue
            }
            var rows: [String] = []
            for n in sorted {
                let ch = n.wlanChannel
                rows.append("      ch=\(ch.map { "\($0.channelNumber)" } ?? "?") "
                    + "width=\(ch.map { "\($0.channelWidth)" } ?? "?") "
                    + "rssi=\(n.rssiValue) noise=\(n.noiseMeasurement)")
            }
            print(rows.joined(separator: "\n"))
        }
    } catch {
        print(String(format: "  scan %d: FAILED in %.3fs: %@", attempt, now() - s, "\(error)"))
    }
}

print("\n== Q5: cachedScanResults read rate (no radio activity) ==")
let s2 = now()
var counts: [Int] = []
for _ in 0..<20 {
    counts.append(iface.cachedScanResults()?.count ?? -1)
    usleep(50_000)
}
let el2 = now() - s2
print(String(format: "  20 reads in %.3fs (%.0f Hz), counts: %@",
             el2, 20.0 / el2, counts.map { String($0) }.joined(separator: ",")))