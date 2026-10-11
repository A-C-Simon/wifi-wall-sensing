// Experiment 3: does forced traffic increase the RSSI signal's usable bandwidth?
//
// The register updates when the radio receives frames. Beacons alone give a
// low update rate; adding data traffic should raise it. If transition rate
// goes up with traffic, we can average down the 1 dB quantisation noise and
// resolve sub-dB motion signatures.
//
// Usage: exp3 <gateway> [iface]
import CoreWLAN
import Foundation

@inline(__always)
func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

let gw = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "192.168.3.1"
let ifaceName = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "en0"
let iface = CWWiFiClient.shared().interface(withName: ifaceName)!
print("iface=\(ifaceName) gateway=\(gw)\n")

/// Spawns parallel `ping` streams. ICMP is used rather than a raw UDP socket
/// because a VPN utun interface owns the default route on this host, which
/// makes connect()+send() to the LAN gateway fail with EADDRNOTAVAIL.
func startPings(_ gw: String, _ n: Int) -> [Process] {
    var procs: [Process] = []
    for _ in 0..<n {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/sbin/ping")
        p.arguments = ["-i", "0.2", gw]
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        do {
            try p.run()
            procs.append(p)
        } catch {
            for running in procs {
                if running.isRunning { running.terminate() }
                running.waitUntilExit()
            }
            FileHandle.standardError.write(Data("Could not start ping: \(error)\n".utf8))
            exit(2)
        }
    }
    return procs
}

func capture(_ secs: Double) -> ([Double], [Int]) {
    var t: [Double] = []
    var r: [Int] = []
    let t0 = now()
    while now() - t0 < secs {
        let v = iface.rssiValue()
        t.append(now() - t0)
        r.append(v)
    }
    return (t, r)
}

func report(_ label: String, _ t: [Double], _ r: [Int]) {
    let n = r.count
    guard n > 2, let wall = t.last else { return }
    var sum = 0.0; for v in r { sum += Double(v) }
    let mean = sum / Double(n)
    var acc = 0.0; for v in r { let d = Double(v) - mean; acc += d * d }
    let sd = sqrt(acc / Double(n))
    var tr = 0
    for i in 1..<n where r[i] != r[i - 1] { tr += 1 }
    let span = Double(r.max()! - r.min()!)
    // Transitions per second is the real "how fast can we see motion" number.
    let trHz = Double(tr) / wall
    print(String(format: "%-22s n=%5d  %.1fHz  distinct=%2d  span=%ddB  sd=%.3f  trans=%4d (%.2f Hz)",
                 (label as NSString).utf8String!, n, Double(n) / wall,
                 Set(r).count, Int(span), sd, tr, trHz))
    var h = [Int: Int]()
    for v in r { h[v, default: 0] += 1 }
    print("      hist: " + h.sorted { $0.key < $1.key }.map { "\($0.key):\($0.value)" }.joined(separator: " "))
}

print("=== baseline: NO added traffic, 20s ===")
let (t0a, r0a) = capture(20)
report("  quiet", t0a, r0a)

print("\n=== with 8 parallel ping streams, 20s ===")
let pings = startPings(gw, 8)
usleep(1_000_000)
let (t1a, r1a) = capture(20)
report("  8x ping", t1a, r1a)

for p in pings {
    if p.isRunning { p.terminate() }
    p.waitUntilExit()
    guard p.terminationStatus == 0 || p.terminationReason == .uncaughtSignal else {
        FileHandle.standardError.write(Data("Ping failed; traffic comparison is invalid\n".utf8))
        exit(2)
    }
}
usleep(1_500_000)

print("\n=== back to quiet, 20s (recovery check) ===")
let (t2a, r2a) = capture(20)
report("  quiet again", t2a, r2a)

print("\n=== multi-AP scan freshness: 12 scans @250ms ===")
var vectors: [String] = []
for i in 0..<12 {
    let st = now()
    if let nets = try? iface.scanForNetworks(withName: nil, includeHidden: false) {
        vectors.append(nets.map { String($0.rssiValue) }.sorted().joined(separator: ","))
        let top = nets.sorted { (a: CWNetwork, b: CWNetwork) -> Bool in a.rssiValue > b.rssiValue }
        print(String(format: "  %2d: %2d nets %6.1fms  top5: %@", i, nets.count, (now() - st) * 1000,
                     top.prefix(5).map { "\($0.wlanChannel?.channelNumber ?? -1):\($0.rssiValue)" }
                        .joined(separator: " ")))
    }
    usleep(250_000)
}
print("  DISTINCT VECTORS: \(Set(vectors).count) of \(vectors.count)  (1 => cached replay)")
