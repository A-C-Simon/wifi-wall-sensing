// wifisense_core -- RSSI capture daemon for device-free WiFi sensing on macOS.
//
// Two independent sensing channels are exported as JSON Lines:
//
//   fine  : CWInterface.rssiValue() polled at the hardware register rate
//           (~182 Hz measured on this host). Bucketed to --fine-hz with
//           mean/min/max/std so the 1 dB quantisation noise survives as a
//           usable variance feature without writing raw samples.
//   sweep : CWInterface.scanForNetworks() -> RSSI of every visible AP.
//           These act as spatially distributed passive anchors.
//
// Design notes / measured constraints (see README "Measured envelope"):
//   * CSI is unavailable: Broadcom BCM4377 FullMAC, no CSI path to userspace.
//   * SSID/BSSID are TCC-redacted without Location Services + a signed binary,
//     so APs are keyed on (channel, width, rssi).
//   * macOS refreshes the scan cache at most once per ~12 s, globally. That
//     limit is unbreakable from userspace, so `sweep` records are emitted only
//     when the vector actually changes.
//   * The fine and sweep channels run on separate threads with separate error
//     handling so that a scan failure cannot take down the fine channel.
//
// Build:  swiftc -O -framework CoreWLAN -o wifisense_core wifisense_core.swift
import CoreWLAN
import Foundation

@inline(__always)
func mono() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

// ----------------------------------------------------------------- utilities

final class JSONLWriter {
    private let fh: FileHandle
    private let lock = NSLock()

    init?(path: String) {
        if path == "-" {
            fh = FileHandle.standardOutput
        } else {
            FileManager.default.createFile(atPath: path, contents: nil)
            guard let h = FileHandle(forWritingAtPath: path) else { return nil }
            fh = h
        }
    }

    func emit(_ obj: [String: Any]) {
        var parts: [String] = []
        parts.reserveCapacity(obj.count)
        for (k, v) in obj {
            parts.append("\"\(k)\":" + JSONLWriter.value(v))
        }
        let line = "{" + parts.joined(separator: ",") + "}\n"
        lock.lock()
        fh.write(line.data(using: .utf8)!)
        lock.unlock()
    }

    private static func value(_ v: Any) -> String {
        switch v {
        case let d as Double: return "\(d)"
        case let f as Float: return "\(f)"
        case let i as Int: return "\(i)"
        case let b as Bool: return b ? "true" : "false"
        case let s as String: return quote(s)
        case let a as [[String: Any]]:
            return "[" + a.map { o -> String in
                "{" + o.map { "\"\($0.key)\":\(value($0.value))" }.joined(separator: ",") + "}"
            }.joined(separator: ",") + "]"
        case let a as [Int]:
            return "[" + a.map { "\($0)" }.joined(separator: ",") + "]"
        case let a as [Double]:
            return "[" + a.map { "\($0)" }.joined(separator: ",") + "]"
        case let a as [String]:
            return "[" + a.map { quote($0) }.joined(separator: ",") + "]"
        default:
            return quote("\(v)")
        }
    }

    /// Escape only what JSON requires -- RSSI keys are ASCII in practice.
    private static func quote(_ s: String) -> String {
        var out = "\""
        for ch in s.unicodeScalars {
            switch ch {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            default:
                if ch.value < 0x20 { out += String(format: "\\u%04x", ch.value) }
                else { out.unicodeScalars.append(ch) }
            }
        }
        return out + "\""
    }
}

// ------------------------------------------------------------------- options

struct Opts {
    var iface = "en0"
    var out = "-"
    var duration = 0.0          // 0 = run until SIGINT
    var fineHz = 20.0           // output bucket rate for the fine channel
    var sweep = true            // enable the multi-AP sweep channel
    var quiet = false
}

func parseArgs() -> Opts {
    var o = Opts()
    let a = Array(CommandLine.arguments.dropFirst())
    var i = 0
    while i < a.count {
        let k = a[i]
        func next() -> String { i += 1; return i < a.count ? a[i] : "" }
        switch k {
        case "--iface": o.iface = next()
        case "--out": o.out = next()
        case "--duration": o.duration = Double(next()) ?? 0
        case "--fine-hz": o.fineHz = Double(next()) ?? 20
        case "--no-sweep": o.sweep = false
        case "--quiet": o.quiet = true
        default:
            FileHandle.standardError.write("unknown arg \(k)\n".data(using: .utf8)!)
            exit(2)
        }
        i += 1
    }
    return o
}

let opts = parseArgs()
setvbuf(stdout, nil, _IOLBF, 0)

let client = CWWiFiClient.shared()
guard let iface = client.interface(withName: opts.iface) else {
    FileHandle.standardError.write("FATAL: interface \(opts.iface) not found\n".data(using: .utf8)!)
    exit(2)
}
guard let w = JSONLWriter(path: opts.out) else {
    FileHandle.standardError.write("FATAL: cannot open output \(opts.out)\n".data(using: .utf8)!)
    exit(2)
}

let t0 = mono()

// ------------------------------------------------------------------ shutdown

final class Stopper {
    static let flag = NSLock()
    static var stop = false
}
signal(SIGINT) { _ in Stopper.flag.lock(); Stopper.stop = true; Stopper.flag.unlock() }
signal(SIGTERM) { _ in Stopper.flag.lock(); Stopper.stop = true; Stopper.flag.unlock() }

func shouldStop() -> Bool {
    Stopper.flag.lock(); defer { Stopper.flag.unlock() }
    if Stopper.stop { return true }
    if opts.duration > 0 && mono() - t0 >= opts.duration { return true }
    return false
}

// ----------------------------------------------------------------- metadata

func hardwareVersion() -> String {
    // IO80211HardwareVersion is readable from the Skywalk interface node and
    // identifies the Broadcom part (14e4:4488 => BCM4377B3 on MacBookAir9,1).
    let p = Process()
    p.launchPath = "/usr/sbin/ioreg"
    p.arguments = ["-r", "-c", "AppleBCMWLANSkywalkInterface", "-l"]
    let pipe = Pipe()
    p.standardOutput = pipe
    try? p.run()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    let s = String(data: data, encoding: .utf8) ?? ""
    for line in s.split(separator: "\n") where line.contains("IO80211HardwareVersion") {
        return line.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "\"", with: "")
    }
    return "unknown"
}

w.emit([
    "type": "meta",
    "t": 0.0,
    "iface": opts.iface,
    "rssi_dbm": iface.rssiValue(),
    "noise_dbm": iface.noiseMeasurement(),
    "tx_rate_mbps": iface.transmitRate(),
    "fine_hz": opts.fineHz,
    "hw": hardwareVersion(),
    "ssid_visible": iface.ssid() != nil,
    "bssid_visible": iface.bssid() != nil,
    "csid": "chan",
])

// --------------------------------------------------------------- fine thread

var fineStats = [
    "raw_rate_hz": 0.0,
    "fine_samples": 0,
    "fine_errors": 0,
]

let fineThread = Thread {
    let bucketPeriod = 1.0 / max(opts.fineHz, 0.1)
    var rawCount = 0
    var rawStart = mono()

    var acc: [Int] = []
    var bucketStart = mono()
    var bucketFirstPoll = mono()

    while !shouldStop() {
        let v = iface.rssiValue()
        acc.append(v)
        rawCount += 1

        let nowT = mono()
        if nowT - rawStart >= 1.0 {
            fineStats["raw_rate_hz"] = Double(rawCount) / (nowT - rawStart)
            rawCount = 0
            rawStart = nowT
        }

        if nowT - bucketStart >= bucketPeriod && acc.count > 1 {
            var mn = acc[0], mx = acc[0]
            var sum = 0.0
            for x in acc {
                if x < mn { mn = x }
                if x > mx { mx = x }
                sum += Double(x)
            }
            let mean = sum / Double(acc.count)
            var varr = 0.0
            for x in acc { let d = Double(x) - mean; varr += d * d }
            let sd = sqrt(varr / Double(acc.count))
            var changes = 0
            for i in 1..<acc.count where acc[i] != acc[i - 1] { changes += 1 }

            w.emit([
                "type": "fine",
                "t": nowT - t0,
                "n": acc.count,
                "rssi": mean,
                "min": mn,
                "max": mx,
                "sd": sd,
                "chg": changes,
                "noise": iface.noiseMeasurement(),
                "rate": iface.transmitRate(),
            ])
            acc.removeAll(keepingCapacity: true)
            bucketStart = nowT
            bucketFirstPoll = nowT
        }
    }
    _ = bucketFirstPoll
    w.emit(["type": "stats", "t": mono() - t0,
            "fine_samples": rawCount,
            "fine_bucket_rate": opts.fineHz])
}
fineThread.stackSize = 1024 * 1024
fineThread.start()

// -------------------------------------------------------------- sweep thread

var sweepStats = ["sweeps": 0, "sweep_errors": 0]

if opts.sweep {
    let sweepThread = Thread {
        var lastKey = ""
        while !shouldStop() {
            let c0 = mono()
            do {
                let nets = try iface.scanForNetworks(withName: nil, includeHidden: false)
                var rows: [(Int, Int, Int)] = []
                for n in nets {
                    let ch = n.wlanChannel
                    rows.append((ch?.channelNumber ?? -1,
                                 ch?.channelWidth.rawValue ?? -1,
                                 n.rssiValue))
                }
                rows.sort { (a: (Int, Int, Int), b: (Int, Int, Int)) -> Bool in
                    (a.0, a.1, a.2) < (b.0, b.1, b.2)
                }
                let key = rows.map { "\($0.0)/\($0.1)=\($0.2)" }.joined(separator: ",")
                let cost = mono() - c0
                if key != lastKey {
                    lastKey = key
                    sweepStats["sweeps"] = Int(sweepStats["sweeps"]! + 1)
                    w.emit([
                        "type": "sweep",
                        "t": mono() - t0,
                        "n": rows.count,
                        "cost_ms": cost * 1000.0,
                        "aps": rows.map { ["ch": $0.0, "w": $0.1, "rssi": $0.2] },
                    ])
                }
            } catch {
                sweepStats["sweep_errors"] = Int(sweepStats["sweep_errors"]! + 1)
            }
            // Poll faster than the sweep cost so we catch each new sweep as
            // soon as the OS publishes it, but never busy-spin the driver.
            let elapsed = mono() - c0
            let sleep = elapsed > 1.0 ? 0.05 : 0.25
            usleep(UInt32(sleep * 1_000_000))
        }
    }
    sweepThread.stackSize = 1024 * 1024
    sweepThread.start()
}

// ------------------------------------------------------------------- main

if !opts.quiet {
    FileHandle.standardError.write(
        "wifisense_core: iface=\(opts.iface) fine=\(opts.fineHz)Hz sweep=\(opts.sweep) out=\(opts.out)\n"
            .data(using: .utf8)!)
}

while !shouldStop() {
    usleep(200_000)
}

w.emit([
    "type": "stats",
    "t": mono() - t0,
    "raw_rate_hz": fineStats["raw_rate_hz"] ?? 0.0,
    "sweeps": sweepStats["sweeps"] ?? 0,
    "sweep_errors": sweepStats["sweep_errors"] ?? 0,
])
if !opts.quiet {
    FileHandle.standardError.write("wifisense_core: done after \(Int(mono() - t0))s\n".data(using: .utf8)!)
}
exit(0)