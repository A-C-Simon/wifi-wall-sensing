"""Measure what the machine can do, and record and analyse sessions."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sensing import detect, io, viz

from common import CORE, OUT, ROOT, build_core, default_iface


# --------------------------------------------------------------------- doctor


def cmd_doctor(args) -> int:
    """Measure, do not assume. Everything the design depends on is re-derived
    here so the numbers in the README are reproducible on this host."""
    iface = args.iface
    build_core()
    print(f"== wifisense doctor ({iface}) ==\n")

    print("[1] hardware / driver identity")
    p = subprocess.run(["system_profiler", "SPAirPortDataType"],
                       capture_output=True, text=True)
    for line in p.stdout.splitlines():
        s = line.strip()
        if s.startswith(("Card Type", "Firmware Version", "Country Code")):
            print("   ", s)
    p = subprocess.run(["ioreg", "-r", "-c", "AppleBCMWLANSkywalkInterface", "-l"],
                       capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if "IO80211HardwareVersion" in line:
            print("   ", line.strip().strip('"'))
            break

    print("\n[2] associated-link scalar poll rate and resolution")
    raw = OUT / "doctor.jsonl"
    dur = args.duration
    build_core()
    subprocess.run([str(CORE), "--iface", iface, "--out", str(raw),
                    "--duration", str(dur), "--fine-hz", "20", "--quiet"])
    s = io.load(raw)
    if s.meta:
        print(f"    rssi={s.meta.get('rssi_dbm')} dBm  noise={s.meta.get('noise_dbm')} dBm"
              f"  tx_rate={s.meta.get('tx_rate_mbps')} Mbps")
        print(f"    SSID visible={s.meta.get('ssid_visible')}  "
              f"BSSID visible={s.meta.get('bssid_visible')}  (False => TCC redaction)")
    if s.fine:
        ft = [f.t for f in s.fine]
        n = [f.n for f in s.fine]
        raw_hz = sum(n) / max(ft[-1] - ft[0], 1e-9)
        # mn/mx are the raw integer register readings bounding each bucket, so
        # their union is the true set of quantisation levels. Using the bucket
        # means instead would report float averages and overstate resolution.
        levels = sorted({f.mn for f in s.fine} | {f.mx for f in s.fine})
        span = (max(levels) - min(levels)) if levels else 0
        n_floats = len({f.rssi for f in s.fine})
        print(f"    register poll rate   : {raw_hz:.0f} Hz")
        print(f"    raw RSSI levels seen : {len(levels)} -> {levels}")
        print(f"    RSSI span            : {span} dB")
        print(f"    (bucket means take {n_floats} distinct float values, but they are")
        print(f"     averages of 1 dB-quantised samples, not extra resolution)")
    else:
        raw_hz, span, levels = 0.0, 0, []

    print("\n[3] multi-AP sweep refresh rate (the real time resolution)")
    if len(s.sweeps) >= 2:
        ts = [x.t for x in s.sweeps]
        iv = [ts[i] - ts[i - 1] for i in range(1, len(ts))]
        iv.sort()
        med = iv[len(iv) // 2]
        # A real sweep costs ~2 s; subsequent calls in the same burst just read
        # the repopulated cache, so the slowest call is the honest figure.
        max_cost = max(x.cost_ms for x in s.sweeps)
        print(f"    fresh vectors        : {len(s.sweeps)} in {dur}s")
        print(f"    median interval      : {med:.1f}s  (=> {1/med:.2f} Hz)")
        print(f"    interval range       : {iv[0]:.1f}s .. {iv[-1]:.1f}s")
        print(f"    AP anchors visible   : {len(s.ap_keys())}")
        print(f"    slowest sweep call   : {max_cost:.0f} ms (a real off-channel sweep)")
    else:
        print("    not enough sweeps in this window")

    print("\n[4] verdict")
    print("    CSI            : UNAVAILABLE (Broadcom FullMAC, no userspace CSI path)")
    print("    monitor mode   : UNAVAILABLE (airport binary is a deprecation stub)")
    print("    fast channel   : ~%.0f Hz polls but only %d dB of RSSI span"
          % (raw_hz if s.fine else 0, span))
    if len(s.sweeps) >= 2:
        print(f"    spatial channel: multi-AP vector every ~{med:.0f}s")
    print("\n    Real time body mapping is not attainable on this hardware.")
    print("    Measured outcomes for this class of machine, see README:")
    print("      localisation : 0 percent accuracy against 25 percent chance")
    print("      motion       : walking scores below an empty room")
    print("    The scan interval above is the reason for both. Faster,")
    print("    higher resolution samples need CSI, which needs different")
    print("    hardware, not different software.")
    return 0


# -------------------------------------------------------------------- capture


def cmd_capture(args) -> int:
    build_core()
    out = Path(args.out) if args.out else OUT / f"capture_{int(time.time())}.jsonl"
    print(f"recording {args.duration}s -> {out}", flush=True)
    cmd = [str(CORE), "--iface", args.iface, "--out", str(out),
           "--duration", str(args.duration), "--fine-hz", str(args.fine_hz)]
    if args.no_sweep:
        cmd.append("--no-sweep")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        pass
    s = io.load(out)
    print(f"done: {len(s.fine)} fine buckets, {len(s.sweeps)} sweeps, {s.duration:.0f}s")
    return 0


# ------------------------------------------------------------------- baseline


def cmd_baseline(args) -> int:
    src = Path(args.capture) if args.capture else OUT / f"baseline_{int(time.time())}.jsonl"
    if not src.exists():
        print("recording empty-room baseline ...", flush=True)
        print("LEAVE THE ROOM EMPTY for the whole recording.", flush=True)
        build_core()
        subprocess.run([str(CORE), "--iface", args.iface, "--out", str(src),
                        "--duration", str(args.duration), "--fine-hz", "20"])
    s = io.load(src)
    info = detect.baseline_threshold(s, win_s=args.win, k=args.k)
    if not info:
        print("no data"); return 1
    info["capture"] = str(src)
    (OUT / "baseline.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    viz.plot_channels(s, str(OUT / "baseline.png"), detect.detect(s, threshold=info["threshold"]))
    print(f"\nwrote {OUT/'baseline.json'} and {OUT/'baseline.png'}")
    return 0


# -------------------------------------------------------------------- analyze


def cmd_analyze(args) -> int:
    src = Path(args.capture)
    s = io.load(src)
    thr = None
    bp = OUT / "baseline.json"
    if bp.exists() and not args.no_baseline:
        thr = json.loads(bp.read_text()).get("threshold")
        print(f"using calibrated threshold {thr:.3f} from {bp}")
    r = detect.detect(s, win_s=args.win, threshold=thr)
    segs = r.segments()
    print(f"\nfine buckets : {len(s.fine)}  ({r.fine_rate_hz:.2f} transitions/s)")
    print(f"sweeps       : {len(s.sweeps)}  (interval {r.sweep_interval_s:.1f}s)")
    print(f"anchors      : {len(s.ap_keys())}")
    print(f"threshold    : {r.threshold:.3f}")
    print(f"occupancy    : {100*r.occupancy_fraction:.1f}% of the capture")
    print(f"segments     : {len(segs)}")
    for a, b in segs[:20]:
        print(f"   {a:7.1f}s -> {b:7.1f}s   ({b-a:.1f}s)")
    viz.plot_channels(s, str(OUT / "channels.png"), r)
    viz.plot_ap_heights(s, str(OUT / "ap_anchors.png"))
    print(f"\nwrote {OUT/'channels.png'} and {OUT/'ap_anchors.png'}")
    return 0


# ------------------------------------------------------------------ calibrate
