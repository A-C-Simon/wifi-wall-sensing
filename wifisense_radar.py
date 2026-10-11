#!/usr/bin/env python3
"""Entry point for the radar display: parse arguments, run the view."""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from common import finite_float, positive_float, positive_int, nonnegative_int
from common import CORE, OUT, build_core
from radar_view import Blips, Radar
from wifisense_live import State, reader


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="en0")
    ap.add_argument("--mode", choices=["monitor", "walktest"], default="monitor")
    ap.add_argument("--lead", type=nonnegative_int, default=20,
                    help="seconds before phase 1, to read the schedule")
    ap.add_argument("--empty", type=positive_int, default=40)
    ap.add_argument("--walk", type=positive_int, default=60)
    ap.add_argument("--still", type=positive_int, default=40)
    ap.add_argument("--threshold", type=finite_float, default=None)
    ap.add_argument("--win", type=positive_float, default=10.0)
    ap.add_argument("--room-x", type=positive_float, default=7.0)
    ap.add_argument("--room-y", type=positive_float, default=6.0)
    ap.add_argument("--laptop-x", type=finite_float, default=1.0)
    ap.add_argument("--laptop-y", type=finite_float, default=1.0)
    ap.add_argument("--router-range", type=positive_float, default=4.0)
    ap.add_argument("--router-bearing", type=finite_float, default=315.0,
                    help="degrees clockwise from north, laptop -> router "
                         "(315 = NW, 0 = N, 90 = E)")
    ap.add_argument("--router-rssi", type=finite_float, default=-55.0,
                    help="measured RSSI of the router at the laptop")
    ap.add_argument("--pathloss", type=positive_float, default=2.7)
    ap.add_argument("--radiomap", default=None)
    args = ap.parse_args()
    matplotlib.use("TkAgg")

    build_core()

    r = Radar(args)
    cap = OUT / f"radar_{args.mode}.jsonl"
    if cap.exists():
        cap.unlink()

    dur = 10 ** 6
    phases = []
    if args.mode == "walktest":
        a, b, c = float(args.empty), float(args.walk), float(args.still)
        lead = float(args.lead)
        dur = lead + a + b + c + args.empty
        phases = [(lead, "GET READY\nread the schedule, then stay out of the area"),
                  (a, "PHASE 1 - EMPTY\nstay out of the sensing area"),
                  (b, "PHASE 2 - WALK\nwalk across the area, back and forth"),
                  (c, "PHASE 3 - STAND STILL\nstop moving"),
                  (float(args.empty), "PHASE 4 - EMPTY\nleave the area")]

    proc = subprocess.Popen([str(CORE), "--iface", args.iface, "--out", str(cap),
                             "--duration", str(dur), "--fine-hz", "20", "--quiet"])
    threading.Thread(target=reader, args=(cap, r.st, {"proc": proc}), daemon=True).start()
    if not r.st.ready.wait(timeout=10):
        r.st.stop = True
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError("capture did not publish metadata")
    print(f"radar capture -> {cap}")
    if not r.rmap or not r.rmap.points:
        print("NOTE: no radio map yet, so the radar will show anchor evidence "
              "only. Build one with:  python3 wifisense.py calibrate")

    def sched():
        end = 0.0
        for dur_i, label in phases:
            r.st.phase, r.st.phase_total = label, dur_i
            end += dur_i
            while r.st.elapsed() < end and proc.poll() is None:
                r.st.phase_left = end - r.st.elapsed()
                time.sleep(0.1)
        r.st.phase = ""
    if phases:
        threading.Thread(target=sched, daemon=True).start()

    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(r.fig, r.update, interval=200, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        r.st.stop = True
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print(f"capture saved: {cap}")
    if args.mode == "walktest":
        from validate.score import score_file
        score_file(cap, r.st.threshold, args.win, args.empty, args.walk,
                   args.still, args.empty, lead=args.lead,
                   calibration=r.st.calibration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
