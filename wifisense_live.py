#!/usr/bin/env python3
"""wifisense live -- real-time GUI for RSSI-only through-wall presence sensing.

Modes:
  monitor    free-running dashboard, no scripted phases
  walktest   scripted EMPTY -> WALK -> STILL -> EMPTY protocol with on-screen
             prompts and countdown, so the operator can follow along

The dashboard shows what each sensing channel is actually contributing:
  * the associated-link RSSI trace (1 dB quantised, visually obvious)
  * the fused motion score against the calibrated threshold
  * every visible access point as a live anchor bar -- these are the
    spatially distributed receivers that make attribution possible
  * the frame-to-frame multi-AP change, which is the spatial evidence

Usage:
  python3 wifisense_live.py monitor
  python3 wifisense_live.py walktest --empty 45 --walk 45 --still 45
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sensing import features as F  # noqa: E402
from common import build_core  # noqa: E402

OUT = ROOT / "out"
CORE = ROOT / "native" / "wifisense_core"

BG = "#12151c"
PANEL = "#1b2029"
FG = "#e6e9ef"
DIM = "#8b93a7"
GOOD = "#3ddc84"
WARN = "#ffb020"
ALERT = "#ff4d5a"
ACCENT = "#4da3ff"
IDLE = "#39414f"

SPAN_S = 150.0     # seconds visible in the scrolling traces


class State:
    """Everything the renderer needs, filled by a background file reader."""
    def __init__(self):
        self.fine_t: collections.deque = collections.deque(maxlen=40000)
        self.fine_rssi: collections.deque = collections.deque(maxlen=40000)
        self.fine_mn: collections.deque = collections.deque(maxlen=40000)
        self.fine_mx: collections.deque = collections.deque(maxlen=40000)
        self.sweeps: list[tuple[float, dict]] = []
        self.ap_keys: list[str] = []
        self.lock = threading.Lock()
        self.stop = False
        self.meta: dict = {}
        self.error: str | None = None
        self.start = time.time()
        self.win_s = 10.0
        self.threshold = 3.0
        self.phase: str = ""
        self.phase_left: float = 0.0
        self.phase_total: float = 0.0

    def snapshot(self) -> dict:
        """Consistent copy of every series, taken under a single lock.

        The reader thread appends to the deques continuously, so reading the
        arrays in separate critical sections would yield mismatched lengths.
        """
        with self.lock:
            return {
                "t": np.array(self.fine_t),
                "rssi": np.array(self.fine_rssi),
                "mn": np.array(self.fine_mn),
                "mx": np.array(self.fine_mx),
                "sweeps": [(x[0], dict(x[1])) for x in self.sweeps[-60:]],
                "keys": list(self.ap_keys),
                "meta": dict(self.meta),
            }

    def elapsed(self) -> float:
        return time.time() - self.start


def reader(path: Path, st: State, proc_holder: dict) -> None:
    """Tail the capture file and update state. Never blocks the GUI."""
    fh = None
    while not st.stop:
        if fh is None:
            if not path.exists():
                time.sleep(0.05)
                continue
            fh = open(path)
        line = fh.readline()
        if not line:
            time.sleep(0.02)
            continue
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = o.get("type")
        with st.lock:
            if kind == "meta":
                st.meta = o
            elif kind == "fine":
                st.fine_t.append(o["t"])
                st.fine_rssi.append(o["rssi"])
                st.fine_mn.append(o["min"])
                st.fine_mx.append(o["max"])
            elif kind == "sweep":
                aps = {f"{a['ch']}/{a['w']}": a["rssi"] for a in o["aps"]}
                st.sweeps.append((o["t"], aps))
                if len(st.sweeps) > 400:
                    st.sweeps.pop(0)
                keys = {k for _t, m in st.sweeps[-6:] for k in m}
                st.ap_keys = sorted(keys)
            elif kind == "stats":
                st.meta.setdefault("raw_rate_hz", o.get("raw_rate_hz"))


def scores(snap: dict, win_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Fused motion score over the captured series.

    Takes the snapshot dict rather than the live State so that the timestamps,
    RSSI arrays and sweep list are guaranteed to be mutually consistent.
    """
    t = snap["t"]
    if len(t) < 3:
        return t, np.zeros(len(t))
    r, mn, mx = snap["rssi"], snap["mn"], snap["mx"]
    sw, keys = snap["sweeps"], snap["keys"]
    dt = float(np.median(np.diff(t)))
    chg = np.zeros_like(t)
    chg[1:] = (np.diff(mn) != 0).astype(float) + (np.diff(mx) != 0).astype(float)
    sd = (mx - mn).astype(float)

    ff = F.fine_features(t, r, sd, chg, win_s, dt)
    span = float(t[-1] - t[0]) if len(t) > 1 else 600.0
    half_life = max(60.0, min(max(span * 0.6, 3 * win_s), 1800.0))
    res = F.detrend(ff[:, 0], t, half_life_s=half_life)
    fine = F.zscore_robust(F.envelope(res))

    sweep = np.zeros(len(t))
    if len(sw) >= 3 and keys:
        st_ = np.array([x[0] for x in sw])
        mat = np.array([[float(m.get(k, np.nan)) for k in keys] for _t, m in sw])
        sf = F.sweep_features(st_, mat)
        if len(sf):
            l1z = F.ratio_score(sf[:, 0])
            se = st_[1:]
            interval = float(np.median(np.diff(st_)))
            sweep = F.hold_sweeps(t, se, min(1.0, len(l1z) / 12.0) * l1z,
                                  max(2.0 * interval, 10.0))
    return t, np.maximum(fine, sweep)


class Dash:
    def __init__(self, st: State):
        self.st = st
        plt.rcParams.update({
            "figure.facecolor": BG, "axes.facecolor": PANEL,
            "savefig.facecolor": BG, "text.color": FG,
            "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
            "axes.edgecolor": "#2b3240", "font.size": 9,
        })
        self.fig = plt.figure(figsize=(15.5, 9.6), facecolor=BG)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[0.42, 1.0, 1.0],
                                   hspace=0.42, wspace=0.18,
                                   left=0.055, right=0.975, top=0.955, bottom=0.06)
        self.ax_state = self.fig.add_subplot(gs[0, :]); self.ax_state.axis("off")
        self.ax_rssi = self.fig.add_subplot(gs[1, 0])
        self.ax_ap = self.fig.add_subplot(gs[1, 1])
        self.ax_score = self.fig.add_subplot(gs[2, 0])
        self.ax_sweep = self.fig.add_subplot(gs[2, 1])
        for a in (self.ax_rssi, self.ax_ap, self.ax_score, self.ax_sweep):
            a.grid(True, alpha=0.18, lw=0.5)
            a.set_axisbelow(True)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            a.tick_params(labelsize=7.5)

    # ---------------------------------------------------------------- header
    def _header(self, motion_now: float, occupied: bool):
        a = self.ax_state
        a.clear(); a.axis("off")
        a.add_patch(FancyBboxPatch((0.005, 0.30), 0.20, 0.42,
                                    boxstyle="round,pad=0.012",
                                    transform=a.transAxes,
                                    facecolor=ALERT if occupied else IDLE,
                                    edgecolor="none"))
        a.text(0.105, 0.51, "MOTION" if occupied else "CLEAR",
               transform=a.transAxes, ha="center", va="center",
               fontsize=17, fontweight="bold",
               color="white")
        a.text(0.225, 0.60, f"score {motion_now:+.2f}   threshold {self.st.threshold:.2f}",
               transform=a.transAxes, fontsize=11, color=FG)
        a.text(0.225, 0.36, f"elapsed {self.st.elapsed():6.1f}s", transform=a.transAxes,
               fontsize=9, color=DIM)

        st = self.st
        rr = st.meta.get("raw_rate_hz")
        info = f"RSSI {st.meta.get('rssi_dbm','?')} dBm   noise {st.meta.get('noise_dbm','?')} dBm"
        if rr:
            info += f"   register poll {rr:.0f} Hz"
        a.text(0.995, 0.60, info, transform=a.transAxes, ha="right", fontsize=9, color=DIM)
        a.text(0.995, 0.36, "RSSI-only sensing - no CSI, 1 dB quantisation",
               transform=a.transAxes, ha="right", fontsize=8.5, color=DIM)

        if st.phase:
            frac = 1.0 - min(st.phase_left / max(st.phase_total, 1e-9), 1.0)
            a.add_patch(FancyBboxPatch((0.005, 0.02), 0.99 * max(frac, 0.0), 0.13,
                                       boxstyle="square,pad=0",
                                       transform=a.transAxes,
                                       facecolor=ACCENT, edgecolor="none", alpha=0.35))
            head, _, sub = st.phase.partition("\n")
            a.text(0.47, 0.70, head, transform=a.transAxes, ha="center",
                   va="center", fontsize=19, fontweight="bold", color=WARN)
            if sub:
                a.text(0.47, 0.45, sub, transform=a.transAxes, ha="center",
                       va="center", fontsize=10.5, color=FG)
            # Countdown sits under the state badge so it never collides with
            # the right-hand telemetry block.
            a.text(0.105, 0.17, f"{max(st.phase_left,0):.0f}s",
                   transform=a.transAxes, ha="center", va="center",
                   fontsize=17, fontweight="bold", color=FG)
        elif st.error:
            a.text(0.5, 0.5, st.error, transform=a.transAxes, ha="center",
                   color=ALERT, fontsize=13)

    # ---------------------------------------------------------------- update
    def update(self, _frame=None):
        st = self.st
        snap = st.snapshot()
        st.meta = snap["meta"]
        t, motion = scores(snap, st.win_s)
        occ = bool(len(motion) and motion[-1] >= st.threshold)
        self._header(float(motion[-1]) if len(motion) else 0.0, occ)

        now = st.elapsed()
        t0 = max(0.0, now - SPAN_S)

        # -- associated RSSI
        a = self.ax_rssi; a.clear()
        if len(t):
            m = (t >= t0) & (t <= now)
            tt, rr = t[m], snap["rssi"][m]
            mn = snap["mn"][m]
            mx = snap["mx"][m]
            if len(tt):
                a.fill_between(tt, mn, mx, color=ACCENT, alpha=0.22, lw=0,
                               label="min/max per 50 ms bucket")
                a.plot(tt, rr, color=ACCENT, lw=1.0, label="bucket mean")
                a.set_ylim(mn.min() - 1.5, mx.max() + 1.5)
        a.set_xlim(t0, now)
        a.set_title("Associated-link RSSI  (1 dB quantised: note the staircase)",
                    fontsize=9.5, color=FG, loc="left")
        a.set_ylabel("dBm", fontsize=8.5)
        a.legend(fontsize=7, loc="lower left", framealpha=0.15)

        # -- anchors
        a = self.ax_ap; a.clear()
        sw_all = snap["sweeps"]
        latest = dict(sw_all[-1][1]) if sw_all else {}
        if latest:
            ks = sorted(latest, key=lambda k: -latest[k])
            vals = [latest[k] for k in ks]
            cols = [ALERT if v > -55 else (WARN if v > -68 else ACCENT) for v in vals]
            a.barh(range(len(ks)), vals, color=cols, height=0.72)
            a.set_yticks(range(len(ks)))
            a.set_yticklabels(ks, fontsize=6.5)
            # Bars run leftwards from 0 because the values are negative, so the
            # axis must cover the strongest (most negative) anchor.
            a.set_xlim(0, min(vals) - 6)
            a.invert_xaxis()
            a.set_ylim(-0.7, len(ks) - 0.3)
        a.set_title(f"Multi-AP anchors - spatially distributed receivers "
                    f"({len(latest)} heard)", fontsize=9.5, color=FG, loc="left")
        a.set_xlabel("RSSI dBm", fontsize=8.5)

        # -- fused score
        a = self.ax_score; a.clear()
        if len(t):
            m = (t >= t0) & (t <= now)
            tt, mm = t[m], motion[m]
            if len(tt):
                a.plot(tt, mm, color="#c8d0e0", lw=1.0, label="motion score")
                a.axhline(st.threshold, color=ALERT, ls="--", lw=1.1,
                          label=f"threshold {st.threshold:.2f}")
                a.fill_between(tt, st.threshold, mm, where=mm >= st.threshold,
                               color=ALERT, alpha=0.30, lw=0, label="occupied")
                a.set_ylim(min(-1.0, float(np.percentile(mm, 5)) - 0.5),
                           max(st.threshold + 1.5, float(np.percentile(mm, 99)) + 0.8))
                a.legend(fontsize=7, loc="upper left", framealpha=0.15)
        a.set_xlim(t0, now)
        a.set_title("Fused motion decision (detrended RSSI + multi-AP change)",
                    fontsize=9.5, color=FG, loc="left")
        a.set_ylabel("robust z-score", fontsize=8.5)

        # -- sweep L1
        a = self.ax_sweep; a.clear()
        sw = snap["sweeps"][-40:]
        keys = snap["keys"]
        if len(sw) >= 2 and keys:
            arr = np.array([[float(m.get(k, np.nan)) for k in keys] for _t, m in sw])
            f = F._fill_columns(arr)
            L1 = np.abs(np.diff(f, axis=0)).sum(axis=1)
            stt = [x[0] for x in sw[1:]]
            a.bar(stt, L1, width=max(6.0, SPAN_S * 0.03), color=WARN)
            a.axhline(np.median(L1) if len(L1) else 0, color=DIM, ls=":", lw=1)
            a.set_ylim(0, max(float(L1.max()) * 1.25, 1.0) if len(L1) else 1)
            # Early on, every sweep timestamp may precede the window start; fall
            # back to showing the full history rather than an empty panel.
            if len(stt) and stt[-1] >= t0:
                a.set_xlim(t0, now)
            else:
                a.set_xlim(min(stt), max(stt[-1], min(stt) + 1.0))
        a.set_title("Multi-AP frame-to-frame change  (sum |dRSSI|, spatial evidence)",
                    fontsize=9.5, color=FG, loc="left")
        a.set_ylabel("dB (summed over anchors)", fontsize=8.5)
        a.set_xlabel("seconds", fontsize=8.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["monitor", "walktest"])
    ap.add_argument("--iface", default="en0")
    ap.add_argument("--empty", type=int, default=45)
    ap.add_argument("--walk", type=int, default=45)
    ap.add_argument("--still", type=int, default=45)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--win", type=float, default=10.0)
    args = ap.parse_args()

    build_core()

    st = State()
    st.win_s = args.win
    st.threshold = args.threshold

    cap = OUT / ("live_walktest.jsonl" if args.mode == "walktest" else "live_monitor.jsonl")
    if cap.exists():
        cap.unlink()

    dur = 10 ** 6
    phases: list[tuple[float, str]] = []
    if args.mode == "walktest":
        a, b, c = float(args.empty), float(args.walk), float(args.still)
        dur = a + b + c + args.empty
        phases = [
            (a, "PHASE 1 - EMPTY ROOM\nstay out of the area, completely still"),
            (b, "PHASE 2 - WALK\nwalk back and forth across the space"),
            (c, "PHASE 3 - STAND STILL\nstop moving, do not leave"),
            (float(args.empty), "PHASE 4 - EMPTY\nwalk away and stay out"),
        ]

    proc = subprocess.Popen([str(CORE), "--iface", args.iface, "--out", str(cap),
                             "--duration", str(dur), "--fine-hz", "20", "--quiet"])
    threading.Thread(target=reader, args=(cap, st, {}), daemon=True).start()
    time.sleep(0.4)
    st.start = time.time()
    print(f"capturing -> {cap}   (threshold {st.threshold})")
    if args.mode == "walktest":
        print("Follow the on-screen prompts exactly. Press Ctrl-C in this "
              "terminal, or close the window, to stop.")

    # Phase scheduler
    def schedule():
        t = 0.0
        for dur_i, label in phases:
            t += dur_i
            st.phase, st.phase_total, st.phase_left = label, dur_i, dur_i
            end = st.elapsed() + dur_i
            while st.elapsed() < end and proc.poll() is None:
                st.phase_left = end - st.elapsed()
                time.sleep(0.1)
        st.phase, st.phase_left = "", 0.0
    if phases:
        threading.Thread(target=schedule, daemon=True).start()

    dash = Dash(st)
    from matplotlib.animation import FuncAnimation
    anim = FuncAnimation(dash.fig, dash.update, interval=180, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        st.stop = True
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print(f"\ncapture saved: {cap}")

    # Score the scripted phases for the record.
    if args.mode == "walktest":
        from validate.score import score_file
        score_file(cap, args.threshold, args.win, args.empty, args.walk,
                   args.still, args.empty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
