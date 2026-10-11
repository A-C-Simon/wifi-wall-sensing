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
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sensing import detect, features as F, io  # noqa: E402
from sensing.stream import FineEnergy  # noqa: E402
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
        self.fine_sd: collections.deque = collections.deque(maxlen=40000)
        self.fine_chg: collections.deque = collections.deque(maxlen=40000)
        self.fine_energy = collections.deque(maxlen=40000)
        self.energy_filter = None
        self.calibration = None
        self.sweeps: list[tuple[float, dict]] = []
        self.ap_keys: list[str] = []
        self.lock = threading.Lock()
        self.stop = False
        self.meta: dict = {}
        self.error: str | None = None
        self.start = time.monotonic()
        self.ready = threading.Event()
        self.win_s = 10.0
        self.threshold = 3.0
        self.phase: str = ""
        self.phase_left: float = 0.0
        self.phase_total: float = 0.0

    def append_fine(self, record: dict):
        """Append a bucket under the state lock, preserving filter continuity."""
        if self.energy_filter is None:
            self.energy_filter = FineEnergy(self.win_s)
        energy = self.energy_filter.update(record["t"], record["rssi"])
        for target, key in ((self.fine_t, "t"), (self.fine_rssi, "rssi"),
                            (self.fine_mn, "min"), (self.fine_mx, "max"),
                            (self.fine_sd, "sd"), (self.fine_chg, "chg")):
            target.append(record[key])
        self.fine_energy.append(energy)

    def snapshot(self) -> dict:
        """Consistent copy of every series, taken under a single lock.

        The reader thread appends to the deques continuously, so reading the
        arrays in separate critical sections would yield mismatched lengths.
        """
        with self.lock:
            fresh = (bool(self.fine_t) and self.error is None
                     and -.1 <= self.elapsed() - self.fine_t[-1] <= 1.0)
            return {
                "t": np.array(self.fine_t),
                "rssi": np.array(self.fine_rssi),
                "mn": np.array(self.fine_mn),
                "mx": np.array(self.fine_mx),
                "sd": np.array(self.fine_sd),
                "chg": np.array(self.fine_chg),
                "energy": np.array(self.fine_energy),
                "sweeps": [(x[0], dict(x[1])) for x in self.sweeps[-60:]],
                "keys": list(self.ap_keys),
                "meta": dict(self.meta),
                "fresh": fresh,
                "error": self.error if self.error else
                         None if fresh else "waiting for fresh RSSI samples",
            }

    def elapsed(self) -> float:
        return time.monotonic() - self.start


def reader(path: Path, st: State, proc_holder: dict) -> None:
    """Tail the capture file and update state. Never blocks the GUI."""
    fh = None
    def capture_ended():
        proc = proc_holder.get("proc")
        if proc is not None and proc.poll() is not None:
            with st.lock:
                st.error = f"capture ended (exit {proc.returncode})"
            return True
        return False

    while not st.stop:
        if fh is None:
            if not path.exists():
                if capture_ended():
                    break
                time.sleep(0.05)
                continue
            fh = open(path)
        position = fh.tell()
        line = fh.readline()
        if not line:
            if capture_ended():
                break
            time.sleep(0.02)
            continue
        if not line.endswith("\n"):
            if capture_ended():
                break
            # A writer may be midway through a record. Retry from its start
            # instead of dropping the fragment and the rest of the record.
            fh.seek(position)
            time.sleep(.02)
            continue
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
            io.validate_record(o)
            if o.get("type") == "fine" and st.fine_t and o["t"] <= st.fine_t[-1]:
                raise ValueError("fine timestamps must be strictly increasing")
        except (json.JSONDecodeError, ValueError) as e:
            with st.lock:
                st.error = f"invalid capture record: {e}"
            break
        kind = o.get("type")
        with st.lock:
            if kind == "meta":
                st.meta = o
                if "started_at" in o:
                    st.start = time.monotonic() - (time.time() - o["started_at"])
                st.ready.set()
            elif kind == "fine":
                st.append_fine(o)
            elif kind == "sweep":
                aps = io.scan_aps(o["aps"])
                if st.sweeps and o["t"] - st.sweeps[-1][0] < 1.0:
                    if len(aps) > len(st.sweeps[-1][1]):
                        st.sweeps[-1] = (o["t"], aps)
                    continue
                st.sweeps.append((o["t"], aps))
                if len(st.sweeps) > 400:
                    st.sweeps.pop(0)
                keys = {k for _t, m in st.sweeps[-6:] for k in m}
                st.ap_keys = sorted(keys)
            elif kind == "stats":
                if "raw_rate_hz" in o:
                    st.meta["raw_rate_hz"] = o["raw_rate_hz"]
    if fh is not None:
        fh.close()


def scores(snap: dict, win_s: float, calibration=None) -> tuple[np.ndarray, np.ndarray]:
    """Fused motion score over the captured series.

    Takes the snapshot dict rather than the live State so that the timestamps,
    RSSI arrays and sweep list are guaranteed to be mutually consistent.
    """
    if calibration is not None:
        detect.validate_calibration(calibration, win_s)
    keys = (calibration["sweep_keys"] if calibration is not None else
            sorted({k for _, aps in snap["sweeps"] for k in aps}))
    counts = {k: sum(k in aps for _, aps in snap["sweeps"]) for k in keys}
    if calibration is None:
        keys = [k for k in keys if counts[k] >= max(1, len(snap["sweeps"]) // 3)]
    sw = snap["sweeps"]
    mat = np.array([[float(aps.get(k, np.nan)) for k in keys] for _, aps in sw])
    if "energy" in snap:
        motion = detect.fuse_scores(
            snap["t"], snap["energy"], np.array([t for t, _ in sw]),
            mat.reshape(len(sw), len(keys)), calibration=calibration)[0]
        return snap["t"], motion
    result = detect.motion_score_arrays(
        snap["t"], snap["rssi"], snap["sd"], snap["chg"],
        np.array([t for t, _ in sw]), mat.reshape(len(sw), len(keys)),
        win_s=win_s, calibration=calibration)
    return result[0], result[1]


def configure_calibration(st, threshold):
    """Use the full saved calibration unless an explicit threshold is given."""
    path = OUT / "baseline.json"
    if threshold is None and path.exists():
        st.calibration = json.loads(path.read_text())
        detect.validate_calibration(st.calibration, st.win_s)
        st.threshold = st.calibration["threshold"]
    else:
        st.threshold = threshold if threshold is not None else 3.0


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
    def _header(self, motion_now: float, occupied: bool, available: bool = True):
        a = self.ax_state
        a.clear(); a.axis("off")
        a.add_patch(FancyBboxPatch((0.005, 0.30), 0.20, 0.42,
                                    boxstyle="round,pad=0.012",
                                    transform=a.transAxes,
                                    facecolor=ALERT if occupied else IDLE,
                                    edgecolor="none"))
        a.text(0.105, 0.51, "NO DATA" if not available else "MOTION" if occupied else "CLEAR",
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
        t, motion = scores(snap, st.win_s, st.calibration)
        occ = bool(snap["fresh"] and len(motion) and motion[-1] >= st.threshold)
        self._header(float(motion[-1]) if len(motion) else 0.0, occ, snap["fresh"])

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
    matplotlib.use("TkAgg")
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["monitor", "walktest"])
    ap.add_argument("--iface", default="en0")
    ap.add_argument("--empty", type=int, default=45)
    ap.add_argument("--walk", type=int, default=45)
    ap.add_argument("--still", type=int, default=45)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--win", type=float, default=10.0)
    args = ap.parse_args()

    build_core()

    st = State()
    st.win_s = args.win
    try:
        configure_calibration(st, args.threshold)
    except ValueError as e:
        ap.error(str(e))

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
    threading.Thread(target=reader, args=(cap, st, {"proc": proc}), daemon=True).start()
    if not st.ready.wait(timeout=10):
        st.stop = True
        proc.terminate()
        proc.wait(timeout=5)
        raise RuntimeError("capture did not publish metadata")
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
            end = t
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
            proc.wait()
        print(f"\ncapture saved: {cap}")

    # Score the scripted phases for the record.
    if args.mode == "walktest":
        from validate.score import score_file
        score_file(cap, st.threshold, args.win, args.empty, args.walk,
                   args.still, args.empty, calibration=st.calibration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
