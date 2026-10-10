"""Radar view: a plan position indicator for movement sensed through WiFi.

The laptop is drawn at the origin as the only receiver. Every visible access
point is a separate transmitter whose link to the laptop a body disturbs, so
each one is a sensing element. Positions come from fingerprint matching against
a surveyed map and therefore resolve to cells rather than a continuous track.
Evidence arrives only when macOS publishes a new scan, roughly every 12 s on
this class of machine.

Nothing here is simulated. When no map exists, or the map failed validation,
the view says so instead of inventing positions.
"""
from __future__ import annotations

import collections
import math
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from common import CORE, OUT, ROOT
from sensing import features as F, linkmodel, radiomap
from wifisense_live import State, configure_calibration, reader, scores


BG = "#0b0e14"
PANEL = "#161b25"
FG = "#e8ecf4"
DIM = "#79839a"
GRID = "#243040"
GREEN = "#39ff88"
AMBER = "#ffb020"
RED = "#ff4d5a"
BLUE = "#4da3ff"
LAPTOP_C = "#c77dff"
ROUTER_C = "#00e5ff"
TRAIL_MAX = 260          # blip history points retained
TRAIL_TTL = 45.0         # seconds a blip remains visible


class Blips:
    """Decaying contact history. Each entry keeps the runner-up cell too, so
    the display can show how ambiguous the match actually is."""

    def __init__(self):
        self.d: collections.deque = collections.deque(maxlen=TRAIL_MAX)

    def add(self, t: float, x: float, y: float, strength: float, occupied: bool,
            alt: tuple[float, float] | None = None, margin: float = float("nan")):
        self.d.append((t, x, y, strength, occupied, alt, margin))

    def live(self, now: float):
        return [b for b in self.d if now - b[0] < TRAIL_TTL]


class Radar:
    def __init__(self, args):
        self.a = args
        self.st = State()
        self.st.win_s = args.win
        configure_calibration(self.st, args.threshold)
        self.blips = Blips()
        self.rmap = None
        self.laptop = (args.laptop_x, args.laptop_y)

        mp = Path(args.radiomap) if args.radiomap else OUT / "radiomap.json"
        if mp.exists():
            try:
                self.rmap = radiomap.RadioMap.load(mp)
            except Exception as e:  # a bad map must not kill the display
                print(f"could not load radio map: {e}")

        # Fitted RSSI<->range model, if the operator has surveyed one. Without
        # it, anchor ranges fall back to a stated default exponent and are
        # labelled as estimates on the display.
        # A map is only usable if it was validated. Without this gate the radar
        # would confidently paint a position for every scan while the matcher is
        # effectively guessing -- measured at 0% accuracy on this host.
        self.map_ok = False
        self.map_why = "no radio map"
        if self.rmap and self.rmap.points:
            v = self.rmap.meta.get("verdict")
            if v == "usable":
                self.map_ok = True
                self.map_why = ""
            elif v:
                sr = self.rmap.meta.get("separability_ratio")
                rp = (self.rmap.meta.get("repeatability") or {}).get(
                    "mean_same_position_db")
                self.map_why = f"map failed validation (separability {sr:.2f}x" if sr else "map failed validation"
                if rp:
                    self.map_why += f", session drift {rp:.0f} dB"
                self.map_why += ")"
            else:
                self.map_why = "map not validated - run: wifisense.py validate-map"

        self.lmodel = None
        lmp = OUT / "linkmodel.json"
        if lmp.exists():
            try:
                self.lmodel = linkmodel.LinkModel.load(lmp)
                print(f"link model: RSSI(d) = {self.lmodel.A:.2f} - "
                      f"10*{self.lmodel.n:.2f}*log10(d)  [{self.lmodel.note}]")
            except Exception as e:
                print(f"could not load link model: {e}")
        else:
            print("no link model: anchor ranges use the default path-loss "
                  f"exponent n={args.pathloss}. Run "
                  "'python3 wifisense.py calibrate-distance' to fit your own.")

        self._last_sweep = -1.0
        self.matcher = radiomap.Matcher(self.rmap) if self.rmap else None

        plt.rcParams.update({
            "figure.facecolor": BG, "axes.facecolor": BG, "text.color": FG,
            "axes.labelcolor": DIM, "xtick.color": DIM, "ytick.color": DIM,
            "font.size": 9, "savefig.facecolor": BG,
        })
        # Explicit placement in disjoint rectangles. Free-floating fig.text
        # cannot be reasoned about once several strings share a band, which is
        # how the header, phase banner, legend and panel titles ended up
        # printing on top of each other. Each band owns its own axes and only
        # draws inside its own x-range.
        #
        #   header   y 0.895-0.970   state | score | telemetry
        #   phase    y 0.825-0.885   phase title + countdown + progress bar
        #   radar    y 0.100-0.800   square, x 0.045-0.512
        #   legend   y 0.580-0.800   own axes, x 0.525-0.655
        #   map      y 0.550-0.800   x 0.680-0.970
        #   evidence y 0.100-0.480   x 0.680-0.970
        #   footer   y 0.020-0.070
        #
        # The radar axes is sized as a true square (width fraction = height
        # fraction x figheight/figwidth) so the range rings stay circular.
        self.W, self.H = 15.0, 10.0
        self.fig = plt.figure(figsize=(self.W, self.H), facecolor=BG)
        side = 0.70
        wide = side * self.H / self.W
        self.wide = wide
        self.ax_p = self.fig.add_axes([0.045, 0.10, wide, side])
        self.ax_leg = self.fig.add_axes([0.525, 0.58, 0.135, 0.22]); self.ax_leg.axis("off")
        self.ax_map = self.fig.add_axes([0.68, 0.55, 0.29, 0.25])
        self.ax_ev = self.fig.add_axes([0.68, 0.10, 0.29, 0.38])
        self.ax_head = self.fig.add_axes([0.03, 0.895, 0.94, 0.075]); self.ax_head.axis("off")
        self.ax_phase = self.fig.add_axes([0.03, 0.825, 0.94, 0.06]); self.ax_phase.axis("off")
        self.ax_foot = self.fig.add_axes([0.03, 0.020, 0.94, 0.05]); self.ax_foot.axis("off")

        self.ax_p.set_facecolor("#070a10")
        self.ax_map.set_facecolor(PANEL)
        self.ax_ev.set_facecolor(PANEL)

    # ------------------------------------------------------------- geometry
    def to_polar(self, x: float, y: float):
        dx, dy = x - self.laptop[0], y - self.laptop[1]
        r = math.hypot(dx, dy)
        th = math.degrees(math.atan2(dx, dy)) % 360.0
        return r, th

    def rmax(self) -> float:
        vals = [self.a.room_x, self.a.room_y]
        if self.rmap and self.rmap.points:
            for p in self.rmap.points:
                vals.append(math.hypot(p.x - self.laptop[0], p.y - self.laptop[1]))
        vals.append(self.a.router_range * 1.25)
        return max(max(vals), 3.0) * 1.12

    # --------------------------------------------------------------- update
    def update(self, _f=None):
        st = self.st
        snap = st.snapshot()
        now = st.elapsed()
        t, motion = scores(snap, st.win_s, st.calibration)
        cur = float(motion[-1]) if len(motion) else 0.0
        occ = bool(snap["fresh"] and len(motion) and cur >= st.threshold)

        # New data point only when macOS published a genuinely new scan.
        if snap["sweeps"] and snap["sweeps"][-1][0] > self._last_sweep:
            self._last_sweep = snap["sweeps"][-1][0]
            self._ingest(snap["sweeps"][-1][0], snap["sweeps"][-1][1], occ)

        self._draw_polar(snap, now, occ, cur)
        self._draw_map(snap, now, occ)
        self._draw_evidence(snap, t, motion, now)
        self._draw_header(snap, now, occ, cur)

    def _ingest(self, t: float, aps: dict, occ: bool):
        """Attribute a scan to a map cell and record a blip."""
        if not self.rmap or not self.rmap.points:
            return
        if not self.map_ok:
            return
        res = self.matcher.localize(aps, top=2)
        if not res:
            return
        p, dist = res[0]
        strength = max(0.0, min(1.0, 1.0 - dist / 60.0))
        alt = (res[1][0].x, res[1][0].y) if len(res) > 1 else None
        margin = res[1][1] - dist if len(res) > 1 else float("nan")
        self.blips.add(t, p.x, p.y, strength, occ, alt, margin)

    # --------------------------------------------------------------- panels
    @staticmethod
    def xy(bearing_deg: float, r: float):
        """Compass bearing + range -> plot coordinates.

        North is +y and east is +x by construction. matplotlib's polar axes
        (theta_zero_location / theta_direction) did not compose into a
        clockwise-from-north frame on this version, so the radar is drawn in
        Cartesian coordinates with the conversion done explicitly here.
        """
        b = math.radians(bearing_deg)
        return r * math.sin(b), r * math.cos(b)

    def _draw_polar(self, snap, now, occ, cur):
        ax = self.ax_p
        ax.clear()
        ax.set_facecolor("#070a10")
        ax.set_aspect("equal", adjustable="box")
        rm = self.rmax()
        ax.set_xlim(-rm * 1.12, rm * 1.12)
        ax.set_ylim(-rm * 1.12, rm * 1.12)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        ring = np.linspace(0.0, 2.0 * np.pi, 361)
        rx = rm * np.sin(ring)
        ry = rm * np.cos(ring)
        for rr in range(1, int(rm) + 1):
            ax.plot(rx * rr / rm, ry * rr / rm, color=GRID, lw=0.6, zorder=0)
            ax.text(0.0, rr, f"{rr}m", color=DIM, fontsize=6.5,
                    ha="center", va="bottom", zorder=9)
        for a_ in range(0, 360, 30):
            ex, ey = self.xy(a_, rm)
            ax.plot([0, ex], [0, ey], color=GRID, lw=0.5, zorder=0)
        for a_, lab, sz, col in ([(0, "N", 10, FG), (90, "E", 10, FG),
                                  (180, "S", 10, FG), (270, "W", 10, FG)] +
                                 [(a_, str(a_), 6, DIM) for a_ in
                                  (30, 60, 120, 150, 210, 240, 300, 330)]):
            lx, ly = self.xy(a_, rm * (1.10 if sz > 8 else 1.045))
            ax.text(lx, ly, lab, color=col, fontsize=sz, ha="center", va="center",
                    fontweight="bold" if sz > 8 else "normal", zorder=9)

        live = self.blips.live(now)

        # known anchor: the router
        rx_, ry_ = self.xy(self.a.router_bearing, self.a.router_range)
        ax.scatter([rx_], [ry_], s=190, marker="s", color=ROUTER_C,
                   edgecolor="white", lw=1.0, zorder=7)
        ax.text(rx_, ry_ + rm * 0.055, "ROUTER", color=ROUTER_C, fontsize=7.5,
                ha="center", va="bottom", zorder=9)

        # other visible APs: range estimated from the fitted log-distance model
        sw = snap["sweeps"][-1][1] if snap["sweeps"] else {}
        for k, v in sw.items():
            px, py = self.xy(self.ap_bearing(k), self.est_range(v))
            ax.scatter([px], [py], s=26, marker=".", color="#5f7fa8", zorder=4)

        # the receiver
        ax.scatter([0], [0], s=230, marker="*", color=LAPTOP_C,
                   edgecolor="white", lw=0.8, zorder=7)
        ax.text(0, rm * 0.055, "LAPTOP", color=LAPTOP_C, fontsize=7.5,
                ha="center", va="bottom", fontweight="bold", zorder=9)

        # Motion contacts: red circles only. A non-motion sample is not drawn
        # at all, so anything visible on the radar is a positive detection.
        for b in live:
            bt, bx, by, stg, bo, alt, margin = b
            if not bo:
                continue
            age = (now - bt) / TRAIL_TTL
            a = max(0.10, 1.0 - age)
            r_, th_ = self.to_polar(bx, by)
            px, py = self.xy(th_, r_)
            if alt is not None:
                r2, t2 = self.to_polar(alt[0], alt[1])
                qx, qy = self.xy(t2, r2)
                ax.plot([px, qx], [py, qy], color=RED, lw=0.9, alpha=0.35 * a,
                        ls=":", zorder=4)
                ax.scatter([qx], [qy], s=70, marker="o", facecolor="none",
                           edgecolor=RED, lw=1.0, alpha=0.5 * a, zorder=5)
            ax.scatter([px], [py], s=90 + 300 * stg, marker="o", color=RED,
                       alpha=a, edgecolor="white", lw=0.8, zorder=6)

        self._draw_legend()

    def _draw_legend(self):
        """Legend in its own axes so it can never overlap the radar."""
        a = self.ax_leg
        a.clear()
        a.axis("off")
        a.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0,
                                    boxstyle="round,pad=0.02",
                                    transform=a.transAxes,
                                    facecolor=PANEL, edgecolor=GRID, lw=0.8))
        a.text(0.5, 0.88, "KEY", transform=a.transAxes, ha="center",
               color=FG, fontsize=8.5, fontweight="bold")
        rows = [("*", LAPTOP_C, "laptop (rx)", False),
                ("s", ROUTER_C, "router", False),
                (".", "#5f7fa8", "other APs", False),
                ("o", RED, "motion top-1", False),
                ("o", RED, "runner-up", True)]
        y = 0.70
        for mk, col, lab, hollow in rows:
            a.plot([0.12], [y], marker=mk, color=col, ms=11, ls="",
                   transform=a.transAxes,
                   markerfacecolor="none" if hollow else col,
                   markeredgewidth=1.2)
            a.text(0.25, y, lab, transform=a.transAxes, color=FG,
                   fontsize=7.5, va="center")
            y -= 0.155

    def _draw_map(self, snap, now, occ):
        ax = self.ax_map; ax.clear()
        ax.set_facecolor(PANEL)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, alpha=0.15, lw=0.5); ax.set_axisbelow(True)

        # room outline
        ax.add_patch(Rectangle((0, 0), self.a.room_x, self.a.room_y,
                               fill=False, ec=GRID, lw=1.4))
        if self.rmap and self.rmap.points:
            for p in self.rmap.points:
                ax.add_patch(Rectangle((p.x - 0.35, p.y - 0.35), 0.7, 0.7,
                                       fill=False, ec="#31445e", lw=0.9, ls=":"))
        ax.scatter([self.laptop[0]], [self.laptop[1]], marker="*", s=320,
                   color=LAPTOP_C, edgecolor="white", lw=0.8, zorder=6)
        ax.annotate("laptop", (self.laptop[0], self.laptop[1]),
                    textcoords="offset points", xytext=(0, -16),
                    ha="center", color=LAPTOP_C, fontsize=8)
        rr, rt = self._router_xy()
        ax.scatter([rr], [rt], marker="s", s=150, color=ROUTER_C,
                   edgecolor="white", lw=0.9, zorder=6)
        ax.annotate("router", (rr, rt), textcoords="offset points",
                    xytext=(0, -16), ha="center", color=ROUTER_C, fontsize=8)

        for b in self.blips.live(now):
            bt, bx, by, stg, bo, alt, margin = b
            if not bo:
                continue
            age = (now - bt) / TRAIL_TTL
            a = max(0.06, 1.0 - age)
            if alt is not None:
                ax.plot([bx, alt[0]], [by, alt[1]], color=RED, lw=0.9,
                        alpha=0.4 * a, ls=":", zorder=4)
                ax.scatter([alt[0]], [alt[1]], s=170, marker="o",
                           facecolor="none", edgecolor=RED, lw=1.1,
                           alpha=0.6 * a, zorder=5)
            ax.scatter([bx], [by], s=170 + 500 * stg, color=RED, alpha=a,
                       edgecolor="white", lw=0.6, zorder=6)
        if self.rmap and self.rmap.points:
            for p in self.rmap.points:
                ax.annotate(p.label, (p.x, p.y), fontsize=6, color=DIM,
                            ha="center", va="center", zorder=3)
        ax.set_xlim(-0.8, self.a.room_x + 0.8)
        ax.set_ylim(-0.8, self.a.room_y + 0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("FINGERPRINT GRID  -  resolves to cells, not a track",
                     fontsize=9, color=FG, loc="left", pad=8)

    def _draw_evidence(self, snap, t, motion, now):
        ax = self.ax_ev; ax.clear()
        ax.set_facecolor(PANEL)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(True, alpha=0.15, lw=0.5); ax.set_axisbelow(True)

        sw = snap["sweeps"][-40:]
        keys = snap["keys"]
        if len(sw) >= 2 and keys:
            arr = np.array([[float(m.get(k, np.nan)) for k in keys] for _t, m in sw])
            f = F._fill_columns(arr)
            d = np.abs(np.diff(f, axis=0))
            L1 = d.sum(axis=1)
            stt = [x[0] for x in sw[1:]]
            ax.bar(stt, L1, width=max(5.0, (now - stt[0]) * 0.02),
                   color=[RED if v > np.median(L1) * 1.35 else AMBER for v in L1])
            ax.axhline(np.median(L1), color=DIM, ls=":", lw=1)
            ax.set_xlim(max(0.0, now - 150.0), max(now, stt[-1] + 1))
            ax.set_ylim(0, max(float(L1.max()) * 1.2, 1))
            # which anchors moved most recently
            last = d[-1]
            order = np.argsort(last)[::-1][:6]
            lbl = "  ".join(f"{keys[i]}:{last[i]:.0f}" for i in order if last[i] > 0)
            ax.set_title(f"SPATIAL EVIDENCE   sum|dRSSI| per scan   |   "
                         f"most-changed: {lbl}", fontsize=8.5,
                         color=FG, loc="left", pad=8)
        else:
            ax.text(0.5, 0.5, "waiting for scans (one every ~12 s)",
                    ha="center", va="center", color=DIM, fontsize=9,
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("seconds", fontsize=8)
        ax.set_ylabel("dB summed", fontsize=8)
        ax.tick_params(labelsize=7)

    def _draw_header(self, snap, now, occ, cur):
        """Header / phase / footer, each drawn inside its own axes.

        The header is split into three fixed x-zones (state | readout |
        telemetry) so no two strings can ever share horizontal space.
        """
        live = self.blips.live(now)
        mg = [b[6] for b in live if b[4] and b[6] == b[6]]
        marg = sorted(mg)[len(mg) // 2] if mg else None

        # ---- header band
        a = self.ax_head
        a.clear(); a.axis("off")
        # zone 1: state badge, x 0.00-0.15
        a.add_patch(FancyBboxPatch((0.005, 0.22), 0.115, 0.60,
                                   boxstyle="round,pad=0.008",
                                   transform=a.transAxes,
                                   facecolor=("#39414f" if not snap["fresh"] else
                                              RED if occ else "#1f7a4d"),
                                   edgecolor="none"))
        a.text(0.0625, 0.52, "NO DATA" if not snap["fresh"] else "MOTION" if occ else "CLEAR",
               transform=a.transAxes, ha="center", va="center",
               fontsize=14, fontweight="bold", color="white")
        # zone 2: readout, x 0.17-0.60
        a.text(0.175, 0.68, f"score {cur:+.2f}   threshold {self.st.threshold:.2f}",
               transform=a.transAxes, fontsize=11, color=FG, va="center")
        a.text(0.175, 0.26,
               f"elapsed {now:6.1f}s      contacts {len(live)}",
               transform=a.transAxes, fontsize=9.5, color=DIM, va="center")
        # zone 3: match margin, x 0.62-0.78
        if marg is not None:
            ok = self.map_ok and marg >= 5.0
            a.text(0.625, 0.68, f"match margin {marg:.1f} dB",
                   transform=a.transAxes, fontsize=9.5,
                   color=(DIM if ok else RED), va="center")
            a.text(0.625, 0.26, "top1 - top2 separation",
                   transform=a.transAxes, fontsize=8, color=DIM, va="center")
        # zone 4: telemetry, right-aligned, x 0.80-1.00
        rr = snap["meta"].get("raw_rate_hz")
        t1 = (f"RSSI {snap['meta'].get('rssi_dbm','?')} dBm    "
              f"noise {snap['meta'].get('noise_dbm','?')} dBm")
        t2 = ((f"register {rr:.0f} Hz    " if rr else "") +
              f"anchors {len(snap['keys'])}    scan ~12 s")
        a.text(0.998, 0.68, t1, transform=a.transAxes, fontsize=9,
               color=DIM, ha="right", va="center")
        a.text(0.998, 0.26, t2, transform=a.transAxes, fontsize=9,
               color=DIM, ha="right", va="center")

        # ---- phase band
        a = self.ax_phase
        a.clear(); a.axis("off")
        if self.st.phase:
            head, _, sub = self.st.phase.partition("\n")
            left = max(self.st.phase_left, 0.0)
            frac = 1.0 - min(left / max(self.st.phase_total, 1e-9), 1.0)
            a.text(0.0, 0.62, head, transform=a.transAxes, color=AMBER,
                   fontsize=15, fontweight="bold", va="center")
            a.text(0.0, 0.12, sub, transform=a.transAxes, color=FG,
                   fontsize=9.5, va="center")
            a.text(1.0, 0.62, f"{left:4.0f}s", transform=a.transAxes,
                   color=FG, fontsize=19, fontweight="bold",
                   ha="right", va="center")
            a.add_patch(FancyBboxPatch((0.0, 0.86), 1.0, 0.07,
                                       boxstyle="square,pad=0",
                                       transform=a.transAxes,
                                       facecolor=GRID, edgecolor="none"))
            a.add_patch(FancyBboxPatch((0.0, 0.86), max(frac, 0.0), 0.07,
                                       boxstyle="square,pad=0",
                                       transform=a.transAxes,
                                       facecolor=AMBER, edgecolor="none"))
        else:
            a.text(0.0, 0.5, "free-running monitor - no scripted phases",
                   transform=a.transAxes, color=DIM, fontsize=10, va="center")

        # ---- footer band
        a = self.ax_foot
        a.clear(); a.axis("off")
        if snap["error"]:
            a.text(0.0, 0.3, snap["error"], transform=a.transAxes,
                   color=AMBER, fontsize=9, va="center")
        elif self.rmap and self.rmap.points and not self.map_ok:
            a.text(0.0, 0.55,
                   "RED DOT = TOP-1 MATCH, NOT A MEASURED POSITION",
                   transform=a.transAxes, color=RED, fontsize=9.5,
                   fontweight="bold", va="center")
            a.text(0.0, 0.05, self.map_why, transform=a.transAxes,
                   color=DIM, fontsize=8, va="center")
        elif not self.rmap or not self.rmap.points:
            a.text(0.0, 0.3, "LOCALISATION OFFLINE - build a radio map:  "
                            "python3 wifisense.py calibrate",
                   transform=a.transAxes, color=AMBER, fontsize=9, va="center")
        else:
            a.text(0.0, 0.3, "Localisation validated - contacts are measured "
                              "fingerprint matches",
                   transform=a.transAxes, color=DIM, fontsize=9, va="center")

    # ------------------------------------------------------------- geometry
    def _router_xy(self):
        th = math.radians(self.a.router_bearing)
        return (self.laptop[0] + self.a.router_range * math.sin(th),
                self.laptop[1] + self.a.router_range * math.cos(th))

    def est_range(self, rssi: int) -> float:
        """Log-distance path-loss inversion, anchored on the router measurement.

        P(d) = P(r0) - 10 n log10(d/r0), with n=2.7 typical for indoor office.
        Relative geometry only -- it is not a survey, and the display labels it
        as estimated.
        """
        if rssi >= 0:
            return 0.2
        cap = self.a.room_x + self.a.room_y
        if self.lmodel is not None:
            return float(np.clip(self.lmodel.range_of(rssi), 0.3, cap))
        n = self.a.pathloss
        r0 = max(self.a.router_range, 0.3)
        p0 = -self.a.router_rssi
        d = r0 * 10 ** ((-rssi - p0) / (10.0 * n))
        return float(np.clip(d, 0.3, cap))

    def ap_bearing(self, key: str) -> float:
        """Stable pseudo-bearing for an AP.

        We cannot observe an AP's bearing from one receiver without knowing the
        AP's position, so this spreads anchors deterministically by channel hash.
        It keeps the display readable and reproducible, but it is NOT a measured
        bearing and must not be read as one.
        """
        h = sum(ord(c) * (i + 3) for i, c in enumerate(key))
        return float(h % 360)
