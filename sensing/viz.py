"""Plotting for wifisense: traces, detection overlay, radio map, occupancy heatmap."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def _grid(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title, fontsize=10, loc="left")


def plot_channels(session, path, res=None):
    """Four-panel view of both raw channels and the derived decision."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 2]})
    ft = np.array([f.t for f in session.fine])
    if len(ft) == 0:
        plt.close(fig)
        return
    rssi = np.array([f.rssi for f in session.fine])
    span = np.array([f.mx - f.mn for f in session.fine])
    chg = np.array([f.chg for f in session.fine])

    axes[0].plot(ft, rssi, lw=0.8, color="#1f77b4")
    axes[0].fill_between(ft, np.array([f.mn for f in session.fine]),
                         np.array([f.mx for f in session.fine]),
                         color="#1f77b4", alpha=0.25, lw=0)
    _grid(axes[0], ylabel="dBm", title="Fine channel: associated-link RSSI (band = min/max within bucket)")

    axes[1].plot(ft, span, lw=0.8, color="#d62728")
    _grid(axes[1], ylabel="dB", title="RSSI excursion within bucket (1 dB quantisation visible as steps)")

    axes[2].plot(ft, chg, lw=0.8, color="#2ca02c")
    _grid(axes[2], ylabel="transitions", title="Register transitions per bucket")

    sw_t = [s.t for s in session.sweeps]
    if sw_t:
        axes[2].axvline(sw_t[0], color="grey", ls=":", lw=0.8)
        for t in sw_t[1:]:
            axes[2].axvline(t, color="grey", ls=":", lw=0.8)
        axes[2].text(sw_t[0], axes[2].get_ylim()[1], " sweeps ~12 s",
                     fontsize=7, color="grey", va="top")
    if res and res.detections:
        dt = np.array([d.t for d in res.detections])
        mo = np.array([d.motion for d in res.detections])
        axes[3].plot(dt, mo, lw=0.9, color="#333333", label="motion score")
        axes[3].axhline(res.threshold, color="#d62728", ls="--", lw=1,
                        label=f"threshold {res.threshold:.2f}")
        occ = np.array([d.occupancy for d in res.detections])
        axes[3].fill_between(dt, 0, mo, where=occ, color="#ff7f0e", alpha=0.35,
                            label="occupied")
        axes[3].legend(fontsize=7, loc="upper right")
    _grid(axes[3], xlabel="time (s)", ylabel="score",
          title="Fused motion decision")
    axes[3].set_xlim(ft[0], ft[-1])

    fig.suptitle("wifisense capture", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_ap_heights(session, path, last_n: int = 1):
    """RSSI of every visible anchor at the most recent sweeps."""
    if not session.sweeps:
        return
    sw = session.sweeps[-last_n:]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n = len(sw)
    labels, data = [], []
    for p in sw:
        for k, v in sorted(p.aps.items(), key=lambda kv: -kv[1]):
            labels.append(k)
            data.append(v)
    cmap = plt.get_cmap("viridis")
    for i, (lb, v) in enumerate(zip(labels, data)):
        c = cmap(i / max(len(labels) - 1, 1))
        ax.barh(i, v, color=c, height=0.7)
        ax.text(v - 1, i, f"{v}", va="center", ha="right", fontsize=7, color="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="k", lw=0.6)
    _grid(ax, xlabel="RSSI (dBm)",
          title="Multi-AP anchors (spatial sensing array) - most recent sweep")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_radio_map(rmap, path):
    fig, ax = plt.subplots(figsize=(7.5, 7))
    if not rmap.points:
        ax.text(0.5, 0.5, "empty radio map", ha="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return
    xs = [p.x for p in rmap.points]
    ys = [p.y for p in rmap.points]
    sc = ax.scatter(xs, ys, c=range(len(xs)), cmap="turbo", s=340,
                    edgecolor="k", zorder=3)
    for p in rmap.points:
        ax.annotate(p.label or f"{p.x:.1f},{p.y:.1f}", (p.x, p.y),
                    fontsize=7, ha="center", va="center", zorder=4)
    _grid(ax, xlabel="x (m)", ylabel="y (m)",
          title=f"Surveyed fingerprint cells (n={len(rmap.points)})")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_occupancy(rmap, visits, path, title="Fingerprint matches per cell"):
    """Render accepted fingerprint matches, without inferring presence.

    visits: dict[(x,y)] -> count of scans attributed to that cell.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    if not rmap.points:
        ax.text(0.5, 0.5, "no radio map - run `wifisense calibrate` first",
                ha="center", fontsize=9)
        ax.set_axis_off()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return
    xs = np.array([p.x for p in rmap.points])
    ys = np.array([p.y for p in rmap.points])
    v = np.array([visits.get((x, y), 0) for x, y in zip(xs, ys)],
                 dtype=float)
    cmap = LinearSegmentedColormap.from_list("occ", ["#08306b", "#2171b5", "#6baed6",
                                                    "#c6dbef", "#fee391", "#ec7014",
                                                    "#cc4c02", "#7f0000"])
    sc = ax.scatter(xs, ys, c=v, cmap=cmap, s=1500, marker="s",
                    edgecolor="k", linewidth=0.6)
    for x, y, vv in zip(xs, ys, v):
        ax.annotate(f"{int(vv)}", (x, y), ha="center", va="center", fontsize=9,
                    color="white" if vv > v.max() * 0.55 else "black")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("matched scans", fontsize=8)
    _grid(ax, xlabel="x (m)", ylabel="y (m)", title=title)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_trace(series, path, title, ylabel):
    fig, ax = plt.subplots(figsize=(13, 3.2))
    if len(series):
        t = np.array([s[0] for s in series])
        v = np.array([s[1] for s in series])
        ax.plot(t, v, lw=0.8, drawstyle="steps-post")
        ax.set_xlim(t[0], t[-1])
    _grid(ax, xlabel="time (s)", ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
