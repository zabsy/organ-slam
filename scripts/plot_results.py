"""Plot SLAM results against ground truth (reads out/<run>/eval.npz)."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GT, EST, MAP = "#2a78d6", "#eb6834", "#b9b8b2"


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/synthetic")
    ap.add_argument("--seq", default="data/synthetic")
    ap.add_argument("--fig-dir", default="docs")
    args = ap.parse_args()
    run, figs = Path(args.run), Path(args.fig_dir)
    figs.mkdir(parents=True, exist_ok=True)
    ev = np.load(run / "eval.npz")
    m = json.loads((run / "metrics.json").read_text())
    gt, est, pts = ev["gt"], ev["aligned"], ev["map_aligned"]

    fig = plt.figure(figsize=(12, 7.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.2)
    views = [(fig.add_subplot(gs[0, 0]), 0, "x (mm)", "Top view"),
             (fig.add_subplot(gs[0, 1]), 1, "y (mm)", "Side view")]
    for ax, axis, label, title in views:
        style(ax, "z, along the lumen (mm)", label)
        if len(pts):
            ax.scatter(pts[:, 2], pts[:, axis], s=1.5, color=MAP, linewidths=0, label="Map points (wall)")
        ax.plot(gt[:, 2], gt[:, axis], color=GT, linewidth=2, label="Ground truth camera path")
        ax.plot(est[:, 2], est[:, axis], color=EST, linewidth=2, linestyle=(0, (4, 2)), label="SLAM estimate")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim(gt[:, 2].min() - 25, gt[:, 2].max() + 45)
        ax.set_ylim(gt[:, axis].min() - 25, gt[:, axis].max() + 25)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
    handles, labels = views[0][0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.9, 0.955), ncol=3, frameon=False,
                     fontsize=9.5, markerscale=5)
    for text in leg.get_texts():
        text.set_color(MUTED)

    ax = fig.add_subplot(gs[1, :])
    style(ax, "Frame", "Position error (mm)")
    ax.plot(ev["frame_ids"], ev["err"], color=EST, linewidth=2)
    ax.set_ylim(0, None)
    ax.set_xlim(ev["frame_ids"][0], ev["frame_ids"][-1])
    ax.set_title(f"Camera position error per frame (RMSE {m['ate_rmse']:.2f} mm over a "
                 f"{m['path_length']:.0f} mm path, {m['ate_pct']:.2f}%)", color=INK, fontsize=11, loc="left")
    fig.suptitle("Monocular SLAM vs ground truth, synthetic colon", color=INK, fontsize=13, x=0.125, ha="left", y=0.965)
    fig.savefig(figs / "results.png", dpi=160, facecolor=SURFACE, bbox_inches="tight")

    rgb = sorted((Path(args.seq) / "rgb").glob("*.png"))
    if rgb:
        pick = np.linspace(0, len(rgb) - 1, 5).astype(int)
        strip = np.hstack([cv2.imread(str(rgb[i])) for i in pick])
        cv2.imwrite(str(figs / "frames.png"), strip)
    print(f"wrote {figs / 'results.png'}")


if __name__ == "__main__":
    main()
