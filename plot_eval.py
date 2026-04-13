"""
Generate report-quality evaluation figures from eval_detection.py outputs.

Figures produced (depending on available data):
1. Per-class detection P/R/F1 grouped bar chart  (needs detection.csv)
2. Latency breakdown pie + per-module bar         (needs latency.csv)
3. Latency CDF with p50/p95/p99                   (needs latency.csv)
4. Depth temporal stability histogram              (needs timeseries.csv)
5. Closing velocity distribution                   (needs timeseries.csv)
6. Warning level distribution per frame            (needs timeseries.csv)
7. Track duration distribution                     (needs timeseries.csv)

Usage
-----
python plot_eval.py --eval-dir eval_results_val --save
python plot_eval.py --eval-dir eval_results_test_seq30 --save
python plot_eval.py --eval-dir eval_results_val,eval_results_test_seq30 --save  # merge
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.titlesize"] = 13
matplotlib.rcParams["figure.dpi"] = 120

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _float_col(rows: list[dict], col: str) -> np.ndarray:
    vals = []
    for r in rows:
        v = r.get(col, "")
        if v != "":
            vals.append(float(v))
    return np.array(vals, dtype=np.float64)


# ---------------------------------------------------------------------------
# Fig 1: Per-class Detection P / R / F1
# ---------------------------------------------------------------------------
def plot_detection(det_rows: list[dict], save_dir: Path | None):
    if not det_rows:
        return
    # Skip "Overall" for grouped bar; show it as annotation
    classes = [r for r in det_rows if r["class_name"] != "Overall"]
    overall = [r for r in det_rows if r["class_name"] == "Overall"]
    if not classes:
        return

    names = [r["class_name"] for r in classes]
    P = [float(r["precision"]) for r in classes]
    R = [float(r["recall"]) for r in classes]
    F = [float(r["f1"]) for r in classes]

    x = np.arange(len(names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, P, w, label="Precision", color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.bar(x, R, w, label="Recall", color="#55A868", edgecolor="black", linewidth=0.4)
    ax.bar(x + w, F, w, label="F1", color="#C44E52", edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Obstacle Detection Performance (IoU ≥ 0.5)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Annotate overall F1
    if overall:
        o = overall[0]
        ax.text(0.02, 0.95,
                f"Overall: P={float(o['precision']):.3f}  R={float(o['recall']):.3f}  F1={float(o['f1']):.3f}",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))

    plt.tight_layout()
    _save_or_show(fig, save_dir, "detection_per_class.png")


# ---------------------------------------------------------------------------
# Fig 2: Latency breakdown (stacked bar + pie)
# ---------------------------------------------------------------------------
def plot_latency_breakdown(lat_rows: list[dict], save_dir: Path | None):
    if not lat_rows:
        return

    components = ["seg_ms", "depth_ms", "fusion_ms", "track_ms", "risk_ms"]
    labels = ["Segmentation", "Depth", "Fusion", "Tracking", "Risk"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    means = [_float_col(lat_rows, c).mean() for c in components]
    total_mean = sum(means)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1.6, 1]})

    # Bar chart
    bars = ax1.barh(labels[::-1], means[::-1], color=colors[::-1],
                     edgecolor="black", linewidth=0.4)
    for bar, m in zip(bars, means[::-1]):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{m:.1f} ms", va="center", fontsize=10)
    ax1.set_xlabel("Time (ms)")
    ax1.set_title(f"Per-Module Latency (total mean: {total_mean:.0f} ms)")
    ax1.set_xlim(0, max(means) * 1.3)

    # Pie chart (only show components > 1%)
    filtered = [(l, m, c) for l, m, c in zip(labels, means, colors) if m / total_mean > 0.01]
    pie_labels = [f"{l}\n{m:.0f}ms ({m/total_mean*100:.0f}%)" for l, m, _ in filtered]
    pie_vals = [m for _, m, _ in filtered]
    pie_colors = [c for _, _, c in filtered]
    ax2.pie(pie_vals, labels=pie_labels, colors=pie_colors, startangle=90,
            textprops={"fontsize": 9})
    ax2.set_title("Proportion")

    plt.tight_layout()
    _save_or_show(fig, save_dir, "latency_breakdown.png")


# ---------------------------------------------------------------------------
# Fig 3: Latency CDF
# ---------------------------------------------------------------------------
def plot_latency_cdf(lat_rows: list[dict], save_dir: Path | None):
    if not lat_rows:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    total = np.sort(_float_col(lat_rows, "total_ms"))
    # Skip first few frames (cold start) for cleaner CDF
    if len(total) > 10:
        total_warm = np.sort(total[5:])
    else:
        total_warm = total
    cdf = np.arange(1, len(total_warm) + 1) / len(total_warm)
    ax.plot(total_warm, cdf, linewidth=2, color="#4C72B0", label="Total (excl. warmup)")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_title("End-to-End Latency CDF")
    ax.grid(True, alpha=0.3)

    # Percentile markers
    for p, color, ls in [(50, "#55A868", "--"), (95, "#C44E52", ":"), (99, "#8172B2", "-.")]:
        val = np.percentile(total_warm, p)
        ax.axvline(val, color=color, linestyle=ls, alpha=0.7, linewidth=1.5)
        ax.text(val + 1, 0.3 + (p - 50) / 150, f"p{p} = {val:.0f} ms",
                fontsize=9, color=color, rotation=0)

    fps_mean = 1000.0 / total_warm.mean()
    ax.text(0.98, 0.05, f"Mean FPS: {fps_mean:.1f}", transform=ax.transAxes,
            fontsize=10, ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))

    ax.legend(fontsize=10)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "latency_cdf.png")


# ---------------------------------------------------------------------------
# Fig 4: Depth temporal stability
# ---------------------------------------------------------------------------
def plot_depth_stability(ts_rows: list[dict], save_dir: Path | None):
    if not ts_rows:
        return

    # Compute frame-to-frame depth change per track
    by_track: dict[int, list[tuple[int, float]]] = {}
    for r in ts_rows:
        d = r.get("depth_m", "")
        if d == "":
            continue
        tid = int(r["track_id"])
        by_track.setdefault(tid, []).append((int(r["frame"]), float(d)))

    delta_depths = []
    for tid, pts in by_track.items():
        pts.sort()
        for i in range(1, len(pts)):
            if pts[i][0] - pts[i - 1][0] == 1:  # consecutive frames only
                delta_depths.append(abs(pts[i][1] - pts[i - 1][1]))

    if not delta_depths:
        return

    dd = np.array(delta_depths)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dd, bins=50, color="#4C72B0", edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.axvline(np.median(dd), color="#C44E52", linestyle="--", linewidth=2,
               label=f"Median: {np.median(dd):.2f} m")
    ax.axvline(np.percentile(dd, 95), color="#55A868", linestyle=":", linewidth=2,
               label=f"p95: {np.percentile(dd, 95):.2f} m")

    ax.set_xlabel("Frame-to-Frame Depth Change (m)")
    ax.set_ylabel("Count")
    ax.set_title("Depth Estimation Temporal Stability (consecutive frames, same track)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "depth_stability.png")


# ---------------------------------------------------------------------------
# Fig 5: Closing velocity distribution
# ---------------------------------------------------------------------------
def plot_velocity_distribution(ts_rows: list[dict], save_dir: Path | None):
    if not ts_rows:
        return

    vels = _float_col(ts_rows, "v_closing_ms")
    if len(vels) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Split approaching vs receding
    approaching = vels[vels > 0.1]
    receding = vels[vels < -0.1]
    near_zero = vels[(vels >= -0.1) & (vels <= 0.1)]

    bins = np.linspace(vels.min() - 0.5, vels.max() + 0.5, 40)
    ax.hist(vels, bins=bins, color="#4C72B0", edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.axvline(0, color="gray", linestyle="-", alpha=0.5)

    ax.set_xlabel("Closing Velocity (m/s)")
    ax.set_ylabel("Count")
    ax.set_title("Closing Velocity Distribution Across All Tracks")

    # Stats annotation
    stats_text = (f"Approaching (>0.1): {len(approaching)} ({len(approaching)/len(vels)*100:.0f}%)\n"
                  f"Stationary (±0.1): {len(near_zero)} ({len(near_zero)/len(vels)*100:.0f}%)\n"
                  f"Receding (<-0.1): {len(receding)} ({len(receding)/len(vels)*100:.0f}%)")
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.8))

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig, save_dir, "velocity_distribution.png")


# ---------------------------------------------------------------------------
# Fig 6: Warning level distribution
# ---------------------------------------------------------------------------
def plot_warning_distribution(ts_rows: list[dict], save_dir: Path | None):
    """Per-frame warning level distribution (max level per frame)."""
    if not ts_rows:
        return

    # Get max warning level per frame
    frame_levels: dict[int, str] = {}
    level_order = {"SAFE": 0, "CAUTION": 1, "IMMEDIATE_WARNING": 2}
    for r in ts_rows:
        fi = int(r["frame"])
        lv = r.get("warning_level", "SAFE")
        if fi not in frame_levels or level_order.get(lv, 0) > level_order.get(frame_levels[fi], 0):
            frame_levels[fi] = lv

    if not frame_levels:
        return

    levels = list(frame_levels.values())
    counts = {
        "SAFE": levels.count("SAFE"),
        "CAUTION": levels.count("CAUTION"),
        "IMMEDIATE": levels.count("IMMEDIATE_WARNING"),
    }
    # Also count frames with no tracked obstacles as SAFE
    all_frames_with_tracks = set(frame_levels.keys())
    # We don't know total frames from timeseries alone, so just use what we have

    total = sum(counts.values())
    if total == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(counts.keys())
    vals = [counts[n] for n in names]
    colors_map = {"SAFE": "#55A868", "CAUTION": "#DDAA33", "IMMEDIATE": "#C44E52"}
    bar_colors = [colors_map[n] for n in names]

    bars = ax.bar(names, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, vals):
        pct = v / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v} ({pct:.0f}%)", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Number of Frames")
    ax.set_title(f"Warning Level Distribution ({total} frames with tracked obstacles)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "warning_distribution.png")


# ---------------------------------------------------------------------------
# Fig 7: Track duration distribution
# ---------------------------------------------------------------------------
def plot_track_duration(ts_rows: list[dict], save_dir: Path | None):
    if not ts_rows:
        return

    by_track: dict[int, list[int]] = {}
    for r in ts_rows:
        tid = int(r["track_id"])
        by_track.setdefault(tid, []).append(int(r["frame"]))

    durations = []
    for tid, frames in by_track.items():
        durations.append(max(frames) - min(frames) + 1)

    if not durations:
        return

    dur = np.array(durations)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dur, bins=max(10, min(30, len(dur))), color="#8172B2",
            edgecolor="black", linewidth=0.4, alpha=0.85)
    ax.axvline(np.median(dur), color="#C44E52", linestyle="--", linewidth=2,
               label=f"Median: {np.median(dur):.0f} frames")
    ax.axvline(np.mean(dur), color="#55A868", linestyle=":", linewidth=2,
               label=f"Mean: {np.mean(dur):.0f} frames")

    ax.set_xlabel("Track Duration (frames)")
    ax.set_ylabel("Count")
    ax.set_title(f"Track Duration Distribution ({len(durations)} total tracks)")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_dir, "track_duration.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save_or_show(fig, save_dir: Path | None, filename: str):
    if save_dir:
        path = save_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate report-quality evaluation plots")
    parser.add_argument("--eval-dir", type=str, default="eval_results",
                        help="Comma-separated eval directories to merge data from")
    parser.add_argument("--save", action="store_true",
                        help="Save PNGs to first eval-dir instead of showing interactively")
    args = parser.parse_args()

    dirs = [Path(d.strip()) for d in args.eval_dir.split(",")]
    save_dir = dirs[0] if args.save else None

    # Merge data from all directories
    det_rows: list[dict] = []
    lat_rows: list[dict] = []
    ts_rows: list[dict] = []

    for d in dirs:
        det_rows.extend(_load_csv(d / "detection.csv"))
        lat_rows.extend(_load_csv(d / "latency.csv"))
        ts_rows.extend(_load_csv(d / "timeseries.csv"))

    print(f"Loaded: {len(det_rows)} detection rows, {len(lat_rows)} latency rows, {len(ts_rows)} timeseries rows")

    if det_rows:
        print("\nPlotting detection P/R/F1...")
        plot_detection(det_rows, save_dir)

    if lat_rows:
        print("Plotting latency breakdown...")
        plot_latency_breakdown(lat_rows, save_dir)
        print("Plotting latency CDF...")
        plot_latency_cdf(lat_rows, save_dir)

    if ts_rows:
        print("Plotting depth stability...")
        plot_depth_stability(ts_rows, save_dir)
        print("Plotting velocity distribution...")
        plot_velocity_distribution(ts_rows, save_dir)
        print("Plotting warning distribution...")
        plot_warning_distribution(ts_rows, save_dir)
        print("Plotting track duration...")
        plot_track_duration(ts_rows, save_dir)

    if save_dir:
        print(f"\nAll plots saved to {save_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
