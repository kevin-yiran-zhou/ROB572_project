"""
Smoke test for the fusion pipeline (obstacle extraction + risk scoring).

Loads one or more frames from the configured sequence, runs SegFormer +
Depth-Anything to produce a seg mask and a metric depth map, then feeds
them into fusion.extract_obstacles and fusion.assess_frame and prints the
result.

Run:
    python test_fusion.py              # first frame of seq
    python test_fusion.py 0 10 20      # frames at indices 0, 10, 20
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the already-tuned loaders from combined.py so the test mirrors
# exactly what the GUI does at inference time.
from combined import (  # noqa: E402
    DEPTH_MODEL,
    EST_SCALE,
    INFER_MAX_SIDE,
    SEG_DEVICE,
    SEG_WEIGHTS,
    _DEPTH_PKG,
    _list_sequence_images,
    _load_rgb_u8,
    _resize_rgb_max_long_side,
    SEQ_DIR,
)
from depth._pipeline import _compute_depth_map, _load_depth_pipe  # noqa: E402
from segmentation._pipeline import compute_segmentation_mask  # noqa: E402

from fusion import assess_frame, extract_obstacles  # noqa: E402


_LEVEL_COLOR = {
    "SAFE": "\033[32m",
    "CAUTION": "\033[33m",
    "IMMEDIATE WARNING": "\033[31m",
}
_RESET = "\033[0m"


def _run_one_frame(
    path: Path,
    depth_pipe,
    seg_weights: Path,
    seg_device,
) -> None:
    import torch

    rgb = _resize_rgb_max_long_side(_load_rgb_u8(path), INFER_MAX_SIDE)
    H, W = rgb.shape[:2]

    seg_mask = compute_segmentation_mask(rgb, seg_weights, seg_device)

    depth, _ = _compute_depth_map(rgb, depth_pipe)
    depth = np.asarray(depth, dtype=np.float32) * float(EST_SCALE)

    obstacles = extract_obstacles(seg_mask, depth)
    frame_risk = assess_frame(obstacles)

    print(f"\n=== {path.name}  ({W}x{H}) ===")
    print(f"seg mask unique classes: {sorted(np.unique(seg_mask).tolist())}")
    print(
        f"depth range: min={float(depth.min()):.2f} m  "
        f"max={float(depth.max()):.2f} m  "
        f"mean={float(depth.mean()):.2f} m"
    )
    print(f"obstacles found: {len(obstacles)}")

    if not obstacles:
        print("  (no obstacle pixels above min_area)")
        print(f"  global warning: {frame_risk.global_warning}")
        return

    print(
        f"  {'idx':>3}  {'type':>5}  {'depth_p5':>9}  {'d_fwd':>7}  {'d_lat':>7}  "
        f"{'lat_exc':>8}  {'area_px':>8}  {'risk':>7}  level"
    )
    for i, r in enumerate(frame_risk.obstacle_risks):
        obs = r.obstacle
        tag = "stat" if obs.is_static else "dyn"
        lvl = str(r.warning_level)
        color = _LEVEL_COLOR.get(lvl, "")
        print(
            f"  {i:>3}  {tag:>5}  {obs.depth_p5:>7.2f} m  "
            f"{r.d_forward:>5.2f} m  {r.d_lateral:>+5.2f} m  "
            f"{r.lat_excess:>6.2f} m  {obs.pixel_area:>8d}  "
            f"{r.risk_score:>7.3f}  {color}{lvl}{_RESET}"
        )

    most = frame_risk.most_critical
    if most is not None:
        print(
            f"  most critical: obstacle at d_fwd={most.d_forward:.2f} m "
            f"d_lat={most.d_lateral:+.2f} m  score={most.risk_score:.3f}"
        )
    lvl = str(frame_risk.global_warning)
    color = _LEVEL_COLOR.get(lvl, "")
    print(f"  global warning: {color}{lvl}{_RESET}")


def main() -> None:
    seq_dir = Path(SEQ_DIR).resolve()
    paths = _list_sequence_images(seq_dir)
    if not paths:
        raise SystemExit(f"No sequence images found under {seq_dir}")

    if len(sys.argv) > 1:
        indices = [int(a) for a in sys.argv[1:]]
    else:
        indices = [0]

    for i in indices:
        if not 0 <= i < len(paths):
            print(f"skipping index {i} (only {len(paths)} frames available)")
            continue

    import torch

    seg_weights = Path(SEG_WEIGHTS).resolve()
    seg_device = (
        torch.device(SEG_DEVICE)
        if SEG_DEVICE
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"loading depth pipe: {DEPTH_MODEL}")
    depth_pipe, label = _load_depth_pipe(_DEPTH_PKG, DEPTH_MODEL)
    print(f"depth device: {label}")
    print(f"seg device:   {seg_device}")
    print(f"frames to run: {indices}")

    for i in indices:
        if 0 <= i < len(paths):
            _run_one_frame(paths[i], depth_pipe, seg_weights, seg_device)


if __name__ == "__main__":
    main()
