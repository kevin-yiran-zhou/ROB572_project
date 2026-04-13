"""
Evaluation script for the risk-aware ASV pipeline.

Supports three evaluation modes (can run together in one pass):

1. **Detection P/R/F1** (requires LaRS panoptic annotations)
   --panoptic-json  / --panoptic-mask-dir  / --image-dir
   Compares our obstacle extraction against panoptic instance bboxes.

2. **Risk + tracking time-series export** (no GT needed)
   Writes per-frame CSV: frame, track_id, class_id, depth, v_closing, ttc,
   risk_score, warning_level.  Used for case-study plots and ablation.

3. **Latency profiling** (always on)
   Per-frame breakdown: seg_ms, depth_ms, fusion_ms, track_ms, total_ms.

Usage examples
--------------
# Run on a sequence directory (mode 2+3 only, no GT):
python eval_detection.py --seq-dir lars_v1.0.0_images_seq/val/images_seq --prefix davimar_seq_08

# Run detection eval with panoptic GT:
python eval_detection.py --image-dir lars_v1.0.0_images/val/images \
    --panoptic-json lars_v1.0.0_annotations/val/panoptic_annotations.json \
    --panoptic-mask-dir lars_v1.0.0_annotations/val/panoptic_masks

# Specify output directory:
python eval_detection.py --seq-dir ... --prefix ... --out-dir eval_results/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Lazy torch import (heavy)
# ---------------------------------------------------------------------------
_torch = None
_seg_device = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_device():
    global _seg_device
    if _seg_device is None:
        torch = _get_torch()
        _seg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _seg_device


# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from depth._constants import DEFAULT_EST_SCALE
from depth._pipeline import _compute_depth_map, _load_depth_pipe, load_image_for_depth
from segmentation._pipeline import (
    compute_segmentation_and_boundary,
    get_class_names,
)
from fusion import (
    assess_frame,
    extract_obstacles,
    extract_obstacles_multiclass,
)
from fusion.obstacle import Obstacle, IGNORE_CLASSES
from tracking import ObstacleTracker, TrackedObstacle

# ---------------------------------------------------------------------------
# Default config (mirrors combined.py)
# ---------------------------------------------------------------------------
_DEPTH_PKG = _ROOT / "depth"
_SEG_PKG = _ROOT / "segmentation"

DEPTH_MODEL = "small"
EST_SCALE = DEFAULT_EST_SCALE
INFER_MAX_SIDE = 552
SEG_WEIGHTS = _SEG_PKG / "model" / "segformer_instance_aware_best.pth"

HFOV_DEG = 70.0
W_BOAT = 2.0
LAT_MARGIN = 1.0
D_SAFE = 12.0
L_SAFE = 2.0
V_REF = 3.0
ALPHA_V = 1.0

TRACK_MAX_AGE = 5
TRACK_MIN_HITS = 2
TRACK_IOU_THRESH = 0.20
TRACK_FPS = 10.0

# Panoptic category mapping (same as train_segformer_instance_aware.py)
CATEGORY_ID_TO_CLASS_NAME = {
    1: "Static Obstacle",
    3: "Water",
    5: "Sky",
    11: "Boat",
    12: "Boat",
    13: "Boat",
    14: "Buoy",
    15: "Swimmer",
    16: "Animal",
    17: "Float",
    19: "Other",
}
CLASS_NAMES = [
    "Static Obstacle", "Water", "Sky", "Boat", "Buoy",
    "Swimmer", "Animal", "Float", "Other",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------
def _load_rgb_u8(path: Path) -> np.ndarray:
    arr = load_image_for_depth(path)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _resize_max_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return rgb
    h, w = rgb.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale >= 1.0:
        return rgb
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return np.array(Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS))


# ---------------------------------------------------------------------------
# Panoptic GT → instance bboxes
# ---------------------------------------------------------------------------
def rgb2id(color: np.ndarray) -> np.ndarray:
    color = np.asarray(color, dtype=np.int64)
    return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]


@dataclass
class GTBox:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str


def load_panoptic_gt(
    panoptic_json_path: Path,
    panoptic_mask_dir: Path,
) -> dict[str, list[GTBox]]:
    """Load panoptic annotations → {filename: [GTBox, ...]}."""
    with open(panoptic_json_path) as f:
        data = json.load(f)

    ann_by_file: dict[str, list[GTBox]] = {}
    for ann in data["annotations"]:
        fname = ann["file_name"]
        mask_path = panoptic_mask_dir / fname
        if not mask_path.exists():
            continue

        panoptic_rgb = np.array(Image.open(mask_path).convert("RGB"))
        panoptic_ids = rgb2id(panoptic_rgb)

        boxes: list[GTBox] = []
        for seg in ann["segments_info"]:
            cat_id = seg["category_id"]
            if cat_id not in CATEGORY_ID_TO_CLASS_NAME:
                continue
            cls_name = CATEGORY_ID_TO_CLASS_NAME[cat_id]
            cls_idx = CLASS_TO_IDX[cls_name]
            # Skip Water and Sky
            if cls_idx in IGNORE_CLASSES:
                continue

            seg_id = seg["id"]
            mask = panoptic_ids == seg_id
            if mask.sum() == 0:
                continue

            ys, xs = np.where(mask)
            x1, y1 = int(xs.min()), int(ys.min())
            x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
            boxes.append(GTBox(x1=x1, y1=y1, x2=x2, y2=y2,
                               class_id=cls_idx, class_name=cls_name))

        # Map annotation filename to image filename (annotations use .png)
        img_base = os.path.splitext(fname)[0]
        ann_by_file[img_base] = boxes

    return ann_by_file


# ---------------------------------------------------------------------------
# IoU matching for detection evaluation
# ---------------------------------------------------------------------------
def _iou(a: tuple, b: tuple) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / max(union, 1e-10)


@dataclass
class DetectionMetrics:
    """Accumulates TP/FP/FN for P/R/F1 calculation."""
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-10)


def match_detections(
    pred_boxes: list[tuple[int, int, int, int, int]],  # (x1,y1,x2,y2,class_id)
    gt_boxes: list[GTBox],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy IoU matching. Returns (tp, fp, fn)."""
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp = 0

    for px1, py1, px2, py2, _ in pred_boxes:
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou_val = _iou((px1, py1, px2, py2), (gt.x1, gt.y1, gt.x2, gt.y2))
            if iou_val > best_iou:
                best_iou = iou_val
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            tp += 1
            matched_gt.add(best_gi)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Collect image paths
# ---------------------------------------------------------------------------
def collect_seq_paths(seq_dir: Path, prefixes: list[str], suffixes: list[str] = [".jpg"]) -> list[Path]:
    """Collect and sort image paths from a sequence directory by prefix groups."""
    import re
    suffix_set = {s.lower() for s in suffixes}
    all_paths: list[Path] = []
    for prefix in prefixes:
        group = sorted(
            [p for p in seq_dir.iterdir()
             if p.suffix.lower() in suffix_set and p.name.startswith(prefix)],
            key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.stem)],
        )
        all_paths.extend(group)
    return all_paths


def collect_image_dir_paths(image_dir: Path, suffixes: list[str] = [".jpg", ".png"]) -> list[Path]:
    """Collect all images from a flat directory."""
    suffix_set = {s.lower() for s in suffixes}
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in suffix_set)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_eval(args: argparse.Namespace) -> None:
    torch = _get_torch()
    device = _get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect images ----
    if args.seq_dir:
        prefixes = [p.strip() for p in args.prefix.split(",")]
        paths = collect_seq_paths(Path(args.seq_dir), prefixes)
        print(f"Sequence mode: {len(paths)} images from {args.seq_dir}")
    elif args.image_dir:
        paths = collect_image_dir_paths(Path(args.image_dir))
        print(f"Image-dir mode: {len(paths)} images from {args.image_dir}")
    else:
        print("ERROR: Provide --seq-dir (with --prefix) or --image-dir")
        sys.exit(1)

    if not paths:
        print("No images found.")
        sys.exit(1)

    # ---- Load panoptic GT if available ----
    gt_data: dict[str, list[GTBox]] | None = None
    if args.panoptic_json and args.panoptic_mask_dir:
        print("Loading panoptic GT annotations...")
        gt_data = load_panoptic_gt(
            Path(args.panoptic_json), Path(args.panoptic_mask_dir),
        )
        print(f"  Loaded GT for {len(gt_data)} images")

    # ---- Load models ----
    print(f"Loading models (seg: {SEG_WEIGHTS.name}, depth: {DEPTH_MODEL})...")
    depth_pipe, _ = _load_depth_pipe(_DEPTH_PKG, DEPTH_MODEL)
    # Warm up seg model
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    compute_segmentation_and_boundary(dummy, SEG_WEIGHTS, device)
    class_names = get_class_names(SEG_WEIGHTS, device)
    is_multiclass = class_names is not None
    print(f"  Model type: {'multi-class (' + str(len(class_names)) + ')' if is_multiclass else '3-class legacy'}")

    # ---- Tracker ----
    tracker = ObstacleTracker(
        max_age=TRACK_MAX_AGE,
        min_hits=TRACK_MIN_HITS,
        iou_threshold=TRACK_IOU_THRESH,
        fps=TRACK_FPS,
    )

    # ---- Metrics accumulators ----
    overall_det = DetectionMetrics()
    per_class_det: dict[str, DetectionMetrics] = defaultdict(DetectionMetrics)

    # ---- CSV writers ----
    ts_path = out_dir / "timeseries.csv"
    lat_path = out_dir / "latency.csv"
    ts_file = open(ts_path, "w", newline="")
    lat_file = open(lat_path, "w", newline="")

    ts_writer = csv.writer(ts_file)
    ts_writer.writerow([
        "frame", "track_id", "class_id", "class_name",
        "depth_m", "v_closing_ms", "ttc_s",
        "risk_score", "warning_level",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    ])

    lat_writer = csv.writer(lat_file)
    lat_writer.writerow([
        "frame", "seg_ms", "depth_ms", "fusion_ms", "track_ms", "risk_ms",
        "total_ms", "n_obstacles", "n_tracked",
    ])

    id_switches = 0
    prev_track_ids: set[int] = set()

    print(f"\nRunning evaluation on {len(paths)} frames...")
    for fi, path in enumerate(paths):
        t_total_0 = time.perf_counter()

        rgb = _load_rgb_u8(path)
        rgb_small = _resize_max_side(rgb, INFER_MAX_SIDE)

        # -- Segmentation --
        t0 = time.perf_counter()
        mask, boundary = compute_segmentation_and_boundary(rgb_small, SEG_WEIGHTS, device)
        seg_ms = (time.perf_counter() - t0) * 1000

        # -- Depth --
        t0 = time.perf_counter()
        depth, _ = _compute_depth_map(rgb_small, depth_pipe)
        depth = np.asarray(depth, dtype=np.float32)
        if EST_SCALE != 1.0:
            depth = depth * float(EST_SCALE)
        depth_ms = (time.perf_counter() - t0) * 1000

        # -- Fusion (obstacle extraction) --
        t0 = time.perf_counter()
        if boundary is not None:
            obstacles = extract_obstacles_multiclass(
                mask, depth, return_masks=False, boundary_prob=boundary,
            )
        else:
            obstacles = extract_obstacles(mask, depth, return_masks=False)
        fusion_ms = (time.perf_counter() - t0) * 1000

        # -- Tracking --
        t0 = time.perf_counter()
        tracked = tracker.update(obstacles)
        track_ms = (time.perf_counter() - t0) * 1000

        # -- Risk --
        t0 = time.perf_counter()
        frame_risk = assess_frame(
            obstacles,
            hfov_deg=HFOV_DEG, w_boat=W_BOAT, lat_margin=LAT_MARGIN,
            d_safe=D_SAFE, l_safe=L_SAFE,
            v_ref=V_REF, alpha_v=ALPHA_V, tracked=tracked,
        )
        risk_ms = (time.perf_counter() - t0) * 1000

        total_ms = (time.perf_counter() - t_total_0) * 1000

        # -- ID switch counting --
        cur_track_ids = {t.track_id for t in tracked}
        # A simple proxy: tracks that existed last frame but vanished
        # (while new ones appeared) suggest an ID switch.
        # More precise MOTA-style counting requires GT association.
        if fi > 0:
            lost = prev_track_ids - cur_track_ids
            new = cur_track_ids - prev_track_ids
            id_switches += min(len(lost), len(new))
        prev_track_ids = cur_track_ids

        # -- Detection eval (if GT available) --
        if gt_data is not None:
            img_base = os.path.splitext(path.name)[0]
            gt_boxes = gt_data.get(img_base, [])

            # Scale GT boxes to match resized image
            h_orig, w_orig = rgb.shape[:2]
            h_small, w_small = rgb_small.shape[:2]
            sx, sy = w_small / w_orig, h_small / h_orig
            scaled_gt = [
                GTBox(
                    x1=int(g.x1 * sx), y1=int(g.y1 * sy),
                    x2=int(g.x2 * sx), y2=int(g.y2 * sy),
                    class_id=g.class_id, class_name=g.class_name,
                )
                for g in gt_boxes
            ]

            pred_boxes = [
                (o.x1, o.y1, o.x2, o.y2, o.class_id if o.class_id is not None else 0)
                for o in obstacles
            ]

            tp, fp, fn = match_detections(pred_boxes, scaled_gt, iou_threshold=0.5)
            overall_det.tp += tp
            overall_det.fp += fp
            overall_det.fn += fn

            # Per-class (GT side)
            for gt in scaled_gt:
                # Check if this GT box was matched
                matched = False
                for px1, py1, px2, py2, _ in pred_boxes:
                    if _iou((px1, py1, px2, py2), (gt.x1, gt.y1, gt.x2, gt.y2)) >= 0.5:
                        matched = True
                        break
                if matched:
                    per_class_det[gt.class_name].tp += 1
                else:
                    per_class_det[gt.class_name].fn += 1

            # FP counted per-class from pred side
            for px1, py1, px2, py2, pcls in pred_boxes:
                best_match = False
                for gt in scaled_gt:
                    if _iou((px1, py1, px2, py2), (gt.x1, gt.y1, gt.x2, gt.y2)) >= 0.5:
                        best_match = True
                        break
                if not best_match:
                    cls_name = CLASS_NAMES[pcls] if pcls < len(CLASS_NAMES) else "Unknown"
                    per_class_det[cls_name].fp += 1

        # -- Write time-series CSV --
        for t in tracked:
            obs = t.obstacle
            cls_name = CLASS_NAMES[obs.class_id] if obs.class_id is not None and obs.class_id < len(CLASS_NAMES) else "obstacle"
            ts_writer.writerow([
                fi, t.track_id,
                obs.class_id if obs.class_id is not None else -1,
                cls_name,
                f"{obs.effective_depth:.2f}",
                f"{t.v_closing:.2f}" if t.v_closing is not None else "",
                f"{t.ttc:.1f}" if t.ttc is not None else "",
                f"{0.0:.3f}",  # placeholder — we'll fill from frame_risk below
                "",
                obs.x1, obs.y1, obs.x2, obs.y2,
            ])

        # Also write risk per obstacle (for all obstacles, not just tracked)
        # We already wrote tracked; for risk we overwrite the score column
        # Actually, let's write risk to a separate row set — simpler approach:
        # The timeseries already has tracked obstacles. Let's update risk info
        # by matching track_id to obstacle_risk.

        # -- Write latency CSV --
        lat_writer.writerow([
            fi,
            f"{seg_ms:.1f}", f"{depth_ms:.1f}", f"{fusion_ms:.1f}",
            f"{track_ms:.1f}", f"{risk_ms:.1f}", f"{total_ms:.1f}",
            len(obstacles), len(tracked),
        ])

        # Progress
        if (fi + 1) % 50 == 0 or fi == len(paths) - 1:
            fps = 1000.0 / max(total_ms, 0.1)
            print(f"  [{fi+1}/{len(paths)}] total={total_ms:.0f}ms "
                  f"(seg={seg_ms:.0f} dep={depth_ms:.0f} fus={fusion_ms:.0f} "
                  f"trk={track_ms:.0f} risk={risk_ms:.0f}) "
                  f"obs={len(obstacles)} trk={len(tracked)} fps={fps:.1f}")

    ts_file.close()
    lat_file.close()

    # ---- Rewrite timeseries with risk scores ----
    # (The simple approach above left risk_score as 0.0 for tracked obstacles.
    #  To properly fill it, we'd need to re-run or store risk per track.
    #  For now, let's do a second pass to add risk — actually, let's fix inline.)
    # The above CSV already captures the key fields. Risk score per obstacle
    # requires matching tracked obstacles to frame_risk.obstacle_risks.
    # Let's redo the CSV write properly by collecting all rows first.

    # Actually the CSV is already written. The risk_score column has placeholder 0.0.
    # Let me fix this by rewriting the main loop to collect risk properly.
    # For now, the timeseries CSV is useful for depth/velocity/TTC analysis.
    # Risk scores can be added in a follow-up.

    # ---- Print results ----
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Latency summary
    print("\n--- Latency ---")
    lat_rows = []
    with open(lat_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat_rows.append(row)
    if lat_rows:
        for col in ["seg_ms", "depth_ms", "fusion_ms", "track_ms", "risk_ms", "total_ms"]:
            vals = [float(r[col]) for r in lat_rows]
            print(f"  {col:>10s}: mean={np.mean(vals):.1f}  std={np.std(vals):.1f}  "
                  f"min={np.min(vals):.1f}  max={np.max(vals):.1f}")
        total_vals = [float(r["total_ms"]) for r in lat_rows]
        print(f"  {'FPS':>10s}: mean={1000/np.mean(total_vals):.1f}  "
              f"min={1000/np.max(total_vals):.1f}  max={1000/np.min(total_vals):.1f}")

    # Tracking
    print(f"\n--- Tracking ---")
    print(f"  Approx ID switches: {id_switches}")

    # Detection (if GT)
    if gt_data is not None:
        print(f"\n--- Detection (IoU >= 0.5) ---")
        print(f"  Overall: P={overall_det.precision:.3f}  R={overall_det.recall:.3f}  "
              f"F1={overall_det.f1:.3f}  (TP={overall_det.tp} FP={overall_det.fp} FN={overall_det.fn})")
        if per_class_det:
            print(f"\n  Per-class:")
            for cls_name in CLASS_NAMES:
                if cls_name in per_class_det:
                    m = per_class_det[cls_name]
                    print(f"    {cls_name:>18s}: P={m.precision:.3f}  R={m.recall:.3f}  "
                          f"F1={m.f1:.3f}  (TP={m.tp} FP={m.fp} FN={m.fn})")
    else:
        print("\n--- Detection ---")
        print("  (No panoptic GT provided — skipping P/R/F1)")

    # Output files
    print(f"\n--- Output files ---")
    print(f"  {ts_path}")
    print(f"  {lat_path}")
    print()


# ---------------------------------------------------------------------------
# Fix: rewrite the evaluation loop to properly capture risk scores
# ---------------------------------------------------------------------------
def run_eval_v2(args: argparse.Namespace) -> None:
    """Main evaluation with proper risk score capture in time-series CSV."""
    torch = _get_torch()
    device = _get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect images ----
    if args.seq_dir:
        prefixes = [p.strip() for p in args.prefix.split(",")]
        paths = collect_seq_paths(Path(args.seq_dir), prefixes)
        print(f"Sequence mode: {len(paths)} images from {args.seq_dir}")
    elif args.image_dir:
        paths = collect_image_dir_paths(Path(args.image_dir))
        print(f"Image-dir mode: {len(paths)} images from {args.image_dir}")
    else:
        print("ERROR: Provide --seq-dir (with --prefix) or --image-dir")
        sys.exit(1)

    if not paths:
        print("No images found.")
        sys.exit(1)

    # ---- Load panoptic GT if available ----
    gt_data: dict[str, list[GTBox]] | None = None
    if args.panoptic_json and args.panoptic_mask_dir:
        print("Loading panoptic GT annotations...")
        gt_data = load_panoptic_gt(
            Path(args.panoptic_json), Path(args.panoptic_mask_dir),
        )
        print(f"  Loaded GT for {len(gt_data)} images")

    # ---- Load models ----
    print(f"Loading models (seg: {SEG_WEIGHTS.name}, depth: {DEPTH_MODEL})...")
    depth_pipe, _ = _load_depth_pipe(_DEPTH_PKG, DEPTH_MODEL)
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    compute_segmentation_and_boundary(dummy, SEG_WEIGHTS, device)
    class_names = get_class_names(SEG_WEIGHTS, device)
    is_multiclass = class_names is not None
    print(f"  Model type: {'multi-class (' + str(len(class_names)) + ')' if is_multiclass else '3-class legacy'}")

    # ---- Tracker ----
    tracker = ObstacleTracker(
        max_age=TRACK_MAX_AGE, min_hits=TRACK_MIN_HITS,
        iou_threshold=TRACK_IOU_THRESH, fps=TRACK_FPS,
    )

    # ---- Accumulators ----
    overall_det = DetectionMetrics()
    per_class_det: dict[str, DetectionMetrics] = defaultdict(DetectionMetrics)
    ts_rows: list[list] = []
    lat_rows: list[list] = []
    id_switches = 0
    prev_track_ids: set[int] = set()

    print(f"\nRunning evaluation on {len(paths)} frames...")
    for fi, path in enumerate(paths):
        t_total_0 = time.perf_counter()

        rgb = _load_rgb_u8(path)
        rgb_small = _resize_max_side(rgb, INFER_MAX_SIDE)

        # -- Segmentation --
        t0 = time.perf_counter()
        mask, boundary = compute_segmentation_and_boundary(rgb_small, SEG_WEIGHTS, device)
        seg_ms = (time.perf_counter() - t0) * 1000

        # -- Depth --
        t0 = time.perf_counter()
        depth, _ = _compute_depth_map(rgb_small, depth_pipe)
        depth = np.asarray(depth, dtype=np.float32)
        if EST_SCALE != 1.0:
            depth = depth * float(EST_SCALE)
        depth_ms = (time.perf_counter() - t0) * 1000

        # -- Fusion --
        t0 = time.perf_counter()
        if boundary is not None:
            obstacles = extract_obstacles_multiclass(
                mask, depth, return_masks=False, boundary_prob=boundary,
            )
        else:
            obstacles = extract_obstacles(mask, depth, return_masks=False)
        fusion_ms = (time.perf_counter() - t0) * 1000

        # -- Tracking --
        t0 = time.perf_counter()
        tracked = tracker.update(obstacles)
        track_ms = (time.perf_counter() - t0) * 1000

        # -- Risk --
        t0 = time.perf_counter()
        frame_risk = assess_frame(
            obstacles,
            hfov_deg=HFOV_DEG, w_boat=W_BOAT, lat_margin=LAT_MARGIN,
            d_safe=D_SAFE, l_safe=L_SAFE,
            v_ref=V_REF, alpha_v=ALPHA_V, tracked=tracked,
        )
        risk_ms = (time.perf_counter() - t0) * 1000
        total_ms = (time.perf_counter() - t_total_0) * 1000

        # -- Build risk lookup: obstacle id → risk_score, warning_level --
        risk_lookup: dict[int, tuple[float, str]] = {}
        for orisk in frame_risk.obstacle_risks:
            risk_lookup[id(orisk.obstacle)] = (orisk.risk_score, orisk.warning_level.name)

        # -- ID switch counting --
        cur_track_ids = {t.track_id for t in tracked}
        if fi > 0:
            lost = prev_track_ids - cur_track_ids
            new = cur_track_ids - prev_track_ids
            id_switches += min(len(lost), len(new))
        prev_track_ids = cur_track_ids

        # -- Time-series rows --
        for t in tracked:
            obs = t.obstacle
            cls_name = (CLASS_NAMES[obs.class_id]
                        if obs.class_id is not None and obs.class_id < len(CLASS_NAMES)
                        else "obstacle")
            r_score, r_level = risk_lookup.get(id(obs), (0.0, "SAFE"))
            ts_rows.append([
                fi, t.track_id,
                obs.class_id if obs.class_id is not None else -1,
                cls_name,
                f"{obs.effective_depth:.2f}",
                f"{t.v_closing:.2f}" if t.v_closing is not None else "",
                f"{t.ttc:.1f}" if t.ttc is not None else "",
                f"{r_score:.3f}", r_level,
                obs.x1, obs.y1, obs.x2, obs.y2,
            ])

        # -- Latency row --
        lat_rows.append([
            fi,
            f"{seg_ms:.1f}", f"{depth_ms:.1f}", f"{fusion_ms:.1f}",
            f"{track_ms:.1f}", f"{risk_ms:.1f}", f"{total_ms:.1f}",
            len(obstacles), len(tracked),
        ])

        # -- Detection eval --
        if gt_data is not None:
            img_base = os.path.splitext(path.name)[0]
            gt_boxes = gt_data.get(img_base, [])
            h_orig, w_orig = rgb.shape[:2]
            h_small, w_small = rgb_small.shape[:2]
            sx, sy = w_small / w_orig, h_small / h_orig
            scaled_gt = [
                GTBox(x1=int(g.x1 * sx), y1=int(g.y1 * sy),
                      x2=int(g.x2 * sx), y2=int(g.y2 * sy),
                      class_id=g.class_id, class_name=g.class_name)
                for g in gt_boxes
            ]
            pred_boxes = [
                (o.x1, o.y1, o.x2, o.y2, o.class_id if o.class_id is not None else 0)
                for o in obstacles
            ]
            tp, fp, fn = match_detections(pred_boxes, scaled_gt, iou_threshold=0.5)
            overall_det.tp += tp
            overall_det.fp += fp
            overall_det.fn += fn

            for gt in scaled_gt:
                matched = any(
                    _iou((px1, py1, px2, py2), (gt.x1, gt.y1, gt.x2, gt.y2)) >= 0.5
                    for px1, py1, px2, py2, _ in pred_boxes
                )
                if matched:
                    per_class_det[gt.class_name].tp += 1
                else:
                    per_class_det[gt.class_name].fn += 1

            for px1, py1, px2, py2, pcls in pred_boxes:
                if not any(
                    _iou((px1, py1, px2, py2), (gt.x1, gt.y1, gt.x2, gt.y2)) >= 0.5
                    for gt in scaled_gt
                ):
                    cls_name = CLASS_NAMES[pcls] if pcls < len(CLASS_NAMES) else "Unknown"
                    per_class_det[cls_name].fp += 1

        # Progress
        if (fi + 1) % 50 == 0 or fi == len(paths) - 1:
            fps_val = 1000.0 / max(total_ms, 0.1)
            print(f"  [{fi+1}/{len(paths)}] {total_ms:.0f}ms "
                  f"(seg={seg_ms:.0f} dep={depth_ms:.0f} fus={fusion_ms:.0f} "
                  f"trk={track_ms:.0f} risk={risk_ms:.0f}) "
                  f"obs={len(obstacles)} trk={len(tracked)} fps={fps_val:.1f}")

    # ---- Write CSVs ----
    ts_path = out_dir / "timeseries.csv"
    with open(ts_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class_id", "class_name",
                     "depth_m", "v_closing_ms", "ttc_s",
                     "risk_score", "warning_level",
                     "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
        w.writerows(ts_rows)

    lat_path = out_dir / "latency.csv"
    with open(lat_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "seg_ms", "depth_ms", "fusion_ms",
                     "track_ms", "risk_ms", "total_ms",
                     "n_obstacles", "n_tracked"])
        w.writerows(lat_rows)

    # ---- Print results ----
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Latency
    print("\n--- Latency ---")
    if lat_rows:
        cols = {"seg_ms": 1, "depth_ms": 2, "fusion_ms": 3,
                "track_ms": 4, "risk_ms": 5, "total_ms": 6}
        for name, ci in cols.items():
            vals = np.array([float(r[ci]) for r in lat_rows])
            print(f"  {name:>10s}: mean={vals.mean():.1f}  std={vals.std():.1f}  "
                  f"min={vals.min():.1f}  max={vals.max():.1f}")
        total_vals = np.array([float(r[6]) for r in lat_rows])
        print(f"  {'FPS':>10s}: mean={1000/total_vals.mean():.1f}  "
              f"min={1000/total_vals.max():.1f}  max={1000/total_vals.min():.1f}")

    # Tracking
    print(f"\n--- Tracking ---")
    print(f"  Approx ID switches: {id_switches}")

    # Detection
    if gt_data is not None:
        print(f"\n--- Detection (IoU >= 0.5) ---")
        print(f"  Overall: P={overall_det.precision:.3f}  R={overall_det.recall:.3f}  "
              f"F1={overall_det.f1:.3f}  (TP={overall_det.tp} FP={overall_det.fp} FN={overall_det.fn})")
        if per_class_det:
            print(f"\n  Per-class:")
            for cls_name in CLASS_NAMES:
                if cls_name in per_class_det:
                    m = per_class_det[cls_name]
                    print(f"    {cls_name:>18s}: P={m.precision:.3f}  R={m.recall:.3f}  "
                          f"F1={m.f1:.3f}  (TP={m.tp} FP={m.fp} FN={m.fn})")
    else:
        print("\n--- Detection ---")
        print("  (No panoptic GT provided — skipping P/R/F1)")

    # Save detection CSV (for plotting)
    if gt_data is not None:
        det_path = out_dir / "detection.csv"
        with open(det_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class_name", "precision", "recall", "f1", "tp", "fp", "fn"])
            w.writerow(["Overall", f"{overall_det.precision:.4f}", f"{overall_det.recall:.4f}",
                         f"{overall_det.f1:.4f}", overall_det.tp, overall_det.fp, overall_det.fn])
            for cls_name in CLASS_NAMES:
                if cls_name in per_class_det:
                    m = per_class_det[cls_name]
                    w.writerow([cls_name, f"{m.precision:.4f}", f"{m.recall:.4f}",
                                f"{m.f1:.4f}", m.tp, m.fp, m.fn])
        print(f"  {det_path}")

    # Save summary text
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Evaluation Summary\n")
        f.write(f"Frames: {len(paths)}\n")
        f.write(f"Model: {SEG_WEIGHTS.name}\n")
        if lat_rows:
            total_vals = np.array([float(r[6]) for r in lat_rows])
            f.write(f"Mean FPS: {1000/total_vals.mean():.1f}\n")
            f.write(f"Mean total_ms: {total_vals.mean():.1f}\n")
        f.write(f"ID switches: {id_switches}\n")
        if gt_data is not None:
            f.write(f"Detection P={overall_det.precision:.3f} R={overall_det.recall:.3f} F1={overall_det.f1:.3f}\n")

    print(f"\n--- Output files ---")
    print(f"  {ts_path}")
    print(f"  {lat_path}")
    print(f"  {summary_path}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pipeline evaluation")
    # Image source (mutually exclusive)
    parser.add_argument("--seq-dir", type=str, default=None,
                        help="Sequence image directory")
    parser.add_argument("--prefix", type=str, default="davimar_seq_08",
                        help="Comma-separated filename prefixes (for --seq-dir)")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Flat image directory (alternative to --seq-dir)")

    # GT annotations (optional)
    parser.add_argument("--panoptic-json", type=str, default=None,
                        help="Path to panoptic_annotations.json")
    parser.add_argument("--panoptic-mask-dir", type=str, default=None,
                        help="Path to panoptic_masks/ directory")

    # Output
    parser.add_argument("--out-dir", type=str, default="eval_results",
                        help="Output directory for CSVs and summary")

    args = parser.parse_args()
    run_eval_v2(args)


if __name__ == "__main__":
    main()
