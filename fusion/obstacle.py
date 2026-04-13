"""
fusion.obstacle — extract obstacle instances from a seg mask + metric depth map.

Supports two modes:
1. **3-class (legacy)**: seg_mask labels 0=obstacle, 1=water, 2=sky.
   Uses geometric heuristic + Otsu depth split to separate static/dynamic.
2. **Multi-class (instance-aware)**: seg_mask labels per class_names in
   checkpoint (e.g. 0=StaticObstacle, 3=Boat, 4=Buoy, …).
   Uses class ID directly — no geometric/depth heuristic needed.

Output
------
list[Obstacle], one per connected region that survives the min-area filter,
sorted nearest-first by `effective_depth`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# SegFormer LUT in combined.py: class 0 is the obstacle class (3-class model).
OBSTACLE_CLASS: int = 0

# Regions smaller than this (pixels) are treated as segmentation noise.
MIN_OBSTACLE_AREA: int = 300

# Percentile of the obstacle depth distribution used as its representative
# distance. p5 ≈ "closest 5% of the blob", a conservative proxy for the
# nearest point of approach.
DEPTH_PERCENTILE: float = 5.0

# Extra metres subtracted from depth_p5 before risk scoring.
DEPTH_MARGIN_M: float = 0.0

# --- 3-class fallback: geometric + depth split constants ---
STATIC_EDGE_MARGIN: int = 5
STATIC_MIN_WIDTH_FRAC: float = 0.40
MIN_DEPTH_GAP_M: float = 5.0

# Risk weights
STATIC_CLASS_WEIGHT: float = 0.3
DYNAMIC_CLASS_WEIGHT: float = 1.0

# --- Multi-class model: class definitions ---
# Class IDs that should be ignored (not extracted as obstacles).
IGNORE_CLASSES: frozenset[int] = frozenset({1, 2})  # Water, Sky

# Class IDs considered static obstacles.
STATIC_CLASSES: frozenset[int] = frozenset({0})  # Static Obstacle

# Per-class risk weights for the multi-class model.
# IDs not listed here default to DYNAMIC_CLASS_WEIGHT (1.0).
CLASS_WEIGHTS: dict[int, float] = {
    0: 0.3,   # Static Obstacle
    3: 1.0,   # Boat
    4: 0.7,   # Buoy
    5: 1.0,   # Swimmer
    6: 0.8,   # Animal
    7: 0.7,   # Float
    8: 0.5,   # Other
}


@dataclass
class Obstacle:
    """Per-obstacle information consumed by fusion.risk."""

    x1: int
    y1: int
    x2: int
    y2: int

    depth_p5: float
    effective_depth: float
    lateral_offset: float
    pixel_area: int

    class_weight: float = 1.0
    is_static: bool = False

    # Semantic class ID from the segmentation model (None for 3-class legacy).
    class_id: int | None = None

    mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def bbox_center_x(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def bbox_center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def bbox_width(self) -> int:
        return self.x2 - self.x1

    @property
    def bbox_height(self) -> int:
        return self.y2 - self.y1


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_obstacle(
    x1: int, y1: int, x2: int, y2: int,
    xs: np.ndarray, depths: np.ndarray,
    area: int, W: int,
    *,
    depth_percentile: float,
    depth_margin: float,
    is_static: bool,
    class_weight: float,
    class_id: int | None,
    instance_mask: np.ndarray | None,
    return_masks: bool,
) -> Obstacle | None:
    """Build an Obstacle from pixel arrays. Returns None if no valid depth."""
    valid = np.isfinite(depths) & (depths > 0)
    if not valid.any():
        return None
    xs_v = xs[valid]
    depths_v = depths[valid]

    depth_p5 = float(np.percentile(depths_v, depth_percentile))
    effective_depth = max(depth_p5 - depth_margin, 0.1)

    nearest_xs = xs_v[depths_v <= depth_p5]
    cx_nearest = float(np.median(nearest_xs)) if nearest_xs.size > 0 else (x1 + x2) / 2.0
    lateral_offset = (cx_nearest - W / 2.0) / (W / 2.0)

    return Obstacle(
        x1=x1, y1=y1, x2=x2, y2=y2,
        depth_p5=depth_p5,
        effective_depth=effective_depth,
        lateral_offset=lateral_offset,
        pixel_area=area,
        class_weight=class_weight,
        is_static=is_static,
        class_id=class_id,
        mask=instance_mask if return_masks else None,
    )


# ---------------------------------------------------------------------------
# Multi-class extraction (new instance-aware model)
# ---------------------------------------------------------------------------

def extract_obstacles_multiclass(
    seg_mask: np.ndarray,
    depth_map: np.ndarray,
    *,
    min_area: int = MIN_OBSTACLE_AREA,
    depth_percentile: float = DEPTH_PERCENTILE,
    depth_margin: float = DEPTH_MARGIN_M,
    return_masks: bool = False,
    boundary_prob: np.ndarray | None = None,
    boundary_thresh: float = 0.4,
) -> list[Obstacle]:
    """Extract obstacles from a multi-class segmentation mask.

    Each non-ignored class is processed independently. Connected components
    within each class become separate obstacle instances. If `boundary_prob`
    is provided, it is used to split touching instances within the same class
    (same technique as the teammate's visualize_instance_aware.py).
    """
    if seg_mask.shape[:2] != depth_map.shape[:2]:
        raise ValueError(
            f"seg_mask {seg_mask.shape} vs depth_map {depth_map.shape}"
        )

    H, W = seg_mask.shape[:2]
    depth_f = depth_map.astype(np.float32, copy=False)
    obstacles: list[Obstacle] = []

    unique_classes = set(np.unique(seg_mask).tolist())
    obstacle_classes = unique_classes - IGNORE_CLASSES

    for cls_id in sorted(obstacle_classes):
        cls_mask = seg_mask == cls_id

        # Use boundary to split touching instances (if available)
        if boundary_prob is not None:
            split_mask = cls_mask & (boundary_prob < boundary_thresh)
        else:
            split_mask = cls_mask

        num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
            split_mask.astype(np.uint8), connectivity=8,
        )

        is_static = cls_id in STATIC_CLASSES
        cw = CLASS_WEIGHTS.get(cls_id, DYNAMIC_CLASS_WEIGHT)

        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            x1 = int(stats[label_id, cv2.CC_STAT_LEFT])
            y1 = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

            instance_mask = label_map == label_id
            ys, xs = np.where(instance_mask)
            depths = depth_f[ys, xs]

            obs = _build_obstacle(
                x1, y1, x1 + w, y1 + h,
                xs, depths, area, W,
                depth_percentile=depth_percentile,
                depth_margin=depth_margin,
                is_static=is_static,
                class_weight=cw,
                class_id=cls_id,
                instance_mask=instance_mask,
                return_masks=return_masks,
            )
            if obs is not None:
                obstacles.append(obs)

    obstacles.sort(key=lambda o: o.effective_depth)
    return obstacles


# ---------------------------------------------------------------------------
# 3-class legacy extraction (geometric + depth split)
# ---------------------------------------------------------------------------

def _is_static_candidate(
    x1: int, y1: int, x2: int, y2: int, H: int, W: int,
) -> bool:
    touches_left = x1 <= STATIC_EDGE_MARGIN
    touches_right = x2 >= W - STATIC_EDGE_MARGIN
    touches_top = y1 <= STATIC_EDGE_MARGIN
    return (int(touches_left) + int(touches_right) + int(touches_top)) >= 2 and (x2 - x1) > W * STATIC_MIN_WIDTH_FRAC


def _depth_split_static(
    component_mask: np.ndarray, depth_f: np.ndarray, H: int, W: int,
    *, min_area: int, depth_percentile: float, depth_margin: float, return_masks: bool,
) -> list[Obstacle]:
    ys, xs = np.where(component_mask)
    depths = depth_f[ys, xs]
    valid = np.isfinite(depths) & (depths > 0)
    if valid.sum() < 2:
        return []
    d_valid = depths[valid]
    d_min, d_max = float(d_valid.min()), float(d_valid.max())
    if d_max - d_min < MIN_DEPTH_GAP_M:
        return []

    d_norm = ((d_valid - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    thresh_val, _ = cv2.threshold(d_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    split_depth = d_min + (thresh_val / 255.0) * (d_max - d_min)

    near_mask = component_mask & (depth_f < split_depth)
    if int(near_mask.sum()) < min_area:
        return []

    num_sub, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(
        near_mask.astype(np.uint8), connectivity=8,
    )
    dynamic_obs: list[Obstacle] = []
    for sid in range(1, num_sub):
        sub_area = int(sub_stats[sid, cv2.CC_STAT_AREA])
        if sub_area < min_area:
            continue
        sx1 = int(sub_stats[sid, cv2.CC_STAT_LEFT])
        sy1 = int(sub_stats[sid, cv2.CC_STAT_TOP])
        sx2 = sx1 + int(sub_stats[sid, cv2.CC_STAT_WIDTH])
        sy2 = sy1 + int(sub_stats[sid, cv2.CC_STAT_HEIGHT])
        if sx1 <= STATIC_EDGE_MARGIN or sx2 >= W - STATIC_EDGE_MARGIN or sy1 <= STATIC_EDGE_MARGIN:
            continue
        sub_mask = sub_labels == sid
        sub_ys, sub_xs = np.where(sub_mask)
        obs = _build_obstacle(
            sx1, sy1, sx2, sy2, sub_xs, depth_f[sub_ys, sub_xs], sub_area, W,
            depth_percentile=depth_percentile, depth_margin=depth_margin,
            is_static=False, class_weight=DYNAMIC_CLASS_WEIGHT, class_id=None,
            instance_mask=sub_mask, return_masks=return_masks,
        )
        if obs is not None:
            dynamic_obs.append(obs)
    return dynamic_obs


def extract_obstacles(
    seg_mask: np.ndarray,
    depth_map: np.ndarray,
    *,
    obstacle_class: int = OBSTACLE_CLASS,
    min_area: int = MIN_OBSTACLE_AREA,
    depth_percentile: float = DEPTH_PERCENTILE,
    depth_margin: float = DEPTH_MARGIN_M,
    return_masks: bool = False,
) -> list[Obstacle]:
    """Extract obstacles from a 3-class segmentation mask (legacy path)."""
    if seg_mask.shape[:2] != depth_map.shape[:2]:
        raise ValueError(
            f"seg_mask {seg_mask.shape} vs depth_map {depth_map.shape}"
        )
    H, W = seg_mask.shape[:2]
    obstacle_binary = (seg_mask == obstacle_class).astype(np.uint8)
    if obstacle_binary.sum() == 0:
        return []

    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(obstacle_binary, connectivity=8)
    obstacles: list[Obstacle] = []
    depth_f = depth_map.astype(np.float32, copy=False)

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x1 = int(stats[label_id, cv2.CC_STAT_LEFT])
        y1 = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        x2, y2 = x1 + w, y1 + h
        instance_mask = label_map == label_id
        ys, xs = np.where(instance_mask)
        depths = depth_f[ys, xs]

        static_cand = _is_static_candidate(x1, y1, x2, y2, H, W)
        if static_cand:
            obstacles.extend(_depth_split_static(
                instance_mask, depth_f, H, W,
                min_area=min_area, depth_percentile=depth_percentile,
                depth_margin=depth_margin, return_masks=return_masks,
            ))

        obs = _build_obstacle(
            x1, y1, x2, y2, xs, depths, area, W,
            depth_percentile=depth_percentile, depth_margin=depth_margin,
            is_static=static_cand, class_weight=STATIC_CLASS_WEIGHT if static_cand else DYNAMIC_CLASS_WEIGHT,
            class_id=None, instance_mask=instance_mask, return_masks=return_masks,
        )
        if obs is not None:
            obstacles.append(obs)

    obstacles.sort(key=lambda o: o.effective_depth)
    return obstacles
