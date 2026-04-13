"""
fusion.risk — collision-corridor risk scoring and warning generation.

Model
-----
We adapt the forward-collision-warning (FCW) geometry from Mobileye's RSS
framework (Shalev-Shwartz et al., 2017) and ISO 22839, while inheriting
the ship-domain sizing intuition of Fujii & Tanaka (1971):

    d_lat   = lateral_offset · d_forward · tan(HFOV / 2)
    lat_exc = max(0, |d_lat| - (W_BOAT / 2 + LAT_MARGIN))
    R       = w_class · exp(-d_forward / D_SAFE) · exp(-lat_exc / L_SAFE)

- W_BOAT / LAT_MARGIN define a forward collision corridor whose half-width
  comes from a virtual ASV hull plus a safety buffer (ship-domain flavour).
- D_SAFE controls how fast risk decays with forward distance. It is kept
  inside the 20 m reliable-depth range reported in the progress report.
- L_SAFE controls how fast risk decays once an obstacle leaves the corridor.
- An obstacle directly ahead at d_forward = 0 gives R = w_class (≈ 1.0);
  an obstacle at the edge of the corridor decays like exp(-d_forward/D_SAFE);
  an obstacle far to the side decays additionally by exp(-lat_exc/L_SAFE).

Phase 2 (after SORT tracking is wired up) will replace the proximity term
with exp(-TTC / T_SAFE) where TTC = d_forward / v_closing, keeping the
corridor geometry intact.

Warning levels
--------------
    SAFE              no meaningful risk
    CAUTION           operator should monitor
    IMMEDIATE_WARNING potential collision; evasive action required
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from fusion.obstacle import Obstacle

# Camera geometry (LaRS / WaterScenes nominal). Override via assess_frame().
HFOV_DEG: float = 70.0

# Virtual ASV ship geometry (metres). Corridor half-width = W_BOAT/2 + LAT_MARGIN.
W_BOAT: float = 2.0
LAT_MARGIN: float = 1.5

# Risk decay scales (metres).
D_SAFE: float = 12.0   # forward; chosen within the 0–20 m reliable depth band
L_SAFE: float = 4.0    # lateral (only kicks in outside the corridor)

# Warning thresholds on R ∈ [0, class_weight].
THRESHOLD_IMMEDIATE: float = 0.50
THRESHOLD_CAUTION: float = 0.15


class WarningLevel(Enum):
    SAFE = 0
    CAUTION = 1
    IMMEDIATE_WARNING = 2

    def __str__(self) -> str:
        return self.name.replace("_", " ")


@dataclass
class ObstacleRisk:
    """Risk assessment result for a single obstacle."""

    obstacle: Obstacle
    risk_score: float
    warning_level: WarningLevel

    # Geometry in metric ego-frame (for debugging + GUI overlays)
    d_forward: float   # metres
    d_lateral: float   # metres, signed (+ right, − left)
    lat_excess: float  # metres beyond the corridor edge (0 if inside)


@dataclass
class FrameRisk:
    """Aggregated risk result for one frame."""

    obstacle_risks: list[ObstacleRisk]
    global_warning: WarningLevel  # max level across all obstacles

    @property
    def most_critical(self) -> Optional[ObstacleRisk]:
        if not self.obstacle_risks:
            return None
        return max(self.obstacle_risks, key=lambda r: r.risk_score)


def _score_to_level(score: float) -> WarningLevel:
    if score >= THRESHOLD_IMMEDIATE:
        return WarningLevel.IMMEDIATE_WARNING
    if score >= THRESHOLD_CAUTION:
        return WarningLevel.CAUTION
    return WarningLevel.SAFE


def _corridor_score(
    d_forward: float,
    d_lateral: float,
    *,
    w_boat: float,
    lat_margin: float,
    d_safe: float,
    l_safe: float,
    class_weight: float,
) -> tuple[float, float]:
    """Return (risk_score, lat_excess_m)."""
    corridor_half = w_boat / 2.0 + lat_margin
    lat_excess = max(0.0, abs(d_lateral) - corridor_half)

    proximity = math.exp(-max(d_forward, 0.0) / d_safe)
    lateral = math.exp(-lat_excess / l_safe)

    return class_weight * proximity * lateral, lat_excess


def assess_frame(
    obstacles: list[Obstacle],
    *,
    hfov_deg: float = HFOV_DEG,
    w_boat: float = W_BOAT,
    lat_margin: float = LAT_MARGIN,
    d_safe: float = D_SAFE,
    l_safe: float = L_SAFE,
) -> FrameRisk:
    """
    Compute per-obstacle risk scores and the global warning level for one frame.

    Parameters
    ----------
    obstacles : list of Obstacle from fusion.obstacle.extract_obstacles().
    hfov_deg  : camera horizontal field of view in degrees (used to convert
                lateral_offset to metric lateral distance).
    w_boat, lat_margin : virtual ASV beam + safety margin, define corridor.
    d_safe, l_safe     : forward / lateral risk decay scales.

    Returns
    -------
    FrameRisk with per-obstacle results and the highest global warning level.
    """
    if not obstacles:
        return FrameRisk(obstacle_risks=[], global_warning=WarningLevel.SAFE)

    tan_half_fov = math.tan(math.radians(hfov_deg) / 2.0)
    results: list[ObstacleRisk] = []

    for obs in obstacles:
        d_fwd = float(obs.effective_depth)
        d_lat = obs.lateral_offset * d_fwd * tan_half_fov

        score, lat_excess = _corridor_score(
            d_fwd,
            d_lat,
            w_boat=w_boat,
            lat_margin=lat_margin,
            d_safe=d_safe,
            l_safe=l_safe,
            class_weight=obs.class_weight,
        )
        results.append(
            ObstacleRisk(
                obstacle=obs,
                risk_score=score,
                warning_level=_score_to_level(score),
                d_forward=d_fwd,
                d_lateral=d_lat,
                lat_excess=lat_excess,
            )
        )

    global_level = max(
        (r.warning_level for r in results),
        key=lambda lv: lv.value,
    )
    return FrameRisk(obstacle_risks=results, global_warning=global_level)
