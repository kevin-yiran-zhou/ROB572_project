"""
fusion — obstacle extraction and risk-aware warning.

Modules:
    obstacle  Connected-component extraction of obstacle instances from a
              segmentation mask + metric depth map.
    risk      Collision-corridor risk scoring and warning-level generation.
"""

from fusion.obstacle import Obstacle, extract_obstacles, extract_obstacles_multiclass
from fusion.risk import (
    FrameRisk,
    ObstacleRisk,
    WarningLevel,
    assess_frame,
)

__all__ = [
    "Obstacle",
    "extract_obstacles",
    "extract_obstacles_multiclass",
    "FrameRisk",
    "ObstacleRisk",
    "WarningLevel",
    "assess_frame",
]
