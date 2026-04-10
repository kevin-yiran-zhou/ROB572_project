"""
Depth package public API: ``run_depth`` only.

Inference implementation lives in ``depth._pipeline``; the radar/visualization CLI is
``depth.test`` (run ``python -m depth.test`` from the repo root).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from ._constants import DEFAULT_EST_SCALE

ModelName = Literal["small", "base", "large"]

__all__ = ["run_depth", "ModelName"]


def run_depth(
    image: str | Path | np.ndarray,
    model: ModelName = "base",
    *,
    est_scale: float = DEFAULT_EST_SCALE,
    repo_root: Path | None = None,
) -> np.ndarray:
    """
    Run Depth-Anything on one image and return the predicted depth map.

    Same pipeline as ``depth.test.visualize_image_with_radar`` / ``_run_all``:
    ``load_image_for_depth`` → ``_load_depth_pipe`` → ``_compute_depth_map``, then
    optional ``est_scale`` on the returned array.

    Parameters
    ----------
    image
        Path to an image (read via ``matplotlib.image.imread``) or an array in the same
        layout as that output (including 2-D grayscale or RGBA, unchanged).
    model
        Checkpoint folder under ``repo_root/model/``: ``small``, ``base``, or ``large``.
    est_scale
        Multiply the depth map by this scalar before returning (default
        ``depth._constants.DEFAULT_EST_SCALE``, currently ``0.4`` for radar calibration).
    repo_root
        Directory that contains the ``model/`` folder. Defaults to this package
        directory (where ``model/`` lives).

    Returns
    -------
    numpy.ndarray
        2-D ``float32`` array, shape ``(H, W)``.
    """
    from ._pipeline import _compute_depth_map, _load_depth_pipe, load_image_for_depth

    if model not in ("small", "base", "large"):
        raise ValueError(f"model must be 'small', 'base', or 'large', got {model!r}")

    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parent
    img_to_show = load_image_for_depth(image)
    pipe, _ = _load_depth_pipe(root, model_variant=model)
    depth_map, _ = _compute_depth_map(img_to_show, pipe)
    out = np.asarray(depth_map, dtype=np.float32)
    if est_scale != 1.0:
        out = out * float(est_scale)
    return out
