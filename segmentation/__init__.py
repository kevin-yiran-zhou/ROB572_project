"""
Segmentation package public API: ``run_segmentation``.

Inference mirrors ``segmentation.vis`` (SegFormer mit-b0, 3 classes). Visualization CLI
logic remains in ``segmentation.vis``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

__all__ = ["run_segmentation"]


def run_segmentation(
    image: str | Path | np.ndarray,
    *,
    repo_root: Path | None = None,
    weights_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """
    Run SegFormer semantic segmentation on one image and return the class map.

    Same model setup as ``segmentation.vis``: ``nvidia/mit-b0`` with 3 labels
    (obstacle, water, sky), checkpoint loaded from ``weights_path``.

    Parameters
    ----------
    image
        Path to an image (read with OpenCV, BGR→RGB) or an ``HxWx3`` RGB array
        (``uint8`` preferred).
    repo_root
        Directory that contains the ``model/`` folder (same idea as ``depth.run_depth``).
        Defaults to this package directory, where ``model/segformer_baseline.pth`` lives.
    weights_path
        Path to ``.pth`` weights. Default: ``<repo_root>/model/segformer_baseline.pth``.
    device
        ``"cuda"``, ``"cpu"``, or a ``torch.device``. Default: CUDA if available.

    Returns
    -------
    numpy.ndarray
        2-D ``uint8``, shape ``(H, W)``, class indices ``0``, ``1``, ``2``.
    """
    from ._pipeline import compute_segmentation_mask, load_image_for_segmentation

    seg_pkg = Path(__file__).resolve().parent
    root = Path(repo_root).resolve() if repo_root is not None else seg_pkg
    wpath = (
        Path(weights_path).expanduser().resolve()
        if weights_path is not None
        else (root / "model" / "segformer_baseline.pth").resolve()
    )
    if not wpath.is_file():
        raise FileNotFoundError(f"Segmentation weights not found: {wpath}")

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device) if isinstance(device, str) else device

    rgb = load_image_for_segmentation(image)
    return compute_segmentation_mask(rgb, wpath, dev)
