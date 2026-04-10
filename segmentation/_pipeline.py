"""SegFormer inference helpers (internal). Used by ``segmentation.run_segmentation``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

_CACHE: dict[tuple[str, str], tuple[SegformerImageProcessor, SegformerForSemanticSegmentation, torch.device]] = {}


def load_image_for_segmentation(image: str | Path | np.ndarray) -> np.ndarray:
    """Return RGB uint8 ``(H, W, 3)`` (same layout as ``segmentation.vis``)."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB image array, got shape {arr.shape}")
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _load_checkpoint_state(model: SegformerForSemanticSegmentation, ckpt: Any) -> None:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                model.load_state_dict(ckpt[key])
                return
        model.load_state_dict(ckpt)
        return
    model.load_state_dict(ckpt)


def _load_segformer(
    weights_path: Path,
    device: torch.device,
) -> tuple[SegformerImageProcessor, SegformerForSemanticSegmentation, torch.device]:
    key = (str(weights_path.resolve()), str(device))
    if key in _CACHE:
        return _CACHE[key]

    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    ckpt = torch.load(weights_path, map_location=device)
    _load_checkpoint_state(model, ckpt)
    model.to(device)
    model.eval()
    _CACHE[key] = (processor, model, device)
    return _CACHE[key]


def compute_segmentation_mask(
    rgb: np.ndarray,
    weights_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Run SegFormer and return ``uint8`` class map ``(H, W)`` (labels 0, 1, 2)."""
    h, w = rgb.shape[:2]
    processor, model, dev = _load_segformer(weights_path, device)
    inputs = processor(images=rgb, return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        pred = torch.argmax(upsampled, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    return pred
