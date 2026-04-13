"""SegFormer inference helpers (internal). Used by ``segmentation.run_segmentation``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, SegformerModel

_CACHE: dict[tuple[str, str], tuple] = {}


# ---------------------------------------------------------------------------
# SegFormerInstanceAware — dual-head model (semantic + boundary)
# Copied from train_segformer_instance_aware.py so we don't need to import it.
# ---------------------------------------------------------------------------

class SegFormerInstanceAware(nn.Module):
    def __init__(self, backbone_name: str = "nvidia/mit-b0", num_classes: int = 9):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained(
            backbone_name, use_safetensors=True,
        )
        hidden_dim = self.backbone.config.hidden_sizes[-1]
        self.semantic_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        feat = outputs.last_hidden_state
        return self.semantic_head(feat), self.boundary_head(feat)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Checkpoint detection — is it the old 3-class or the new instance-aware?
# ---------------------------------------------------------------------------

def _is_instance_aware_ckpt(ckpt: Any) -> bool:
    """True if the checkpoint was saved by train_segformer_instance_aware.py."""
    return isinstance(ckpt, dict) and "class_names" in ckpt


# ---------------------------------------------------------------------------
# Loader: old 3-class SegformerForSemanticSegmentation
# ---------------------------------------------------------------------------

def _load_checkpoint_state(model: SegformerForSemanticSegmentation, ckpt: Any) -> None:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                model.load_state_dict(ckpt[key])
                return
        model.load_state_dict(ckpt)
        return
    model.load_state_dict(ckpt)


def _load_segformer_3class(
    weights_path: Path,
    device: torch.device,
) -> tuple[SegformerImageProcessor, SegformerForSemanticSegmentation, torch.device, None, None]:
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
    result = (processor, model, device, None, None)
    _CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Loader: new instance-aware dual-head model
# ---------------------------------------------------------------------------

def _load_segformer_instance_aware(
    weights_path: Path,
    device: torch.device,
) -> tuple[SegformerImageProcessor, SegFormerInstanceAware, torch.device, list[str], int]:
    key = (str(weights_path.resolve()), str(device))
    if key in _CACHE:
        return _CACHE[key]

    ckpt = torch.load(weights_path, map_location=device)
    class_names: list[str] = ckpt["class_names"]
    img_size: int = ckpt["img_size"]
    num_classes = len(class_names)

    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    model = SegFormerInstanceAware(
        backbone_name="nvidia/mit-b0",
        num_classes=num_classes,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    result = (processor, model, device, class_names, img_size)
    _CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

def _load_segformer(
    weights_path: Path,
    device: torch.device,
) -> tuple[SegformerImageProcessor, nn.Module, torch.device, list[str] | None, int | None]:
    """Load either the old 3-class or new instance-aware model.

    Returns (processor, model, device, class_names_or_None, img_size_or_None).
    class_names is None for the old 3-class model.
    """
    key = (str(weights_path.resolve()), str(device))
    if key in _CACHE:
        return _CACHE[key]

    # Peek at checkpoint to detect type
    ckpt = torch.load(weights_path, map_location=device)
    if _is_instance_aware_ckpt(ckpt):
        class_names: list[str] = ckpt["class_names"]
        img_size: int = ckpt["img_size"]
        num_classes = len(class_names)

        processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        model = SegFormerInstanceAware(
            backbone_name="nvidia/mit-b0",
            num_classes=num_classes,
        )
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
        result = (processor, model, device, class_names, img_size)
    else:
        processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
        model_3c = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=3,
            ignore_mismatched_sizes=True,
        )
        _load_checkpoint_state(model_3c, ckpt)
        model_3c.to(device)
        model_3c.eval()
        result = (processor, model_3c, device, None, None)

    _CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_segmentation_mask(
    rgb: np.ndarray,
    weights_path: Path,
    device: torch.device,
) -> np.ndarray:
    """Run SegFormer and return ``uint8`` class map ``(H, W)``.

    For the old 3-class model: labels 0 (obstacle), 1 (water), 2 (sky).
    For the instance-aware model: labels per class_names in checkpoint.
    """
    h, w = rgb.shape[:2]
    processor, model, dev, class_names, img_size = _load_segformer(weights_path, device)

    if class_names is not None:
        # Instance-aware model: use img_size from checkpoint
        inputs = processor(
            images=rgb,
            return_tensors="pt",
            do_resize=True,
            size={"height": img_size, "width": img_size},
        )
        pixel_values = inputs["pixel_values"].to(dev)
        with torch.no_grad():
            semantic_logits, _boundary_logits = model(pixel_values)
            upsampled = F.interpolate(
                semantic_logits, size=(h, w), mode="bilinear", align_corners=False,
            )
            pred = torch.argmax(upsampled, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    else:
        # Old 3-class model
        inputs = processor(images=rgb, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            pred = torch.argmax(upsampled, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    return pred


def compute_segmentation_and_boundary(
    rgb: np.ndarray,
    weights_path: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Single forward pass returning (seg_mask, boundary_prob_or_None).

    For the instance-aware model both outputs come from one inference.
    For the old 3-class model, boundary_prob is None.
    """
    h, w = rgb.shape[:2]
    processor, model, dev, class_names, img_size = _load_segformer(weights_path, device)

    if class_names is not None:
        inputs = processor(
            images=rgb, return_tensors="pt", do_resize=True,
            size={"height": img_size, "width": img_size},
        )
        pixel_values = inputs["pixel_values"].to(dev)
        with torch.no_grad():
            semantic_logits, boundary_logits = model(pixel_values)
            sem_up = F.interpolate(semantic_logits, size=(h, w), mode="bilinear", align_corners=False)
            pred = torch.argmax(sem_up, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            bnd_up = F.interpolate(boundary_logits, size=(h, w), mode="bilinear", align_corners=False)
            boundary_prob = torch.sigmoid(bnd_up).squeeze().cpu().numpy().astype(np.float32)
        return pred, boundary_prob
    else:
        inputs = processor(images=rgb, return_tensors="pt").to(dev)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            pred = torch.argmax(upsampled, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return pred, None


def compute_boundary_prob(
    rgb: np.ndarray,
    weights_path: Path,
    device: torch.device,
) -> np.ndarray | None:
    """Return boundary probability map ``(H, W)`` float32, or None if old model."""
    _, boundary = compute_segmentation_and_boundary(rgb, weights_path, device)
    return boundary


def get_class_names(weights_path: Path, device: torch.device) -> list[str] | None:
    """Return class names if instance-aware model, else None."""
    _, _, _, class_names, _ = _load_segformer(weights_path, device)
    return class_names
