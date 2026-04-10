"""
Depth-Anything inference helpers (internal). Imported by ``depth`` (``run_depth``) and ``depth.test``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Tuple, Union

import matplotlib.image as mpimg
import numpy as np
from PIL import Image


def _load_radar_uvr(radar_csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd

    radar = pd.read_csv(str(radar_csv_path))
    required_cols = {"u", "v", "range"}
    missing = required_cols - set(radar.columns)
    if missing:
        raise ValueError(
            f"Radar CSV missing required columns {sorted(missing)}. "
            f"Got columns: {list(radar.columns)}"
        )
    return radar["u"].to_numpy(), radar["v"].to_numpy(), radar["range"].to_numpy()


def _sample_est_depth_at_uv(depth_map: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = depth_map.shape[:2]
    u_int = np.clip(np.round(u).astype(int), 0, w - 1)
    v_int = np.clip(np.round(v).astype(int), 0, h - 1)
    return depth_map[v_int, u_int].astype(float)


def load_image_for_depth(image: str | Path | np.ndarray) -> np.ndarray:
    """
    Load or pass through an image array exactly as used by ``_compute_depth_map``
    (same rules as ``visualize_image_with_radar`` / ``_run_all``).
    """
    if isinstance(image, np.ndarray):
        return np.asarray(image)
    image_path = Path(image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return mpimg.imread(str(image_path))


def _select_depth_device() -> tuple[Union[int, str], str]:
    """Return (transformers ``device`` arg, human-readable label). Prefer CUDA, then MPS, else CPU."""
    import torch

    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA device"
        return 0, f"CUDA:0 — {name}"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps", "MPS (Apple GPU)"
    return -1, "CPU (no CUDA/MPS — install CUDA build of torch for GPU)"


def _load_depth_pipe(repo_root: Path, model_variant: str) -> tuple[Any, str]:
    try:
        from transformers import pipeline
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency: `transformers`. Install in your env, e.g.\n"
            "  pip install transformers\n"
            "Then re-run this script."
        ) from e

    model_dir = repo_root / "model" / model_variant
    model_name = str(model_dir)

    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Missing dependency: `torch`. Install it in your conda env, e.g.\n"
            "  pip install torch\n"
            "Then re-run this script."
        )

    device, device_label = _select_depth_device()

    weight_candidates = (
        list(model_dir.glob("*.safetensors"))
        + list(model_dir.glob("*.bin"))
        + list(model_dir.glob("*.pt"))
        + list(model_dir.glob("*.pth"))
        + list(model_dir.glob("*.ckpt"))
    )
    has_sharded_index = (model_dir / "model.safetensors.index.json").is_file()
    if not weight_candidates and not has_sharded_index:
        raise FileNotFoundError(
            f"Depth-Anything weights not found in `./model/{model_variant}/`.\n"
            "Put the checkpoint files there (e.g. `*.safetensors` or `*.bin`).\n"
            f"Currently found: {[p.name for p in model_dir.iterdir()]}"
        )

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        pipe = pipeline(task="depth-estimation", model=model_name, device=device)
    except Exception:
        if device != -1:
            pipe = pipeline(task="depth-estimation", model=model_name, device=-1)
            device_label = f"{device_label} [pipeline fell back to CPU]"
        else:
            raise
    return pipe, device_label


def _compute_depth_map(img_to_show: np.ndarray, depth_pipe) -> tuple[np.ndarray, float]:
    import torch

    pil_image = Image.fromarray(
        (img_to_show * 255).astype("uint8")
    ) if img_to_show.dtype != np.uint8 else Image.fromarray(img_to_show)

    t0 = time.perf_counter()
    depth_result = depth_pipe(pil_image)
    process_time_s = time.perf_counter() - t0
    if isinstance(depth_result, list):
        depth_result = depth_result[0]

    img_h, img_w = img_to_show.shape[0], img_to_show.shape[1]

    if "predicted_depth" in depth_result:
        raw = depth_result["predicted_depth"]
        if not isinstance(raw, torch.Tensor):
            raise RuntimeError(
                f"Expected `predicted_depth` to be a torch.Tensor, got {type(raw)}"
            )
        raw_2d = raw[0] if raw.ndim == 3 else raw
        if raw_2d.ndim != 2:
            raise RuntimeError(f"Unexpected predicted_depth shape: {tuple(raw.shape)}")

        depth_resized = torch.nn.functional.interpolate(
            raw_2d[None, None, ...],
            size=(img_h, img_w),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        depth_map = depth_resized.detach().float().cpu().numpy()
    elif "depth" in depth_result:
        depth = depth_result["depth"]
        if isinstance(depth, torch.Tensor):
            depth_map = depth.detach().float().cpu().numpy()
        elif isinstance(depth, np.ndarray):
            depth_map = depth.astype(np.float32)
        else:
            depth_map = np.array(depth, dtype=np.float32)

        if depth_map.ndim == 2 and depth_map.shape != (img_h, img_w):
            depth_t = torch.from_numpy(depth_map)[None, None, ...].float()
            depth_map = (
                torch.nn.functional.interpolate(
                    depth_t,
                    size=(img_h, img_w),
                    mode="bicubic",
                    align_corners=False,
                )[0, 0]
                .cpu()
                .numpy()
            )
    else:
        raise RuntimeError(
            f"Unexpected depth pipeline output keys: {list(depth_result.keys())}"
        )

    return np.squeeze(depth_map), process_time_s
