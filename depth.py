"""
Overlay radar points on top of the corresponding camera frame.

Expected radar CSV columns (at minimum):
- u: x pixel coordinate in the image
- v: y pixel coordinate in the image
- range: radar range value (used for point color)

This script is written for the WaterScenes_Samples folder layout in this repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib as mpl
from PIL import Image
from transformers import pipeline
import random
from typing import Tuple
import time


def _load_radar_uvr(radar_csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _load_depth_pipe(repo_root: Path, model_variant: str):
    model_dir = repo_root / "model" / model_variant
    model_name = str(model_dir)

    try:
        import torch

        device = 0 if torch.cuda.is_available() else -1
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Missing dependency: `torch`. Install it in your conda env, e.g.\n"
            "  pip install torch\n"
            "Then re-run this script."
        )

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
    return pipeline(task="depth-estimation", model=model_name, device=device)


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


def visualize_image_with_radar(
    image_path: str | Path,
    radar_csv_path: str | Path,
    output_path: str | Path | None,
    *,
    point_size: float = 18.0,
    alpha: float = 0.65,
    cmap: str = "inferno",
    show_colorbar: bool = True,
    within: float | None = None,
    est_scale: float = 1.0,
    model_variant: str = "base",
) -> Path | None:
    image_path = Path(image_path)
    radar_csv_path = Path(radar_csv_path)
    output_path_obj = Path(output_path) if output_path is not None else None

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not radar_csv_path.is_file():
        raise FileNotFoundError(f"Radar CSV not found: {radar_csv_path}")

    img = mpimg.imread(str(image_path))
    if img.ndim == 2:
        # Grayscale -> make it RGB-like for imshow consistency.
        img_to_show = img
    else:
        img_to_show = img

    u, v, r = _load_radar_uvr(radar_csv_path)

    repo_root = Path(__file__).resolve().parent
    depth_pipe = _load_depth_pipe(repo_root, model_variant=model_variant)
    depth_map, process_time_s = _compute_depth_map(img_to_show, depth_pipe)

    # Basic stats of metric depth
    depth_min = float(np.min(depth_map))
    depth_max = float(np.max(depth_map))
    print(f"Raw metric depth stats: min={depth_min:.6f}, max={depth_max:.6f}")

    # --- Per-radar-point depth comparison (GT vs estimated) ---
    est_depth = _sample_est_depth_at_uv(depth_map, u, v)
    if est_scale != 1.0:
        est_depth = est_depth * float(est_scale)
    gt_depth = r.astype(float)
    if within is not None:
        within_mask = gt_depth <= float(within)
        gt_depth = gt_depth[within_mask]
        est_depth = est_depth[within_mask]
        print(
            f"Points (total={r.size}, within(GT<={within:g}m)={gt_depth.size})"
        )
    else:
        print(f"Points (total={gt_depth.size})")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    single_out = results_dir / f"{model_variant}_{image_path.stem}.csv"
    pd.DataFrame({"GT": gt_depth, "est": est_depth}).to_csv(single_out, index=False)
    print(f"Saved results: {single_out}")
    print(f"Depth-Anything process time: {process_time_s:.4f}s")

    # Use ONE shared color scale so "same distance => same color".
    # Pick the larger max across (radar range, predicted depth).
    shared_vmin = 0.0
    shared_vmax = float(max(depth_max, float(np.max(r))))
    norm = mpl.colors.Normalize(vmin=shared_vmin, vmax=shared_vmax)
    shared_cmap = plt.get_cmap("magma")

    # --- Build figure with 3 panels ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Reserve space on the right for a single shared colorbar.
    fig.subplots_adjust(right=0.88, wspace=0.10)

    # 1) Original image with radar range overlay
    ax0 = axes[0]
    ax0.imshow(img_to_show, origin="upper")
    sc0 = ax0.scatter(
        u,
        v,
        c=r,
        cmap=shared_cmap,
        norm=norm,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )
    ax0.set_xlim(0, img_to_show.shape[1] - 1)
    ax0.set_ylim(img_to_show.shape[0] - 1, 0)
    ax0.set_axis_off()
    ax0.set_title("RGB + radar range")

    # 2) Depth map
    ax1 = axes[1]
    im1 = ax1.imshow(depth_map, cmap=shared_cmap, norm=norm)
    ax1.set_title("Depth (metric)")
    ax1.set_axis_off()

    # 3) GT vs Estimated scatter (same axes range + 45° line)
    ax2 = axes[2]
    if within is not None:
        within_mask = gt_depth <= float(within)
        gt_plot = gt_depth[within_mask]
        est_plot = est_depth[within_mask]
    else:
        gt_plot = gt_depth
        est_plot = est_depth

    max_xy = float(
        max(
            np.max(gt_plot) if gt_plot.size else 0.0,
            np.max(est_plot) if est_plot.size else 0.0,
        )
    )
    # Use the shared range so it matches the other panels' scale.
    max_xy = max(max_xy, shared_vmax)
    ax2.scatter(gt_plot, est_plot, s=14, alpha=0.7, color="tab:blue", edgecolors="none")
    ax2.plot([0.0, max_xy], [0.0, max_xy], linestyle="--", linewidth=1.5, color="tab:red", label="y = x")
    ax2.set_xlim(0.0, max_xy)
    ax2.set_ylim(0.0, max_xy)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("GT depth (radar range, m)")
    ax2.set_ylabel("Estimated depth (m)")
    ax2.set_title("GT vs Estimated" + (f" (GT<={within:g}m)" if within is not None else ""))
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper left")
    if show_colorbar:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=shared_cmap)
        sm.set_array([])
        # Put the bar outside the plots (avoid overlaying the depth map).
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("Distance (m)")

    if output_path_obj is not None:
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path_obj), dpi=220, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
        return output_path_obj

    # Leave the figure open so `plt.show()` can display it.
    return None


def _pick_random_timestamp(image_dir: Path, *, seed: int | None = None) -> str:
    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg files found in {image_dir}")
    rng = random.Random(seed)
    return rng.choice(images).stem


def _run_all(
    image_dir: Path,
    radar_dir: Path,
    *,
    seed: int | None,
    show_display: bool,
    output_path: Path | None,
    within: float | None,
    est_scale: float,
    model_variant: str,
) -> None:
    repo_root = Path(__file__).resolve().parent
    depth_pipe = _load_depth_pipe(repo_root, model_variant=model_variant)

    images = sorted(image_dir.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg files found in {image_dir}")

    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(images)

    all_gt: list[np.ndarray] = []
    all_est: list[np.ndarray] = []
    all_process_times_s: list[float] = []
    skipped = 0

    for idx, img_path in enumerate(images, start=1):
        ts = img_path.stem
        radar_csv = radar_dir / f"{ts}.csv"
        if not radar_csv.is_file():
            skipped += 1
            continue

        img = mpimg.imread(str(img_path))
        img_to_show = img

        try:
            u, v, r = _load_radar_uvr(radar_csv)
            depth_map, process_time_s = _compute_depth_map(img_to_show, depth_pipe)
            est = _sample_est_depth_at_uv(depth_map, u, v)
            if est_scale != 1.0:
                est = est * float(est_scale)
            gt = r.astype(float)
        except Exception:
            skipped += 1
            continue

        all_gt.append(gt)
        all_est.append(est)
        all_process_times_s.append(process_time_s)

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(images)} images (skipped {skipped})...")

    if not all_gt:
        raise RuntimeError("No valid (image, radar) pairs were processed.")

    gt_depth = np.concatenate(all_gt)
    est_depth = np.concatenate(all_est)
    total_points = gt_depth.size
    if within is not None:
        within_mask = gt_depth <= float(within)
        gt_depth = gt_depth[within_mask]
        est_depth = est_depth[within_mask]
        print(f"[ALL] Points (total={total_points}, within(GT<={within:g}m)={gt_depth.size})")
    else:
        print(f"[ALL] Points (total={total_points})")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_out = results_dir / f"{model_variant}.csv"
    pd.DataFrame({"GT": gt_depth, "est": est_depth}).to_csv(all_out, index=False)
    print(f"[ALL] Saved results: {all_out}")

    avg_process_time_s = float(np.mean(all_process_times_s)) if all_process_times_s else float("nan")
    print(f"[ALL] Depth-Anything average process time: {avg_process_time_s:.4f}s")

    if within is not None:
        within_mask = gt_depth <= float(within)
        gt_plot = gt_depth[within_mask]
        est_plot = est_depth[within_mask]
    else:
        gt_plot = gt_depth
        est_plot = est_depth

    max_xy = float(max(np.max(gt_plot), np.max(est_plot))) if gt_plot.size else 0.0
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 7.0))
    ax.scatter(gt_plot, est_plot, s=6, alpha=0.35, color="tab:blue", edgecolors="none")
    ax.plot([0.0, max_xy], [0.0, max_xy], linestyle="--", linewidth=1.5, color="tab:red", label="y = x")
    ax.set_xlim(0.0, max_xy)
    ax.set_ylim(0.0, max_xy)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("GT depth (radar range, m)")
    ax.set_ylabel("Estimated depth (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=220, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return

    if show_display:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    data_root = repo_root / "WaterScenes_Samples"
    image_dir = data_root / "image"
    radar_dir = data_root / "radar"

    parser = argparse.ArgumentParser(
        description="Run depth + radar export/visualization."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process ALL images and save combined GT/est results for the selected model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["small", "base", "large"],
        default="base",
        help="Depth-Anything model size to load from model/<size>/.",
    )
    parser.add_argument(
        "--est_scale",
        type=float,
        default=1.0,
        help="Multiply ALL estimated depths by this scalar before saving (default: 1.0).",
    )
    args = parser.parse_args()

    if args.all:
        _run_all(
            image_dir=image_dir,
            radar_dir=radar_dir,
            seed=None,
            show_display=True,
            output_path=None,
            within=None,
            est_scale=args.est_scale,
            model_variant=args.model,
        )
        return

    timestamp = _pick_random_timestamp(image_dir=image_dir, seed=None)
    image_path = image_dir / f"{timestamp}.jpg"
    radar_csv_path = radar_dir / f"{timestamp}.csv"

    _ = visualize_image_with_radar(
        image_path=image_path,
        radar_csv_path=radar_csv_path,
        output_path=None,
        point_size=18.0,
        alpha=0.65,
        cmap="inferno",
        show_colorbar=True,
        within=None,
        est_scale=args.est_scale,
        model_variant=args.model,
    )
    plt.show()


if __name__ == "__main__":
    main()

