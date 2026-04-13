"""
Tkinter GUI: RGB + SegFormer + Depth-Anything on a prefixed image sequence.

Edit the **Config** block below (no CLI). ``INFER_MAX_SIDE`` applies to both seg and depth.
Starts fullscreen; Esc toggles fullscreen. Auto-plays once through the sequence then closes.
"""

from __future__ import annotations

import queue
import re
import sys
import threading
import time
import tkinter.font as tkfont
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from depth._constants import DEFAULT_EST_SCALE
from depth._pipeline import _compute_depth_map, _load_depth_pipe, load_image_for_depth

import math

import cv2
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.cm as mpl_cm

from fusion import FrameRisk, WarningLevel, assess_frame, extract_obstacles, extract_obstacles_multiclass
from tracking import ObstacleTracker, TrackedObstacle
from segmentation._pipeline import compute_segmentation_and_boundary

# Depth-Anything
_DEPTH_PKG = _ROOT / "depth"
_DEPTH_MODEL_CHOICES = ("small", "base", "large")

# SegFormer package root (weights path default below)
_SEG_PKG = _ROOT / "segmentation"

# ---------------------------------------------------------------------------
# Config - edit here (no command-line flags)
# ---------------------------------------------------------------------------
DEPTH_MODEL: str = "small"  # "small" | "base" | "large"
EST_SCALE: float = DEFAULT_EST_SCALE
INFER_MAX_SIDE: int = 552  # 0 = no resize before seg/depth; else max long edge (px)
GUI_PREVIEW_MAX: int = 1280  # longest edge for on-screen thumbnails
GUI_FONT_PT: int = 16  # base UI font size (labels, buttons, panel titles)

SEQ_DIR: Path = _ROOT / "MODD2_video_data_rectified/video_data/kope67-00-00025200-00025670/framesRectified"
IMAGE_GLOB_PATTERN: str = "*L.jpg"

SEG_WEIGHTS: Path = _SEG_PKG / "model" / "segformer_instance_aware_best.pth"
# SEG_WEIGHTS: Path = _SEG_PKG / "model" / "segformer_baseline.pth"  # old 3-class
SEG_DEVICE: str | None = None  # None = auto; or "cpu", "cuda:0", ...

# --- Risk-aware warning (fusion module) ---
HFOV_DEG: float = 70.0       # camera horizontal FOV (LaRS nominal)
W_BOAT: float = 2.0          # virtual ASV beam (m)
LAT_MARGIN: float = 1.0      # lateral safety margin outside beam (m)
D_SAFE: float = 12.0         # forward risk decay scale (m)
L_SAFE: float = 2.0          # lateral risk decay scale (m); sharper than corridor
DRAW_CORRIDOR: bool = True   # overlay the forward collision corridor
CORRIDOR_NEAR_M: float = 4.0 # near plane of corridor trapezoid (visual only)
CORRIDOR_FAR_M: float = 20.0 # far plane of corridor trapezoid (visual only)

# Frame-level warning hysteresis. Enter a higher level when score crosses the
# upper threshold; only drop back once it falls below the (lower) exit
# threshold. Reduces flicker between SAFE/CAUTION/IMMEDIATE on noisy depth.
# Proper fix is per-track smoothing after SORT lands — this is a stopgap.
ENTER_CAUTION: float = 0.15
EXIT_CAUTION: float = 0.10
ENTER_IMMEDIATE: float = 0.50
EXIT_IMMEDIATE: float = 0.38

# --- SORT tracking ---
TRACK_MAX_AGE: int = 5       # frames before a lost track is deleted
TRACK_MIN_HITS: int = 2      # consecutive hits before track is reported
TRACK_IOU_THRESH: float = 0.20
TRACK_FPS: float = 10.0      # assumed frame rate for velocity estimation
# ---------------------------------------------------------------------------

# LUT for depth colormap (faster than calling inferno per pixel each frame).
_INFERNO_LUT = (mpl_cm.inferno(np.linspace(0.0, 1.0, 256, dtype=np.float64))[:, :3] * 255.0).astype(
    np.uint8
)
_SEG_LUT = np.zeros((256, 3), dtype=np.uint8)
# 3-class: 0=obstacle(red), 1=water(blue), 2=sky(green)
# 9-class: 0=StaticObstacle, 1=Water, 2=Sky, 3=Boat, 4=Buoy,
#          5=Swimmer, 6=Animal, 7=Float, 8=Other
_SEG_LUT[0] = (255, 0, 0)       # Static Obstacle — red
_SEG_LUT[1] = (0, 0, 255)       # Water — blue
_SEG_LUT[2] = (0, 255, 0)       # Sky — green
_SEG_LUT[3] = (255, 128, 0)     # Boat — orange
_SEG_LUT[4] = (255, 0, 255)     # Buoy — magenta
_SEG_LUT[5] = (255, 255, 0)     # Swimmer — yellow
_SEG_LUT[6] = (128, 255, 255)   # Animal — cyan
_SEG_LUT[7] = (180, 80, 255)    # Float — purple
_SEG_LUT[8] = (255, 255, 255)   # Other — white
_SEG_LUT[255] = (0, 0, 0)

# Warning-level colours drawn on the RGB overlay (tuples are in RGB order
# because we draw directly on the float/uint8 RGB array that PIL later
# consumes — cv2 treats the tuple channel-blind).
_WARNING_COLORS: dict[WarningLevel, tuple[int, int, int]] = {
    WarningLevel.SAFE: (40, 200, 40),
    WarningLevel.CAUTION: (255, 200, 40),
    WarningLevel.IMMEDIATE_WARNING: (255, 50, 50),
}
_BANNER_HEIGHT: int = 40

_SEG_HF_ID = "nvidia/mit-b0"  # matches ``segmentation._pipeline`` backbone

_rs = getattr(Image, "Resampling", Image)
_PREVIEW_DOWN_RESAMPLE = getattr(_rs, "BOX", getattr(_rs, "NEAREST", Image.NEAREST))


def _natural_sort_key(path: Path) -> list:
    parts = re.split(r"(\d+)", path.name)
    key: list = []
    for x in parts:
        if x.isdigit():
            key.append(int(x))
        else:
            key.append(x.casefold())
    return key


def _normalized_suffixes(suffixes: list[str]) -> frozenset[str]:
    out: set[str] = set()
    for s in suffixes:
        t = s.strip().lower()
        if not t.startswith("."):
            t = "." + t
        out.add(t)
    return frozenset(out)


def _list_sequence_images(seq_dir: Path) -> list[Path]:
    if not seq_dir.is_dir():
        return []
    return sorted(
        [p for p in seq_dir.glob(IMAGE_GLOB_PATTERN) if p.is_file()],
        key=_natural_sort_key,
    )


def _seg_mask_to_rgb_u8(mask: np.ndarray) -> np.ndarray:
    m = np.clip(np.asarray(mask, dtype=np.intp), 0, 255)
    return _SEG_LUT[m]


def _depth_to_rgb_u8(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    lo, hi = float(np.min(d)), float(np.max(d))
    if hi <= lo:
        return np.zeros((*d.shape, 3), dtype=np.uint8)
    idx = ((d - lo) / (hi - lo + 1e-12) * 255.0).clip(0, 255).astype(np.uint8)
    return _INFERNO_LUT[idx]


def _load_rgb_u8(path: Path) -> np.ndarray:
    arr = load_image_for_depth(path)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def _resize_rgb_max_long_side(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """If max(H,W) > max_side, shrink with aspect ratio preserved (longest edge = max_side)."""
    if max_side <= 0:
        return rgb
    h, w = rgb.shape[:2]
    nw, nh = _fit_size(w, h, max_side)
    if (nw, nh) == (w, h):
        return rgb
    pil = Image.fromarray(rgb).resize((nw, nh), _PREVIEW_DOWN_RESAMPLE)
    return np.asarray(pil)


def _fit_size(w: int, h: int, max_side: int) -> tuple[int, int]:
    m = max(w, h)
    if m <= max_side:
        return w, h
    s = max_side / m
    return max(1, int(w * s)), max(1, int(h * s))


def _array_to_photo(arr: np.ndarray, max_side: int) -> ImageTk.PhotoImage:
    h, w = arr.shape[:2]
    nw, nh = _fit_size(w, h, max_side)
    pil = Image.fromarray(arr)
    if (nw, nh) != (w, h):
        pil = pil.resize((nw, nh), _PREVIEW_DOWN_RESAMPLE)
    return ImageTk.PhotoImage(pil)


def _draw_text_with_bg(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    bg_color: tuple[int, int, int],
    *,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Draw text with a filled background rectangle onto an RGB array in-place."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    x = max(0, x)
    y = max(th + 4, y)
    cv2.rectangle(
        img,
        (x, y - th - 4),
        (x + tw + 6, y + bl),
        bg_color,
        thickness=-1,
    )
    cv2.putText(
        img,
        text,
        (x + 3, y - 2),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _draw_status_banner(img: np.ndarray, level: WarningLevel) -> None:
    """Draw the top status banner with the global warning level."""
    H, W = img.shape[:2]
    color = _WARNING_COLORS[level]
    cv2.rectangle(img, (0, 0), (W, _BANNER_HEIGHT), color, thickness=-1)
    text = f"STATUS: {level}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.95
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx = max(0, (W - tw) // 2)
    ty = _BANNER_HEIGHT // 2 + th // 2
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _apply_hysteresis(prev: WarningLevel, score: float) -> WarningLevel:
    """Return the next warning level given previous level and current score.

    Uses asymmetric enter/exit thresholds to suppress single-frame flicker.
    `score` is the most-critical obstacle's risk score for the frame (0 if none).
    """
    if prev == WarningLevel.IMMEDIATE_WARNING:
        if score < EXIT_IMMEDIATE:
            return (
                WarningLevel.CAUTION
                if score >= EXIT_CAUTION
                else WarningLevel.SAFE
            )
        return WarningLevel.IMMEDIATE_WARNING
    if prev == WarningLevel.CAUTION:
        if score >= ENTER_IMMEDIATE:
            return WarningLevel.IMMEDIATE_WARNING
        if score < EXIT_CAUTION:
            return WarningLevel.SAFE
        return WarningLevel.CAUTION
    # prev == SAFE
    if score >= ENTER_IMMEDIATE:
        return WarningLevel.IMMEDIATE_WARNING
    if score >= ENTER_CAUTION:
        return WarningLevel.CAUTION
    return WarningLevel.SAFE


def _draw_corridor(
    img: np.ndarray,
    *,
    hfov_deg: float,
    corridor_half_m: float,
    near_depth_m: float = 4.0,
    far_depth_m: float = 20.0,
) -> None:
    """
    Overlay the forward collision corridor as a pair of white lines forming
    a trapezoid. Near depth maps to the image bottom; far depth maps to a
    heuristic horizon row (≈ 45 % from the top). This is a rough visual
    aid — no ground-plane calibration is performed.
    """
    H, W = img.shape[:2]
    tan_half_fov = math.tan(math.radians(hfov_deg) / 2.0)
    if tan_half_fov <= 0:
        return
    cx = W / 2.0

    def _offset_px(depth_m: float) -> float:
        if depth_m <= 1e-3:
            return W / 2.0
        lat_norm = corridor_half_m / (depth_m * tan_half_fov)
        return lat_norm * (W / 2.0)

    off_near = _offset_px(near_depth_m)
    off_far = _offset_px(far_depth_m)

    y_near = H - 1
    y_far = int(H * 0.45)

    def _clip_x(x: float) -> int:
        return int(max(0, min(W - 1, x)))

    pts_left = np.array(
        [
            [_clip_x(cx - off_near), y_near],
            [_clip_x(cx - off_far), y_far],
        ],
        dtype=np.int32,
    )
    pts_right = np.array(
        [
            [_clip_x(cx + off_near), y_near],
            [_clip_x(cx + off_far), y_far],
        ],
        dtype=np.int32,
    )
    line_color = (255, 255, 255)
    cv2.polylines(img, [pts_left], False, line_color, 2, cv2.LINE_AA)
    cv2.polylines(img, [pts_right], False, line_color, 2, cv2.LINE_AA)


def _draw_risk_overlay(
    rgb: np.ndarray,
    frame_risk: FrameRisk,
    *,
    hfov_deg: float,
    banner_level: WarningLevel | None = None,
    tracked: list[TrackedObstacle] | None = None,
) -> np.ndarray:
    """Return a copy of `rgb` with corridor, bboxes, labels, and banner.

    `banner_level` overrides the status banner (e.g. after hysteresis
    smoothing); per-obstacle bbox colours still use the raw per-frame level.
    `tracked` overlays track IDs + velocity/TTC on dynamic obstacles.
    """
    out = np.ascontiguousarray(rgb.copy())

    if DRAW_CORRIDOR:
        _draw_corridor(
            out,
            hfov_deg=hfov_deg,
            corridor_half_m=W_BOAT / 2.0 + LAT_MARGIN,
            near_depth_m=CORRIDOR_NEAR_M,
            far_depth_m=CORRIDOR_FAR_M,
        )

    # Build a lookup from obstacle id() to TrackedObstacle for annotation
    _track_lut: dict[int, TrackedObstacle] = {}
    if tracked:
        for t in tracked:
            _track_lut[id(t.obstacle)] = t

    _STATIC_COLOR = (160, 160, 160)
    _MASK_ALPHA = 0.35  # transparency for dynamic obstacle mask overlay

    for r in frame_risk.obstacle_risks:
        obs = r.obstacle
        trk = _track_lut.get(id(obs))

        if obs.is_static:
            # --- Static (shoreline): thin gray contour, no label ---
            if obs.mask is not None:
                contours, _ = cv2.findContours(
                    obs.mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(out, contours, -1, _STATIC_COLOR, 1, cv2.LINE_AA)
        else:
            # --- Dynamic obstacle: mask overlay + corner brackets + label ---
            color = _WARNING_COLORS[r.warning_level]

            # Semi-transparent mask overlay
            if obs.mask is not None:
                tint = np.array(color, dtype=np.uint8)
                overlay_region = out[obs.mask]
                blended = (
                    overlay_region.astype(np.float32) * (1 - _MASK_ALPHA)
                    + tint.astype(np.float32) * _MASK_ALPHA
                ).astype(np.uint8)
                out[obs.mask] = blended

            # Label position from Kalman-smoothed bbox if available
            if trk is not None and trk.smoothed_bbox is not None:
                bx1, by1, _, _ = trk.smoothed_bbox
            else:
                bx1, by1 = obs.x1, obs.y1

            # Info label
            parts = []
            if obs.class_id is not None:
                _CLS_SHORT = {3: "Boat", 4: "Buoy", 5: "Swim", 6: "Anim", 7: "Float", 8: "Oth"}
                parts.append(_CLS_SHORT.get(obs.class_id, f"C{obs.class_id}"))
            if trk is not None:
                parts.append(f"T{trk.track_id}")
            parts.append(f"d={r.d_forward:.1f}m")
            if trk is not None and trk.v_closing is not None:
                if trk.v_closing > 0:
                    parts.append(f"v={trk.v_closing:.1f}m/s")
                else:
                    parts.append("receding")
            if trk is not None and trk.ttc is not None:
                parts.append(f"TTC={trk.ttc:.1f}s")
            parts.append(f"R={r.risk_score:.2f}")
            label = " ".join(parts)
            y_text = by1 - 6
            if y_text < _BANNER_HEIGHT + 18:
                y_text = by1 + 20
            _draw_text_with_bg(out, label, (bx1, y_text), color)

    _draw_status_banner(
        out,
        banner_level if banner_level is not None else frame_risk.global_warning,
    )
    return out


def _format_current_frame_stats(idx: int, n: int, seg_s: float, depth_s: float) -> str:
    """This frame only: wall time from worker (GUI shows ms only)."""
    seg_s = max(float(seg_s), 1e-9)
    depth_s = max(float(depth_s), 1e-9)
    total_s = seg_s + depth_s
    return (
        f"Frame {idx + 1}/{n}\n"
        f"Seg: {seg_s * 1000:.1f} ms   Depth: {depth_s * 1000:.1f} ms   Total: {total_s * 1000:.1f} ms"
    )


class CombinedSequenceViewer:
    def __init__(
        self,
        seq_dir: Path,
        *,
        model: str = "small",
        est_scale: float = DEFAULT_EST_SCALE,
        infer_max_side: int = 552,
        gui_preview_max: int = GUI_PREVIEW_MAX,
        seg_weights: Path,
        seg_device: str | None = None,
    ) -> None:
        if model not in _DEPTH_MODEL_CHOICES:
            raise ValueError(f"model must be one of {_DEPTH_MODEL_CHOICES}, got {model!r}")
        if infer_max_side < 0:
            raise ValueError(f"infer_max_side must be >= 0 (0 = no limit), got {infer_max_side}")
        if 0 < infer_max_side < 128:
            raise ValueError(f"infer_max_side must be 0 or >= 128, got {infer_max_side}")
        if gui_preview_max < 320:
            raise ValueError(f"gui_preview_max must be >= 320, got {gui_preview_max}")
        self.seq_dir = seq_dir
        self.paths = _list_sequence_images(seq_dir)
        self.model = model
        self.est_scale = est_scale
        self.infer_max_side = infer_max_side
        self._seg_weights = Path(seg_weights).resolve()
        self._seg_device = seg_device
        self._index = 0
        self._pipe = None
        self._pipe_lock = threading.Lock()
        self._work_q: queue.Queue[tuple[str, int | None, int]] = queue.Queue()
        self._result_q: queue.Queue[tuple | None] = queue.Queue()
        self._worker_stop = threading.Event()
        self._req_id = 0
        self._latest_req = 0
        self._photo_rgb: ImageTk.PhotoImage | None = None
        self._photo_seg: ImageTk.PhotoImage | None = None
        self._photo_depth: ImageTk.PhotoImage | None = None
        self._user_preview_cap = gui_preview_max
        self._gui_preview_max = gui_preview_max
        self._playing = True
        self._skip_auto_advance_once = False
        self._fullscreen = True
        self._closing = False
        self._acc_seg_s = 0.0
        self._acc_depth_s = 0.0
        self._acc_timed_frames = 0
        # Frame-level hysteresis state; updated only by the worker thread.
        self._prev_warning_level: WarningLevel = WarningLevel.SAFE
        # SORT tracker instance; lives in the worker thread.
        self._tracker = ObstacleTracker(
            max_age=TRACK_MAX_AGE,
            min_hits=TRACK_MIN_HITS,
            iou_threshold=TRACK_IOU_THRESH,
            fps=TRACK_FPS,
        )

        self.root = tk.Tk()
        self.root.title(f"RGB | Seg | Depth ({model}) - {IMAGE_GLOB_PATTERN}")
        self.root.minsize(1200, 480)
        self._apply_ui_fonts()

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        self._info_wrap = 880
        self._lbl_models = ttk.Label(
            main,
            text=self._models_banner_text(),
            wraplength=self._info_wrap,
            justify=tk.LEFT,
        )
        self._lbl_models.pack(fill=tk.X, pady=(0, 4))
        self._lbl_stats = ttk.Label(
            main,
            text=self._stats_placeholder_text(),
            wraplength=self._info_wrap,
            justify=tk.LEFT,
        )
        self._lbl_stats.pack(fill=tk.X, pady=(0, 6))

        panes = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)

        rgb_f = ttk.LabelFrame(panes, text="RGB + Risk Overlay", padding=4)
        seg_f = ttk.LabelFrame(panes, text=f"Segmentation | SegFormer ({_SEG_HF_ID})", padding=4)
        dep_f = ttk.LabelFrame(
            panes,
            text=f"Depth | Depth-Anything ({model}, inferno)",
            padding=4,
        )
        panes.add(rgb_f, weight=1)
        panes.add(seg_f, weight=1)
        panes.add(dep_f, weight=1)

        self._lbl_rgb = ttk.Label(rgb_f)
        self._lbl_rgb.pack(expand=True)
        self._lbl_seg = ttk.Label(seg_f)
        self._lbl_seg.pack(expand=True)
        self._lbl_depth = ttk.Label(dep_f)
        self._lbl_depth.pack(expand=True)

        nav = ttk.Frame(main)
        nav.pack(fill=tk.X, pady=8)
        self._btn_play = ttk.Button(nav, text="Pause", command=self._toggle_play)
        self._btn_play.pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="Prev", command=self._prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="Next", command=self._next).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav, text="Reload", command=self._reload_current).pack(side=tk.LEFT, padx=12)

        self.root.bind("<Left>", lambda e: self._prev())
        self.root.bind("<Right>", lambda e: self._next())
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("<Escape>", lambda e: self._toggle_fullscreen())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.update_idletasks()
        self._apply_large_display_and_fullscreen()

        if not self.paths:
            messagebox.showerror(
                "No images",
                f"No files matching {IMAGE_GLOB_PATTERN!r} under:\n{seq_dir.resolve()}",
            )
        else:
            self._start_worker()
            self._enqueue_infer(0)
            self._poll_results()

    def _apply_large_display_and_fullscreen(self) -> None:
        try:
            sw = max(self.root.winfo_screenwidth(), 800)
            sh = max(self.root.winfo_screenheight(), 600)
        except tk.TclError:
            sw, sh = 1920, 1080
        want = int(min(max(sw // 2 - 96, sh - 240, 960), 4096))
        self._gui_preview_max = min(want, self._user_preview_cap)
        self._info_wrap = max(sw - 80, 600)
        self._lbl_models.configure(wraplength=self._info_wrap)
        self._lbl_stats.configure(wraplength=self._info_wrap)
        self._set_fullscreen(True)

    def _apply_ui_fonts(self) -> None:
        pt = GUI_FONT_PT
        for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
            try:
                tkfont.nametofont(name).configure(size=pt)
            except tk.TclError:
                pass
        f = tkfont.nametofont("TkDefaultFont")
        fam = f.actual("family")
        style = ttk.Style()
        style.configure("TLabel", font=(fam, pt))
        style.configure("TButton", font=(fam, pt))
        style.configure("TLabelframe.Label", font=(fam, pt))

    def _models_banner_text(self) -> str:
        return (
            f"Depth model: Depth-Anything ({self.model})\n"
            f"Seg model: SegFormer ({_SEG_HF_ID}), weights: {self._seg_weights.name}"
        )

    def _stats_placeholder_text(self) -> str:
        n = len(self.paths)
        if n == 0:
            return "No frames loaded."
        return "Starting..."

    def _stats_running_text(self, idx: int) -> str:
        n = len(self.paths)
        return f"Frame {idx + 1}/{n} (running...)"

    def _print_timing_summary(self) -> None:
        n = self._acc_timed_frames
        print("", flush=True)
        if n <= 0:
            print("=== Inference timing summary: no frames completed ===", flush=True)
            return
        s_seg = self._acc_seg_s
        s_dep = self._acc_depth_s
        s_tot = s_seg + s_dep
        eps = 1e-12
        print(f"=== Inference timing summary ({n} frames) ===", flush=True)
        print(
            f"  Seg:   {s_seg:.3f} s total   {s_seg / n * 1000:.1f} ms/frame avg   "
            f"{n / max(s_seg, eps):.2f} FPS",
            flush=True,
        )
        print(
            f"  Depth: {s_dep:.3f} s total   {s_dep / n * 1000:.1f} ms/frame avg   "
            f"{n / max(s_dep, eps):.2f} FPS",
            flush=True,
        )
        print(
            f"  Serial (seg then depth, end-to-end): {s_tot:.3f} s total   "
            f"{s_tot / n * 1000:.1f} ms/frame avg   {n / max(s_tot, eps):.2f} FPS",
            flush=True,
        )

    def _set_fullscreen(self, on: bool) -> None:
        self._fullscreen = on
        try:
            if sys.platform == "win32":
                if on:
                    self.root.state("zoomed")
                else:
                    self.root.state("normal")
            elif sys.platform == "darwin":
                try:
                    self.root.attributes("-fullscreen", on)
                except tk.TclError:
                    self.root.state("zoomed" if on else "normal")
            else:
                self.root.attributes("-fullscreen", on)
        except tk.TclError:
            try:
                self.root.state("zoomed" if on else "normal")
            except tk.TclError:
                pass

    def _toggle_fullscreen(self) -> None:
        self._set_fullscreen(not self._fullscreen)

    def _start_worker(self) -> None:
        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def _worker(self) -> None:
        while not self._worker_stop.is_set():
            try:
                msg, idx, rid = self._work_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if msg == "stop":
                break
            if msg != "infer" or idx is None:
                continue
            try:
                import torch

                path = self.paths[idx]
                rgb_small = _resize_rgb_max_long_side(_load_rgb_u8(path), self.infer_max_side)
                seg_dev = (
                    torch.device(self._seg_device)
                    if self._seg_device
                    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                t0 = time.perf_counter()
                mask, boundary = compute_segmentation_and_boundary(
                    rgb_small, self._seg_weights, seg_dev,
                )
                seg_s = time.perf_counter() - t0
                seg_rgb = _seg_mask_to_rgb_u8(mask)

                with self._pipe_lock:
                    if self._pipe is None:
                        self._pipe, _ = _load_depth_pipe(_DEPTH_PKG, self.model)
                    t1 = time.perf_counter()
                    depth, _infer_s = _compute_depth_map(rgb_small, self._pipe)
                    depth_s = time.perf_counter() - t1
                    depth = np.asarray(depth, dtype=np.float32)
                    if self.est_scale != 1.0:
                        depth = depth * float(self.est_scale)

                # Auto-detect model type and extract obstacles accordingly
                if boundary is not None:
                    # Multi-class instance-aware model
                    obstacles = extract_obstacles_multiclass(
                        mask, depth, return_masks=True, boundary_prob=boundary,
                    )
                else:
                    # Legacy 3-class model
                    obstacles = extract_obstacles(mask, depth, return_masks=True)
                tracked = self._tracker.update(obstacles)
                frame_risk = assess_frame(
                    obstacles,
                    hfov_deg=HFOV_DEG,
                    w_boat=W_BOAT,
                    lat_margin=LAT_MARGIN,
                    d_safe=D_SAFE,
                    l_safe=L_SAFE,
                )
                most = frame_risk.most_critical
                cur_score = most.risk_score if most is not None else 0.0
                smoothed_level = _apply_hysteresis(self._prev_warning_level, cur_score)
                self._prev_warning_level = smoothed_level
                rgb_overlay = _draw_risk_overlay(
                    rgb_small,
                    frame_risk,
                    hfov_deg=HFOV_DEG,
                    banner_level=smoothed_level,
                    tracked=tracked,
                )
                dep_rgb = _depth_to_rgb_u8(depth)
                self._result_q.put(
                    ("ok", rid, idx, rgb_overlay, seg_rgb, dep_rgb, seg_s, depth_s, frame_risk)
                )
            except Exception as e:
                self._result_q.put(("err", rid, idx, str(e)))

    def _enqueue_infer(self, idx: int) -> None:
        if not self.paths:
            return
        idx = max(0, min(idx, len(self.paths) - 1))
        self._index = idx
        self._req_id += 1
        self._latest_req = self._req_id
        self._lbl_stats.config(text=self._stats_running_text(idx))
        self._work_q.put(("infer", idx, self._req_id))

    def _poll_results(self) -> None:
        try:
            while True:
                item = self._result_q.get_nowait()
                if item is None:
                    continue
                if item[0] == "ok":
                    _, rid, idx, rgb, seg_rgb, dep_rgb, seg_s, depth_s, frame_risk = item
                    if rid != self._latest_req:
                        continue
                    self._photo_rgb = _array_to_photo(rgb, self._gui_preview_max)
                    self._photo_seg = _array_to_photo(seg_rgb, self._gui_preview_max)
                    self._photo_depth = _array_to_photo(dep_rgb, self._gui_preview_max)
                    self._lbl_rgb.configure(image=self._photo_rgb)
                    self._lbl_seg.configure(image=self._photo_seg)
                    self._lbl_depth.configure(image=self._photo_depth)
                    self._lbl_stats.config(
                        text=_format_current_frame_stats(idx, len(self.paths), seg_s, depth_s)
                    )
                    self._acc_seg_s += float(seg_s)
                    self._acc_depth_s += float(depth_s)
                    self._acc_timed_frames += 1

                    skip = self._skip_auto_advance_once
                    if skip:
                        self._skip_auto_advance_once = False
                    elif self._playing:
                        n = len(self.paths)
                        if idx >= n - 1:
                            self._playing = False
                            self._btn_play.config(text="Play")
                            self.root.after(200, self._on_close)
                        else:
                            self._enqueue_infer(idx + 1)
                else:
                    _, rid, idx, err = item
                    if rid != self._latest_req:
                        continue
                    self._playing = False
                    self._btn_play.config(text="Play")
                    self._lbl_stats.config(text=f"Error on frame {idx + 1}/{len(self.paths)}\n{err}")
                    messagebox.showerror("Inference failed", str(err))
        except queue.Empty:
            pass
        if not self._closing:
            try:
                if self.root.winfo_exists():
                    self.root.after(16, self._poll_results)
            except tk.TclError:
                pass

    def _toggle_play(self) -> None:
        if not self.paths:
            return
        self._playing = not self._playing
        self._btn_play.config(text="Pause" if self._playing else "Play")
        if self._playing:
            n = len(self.paths)
            nxt = self._index + 1
            if nxt >= n:
                nxt = 0
            self._enqueue_infer(nxt)

    def _prev(self) -> None:
        if not self.paths:
            return
        self._playing = False
        self._btn_play.config(text="Play")
        self._enqueue_infer(self._index - 1)

    def _next(self) -> None:
        if not self.paths:
            return
        self._playing = False
        self._btn_play.config(text="Play")
        self._enqueue_infer(self._index + 1)

    def _reload_current(self) -> None:
        self._playing = False
        self._btn_play.config(text="Play")
        self._skip_auto_advance_once = True
        self._enqueue_infer(self._index)

    def _on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._worker_stop.set()
        self._work_q.put(("stop", None, -1))
        self._print_timing_summary()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if DEPTH_MODEL not in _DEPTH_MODEL_CHOICES:
        raise ValueError(f"DEPTH_MODEL must be one of {_DEPTH_MODEL_CHOICES}, got {DEPTH_MODEL!r}")
    if INFER_MAX_SIDE < 0:
        raise ValueError(f"INFER_MAX_SIDE must be >= 0, got {INFER_MAX_SIDE}")
    if 0 < INFER_MAX_SIDE < 128:
        raise ValueError(f"INFER_MAX_SIDE must be 0 or >= 128, got {INFER_MAX_SIDE}")
    if GUI_PREVIEW_MAX < 320:
        raise ValueError(f"GUI_PREVIEW_MAX must be >= 320, got {GUI_PREVIEW_MAX}")
    if not isinstance(IMAGE_GLOB_PATTERN, str) or not IMAGE_GLOB_PATTERN:
        raise ValueError("IMAGE_GLOB_PATTERN must be a non-empty string")

    seq_dir = Path(SEQ_DIR).resolve()
    seg_w = Path(SEG_WEIGHTS).resolve()
    if not seg_w.is_file():
        raise FileNotFoundError(f"SEG_WEIGHTS not found: {seg_w}")

    app = CombinedSequenceViewer(
        seq_dir,
        model=DEPTH_MODEL,
        est_scale=EST_SCALE,
        infer_max_side=INFER_MAX_SIDE,
        gui_preview_max=GUI_PREVIEW_MAX,
        seg_weights=seg_w,
        seg_device=SEG_DEVICE,
    )
    app.run()


if __name__ == "__main__":
    main()
