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
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from depth._constants import DEFAULT_EST_SCALE
from depth._pipeline import _compute_depth_map, _load_depth_pipe, load_image_for_depth
from segmentation._pipeline import compute_segmentation_mask

import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.cm as mpl_cm

# Depth-Anything
_DEPTH_PKG = _ROOT / "depth"
_DEPTH_MODEL_CHOICES = ("small", "base", "large")

# SegFormer package root (weights path default below)
_SEG_PKG = _ROOT / "segmentation"

# ---------------------------------------------------------------------------
# Config — edit here (no command-line flags)
# ---------------------------------------------------------------------------
DEPTH_MODEL: str = "small"  # "small" | "base" | "large"
EST_SCALE: float = DEFAULT_EST_SCALE
INFER_MAX_SIDE: int = 552  # 0 = no resize before seg/depth; else max long edge (px)
GUI_PREVIEW_MAX: int = 1280  # longest edge for on-screen thumbnails

SEQ_DIR: Path = _ROOT / "lars_v1.0.0_images_seq/test/images_seq"
# Multiple prefixes: each group is sorted (natural order), then groups are concatenated in list order.
FILENAME_PREFIXES: list[str] = ["davimar_seq_30", "davimar_seq_31", "davimar_seq_32"]
IMAGE_SUFFIXES: list[str] = [".jpg"]

SEG_WEIGHTS: Path = _SEG_PKG / "model" / "segformer_baseline.pth"
SEG_DEVICE: str | None = None  # None = auto; or "cpu", "cuda:0", ...
# ---------------------------------------------------------------------------

# LUT for depth colormap (faster than calling inferno per pixel each frame).
_INFERNO_LUT = (mpl_cm.inferno(np.linspace(0.0, 1.0, 256, dtype=np.float64))[:, :3] * 255.0).astype(
    np.uint8
)
_SEG_LUT = np.zeros((256, 3), dtype=np.uint8)
_SEG_LUT[0] = (255, 0, 0)
_SEG_LUT[1] = (0, 0, 255)
_SEG_LUT[2] = (0, 255, 0)
_SEG_LUT[255] = (0, 0, 0)

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
    suf = _normalized_suffixes(list(IMAGE_SUFFIXES))
    seen: set[str] = set()
    out: list[Path] = []
    for prefix in FILENAME_PREFIXES:
        batch = [
            p
            for p in seq_dir.iterdir()
            if p.is_file() and p.suffix.lower() in suf and p.name.startswith(prefix)
        ]
        for p in sorted(batch, key=_natural_sort_key):
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out


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

        self.root = tk.Tk()
        pref = " + ".join(f"{p}*" for p in FILENAME_PREFIXES)
        self.root.title(f"RGB · Seg · Depth ({model}) — {pref}")
        self.root.minsize(1200, 480)

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        self._info_wrap = 880
        info = ttk.Label(
            main,
            text=self._status_text(),
            wraplength=self._info_wrap,
        )
        info.pack(fill=tk.X, pady=(0, 6))
        self._info = info

        panes = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)

        rgb_f = ttk.LabelFrame(panes, text="RGB", padding=4)
        seg_f = ttk.LabelFrame(panes, text="Segmentation (SegFormer)", padding=4)
        dep_f = ttk.LabelFrame(panes, text=f"Depth ({model}, inferno)", padding=4)
        panes.add(rgb_f, weight=1)
        panes.add(seg_f, weight=1)
        panes.add(dep_f, weight=1)

        self._lbl_rgb = ttk.Label(rgb_f)
        self._lbl_rgb.pack(expand=True)
        self._lbl_seg = ttk.Label(seg_f)
        self._lbl_seg.pack(expand=True)
        self._lbl_depth = ttk.Label(dep_f)
        self._lbl_depth.pack(expand=True)

        self._lbl_busy = ttk.Label(main, text="")
        self._lbl_busy.pack(fill=tk.X, pady=4)

        nav = ttk.Frame(main)
        nav.pack(fill=tk.X, pady=6)
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
                f"No files for prefixes {FILENAME_PREFIXES!r} under:\n{seq_dir.resolve()}",
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
        self._info.configure(wraplength=self._info_wrap)
        self._set_fullscreen(True)

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

    def _status_text(self) -> str:
        n = len(self.paths)
        if n == 0:
            return f"{self.seq_dir.resolve()}\n(no images for prefixes {FILENAME_PREFIXES})"
        ims = self.infer_max_side
        infer_note = "full res" if ims <= 0 else f"long edge ≤ {ims}px"
        seg_name = self._seg_weights.name
        return (
            f"{self.seq_dir.resolve()}\n"
            f"{n} frames · depth={self.model} · seg={seg_name} · {infer_note} · "
            f"preview ≤ {self._gui_preview_max}px"
        )

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
                mask = compute_segmentation_mask(rgb_small, self._seg_weights, seg_dev)
                seg_rgb = _seg_mask_to_rgb_u8(mask)

                with self._pipe_lock:
                    if self._pipe is None:
                        self._pipe, _ = _load_depth_pipe(_DEPTH_PKG, self.model)
                    depth, _infer_s = _compute_depth_map(rgb_small, self._pipe)
                    depth = np.asarray(depth, dtype=np.float32)
                    if self.est_scale != 1.0:
                        depth = depth * float(self.est_scale)

                rgb = rgb_small
                dep_rgb = _depth_to_rgb_u8(depth)
                self._result_q.put(("ok", rid, idx, rgb, seg_rgb, dep_rgb, str(path)))
            except Exception as e:
                self._result_q.put(("err", rid, idx, str(e)))

    def _enqueue_infer(self, idx: int) -> None:
        if not self.paths:
            return
        idx = max(0, min(idx, len(self.paths) - 1))
        self._index = idx
        self._req_id += 1
        self._latest_req = self._req_id
        self._lbl_busy.config(text=f"Inferring… {idx + 1}/{len(self.paths)}")
        self._work_q.put(("infer", idx, self._req_id))

    def _poll_results(self) -> None:
        try:
            while True:
                item = self._result_q.get_nowait()
                if item is None:
                    continue
                if item[0] == "ok":
                    _, rid, idx, rgb, seg_rgb, dep_rgb, path_str = item
                    if rid != self._latest_req:
                        continue
                    self._photo_rgb = _array_to_photo(rgb, self._gui_preview_max)
                    self._photo_seg = _array_to_photo(seg_rgb, self._gui_preview_max)
                    self._photo_depth = _array_to_photo(dep_rgb, self._gui_preview_max)
                    self._lbl_rgb.configure(image=self._photo_rgb)
                    self._lbl_seg.configure(image=self._photo_seg)
                    self._lbl_depth.configure(image=self._photo_depth)
                    self._lbl_busy.config(
                        text=f"{idx + 1}/{len(self.paths)} — {Path(path_str).name}"
                    )

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
                    self._lbl_busy.config(text=f"Error: {err}")
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
    if not FILENAME_PREFIXES or not all(isinstance(p, str) and p for p in FILENAME_PREFIXES):
        raise ValueError("FILENAME_PREFIXES must be a non-empty list of non-empty strings")
    if not IMAGE_SUFFIXES:
        raise ValueError("IMAGE_SUFFIXES must be a non-empty list")

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
