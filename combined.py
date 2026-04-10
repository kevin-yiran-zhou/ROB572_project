"""
GUI: Depth-Anything on the davimar_seq_02* image sequence.

Model is selectable via ``--model`` (default: small). Default ``--infer-max-side`` caps the longest image edge (552 px, aspect preserved) before depth. Starts fullscreen (or
maximized); Esc exits fullscreen. Auto-plays once through the sequence then closes the window.
The on-screen **Infer FPS** uses only ``depth_pipe(...)`` time from ``depth._pipeline`` (not GUI
resize/colormap). Display work can still limit how fast frames appear end-to-end.
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

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except ImportError as e:
    raise SystemExit("tkinter is required (usually bundled with Python).") from e

try:
    from PIL import Image, ImageTk
except ImportError as e:
    raise SystemExit("Install Pillow: pip install pillow") from e

try:
    import matplotlib.cm as mpl_cm
except ImportError as e:
    raise SystemExit("Install matplotlib: pip install matplotlib") from e

# One-time inferno table: per-pixel mpl_cm.inferno() was a major CPU bottleneck each frame.
_INFERNO_LUT = (mpl_cm.inferno(np.linspace(0.0, 1.0, 256, dtype=np.float64))[:, :3] * 255.0).astype(
    np.uint8
)

_SEQ_SUBDIR = Path("lars_v1.0.0_images_seq/test/images_seq")
_PREFIX = "davimar_seq_28"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_PKG_DEPTH = _ROOT / "depth"
_MODEL_CHOICES = ("small", "base", "large")

# GUI preview: cap longest edge so PIL+PhotoImage don’t dominate frame time; BOX/NEAREST is fast.
_GUI_PREVIEW_SIDE_CAP = 1280
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


def _list_sequence_images(seq_dir: Path) -> list[Path]:
    if not seq_dir.is_dir():
        return []
    paths = [
        p
        for p in seq_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in _IMAGE_SUFFIXES
        and p.name.startswith(_PREFIX)
    ]
    return sorted(paths, key=_natural_sort_key)


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


class DepthSequenceViewer:
    def __init__(
        self,
        seq_dir: Path,
        *,
        model: str = "small",
        est_scale: float = DEFAULT_EST_SCALE,
        infer_max_side: int = 552,
        gui_preview_max: int = _GUI_PREVIEW_SIDE_CAP,
    ) -> None:
        if model not in _MODEL_CHOICES:
            raise ValueError(f"model must be one of {_MODEL_CHOICES}, got {model!r}")
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
        self._index = 0
        self._pipe = None
        self._pipe_lock = threading.Lock()
        self._work_q: queue.Queue[tuple[str, int | None, int]] = queue.Queue()
        self._result_q: queue.Queue[tuple | None] = queue.Queue()
        self._worker_stop = threading.Event()
        self._req_id = 0
        self._latest_req = 0
        self._photo_left: ImageTk.PhotoImage | None = None
        self._photo_right: ImageTk.PhotoImage | None = None
        self._user_preview_cap = gui_preview_max
        self._gui_preview_max = gui_preview_max
        self._infer_device_label: str | None = None
        self._device_line_applied = False
        self._playing = True
        self._skip_auto_advance_once = False
        self._infer_fps_ema = 0.0
        self._fullscreen = True
        self._closing = False

        self.root = tk.Tk()
        self.root.title(f"Depth-Anything ({model}) — davimar_seq_02")
        self.root.minsize(900, 480)

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

        left_f = ttk.LabelFrame(panes, text="RGB", padding=4)
        right_f = ttk.LabelFrame(panes, text=f"Depth ({model}, inferno)", padding=4)
        panes.add(left_f, weight=1)
        panes.add(right_f, weight=1)

        self._lbl_left = ttk.Label(left_f)
        self._lbl_left.pack(expand=True)
        self._lbl_right = ttk.Label(right_f)
        self._lbl_right.pack(expand=True)

        status_row = ttk.Frame(main)
        status_row.pack(fill=tk.X, pady=4)
        self._lbl_busy = ttk.Label(status_row, text="")
        self._lbl_busy.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._lbl_fps = ttk.Label(status_row, text="Infer FPS: —", width=18)
        self._lbl_fps.pack(side=tk.RIGHT, padx=(8, 0))

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
                f"No files starting with {_PREFIX!r} under:\n{seq_dir.resolve()}",
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
        if hasattr(self, "_info"):
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
            return f"Folder: {self.seq_dir.resolve()}\n(no matching images)"
        ims = self.infer_max_side
        infer_note = (
            "model input: full resolution (no resize)"
            if ims <= 0
            else f"model input: long edge ≤ {ims}px (aspect preserved)"
        )
        lines = [
            f"Folder: {self.seq_dir.resolve()}",
            (
                f"{n} frames ({_PREFIX}*), model={self.model!r}, est_scale={self.est_scale}, "
                f"{infer_note}, GUI preview max side {self._gui_preview_max}px"
            ),
        ]
        if self._infer_device_label:
            lines.append(self._infer_device_label)
        return "\n".join(lines)

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
                path = self.paths[idx]
                rgb_small = _resize_rgb_max_long_side(_load_rgb_u8(path), self.infer_max_side)
                with self._pipe_lock:
                    if self._pipe is None:
                        self._pipe, self._infer_device_label = _load_depth_pipe(_PKG_DEPTH, self.model)
                    depth, infer_s = _compute_depth_map(rgb_small, self._pipe)
                    depth = np.asarray(depth, dtype=np.float32)
                    if self.est_scale != 1.0:
                        depth = depth * float(self.est_scale)

                rgb = rgb_small
                dep_rgb = _depth_to_rgb_u8(depth)
                self._result_q.put(("ok", rid, idx, rgb, dep_rgb, str(path), float(infer_s)))
            except Exception as e:
                self._result_q.put(("err", rid, idx, str(e)))

    def _enqueue_infer(self, idx: int) -> None:
        if not self.paths:
            return
        idx = max(0, min(idx, len(self.paths) - 1))
        self._index = idx
        self._req_id += 1
        self._latest_req = self._req_id
        self._lbl_busy.config(text=f"Running depth… {idx + 1}/{len(self.paths)}")
        self._work_q.put(("infer", idx, self._req_id))

    def _poll_results(self) -> None:
        try:
            while True:
                item = self._result_q.get_nowait()
                if item is None:
                    continue
                if item[0] == "ok":
                    _, rid, idx, rgb, dep_rgb, path_str, infer_s = item
                    if rid != self._latest_req:
                        continue
                    self._photo_left = _array_to_photo(rgb, self._gui_preview_max)
                    self._photo_right = _array_to_photo(dep_rgb, self._gui_preview_max)
                    self._lbl_left.configure(image=self._photo_left)
                    self._lbl_right.configure(image=self._photo_right)
                    self._lbl_busy.config(
                        text=f"{idx + 1}/{len(self.paths)} — {Path(path_str).name}"
                    )
                    if not self._device_line_applied and self._infer_device_label:
                        self._device_line_applied = True
                        self._info.config(text=self._status_text())
                    t_inf = max(float(infer_s), 1e-9)
                    inst = 1.0 / t_inf
                    self._infer_fps_ema = (
                        inst if self._infer_fps_ema <= 0 else 0.85 * self._infer_fps_ema + 0.15 * inst
                    )
                    self._lbl_fps.config(text=f"Infer FPS: {self._infer_fps_ema:.2f}")

                    skip = self._skip_auto_advance_once
                    if skip:
                        self._skip_auto_advance_once = False
                    elif self._playing:
                        n = len(self.paths)
                        if idx >= n - 1:
                            self._playing = False
                            if hasattr(self, "_btn_play"):
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
                    messagebox.showerror("Depth failed", str(err))
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
    import argparse

    p = argparse.ArgumentParser(description="Depth-Anything GUI on davimar_seq_02* frames.")
    p.add_argument(
        "--model",
        type=str,
        choices=list(_MODEL_CHOICES),
        default="small",
        help="Checkpoint under depth/model/<model>/ (default: small).",
    )
    p.add_argument(
        "--infer-max-side",
        type=int,
        default=552,
        metavar="N",
        help=(
            "Before depth: if N>0, resize so the longest side is at most N pixels (aspect ratio kept). "
            "Uses fast BOX downscaling. N=0 means no resize (full camera resolution). "
            "Default: %(default)s."
        ),
    )
    p.add_argument(
        "--preview-max",
        type=int,
        default=_GUI_PREVIEW_SIDE_CAP,
        metavar="PX",
        help=(
            "Max longest edge (pixels) for RGB/depth thumbnails in the GUI only. "
            "Separate from --infer-max-side. Default: %(default)s."
        ),
    )
    p.add_argument(
        "--seq-dir",
        type=Path,
        default=None,
        help=f"Override image folder (default: <repo>/{_SEQ_SUBDIR.as_posix()})",
    )
    args = p.parse_args()
    if args.infer_max_side < 0:
        p.error("--infer-max-side must be >= 0")
    if 0 < args.infer_max_side < 128:
        p.error("--infer-max-side must be 0 or >= 128")
    if args.preview_max < 320:
        p.error("--preview-max must be >= 320")
    seq_dir = (args.seq_dir or (_ROOT / _SEQ_SUBDIR)).resolve()
    app = DepthSequenceViewer(
        seq_dir,
        model=args.model,
        infer_max_side=args.infer_max_side,
        gui_preview_max=args.preview_max,
    )
    app.run()


if __name__ == "__main__":
    main()
