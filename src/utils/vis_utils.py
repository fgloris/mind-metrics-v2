import os
import json
import math
from typing import List, Tuple, Dict, Any, Optional

import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button


def find_latest_metrics_json(output_dir: str = "output") -> str:
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    json_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".json")
    ]
    if not json_files:
        raise FileNotFoundError(f"No json files found in: {output_dir}")

    json_files.sort(key=os.path.getmtime, reverse=True)
    return json_files[0]


def timestamped_json_path(output_dir: str = "output") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"metrics_{ts}.json"


def tensor_to_nested_list(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [tensor_to_nested_list(v) for v in x]
    if isinstance(x, dict):
        return {k: tensor_to_nested_list(v) for k, v in x.items()}
    return x


def save_metrics_json(payload: Dict[str, Any], output_dir: str = "output") -> Path:
    json_path = timestamped_json_path(output_dir)
    serializable = tensor_to_nested_list(payload)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    return json_path


def load_metrics_json(output_dir: str = "output") -> dict:
    json_path = find_latest_metrics_json(output_dir)
    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print(f"[vis] Loaded metrics json: {json_path}")
    metrics["_json_path"] = json_path
    return metrics


def load_soft_mat_npy_from_metrics(metrics: dict) -> Optional[np.ndarray]:
    meta = metrics.get("meta", {})
    npy_path = meta.get("soft_mat_npy_path")
    if not npy_path:
        print("[vis] soft_mat_npy_path not found in metrics json.")
        return None

    json_path = metrics.get("_json_path")
    candidate_paths = [Path(npy_path)]
    if json_path is not None:
        candidate_paths.append(Path(json_path).parent / Path(npy_path).name)

    for path in candidate_paths:
        if path.exists():
            soft_mats = np.load(path)
            print(f"[vis] Loaded soft_mat npy: {path}, shape={soft_mats.shape}")
            return soft_mats

    print(f"[vis] soft_mat npy not found. Tried: {[str(p) for p in candidate_paths]}")
    return None


def _read_video_frames(video_path: str, start_frame: int, num_frames: int) -> List[np.ndarray]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(video_path)
    frames = []
    for idx in range(start_frame, start_frame + num_frames):
        try:
            frame = reader.get_data(idx)
        except Exception:
            break
        frames.append(frame)
    reader.close()
    return frames


def load_aligned_video_frames_from_metrics(metrics: dict) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    meta = metrics["meta"]
    frames_data = metrics["frames"]

    num_frames = len(frames_data)
    gt_video_path = meta["gt_video_path"]
    pred_video_path = meta["pred_video_path"]
    gt_start_frame = int(meta["gt_start_frame"])
    pred_start_frame = int(meta["pred_start_frame"])

    gt_frames = _read_video_frames(gt_video_path, gt_start_frame, num_frames)
    pred_frames = _read_video_frames(pred_video_path, pred_start_frame, num_frames)

    valid_len = min(len(gt_frames), len(pred_frames), num_frames)
    if valid_len == 0:
        raise RuntimeError("No valid aligned frames could be loaded from videos.")

    gt_frames = gt_frames[:valid_len]
    pred_frames = pred_frames[:valid_len]
    metrics["frames"] = metrics["frames"][:valid_len]

    print(f"[vis] Loaded {valid_len} aligned frames.")
    return gt_frames, pred_frames


def _to_display_image(img):
    if isinstance(img, np.ndarray):
        out = img
    else:
        out = np.asarray(img)

    if out.ndim == 3 and out.shape[0] in (1, 3):
        out = np.transpose(out, (1, 2, 0))

    if out.ndim == 2:
        out = np.stack([out, out, out], axis=-1)

    out = out.astype(np.float32)
    if out.max() > 1.0:
        out = out / 255.0
    out = np.clip(out, 0.0, 1.0)
    return out


def _infer_patch_grid(num_patches: int):
    side = int(math.sqrt(num_patches))
    if side * side != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a square number")
    return side, side


def _patch_index_from_xy(x, y, img_w, img_h, grid_w, grid_h):
    if x is None or y is None:
        return None
    if x < 0 or x >= img_w or y < 0 or y >= img_h:
        return None

    col = min(int(x / img_w * grid_w), grid_w - 1)
    row = min(int(y / img_h * grid_h), grid_h - 1)
    idx = row * grid_w + col
    return idx, row, col


def _patch_rect(row, col, img_w, img_h, grid_w, grid_h):
    patch_w = img_w / grid_w
    patch_h = img_h / grid_h
    x0 = col * patch_w
    y0 = row * patch_h
    return x0, y0, patch_w, patch_h


def _normalize_map(sim_map, mode="minmax", vmin=None, vmax=None):
    sim_map = sim_map.astype(np.float32)

    if mode == "fixed":
        if vmin is None or vmax is None:
            raise ValueError("fixed mode requires vmin and vmax")
        denom = max(vmax - vmin, 1e-8)
        out = (sim_map - vmin) / denom
    else:
        smin = sim_map.min()
        smax = sim_map.max()
        denom = max(smax - smin, 1e-8)
        out = (sim_map - smin) / denom

    return np.clip(out, 0.0, 1.0)


class MetricsFrameViewer:
    def __init__(self, gt_frames: List[np.ndarray], pred_frames: List[np.ndarray], metrics: dict, soft_mats_npy: Optional[np.ndarray] = None):
        self.gt_frames = gt_frames
        self.pred_frames = pred_frames
        self.metrics = metrics
        self.frames_data = metrics["frames"]
        self.soft_mats_npy = soft_mats_npy

        if len(self.gt_frames) != len(self.pred_frames) or len(self.gt_frames) != len(self.frames_data):
            raise ValueError("Frame count mismatch among gt_frames, pred_frames, and metrics['frames'].")
        if self.soft_mats_npy is not None:
            valid_len = min(len(self.frames_data), self.soft_mats_npy.shape[0])
            self.gt_frames = self.gt_frames[:valid_len]
            self.pred_frames = self.pred_frames[:valid_len]
            self.frames_data = self.frames_data[:valid_len]
            self.soft_mats_npy = self.soft_mats_npy[:valid_len]

        self.frame_idx = 0
        self.last_patch_idx = 0

        self.fig = None
        self.ax_a = None
        self.ax_b = None
        self.img_artist_a = None
        self.img_artist_b = None
        self.heat_artist = None
        self.rect_a = None
        self.rect_b = None
        self.text_artist = None
        self.frame_text_artist = None
        self.btn_prev = None
        self.btn_next = None

        self.current_img_a = None
        self.current_img_b = None
        self.current_soft_mat = None

        self.h_a = None
        self.w_a = None
        self.h_b = None
        self.w_b = None

        self.grid_h = None
        self.grid_w = None
        self.has_soft_mat = False

        self._set_current_frame(0)

    def _set_current_frame(self, frame_idx: int):
        self.frame_idx = max(0, min(frame_idx, len(self.frames_data) - 1))
        self.current_img_a = _to_display_image(self.gt_frames[self.frame_idx])
        self.current_img_b = _to_display_image(self.pred_frames[self.frame_idx])

        frame_data = self.frames_data[self.frame_idx]
        self.h_a, self.w_a = self.current_img_a.shape[:2]
        self.h_b, self.w_b = self.current_img_b.shape[:2]

        if self.soft_mats_npy is not None and self.frame_idx < len(self.soft_mats_npy):
            self.current_soft_mat = np.asarray(self.soft_mats_npy[self.frame_idx], dtype=np.float32)
            self.has_soft_mat = True
        elif ("soft_mat" in frame_data) and (frame_data["soft_mat"] is not None):
            self.current_soft_mat = np.asarray(frame_data["soft_mat"], dtype=np.float32)
            self.has_soft_mat = True
        else:
            self.current_soft_mat = None
            self.has_soft_mat = False

        if self.has_soft_mat:
            num_patches = self.current_soft_mat.shape[0]
            self.grid_h, self.grid_w = _infer_patch_grid(num_patches)
        else:
            self.grid_h, self.grid_w = None, None

    def _update_frame_visuals(self):
        self._set_current_frame(self.frame_idx)

        self.img_artist_a.set_data(self.current_img_a)
        self.img_artist_b.set_data(self.current_img_b)
        self.ax_a.set_title(f"Image A - frame {self.frame_idx}")
        self.ax_b.set_title(f"Image B - frame {self.frame_idx}")

        self.rect_a.set_visible(False)
        self.rect_b.set_visible(False)

        frame_data = self.frames_data[self.frame_idx]
        self.frame_text_artist.set_text(
            f"frame={self.frame_idx} / {len(self.frames_data)-1}\n"
            f"soft_score={frame_data['soft_score']:.6f}\n"
            f"hard_score={frame_data['hard_score']:.6f}\n"
            f"soft_mat={'yes' if self.has_soft_mat else 'no'}"
        )

        if self.has_soft_mat:
            self.text_artist.set_text("Move mouse on Image A to visualize soft_mat")
            self.heat_artist.set_data(np.zeros((self.grid_h, self.grid_w), dtype=np.float32))
            self.heat_artist.set_extent((0, self.w_b, self.h_b, 0))
            self.heat_artist.set_alpha(0.0)
        else:
            self.text_artist.set_text("soft_mat not found in json/npy; heatmap disabled")
            self.heat_artist.set_alpha(0.0)

        self.fig.canvas.draw_idle()

    def _update_patch_visuals(self, idx: int, row: int, col: int):
        if not self.has_soft_mat:
            return

        self.last_patch_idx = idx

        x0, y0, pw, ph = _patch_rect(row, col, self.w_a, self.h_a, self.grid_w, self.grid_h)
        self.rect_a.set_xy((x0, y0))
        self.rect_a.set_width(pw)
        self.rect_a.set_height(ph)
        self.rect_a.set_visible(True)

        sim_vec = self.current_soft_mat[idx]
        sim_map = sim_vec.reshape(self.grid_h, self.grid_w)
        sim_map_norm = _normalize_map(sim_map, mode="minmax")

        self.heat_artist.set_data(sim_map_norm)
        self.heat_artist.set_alpha(0.45)

        best_idx = int(np.argmax(sim_vec))
        best_row = best_idx // self.grid_w
        best_col = best_idx % self.grid_w

        bx0, by0, bpw, bph = _patch_rect(best_row, best_col, self.w_b, self.h_b, self.grid_w, self.grid_h)
        self.rect_b.set_xy((bx0, by0))
        self.rect_b.set_width(bpw)
        self.rect_b.set_height(bph)
        self.rect_b.set_visible(True)

        frame_data = self.frames_data[self.frame_idx]
        self.text_artist.set_text(
            f"A patch idx={idx} (row={row}, col={col})\n"
            f"B best idx={best_idx} (row={best_row}, col={best_col})\n"
            f"sim={float(sim_vec[best_idx]):.6f}\n"
            f"soft_score={frame_data['soft_score']:.6f} | hard_score={frame_data['hard_score']:.6f}"
        )
        self.fig.canvas.draw_idle()

    def _redraw_last_patch(self):
        if not self.has_soft_mat:
            return
        idx = max(0, min(self.last_patch_idx, self.current_soft_mat.shape[0] - 1))
        row = idx // self.grid_w
        col = idx % self.grid_w
        self._update_patch_visuals(idx, row, col)

    def _on_move(self, event):
        if not self.has_soft_mat:
            return
        if event.inaxes != self.ax_a:
            return

        parsed = _patch_index_from_xy(
            event.xdata, event.ydata,
            img_w=self.w_a, img_h=self.h_a,
            grid_w=self.grid_w, grid_h=self.grid_h,
        )
        if parsed is None:
            return

        idx, row, col = parsed
        self._update_patch_visuals(idx, row, col)

    def _on_prev(self, event):
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self._update_frame_visuals()
            self._redraw_last_patch()

    def _on_next(self, event):
        if self.frame_idx < len(self.frames_data) - 1:
            self.frame_idx += 1
            self._update_frame_visuals()
            self._redraw_last_patch()

    def show(self):
        self.fig, (self.ax_a, self.ax_b) = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(bottom=0.18)

        self.img_artist_a = self.ax_a.imshow(self.current_img_a)
        self.img_artist_b = self.ax_b.imshow(self.current_img_b)

        self.ax_a.set_title("Image A")
        self.ax_b.set_title("Image B")
        self.ax_a.axis("off")
        self.ax_b.axis("off")

        self.rect_a = Rectangle((0, 0), 10, 10, fill=False, edgecolor="cyan", linewidth=2)
        self.rect_b = Rectangle((0, 0), 10, 10, fill=False, edgecolor="lime", linewidth=2)
        self.ax_a.add_patch(self.rect_a)
        self.ax_b.add_patch(self.rect_b)
        self.rect_a.set_visible(False)
        self.rect_b.set_visible(False)

        heat_h = self.grid_h if self.grid_h is not None else 2
        heat_w = self.grid_w if self.grid_w is not None else 2

        self.heat_artist = self.ax_b.imshow(
            np.zeros((heat_h, heat_w), dtype=np.float32),
            cmap="jet",
            alpha=0.0,
            interpolation="nearest",
            extent=(0, self.w_b, self.h_b, 0),
            vmin=0.0,
            vmax=1.0,
        )

        self.text_artist = self.ax_b.text(
            0.02, 0.02, "",
            transform=self.ax_b.transAxes,
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=4),
        )

        self.frame_text_artist = self.ax_a.text(
            0.02, 0.02, "",
            transform=self.ax_a.transAxes,
            color="white",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=4),
        )

        ax_prev = plt.axes([0.38, 0.05, 0.10, 0.06])
        ax_next = plt.axes([0.52, 0.05, 0.10, 0.06])

        self.btn_prev = Button(ax_prev, "Prev Frame")
        self.btn_next = Button(ax_next, "Next Frame")

        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)

        self._update_frame_visuals()
        plt.show()