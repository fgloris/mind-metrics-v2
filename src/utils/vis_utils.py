import numpy as np
import torch
import cv2
from typing import Dict, List, Any


def _video_to_numpy_uint8(video_frames):
    """
    video_frames:
        - torch.Tensor [T, C, H, W], float in [0,1]
        - or np.ndarray  [T, C, H, W] / [T, H, W, C]

    return:
        np.ndarray [T, H, W, 3], uint8
    """
    if isinstance(video_frames, torch.Tensor):
        arr = video_frames.detach().cpu().float().numpy()
    else:
        arr = np.asarray(video_frames)

    if arr.ndim != 4:
        raise ValueError(f"Expected 4D video frames, got shape {arr.shape}")

    # [T, C, H, W] -> [T, H, W, C]
    if arr.shape[1] in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.shape[-1] in (1, 3):
        pass
    else:
        raise ValueError(f"Cannot infer channel dimension from shape {arr.shape}")

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)

    return arr


def _stable_color_from_obj_id(obj_id: int) -> tuple[int, int, int]:
    """
    给 obj_id 一个稳定的 RGB 颜色
    """
    rng = np.random.default_rng(int(obj_id) * 9973 + 17)
    color = rng.integers(low=40, high=256, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def _mask_center(mask: np.ndarray):
    """
    mask: [H, W] bool / 0-1
    return: (x, y) or None
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x = int(xs.mean())
    y = int(ys.mean())
    return x, y


def visualize_video_segments(
    video_frames,
    video_segments: Dict[int, Dict[int, np.ndarray]],
    alpha: float = 0.45,
    draw_border: bool = True,
    border_thickness: int = 2,
    draw_label: bool = True,
    label_font_scale: float = 0.6,
    label_thickness: int = 2,
) -> List[np.ndarray]:
    """
    可视化 main 里输出格式的 video_segments

    Args:
        video_frames:
            [T, C, H, W] torch.Tensor / np.ndarray, 值域 [0,1]
            或 [T, H, W, C]
        video_segments:
            dict[frame_idx][obj_id] = mask
            其中 mask 可以是 [H, W] / [1, H, W] / torch.Tensor / bool / float
        alpha:
            mask overlay 透明度
        draw_border:
            是否画边界
        border_thickness:
            边界粗细
        draw_label:
            是否在 mask 中心写 obj_id
        label_font_scale:
            文字大小
        label_thickness:
            文字粗细

    Returns:
        vis_frames: list[np.ndarray], 每帧 [H, W, 3], uint8 RGB
    """
    frames = _video_to_numpy_uint8(video_frames)  # [T, H, W, 3]
    T, H, W, _ = frames.shape

    vis_frames: List[np.ndarray] = []

    for t in range(T):
        img = frames[t].copy()

        if t not in video_segments:
            vis_frames.append(img)
            continue

        obj_dict = video_segments[t]

        # 先做 overlay，再画边界和文字
        overlay = img.copy()

        for obj_id, mask_like in obj_dict.items():
            if isinstance(mask_like, torch.Tensor):
                mask = mask_like.detach().cpu().numpy()
            else:
                mask = np.asarray(mask_like)

            mask = np.squeeze(mask)
            if mask.ndim != 2:
                raise ValueError(
                    f"Frame {t}, obj_id {obj_id}: expected 2D mask after squeeze, got {mask.shape}"
                )

            mask_bool = mask > 0
            if not np.any(mask_bool):
                continue

            color = np.array(_stable_color_from_obj_id(obj_id), dtype=np.uint8)
            overlay[mask_bool] = color

        img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)

        for obj_id, mask_like in obj_dict.items():
            if isinstance(mask_like, torch.Tensor):
                mask = mask_like.detach().cpu().numpy()
            else:
                mask = np.asarray(mask_like)

            mask = np.squeeze(mask)
            mask_bool = mask > 0
            if not np.any(mask_bool):
                continue

            color = _stable_color_from_obj_id(obj_id)

            if draw_border:
                mask_u8 = (mask_bool.astype(np.uint8) * 255)
                contours, _ = cv2.findContours(
                    mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    img,
                    contours,
                    contourIdx=-1,
                    color=color,
                    thickness=border_thickness,
                )

            if draw_label:
                center = _mask_center(mask_bool)
                if center is not None:
                    x, y = center
                    text = str(obj_id)

                    # 先画黑底描边，避免看不清
                    cv2.putText(
                        img,
                        text,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_font_scale,
                        (0, 0, 0),
                        thickness=label_thickness + 2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        img,
                        text,
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        label_font_scale,
                        color,
                        thickness=label_thickness,
                        lineType=cv2.LINE_AA,
                    )

        vis_frames.append(img)

    return vis_frames    