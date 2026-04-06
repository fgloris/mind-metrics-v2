from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, ParallelEncoderVideoMaskGenerator

ArrayMask = np.ndarray

@dataclass
class FrameMaskData:
    frame_idx: int
    mask_id: int
    mask: ArrayMask                    # [H, W] bool, frame resolution
    area: int
    bbox_xywh: Tuple[int, int, int, int]
    patch_indices: Optional[np.ndarray] = None
    predicted_iou: Optional[float] = None
    stability_score: Optional[float] = None

def frames_tensor_to_rgb_list(frames: torch.Tensor) -> List[np.ndarray]:
    """
    Convert [T, C, H, W] float tensor in [0, 1] to a list of uint8 RGB frames.
    """
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"frames must be torch.Tensor, got {type(frames)}")
    if frames.ndim != 4:
        raise ValueError(f"frames must have shape [T, C, H, W], got {tuple(frames.shape)}")
    if frames.shape[1] != 3:
        raise ValueError(f"frames channel dimension must be 3, got {frames.shape[1]}")

    frames = frames.detach().cpu().clamp(0.0, 1.0)
    frames = (frames * 255.0).round().to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).contiguous()
    return [frame.numpy() for frame in frames]

def build_sam2_generator(
    checkpoint: str,
    device: str,
    model_type: str = "vit_h",
    *,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    min_mask_region_area: int = 200,
    crop_n_layers: int = 0,
    crop_n_points_downscale_factor: int = 1,
) -> SamAutomaticMaskGenerator:
    """
    Build a frame-wise automatic mask generator.

    Note:
        This utility uses the `segment_anything` package's `SamAutomaticMaskGenerator`.
        Although this file is named sam2_utils.py to match your project naming,
        the generator interface here is the classic automatic mask generator API.
    """
    tqdm.write(f"[SAM2/framewise] Loading automatic mask generator from {checkpoint} on {device}...")

    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    sam.eval()

    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
    )

def build_fast_video_sam_generator(
    checkpoint: str,
    device: str = "cuda:0",
    model_type: str = "vit_h",
    **kwargs: Any,
) -> ParallelEncoderVideoMaskGenerator:
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device=device)
    model.eval()
    return ParallelEncoderVideoMaskGenerator(model=model, **kwargs)

def build_nonoverlap_id_mask(anns: List[Dict[str, Any]], img_h: int, img_w: int) -> np.ndarray:
    """
    Convert overlapping SAM annotations into a single non-overlap id mask.

    Larger regions are filled first, and smaller regions overwrite later,
    which is the same behavior as your previous minimal prototype.
    """
    id_mask = np.zeros((img_h, img_w), dtype=np.int32)
    anns_sorted = sorted(enumerate(anns, start=1), key=lambda x: x[1]["area"], reverse=True)
    for mask_id, ann in anns_sorted:
        seg = np.asarray(ann["segmentation"]).astype(bool)
        id_mask[seg] = mask_id
    return id_mask



def mask_to_patch_indices(
    mask: np.ndarray,
    patch_grid_size: int,
    coverage_thresh: float = 0.3,
) -> np.ndarray:
    """
    Map a mask to patch indices on a square patch grid.

    Requirements:
        - mask shape must be [H, W]
        - H == W
        - H must be divisible by patch_grid_size
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    h, w = mask.shape
    if h != w:
        raise ValueError(
            f"mask_to_patch_indices expects square masks, got {mask.shape}. "
            "Resize frames to a square first if you need patch indices."
        )
    if h % patch_grid_size != 0:
        raise ValueError(
            f"mask size {h} is not divisible by patch_grid_size={patch_grid_size}"
        )

    patch = h // patch_grid_size
    m = mask.astype(np.float32)
    coverage = m.reshape(patch_grid_size, patch, patch_grid_size, patch).transpose(0, 2, 1, 3).mean(axis=(2, 3))
    valid = coverage >= coverage_thresh
    return np.flatnonzero(valid.reshape(-1)).astype(np.int64)



def _ann_to_mask_data(
    ann: Dict[str, Any],
    frame_idx: int,
    mask_id: int,
    patch_grid_size: Optional[int] = None,
    coverage_thresh: float = 0.3,
) -> FrameMaskData:
    mask = np.asarray(ann["segmentation"]).astype(bool)
    x, y, w, h = ann["bbox"]

    patch_indices = None
    if patch_grid_size is not None:
        patch_indices = mask_to_patch_indices(mask, patch_grid_size, coverage_thresh)

    return FrameMaskData(
        frame_idx=frame_idx,
        mask_id=mask_id,
        mask=mask,
        area=int(mask.sum()),
        bbox_xywh=(int(x), int(y), int(w), int(h)),
        patch_indices=patch_indices,
        predicted_iou=float(ann["predicted_iou"]) if "predicted_iou" in ann else None,
        stability_score=float(ann["stability_score"]) if "stability_score" in ann else None,
    )



def generate_masks_on_frame(
    image_rgb: np.ndarray,
    sam_generator: SamAutomaticMaskGenerator,
    frame_idx: int = 0,
    patch_grid_size: Optional[int] = None,
    coverage_thresh: float = 0.3,
    min_patch_count: int = 1,
) -> Tuple[np.ndarray, List[FrameMaskData], List[Dict[str, Any]]]:
    """
    Run automatic mask generation on a single RGB frame.

    Returns:
        id_mask: [H, W] int32, non-overlap mask ids starting from 1
        mask_list: per-mask structured metadata
        anns: raw SAM annotations
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be [H, W, 3], got {image_rgb.shape}")
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    anns = sam_generator.generate(
        image_rgb
    )
    id_mask = build_nonoverlap_id_mask(anns, image_rgb.shape[0], image_rgb.shape[1])

    mask_list: List[FrameMaskData] = []
    for mask_id, ann in enumerate(anns, start=1):
        item = _ann_to_mask_data(
            ann=ann,
            frame_idx=frame_idx,
            mask_id=mask_id,
            patch_grid_size=patch_grid_size,
            coverage_thresh=coverage_thresh,
        )
        if item.patch_indices is not None and len(item.patch_indices) < min_patch_count:
            continue
        mask_list.append(item)

    return id_mask, mask_list, anns



def run_framewise_sam2_on_tensor(
    frames: torch.Tensor,
    generator: Optional[SamAutomaticMaskGenerator],
    patch_grid_size: Optional[int] = None,
    coverage_thresh: float = 0.3,
    min_patch_count: int = 1,
) -> Dict[str, Any]:
    """
    Run frame-wise automatic segmentation on a video tensor.

    Important:
        - Each frame is segmented independently.
        - Object ids are only meaningful within the same frame.
        - There is no temporal association across frames.

    Returns:
        {
            "video_segments": {frame_idx: {mask_id: bool_mask}},
            "frame_mask_data": {frame_idx: [FrameMaskData, ...]},
            "id_masks": {frame_idx: id_mask_int32},
            "video_size": (H, W),
            "mode": "framewise_automatic_mask_generator",
        }
    """
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(f"frames must be non-empty [T, C, H, W], got {tuple(frames.shape)}")

    rgb_frames = frames_tensor_to_rgb_list(frames)
    h, w = rgb_frames[0].shape[:2]

    if patch_grid_size is not None and h != w:
        raise ValueError(
            f"patch_grid_size={patch_grid_size} requires square frames, but got {(h, w)}. "
            "Use patch_grid_size=None or resize frames to square before calling this function."
        )

    video_segments: Dict[int, Dict[int, ArrayMask]] = {}
    frame_mask_data: Dict[int, List[FrameMaskData]] = {}
    id_masks: Dict[int, np.ndarray] = {}

    for frame_idx, frame_rgb in enumerate(rgb_frames):
        id_mask, mask_list, _ = generate_masks_on_frame(
            image_rgb=frame_rgb,
            sam_generator=generator,
            frame_idx=frame_idx,
            patch_grid_size=patch_grid_size,
            coverage_thresh=coverage_thresh,
            min_patch_count=min_patch_count,
        )
        id_masks[frame_idx] = id_mask
        frame_mask_data[frame_idx] = mask_list
        video_segments[frame_idx] = {item.mask_id: item.mask for item in mask_list}
        tqdm.write(f"[SAM2/framewise] frame={frame_idx:03d} masks={len(mask_list)}")

    return {
        "video_segments": video_segments,
        "frame_mask_data": frame_mask_data,
        "id_masks": id_masks,
        "video_size": (h, w),
        "mode": "framewise_automatic_mask_generator",
    }
