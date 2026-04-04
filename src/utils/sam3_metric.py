import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor


ArrayMask = np.ndarray


def _resolve_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str, dtype: Optional[torch.dtype] = None) -> torch.dtype:
    if dtype is not None:
        return dtype
    return torch.bfloat16 if device.startswith("cuda") else torch.float32

def _to_single_mask_2d(mask_like) -> np.ndarray:
    arr = np.asarray(mask_like)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a single 2D mask after squeeze, got shape {arr.shape}")
    return arr

@torch.inference_mode()
def load_sam3_model(model_name: str, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
    device = _resolve_device(device)
    dtype = _resolve_dtype(device, dtype)
    tqdm.write(f"[SAM3] Loading model from {model_name} on {device} ({dtype})...")
    model = Sam3TrackerVideoModel.from_pretrained(model_name).to(device, dtype=dtype)
    processor = Sam3TrackerVideoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor, device, dtype



def frames_tensor_to_sam3_video(frames: torch.Tensor) -> List[np.ndarray]:
    """
    Convert VideoStreamReader output to SAM3 video format.

    Args:
        frames: [T, C, H, W], float in [0, 1]

    Returns:
        list[np.ndarray], each [H, W, C] uint8 RGB
    """
    if not isinstance(frames, torch.Tensor):
        raise TypeError(f"frames must be torch.Tensor, got {type(frames)}")
    if frames.ndim != 4:
        raise ValueError(f"frames must have shape [T, C, H, W], got {tuple(frames.shape)}")
    if frames.shape[1] != 3:
        raise ValueError(f"frames channel dimension must be 3, got {frames.shape[1]}")

    frames = frames.detach().cpu().clamp(0.0, 1.0)
    frames = (frames * 255.0).round().to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).contiguous()  # [T, H, W, C]
    return [frame.numpy() for frame in frames]



def make_grid_points(width: int, height: int, grid_size: int = 8, margin_ratio: float = 0.08) -> List[Tuple[int, int]]:
    xs = np.linspace(int(width * margin_ratio), int(width * (1 - margin_ratio)), grid_size)
    ys = np.linspace(int(height * margin_ratio), int(height * (1 - margin_ratio)), grid_size)
    return [(int(round(x)), int(round(y))) for y in ys for x in xs]



def mask_iou(mask_a: ArrayMask, mask_b: ArrayMask) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)



def mask_center_point(mask: ArrayMask) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [int(round(xs.mean())), int(round(ys.mean()))]


def mask_xyxy_box(mask: ArrayMask) -> Optional[List[int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max())
    y1 = int(ys.max())
    return [x0, y0, x1, y1]


def _build_reinit_prompts(filtered_items: List[Dict[str, Any]], prefer_mask_prompt: bool = True):
    obj_ids: List[int] = []
    object_masks: List[np.ndarray] = []
    object_boxes: List[List[int]] = []
    object_points: List[List[List[int]]] = []
    object_labels: List[List[int]] = []

    cur_id = 1
    for item in filtered_items:
        mask = np.asarray(item["mask"]).astype(bool)
        point = mask_center_point(mask)
        box = mask_xyxy_box(mask)
        if point is None:
            continue
        obj_ids.append(cur_id)
        object_points.append([point])
        object_labels.append([1])
        object_masks.append(mask.astype(np.float32))
        object_boxes.append(box if box is not None else [point[0], point[1], point[0], point[1]])
        cur_id += 1

    return obj_ids, object_masks, object_boxes, object_points, object_labels


def _add_reinit_inputs(
    processor,
    inference_session,
    obj_ids: List[int],
    object_masks: List[np.ndarray],
    object_boxes: List[List[int]],
    object_points: List[List[List[int]]],
    object_labels: List[List[int]],
    frame_idx: int = 0,
) -> str:
    if len(obj_ids) == 0:
        return "none"

    errors = []

    mask_candidates = [
        {"input_masks": [object_masks]},
        {"input_masks": np.stack(object_masks, axis=0)},
    ]
    for kwargs in mask_candidates:
        try:
            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_ids=copy.deepcopy(obj_ids),
                **kwargs,
            )
            return "mask"
        except Exception as e:
            errors.append(f"mask_prompt_failed: {type(e).__name__}: {e}")

    box_candidates = [
        {"input_boxes": [object_boxes]},
        {"input_boxes": object_boxes},
    ]
    for kwargs in box_candidates:
        try:
            processor.add_inputs_to_inference_session(
                inference_session=inference_session,
                frame_idx=frame_idx,
                obj_ids=copy.deepcopy(obj_ids),
                **kwargs,
            )
            return "box"
        except Exception as e:
            errors.append(f"box_prompt_failed: {type(e).__name__}: {e}")

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=frame_idx,
        obj_ids=copy.deepcopy(obj_ids),
        input_points=[object_points],
        input_labels=[object_labels],
    )
    if errors:
        tqdm.write("[SAM3] Reinit fallback to point prompt. " + " | ".join(errors))
    return "point"



def filter_masks(
    masks: List[ArrayMask],
    scores: Optional[List[float]] = None,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.5,
    max_border_touch_ratio: float = 0.5,
    iou_dedup_thresh: float = 0.8,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    if len(masks) == 0:
        return []

    h, w = masks[0].shape
    total_area = h * w

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    items = []
    for i, mask in enumerate(masks):
        m = mask.astype(bool)
        area = int(m.sum())
        area_ratio = area / max(total_area, 1)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        border_touch = int(np.logical_and(m, border).sum())
        border_touch_ratio = border_touch / max(area, 1)
        if border_touch_ratio > max_border_touch_ratio:
            continue

        score = 0.0 if scores is None or i >= len(scores) else float(scores[i])
        items.append({"mask": m, "score": score, "area": area})

    items.sort(key=lambda x: (x["score"], x["area"]), reverse=True)

    kept = []
    for item in items:
        duplicated = any(mask_iou(item["mask"], kept_item["mask"]) > iou_dedup_thresh for kept_item in kept)
        if not duplicated:
            kept.append(item)
        if len(kept) >= top_k:
            break
    return kept


@torch.inference_mode()
def propose_objects_from_grid(
    model,
    processor,
    inference_session,
    frame_idx: int,
    width: int,
    height: int,
    grid_size: int = 8,
    mask_threshold: float = 0.3,
) -> Tuple[List[ArrayMask], Optional[List[float]]]:
    grid_points = make_grid_points(width, height, grid_size=grid_size)

    obj_ids = list(range(1, len(grid_points) + 1))
    input_points = [[[list(point)] for point in grid_points]]
    input_labels = [[[1] for _ in grid_points]]

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=frame_idx,
        obj_ids=obj_ids,
        input_points=input_points,
        input_labels=input_labels,
    )

    outputs = model(inference_session=inference_session, frame_idx=frame_idx)
    masks = processor.post_process_masks(
        [outputs.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
        binarize=False,
    )[0]

    masks_np = masks.detach().float().cpu().numpy()
    scores = None
    for candidate_name in ["pred_iou", "iou_scores", "object_scores", "pred_scores"]:
        if hasattr(outputs, candidate_name):
            value = getattr(outputs, candidate_name)
            if value is not None:
                scores = value.detach().float().cpu().numpy().reshape(-1).tolist()
                break

    bin_masks = []
    for m in masks_np:
        m2d = _to_single_mask_2d(m)
        bin_masks.append(m2d > mask_threshold)
    return bin_masks, scores


@torch.inference_mode()
def reinitialize_with_filtered_objects(
    model,
    processor,
    video_frames: List[np.ndarray],
    filtered_items: List[Dict[str, Any]],
    device: str,
    dtype: torch.dtype,
):
    new_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        dtype=dtype,
    )

    obj_ids, object_masks, object_boxes, object_points, object_labels = _build_reinit_prompts(filtered_items)

    if len(obj_ids) == 0:
        print('warning: obj_ids is empty!')
        return new_session, {"obj_ids": [], "first_frame_masks": None}

    prompt_mode = _add_reinit_inputs(
        processor=processor,
        inference_session=new_session,
        obj_ids=obj_ids,
        object_masks=object_masks,
        object_boxes=object_boxes,
        object_points=object_points,
        object_labels=object_labels,
        frame_idx=0,
    )

    outputs = model(inference_session=new_session, frame_idx=0)
    first_frame_masks = processor.post_process_masks(
        [outputs.pred_masks],
        original_sizes=[[new_session.video_height, new_session.video_width]],
        binarize=False,
    )[0]

    print('obj_ids reinit:', obj_ids)
    tqdm.write(f"[SAM3] reinit prompt mode: {prompt_mode}")
    return new_session, {"obj_ids": obj_ids, "first_frame_masks": first_frame_masks, "prompt_mode": prompt_mode}


@torch.inference_mode()
def run_grid_prompt_video_tracking_on_tensor(
    frames: torch.Tensor,
    model,
    processor,
    device: str,
    dtype: torch.dtype,
    grid_size: int = 8,
    max_objects: int = 10,
    min_area_ratio: float = 0.002,
    max_area_ratio: float = 0.5,
    max_border_touch_ratio: float = 0.5,
    iou_dedup_thresh: float = 0.8,
    mask_threshold: float = 0.3,
) -> Dict[str, Any]:
    if frames.ndim != 4 or frames.shape[0] == 0:
        raise ValueError(f"frames must be non-empty [T, C, H, W], got {tuple(frames.shape)}")

    video_frames = frames_tensor_to_sam3_video(frames)
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        dtype=dtype,
    )

    width = inference_session.video_width
    height = inference_session.video_height

    raw_masks, raw_scores = propose_objects_from_grid(
        model=model,
        processor=processor,
        inference_session=inference_session,
        frame_idx=0,
        width=width,
        height=height,
        grid_size=grid_size,
        mask_threshold=mask_threshold,
    )

    filtered_items = filter_masks(
        raw_masks,
        scores=raw_scores,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        max_border_touch_ratio=max_border_touch_ratio,
        iou_dedup_thresh=iou_dedup_thresh,
        top_k=max_objects,
    )

    tqdm.write(f"[SAM3] raw candidates: {len(raw_masks)}")
    tqdm.write(f"[SAM3] kept objects: {len(filtered_items)}")

    clean_session, init_info = reinitialize_with_filtered_objects(
        model=model,
        processor=processor,
        video_frames=video_frames,
        filtered_items=filtered_items,
        device=device,
        dtype=dtype,
    )
    obj_ids = init_info["obj_ids"]
    first_frame_masks = init_info["first_frame_masks"]
    prompt_mode = init_info.get("prompt_mode", "unknown")

    video_segments: Dict[int, Dict[int, ArrayMask]] = {}

    # 先把首帧结果存进去
    if first_frame_masks is not None:
        first_masks_np = first_frame_masks.detach().float().cpu().numpy()
        frame0_result: Dict[int, ArrayMask] = {}
        for i, obj_id in enumerate(obj_ids):
            if i < first_masks_np.shape[0]:
                mask_2d = np.squeeze(first_masks_np[i])
                if mask_2d.ndim != 2:
                    raise ValueError(
                        f"first_frame mask for obj {obj_id} is not 2D after squeeze: {mask_2d.shape}"
                    )
                frame0_result[obj_id] = mask_2d > mask_threshold
        video_segments[0] = frame0_result
    else:
        print('first_frame_masks is None!')
    
    if len(obj_ids) == 0:
        return {
            "obj_ids": [],
            "filtered_items": filtered_items,
            "video_segments": video_segments,
            "video_size": (height, width),
            "prompt_mode": prompt_mode,
            "mask_threshold": mask_threshold,
        }

    # 再传播后续帧
    for output in model.propagate_in_video_iterator(clean_session):
        masks = processor.post_process_masks(
            [output.pred_masks],
            original_sizes=[[clean_session.video_height, clean_session.video_width]],
            binarize=False,
        )[0]
        masks_np = masks.detach().float().cpu().numpy()

        frame_result: Dict[int, ArrayMask] = {}
        for i, obj_id in enumerate(obj_ids):
            if i < masks_np.shape[0]:
                mask_2d = np.squeeze(masks_np[i])
                if mask_2d.ndim != 2:
                    raise ValueError(
                        f"propagated mask for frame {int(output.frame_idx)}, obj {obj_id} "
                        f"is not 2D after squeeze: {mask_2d.shape}"
                    )
                frame_result[obj_id] = mask_2d > mask_threshold

        video_segments[int(output.frame_idx)] = frame_result

    return {
        "obj_ids": obj_ids,
        "filtered_items": filtered_items,
        "video_segments": video_segments,
        "video_size": (height, width),
    }