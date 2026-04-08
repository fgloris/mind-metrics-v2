import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch

from utils.utils import VideoStreamReader
from utils.dino_utils import load_dinov3_model, extract_dinov3_features
from utils.compute_patch_sim import compute_patch_sim
from utils.vis_utils import save_metrics_json


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _stack_stats(values):
    tensors = [v.detach().cpu() if isinstance(v, torch.Tensor) else torch.tensor(v) for v in values]
    t = torch.stack(tensors)
    return {
        "mean": float(t.mean().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def _save_soft_mats_npy(soft_mats_list, output_dir="output"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    npy_path = output_path / f"soft_mat_{ts}.npy"

    soft_mats_np = np.stack([
        sm.detach().cpu().numpy() if isinstance(sm, torch.Tensor) else np.asarray(sm, dtype=np.float32)
        for sm in soft_mats_list
    ], axis=0).astype(np.float32)
    np.save(npy_path, soft_mats_np)
    return npy_path, list(soft_mats_np.shape)



def main(video_max_frames=512, video_read_batch_size=128, device='cuda:0'):
    gt_dir = '/media/wjp/gingerBackup/mind/mind_data_with_baselines/gt/world_model_1st/2025-10-18-16-31-BoardcastStudio_A-1m01s'
    pred_dir = '/media/wjp/gingerBackup/mind/structured_baselines/v2v/1st_data/mem_test/2025-10-18-16-31-BoardcastStudio_A-1m01s'

    tqdm.write('[Total Stage 1/5]: initialize video stream reader...')

    mark_time = 488
    total_time = 1466
    real_time = min(total_time - mark_time, video_max_frames)

    gt_video_path = os.path.join(gt_dir, 'origin.mp4')
    pred_video_path = os.path.join(pred_dir, 'video.mp4')

    gt_reader = VideoStreamReader(gt_video_path, start_frame=mark_time, total_frames=real_time + mark_time, skip_frames=15)
    pred_reader = VideoStreamReader(pred_video_path, start_frame=0, total_frames=real_time, skip_frames=15)

    dino_model_path = '/home/wjp/Documents/Metric/dinov3/dinov3_vit7b16'
    device = 'cuda:0'

    tqdm.write(f'[Total Stage 2/5]: loading DINOv3 from {dino_model_path} ...')
    dino_model, dino_processor = load_dinov3_model(dino_model_path, device=device)
    dino_processor.size = {"height": 672, "width": 672}

    start_time = time.time()
    tqdm.write('[Total Stage 3/5]: Computing DINOv3 features...')

    gt_feature_list = []
    pred_feature_list = []
    total_processed_frames = 0

    while True:
        tqdm.write('[DINO 1/3] Reading videos...')
        is_ended, gt_frames = gt_reader.read_batch(video_read_batch_size)
        _, pred_frames = pred_reader.read_batch(video_read_batch_size)

        if gt_frames is None or pred_frames is None:
            break

        tqdm.write('[DINO 2/3] Extracting DINOv3 features for gt_frames...')
        gt_feature_batch = extract_dinov3_features(
            gt_frames,
            model=dino_model,
            processor=dino_processor,
            device=device,
            batch_size=2
        )
        gt_feature_list.append(gt_feature_batch)

        tqdm.write('[DINO 3/3] Extracting DINOv3 features for pred_frames...')
        pred_feature_batch = extract_dinov3_features(
            pred_frames,
            model=dino_model,
            processor=dino_processor,
            device=device,
            batch_size=2
        )
        pred_feature_list.append(pred_feature_batch)

        total_processed_frames += int(gt_feature_batch.shape[0])

        if is_ended:
            break

    elapsed_time = time.time() - start_time
    tqdm.write(f'[Total Stage 3/5] Extracting DINOv3 features completed in {elapsed_time:.2f}s')

    if not gt_feature_list or not pred_feature_list:
        raise RuntimeError('No features were extracted from the input videos.')

    gt_feature_list = torch.cat(gt_feature_list, dim=0)
    pred_feature_list = torch.cat(pred_feature_list, dim=0)

    print(gt_feature_list.shape, pred_feature_list.shape)

    soft_score_list, hard_score_list, soft_mats_list = compute_patch_sim(gt_feature_list, pred_feature_list)

    soft_mat_npy_path = None
    soft_mat_npy_shape = None
    if soft_mats_list is not None:
        soft_mat_npy_path, soft_mat_npy_shape = _save_soft_mats_npy(soft_mats_list, output_dir='output')
        tqdm.write(f'[Total Stage 4/5] soft_mat npy saved to {soft_mat_npy_path}')

    frames_payload = []
    for frame_idx, (soft_score, hard_score) in enumerate(zip(soft_score_list, hard_score_list)):
        frames_payload.append({
            'frame_idx': frame_idx,
            'soft_score': _to_float(soft_score),
            'hard_score': _to_float(hard_score),
        })

    soft_stats = _stack_stats(soft_score_list)
    hard_stats = _stack_stats(hard_score_list)

    metrics_payload = {
        'meta': {
            'gt_dir': gt_dir,
            'pred_dir': pred_dir,
            'gt_video_path': gt_video_path,
            'pred_video_path': pred_video_path,
            'gt_start_frame': int(mark_time),
            'pred_start_frame': 0,
            'skip_frames': 15,
            'video_max_frames': int(video_max_frames),
            'video_read_batch_size': int(video_read_batch_size),
            'processed_frames': int(total_processed_frames),
            'dino_model_path': dino_model_path,
            'dino_processor_size': dino_processor.size,
            'feature_shape': list(gt_feature_list.shape),
            'elapsed_seconds': elapsed_time,
            'has_soft_mat_npy': soft_mats_list is not None,
            'soft_mat_npy_path': str(soft_mat_npy_path) if soft_mat_npy_path is not None else None,
            'soft_mat_npy_shape': soft_mat_npy_shape,
        },
        'summary': {
            'num_frames': len(frames_payload),
            'soft_score_mean': soft_stats['mean'],
            'hard_score_mean': hard_stats['mean'],
            'soft_score_min': soft_stats['min'],
            'soft_score_max': soft_stats['max'],
            'hard_score_min': hard_stats['min'],
            'hard_score_max': hard_stats['max'],
        },
        'frames': frames_payload,
    }

    json_path = save_metrics_json(metrics_payload, output_dir='output')
    tqdm.write(f'[Total Stage 5/5] Metrics saved to {json_path}')

    del dino_model, dino_processor
    del gt_reader, pred_reader
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()