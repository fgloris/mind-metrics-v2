import numpy as np
import os
import time
from tqdm import tqdm
import torch
import imageio.v2 as imageio

from utils.utils import VideoStreamReader
from utils.dino_utils import load_dinov3_model, extract_dinov3_features

from utils.compute_patch_sim import compute_patch_sim

from vis import visualize_patch_similarity_interactive

def main(process_batch_size=4, device='cuda:0'):
    gt_dir = '/media/wjp/gingerBackup/mind/mind_data_with_baselines/gt/world_model_1st/2025-10-18-16-31-BoardcastStudio_A-1m01s'
    pred_dir = '/media/wjp/gingerBackup/mind/structured_baselines/v2v/1st_data/mem_test/2025-10-18-16-31-BoardcastStudio_A-1m01s'

    # Stage 1/5: initialize video stream reader
    tqdm.write(f'[Total Stage 1/5]: initialize video stream reader...')

    gt_reader = VideoStreamReader(os.path.join(gt_dir, 'origin.mp4'), start_frame=495)
    pred_reader = VideoStreamReader(os.path.join(pred_dir, 'video.mp4'), start_frame=0)

    # Stage 2/5: loading DINOv3
    dino_model_path = '/home/wjp/Documents/Metric/dinov3/dinov3_vit7b16'
    device = 'cuda:0'
    
    tqdm.write(f'[Total Stage 2/5]: loading DINOv3 from {dino_model_path} ...')
    
    dino_model, dino_processor = load_dinov3_model(dino_model_path, device=device)
    dino_processor.size = {"height": 672, "width": 672}

    # Stage 3/5: Computing DINOv3 features
    start_time = time.time()
    tqdm.write(f'[Total Stage 3/5]: Computing DINOv3 features...')
    
    gt_feature_list = []
    pred_feature_list = []
    while True:
        # Step 1/5: Reading videos
        tqdm.write(f"[DINO 1/3] Reading videos...")
        is_ended, gt_frames = gt_reader.read_batch(process_batch_size)
        _, pred_frames = pred_reader.read_batch(process_batch_size)
        elapsed_time = time.time() - start_time

        if is_ended or gt_frames is None or pred_frames is None:
            break

        # Step 2/5: Extracting DINOv3 features for gt_frames
        tqdm.write(f"[DINO 2/3] Extracting DINOv3 features for gt_frames...")
        gt_feature_batch = extract_dinov3_features(
            gt_frames,
            model=dino_model,
            processor=dino_processor,
            device=device,
            batch_size=2
        ) # [f, 896*896, 768]
        gt_feature_list.append(gt_feature_batch)

        # Step 3/5: Extracting DINOv3 features for pred_frames
        tqdm.write(f"[DINO 3/3] Extracting DINOv3 features for pred_frames...")
        pred_feature_batch = extract_dinov3_features(
            pred_frames,
            model=dino_model,
            processor=dino_processor,
            device=device,
            batch_size=2
        ) # [f, 896*896, 768]
        pred_feature_list.append(pred_feature_batch)

        break

    elapsed_time = time.time() - start_time
    tqdm.write(f"[Total Stage 3/5] Extracting DINOv3 features completed in {elapsed_time:.2f}s")

    gt_feature_list = torch.cat(gt_feature_list, dim=0)
    pred_feature_list = torch.cat(pred_feature_list, dim=0)

    print(gt_feature_list.shape, pred_feature_list.shape)

    result = compute_patch_sim(gt_feature_list, pred_feature_list)

    img_a = gt_frames[0]
    img_b = pred_frames[0]
    joint_sim_0 = result[0]   # [N, N]

    visualize_patch_similarity_interactive(
        img_a=img_a,
        img_b=img_b,
        joint_sim=joint_sim_0,
        title="Joint Similarity Visualization",
        heat_alpha=0.45,
        norm_mode="minmax",   # 或者 "fixed"
        fixed_vmin=0.0,
        fixed_vmax=1.0,
    )

    del dino_model, dino_processor
    del gt_reader, pred_reader
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()