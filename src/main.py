import numpy as np
import os
import time
from tqdm import tqdm
import torch
import imageio.v2 as imageio

from utils.utils import VideoStreamReader
from utils.dino_utils import load_dinov3_model, extract_dinov3_features
from utils.sam3_metric import load_sam3_model, run_grid_prompt_video_tracking_on_tensor
from utils.sam2_utils import build_sam2_generator, run_framewise_sam2_on_tensor
from utils.vis_utils import visualize_video_segments

def main(process_batch_size=16, device='cuda:0'):
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

    # Stage 3/5: Computing DINOv3 features
    tqdm.write(f'[Total Stage 3/5]: Computing DINOv3 features...')
    
    gt_feature_list = []
    pred_feature_list = []
    while True:
        # Step 1/5: Reading videos
        start_time = time.time()
        tqdm.write(f"[DINO 1/3] Reading videos...")
        is_ended, gt_frames = gt_reader.read_batch(process_batch_size * 8)
        _, pred_frames = pred_reader.read_batch(process_batch_size * 8)
        elapsed_time = time.time() - start_time
        tqdm.write(f"[DINO 1/3] Reading videos completed in {elapsed_time:.2f}s")

        if is_ended or gt_frames is None or pred_frames is None:
            break

        # Step 2/5: Extracting DINOv3 features for gt_frames
        start_time = time.time()
        tqdm.write(f"[DINO 2/3] Extracting DINOv3 features for gt_frames...")
        gt_feature_batch = extract_dinov3_features(
            gt_frames,
            model=dino_model,
            processor=dino_processor,
            device=device,
            batch_size=process_batch_size
        ) # [f, 196, 4096]
        gt_feature_list.append(gt_feature_batch)
        elapsed_time = time.time() - start_time
        tqdm.write(f"[DINO 2/3] Extracting DINOv3 features for gt_frames completed in {elapsed_time:.2f}s")

        # Step 3/5: Extracting DINOv3 features for pred_frames
        tqdm.write(f"[DINO 3/3] Extracting DINOv3 features for pred_frames...")
        pred_feature_batch = extract_dinov3_features(
            pred_frames,
            model=dino_model,
            processor=dino_processor,
            device=device,
            batch_size=process_batch_size
        ) # [f, 196, 4096]
        pred_feature_list.append(pred_feature_batch)
        tqdm.write(f"[DINO 3/3] Extracting DINOv3 features for pred_frames completed in {elapsed_time:.2f}s")

        break

    gt_feature_list = torch.cat(gt_feature_list, dim=0)
    pred_feature_list = torch.cat(pred_feature_list, dim=0)

    del dino_model, dino_processor
    del gt_reader, pred_reader
    torch.cuda.empty_cache()

    # Optional alternative: frame-wise SAM automatic mask generation (no temporal tracking)
    sam2_ckpt = '/home/wjp/Documents/GitHub/segment-anything/weights/sam_vit_h_4b8939.pth'
    sam2_generator = build_sam2_generator(checkpoint=sam2_ckpt, model_type='vit_h', device=device)

    # Stage 4/5: Computing SAM masks
    #sam3_model_path = '/home/wjp/Documents/GitHub/sam3/weights'
    #sam3_grid_size = 16
    #sam3_max_objects = 20
    #sam3_model, sam3_processor, sam3_device, sam3_dtype = load_sam3_model(
    #    model_name=sam3_model_path,
    #    device=device,
    #)

    start_time = time.time()
    tqdm.write(f'[Total Stage 4/5]: Computing SAM masks...')

    gt_reader = VideoStreamReader(os.path.join(gt_dir, 'origin.mp4'), start_frame=495)
    pred_reader = VideoStreamReader(os.path.join(pred_dir, 'video.mp4'), start_frame=0)

    while True:
        tqdm.write(f"[SAM 1/3] Reading videos...")
        is_ended, gt_frames = gt_reader.read_batch(process_batch_size * 8)
        _, pred_frames = pred_reader.read_batch(process_batch_size * 8)

        if is_ended or gt_frames is None or pred_frames is None:
            break

        tqdm.write(f'[SAM 2/3] Tracking masks for gt video...')
        #gt_sam3_result = run_grid_prompt_video_tracking_on_tensor(
        #    frames=gt_frames,
        #    model=sam3_model,
        #    processor=sam3_processor,
        #    device=sam3_device,
        #    dtype=sam3_dtype,
        #    grid_size=sam3_grid_size,
        #    max_objects=sam3_max_objects,
        #)
        gt_sam2_result = run_framewise_sam2_on_tensor(
            gt_frames,
            generator=sam2_generator
        )

        tqdm.write(f'[SAM 3/3] Tracking masks for pred video...')

        pred_sam2_result = run_framewise_sam2_on_tensor(
            pred_frames,
            generator=sam2_generator
        )
        break

    del sam3_model, sam3_processor
    del gt_reader, pred_reader
    torch.cuda.empty_cache()

    elapsed_time = time.time() - start_time
    tqdm.write(f'[Total Stage 4/5] completed in {elapsed_time:.2f}s')

    # Stage 5/5: placeholder for metric aggregation
    tqdm.write(f'[Total Stage 5/5]: Ready for downstream mask-based metric computation.')


if __name__ == '__main__':
    main()