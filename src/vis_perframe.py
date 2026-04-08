from utils.vis_utils import (
    load_metrics_json,
    load_aligned_video_frames_from_metrics,
    load_soft_mat_npy_from_metrics,
    MetricsFrameViewer,
)



def main():
    metrics = load_metrics_json(output_dir="output")
    gt_frames, pred_frames = load_aligned_video_frames_from_metrics(metrics)
    soft_mats_npy = load_soft_mat_npy_from_metrics(metrics)
    viewer = MetricsFrameViewer(gt_frames, pred_frames, metrics, soft_mats_npy=soft_mats_npy)
    viewer.show()


if __name__ == '__main__':
    main()