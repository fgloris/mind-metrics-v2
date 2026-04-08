import torch
import math

def make_patch_positions(num_patches, device=None, dtype=torch.float32):
    """
    num_patches: e.g. 1764 -> 42x42, 3136 -> 56x56
    return: [num_patches, 2], each row is (y, x) in [0, 1]
    """
    side = int(math.sqrt(num_patches))
    assert side * side == num_patches, f"num_patches={num_patches} is not a square number"

    ys = torch.linspace(0.0, 1.0, steps=side, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, steps=side, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)  # [N, 2]
    return pos


def compute_patch_sim(
    gt_feature_list,
    pred_feature_list,
    sigma=1.0,         # position gaussian bandwidth
):
    """
    gt_feature_list, pred_feature_list:
        each is [B, N, C], features should ideally already be L2-normalized

    return: list of joint similarity matrices
    """
    assert gt_feature_list.shape == pred_feature_list.shape
    B, N, C = gt_feature_list.shape

    soft_mats_list = []
    soft_score_list = []
    hard_score_list = []

    # positions: [N, 2]
    gt_pos = make_patch_positions(N, device=gt_feature_list.device, dtype=gt_feature_list.dtype)
    pred_pos = make_patch_positions(N, device=pred_feature_list.device, dtype=pred_feature_list.dtype)

    # pairwise position distance: [N, N]
    pos_dist = torch.cdist(gt_pos, pred_pos, p=2)

    # convert distance to similarity with Gaussian kernel
    dist_sim = torch.exp(-(pos_dist ** 2) / (2 * sigma * sigma))  # [N, N], in (0,1]

    for i in range(B):
        gt_feats = gt_feature_list[i]      # [N, C]
        pred_feats = pred_feature_list[i]  # [N, C]

        # 1) normalize features just in case
        gt_feats = torch.nn.functional.normalize(gt_feats, p=2, dim=-1)
        pred_feats = torch.nn.functional.normalize(pred_feats, p=2, dim=-1)

        # 2) feature similarity: [N, N]
        feat_sim = gt_feats @ pred_feats.T   # cosine sim if normalized

        # 6) joint similarity
        joint_soft_sim = feat_sim * dist_sim

        # 7) gt self-similarity
        self_sim = (gt_feats @ gt_feats.T) * dist_sim

        soft_score = joint_soft_sim.sum() / self_sim.sum()

        #soft_mats_list.append(joint_soft_sim)
        soft_score_list.append(soft_score)

        best_idx = torch.argmax(joint_soft_sim, dim=1, keepdim=True)
        hard_mask = torch.zeros_like(joint_soft_sim)
        hard_mask.scatter_(1, best_idx, 1.0)
        hard_score = (hard_mask * joint_soft_sim).sum()

        hard_score_list.append(hard_score)

    return soft_score_list, hard_score_list#, soft_mats_list