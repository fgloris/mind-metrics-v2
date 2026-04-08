import torch
import math

def make_patch_positions(num_patches, device=None, dtype=torch.float32):
    side = int(math.sqrt(num_patches))
    assert side * side == num_patches, f"num_patches={num_patches} is not a square number"

    ys = torch.linspace(0.0, 1.0, steps=side, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, steps=side, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=-1)
    return pos


def compute_patch_sim(
    gt_feature_list,
    pred_feature_list,
    sigma=1.0,
    tau=0.07,
    lambda_entropy=1.0,
):
    """
    gt_feature_list, pred_feature_list: [B, N, C]

    return:
        soft_score_list, hard_score_list, soft_mats_list
    """
    assert gt_feature_list.shape == pred_feature_list.shape
    B, N, C = gt_feature_list.shape

    soft_mats_list = []
    soft_score_list = []
    hard_score_list = []

    # positions: [N, 2]
    gt_pos = make_patch_positions(N, device=gt_feature_list.device, dtype=gt_feature_list.dtype)
    pred_pos = make_patch_positions(N, device=pred_feature_list.device, dtype=pred_feature_list.dtype)

    pos_dist = torch.cdist(gt_pos, pred_pos, p=2)
    dist_sim = torch.exp(-(pos_dist ** 2) / (2 * sigma * sigma))

    eps = 1e-8
    logN = math.log(N)

    for i in range(B):
        gt_feats = gt_feature_list[i]      # [N, C]
        pred_feats = pred_feature_list[i]  # [N, C]

        gt_feats = torch.nn.functional.normalize(gt_feats, p=2, dim=-1)
        pred_feats = torch.nn.functional.normalize(pred_feats, p=2, dim=-1)

        # similarity matrix
        feat_sim = gt_feats @ pred_feats.T   # [N, N]

        # 你现在先不用位置项，就保持这样
        joint_soft_sim = feat_sim  # * dist_sim

        soft_mats_list.append(joint_soft_sim)

        # ----------------------------
        # hard score: row-wise top1 mean
        # ----------------------------
        best_vals, best_idx = torch.max(joint_soft_sim, dim=1)   # [N]
        hard_score = best_vals.mean()
        hard_score_list.append(hard_score)

        # ----------------------------
        # entropy penalty
        # ----------------------------
        prob = torch.softmax(joint_soft_sim / tau, dim=1)   # [N, N]

        entropy = -(prob * torch.log(prob + eps)).sum(dim=1)   # [N]
        normalized_entropy = entropy / (logN + eps)            # [N], about [0,1]

        # 越接近 one-hot，confidence 越大
        confidence = 1.0 - normalized_entropy                  # [N]

        # 也可以直接当惩罚项：
        # penalty = normalized_entropy

        # 最终 soft score：高相似 + 低熵 才高
        weighted_score = confidence * best_vals
        soft_score = weighted_score.sum() / (confidence.sum() + eps)

        # 如果你想显式减惩罚项，也可以用这个版本：
        # soft_score = best_vals.mean() - lambda_entropy * normalized_entropy.mean()

        soft_score_list.append(soft_score)

    return soft_score_list, hard_score_list, soft_mats_list