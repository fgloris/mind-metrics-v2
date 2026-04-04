import numpy as np

def make_grid_points(width, height, grid_size=8, margin_ratio=0.08):
    """
    在首帧生成规则网格点，返回 [(x, y), ...]
    grid_size=8 表示 8x8 网格
    """
    xs = np.linspace(int(width * margin_ratio), int(width * (1 - margin_ratio)), grid_size)
    ys = np.linspace(int(height * margin_ratio), int(height * (1 - margin_ratio)), grid_size)
    points = []
    for y in ys:
        for x in xs:
            points.append((int(round(x)), int(round(y))))
    return points


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def filter_masks(
    masks,
    scores=None,
    min_area_ratio=0.002,
    max_area_ratio=0.5,
    max_border_touch_ratio=0.5,
    iou_dedup_thresh=0.8,
    top_k=20,
):
    """
    对首帧候选 mask 做过滤和去重
    masks: list[np.ndarray]   shape [H, W], bool/0-1
    scores: list[float] or None
    """
    if len(masks) == 0:
        return []
    
    b, h, w = masks[0].shape
    total_area = h * w

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    items = []
    for i, m in enumerate(masks):
        m = m.astype(bool)
        area = m.sum()
        area_ratio = area / total_area

        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue

        border_touch = np.logical_and(m, border).sum()
        border_touch_ratio = border_touch / max(area, 1)
        if border_touch_ratio > max_border_touch_ratio:
            continue

        score = 0.0 if scores is None else float(scores[i])
        items.append({"mask": m, "score": score, "area": int(area)})

    # 优先按 score，再按面积
    items.sort(key=lambda x: (x["score"], x["area"]), reverse=True)

    kept = []
    for item in items:
        duplicated = False
        for k in kept:
            if mask_iou(item["mask"], k["mask"]) > iou_dedup_thresh:
                duplicated = True
                break
        if not duplicated:
            kept.append(item)
        if len(kept) >= top_k:
            break

    return kept


def mask_center_point(mask: np.ndarray):
    """
    用 mask 内像素均值近似中心点；更稳可以改成 distance transform 取最内点
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x = int(round(xs.mean()))
    y = int(round(ys.mean()))
    return [x, y]