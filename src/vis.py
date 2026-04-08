import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def _to_display_image(img):
    """
    img: torch.Tensor [C,H,W] in [0,1]  or numpy [H,W,C]
    return: numpy [H,W,C] in [0,1]
    """
    if hasattr(img, "detach"):
        img = img.detach().cpu().float().numpy()

    if img.ndim == 3 and img.shape[0] in (1, 3):  # [C,H,W] -> [H,W,C]
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    img = np.clip(img, 0.0, 1.0)
    return img


def _infer_patch_grid(num_patches):
    side = int(math.sqrt(num_patches))
    assert side * side == num_patches, f"num_patches={num_patches} is not square"
    return side, side


def _patch_index_from_xy(x, y, img_w, img_h, grid_w, grid_h):
    """
    Map mouse position on image to patch index.
    """
    if x is None or y is None:
        return None

    if x < 0 or x >= img_w or y < 0 or y >= img_h:
        return None

    col = min(int(x / img_w * grid_w), grid_w - 1)
    row = min(int(y / img_h * grid_h), grid_h - 1)
    idx = row * grid_w + col
    return idx, row, col


def _patch_rect(row, col, img_w, img_h, grid_w, grid_h):
    patch_w = img_w / grid_w
    patch_h = img_h / grid_h
    x0 = col * patch_w
    y0 = row * patch_h
    return x0, y0, patch_w, patch_h


def _normalize_map(sim_map, mode="minmax", vmin=None, vmax=None):
    """
    sim_map: [gh, gw]
    return normalized heatmap in [0,1]
    """
    sim_map = sim_map.astype(np.float32)

    if mode == "fixed":
        assert vmin is not None and vmax is not None
        denom = max(vmax - vmin, 1e-8)
        out = (sim_map - vmin) / denom
    else:
        smin = sim_map.min()
        smax = sim_map.max()
        denom = max(smax - smin, 1e-8)
        out = (sim_map - smin) / denom

    return np.clip(out, 0.0, 1.0)

def visualize_patch_similarity_interactive(
    img_a,
    img_b,
    joint_sim,
    title="Patch Similarity Viewer",
    heat_alpha=0.45,
    norm_mode="minmax",   # "minmax" or "fixed"
    fixed_vmin=0.0,
    fixed_vmax=1.0,
):
    """
    img_a, img_b:
        torch.Tensor [C,H,W] or numpy [H,W,C]

    joint_sim:
        torch.Tensor or numpy, shape [N, N]
        row i = similarity from patch i in A to all patches in B

    功能：
    - 鼠标移动到 A 图某个 patch
    - B 图按 joint_sim[idx] reshape 后显示热力图
    """

    img_a = _to_display_image(img_a)
    img_b = _to_display_image(img_b)

    if hasattr(joint_sim, "detach"):
        joint_sim = joint_sim.detach().cpu().float().numpy()

    assert joint_sim.ndim == 2 and joint_sim.shape[0] == joint_sim.shape[1], \
        f"joint_sim shape should be [N,N], got {joint_sim.shape}"

    num_patches = joint_sim.shape[0]
    grid_h, grid_w = _infer_patch_grid(num_patches)

    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(title)

    ax_a.imshow(img_a)
    ax_a.set_title("Image A")
    ax_a.axis("off")

    ax_b.imshow(img_b)
    ax_b.set_title("Image B")
    ax_b.axis("off")

    # 左图当前 patch 框
    rect_a = Rectangle((0, 0), 10, 10, fill=False, edgecolor="cyan", linewidth=2)
    ax_a.add_patch(rect_a)
    rect_a.set_visible(False)

    # 右图 best-match 框
    rect_b = Rectangle((0, 0), 10, 10, fill=False, edgecolor="lime", linewidth=2)
    ax_b.add_patch(rect_b)
    rect_b.set_visible(False)

    # 右图热力图 overlay
    heat_init = np.zeros((grid_h, grid_w), dtype=np.float32)
    heat_artist = ax_b.imshow(
        heat_init,
        cmap="jet",
        alpha=0.0,
        interpolation="nearest",
        extent=(0, w_b, h_b, 0),  # align with image coordinates
        vmin=0.0,
        vmax=1.0,
    )

    # 文本信息
    text_artist = ax_b.text(
        0.02, 0.02, "",
        transform=ax_b.transAxes,
        color="white",
        fontsize=10,
        bbox=dict(facecolor="black", alpha=0.6, pad=4)
    )

    def on_move(event):
        if event.inaxes != ax_a:
            return

        parsed = _patch_index_from_xy(
            event.xdata, event.ydata,
            img_w=w_a, img_h=h_a,
            grid_w=grid_w, grid_h=grid_h
        )
        if parsed is None:
            return

        idx, row, col = parsed

        # 左图框
        x0, y0, pw, ph = _patch_rect(row, col, w_a, h_a, grid_w, grid_h)
        rect_a.set_xy((x0, y0))
        rect_a.set_width(pw)
        rect_a.set_height(ph)
        rect_a.set_visible(True)

        # 取 A 中当前 patch 对 B 所有 patch 的相似度
        sim_vec = joint_sim[idx]                  # [N]
        sim_map = sim_vec.reshape(grid_h, grid_w)

        sim_map_norm = sim_map#_normalize_map(
        #    sim_map,
        #    mode=norm_mode,
        #    vmin=fixed_vmin,
        #    vmax=fixed_vmax,
        #)

        heat_artist.set_data(sim_map_norm)
        heat_artist.set_alpha(heat_alpha)

        # 找 B 中最相似 patch
        best_idx = idx # int(np.argmax(sim_vec))
        best_row = best_idx // grid_w
        best_col = best_idx % grid_w

        bx0, by0, bpw, bph = _patch_rect(best_row, best_col, w_b, h_b, grid_w, grid_h)
        rect_b.set_xy((bx0, by0))
        rect_b.set_width(bpw)
        rect_b.set_height(bph)
        rect_b.set_visible(True)

        text_artist.set_text(
            f"A patch idx={idx} (row={row}, col={col})\n"
            f"B best idx={best_idx} (row={best_row}, col={best_col})\n"
            f"sim max={sim_vec[best_idx]:.4f}"
        )

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    plt.tight_layout()
    plt.show()