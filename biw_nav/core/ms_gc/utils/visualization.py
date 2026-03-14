import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import torch


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight"
})


# 如果您安装了 seaborn，可以使用它的调色板，没有也没关系
try:
    import seaborn as sns
except ImportError:
    sns = None

def get_nmi_cmap(style="neuro_classic"):
    style = style.lower() # 忽略大小写

    # --- 1. Matplotlib 标准感知均匀色系 (推荐) ---
    if style == "viridis":
        # 经典：蓝 -> 绿 -> 黄 (最稳健，默认)
        return plt.get_cmap("viridis")
    
    elif style == "plasma":
        # 活力：紫 -> 红 -> 黄 (对比度更高，视觉更暖)
        return plt.get_cmap("plasma")
    
    elif style == "magma":
        # 深邃：黑 -> 红 -> 白 (适合白底，像岩浆)
        return plt.get_cmap("magma")
    
    elif style == "inferno":
        # 强烈：黑 -> 火红 -> 黄 (极高对比度)
        return plt.get_cmap("inferno")

    # --- 2. 自定义神经科学风格 ---
    elif style == "neuro_classic":
        # 白 -> 黄 -> 红 -> 深红 (背景纯白，突出结构)
        colors = ["#ffffff", "#ffcc00", "#ff6600", "#990000"]
        nodes = [0.0, 0.3, 0.7, 1.0]
        return mcolors.LinearSegmentedColormap.from_list("neuro_heat", list(zip(nodes, colors)))
    
    # --- 3. 兜底 ---
    else:
        try:
            return plt.get_cmap(style)
        except ValueError:
            print(f"Warning: Colormap '{style}' not found. Using 'viridis' instead.")
            return plt.get_cmap("viridis")


def plot_position_decoding(
    positions,
    decoded_positions,
    grid_scales,
    save_path=None
):
    """
    Position decoding visualization in 1×3 layout (NMI-style)
    """

    num_scales = len(decoded_positions)
    assert num_scales == 3, "1×3 layout assumes exactly 3 grid scales"

    fig, axs = plt.subplots(
        1, 3,
        figsize=(9.5, 3.2),
        sharex=True,
        sharey=True
    )

    for i, ax in enumerate(axs):
        decoded = decoded_positions[i].detach().cpu().numpy()

        # Ground truth (background)
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=8,
            c="black",
            alpha=0.25,
            label="Ground truth" if i == 0 else None
        )

        # Decoded positions
        ax.scatter(
            decoded[:, 0],
            decoded[:, 1],
            s=10,
            c="#1f77b4",
            alpha=0.7,
            label="Decoded" if i == 0 else None
        )

        ax.set_title(f"Scale = {grid_scales[i]}", pad=4)
        ax.set_aspect("equal", adjustable="box")

        if i == 0:
            ax.set_ylabel("Y position")
        ax.set_xlabel("X position")

    # Shared legend (only once)
    axs[0].legend(
        frameon=False,
        loc="upper right",
        handletextpad=0.3
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    plt.show()



def plot_position_decoding_with_error_1x3(
    positions,
    decoded_positions,
    grid_scales,
    error_stride=5,
    save_path=None
):
    """
    NMI Figure-3 style position decoding visualization
    with decoding error vectors (1×3 layout).
    """

    num_scales = len(decoded_positions)
    assert num_scales == 3, "Figure 3 layout assumes exactly 3 scales"

    fig, axs = plt.subplots(
        1, 3,
        figsize=(9.5, 3.4),
        sharex=True,
        sharey=True
    )

    for i, ax in enumerate(axs):
        decoded = decoded_positions[i].detach().cpu().numpy()

        # ---- Ground truth ----
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=8,
            c="black",
            alpha=0.22,
            label="Ground truth" if i == 0 else None,
            zorder=1
        )

        # ---- Decoded positions ----
        ax.scatter(
            decoded[:, 0],
            decoded[:, 1],
            s=10,
            c="#1f77b4",
            alpha=0.75,
            label="Decoded" if i == 0 else None,
            zorder=2
        )

        # ---- Decoding error vectors (downsampled) ----
        idx = slice(0, len(positions), error_stride)

        ax.quiver(
        positions[idx, 0],
        positions[idx, 1],
        decoded[idx, 0] - positions[idx, 0],
        decoded[idx, 1] - positions[idx, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="#d62728",
        alpha=0.65,
        width=0.004,        # ← 关键：线条明显变粗
        headwidth=4.5,      # ← 箭头头部同步加粗
        headlength=6,
        zorder=3,
        label="Decoding error" if i == 0 else None
        )

        ax.set_title(f"Scale = {grid_scales[i]}", pad=4)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("X position")
        if i == 0:
            ax.set_ylabel("Y position")

        # 子图标注 (a)(b)(c)
        ax.text(
            0.02, 0.95,
            f"({chr(97+i)})",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top"
        )

    # ---- Shared legend ----
    axs[0].legend(
        frameon=False,
        loc="upper right",
        handlelength=1.2,
        handletextpad=0.4
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)

    plt.show()



def plot_loss_curve(
    train_losses,
    val_losses,
    save_path=None
):
    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(
        epochs,
        train_losses,
        label="Training",
        color="#1f77b4"
    )
    ax.plot(
        epochs,
        val_losses,
        label="Validation",
        color="#ff7f0e"
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)

    if save_path is not None:
        fig.savefig(save_path)

    # plt.show()


def visualize_grid_activities(
    positions,
    grid_activities,
    grid_scales,
    save_path=None,
    color_style="neuro_classic"  # <--- 新增参数：选择配色
):
    """
    NMI Figure-3 style grid cell activity visualization (1×3 layout)
    """

    # 获取选定的 Colormap
    my_cmap = get_nmi_cmap(color_style)

    # 数据处理
    num_scales = len(grid_activities)
    processed_activities = []
    for g in grid_activities:
        if isinstance(g, torch.Tensor):
            processed_activities.append(g.detach().cpu().numpy())
        else:
            processed_activities.append(g)
            
    all_activity = np.concatenate([g.reshape(-1) for g in processed_activities])
    vmin, vmax = all_activity.min(), all_activity.max()

    # 如果使用 "neuro_classic" (白色背景)，建议稍微调高一点 vmin
    # 这样接近 0 的噪音就会变成纯白，画面更干净
    if color_style == "neuro_classic":
        vmin = vmin + (vmax - vmin) * 0.05 

    fig, axs = plt.subplots(
        1, 3,
        figsize=(9.5, 2.5),
        sharey=True,
        layout='constrained'
    )

    im = None
    for i, ax in enumerate(axs):
        activity = processed_activities[i]

        im = ax.imshow(
            activity.T,
            cmap=my_cmap,  # <--- 使用新配色
            aspect="auto",
            # interpolation="bicubic", # 如果想要更平滑的效果，可以用 bicubic
            interpolation="nearest",   # 科学严谨通常用 nearest，看像素真实值
            vmin=vmin,
            vmax=vmax
        )

        ax.set_title(f"Scale = {grid_scales[i]:.2f}", pad=4, fontweight='medium')
        ax.set_xlabel("Position index")
        
        # 美化坐标轴：去掉上方和右方的边框 (Despine)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # 加粗一点左下边框
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        if i == 0:
            ax.set_ylabel("Cell index")
        
        ax.text(
            -0.1, 1.05, f"({chr(100+i)})",
            transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right"
        )

    # Colorbar
    cbar = fig.colorbar(im, ax=axs, orientation='vertical', aspect=30, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=7, width=0.5)
    cbar.outline.set_linewidth(0.5) # Colorbar 边框变细
    cbar.set_label("Activity", fontsize=8)
    
    if save_path is not None:
        fig.savefig(save_path)
    
    plt.show()
