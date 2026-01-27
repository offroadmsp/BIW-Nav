import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter
import os

# ==============================
# 0. NMI 风格配置
# ==============================
def set_nmi_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.2
    plt.rcParams['ytick.major.width'] = 1.2
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['image.cmap'] = 'inferno' 

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==============================
# 1. 数据处理 (保持不变)
# ==============================
def load_and_process_data():
    # 路径硬编码
    data_dir = '/media/zhen/Data/cellData/The_place_cell_representation_of_volumetric_space_in_rats/Summarydata'
    pos_path = os.path.join(data_dir, 'best_session_pos_3d.csv')
    spk_path = os.path.join(data_dir, 'best_session_spk_3d.csv')
    
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"找不到数据文件: {pos_path}")

    pos = pd.read_csv(pos_path, header=None).values 
    spk_coords = pd.read_csv(spk_path, header=None).values 
    
    # 假设 50Hz
    vel_vec = np.diff(pos, axis=0) * 50 
    speed = np.linalg.norm(vel_vec, axis=1)
    speed = gaussian_filter(speed, sigma=2) 
    pos = pos[1:] 
    
    return speed, pos, spk_coords

def simulate_lif_multiscale(input_current, alphas=[0.1, 0.9]):
    traces = []
    I = (input_current - np.mean(input_current)) / (np.std(input_current) + 1e-6)
    I = I * 2.0 + 0.5
    for alpha in alphas:
        v = np.zeros_like(I)
        v_curr = 0
        for t in range(len(I)):
            v_curr = (1 - alpha) * v_curr + alpha * I[t]
            v[t] = v_curr
        traces.append(v)
    return traces

def compute_high_res_ratemap(pos, spk_coords, axis_idx, bin_size=3.0, sigma=1.5):
    keep_axes = [i for i in [0,1,2] if i != axis_idx]
    pos_2d = pos[:, keep_axes]
    spk_2d = spk_coords[:, keep_axes]
    
    min_xy = np.min(pos_2d, axis=0)
    max_xy = np.max(pos_2d, axis=0)
    # 稍微扩大一点边界，防止切边
    padding = 2.0
    x_edges = np.arange(min_xy[0]-padding, max_xy[0]+padding, bin_size)
    y_edges = np.arange(min_xy[1]-padding, max_xy[1]+padding, bin_size)
    
    H_occ, _, _ = np.histogram2d(pos_2d[:,0], pos_2d[:,1], bins=[x_edges, y_edges])
    H_occ = H_occ * 0.02 
    H_spk, _, _ = np.histogram2d(spk_2d[:,0], spk_2d[:,1], bins=[x_edges, y_edges])
    
    H_occ_smooth = gaussian_filter(H_occ, sigma=sigma)
    H_spk_smooth = gaussian_filter(H_spk, sigma=sigma)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_map = H_spk_smooth / H_occ_smooth
    
    rate_map[H_occ_smooth < 0.05] = np.nan
    return rate_map.T, x_edges, y_edges

# ==============================
# 2. 绘图逻辑 (GridSpec 深度优化版)
# ==============================
def plot_nmi_final_v5():
    set_nmi_style()
    ensure_dir('./plot')
    
    try:
        speed, pos, spk_coords = load_and_process_data()
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        return
    
    alphas = [0.2, 0.01] 
    traces = simulate_lif_multiscale(speed, alphas)
    
    # 画布设置：宽高比更协调，防止压扁
    fig = plt.figure(figsize=(16, 14), layout='constrained')
    
    # 主 GridSpec：3行
    # height_ratios: 给最下面的 Panel C 留出最大空间
    gs_main = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[0.7, 0.9, 1.4])
    
    # ==========================
    # Panel A: 空间尺度 (Lambda)
    # ==========================
    ax1 = fig.add_subplot(gs_main[0])
    x = np.linspace(-10, 10, 500)
    lambdas = [0.5, 1.5, 3.0]
    colors = ['#E91E63', '#9C27B0', '#3F51B5']
    
    for i, lam in enumerate(lambdas):
        y = np.exp(-x**2 / (2*lam**2))
        ax1.plot(x, y, color=colors[i], lw=3, label=f'Scale $\lambda_{i+1} = {lam}$', alpha=0.8)
        ax1.fill_between(x, y, color=colors[i], alpha=0.1)
    
    ax1.set_title(r"Multi-Scale Spatial Encoding ($\lambda$)", loc='left', pad=10)
    ax1.set_xlabel("Spatial Latent Dimension ($x$)")
    ax1.set_ylabel("Activation")
    ax1.set_yticks([])
    ax1.set_xlim(-8, 8)
    ax1.legend(frameon=False, loc='upper right', ncol=3)
    
    # ==========================
    # Panel B: 时间尺度 (Alpha)
    # ==========================
    ax2 = fig.add_subplot(gs_main[1])
    start_idx, end_idx = 1000, 2500 
    t_axis = np.arange(end_idx - start_idx) * 0.02
    norm_speed = (speed[start_idx:end_idx] - speed.min()) / (speed.max() - speed.min())
    
    ax2.fill_between(t_axis, norm_speed, color='gray', alpha=0.15, label='Real Input Speed')
    ax2.plot(t_axis, traces[0][start_idx:end_idx] + 2.5, color=colors[0], lw=2, 
             label=r'Fast / Local ($\alpha_{large}=' + str(alphas[0]) + r'$)')
    ax2.plot(t_axis, traces[1][start_idx:end_idx], color=colors[2], lw=2, 
             label=r'Slow / Global ($\alpha_{small}=' + str(alphas[1]) + r'$)')
    
    ax2.set_title(r"Temporal Integration Scales ($\alpha$)", loc='left', pad=10)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Neural Activity")
    ax2.set_yticks([])
    ax2.set_xlim(0, t_axis[-1])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False, ncol=3)

    # ==========================
    # Panel C: 3D 空间表征 (Nested GridSpec 修复版)
    # ==========================
    # 创建子网格：1行4列
    # width_ratios: [图1, 图2, 图3, Colorbar]
    # 0.05 的 colorbar 宽度既够用又不会挤占主图
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_main[2], 
                                            width_ratios=[1, 1, 1, 0.05], 
                                            wspace=0.3)
    
    proj_titles = ['YZ Projection (Front)', 'XZ Projection (Side)', 'XY Projection (Top)']
    proj_axes = [0, 1, 2] 
    
    # 预计算范围
    temp_map, _, _ = compute_high_res_ratemap(pos, spk_coords, 2)
    vmax = np.nanpercentile(temp_map, 99)
    last_im = None

    for i, ax_idx in enumerate(proj_axes):
        ax = fig.add_subplot(gs_c[i])
        
        rmap, xe, ye = compute_high_res_ratemap(pos, spk_coords, axis_idx=ax_idx, bin_size=4.0, sigma=1.2)
        
        # --- 关键修改：Aspect Ratio ---
        # 对于 XY (i=2), 它是俯视图，物理上是正方形，必须用 'equal'
        # 对于 YZ/XZ (i=0,1), 它是侧视图，高度通常远小于宽度，用 'auto' 拉伸视觉效果更好，
        # 否则会变成细长条，看不清结构。
        aspect_mode = 'equal' if i == 2 else 'auto'
        
        last_im = ax.imshow(rmap, origin='lower', cmap='inferno', 
                       extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       vmin=0, vmax=vmax, interpolation='gaussian',
                       aspect=aspect_mode)
        
        ax.set_title(proj_titles[i], fontweight='bold', fontsize=14, pad=10)
        ax.set_xlabel("Position (cm)")
        if i == 0:
            ax.set_ylabel("Position (cm)")
        
        # 比例尺 (仅在 Top View 添加，因为比例最标准)
        if i == 2:
            scale_len = 20
            # 找一个合适的位置 (左下角)
            start_x = xe[0] + 10
            start_y = ye[0] + 10
            ax.plot([start_x, start_x+scale_len], [start_y, start_y], color='white', lw=4)
            ax.text(start_x+scale_len/2, start_y+3, '20 cm', color='white', ha='center', fontsize=12, fontweight='bold')

    # --- Colorbar (独立列) ---
    ax_cbar = fig.add_subplot(gs_c[3])
    cbar = plt.colorbar(last_im, cax=ax_cbar)
    cbar.set_label('Firing Rate (Hz)', rotation=270, labelpad=20)
    
    # 总标题位置调整
    fig.text(0.01, 0.2, r"Volumetric Rate Maps (Real Data)", 
             fontsize=16, va='center', rotation=90)

    # 保存
    save_path = './plot/nmi_final_v5_restored.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figures saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_nmi_final_v5()