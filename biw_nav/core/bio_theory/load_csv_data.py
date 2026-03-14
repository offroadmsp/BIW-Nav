import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ==========================================
# 配置
# ==========================================
class Config:
    DATA_ROOT = "/media/zhen/Data/cellData/The_place_cell_representation_of_volumetric_space_in_rats/Summarydata"
    POS_FILE = "best_session_pos.csv"
    SPK_FILE = "best_session_spk.csv"
    
    BIN_SIZE = 2.0 # cm
    SMOOTH_SIGMA = 1.5
    MIN_DWELL = 0.1 # seconds

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

# ==========================================
# 加载与分析
# ==========================================
def load_and_analyze():
    pos_path = os.path.join(Config.DATA_ROOT, Config.POS_FILE)
    spk_path = os.path.join(Config.DATA_ROOT, Config.SPK_FILE)
    
    if not os.path.exists(pos_path):
        print(f"[错误] 找不到文件: {pos_path}\n请先运行 MATLAB 转换脚本！")
        return

    print("[加载] 读取 CSV 数据...")
    # pos: (N, 2) 轨迹坐标
    pos = pd.read_csv(pos_path, header=None).values
    # spk: (M, 2) 脉冲发生时的坐标 (注意：这里不再是时间序列，而是坐标点)
    spk_coords = pd.read_csv(spk_path, header=None).values
    
    print(f"  - 轨迹点数: {len(pos)}")
    print(f"  - 脉冲数量: {len(spk_coords)}")

    # 计算 Rate Map
    print("[分析] 计算 Rate Map (基于坐标点)...")
    ratemap = compute_rate_map_from_coords(pos, spk_coords)
    
    # 绘图
    plot_results(pos, spk_coords, ratemap)

def compute_rate_map_from_coords(pos, spk_coords, bin_size=Config.BIN_SIZE, sigma=Config.SMOOTH_SIGMA):
    """
    基于坐标点计算 Rate Map (复现 mapDATA 逻辑)
    输入:
        pos: (N, 2) 所有轨迹点坐标
        spk_coords: (M, 2) 脉冲发生位置的坐标
    """
    # 1. 确定地图边界
    # 合并 pos 和 spk 以确保边界覆盖所有数据
    all_points = np.vstack([pos, spk_coords])
    min_x, max_x = np.min(all_points[:,0]), np.max(all_points[:,0])
    min_y, max_y = np.min(all_points[:,1]), np.max(all_points[:,1])
    
    # 加上一点 padding
    padding = 2.0
    x_edges = np.arange(min_x - padding, max_x + padding + bin_size, bin_size)
    y_edges = np.arange(min_y - padding, max_y + padding + bin_size, bin_size)
    
    # 2. 计算 Dwell Map (停留时间图)
    # 统计轨迹点落入每个格子的次数
    occupancy, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=[x_edges, y_edges])
    # 转换为时间: 次数 * 采样间隔 (假设 50Hz -> 0.02s)
    dwell_time = occupancy * 0.02
    
    # 3. 计算 Spike Map (脉冲分布图)
    # 统计脉冲点落入每个格子的次数
    spike_counts, _, _ = np.histogram2d(spk_coords[:,0], spk_coords[:,1], bins=[x_edges, y_edges])
    
    # 4. 平滑 (Gaussian Smoothing)
    # 注意: 对 dwell 和 spike map 分别平滑，然后再相除
    smooth_dwell = gaussian_filter(dwell_time, sigma, mode='constant')
    smooth_spikes = gaussian_filter(spike_counts, sigma, mode='constant')
    
    # 5. 计算 Rate (Hz)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratemap = smooth_spikes / smooth_dwell
    
    # 过滤掉未访问区域 (NaN)
    ratemap[smooth_dwell < Config.MIN_DWELL] = np.nan
    
    return ratemap

def plot_results(pos, spk_coords, ratemap):
    plt.figure(figsize=(12, 5))
    
    # 轨迹与脉冲叠加图
    plt.subplot(1, 2, 1)
    plt.plot(pos[:,0], pos[:,1], 'k', alpha=0.1, label='Trajectory')
    plt.scatter(spk_coords[:,0], spk_coords[:,1], c='r', s=5, alpha=0.6, label='Spikes')
    plt.title("Raw Trajectory & Spikes")
    plt.legend()
    plt.axis('equal')
    plt.xlabel("X (cm)")
    plt.ylabel("Y (cm)")
    
    # Rate Map 热力图
    plt.subplot(1, 2, 2)
    # Transpose + Origin Lower 以匹配坐标系
    plt.imshow(ratemap.T, origin='lower', cmap='jet', interpolation='nearest')
    plt.colorbar(label='Firing Rate (Hz)')
    plt.title("Rate Map")
    plt.axis('off')
    
    plt.tight_layout()
    save_path = './plot/real_3d_data.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figures saved to: {save_path}")
    # plt.show()

if __name__ == "__main__":
    load_and_analyze()