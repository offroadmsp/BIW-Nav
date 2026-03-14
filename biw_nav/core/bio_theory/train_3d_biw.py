import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
# 1. 3D 版 Bio-Inspired Encoder
# ==========================================
class BioInspiredEncoder3D(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128):
        super().__init__()
        # LSTM 现在处理 3D 速度向量
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 3) # 预测下一步的 3D 位移
        
    def forward(self, x):
        # x: [Batch, Seq, 3]
        output, (hn, cn) = self.lstm(x)
        pred_next_pos = self.head(output)
        return pred_next_pos, output

# ==========================================
# 2. 3D 数据集加载器
# ==========================================
class RealRatDataset3D(torch.utils.data.Dataset):
    def __init__(self, csv_path, seq_len=50):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到 3D 数据文件: {csv_path}\n请先运行 export_data_3d.m")
            
        print(f"[加载] 读取 3D 轨迹: {csv_path}")
        # 读取 (N, 3) -> X, Y, Z
        self.pos = pd.read_csv(csv_path, header=None).values
        
        # 计算 3D 速度: v_t = p_{t+1} - p_t
        self.velocity = np.diff(self.pos, axis=0)
        self.velocity = np.vstack([self.velocity, [0, 0, 0]]) # 补齐长度
        
        # 归一化速度 (Z轴速度可能比XY小，统一归一化有助于训练)
        self.velocity = self.velocity / (np.std(self.velocity) + 1e-6)
        
        self.seq_len = seq_len
        self.n_samples = len(self.pos) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        v_seq = self.velocity[idx : idx+self.seq_len]
        target_pos = self.pos[idx+1 : idx+self.seq_len+1]
        return torch.FloatTensor(v_seq), torch.FloatTensor(target_pos)

# ==========================================
# 3. 3D Rate Map 计算工具
# ==========================================
def compute_3d_projections(pos, weights, bin_size=5):
    """
    计算 XY, XZ, YZ 三个面的投影 Rate Map
    """
    # 确定全局边界
    min_xyz = np.min(pos, axis=0)
    max_xyz = np.max(pos, axis=0)
    
    # 定义网格边沿
    edges = [
        np.arange(min_xyz[i], max_xyz[i] + bin_size, bin_size) 
        for i in range(3)
    ]
    
    # 计算 3D 直方图 (Occupancy & Activity)
    # weights 是神经元活性
    H_activity, _ = np.histogramdd(pos, bins=edges, weights=weights)
    H_occupancy, _ = np.histogramdd(pos, bins=edges)
    
    # Rate Map = Activity / Occupancy
    # 避免除以0
    H_rate = np.divide(H_activity, H_occupancy, out=np.zeros_like(H_activity), where=H_occupancy!=0)
    
    # 生成投影 (最大值投影或平均值投影)
    # axis=2 (Z) -> XY plane
    proj_xy = np.nanmean(H_rate, axis=2).T 
    # axis=1 (Y) -> XZ plane
    proj_xz = np.nanmean(H_rate, axis=1).T
    # axis=0 (X) -> YZ plane
    proj_yz = np.nanmean(H_rate, axis=0).T
    
    return proj_xy, proj_xz, proj_yz

# ==========================================
# 4. 主训练流程
# ==========================================
def train_3d():
    # 路径配置 (请确认路径正确)
    BASE_DIR = "/media/zhen/Data/cellData/The_place_cell_representation_of_volumetric_space_in_rats/Summarydata"
    POS_FILE = os.path.join(BASE_DIR, "best_session_pos_3d.csv")
    
    # 1. 准备数据
    dataset = RealRatDataset3D(POS_FILE, seq_len=100)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 2. 初始化模型
    model = BioInspiredEncoder3D(input_dim=3, hidden_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(">>> 开始训练 3D 空间模型...")
    for epoch in range(10): # 跑10轮，让模型充分学习
        total_loss = 0
        for v_batch, p_batch in dataloader:
            optimizer.zero_grad()
            # 预测位移变化
            pred, _ = model(v_batch)
            # 简单自监督 Loss: 路径积分误差
            loss = torch.mean((pred - p_batch[:,:,:3])**2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"    Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")

    # ==========================================
    # 5. 结果可视化 (3D Rate Maps)
    # ==========================================
    print(">>> 正在生成 3D Rate Maps (投影视图)...")
    
    # 提取所有时刻的隐层活性
    full_v = torch.FloatTensor(dataset.velocity).unsqueeze(0)
    with torch.no_grad():
        _, hidden = model(full_v)
    hidden = hidden[0].numpy() # [T, 128]
    pos = dataset.pos
    
    # 挑选 3 个代表性神经元展示
    # 为了展示效果，我们挑方差大(活性变化大)的神经元
    neuron_vars = np.var(hidden, axis=0)
    top_indices = np.argsort(neuron_vars)[-3:] # 选最活跃的3个
    
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle("3D Spatial Representation: XY, XZ, YZ Projections", fontsize=16)
    
    for i, neuron_idx in enumerate(top_indices):
        activity = hidden[:, neuron_idx]
        
        # 计算投影
        xy, xz, yz = compute_3d_projections(pos, activity, bin_size=5)
        
        # 绘图：每行一个神经元，三列对应三个面
        # XY Plane
        ax1 = plt.subplot(3, 3, i*3 + 1)
        ax1.imshow(xy, origin='lower', cmap='jet', interpolation='gaussian')
        ax1.set_title(f"Neuron {neuron_idx} - XY (Top View)")
        ax1.axis('off')
        
        # XZ Plane
        ax2 = plt.subplot(3, 3, i*3 + 2)
        ax2.imshow(xz, origin='lower', cmap='jet', interpolation='gaussian')
        ax2.set_title(f"Neuron {neuron_idx} - XZ (Side View)")
        ax2.axis('off')
        
        # YZ Plane
        ax3 = plt.subplot(3, 3, i*3 + 3)
        ax3.imshow(yz, origin='lower', cmap='jet', interpolation='gaussian')
        ax3.set_title(f"Neuron {neuron_idx} - YZ (Front View)")
        ax3.axis('off')
        
    plt.tight_layout()

    save_path = './plot/real_biw.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figures saved to: {save_path}")
    # plt.show()
    print(">>> 完成！请观察三个视角的投影。如果 XZ/YZ 出现类似条纹，说明学到了垂直结构。")

if __name__ == "__main__":
    train_3d()