import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# [修复1] 导入 collate_variable_length
from dataset import VariableLengthTrajectoryDataset, collate_variable_length
from cann_base import cann_model
from scipy.ndimage import gaussian_filter1d

# ==============================
# 0. 配置与 NMI 风格
# ==============================
class Config:
    DATA_ROOT = '/media/zhen/Data/Datasets/nomad_data/go_stanford'
    SAVE_DIR = './plot/ablation'
    BATCH_SIZE = 1
    MIN_LEN = 20
    N_NEURONS = 100
    DT = 0.01  # 仿真步长

def set_nmi_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['legend.frameon'] = False

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==============================
# 1. 核心模型类 (支持多尺度)
# ==============================
class MockArgs:
    def __init__(self, k=0.5, a=0.5, N=100, A=1.0, tau=1.0, z0=0.0):
        self.k = k
        self.a = a
        self.N = N
        self.A = A
        self.z0 = z0
        self.tau = tau

class MultiScaleCANN:
    def __init__(self, spatial_scales, temporal_scales):
        """
        spatial_scales (list): [0, 1] 之间的 lambda
        temporal_scales (list): [0, 1] 之间的 alpha
        """
        self.layers_x = []
        self.layers_y = []
        self.scales = []
        
        # 尺度映射参数
        self.a_min, self.a_max = 0.1, 1.5   # 空间宽度范围 (rad/m)
        self.tau_min, self.tau_max = 1.0, 50.0 # 时间常数范围 (steps)

        for s_lambda, t_alpha in zip(spatial_scales, temporal_scales):
            # 映射 Lambda -> a (Width)
            mapped_a = self.a_min + s_lambda * (self.a_max - self.a_min)
            # 映射 Alpha -> Tau (Time Constant)
            mapped_tau = self.tau_max - t_alpha * (self.tau_max - self.tau_min)
            
            args_x = MockArgs(a=mapped_a, tau=mapped_tau, N=Config.N_NEURONS)
            args_y = MockArgs(a=mapped_a, tau=mapped_tau, N=Config.N_NEURONS)
            
            self.layers_x.append(cann_model(args_x))
            self.layers_y.append(cann_model(args_y))

    def reset(self, start_pos_norm):
        x0, y0 = start_pos_norm
        for cx, cy in zip(self.layers_x, self.layers_y):
            cx.u = np.zeros(cx.N)
            cx.set_input(cx.input[0], x0)
            cx.u = cx.input.copy()
            cy.u = np.zeros(cy.N)
            cy.set_input(cy.input[0], y0)
            cy.u = cy.input.copy()

    def step(self, sensory_input_norm, dt=Config.DT):
        tx, ty = sensory_input_norm
        estimates_x = []
        estimates_y = []
        
        for cx, cy in zip(self.layers_x, self.layers_y):
            cx.set_input(1.0, tx)
            cy.set_input(1.0, ty)
            
            dudt_x = cx.get_dudt(0, cx.u)
            cx.u += dudt_x * dt
            
            dudt_y = cy.get_dudt(0, cy.u)
            cy.u += dudt_y * dt
            
            estimates_x.append(cx.cm_of_u())
            estimates_y.append(cy.cm_of_u())
            
        return estimates_x, estimates_y

# ==============================
# 2. 实验逻辑
# ==============================
def normalize_traj(pos_seq):
    """将物理轨迹映射到 CANN 的 [0, 2pi] 环形空间"""
    min_val = np.min(pos_seq, axis=0)
    max_val = np.max(pos_seq, axis=0)
    range_val = np.max(max_val - min_val) + 1e-6
    
    # 缩放到 [0.5*pi, 1.5*pi] 避免边界问题
    pos_norm = (pos_seq - min_val) / range_val * np.pi + 0.5 * np.pi
    return pos_norm, min_val, range_val

def denormalize_traj(pos_norm, min_val, range_val):
    return (pos_norm - 0.5 * np.pi) / np.pi * range_val + min_val

def run_experiment(model, loader, num_batches=10):
    total_rmse = 0.0
    trajectories = [] 
    
    count = 0
    for batch_idx, batch_data in enumerate(loader):
        # 过滤掉可能的 None batch (如果在 dataset 中有 skip)
        if batch_data is None or batch_data[0] is None: 
            continue
            
        if batch_idx >= num_batches: break
        
        # 现在解包将正常工作，因为 collate_fn 返回 4 个值
        feats, _, _, mask = batch_data
        
        # feats shape: [1, T, 4], (x, y, yaw, ...)
        pos_seq = feats[0, :, :2].numpy()
        valid_len = mask[0].sum().item()
        pos_seq = pos_seq[:valid_len]
        
        # 归一化
        pos_norm, min_v, range_v = normalize_traj(pos_seq)
        
        model.reset(pos_norm[0])
        
        pred_seq = []
        for t in range(valid_len):
            gt_x, gt_y = pos_norm[t]
            
            est_xs, est_ys = model.step((gt_x, gt_y))
            
            final_x = np.mean(est_xs)
            final_y = np.mean(est_ys)
            
            pred_seq.append([final_x, final_y])
            
        pred_seq = np.array(pred_seq)
        
        # 反归一化计算真实 RMSE (米)
        pos_recon = denormalize_traj(pred_seq, min_v, range_v)
        
        rmse = np.sqrt(np.mean(np.sum((pos_seq - pos_recon)**2, axis=1)))
        total_rmse += rmse
        count += 1
        
        if batch_idx < 3: 
            trajectories.append((pos_seq, pos_recon))
            
    if count == 0: return 0.0, []
    return total_rmse / count, trajectories

def main():
    set_nmi_style()
    ensure_dir(Config.SAVE_DIR)
    
    print("Initializing Dataset...")
    dataset = VariableLengthTrajectoryDataset(Config.DATA_ROOT, min_len=Config.MIN_LEN)
    
    # [修复2] 添加 collate_fn 参数
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_variable_length)
    
    # --- Condition A: Single Scale (Baseline) ---
    print("Running Single-Scale Experiment...")
    model_single = MultiScaleCANN(spatial_scales=[0.5], temporal_scales=[0.5])
    rmse_single, traj_single = run_experiment(model_single, loader)
    
    # --- Condition B: Multi Scale (Ours) ---
    print("Running Multi-Scale Experiment...")
    scales_s = [0.1, 0.5, 0.9] 
    scales_t = [0.9, 0.5, 0.1] 
    model_multi = MultiScaleCANN(spatial_scales=scales_s, temporal_scales=scales_t)
    rmse_multi, traj_multi = run_experiment(model_multi, loader)
    
    print(f"\nResults:")
    print(f"Single-Scale RMSE: {rmse_single:.4f} m")
    print(f"Multi-Scale RMSE:  {rmse_multi:.4f} m")
    
    # --- Plotting (NMI Style) ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 1. Trajectory Comparison
    ax = axes[0]
    if traj_single and traj_multi:
        gt, pred_s = traj_single[0]
        _, pred_m = traj_multi[0]
        
        ax.plot(gt[:,0], gt[:,1], 'k-', lw=1.5, alpha=0.6, label='Ground Truth')
        ax.plot(pred_s[:,0], pred_s[:,1], 'r--', lw=1.2, label='Single-Scale')
        ax.plot(pred_m[:,0], pred_m[:,1], 'b-.', lw=1.2, label='Multi-Scale')
        ax.set_title("Trajectory Tracking")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.legend()
        ax.axis('equal')
    
    # 2. Error Bar
    ax = axes[1]
    labels = ['Single-Scale', 'Multi-Scale']
    values = [rmse_single, rmse_multi]
    colors = ['#e74c3c', '#3498db']
    
    bars = ax.bar(labels, values, color=colors, width=0.5, alpha=0.8)
    ax.set_ylabel("RMSE (m)")
    ax.set_title("Tracking Performance")
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(Config.SAVE_DIR, 'Experiment1_MultiScale_Representation.pdf')
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    main()