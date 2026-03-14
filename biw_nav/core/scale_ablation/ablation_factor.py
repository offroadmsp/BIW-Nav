import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# [修复] 导入 collate_variable_length
from dataset import VariableLengthTrajectoryDataset, collate_variable_length
from scale_ablation.model.cann_base import cann_model

class Config:
    DATA_ROOT = '/media/zhen/Data/Datasets/nomad_data/go_stanford'
    SAVE_DIR = './plot/ablation'
    BATCH_SIZE = 1
    MIN_LEN = 20
    N_NEURONS = 100
    DT = 0.01

class MockArgs:
    def __init__(self, k=0.5, a=0.5, N=100, A=1.0, tau=1.0, z0=0.0):
        self.k, self.a, self.N, self.A, self.z0, self.tau = k, a, N, A, z0, tau

class MultiScaleCANN:
    def __init__(self, spatial_scales, temporal_scales):
        self.layers_x, self.layers_y = [], []
        self.a_min, self.a_max = 0.1, 1.5
        self.tau_min, self.tau_max = 1.0, 50.0

        for s_lambda, t_alpha in zip(spatial_scales, temporal_scales):
            mapped_a = self.a_min + s_lambda * (self.a_max - self.a_min)
            mapped_tau = self.tau_max - t_alpha * (self.tau_max - self.tau_min)
            self.layers_x.append(cann_model(MockArgs(a=mapped_a, tau=mapped_tau, N=Config.N_NEURONS)))
            self.layers_y.append(cann_model(MockArgs(a=mapped_a, tau=mapped_tau, N=Config.N_NEURONS)))

    def reset(self, start_pos_norm):
        x0, y0 = start_pos_norm
        for cx, cy in zip(self.layers_x, self.layers_y):
            cx.u = np.zeros(cx.N); cx.set_input(cx.input[0], x0); cx.u = cx.input.copy()
            cy.u = np.zeros(cy.N); cy.set_input(cy.input[0], y0); cy.u = cy.input.copy()

    def step(self, sensory_input_norm, dt=Config.DT):
        tx, ty = sensory_input_norm
        estimates_x, estimates_y = [], []
        for cx, cy in zip(self.layers_x, self.layers_y):
            cx.set_input(1.0, tx); cy.set_input(1.0, ty)
            cx.u += cx.get_dudt(0, cx.u) * dt
            cy.u += cy.get_dudt(0, cy.u) * dt
            estimates_x.append(cx.cm_of_u())
            estimates_y.append(cy.cm_of_u())
        return estimates_x, estimates_y

def normalize_traj(pos_seq):
    min_val, max_val = np.min(pos_seq, axis=0), np.max(pos_seq, axis=0)
    range_val = np.max(max_val - min_val) + 1e-6
    return (pos_seq - min_val) / range_val * np.pi + 0.5 * np.pi, min_val, range_val

def denormalize_traj(pos_norm, min_val, range_val):
    return (pos_norm - 0.5 * np.pi) / np.pi * range_val + min_val

def run_experiment(model, loader, num_batches=15):
    total_rmse, count = 0.0, 0
    for batch_idx, batch_data in enumerate(loader):
        if batch_data is None or batch_data[0] is None: continue
        if batch_idx >= num_batches: break
        
        # batch_data[3] 是 mask，需要 collate_fn 才能正常工作
        pos_seq = batch_data[0][0, :, :2].numpy()[:batch_data[3][0].sum().item()]
        pos_norm, min_v, range_v = normalize_traj(pos_seq)
        
        model.reset(pos_norm[0])
        pred_seq = []
        for t in range(len(pos_seq)):
            est_xs, est_ys = model.step(pos_norm[t])
            pred_seq.append([np.mean(est_xs), np.mean(est_ys)])
            
        rmse = np.sqrt(np.mean(np.sum((pos_seq - denormalize_traj(np.array(pred_seq), min_v, range_v))**2, axis=1)))
        total_rmse += rmse; count += 1
    if count == 0: return 0.0
    return total_rmse / count

def main():
    if not os.path.exists(Config.SAVE_DIR): os.makedirs(Config.SAVE_DIR)
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif', 'axes.spines.top': False, 'axes.spines.right': False})
    
    print("Initializing Dataset...")
    dataset = VariableLengthTrajectoryDataset(Config.DATA_ROOT, min_len=Config.MIN_LEN)
    # [修复] 添加 collate_fn
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_variable_length)
    
    # === 定义不同的尺度组合策略 ===
    s_linear = [0.2, 0.5, 0.8]; t_linear = [0.8, 0.5, 0.2]
    s_exp = [0.1, 0.3, 0.9]; t_exp = [0.9, 0.3, 0.1]
    s_clustered = [0.4, 0.5, 0.6]; t_clustered = [0.6, 0.5, 0.4]
    
    strategies = {
        'Linear': (s_linear, t_linear),
        'Exponential': (s_exp, t_exp),
        'Clustered': (s_clustered, t_clustered)
    }
    
    results = {}
    print("Running Scale Factor Ablation...")
    
    for name, (s_scales, t_scales) in strategies.items():
        print(f"  Testing Strategy: {name}")
        model = MultiScaleCANN(s_scales, t_scales)
        rmse = run_experiment(model, loader)
        results[name] = rmse
        print(f"    -> RMSE: {rmse:.4f}")
        
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 5))
    names = list(results.keys())
    values = list(results.values())
    colors = ['#2ecc71', '#f1c40f', '#95a5a6']
    
    bars = ax.bar(names, values, color=colors, width=0.6, alpha=0.9)
    ax.set_ylabel("Tracking RMSE (m)")
    ax.set_title("Impact of Scale Factor Distribution")
    ax.set_ylim(0, max(values)*1.2)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 1.02*height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=11)
                
    plt.tight_layout()
    save_path = os.path.join(Config.SAVE_DIR, 'Experiment2_Scale_Factors.pdf')
    plt.savefig(save_path, format='pdf', dpi=300)
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    main()