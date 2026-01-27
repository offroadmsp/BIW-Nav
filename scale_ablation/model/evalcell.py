# -*- coding: utf-8 -*-
# Visual–Temporal Time Cell Modeling (Variable-Length Trajectories)
# 读取 **可变长度轨迹数据**（每条轨迹独立文件夹，包含 pkl + 图像序列），并结合 CNN + LSTM 进行视觉时间联合建模。

import os
import pickle
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
import warnings

# --- NMI Style Configuration ---
def set_nmi_style():
    """Sets Matplotlib style parameters for NMI-standard publication plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.format': 'pdf'
    })

# ---------- 安全版 MSE 计算 ----------
def safe_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Sample count mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")
    D_true, D_pred = y_true.shape[1], y_pred.shape[1]
    D_common = min(D_true, D_pred)
    if D_true != D_pred:
        # print(f"Warning: Dimension mismatch ({D_true} vs {D_pred}). Using first {D_common} dimensions.")
        pass
    return mean_squared_error(y_true[:, :D_common], y_pred[:, :D_common])

# --- Analysis Function 2: analyze_cell_ablation (Cell 7 - Optimized) ---
def analyze_cell_ablation(model, loader, results, device, save_dir="./eval_results"): # Pass device
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if results are valid
    if results is None:
        print("Error: 'results' object is None. Cannot perform ablation analysis.")
        return None
    
    hidden_concat = results.get('hidden_concat')
    peak_times = results.get('peak_times')
    
    if hidden_concat is None or peak_times is None:
        print("Error: Missing 'hidden_concat' or 'peak_times' in results.")
        return None

    # Apply style settings
    set_nmi_style()
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

    # ---- 1. 定义时间尺度组 (Define Time Scale Groups) ----
    max_T = hidden_concat.shape[1]
    
    # Define boundaries for Fast, Medium, Slow based on peak time
    # Fast: 0 - 1/3 T, Medium: 1/3 - 2/3 T, Slow: 2/3 - T
    b1 = max_T // 3
    b2 = 2 * max_T // 3
    
    fast_cells = np.where(peak_times < b1)[0]
    medium_cells = np.where((peak_times >= b1) & (peak_times < b2))[0]
    slow_cells = np.where(peak_times >= b2)[0]
    
    print(f"Cell Groups: Fast={len(fast_cells)}, Medium={len(medium_cells)}, Slow={len(slow_cells)}")

    # ---- 2. 消融实验 (Ablation) ----
    # 这里的消融是指：将特定组的细胞活性置零，然后观察对整体表征的影响
    # 评估指标：Reconstruction Error (MSE) - 假设我们试图用剩余细胞重建原始轨迹
    # 由于没有直接的解码器，我们比较 hidden states 的变化（或者如果有 Decoder 可以用 Decoder）
    # 这里我们用 hidden states 自我相关性或简化代替
    
    # 实际上，真正的消融通常需要重新跑模型或 mask 掉部分 neuron 后过 decoder。
    # 如果没有 Decoder，我们可以计算 mask 后表征与原始表征的差异 (MSE)。
    # 但这有点平凡（mask 越多 MSE 越大）。
    # 更得体的做法：计算各组内部的相关性 (Coherence)
    
    # Let's calculate: Intra-group vs Inter-group correlation of activity
    # Flatten: [N_seq * T, H]
    flat_activity = hidden_concat.reshape(-1, hidden_concat.shape[-1])
    
    # Correlation Matrix
    if flat_activity.shape[0] > 1000: # Downsample if too large for corrcoef
        indices = np.random.choice(flat_activity.shape[0], 1000, replace=False)
        flat_activity_sub = flat_activity[indices]
    else:
        flat_activity_sub = flat_activity
        
    corr_matrix = np.corrcoef(flat_activity_sub, rowvar=False) # [H, H]
    np.fill_diagonal(corr_matrix, np.nan) # Remove self-correlation
    
    def get_avg_corr(indices1, indices2):
        if len(indices1) == 0 or len(indices2) == 0: return 0
        sub_mat = corr_matrix[np.ix_(indices1, indices2)]
        return np.nanmean(np.abs(sub_mat))

    corr_ff = get_avg_corr(fast_cells, fast_cells)
    corr_mm = get_avg_corr(medium_cells, medium_cells)
    corr_ss = get_avg_corr(slow_cells, slow_cells)
    
    corr_fm = get_avg_corr(fast_cells, medium_cells)
    corr_ms = get_avg_corr(medium_cells, slow_cells)
    corr_fs = get_avg_corr(fast_cells, slow_cells)
    
    print(f"Intra-Group Corr: Fast={corr_ff:.3f}, Med={corr_mm:.3f}, Slow={corr_ss:.3f}")
    print(f"Inter-Group Corr: F-M={corr_fm:.3f}, M-S={corr_ms:.3f}, F-S={corr_fs:.3f}")

    # ---- 3. 模拟消融后的 MSE (Simulated Ablation MSE) ----
    # 假设全集是 Ground Truth，计算 Mask 掉部分后的 MSE
    # 这衡量了该组细胞携带的信息量（能量）
    
    original_signal = flat_activity
    
    def calc_ablation_mse(mask_indices):
        masked_signal = original_signal.copy()
        masked_signal[:, mask_indices] = 0
        return mean_squared_error(original_signal, masked_signal)

    mse_base = 0 # No ablation
    mse_fast = calc_ablation_mse(fast_cells)
    mse_med = calc_ablation_mse(medium_cells)
    mse_slow = calc_ablation_mse(slow_cells)
    
    print(f"Ablation MSE impact: Fast={mse_fast:.4f}, Med={mse_med:.4f}, Slow={mse_slow:.4f}")

    # ---- 4. 跨轨迹一致性 (Cross-Trajectory Consistency) ----
    # 如果有多个轨迹，计算同一细胞在不同轨迹下的相关性
    # hidden_concat: [N_seq, T, H]
    N_seq = hidden_concat.shape[0]
    avg_cross_corr = 0
    
    if N_seq >= 2:
        # Pad/Crop to same length for correlation
        len1 = hidden_concat.shape[1]
        # Compare trajectory 0 and 1
        traj0 = hidden_concat[0]
        traj1 = hidden_concat[1]
        
        # Pearson correlation per cell
        corr_cross_traj = np.zeros(hidden_concat.shape[-1])
        for cell_idx in range(hidden_concat.shape[-1]):
            act0 = traj0[:, cell_idx]
            act1 = traj1[:, cell_idx]
            # Simple handling of different lengths or padding zeros:
            # Only consider active parts? Or whole padded sequence? 
            # Using whole padded sequence might inflate correlation due to zeros.
            # Let's verify variance.
            if np.std(act0) > 1e-6 and np.std(act1) > 1e-6:
                corr_cross_traj[cell_idx], _ = pearsonr(act0, act1)
            else:
                corr_cross_traj[cell_idx] = np.nan
        
        avg_cross_corr = np.nanmean(np.abs(corr_cross_traj))

        # ---- Plot: Cross-Trajectory Correlation ----
        plt.figure(figsize=(3.5, 2.5))
        # Use absolute correlation
        plot_data = np.abs(corr_cross_traj)
        # Handle NaNs for plotting
        plot_data = np.nan_to_num(plot_data)
        
        plt.bar(np.arange(len(plot_data)), plot_data, color=colors[5], width=0.8, alpha=0.9)
        plt.title(f"Cross-Trajectory Correlation\n(Avg |r| = {avg_cross_corr:.2f})")
        plt.xlabel("Cell Index")
        plt.ylabel("|Correlation|")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cross_traj_corr.pdf"))
        plt.close()
    else:
        print("Skipping cross-trajectory correlation (need at least 2 trajectories).")
        avg_cross_corr = np.nan


    # Save Summary Table
    df_data = {
        "metric": ["MSE_baseline", "MSE_no_fast", "MSE_no_medium", "MSE_no_slow",
                   "AvgCorr_Fast_Fast", "AvgCorr_Med_Med", "AvgCorr_Slow_Slow",
                   "AvgCorr_Fast_Med", "AvgCorr_Med_Slow", "AvgCorr_Fast_Slow",
                   "AvgCorr_CrossTraj"],
        "value": [mse_base, mse_fast, mse_med, mse_slow,
                  corr_ff, corr_mm, corr_ss,
                  corr_fm, corr_ms, corr_fs,
                  avg_cross_corr]
    }
    df = pd.DataFrame(df_data)
    df.to_csv(os.path.join(save_dir, "ablation_metrics.csv"), index=False)
    print("Ablation metrics saved.")
    
    return df