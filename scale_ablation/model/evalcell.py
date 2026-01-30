# -*- coding: utf-8 -*-
# evalcell.py - Optimized for NMI Standards (Merged with fixed Cross-Trajectory Logic)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
import torch
from sklearn.metrics import mean_squared_error
import warnings

# --- NMI Style Configuration (Larger Fonts) ---
def set_nmi_style():
    """Sets Matplotlib style parameters for NMI-standard publication plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        # 字体整体调大，适应论文排版
        'font.size': 10,              
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.linewidth': 0.8,       
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'axes.spines.top': False,    
        'axes.spines.right': False,  
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'pdf.fonttype': 42,          # Embed fonts
        'savefig.format': 'pdf'      
    })

def safe_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
    D_common = min(y_true.shape[1], y_pred.shape[1])
    return np.mean((y_true[:, :D_common] - y_pred[:, :D_common])**2)

def evaluate_time_scales_and_extensions(model, loader, results, device, thresholds=(10, 30), save_dir="./eval_results"):
    os.makedirs(save_dir, exist_ok=True)
    if results is None: return None

    hidden_concat = results.get('hidden_concat')
    cell_timescale = np.array(results.get('cell_timescale'))
    
    fast_thr, slow_thr = thresholds
    fast_idx = np.where(cell_timescale <= fast_thr)[0]
    medium_idx = np.where((cell_timescale > fast_thr) & (cell_timescale <= slow_thr))[0]
    slow_idx = np.where(cell_timescale > slow_thr)[0]

    # --- Ablation Logic (Internal helper) ---
    def ablate_cells(model, mask_idx):
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_data in loader:
                feats, imgs, targets, mask = batch_data
                if feats is None: continue
                
                # Move to GPU
                feats, imgs, mask_dev = feats.to(device), imgs.to(device), mask.to(device)
                
                # Forward pass
                _, lstm_out = model(feats, imgs, mask_dev)
                
                # Apply Ablation Mask
                lstm_out_ablated = lstm_out.clone()
                if len(mask_idx) > 0:
                    lstm_out_ablated[:, :, mask_idx] = 0
                
                # Recompute prediction from ablated hidden states
                lengths = mask_dev.sum(1)
                last_idx = torch.clamp(lengths - 1, min=0)
                last_hidden = lstm_out_ablated[torch.arange(lstm_out_ablated.size(0)), last_idx]
                pred = model.fc(last_hidden)
                
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
        return safe_mean_squared_error(y_true, y_pred)

    print("Running ablation analysis...")
    mse_base = ablate_cells(model, [])
    mse_fast = ablate_cells(model, fast_idx) if len(fast_idx) > 0 else mse_base
    mse_med = ablate_cells(model, medium_idx) if len(medium_idx) > 0 else mse_base
    mse_slow = ablate_cells(model, slow_idx) if len(slow_idx) > 0 else mse_base

    # --- Plotting ---
    set_nmi_style()
    colors = ['#B0B0B0', '#E64B35', '#E18727', '#00A087'] # Grey, Red, Orange, Teal

    # 1. Ablation Bar Chart (Improved)
    plt.figure(figsize=(4.5, 3.5))
    cats = ['Baseline', 'No Fast', 'No Med', 'No Slow']
    cats = ["Multi Scales", "w/o Short term", "w/o Mid-term", "w/o Long term"]

    vals = [mse_base, mse_fast, mse_med, mse_slow]
    
    x = np.arange(len(cats))
    plt.bar(x, vals, color=colors, width=0.7, alpha=0.9, edgecolor='white')
    plt.xticks(x, cats, rotation=60) # No rotation needed with larger figure
    plt.ylabel("Reconstruction MSE")
    plt.title("Impact of Time-Scale Ablation")
    
    # Add value labels
    ylim = max(vals) * 1.15 if vals else 1.0
    plt.ylim(0, ylim)
    for i, v in enumerate(vals):
        plt.text(i, v + ylim*0.02, f"{v:.4f}", ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "ablation_analysis.pdf"))
    plt.close()

    # 2. Correlation Matrix
    hidden_mean = np.nanmean(hidden_concat, axis=0)
    # Filter for active cells to avoid NaNs
    valid_cells = np.where(np.nanstd(hidden_mean, axis=0) > 1e-6)[0]
    
    if len(valid_cells) > 1:
        corr = np.corrcoef(hidden_mean[:, valid_cells].T)
        
        plt.figure(figsize=(4.0, 3.5))
        # Use rasterized for dense matrices
        sns.heatmap(corr, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                    square=True, rasterized=True,
                    cbar_kws={'label': 'Pearson r', 'shrink': 0.8})
        plt.title("Inter-Cell Correlation Matrix")
        plt.xlabel("Cell Index")
        plt.ylabel("Cell Index")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "correlation_matrix.pdf"))
        plt.close()

    # 3. Multi-scale Filtering Example
    plt.figure(figsize=(4.0, 3.0))
    if hidden_concat.shape[2] > 0:
        base = hidden_concat[0, :, 0] # Example cell
        plt.plot(base, label="Raw Activity", color='black', alpha=0.2, linewidth=1)
        
        line_styles = ['-', '--', ':']
        line_colors = ['#3C5488', '#4DBBD5', '#00A087']
        sigmas = [1, 3, 6]
        
        for i, s in enumerate(sigmas):
            smoothed = gaussian_filter1d(base, sigma=s)
            plt.plot(smoothed, label=f"Smoothed ($\sigma={s}$)", 
                     color=line_colors[i], linestyle=line_styles[i], linewidth=1.5)
        
        plt.title("Multi-scale Temporal Response")
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Activity")
        plt.legend(frameon=False, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "multi_scale_filtering.pdf"))
        plt.close()

    # 4. Cross-Trajectory Consistency (Merged from fixed)
    # 功能：计算同一细胞在不同轨迹（Traj 0 vs Traj 1）间的一致性
    avg_cross_corr = np.nan
    if hidden_concat.shape[0] >= 2:
        # Compare Traj 0 and Traj 1
        traj0 = hidden_concat[0]
        traj1 = hidden_concat[1]
        n_cells = hidden_concat.shape[2]
        
        corr_cross_traj = np.zeros(n_cells)
        
        for cell_idx in range(n_cells):
            act0 = traj0[:, cell_idx]
            act1 = traj1[:, cell_idx]
            
            # Check variance to avoid division by zero
            if np.std(act0) > 1e-6 and np.std(act1) > 1e-6:
                r, _ = pearsonr(act0, act1)
                corr_cross_traj[cell_idx] = np.abs(r) # Use absolute correlation
            else:
                corr_cross_traj[cell_idx] = np.nan
        
        avg_cross_corr = np.nanmean(corr_cross_traj)
        
        # Plotting
        plt.figure(figsize=(4.5, 3.0))
        # Use a distinctive color (e.g., Purple)
        plt.bar(np.arange(n_cells), corr_cross_traj, color='#984EA3', width=0.8, alpha=0.9)
        plt.title(f"Cross-Trial Consistency (Avg |r| = {avg_cross_corr:.2f})")
        plt.xlabel("Cell Index")
        plt.ylabel("|Correlation|")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "cross_traj_corr.pdf"))
        plt.close()
    else:
        print("Skipping cross-trajectory correlation (need >= 2 trajectories).")

    # Save CSV Summary
    df = pd.DataFrame({
        "metric": ["MSE_Base", "MSE_Fast", "MSE_Med", "MSE_Slow", "Avg_Cross_Corr"], 
        "value": [mse_base, mse_fast, mse_med, mse_slow, avg_cross_corr]
    })
    df.to_csv(os.path.join(save_dir, "timecell_eval_summary.csv"), index=False)
    print("Ablation evaluation completed.")

    return {"mse_values": [mse_base, mse_fast, mse_med, mse_slow], "csv": "saved"}