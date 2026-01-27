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
from sklearn.metrics import mean_squared_error, r2_score

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

# --- Analysis Function 3: evaluate_timecell_trajectory_correlation (Cell 8 - Optimized) ---
def evaluate_and_visualize(model, loader, device, model_path, save_dir, save_prefix, plot=True):
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载模型权重（如果提供）
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}: {e}")
    
    model.eval()
    model.to(device)

    preds = []
    targets_all = []

    # ------------------------------------
    # 1. 批量预测
    # ------------------------------------
    print("Starting evaluation loop...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            feats, imgs, targets, mask = batch_data
            
            if feats is None: continue

            feats = feats.to(device)
            imgs = imgs.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            # Forward pass: 假设模型返回 (pred, hidden)
            # 根据 main_cell.py 中的调用，model 返回 (pred, hidden)
            # pred shape: [Batch, Output_Dim] (通常是最后一帧的预测)
            # targets shape: [Batch, Output_Dim]
            
            output_tuple = model(feats, imgs, mask)
            if isinstance(output_tuple, tuple):
                pred = output_tuple[0]
            else:
                pred = output_tuple

            # 收集结果
            preds.append(pred.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    if not preds:
        print("Error: No predictions generated.")
        return None, None, None

    preds_all = np.concatenate(preds, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    # ------------------------------------
    # 2. 计算评估指标
    # ------------------------------------
    mse = mean_squared_error(targets_all, preds_all)
    r2 = r2_score(targets_all, preds_all)
    metrics = {"MSE": mse, "R2": r2}
    print(f"Evaluation Results: MSE={mse:.6f}, R²={r2:.4f}")

    # ------------------------------------
    # 3. 保存结果数据
    # ------------------------------------
    save_pred_path = os.path.join(save_dir, f"{save_prefix}_preds.npy")
    save_target_path = os.path.join(save_dir, f"{save_prefix}_targets.npy")
    np.save(save_pred_path, preds_all)
    np.save(save_target_path, targets_all)
    print(f"Saved predictions to {save_pred_path}")

    # ------------------------------------
    # 4. 绘制轨迹对比图 (NMI Style)
    # ------------------------------------
    if plot:
        set_nmi_style()
        colors = ['#377eb8', '#ff7f00', '#4daf4a'] # Blue, Orange, Green
        
        # 假设预测的是 2D 坐标 (x, y)
        if preds_all.shape[1] >= 2:
            plt.figure(figsize=(3.5, 3.5)) # Square figure for spatial plot
            
            # Ground Truth (Black/Grey, solid)
            plt.plot(targets_all[:, 0], targets_all[:, 1], 
                     color='black', linestyle='-', linewidth=1.5, alpha=0.7, label="Ground Truth")
            
            # Prediction (Color, dashed)
            plt.plot(preds_all[:, 0], preds_all[:, 1], 
                     color='#e41a1c', linestyle='--', linewidth=1.5, alpha=0.9, label="Prediction")
            
            plt.legend(frameon=False) # No frame for legend in clean style
            plt.title(f"Trajectory Prediction\n(MSE={mse:.4f})")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            
            # Crucial for spatial data: Equal Aspect Ratio
            plt.axis('equal') 
            
            plt.tight_layout()
            save_plot_path = os.path.join(save_dir, f"{save_prefix}_trajectory_plot.pdf")
            plt.savefig(save_plot_path)
            plt.close()
            print(f"Saved plot to {save_plot_path}")
        else:
            print("Output dimension < 2, skipping 2D trajectory plot.")

    return preds_all, targets_all, metrics