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


# --- Analysis Function 1: analyze_time_cells (Cell 6 - Optimized) ---
def analyze_time_cells(model, loader, device, n_cells_to_plot=16,save_dir="./eval_results"): # Pass device
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_hidden_states = []

    # ---- Collect LSTM outputs ----
    with torch.no_grad():
        for batch_data in loader:
             feats, imgs, targets, mask = batch_data
             # Handle potential None batch from collate_fn error
             if feats is None:
                 print("Warning: Skipping a batch due to collate error.")
                 continue

             # Move data to the specified device
             feats = feats.to(device)
             imgs = imgs.to(device)
             mask = mask.to(device)

             _, lstm_out = model(feats, imgs, mask)

             # Move results back to CPU before converting to numpy
             all_hidden_states.extend(list(lstm_out.cpu().numpy()))

    # Check if any hidden states were collected
    if not all_hidden_states:
         print("Error: No hidden states collected. Cannot perform analysis.")
         return None


    # ---- Padding ----
    # Filter out potential zero-length arrays if they occurred due to earlier errors
    all_hidden_states = [h for h in all_hidden_states if h.shape[0] > 0]
    if not all_hidden_states:
         print("Error: No valid hidden states collected after filtering. Cannot perform analysis.")
         return None

    try:
        max_T = max(h.shape[0] for h in all_hidden_states)
        hidden_dim = all_hidden_states[0].shape[-1]
    except ValueError: # Handles case where all_hidden_states might be empty after filtering
        print("Error: Could not determine dimensions from hidden states.")
        return None

    padded_states = np.zeros((len(all_hidden_states), max_T, hidden_dim))
    for i, h in enumerate(all_hidden_states):
        T = h.shape[0]
        padded_states[i, :T, :] = h
    hidden_concat = padded_states  # [N_seq, T_max, H]
    print(f"Hidden states padded to {hidden_concat.shape}")


    # ---- Plotting Functions ----
    # ---- 1️⃣ Heatmap ----
    plt.figure(figsize=(10,6))
    sns.heatmap(hidden_concat[0].T, cmap='inferno', cbar=True)
    plt.title("Example LSTM Hidden State Heatmap (Time Cell Activation)")
    plt.xlabel("Time step")
    plt.ylabel("Hidden unit")
    plt.savefig(os.path.join(save_dir, "cell_activate_heatmap.png"), dpi=200)
    # plt.show()

    # ---- 2️⃣ Peak Time Distribution ----
    hidden_mean = hidden_concat.mean(axis=0)  # [T, H]
    peak_times = hidden_mean.argmax(axis=0)
    plt.figure(figsize=(6,4))
    plt.hist(peak_times, bins=min(20, max_T), color='orange', edgecolor='k') # Adjust bins based on max_T
    plt.title("Distribution of Peak Activation Times")
    plt.xlabel("Time step of max activation")
    plt.ylabel("Cell count")
    plt.savefig(os.path.join(save_dir, "peak_activation_times.png"), dpi=200)
    # plt.show()

    # ---- 3️⃣ TSI ----
    # Calculate stats carefully, handling potential constant activations
    cell_mean = np.nanmean(hidden_concat, axis=(0, 1))
    cell_max = np.nanmax(hidden_concat, axis=(0, 1))
    cell_std = np.nanstd(hidden_concat, axis=(0, 1)) + 1e-9 # Use smaller epsilon, ensure non-zero std
    tsi = np.divide(cell_max - cell_mean, cell_std, out=np.zeros_like(cell_std), where=cell_std!=0) # Avoid division by zero


    plt.figure(figsize=(8,3))
    plt.bar(np.arange(len(tsi)), tsi)
    plt.title("Temporal Selectivity Index (TSI) per Cell")
    plt.xlabel("Cell index")
    plt.ylabel("TSI value")
    plt.savefig(os.path.join(save_dir, "temporal_selectivity_index.png"), dpi=200)
    # plt.show()

    # ---- 4️⃣ Time-Scale Diversity ----
    cell_timescale = []
    for i in range(hidden_concat.shape[-1]):
        # Analyze each trajectory for this cell
        durations_for_cell = []
        for traj_idx in range(hidden_concat.shape[0]):
             act_curve = hidden_concat[traj_idx, :, i]
             max_act = act_curve.max()
             if max_act > 1e-6: # Only consider if there's significant activation
                 above_thresh = act_curve > (max_act * 0.5)
                 duration = above_thresh.sum()
                 durations_for_cell.append(duration)
        # Append the average duration for this cell across trajectories
        if durations_for_cell:
             cell_timescale.append(np.mean(durations_for_cell))
        else:
             cell_timescale.append(0) # Or handle as NaN if preferred

    plt.figure(figsize=(6,4))
    plt.hist(cell_timescale, bins=min(15, max_T), color='teal', edgecolor='k') # Adjust bins
    plt.title("Time-Scale Diversity of Time Cells (Avg. Duration)")
    plt.xlabel("Avg. Active duration (steps > 0.5·max)")
    plt.ylabel("Cell count")
    plt.savefig(os.path.join(save_dir, "timecell_timescale_diversity.png"), dpi=200)
    # plt.show()

    # ---- 5️⃣ PCA ----
    flat_hidden = hidden_concat.reshape(-1, hidden_concat.shape[-1])
    # Filter out rows corresponding to padding
    valid_mask_flat = np.repeat(np.arange(max_T), len(all_hidden_states)) < np.repeat([h.shape[0] for h in all_hidden_states], max_T)
    flat_hidden_valid = flat_hidden[valid_mask_flat]
    time_color_valid = np.tile(np.arange(max_T), len(all_hidden_states))[valid_mask_flat]


    if flat_hidden_valid.shape[0] > 0 and flat_hidden_valid.shape[1] > 1 : # Need at least 2 samples and 2 features for PCA
        pca = PCA(n_components=2)
        try:
            reduced = pca.fit_transform(flat_hidden_valid)
            plt.figure(figsize=(5,5))
            scatter = plt.scatter(reduced[:,0], reduced[:,1], s=2, alpha=0.4, c=time_color_valid, cmap='viridis')
            plt.colorbar(scatter, label='Time Step')
            plt.title("PCA projection of Time Cell Activity (Valid Steps)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.savefig(os.path.join(save_dir, "pca_timecell_activity.png"), dpi=200)
             # plt.show()   
        except ValueError as e:
             print(f"PCA failed: {e}. Skipping PCA plot.")

    else:
         print("Skipping PCA plot due to insufficient valid data.")


    print("多尺度时间细胞评估完成。")

    # ---- 6️⃣ Cell-wise PCA (3D) by time-scale groups (Fast/Medium/Slow) ----
    # Each cell i is represented by its mean activation curve over time: v_i = mean_i(t) ∈ R^T.
    # Then PCA(3D) is applied across cells (H points), and colored by timescale group derived from cell_timescale.
    pca3_explained_variance = None
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        # hidden_mean: [T, H] was computed above
        cell_vectors = hidden_mean.T  # [H, T]

        if cell_vectors.shape[0] > 3 and cell_vectors.shape[1] > 2:
            ts = np.asarray(cell_timescale, dtype=np.float32)
            if np.all(np.isfinite(ts)) and ts.size == cell_vectors.shape[0]:
                q1, q2 = np.quantile(ts, [1.0/3.0, 2.0/3.0])
                groups = np.zeros_like(ts, dtype=np.int64)  # 0 fast, 1 medium, 2 slow
                groups[ts > q1] = 1
                groups[ts > q2] = 2

                pca_cell = PCA(n_components=3)
                Z = pca_cell.fit_transform(cell_vectors)  # [H,3]
                pca3_explained_variance = pca_cell.explained_variance_ratio_.tolist()

                fig = plt.figure(figsize=(4.4, 3.9))
                ax = fig.add_subplot(111, projection="3d")
                labels = {0: "Fast", 1: "Medium", 2: "Slow"}
                for g in (0, 1, 2):
                    idx = np.where(groups == g)[0]
                    if idx.size == 0:
                        continue
                    ax.scatter(Z[idx, 0], Z[idx, 1], Z[idx, 2],
                               s=14, alpha=0.80, label=labels[g])
                ax.set_title("Cell-wise PCA (3D) by Time-scale Group")
                ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
                ax.legend(loc="best", fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "cellwise_pca_3d_fast_med_slow.pdf"))
                plt.close()
            else:
                print("Skipping cell-wise PCA: invalid cell_timescale.")
        else:
            print("Skipping cell-wise PCA: insufficient cell_vectors shape:", cell_vectors.shape)
    except Exception as e:
        print("Cell-wise PCA(3D) failed:", repr(e))

    return {
        "pca3_explained_variance": pca3_explained_variance,
        "hidden_concat": hidden_concat,
        "peak_times": peak_times,
        "tsi": tsi,
        "cell_timescale": np.array(cell_timescale) # Ensure it's a numpy array
    }