# -*- coding: utf-8 -*-
# Visual–Temporal Time Cell Modeling (Variable-Length Trajectories)
# plotcell.py - Optimized for NMI Standards (Refined Axis Labels & Titles)

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
# 3D plotting import
from mpl_toolkits.mplot3d import Axes3D
import warnings

# --- NMI Style Configuration ---
def set_nmi_style():
    """Sets Matplotlib style parameters for NMI-standard publication plots."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
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
        'pdf.fonttype': 42,          # Embed fonts (Type 42)
        'ps.fonttype': 42,
        'savefig.format': 'pdf'      
    })

def analyze_time_cells(model, loader, device, n_cells_to_plot=16, save_dir="./eval_results"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    all_hidden_states = []

    # ---- Collect LSTM outputs (Truncated to real length) ----
    with torch.no_grad():
        for batch_data in loader:
             feats, imgs, targets, mask = batch_data
             if feats is None: continue

             feats = feats.to(device)
             imgs = imgs.to(device)
             mask = mask.to(device)

             _, lstm_out = model(feats, imgs, mask)   # lstm_out: [B, T_pad, H]
             lstm_out = lstm_out.detach().cpu()
             mask_cpu = mask.detach().cpu()

             B = lstm_out.shape[0]
             for b in range(B):
                 L = int(mask_cpu[b].sum().item())   # ✅ Real length
                 if L <= 0:
                     continue
                 all_hidden_states.append(lstm_out[b, :L, :].numpy())  # ✅ Save valid segment only

    if not all_hidden_states:
         print("Error: No hidden states collected.")
         return None

    # ---- Padding for matrix operations ----
    try:
        max_T = max(h.shape[0] for h in all_hidden_states)
        hidden_dim = all_hidden_states[0].shape[-1]
    except ValueError: return None

    # Use NaN padding to be safe
    padded_states = np.full((len(all_hidden_states), max_T, hidden_dim), np.nan, dtype=np.float32)
    for i, h in enumerate(all_hidden_states):
        T = h.shape[0]
        padded_states[i, :T, :] = h
    hidden_concat = padded_states
    print(f"Hidden states padded to {hidden_concat.shape}")

    # ---- Plotting Setup ----
    set_nmi_style()
    colors = ['#3C5488', '#E64B35', '#00A087', '#4DBBD5']

    # ---- 1️⃣ Heatmap (Sorted) ----
    plt.figure(figsize=(4.0, 3.2)) 
    # Pick the longest trajectory for better visualization
    lengths_arr = np.array([h.shape[0] for h in all_hidden_states])
    best_i = int(np.argmax(lengths_arr))
    T_i = int(lengths_arr[best_i])
    activity_matrix = hidden_concat[best_i, :T_i, :].T  

    # Handle NaNs for sorting
    am = np.nan_to_num(activity_matrix, nan=-1e9)
    peak_times_example = np.argmax(am, axis=1)
    sorted_indices = np.argsort(peak_times_example)
    sorted_activity = activity_matrix[sorted_indices]

    # Dynamic vmin/vmax for better contrast
    vals = sorted_activity[np.isfinite(sorted_activity)]
    if vals.size > 0:
        vmin, vmax = np.percentile(vals, [2, 98])
    else:
        vmin, vmax = None, None

    ax = sns.heatmap(sorted_activity, cmap='magma', vmin=vmin, vmax=vmax,
                     cbar_kws={'label': 'Activity'}, rasterized=True, 
                     xticklabels=max(1, T_i//10), yticklabels=False)
    
    # [NMI Edit] Renamed axis labels
    ax.set_title("Time-cell activation sequence (sorted by peak time)")
    ax.set_xlabel("Relative time step")
    ax.set_ylabel("Time cells (sorted)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cell_activate_heatmap.pdf"))
    plt.close()

    # ---- 2️⃣ Peak Time Distribution ----
    # Mean across trials (ignoring NaNs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        hidden_mean = np.nanmean(hidden_concat, axis=0)
    
    peak_times = np.argmax(np.nan_to_num(hidden_mean, nan=-1), axis=0)
    
    plt.figure(figsize=(4.0, 3.0))
    plt.hist(peak_times, bins=min(20, max_T), color=colors[0], 
             edgecolor='white', linewidth=0.5, alpha=0.9)
    
    # [NMI Edit] Renamed axis labels
    plt.title("Peak time distribution across cells")
    plt.xlabel("Peak time (step)")
    plt.ylabel("Number of cells")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "peak_activation_times.pdf"))
    plt.close()

    # ---- 3️⃣ TSI Bar Chart ----
    cell_mean = np.nanmean(hidden_concat, axis=(0, 1))
    cell_max = np.nanmax(hidden_concat, axis=(0, 1))
    cell_std = np.nanstd(hidden_concat, axis=(0, 1)) + 1e-9
    tsi = np.divide(cell_max - cell_mean, cell_std, out=np.zeros_like(cell_std), where=cell_std!=0)

    plt.figure(figsize=(4.0, 3.0))
    plt.bar(np.arange(len(tsi)), tsi, color=colors[2], width=0.8, alpha=0.8)
    
    # [NMI Edit] Renamed axis labels
    plt.title("Temporal selectivity across cells (TSI)")
    plt.xlabel("Cell index")
    plt.ylabel("TSI")
    
    plt.xlim(-1, len(tsi))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "temporal_selectivity_index.pdf"))
    plt.close()

    # ---- 4️⃣ Time-Scale Diversity ----
    cell_timescale = []
    for i in range(hidden_concat.shape[-1]):
        durations = []
        for traj_idx in range(hidden_concat.shape[0]):
             act_curve = hidden_concat[traj_idx, :, i]
             # Check if valid (not all NaNs)
             if np.isnan(act_curve).all(): continue
             
             curve_clean = np.nan_to_num(act_curve)
             if curve_clean.max() > 1e-6:
                 durations.append((curve_clean > (curve_clean.max() * 0.5)).sum())
        cell_timescale.append(np.mean(durations) if durations else 0)

    plt.figure(figsize=(4.0, 3.0))
    plt.hist(cell_timescale, bins=min(15, max_T), color=colors[3], 
             edgecolor='white', linewidth=0.5, alpha=0.9)
    
    # [NMI Edit] Renamed axis labels
    plt.title("Estimated temporal scale distribution")
    plt.xlabel("Estimated duration (steps)")
    plt.ylabel("Number of cells")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "timecell_timescale_diversity.pdf"))
    plt.close()

    # ---- 5️⃣ 2D PCA (Optimized Logic) ----
    print("Preparing data for PCA...")
    
    # [Optimized Logic for NMI & Stability]
    N_seq, max_T, H = hidden_concat.shape
    lengths = np.array([h.shape[0] for h in all_hidden_states], dtype=np.int64)  # real lengths

    flat_hidden = hidden_concat.reshape(-1, H)            # [N_seq*max_T, H]
    t_idx = np.tile(np.arange(max_T), N_seq)              # [N_seq*max_T]
    len_rep = np.repeat(lengths, max_T)                   # [N_seq*max_T]
    
    # 1. Mask out padding steps based on length
    valid_mask = t_idx < len_rep                          
    
    flat_valid = flat_hidden[valid_mask]                  # [N_valid, H]
    time_color = t_idx[valid_mask]                        # [N_valid]

    # 2. Critical: Filter out any NaNs/Infs that might remain
    finite_mask = np.isfinite(flat_valid).all(axis=1)
    flat_valid = flat_valid[finite_mask]
    time_color = time_color[finite_mask]

    if flat_valid.shape[0] > 0 and flat_valid.shape[1] > 1:
        pca = PCA(n_components=2)
        try:
            reduced = pca.fit_transform(flat_valid)
            plt.figure(figsize=(3.5, 3.5)) 
            sc = plt.scatter(reduced[:,0], reduced[:,1], c=time_color, 
                             s=3, alpha=0.6, cmap='viridis', rasterized=True, linewidth=0)
            cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
            cbar.set_label('Relative time step', rotation=270, labelpad=12) # Consistent with Heatmap
            cbar.outline.set_visible(False)
            
            # [NMI Edit] Renamed axis labels - Conceptual Interpretation
            plt.title("Latent-state manifold over time (PCA)")
            plt.xlabel("Dominant temporal variation")
            plt.ylabel("Secondary temporal variation")
            
            plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "pca_timecell_activity.pdf"))
            plt.close()
        except Exception as e:
             print(f"PCA plot skipped: {e}")

    # ---- 6️⃣ 3D Cell-wise PCA ----
    pca3_explained_variance = None
    try:
        # Transpose to [H, T] -> Each cell is a sample
        # Use hidden_mean (averaged across trials)
        cell_vectors = hidden_mean.T 
        # Handle NaNs in cell vectors (replace with 0 for PCA stability)
        cell_vectors = np.nan_to_num(cell_vectors, nan=0.0)

        if cell_vectors.shape[0] > 3 and cell_vectors.shape[1] > 2:
            ts = np.asarray(cell_timescale, dtype=np.float32)
            if np.all(np.isfinite(ts)) and ts.size == cell_vectors.shape[0]:
                
                # Grouping Logic
                q1, q2 = np.quantile(ts, [1.0/3.0, 2.0/3.0])
                groups = np.zeros_like(ts, dtype=np.int64) # 0:Fast, 1:Med, 2:Slow
                groups[ts > q1] = 1
                groups[ts > q2] = 2

                pca_cell = PCA(n_components=3)
                Z = pca_cell.fit_transform(cell_vectors)  # [H, 3]
                pca3_explained_variance = pca_cell.explained_variance_ratio_.tolist()

                # 3D Plot Settings
                fig = plt.figure(figsize=(5.0, 4.5))
                ax = fig.add_subplot(111, projection="3d")
                
                group_colors = {0: '#3C5488', 1: '#E64B35', 2: '#00A087'}
                labels = {0: "Fast", 1: "Medium", 2: "Slow"}
                
                for g in (0, 1, 2):
                    idx = np.where(groups == g)[0]
                    if idx.size == 0: continue
                    ax.scatter(Z[idx, 0], Z[idx, 1], Z[idx, 2],
                               s=30, alpha=0.85, c=group_colors[g], 
                               label=labels[g], edgecolors='white', linewidth=0.5,
                               depthshade=True)

                # NMI Minimalist 3D Style
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.xaxis.pane.set_edgecolor('w')
                ax.yaxis.pane.set_edgecolor('w')
                ax.zaxis.pane.set_edgecolor('w')
                ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4)
                
                ax.set_title("Time-Scale Manifold (3D PCA)")
                ax.set_xlabel("Short-horizon temporal variation", labelpad=-8)
                ax.set_ylabel("Medium-horizon temporal variation", labelpad=-8)
                ax.set_zlabel("Long-horizon temporal variation", labelpad=-8)
                ax.tick_params(axis='both', which='major', pad=-2)
                ax.view_init(elev=25, azim=135)
                
                ax.legend(loc="upper right", frameon=False, fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "cellwise_pca_3d_fast_med_slow.pdf"))
                plt.close()
            else:
                print("Skipping 3D PCA: invalid timescales.")
    except Exception as e:
        print("Cell-wise PCA(3D) failed:", repr(e))

    return {
        "pca3_explained_variance": pca3_explained_variance,
        "hidden_concat": hidden_concat,
        "peak_times": peak_times,
        "tsi": tsi,
        "cell_timescale": np.array(cell_timescale) 
    }