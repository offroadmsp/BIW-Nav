# -*- coding: utf-8 -*-
# evaltraj.py - Optimized for NMI Standards (Fixed Target Dimension Bug)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
import torch

# --- NMI Style Configuration ---
def set_nmi_style():
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
        'axes.spines.top': False,    
        'axes.spines.right': False,  
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'savefig.format': 'pdf'      
    })

# --- Analysis Function 3: Correlation ---
def evaluate_timecell_trajectory_correlation(model, loader, results, device, save_dir="./eval_results"):
    os.makedirs(save_dir, exist_ok=True)
    
    print("Computing synchronized hidden states and trajectories for correlation...")
    model.eval()
    
    all_hidden_list = [] 
    all_pos_list = []
    all_yaw_list = []
    
    with torch.no_grad():
        for batch_data in loader:
            feats, imgs, _, mask = batch_data # Ignore targets
            if feats is None: continue
            
            feats_dev, imgs_dev, mask_dev = feats.to(device), imgs.to(device), mask.to(device)
            _, lstm_out = model(feats_dev, imgs_dev, mask_dev)
            
            lstm_np = lstm_out.detach().cpu().numpy()
            mask_np = mask.numpy()
            feats_np = feats.numpy()
            
            B = feats_np.shape[0]
            for b in range(B):
                L = int(mask_np[b].sum())
                if L > 0:
                    all_hidden_list.append(lstm_np[b, :L, :])
                    all_pos_list.append(feats_np[b, :L, :2]) # x, y
                    if feats_np.shape[-1] >= 3:
                        all_yaw_list.append(feats_np[b, :L, 2]) # yaw
                    else:
                        all_yaw_list.append(np.zeros(L))

    if not all_hidden_list:
        print("Error: No valid data collected.")
        return None

    # Concatenate (Flatten)
    flat_hidden = np.concatenate(all_hidden_list, axis=0)
    flat_pos = np.concatenate(all_pos_list, axis=0)
    flat_yaw = np.concatenate(all_yaw_list, axis=0)
    
    # Calculate Velocity
    flat_vel_list = []
    for pos_traj in all_pos_list:
        if len(pos_traj) > 1:
            v = np.linalg.norm(np.diff(pos_traj, axis=0, prepend=pos_traj[:1]), axis=1)
            flat_vel_list.append(v)
        else:
            flat_vel_list.append(np.zeros(len(pos_traj)))
    flat_vel = np.concatenate(flat_vel_list, axis=0)

    # Filtering NaNs
    valid_mask = np.isfinite(flat_hidden).all(axis=1) & np.isfinite(flat_pos).all(axis=1)
    flat_hidden = flat_hidden[valid_mask]
    flat_pos = flat_pos[valid_mask]
    flat_yaw = flat_yaw[valid_mask]
    flat_vel = flat_vel[valid_mask]
    
    n_cells = flat_hidden.shape[1]
    corr_pos = np.zeros(n_cells); corr_yaw = np.zeros(n_cells); corr_vel = np.zeros(n_cells)

    for c in range(n_cells):
        h = flat_hidden[:, c]
        if np.std(h) < 1e-6:
            corr_pos[c] = np.nan; corr_yaw[c] = np.nan; corr_vel[c] = np.nan
            continue
        
        rx, _ = pearsonr(h, flat_pos[:, 0])
        ry, _ = pearsonr(h, flat_pos[:, 1])
        corr_pos[c] = (abs(rx) + abs(ry)) / 2
        
        if np.std(flat_yaw) > 1e-6: corr_yaw[c] = abs(pearsonr(h, flat_yaw)[0])
        if np.std(flat_vel) > 1e-6: corr_vel[c] = abs(pearsonr(h, flat_vel)[0])

    # Plotting
    set_nmi_style()
    colors = ['#4DBBD5', '#E64B35', '#00A087'] 
    plt.figure(figsize=(7.5, 3.5))
    plt.subplot(1, 2, 1)
    plt.bar(np.arange(n_cells), np.nan_to_num(corr_pos), color=colors[0], width=0.8, alpha=0.9)
    plt.title("Spatial Correlation (Position)"); plt.ylim(0, 1.0)
    plt.subplot(1, 2, 2)
    plt.bar(np.arange(n_cells), np.nan_to_num(corr_yaw), color=colors[1], width=0.8, alpha=0.9)
    plt.title("Angular Correlation (Yaw)"); plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cell_position_yaw_corr.pdf")); plt.close()
    
    return pd.DataFrame({'corr_pos': corr_pos})


# --- Analysis Function 4: evaluate_and_visualize ---
def evaluate_and_visualize(model, test_loader, device, model_path, save_dir, save_prefix, plot=True):
    os.makedirs(save_dir, exist_ok=True)
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except: pass
    
    model.eval()
    model.to(device)

    preds_list = []
    targets_list = []
    
    print("Starting trajectory evaluation...")
    with torch.no_grad():
        for batch_data in test_loader:
            feats, imgs, _, mask = batch_data # Ignore original targets (endpoint only)
            if feats is None: continue

            # [Fix] Extract Sequence Targets from Feats
            targets_seq = feats[:, :, :2].clone() # [B, T, 2]

            feats = feats.to(device)
            imgs = imgs.to(device)
            targets_seq = targets_seq.to(device)
            mask = mask.to(device)
            
            # Forward
            pred, _ = model(feats, imgs, mask) # [B, T, 2]
            
            # Unpack Batch
            batch_size = pred.shape[0]
            mask_cpu = mask.cpu()
            pred_cpu = pred.cpu().numpy()
            target_cpu = targets_seq.cpu().numpy() # Now this is 3D [B, T, 2]
            
            for b in range(batch_size):
                L = int(mask_cpu[b].sum().item())
                if L == 0: continue
                
                # Extract valid segment
                p_traj = pred_cpu[b, :L, :]
                t_traj = target_cpu[b, :L, :2] # Now this slicing works!
                
                preds_list.append(p_traj)
                targets_list.append(t_traj)

    if not preds_list:
        print("Error: No valid predictions generated.")
        return None, None, None

    # Concatenate
    preds_all_flat = np.concatenate(preds_list, axis=0)
    targets_all_flat = np.concatenate(targets_list, axis=0)

    # Metrics
    mse = mean_squared_error(targets_all_flat, preds_all_flat)
    r2 = r2_score(targets_all_flat, preds_all_flat)
    metrics = {"MSE": mse, "R2": r2}
    print(f"Evaluation Results: MSE={mse:.6f}, R²={r2:.4f}")

    # Save & Plot
    np.save(os.path.join(save_dir, f"{save_prefix}_preds.npy"), preds_all_flat)
    if plot and preds_all_flat.shape[1] >= 2:
        set_nmi_style()
        plt.figure(figsize=(4.0, 4.0))
        # Plot ONLY the first valid trajectory
        t_sample = targets_list[0]; p_sample = preds_list[0]
        plt.plot(t_sample[:,0], t_sample[:,1], 'k-', linewidth=2.0, label="Ground Truth", alpha=0.8)
        plt.plot(p_sample[:,0], p_sample[:,1], 'r--', linewidth=2.0, label="Prediction", alpha=0.9)
        plt.scatter(t_sample[0,0], t_sample[0,1], marker='o', c='k', s=40)
        plt.scatter(t_sample[-1,0], t_sample[-1,1], marker='x', c='k', s=40)
        plt.legend(frameon=False); plt.axis('equal'); plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{save_prefix}_traj.pdf")); plt.close()
        print(f"Trajectory plot saved.")

    return preds_all_flat, targets_all_flat, metrics