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

from model import dataread,evalcell,evaltraj,plotcell,visualNet, patchlength


# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_model(model, train_loader, device, epochs=10, lr=1e-3, save_path="timecell_model.pth"):
    model.to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for feats, imgs, targets, mask in train_loader:
            feats, imgs, targets, mask = feats.to(device), imgs.to(device), targets.to(device), mask.to(device)
            optimizer.zero_grad()
            pred, _ = model(feats, imgs, mask)
            min_dim = min(pred.shape[-1], targets.shape[-1])
            loss = loss_fn(pred[..., :min_dim], targets[..., :min_dim])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")


# --- Main Execution Block (Combines Cell 5 and calls) ---
if __name__ == "__main__":

    # --- Configuration ---
    # Consider making this a command-line argument
    DATA_ROOT_DIR = '/media/zhen/Data/Datasets/nomad_data/go_stanford'
    SAVE_DIR = "./eval_results" # Save results in a sub-directory
    BATCH_SIZE = 16
    NUM_EPOCHS = 400 # Set back to a reasonable number for training
    LEARNING_RATE = 1e-4
    HIDDEN_DIM = 128
    VISUAL_DIM = 256
    USE_YAW = True # Set based on your data and model intention
    MIN_LEN = 1
    train_status = False # Set to True to enable training
    eval_status = True # Set to True to enable evaluation

    # --- Initialize Dataset and DataLoader ---
    try:
        print("Initializing dataset...")
        dataset = dataread.VariableLengthTrajectoryDataset(DATA_ROOT_DIR, min_len=MIN_LEN, use_yaw=USE_YAW)
        if len(dataset) == 0:
             print(f"Error: No valid trajectories loaded from {DATA_ROOT_DIR}. Exiting.")
             exit()
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=patchlength.collate_variable_length)
        print("Dataset and DataLoader initialized.")
    except Exception as e:
        print(f"Error during dataset/dataloader initialization: {e}")
        exit()


    # --- Determine correct dimensions from dataset object ---
    input_feat_dim = dataset.input_feature_dim
    output_target_dim = dataset.target_dim
    print(f"Detected input feature_dim = {input_feat_dim}")
    print(f"Detected output target_dim = {output_target_dim}")


    # --- Initialize Model, Optimizer, Loss ---
    print("Initializing model...")
    model = visualNet.VisualTemporalNet(feature_dim=input_feat_dim,
                              hidden_dim=HIDDEN_DIM,
                              visual_dim=VISUAL_DIM,
                              output_dim=output_target_dim
                             ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    print("Model initialized.")

    # --- Training Loop ---
    if train_status:
        print(f"Starting training for {NUM_EPOCHS} epochs...")
        model.train() # Set model to training mode
        best_loss = float('inf')
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            batch_count = 0
            for batch_data in loader:
                # Unpack batch data
                feats, imgs, targets, mask = batch_data

                # Skip batch if collate_fn returned None due to errors
                if feats is None or imgs is None or targets is None or mask is None:
                    print(f"Skipping problematic batch in epoch {epoch+1}")
                    continue

                # Move data to the selected device
                feats, imgs, targets, mask = feats.to(device), imgs.to(device), targets.to(device), mask.to(device)

                # Forward pass
                try:
                    pred, _ = model(feats, imgs, mask)

                    # Ensure prediction and target shapes match before loss calculation
                    if pred.shape != targets.shape:
                        print(f"Warning: Shape mismatch! Pred: {pred.shape}, Target: {targets.shape}. Skipping batch.")
                        continue # Skip this batch

                    # Loss calculation and backward pass
                    loss = loss_fn(pred, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    batch_count += 1

                    # 每隔若干 epoch 保存一次
                    if (epoch + 1) % 10 == 0:
                        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_epoch_{epoch+1}.pth"))


                except Exception as e:
                    print(f"Error during training step in epoch {epoch+1}: {e}")
                    # Decide whether to continue or stop based on the error
                    continue # Skip this batch on error

                avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
                # Save best model based on loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_timecell_model.pth"))
                    print(f"Epoch {epoch+1}: New best model saved with loss {best_loss:.4f}")


            if batch_count > 0:
                avg_loss = total_loss / batch_count
                print(f'Epoch {epoch+1}, Average Loss={avg_loss:.4f}')
            else:
                print(f"Epoch {epoch+1} had no valid batches.")

        print("Training finished.")

    # --- Run Analyses ---
    if eval_status:
        model.eval() # Set model to evaluation mode
        print("\nRunning Time Cell Analysis...")
        try:
            # Need to create a new loader if the old one is exhausted or shuffling is desired differently for analysis
            # For simplicity, re-using the same loader instance here. If issues arise, create a new one.
            analysis_results = plotcell.analyze_time_cells(model, loader, device) # Pass device
        except Exception as e:
            print(f"Error during analyze_time_cells: {e}")
            analysis_results = None # Ensure it's None if analysis fails


        if analysis_results:
            print("\nRunning Extended Time Cell Evaluations...")
            try:
                extended_analysis_results = evalcell.evaluate_time_scales_and_extensions(model, loader, analysis_results, device, save_dir=SAVE_DIR) # Pass device
            except Exception as e:
                print(f"Error during evaluate_time_scales_and_extensions: {e}")
                extended_analysis_results = None

            print("\nRunning Trajectory Correlation Evaluation...")
            try:
                trajectory_correlation_results = evaltraj.evaluate_timecell_trajectory_correlation(model, loader, analysis_results, device, save_dir=SAVE_DIR) # Pass device
            except Exception as e:
                print(f"Error during evaluate_timecell_trajectory_correlation: {e}")
                trajectory_correlation_results = None

        else:
            print("Skipping extended evaluations due to failure in initial analysis.")


        # --- Final Plotting (Optional - can be done from saved files too) ---
        print("\nGenerating final activation plot...")
        with torch.no_grad(): # Disable gradient calculation
            # Get a fresh batch for plotting if loader might be exhausted
            try:
                plot_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=patchlength.collate_variable_length) # Use non-shuffled loader
                feats, imgs, targets, mask = next(iter(plot_loader))
            except StopIteration:
                print("Warning: Could not get a new batch for final plotting.")
                feats, imgs, targets, mask = None, None, None, None # Set to None


            if feats is not None:
                # Move data to GPU for inference
                feats_gpu = feats.to(device)
                imgs_gpu = imgs.to(device)
                mask_gpu = mask.to(device)

                _, lstm_out_gpu = model(feats_gpu, imgs_gpu, mask_gpu)

                # Move the result back to CPU for plotting
                lstm_out_cpu = lstm_out_gpu.cpu().numpy()

                plt.figure() # Create a new figure explicitly
                plt.imshow(lstm_out_cpu[0].T, aspect='auto', cmap='hot')
                plt.title('Final Time Cell Activation (Hidden States)')
                plt.xlabel('Time Step')
                plt.ylabel('Hidden Unit')
                plt.savefig(os.path.join(SAVE_DIR, "final_activation_heatmap.png"), dpi=200)
                print(f"Saved final activation heatmap to {SAVE_DIR}")
                #   plt.show() # Show plot if running interactively
            else:
                print("Skipping final activation plot.")

        # 调用统一的评估函数
        preds, targets, metrics = evaltraj.evaluate_and_visualize(
            model,
            loader,
            device=device,
            model_path="./eval_results/best_timecell_model.pth",
            save_dir="./eval_results",
            save_prefix="timecell_eval",
            plot=True
        )


    # plt.show()
    print("\nScript finished.")