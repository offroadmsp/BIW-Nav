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


# --- Model Definition (Cell 4 - Optimized) ---
class VisualTemporalNet(nn.Module):
    def __init__(self, feature_dim=4, hidden_dim=128, visual_dim=256, output_dim=None):
        super().__init__()
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim # This is INPUT feature dim (pos+yaw)
        self.output_dim = output_dim # This is OUTPUT target dim (e.g., pos)


        # Simple CNN for visual feature extraction，输入视觉图像信号，输入，输出视觉特征向量
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, visual_dim), nn.ReLU()
        )

        # Initialize temporal_conv to None
        self.temporal_conv = None
        self.attn = nn.MultiheadAttention(embed_dim=visual_dim, num_heads=4, batch_first=True)
        # Input to LSTM should match output of attention layer (visual_dim)
        self.lstm = nn.LSTM(visual_dim, hidden_dim, batch_first=True)
        # Final FC layer output dimension should match the desired prediction target dimension
        self.fc = nn.Linear(hidden_dim, self.output_dim) # Use self.output_dim

    def forward(self, feats, imgs, mask):
        # Ensure model components are on the same device as input tensors
        current_device = imgs.device
        self.cnn.to(current_device)
        self.attn.to(current_device)
        self.lstm.to(current_device)
        self.fc.to(current_device)

        B, T, C, H, W = imgs.shape # Get C, H, W from input
        imgs_flat = imgs.view(B * T, C, H, W) # Reshape for batch processing by CNN
        vis_flat = self.cnn(imgs_flat)
        visual_feats = vis_flat.view(B, T, -1) # Reshape back to (B, T, visual_dim)

        # Ensure 'feats' are also on the correct device (handled in the loop/collate)
        feats = feats.to(current_device)
        mask = mask.to(current_device) # Ensure mask is on the correct device

        # Apply Temporal Conv and Attention ONLY to visual features
        temp_conv_input = visual_feats.transpose(1, 2) # (B, visual_dim, T)

        # Dynamically construct temporal_conv if needed, ensuring it's on the correct device
        # Check based on input channels needed
        if self.temporal_conv is None or self.temporal_conv.in_channels != self.visual_dim:
            # Make sure the dynamically created layer is on the same device as the model/data
            self.temporal_conv = nn.Conv1d(self.visual_dim, self.visual_dim, kernel_size=3, padding=1).to(current_device)

        temp_conv_output = F.relu(self.temporal_conv(temp_conv_input))
        processed_visual = temp_conv_output.transpose(1, 2) # (B, T, visual_dim)

        # Apply attention to the temporally processed visual features
        # key_padding_mask should mask PAD tokens (where mask is False)
        attn_out, _ = self.attn(processed_visual, processed_visual, processed_visual, key_padding_mask=~mask)

        # LSTM input is the output of the attention layer
        lstm_input = attn_out

        # Packing requires lengths on CPU
        lengths = mask.sum(1).cpu().to(torch.int64) # Ensure lengths are integer type
        # Check for zero lengths before packing
        if (lengths == 0).any():
             # Handle sequences with zero length if they occur, e.g., by skipping them or returning zeros
             # This might indicate an issue upstream in data loading or masking
             print("Warning: Detected sequences with zero length.")
             # Simple handling: return zeros of the expected shape
             pred = torch.zeros(B, self.output_dim, device=current_device)
             lstm_out = torch.zeros(B, T, self.hidden_dim, device=current_device)
             return pred, lstm_out


        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)

        lstm_out_packed, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True, total_length=T) # Ensure padding back to max length T

        # Select the last valid hidden state for each sequence
        last_valid_indices = lengths - 1
        last_hidden_states = lstm_out[torch.arange(B, device=current_device), last_valid_indices] # Index on the correct device

        pred = self.fc(last_hidden_states)
        return pred, lstm_out
    

# --- Helper: Safe MSE Calculation (Cell 6 - Optimized) ---
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
        warnings.warn(f"Dimension mismatch: y_true={D_true}, y_pred={D_pred}. Using first {D_common} dims.")
    # Ensure arrays are not empty before calculating mean
    if y_true.shape[0] == 0 or D_common == 0:
         return np.nan # Or return 0.0, depending on desired behavior for empty input
    mse = np.mean((y_true[:, :D_common] - y_pred[:, :D_common])**2)
    return mse