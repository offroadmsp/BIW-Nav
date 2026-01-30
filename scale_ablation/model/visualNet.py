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
import torchvision.models as models


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



class VisualTemporalNet_Optimized(nn.Module):
    def __init__(self, feature_dim=3, hidden_dim=128, visual_dim=256, output_dim=2, ablation_config=None):
        """
        ablation_config: dict, e.g., {'use_visual': True, 'use_kinematics': True}
        """
        super().__init__()
        self.ablation_config = ablation_config or {'use_visual': True, 'use_kinematics': True}
        
        # 1. Visual Branch (CNN)
        # 使用 ResNet18 作为特征提取器 (比简单的 Conv2d 更稳)
        resnet = models.resnet18(pretrained=True)
        self.visual_extractor = nn.Sequential(*list(resnet.children())[:-1]) # Output: [B, 512, 1, 1]
        self.visual_proj = nn.Linear(512, visual_dim)
        
        # 2. Kinematic Branch (Feats Projection) -> 关键修正！
        # 将位置/速度/Yaw 投影到与视觉相同的维度，以便融合
        self.kinematic_proj = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, visual_dim)
        )
        
        # 3. Temporal Convolution (模拟多尺度时间感受野)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(visual_dim, visual_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 4. LSTM Core
        # 输入维度根据融合策略决定
        lstm_input_dim = visual_dim
        self.lstm = nn.LSTM(input_size=lstm_input_dim, hidden_size=hidden_dim, 
                            num_layers=2, batch_first=True, dropout=0.2)
        
        # 5. Output Head
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, feats, imgs, mask):
        """
        feats: [B, T, feature_dim] (Pos, Yaw)
        imgs:  [B, T, C, H, W]
        mask:  [B, T]
        """
        B, T, C, H, W = imgs.shape
        
        # --- Branch 1: Visual ---
        if self.ablation_config.get('use_visual', True):
            # CNN 处理：合并 B*T 维度
            imgs_reshaped = imgs.view(B * T, C, H, W)
            vis_feats = self.visual_extractor(imgs_reshaped) # [B*T, 512, 1, 1]
            vis_feats = vis_feats.view(B, T, -1)             # [B, T, 512]
            vis_feats = self.visual_proj(vis_feats)          # [B, T, visual_dim]
        else:
            vis_feats = torch.zeros(B, T, self.visual_proj.out_features).to(feats.device)

        # --- Branch 2: Kinematics (Feature Fusion) --- 关键修正！
        if self.ablation_config.get('use_kinematics', True):
            kin_feats = self.kinematic_proj(feats)           # [B, T, visual_dim]
        else:
            kin_feats = torch.zeros(B, T, self.visual_proj.out_features).to(feats.device)
            
        # --- Fusion Strategy ---
        # 简单的加和融合 (Residual-like) 或 Concat (需改 LSTM dim)
        # 这里用加和，强调 kinematics 对视觉的校正
        fused_input = vis_feats + kin_feats 
        
        # --- Temporal Processing ---
        # Permute for Conv1d: [B, Dim, T]
        conv_in = fused_input.permute(0, 2, 1)
        conv_out = self.temporal_conv(conv_in)
        lstm_in = conv_out.permute(0, 2, 1) # Back to [B, T, Dim]
        
        # --- LSTM ---
        # Pack sequence to handle variable lengths
        lengths = mask.sum(dim=1).cpu()
        packed_in = nn.utils.rnn.pack_padded_sequence(lstm_in, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (hn, cn) = self.lstm(packed_in)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        
        # --- Prediction ---
        # 取每个序列最后一个有效时间步的 hidden state 进行预测
        # 或者如果是序列预测，则对所有步预测。这里假设是预测整条轨迹或最后位置
        # 根据你之前的代码，貌似是 dense prediction (每个时间步都预测)
        preds = self.fc(lstm_out)
        
        return preds, lstm_out