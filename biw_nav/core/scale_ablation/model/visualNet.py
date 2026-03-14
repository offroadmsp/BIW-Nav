# -*- coding: utf-8 -*-
# model/visualNet.py
# Optimized for NMI: Includes Explicit Multi-Scale Temporal Module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VisualTemporalNet_Optimized(nn.Module):
    def __init__(self, feature_dim=3, hidden_dim=128, visual_dim=256, output_dim=2, ablation_config=None):
        """
        ablation_config: dict, e.g., {'use_visual': True, 'use_kinematics': True}
        """
        super().__init__()
        self.ablation_config = ablation_config or {'use_visual': True, 'use_kinematics': True}
        
        # 1. Visual Branch (ResNet18 Feature Extractor)
        # NMI 标准：使用预训练骨干网络
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.visual_extractor = nn.Sequential(*list(resnet.children())[:-1]) # [B, 512, 1, 1]
        self.visual_proj = nn.Linear(512, visual_dim)
        
        # 2. Kinematic Branch (Feature Fusion)
        # 将低维运动学信息 (x, y, yaw) 投影到高维语义空间
        self.kinematic_proj = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, visual_dim)
        )
        
        # 3. [Core Method Update] Multi-Scale Temporal Module
        # 显式模拟 CANN 中的 Fast / Medium / Slow 时间尺度
        # 通过不同的 dilation rate 实现不同的时间感受野
        self.conv_fast = nn.Conv1d(visual_dim, visual_dim//4, kernel_size=3, padding=1, dilation=1) #对应alpha = 0
        self.conv_med  = nn.Conv1d(visual_dim, visual_dim//4, kernel_size=3, padding=2, dilation=2) #对应0.5
        self.conv_slow = nn.Conv1d(visual_dim, visual_dim//4, kernel_size=3, padding=4, dilation=4) #对应1.0
        
        # 融合层 (将多尺度特征融合回 visual_dim)
        # 输入维度 = (visual_dim // 4) * 3
        self.ms_fusion = nn.Linear((visual_dim // 4) * 3, visual_dim)
        self.ms_norm = nn.LayerNorm(visual_dim)
        
        # 4. LSTM Core (Recurrent Dynamics)
        self.lstm = nn.LSTM(input_size=visual_dim, hidden_size=hidden_dim, 
                            num_layers=2, batch_first=True, dropout=0.2)
        
        # 5. Output Head
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, feats, imgs, mask):
        """
        feats: [B, T, 3] (x, y, yaw)
        imgs:  [B, T, C, H, W]
        mask:  [B, T]
        """
        B, T, C, H, W = imgs.shape
        
        # --- A. Feature Extraction & Fusion ---
        # 1. Visual
        if self.ablation_config.get('use_visual', True):
            imgs_reshaped = imgs.view(B * T, C, H, W)
            vis_feats = self.visual_extractor(imgs_reshaped) 
            vis_feats = vis_feats.view(B, T, -1)             
            vis_feats = self.visual_proj(vis_feats)          # [B, T, visual_dim]
        else:
            vis_feats = torch.zeros(B, T, self.visual_proj.out_features, device=feats.device)

        # 2. Kinematics
        if self.ablation_config.get('use_kinematics', True):
            kin_feats = self.kinematic_proj(feats)           # [B, T, visual_dim]
        else:
            kin_feats = torch.zeros(B, T, self.visual_proj.out_features, device=feats.device)
            
        # 3. Early Fusion (Residual connection)
        fused_input = vis_feats + kin_feats 
        
        # --- B. Multi-Scale Temporal Processing (The NMI Highlight) ---
        # Permute for Conv1d: [B, T, Dim] -> [B, Dim, T]
        x_in = fused_input.permute(0, 2, 1)
        
        # Parallel processing
        c1 = F.relu(self.conv_fast(x_in)) # Fast Dynamics
        c2 = F.relu(self.conv_med(x_in))  # Medium Dynamics
        c3 = F.relu(self.conv_slow(x_in)) # Slow Dynamics
        
        # Concat & Fuse
        c_cat = torch.cat([c1, c2, c3], dim=1) # [B, 3*(Dim/4), T]
        c_cat = c_cat.permute(0, 2, 1)         # [B, T, Combined_Dim]
        
        ms_out = self.ms_fusion(c_cat)         # [B, T, visual_dim]
        ms_out = self.ms_norm(ms_out + fused_input) # Residual + Norm
        
        # --- C. Recurrent Dynamics (LSTM) ---
        # Pack sequence
        lengths = mask.sum(dim=1).cpu().long()
        # Handle case where length is 0 (though dataset should prevent this)
        lengths = torch.clamp(lengths, min=1) 
        
        packed_in = nn.utils.rnn.pack_padded_sequence(ms_out, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_in)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        
        # --- D. Prediction ---
        preds = self.fc(lstm_out)
        
        return preds, lstm_out