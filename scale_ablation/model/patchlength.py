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

# --- Collate Function (Cell 3) ---
def collate_variable_length(batch):
    # Filter out None items that might result from errors in __getitem__
    batch = [item for item in batch if item is not None]
    if not batch: # If all items in the batch failed
        return None, None, None, None # Or raise an error

    feats, imgs, targets = zip(*batch)

    # Check if lists are empty after filtering
    if not feats or not imgs or not targets:
        return None, None, None, None # Or raise an error

    lengths = [f.shape[0] for f in feats]
    max_len = max(lengths) if lengths else 0

    if max_len == 0: # Handle case where all valid trajectories have zero length (unlikely)
         return None, None, None, None

    feat_dim = feats[0].shape[1] if feats else 0
    img_shape = imgs[0].shape[1:] if imgs else (0, 0, 0) # Use shape directly

    feats_padded = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    imgs_padded = torch.zeros(len(batch), max_len, *img_shape, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, (f, im) in enumerate(zip(feats, imgs)):
        L = f.shape[0]
        if L > 0: # Ensure sequence length is positive
            feats_padded[i, :L] = f
            imgs_padded[i, :L] = im
            mask[i, :L] = True

    # Ensure targets are stacked correctly even if some items were filtered
    # targets list might contain tensors of different shapes if __getitem__ logic changes target dim
    # Assuming target is always the last position (fixed dim), stacking should work.
    try:
        targets = torch.stack(targets)
    except RuntimeError as e:
        print(f"Error stacking targets: {e}")
        # Handle error: maybe return None or pad targets if shape mismatch is expected
        return None, None, None, None


    return feats_padded, imgs_padded, targets, mask