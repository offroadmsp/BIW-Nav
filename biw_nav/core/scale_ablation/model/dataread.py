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

from model import dataread,evalcell,evaltraj,plotcell,visualNet

# --- Dataset Definition (Cell 2) ---
class VariableLengthTrajectoryDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_yaw=True, min_len=1):
        self.root_dir = root_dir
        self.use_yaw = use_yaw
        self.min_len = min_len
        self.trajectories = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.transform = transform or transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])
        print(f"Found {len(self.trajectories)} trajectories in {root_dir}")

        # 🧩 自动识别输出维度 (Position dimension only for target)
        if not self.trajectories:
             raise ValueError(f"No trajectories found in {root_dir}")

        sample_traj_path = os.path.join(self.trajectories[0], "traj_data.pkl")
        try:
            with open(sample_traj_path, 'rb') as f:
                sample_data = pickle.load(f)
        except FileNotFoundError:
             raise FileNotFoundError(f"traj_data.pkl not found in the first trajectory: {self.trajectories[0]}")
        except Exception as e:
            raise RuntimeError(f"Error loading sample pkl file: {e}")

        # Target dimension is usually just the position dimension
        self.target_dim = len(sample_data['position'][0])
        # Input feature dimension includes position and optionally yaw
        self.input_feature_dim = self.target_dim + (1 if self.use_yaw and 'yaw' in sample_data else 0)


        print(f"Loaded {len(self.trajectories)} trajectories from {root_dir}")
        print(f"Detected target_dim (position) = {self.target_dim}")
        print(f"Detected input_feature_dim (position + yaw={self.use_yaw}) = {self.input_feature_dim}")


    def __len__(self):
        return len(self.trajectories)

    def _load_traj(self, traj_path):
        pkl_path = os.path.join(traj_path, "traj_data.pkl")
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: traj_data.pkl not found in {traj_path}. Skipping.")
            return None, None
        except Exception as e:
            print(f"Warning: Error loading {pkl_path}: {e}. Skipping.")
            return None, None

        pos = np.array(data.get('position', []), dtype=np.float32)
        yaw_data = data.get('yaw')
        # Handle potential scalar yaw values if they occur
        if yaw_data is not None:
             if isinstance(yaw_data, (int, float)):
                 # If it's a single scalar, assume it applies to the whole trajectory (less likely)
                 # or perhaps only the first step. More robustly, assume it's per step.
                 # If len(pos) > 0, replicate; otherwise, handle empty case.
                 if len(pos) > 0:
                     yaw = np.full(len(pos), float(yaw_data), dtype=np.float32)
                 else:
                     yaw = np.array([], dtype=np.float32)

             elif isinstance(yaw_data, (list, np.ndarray)):
                 try:
                     # Attempt conversion, handling potential nested structures if needed
                     yaw = np.array(yaw_data, dtype=np.float32).flatten() # Flatten in case of nested lists/arrays
                 except ValueError:
                     print(f"Warning: Could not convert yaw data in {pkl_path} to float32 array. Skipping yaw.")
                     yaw = None
             else:
                  print(f"Warning: Unexpected type for yaw data in {pkl_path}: {type(yaw_data)}. Skipping yaw.")
                  yaw = None
        else:
            yaw = None


        if pos.size == 0:
            print(f"Warning: No position data found in {pkl_path}. Skipping.")
            return None, None

        # Ensure yaw has the same length as pos if it exists
        if yaw is not None and len(yaw) != len(pos):
             print(f"Warning: Mismatch between position ({len(pos)}) and yaw ({len(yaw)}) lengths in {pkl_path}. Using min length or skipping yaw.")
             min_len_traj = min(len(pos), len(yaw))
             if min_len_traj > 0:
                 pos = pos[:min_len_traj]
                 yaw = yaw[:min_len_traj]
             else:
                 # If lengths mismatch and one is zero, cannot proceed reliably
                 print(f"Skipping yaw due to length mismatch and potential zero length.")
                 yaw = None # Skip yaw if lengths mismatch significantly


        return pos, yaw


    def _load_imgs(self, traj_path):
        img_files = sorted([
            os.path.join(traj_path, f)
            for f in os.listdir(traj_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) # Added .jpeg
        ])
        imgs = []
        for f in img_files:
             try:
                 img = Image.open(f).convert('RGB')
                 imgs.append(self.transform(img))
             except Exception as e:
                 print(f"Warning: Could not load or transform image {f}: {e}. Skipping image.")
                 # Decide how to handle: either skip the whole trajectory or just this image.
                 # If skipping image, need to align pos/yaw later. For simplicity,
                 # let's return an empty list if any image fails, causing __getitem__ to handle it.
                 # return [] # Option 1: Skip trajectory if any image fails
                 pass # Option 2: Skip only the faulty image, requires alignment later

        return imgs


    def __getitem__(self, idx):
        while idx < len(self.trajectories): # Loop to find a valid trajectory
            traj_path = self.trajectories[idx]
            pos, yaw = self._load_traj(traj_path)

            if pos is None: # Trajectory loading failed
                 print(f"Skipping trajectory at index {idx} due to loading error: {traj_path}")
                 idx += 1 # Try next index
                 if idx == len(self.trajectories): return self.__getitem__(0) # Wrap around if needed, or raise error
                 continue # Go to next iteration of while loop

            imgs = self._load_imgs(traj_path)

            # Align lengths of pos, yaw (if used), and imgs
            n = len(pos)
            if self.use_yaw and yaw is not None:
                n = min(n, len(yaw))
            n = min(n, len(imgs))

            if n < self.min_len:
                 # print(f"Skipping trajectory {traj_path} (length {n} < min_len {self.min_len})")
                 idx += 1 # Try next index
                 if idx == len(self.trajectories): return self.__getitem__(0) # Wrap around or raise error
                 continue # Go to next iteration of while loop


            pos, imgs = pos[:n], imgs[:n]
            if self.use_yaw and yaw is not None:
                yaw = yaw[:n]
                # Ensure yaw has shape (n,) before adding dimension
                if yaw.ndim > 1:
                     yaw = yaw.flatten() # Or handle specific shape issues
                feats = np.concatenate([pos, yaw[:, None]], axis=1)
            else:
                feats = pos

            imgs_tensor = torch.stack(imgs)
            # Target is the final position (use actual pos dimension)
            target = torch.tensor(pos[-1, :self.target_dim], dtype=torch.float32)
            feats_tensor = torch.tensor(feats, dtype=torch.float32)

            return feats_tensor, imgs_tensor, target # Return the valid data

        # If loop finishes without finding a valid trajectory (should ideally not happen with wrap-around)
        raise IndexError(f"Could not find a valid trajectory after checking index {idx} onwards.")
