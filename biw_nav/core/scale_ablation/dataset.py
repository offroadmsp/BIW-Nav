import os
import pickle
import numpy as np
import re
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def sort_key(path):
    """
    从文件名中提取数字用于排序。
    例如: '.../10.jpg' -> 10, 确保 10.jpg 排在 2.jpg 后面。
    """
    nums = re.findall(r'\d+', os.path.basename(path))
    return int(nums[-1]) if nums else 0

class VariableLengthTrajectoryDataset(Dataset):
    """
    可变长度轨迹数据集：
    每个文件夹包含一个 traj_data.pkl 和对应图像序列 (*.jpg)。
    支持变长序列，在 collate_fn 中自动 padding。
    """
    def __init__(self, root_dir, transform=None, use_yaw=True, min_len=3):
        self.root_dir = root_dir
        self.use_yaw = use_yaw
        self.min_len = min_len
        
        # 检查根目录是否存在
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

        self.trajectories = sorted([
            os.path.join(root_dir, d)
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        
        # 默认变换
        self.transform = transform or transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])
        print(f"Found {len(self.trajectories)} trajectories in {root_dir}")

    def __len__(self):
        return len(self.trajectories)

    def _load_traj(self, traj_path):
        pkl_path = os.path.join(traj_path, "traj_data.pkl")
        
        # 增加健壮性检查
        if not os.path.exists(pkl_path):
            print(f"Warning: {pkl_path} not found, skipping trajectory.")
            return None, None
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        pos = np.array(data['position'], dtype=np.float32)
        yaw = np.array(data['yaw'], dtype=np.float32)
        return pos, yaw

    def _load_imgs(self, traj_path):
        # [核心修改] 参考 notebook 的读取方式
        # 1. 直接在 traj_path 下寻找 jpg 文件，而不是 traj_path/images
        img_files = glob(os.path.join(traj_path, '*.jpg'))
        
        # 2. 如果没有jpg，尝试png
        if not img_files:
            img_files = glob(os.path.join(traj_path, '*.png'))
            
        # 3. 使用数字 key 进行排序，防止顺序错乱
        img_files = sorted(img_files, key=sort_key)
        
        imgs = [self.transform(Image.open(f).convert('RGB')) for f in img_files]
        return imgs

    def __getitem__(self, idx):
        traj_path = self.trajectories[idx]
        
        pos, yaw = self._load_traj(traj_path)
        if pos is None: 
            # 如果数据损坏，递归尝试下一个数据（避免报错中断）
            return self.__getitem__((idx + 1) % len(self))
            
        imgs = self._load_imgs(traj_path)

        # 对齐长度：取位置数据和图片数量的最小值
        n = min(len(pos), len(imgs))
        pos, yaw, imgs = pos[:n], yaw[:n], imgs[:n]

        # 过滤过短的轨迹
        if n < self.min_len:
            # 同样跳过处理
            return self.__getitem__((idx + 1) % len(self))

        # 构建特征向量
        if self.use_yaw:
            # [n, 3] -> x, y, yaw
            feats = np.concatenate([pos, yaw[:, None]], axis=1)  
        else:
            feats = pos

        imgs_tensor = torch.stack(imgs)
        # Target: 通常是预测序列的最后位置，或下一个位置
        target = torch.tensor(pos[-1], dtype=torch.float32)
        feats_tensor = torch.tensor(feats, dtype=torch.float32)
        
        return feats_tensor, imgs_tensor, target

def collate_variable_length(batch):
    """
    动态padding批处理函数：
    - 对不同长度序列进行0填充；
    - 返回 mask 用于LSTM有效步长计算。
    """
    # 过滤掉 None
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None, None, None, None

    feats, imgs, targets = zip(*batch)
    lengths = [f.shape[0] for f in feats]
    max_len = max(lengths)

    # Padding
    B = len(batch)
    feat_dim = feats[0].shape[1]
    img_shape = imgs[0].shape[1:] # [C, H, W]
    
    feats_padded = torch.zeros(B, max_len, feat_dim)
    imgs_padded = torch.zeros(B, max_len, *img_shape)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (f, im) in enumerate(zip(feats, imgs)):
        L = f.shape[0]
        feats_padded[i, :L] = f
        imgs_padded[i, :L] = im
        mask[i, :L] = True

    targets = torch.stack(targets)
    return feats_padded, imgs_padded, targets, mask