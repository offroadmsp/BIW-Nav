import torch
from torch.utils.data import Dataset
import numpy as np

class PositionDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        
        # 加载数据
        self.positions, self.targets = self._load_data()
    
    def _load_data(self):
        # 这里可以根据实际数据格式进行加载
        # 示例：生成随机数据
        np.random.seed(42)
        num_samples = 10000 if self.split == 'train' else 2000
        positions = np.random.rand(num_samples, 2) * 10  # 位置范围 [0, 10]
        targets = positions + np.random.randn(num_samples, 2) * 0.1  # 添加一些噪声
        
        return torch.tensor(positions, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        return {
            'positions': self.positions[idx],
            'targets': self.targets[idx]
        }