import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 之前定义的类脑模型 (简化版引用)
# ==========================================
class BioInspiredEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, n_scales=3):
        super().__init__()
        # 模拟不同尺度的模块
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2) # 简单预测下一步位置(自监督)
        
    def forward(self, x):
        # x: [Batch, Seq, 2] (速度向量 v_x, v_y)
        output, (hn, cn) = self.lstm(x)
        pred_next_pos = self.head(output)
        return pred_next_pos, output

# ==========================================
# 2. 真实轨迹数据加载器
# ==========================================
class RealRatDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, seq_len=50):
        """
        加载真实老鼠轨迹，并转化为模型需要的 (速度, 位置) 对
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到轨迹文件: {csv_path}")
            
        print(f"[加载] 读取真实轨迹: {csv_path}")
        # 读取 pos (cm)
        self.pos = pd.read_csv(csv_path, header=None).values
        
        # 计算速度 (Action): v_t = p_{t+1} - p_t
        # 假设 50Hz，这里直接用位移作为速度特征
        self.velocity = np.diff(self.pos, axis=0)
        # 补一个 0 使得长度一致
        self.velocity = np.vstack([self.velocity, [0, 0]])
        
        # 归一化速度 (帮助网络收敛)
        self.velocity = self.velocity / (np.std(self.velocity) + 1e-6)
        
        self.seq_len = seq_len
        self.n_samples = len(self.pos) - seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 输入：一段连续的速度序列 (模拟老鼠的自我运动感知)
        # 目标：预测这段时间内的位移或位置 (路径积分任务)
        v_seq = self.velocity[idx : idx+self.seq_len]
        target_pos = self.pos[idx+1 : idx+self.seq_len+1] # 预测下一步
        
        return torch.FloatTensor(v_seq), torch.FloatTensor(target_pos)

# ==========================================
# 3. 训练与网格细胞可视化
# ==========================================
def train_and_visualize():
    # 配置
    DATA_PATH = "/media/zhen/Data/cellData/The_place_cell_representation_of_volumetric_space_in_rats/Summarydata/best_session_pos.csv"
    SEQ_LEN = 100
    HIDDEN_DIM = 128
    
    # 1. 准备数据
    dataset = RealRatDataset(DATA_PATH, seq_len=SEQ_LEN)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. 初始化模型
    model = BioInspiredEncoder(input_dim=2, hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    print(">>> 开始在真实生物轨迹上训练...")
    # 简短训练演示
    loss_history = []
    for epoch in range(5): 
        total_loss = 0
        for i, (v_batch, p_batch) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # 前向传播: 输入速度，预测位置变化
            # 这里我们做一个简单的自监督任务：路径积分
            pred, hidden_activity = model(v_batch)
            
            # 这里简化 Loss: 让模型学会根据速度推算位置变化
            # 实际 TEM 会更复杂 (预测 Sensory)
            # 这里我们只为了激活隐层神经元
            loss = torch.mean((pred - p_batch[:,:,:2])**2) # Dummy loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"    Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # ==========================================
    # 4. 关键步骤：提取隐层活性并绘制 Rate Map
    # ==========================================
    print(">>> 正在提取隐层神经元活性 (Grid Cell Analysis)...")
    
    # 不打乱顺序，完整跑一遍轨迹
    full_velocity = torch.FloatTensor(dataset.velocity).unsqueeze(0) # [1, T, 2]
    with torch.no_grad():
        _, activites = model(full_velocity) 
        # activites: [1, T, Hidden_Dim]
        
    rate_maps = activites[0].numpy() # [T, N_neurons]
    real_pos = dataset.pos
    
    # 画前 9 个神经元的 Rate Map
    plt.figure(figsize=(12, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        
        # 使用直方图累积法画 Rate Map (类似 mapDATA)
        heatmap, xedges, yedges = np.histogram2d(
            real_pos[:,0], real_pos[:,1], 
            bins=50, 
            weights=rate_maps[:, i] # 权重是神经元活性
        )
        # 归一化 (除以停留时间)
        occupancy, _, _ = np.histogram2d(real_pos[:,0], real_pos[:,1], bins=50)
        heatmap = heatmap / (occupancy + 1e-8)
        
        plt.imshow(heatmap.T, origin='lower', cmap='jet', interpolation='gaussian')
        plt.title(f"Neuron {i+1}")
        plt.axis('off')
        
    plt.suptitle(f"Emergent Grid-like Fields from Real Rat Trajectory\n(Trained on {len(real_pos)} steps)", fontsize=16)
    plt.show()
    print(">>> 完成！如果看到周期性的斑点，说明您的模型学会了空间结构。")

if __name__ == "__main__":
    train_and_visualize()