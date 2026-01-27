import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# -----------------------------
# 参数设置
# -----------------------------
num_cells = 20
T = 10.0
dt = 0.01  # LSTM 不需要太小步长，降低计算量
time = np.arange(0, T, dt)
num_steps = len(time)

# 时间细胞峰值激活
peak_times = np.linspace(1, 9, num_cells)
input_strength = 1.0

# -----------------------------
# 模拟群体时间细胞活动（高斯激活）
# -----------------------------
spikes = np.zeros((num_cells, num_steps))
for i in range(num_cells):
    spikes[i] = input_strength * np.exp(-(time - peak_times[i])**2 / (2*0.2**2))

# 转置为 (num_steps, num_cells)
data = spikes.T.astype(np.float32)
target = time.astype(np.float32)

# -----------------------------
# 构建 PyTorch 数据集
# -----------------------------
seq_len = 10  # LSTM 输入序列长度
X, Y = [], []
for i in range(num_steps - seq_len):
    X.append(data[i:i+seq_len])
    Y.append(target[i+seq_len])
X = np.stack(X)
Y = np.array(Y).reshape(-1,1)

dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------------
# 构建 LSTM 模型
# -----------------------------
class TimeCellLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:,-1,:])  # 取最后时间步输出
        return out

model = TimeCellLSTM(input_size=num_cells, hidden_size=50)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 训练 LSTM
# -----------------------------
epochs = 100
for epoch in range(epochs):
    for xb, yb in dataloader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# -----------------------------
# 时间预测
# -----------------------------
with torch.no_grad():
    X_all = torch.from_numpy(data[np.newaxis,:-seq_len]).float().permute(1,0,2)
    # 使用滑动窗口预测
    preds = []
    for i in range(num_steps - seq_len):
        inp = torch.from_numpy(data[i:i+seq_len][np.newaxis,:,:]).float()
        pred = model(inp)
        preds.append(pred.item())
preds = np.array(preds)
time_true = target[seq_len:]

# -----------------------------
# 绘图
# -----------------------------
# 群体活动热图
plt.figure(figsize=(8,6))
plt.imshow(spikes, aspect='auto', extent=[0, T, 1, num_cells], origin='lower', cmap='hot')
plt.colorbar(label='Activity')
plt.xlabel('Time (s)')
plt.ylabel('Time Cell Index')
plt.title('Population Activity of Time Cells')
plt.show()

# LSTM 解码结果
plt.figure(figsize=(10,4))
plt.plot(time_true, time_true, 'k--', label='True Time')
plt.plot(time_true, preds, 'r', label='Decoded Time (LSTM)')
plt.xlabel('Actual Time (s)')
plt.ylabel('Decoded Time (s)')
plt.title('Time Decoding from Population Activity using LSTM')
plt.legend()
plt.show()
