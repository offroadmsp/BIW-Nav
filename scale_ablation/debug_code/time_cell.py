import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 自定义LSTM，提取门控信息
# -----------------------------
class AnalyzableLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size, 4 * hidden_size)
        self.U = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x, h_prev, c_prev):
        gates = self.W(x) + self.U(h_prev)
        i, f, g, o = gates.chunk(4, dim=1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c, f  # 返回遗忘门 f_t

# -----------------------------
# 运行仿真
# -----------------------------
input_size = 3
hidden_size = 20
T = 100

cell = AnalyzableLSTMCell(input_size, hidden_size)
x = torch.randn(T, input_size)
h = torch.zeros(1, hidden_size)
c = torch.zeros(1, hidden_size)

forget_values = []
for t in range(T):
    h, c, f = cell(x[t].unsqueeze(0), h, c)
    forget_values.append(f.detach().numpy())

forget_values = np.concatenate(forget_values, axis=0)

# -----------------------------
# 计算时间常数
# -----------------------------
mean_forget = forget_values.mean(axis=0)
timescales = -1 / np.log(mean_forget + 1e-6)  # 避免 log(0)

# -----------------------------
# 可视化
# -----------------------------
plt.figure(figsize=(10,4))
plt.bar(np.arange(hidden_size), timescales)
plt.xlabel('LSTM Unit Index')
plt.ylabel('Effective Timescale (τ)')
plt.title('Estimated Timescales of LSTM Time Cells')
plt.savefig('./results/scale_timescales.png')



plt.figure(figsize=(8,6))
plt.imshow(forget_values.T, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Forget Gate Value')
plt.xlabel('Time step')
plt.ylabel('Unit index')
plt.title('Forget Gate Activation Over Time')
plt.savefig('./results/cell_forgets.png')


plt.show()
