import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# -----------------------------
# 模拟参数
# -----------------------------
T = 100  # 时间步
input_size = 3   # 输入维度：模拟3种事件类型
hidden_size = 20 # LSTM隐状态数量（即“时间细胞”数量）
num_layers = 1
noise_level = 0.1  # 噪声强度

# -----------------------------
# 生成输入序列
# -----------------------------
np.random.seed(42)
x = np.zeros((T, input_size))
# 生成事件序列（例如：A→B→C）
x[10:20, 0] = 1.0  # 事件A
x[40:50, 1] = 1.0  # 事件B
x[70:80, 2] = 1.0  # 事件C

# 添加噪声
x += noise_level * np.random.randn(*x.shape)

# -----------------------------
# 转换为Torch张量
# -----------------------------
x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # (T, batch, input)

# -----------------------------
# 定义LSTM模型
# -----------------------------
class TimeCellLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, 1)  # 可用于预测未来事件或时间
    
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        y = self.fc(out)
        return out, y
    
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
# 实例化并前向传播
# -----------------------------
model = TimeCellLSTM(input_size, hidden_size, num_layers)
lstm_out, y_pred = model(x_tensor)

# -----------------------------
# 多时间尺度：人为加入慢速/快速细胞
# -----------------------------
timescales = np.linspace(0.1, 1.0, hidden_size)
scaled_out = []
for i in range(hidden_size):
    alpha = timescales[i]
    filtered = np.convolve(lstm_out[:,0,i].detach().numpy(), np.exp(-np.arange(10)*alpha), mode='same')
    scaled_out.append(filtered)
scaled_out = np.array(scaled_out)

# -----------------------------
# 绘制结果
# -----------------------------
plt.figure(figsize=(10, 6))
plt.imshow(scaled_out, aspect='auto', cmap='hot', origin='lower')
plt.colorbar(label='Activity')
plt.xlabel('Time Step')
plt.ylabel('LSTM Unit (Time Cell)')
plt.title('Time Cell-Like Activation in LSTM (with Noise & Multi-Timescale)')
# 保存图像到文件（必须在 plt.show() 之前调用！）
plt.savefig('./results/cell_activation.png')


# 绘制输入事件
plt.figure(figsize=(10, 2))
plt.plot(x)
plt.legend(['Event A', 'Event B', 'Event C'])
plt.title('Input Event Sequence (with Noise)')
plt.savefig('./results/input_event_sequence.png')


plt.figure(figsize=(10,4))
for i in [0, 5, 10, 15, 19]:
    plt.plot(scaled_out[i], label=f'Cell {i+1} (α={timescales[i]:.2f})')
plt.xlabel('Time step')
plt.ylabel('Filtered activity')
plt.title('Different Temporal Scales of Time Cells')
plt.legend()
plt.savefig('./results/cell_scale.png')


plt.show()
