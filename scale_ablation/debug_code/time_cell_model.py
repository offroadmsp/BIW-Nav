import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# 参数设置
# -----------------------------
num_cells = 20           # 时间细胞数量
T = 10.0                 # 总时间 (秒)
dt = 0.001               # 时间步长
time = np.arange(0, T, dt)

# LIF 模型参数
V_rest = -65.0
V_th = -50.0
V_reset = -65.0
tau_m = 0.02
tau_a = 0.5
g_adapt = 5.0
input_strength = 20.0

# 每个细胞的峰值激活时间
peak_times = np.linspace(1, 9, num_cells)

# -----------------------------
# 初始化变量
# -----------------------------
V = np.ones((num_cells, len(time))) * V_rest
a = np.zeros((num_cells, len(time)))
spikes = np.zeros((num_cells, len(time)))

# -----------------------------
# 模拟群体时间细胞
# -----------------------------
for i in range(num_cells):
    for t in range(1, len(time)):
        # 高斯输入激活
        I_ext = input_strength * np.exp(-(time[t] - peak_times[i])**2 / (2*0.2**2))
        dV = (-(V[i, t-1] - V_rest) + I_ext - g_adapt*a[i, t-1]) * (dt / tau_m)
        V[i, t] = V[i, t-1] + dV
        da = (-a[i, t-1]) * dt / tau_a
        a[i, t] = a[i, t-1] + da
        if V[i, t] >= V_th:
            spikes[i, t] = 1
            V[i, t] = V_reset
            a[i, t] += 1.0

# -----------------------------
# 群体活动可视化
# -----------------------------
plt.figure(figsize=(8, 6))
plt.imshow(spikes, aspect='auto', extent=[0, T, 1, num_cells], origin='lower', cmap='hot')
plt.colorbar(label='Spike')
plt.xlabel('Time (s)')
plt.ylabel('Time Cell Index')
plt.title('Population Raster Plot of Time Cells')
plt.show()

# -----------------------------
# 线性时间解码
# -----------------------------
# 使用滑动窗口计算瞬时发放率
window_size = int(0.05 / dt)  # 50ms 窗口
firing_rates = np.array([np.convolve(spikes[i], np.ones(window_size)/window_size, mode='same') 
                         for i in range(num_cells)]).T

# 训练线性回归解码器
decoder = LinearRegression()
decoder.fit(firing_rates, time)

# 预测时间
time_hat = decoder.predict(firing_rates)

# -----------------------------
# 绘制解码结果
# -----------------------------
plt.figure(figsize=(10, 4))
plt.plot(time, time, 'k--', label='True Time')
plt.plot(time, time_hat, 'r', label='Decoded Time')
plt.xlabel('Actual Time (s)')
plt.ylabel('Decoded Time (s)')
plt.title('Time Decoding from Population Activity')
plt.legend()
plt.show()
