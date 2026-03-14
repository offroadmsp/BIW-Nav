#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Standard library imports
import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import importlib.util
import time

# Own module imports
import analyse
import world
import analyse_cpu
import plot

# 配置matplotlib
plt.rcParams['text.usetex'] = True

# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)

# ==================== CUDA配置 ====================
# 自动检测并使用最佳设备
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

print("=" * 60)
print("TEM Model Testing with CUDA Support")
print("=" * 60)
print(f"Device: {DEVICE}")
if USE_CUDA:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
print("=" * 60 + "\n")

# ==================== 模型加载配置 ====================
# 选择要加载的训练模型
date = '2024-12-22'
run = '1'
index = '39000'

print(f"Loading model: {date}, run {run}, iteration {index}")

# 构建路径
summary_path = f'./Summaries/{date}/run{run}'
script_path = f'{summary_path}/script'
model_path = f'{summary_path}/model'
envs_path = f'{script_path}/envs'

# ==================== 加载模型 ====================
# 动态导入模型模块
model_spec = importlib.util.spec_from_file_location(
    "model", f'{script_path}/model.py'
)
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)

# 加载模型参数（直接加载到目标设备）
params = torch.load(
    f'{model_path}/params_{index}.pt',
    map_location=DEVICE  # 关键：直接加载到目标设备
)

print("Parameters loaded")

# ==================== 移动超参数到设备 ====================
def move_params_to_device(params, device):
    """将张量超参数移动到指定设备"""
    tensor_params = ['W_tile', 'W_repeat', 'g_downsample', 'two_hot_table', 
                    'p_update_mask', 'p_retrieve_mask_inf', 'p_retrieve_mask_gen']
    
    for param_name in tensor_params:
        if param_name in params:
            param_value = params[param_name]
            if isinstance(param_value, list):
                params[param_name] = [
                    p.to(device) if torch.is_tensor(p) else p 
                    for p in param_value
                ]
            elif torch.is_tensor(param_value):
                params[param_name] = param_value.to(device)
    
    return params

params = move_params_to_device(params, DEVICE)
print("Parameters moved to device")

# ==================== 创建模型 ====================
# 创建TEM模型并移动到设备
tem = model_module.Model(params).to(DEVICE)
print("Model created and moved to device")

# 加载训练权重
model_weights = torch.load(
    f'{model_path}/tem_{index}.pt',
    map_location=DEVICE  # 关键：直接加载到目标设备
)
tem.load_state_dict(model_weights)
print("Model weights loaded")

# 设置为评估模式
tem.eval()
print("Model in evaluation mode\n")

# ==================== 准备环境和数据 ====================
# 加载训练时使用的环境
envs = list(glob.iglob(f'{envs_path}/*'))
print(f"Found {len(envs)} environment files")

# 设置哪些环境包含闪亮物体
shiny_envs = [False, False, True, True]
n_walks = len(shiny_envs)

# 创建环境
print("Creating environments...")
environments = [
    world.World(
        np.random.choice(envs), 
        randomise_observations=True, 
        shiny=(params['shiny'] if shiny_envs[env_i] else None)
    ) 
    for env_i in range(n_walks)
]

# 确定步行长度
walk_len = np.median([env.n_locations * 50 for env in environments]).astype(int)
print(f"Walk length: {walk_len}")

# 生成步行数据
print("Generating walks...")
walks = [env.generate_walks(walk_len, 1)[0] for env in environments]

# ==================== 准备模型输入 ====================
print("Preparing model input...")
# 将所有环境的步骤组合在一起以并行输入模型
model_input = [
    [
        [walks[i][j][k] for i in range(len(walks))] 
        for k in range(3)
    ] 
    for j in range(walk_len)
]

# 堆叠观察张量并移动到设备
for i_step, step in enumerate(model_input):
    model_input[i_step][1] = torch.stack(step[1], dim=0).to(DEVICE)  # 移动到设备

print(f"Model input prepared, batch size: {model_input[0][1].shape[0]}")

# ==================== 运行前向传播 ====================
print("\nRunning forward pass...")
start_time = time.time()

with torch.no_grad():  # 禁用梯度计算以节省内存和加速
    forward = tem(model_input, prev_iter=None)

forward_time = time.time() - start_time
print(f"Forward pass completed in {forward_time:.2f} seconds")
print(f"Average time per step: {forward_time/walk_len*1000:.2f} ms")

# ==================== 分析结果 ====================
# 注意：analyse模块中的函数可能需要CPU上的数据
# 我们在传递给analyse函数之前将数据移回CPU

print("\nAnalyzing results...")

# 决定是否包含原地不动的动作
include_stay_still = True

print("Comparing to agents...")
# 比较训练模型与节点代理和边缘代理的性能
correct_model, correct_node, correct_edge = analyse.compare_to_agents(
    forward, tem, environments, include_stay_still=include_stay_still
)

print("Analyzing zero-shot inference...")
# 分析零样本推理
zero_shot = analyse.zero_shot(
    forward, tem, environments, include_stay_still=include_stay_still
)

print("Computing location occupation...")
# 生成占用图
occupation = analyse.location_occupation(forward, tem, environments)

print("Computing rate maps...")
# 生成速率图
g, p = analyse.rate_map(forward, tem, environments)
print("Computing location accuracy...")
# 计算到达和离开每个位置的准确率
from_acc, to_acc = analyse.location_accuracy(forward, tem, environments)

# ==================== 可视化结果 ====================
print("\nGenerating plots...")

# 选择要绘制的环境
env_to_plot = 0
# 平均环境时，决定包含哪些环境
envs_to_avg = shiny_envs if shiny_envs[env_to_plot] else [not s for s in shiny_envs]

# ========== 图1: 代理比较和零样本推理 ==========
print("Plotting agent comparison...")
filt_size = 41
plt.figure(figsize=(3.5, 2.2), dpi=600)

# 计算平均准确率
avg_model = np.mean([env for env_i, env in enumerate(correct_model) if envs_to_avg[env_i]], 0)[1:]
avg_node = np.mean([env for env_i, env in enumerate(correct_node) if envs_to_avg[env_i]], 0)[1:]
avg_edge = np.mean([env for env_i, env in enumerate(correct_edge) if envs_to_avg[env_i]], 0)[1:]

plt.plot(analyse_cpu.smooth(avg_model, filt_size), label='BIW', linewidth=1.5)
plt.plot(analyse_cpu.smooth(avg_node, filt_size), label='node', linewidth=1.5)
plt.plot(analyse_cpu.smooth(avg_edge, filt_size), label='edge', linewidth=1.5)
plt.ylim(0, 1)
plt.legend(frameon=False)
plt.xlabel('Indexed step')
plt.ylabel('Precision')

# 计算零样本推理平均值
zs_avg = np.mean([np.mean(env) for env_i, env in enumerate(zero_shot) if envs_to_avg[env_i]]) * 100
plt.title(f'Zero-shot inference: {zs_avg:.1f}%', fontdict={'family':'Arial', 'size':10})
plt.tight_layout()
plt.savefig('./Pics/fig_infer1.png', dpi=600, bbox_inches='tight')
print("  Saved: ./Pics/fig_infer1.png")

# ========== 图2: 所有细胞的速率图 ==========
print("Plotting cell rate maps...")
n_f_ovc = params.get('n_f_ovc', 0)
plot.plot_cells(
    p[env_to_plot], 
    g[env_to_plot], 
    environments[env_to_plot], 
    n_f_ovc=n_f_ovc, 
    columns=25
)

# ========== 图3: 位置准确率 ==========
print("Plotting location accuracy...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3), dpi=600)

# 绘制两个子图
plot.plot_map(
    environments[env_to_plot], 
    np.array(to_acc[env_to_plot]), 
    ax1
)
ax1.set_title('Accuracy when arriving location', fontsize=9)

plot.plot_map(
    environments[env_to_plot], 
    np.array(from_acc[env_to_plot]), 
    ax2
)
ax2.set_title('Accuracy when leaving location', fontsize=9)

# 设置整体标题
fig.suptitle('Location Accuracy', fontdict={'family':'Arial', 'size':10})
plt.tight_layout()
plt.savefig('./Pics/fig_location1.png', dpi=600, bbox_inches='tight')
print("  Saved: ./Pics/fig_location1.png")

# ========== 图4: 占用图和步行路径 ==========
print("Plotting occupation and walk...")
plt.figure(figsize=(4, 3), dpi=300)

# 计算归一化的占用率
norm_occupation = np.array(occupation[env_to_plot]) / sum(occupation[env_to_plot])
norm_occupation *= environments[env_to_plot].n_locations

# 绘制占用图
ax = plot.plot_map(
    environments[env_to_plot], 
    norm_occupation,
    min_val=0, 
    max_val=2, 
    ax=None, 
    shape='square', 
    radius=1/np.sqrt(environments[env_to_plot].n_locations)
)

# 在占用图上绘制步行路径
n_steps = max(1, int(len(walks[env_to_plot]) / 500))
ax = plot.plot_walk(
    environments[env_to_plot], 
    walks[env_to_plot], 
    ax=ax, 
    n_steps=n_steps
)

plt.title('Walk and average occupation', fontdict={'family':'Arial', 'size':10})
plt.tight_layout()
plt.savefig('./Pics/fig_occupation1.png', dpi=600, bbox_inches='tight')
print("  Saved: ./Pics/fig_occupation1.png")

# ==================== 显示所有图像 ====================
plt.show()

# ==================== 性能统计 ====================
print("\n" + "=" * 60)
print("Performance Summary")
print("=" * 60)
print(f"Device used: {DEVICE}")
print(f"Total steps: {walk_len}")
print(f"Forward pass time: {forward_time:.2f} seconds")
print(f"Average time per step: {forward_time/walk_len*1000:.2f} ms")
print(f"Zero-shot inference: {zs_avg:.2f}%")

if USE_CUDA:
    print(f"\nGPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

print("=" * 60)
print("\n✓ Testing completed successfully!")
