# -*- coding: utf-8 -*-
# model/cann_analysis.py

import numpy as np
import sys
import os

# 尝试导入 cann_model
# 如果 cann_base.py 在 model/ 下 (推荐):
try:
    from .cann_base import cann_model
except ImportError:
    # 如果 cann_base.py 在根目录下 (兼容旧结构):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cann_base import cann_model

class CannArgs:
    """参数容器，用于传递给 CANN 模型"""
    def __init__(self, alpha=1.0, tau=10.0, a=0.5, k=0.1, A=1.0, N=100, eps=0.05):
        self.alpha = alpha  # 时间尺度因子
        self.tau = tau      # 基准时间常数
        self.a = a          # 空间宽度
        self.k = k          # 抑制强度
        self.A = A          # 输入强度
        self.N = N          # 神经元数量
        self.eps = eps      # 最小时间常数比例

def run_single_simulation(alpha, gt_traj, tau0=20.0, noise_std=0.2, dt=0.1, spatial_a=0.5):
    """
    运行单次 CANN 跟踪仿真
    
    Args:
        alpha: 时间尺度因子
        gt_traj: Ground Truth 轨迹数组
        tau0: 基准时间常数
        noise_std: 输入噪声强度
        dt: 仿真步长
        spatial_a: 空间交互宽度 (用于 Lambda 扫描)
        
    Returns:
        pred_traj: 预测的轨迹数组
    """
    # 1. 初始化参数
    args = CannArgs(alpha=alpha, tau=tau0, a=spatial_a)
    model = cann_model(args)
    
    # 2. 状态预热
    model.set_input(args.A, gt_traj[0])
    model.u = model.input.copy()
    
    preds = []
    
    # 3. 仿真循环
    for x in gt_traj:
        # 加入感知噪声 (Perception Noise)
        noisy_x = x + np.random.randn() * noise_std
        model.set_input(args.A, noisy_x)
        
        # 显式欧拉更新
        dudt = model.get_dudt(0, model.u)
        model.u += dudt * dt
        
        # 解码位置
        preds.append(model.cm_of_u())
        
    return np.array(preds)

def calculate_metrics(pred, gt, dt):
    """
    计算 RMSE 和 Lag 指标
    """
    # 1. RMSE (Accuracy)
    rmse = np.sqrt(np.mean((pred - gt)**2))
    
    # 2. Lag (Phase Delay)
    # 使用互相关计算滞后
    xcorr = np.correlate(pred - pred.mean(), gt - gt.mean(), mode="full")
    peak_idx = np.argmax(xcorr)
    center_idx = len(gt) - 1
    lag_steps = peak_idx - center_idx
    lag_time = np.abs(lag_steps * dt)
    
    return rmse, lag_time

def run_alpha_sweep(alphas, gt_traj, **kwargs):
    """
    批量执行 Alpha 扫描实验
    """
    results = {
        'alphas': alphas,
        'rmse': [],
        'lag': []
    }
    
    print(f"Running Alpha Sweep on {len(alphas)} points...")
    for alpha in alphas:
        pred = run_single_simulation(alpha, gt_traj, **kwargs)
        rmse, lag = calculate_metrics(pred, gt_traj, kwargs.get('dt', 0.1))
        
        results['rmse'].append(rmse)
        results['lag'].append(lag)
        
    return results

def run_spatial_sweep(lambdas, gt_traj, fixed_alpha=0.5, **kwargs):
    """
    批量执行空间尺度 (Lambda/a) 扫描实验
    """
    rmse_list = []
    print(f"Running Spatial Sweep on {len(lambdas)} points...")
    for a_val in lambdas:
        pred = run_single_simulation(fixed_alpha, gt_traj, spatial_a=a_val, **kwargs)
        rmse, _ = calculate_metrics(pred, gt_traj, kwargs.get('dt', 0.1))
        rmse_list.append(rmse)
        
    return {'lambdas': lambdas, 'rmse': rmse_list}