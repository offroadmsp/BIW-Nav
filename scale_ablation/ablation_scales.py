import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint
from cann_base import cann_model  # 确保 cann_base.py 在同一目录下

class MockArgs:
    """模拟命令行参数类"""
    def __init__(self, k=0.5, a=0.5, N=128, A=0.5, z0=0.5*np.pi):
        self.k = k
        self.a = a
        self.N = N
        self.A = A
        self.z0 = z0

def run_single_cann(args, duration=50):
    """运行单个 CANN 模型的仿真并返回最终误差"""
    model = cann_model(args)
    
    # 1. 初始化
    if args.k < 1.0:
        model.set_input(np.sqrt(32.0)/args.k, 0)
    else:
        model.set_input(np.sqrt(32.0), 0)
    model.u = model.input
    
    # 2. 预热
    spint.solve_ivp(model.get_dudt, (0, 20), model.u, method="RK45")
    
    # 3. 刺激跳变 (0 -> z0)
    model.set_input(args.A, args.z0)
    
    # 4. 运行仿真
    # 这里我们只关心收敛速度或最终稳态误差
    t_span = (0, duration)
    out = spint.solve_ivp(model.get_dudt, t_span, model.u, method="RK45")
    
    # 获取最终状态的中心位置
    final_u = out.y[:, -1]
    model.u = final_u
    final_pos = model.cm_of_u()
    
    # 计算误差 (目标位置 - 实际位置)
    error = np.abs(final_pos - args.z0)
    return error, out.t, out.y

def experiment_multi_scale_impact():
    print("=== Running Ablation: Impact of Multi-Scale Representation ===")
    
    # 设置：单尺度 vs 多尺度 (通过改变宽度参数 'a' 来模拟不同尺度)
    # Scale 1: a=0.3 (精细), Scale 2: a=0.5 (中等), Scale 3: a=0.8 (粗糙)
    scales = [0.3, 0.5, 0.8]
    errors = []
    
    plt.figure(figsize=(12, 4))
    
    for i, a in enumerate(scales):
        args = MockArgs(a=a)
        err, t, y = run_single_cann(args)
        errors.append(err)
        print(f"Scale a={a}: Final Error = {err:.6f}")
        
        # 可视化不同尺度的响应波包
        plt.subplot(1, 3, i+1)
        plt.plot(y[:, -1])
        plt.title(f"Scale a={a}\nError={err:.4f}")
        plt.xlabel("Neuron Index")
        if i == 0: plt.ylabel("Activity u(x)")

    plt.tight_layout()
    plt.show()
    
    # 在这里，您可以进一步实现"多尺度融合"的逻辑
    # 例如：将多个 CANN 的输出加权求和来解码，对比仅使用单个 CANN 的解码精度
    # 由于 cann_base 是独立运行的，这里主要展示单一尺度的特性差异

def experiment_scale_factors():
    print("\n=== Running Ablation: Impact of Scale Factors ===")
    
    # 比较不同尺度增长因子的组
    # Group A: 密集 (Factor 1.2) -> a = [0.3, 0.36, 0.43]
    # Group B: 稀疏 (Factor 2.0) -> a = [0.3, 0.6, 1.2]
    
    base_a = 0.3
    factors = [1.2, 1.5, 2.0]
    
    results = {}
    
    for f in factors:
        current_scales = [base_a * (f**i) for i in range(3)]
        group_error = 0
        print(f"Testing Factor {f}: Scales {current_scales}")
        
        for a in current_scales:
            # 注意：对于非常大的 a，可能需要调整 N 或 z_range 以避免边界效应
            args = MockArgs(a=a, N=256) 
            err, _, _ = run_single_cann(args)
            group_error += err # 简单累加误差作为指标
            
        results[f] = group_error
        print(f"  -> Total Group Error: {group_error:.6f}")

    # 可视化结果
    plt.figure(figsize=(6, 4))
    plt.bar([str(f) for f in results.keys()], results.values())
    plt.xlabel("Scale Factor")
    plt.ylabel("Cumulative Tracking Error")
    plt.title("Impact of Scale Factors on Error")
    plt.show()

if __name__ == "__main__":
    experiment_multi_scale_impact()
    experiment_scale_factors()