import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint
from cann_base import cann_model

class MockArgs:
    def __init__(self, k=0.5, a=0.5, N=128, A=0.5, z0=0.5*np.pi):
        self.k = k
        self.a = a
        self.N = N
        self.A = A
        self.z0 = z0

def generate_relation_bias(model, target_z0, strength=0.2):
    """
    生成模拟的认知关系偏置 (Cognitive Bias)。
    假设认知图谱预测刺激会移动到 target_z0，因此产生一个微弱的预激活场。
    """
    # 使用与外部输入相同的高斯形式，但强度较弱
    bias = strength * np.exp(-0.25 * np.square(model.dist(model.x - target_z0) / model.a))
    return bias

def run_cognitive_experiment(use_relation=True, noise_level=0.2):
    args = MockArgs(A=1.0) # 增强一点主信号
    model = cann_model(args)
    
    # 1. 初始化
    model.set_input(np.sqrt(32.0), 0)
    model.u = model.input
    spint.solve_ivp(model.get_dudt, (0, 20), model.u, method="RK45")
    
    # 2. 设置场景：含噪声的外部输入
    # 我们通过修改 set_input 逻辑或在 get_dudt 中注入噪声来模拟
    # 这里为了不修改 cann_base，我们在初始 set_input 后手动给 input 数组加噪声
    model.set_input(args.A, args.z0)
    np.random.seed(42)
    noise = np.random.normal(0, noise_level, size=model.N)
    model.input += noise # 注入噪声干扰
    
    # 3. 设置认知关系 (消融点)
    if use_relation:
        # 认知系统预测到了目标位置 (z0)，提供额外的稳定偏置
        bias = generate_relation_bias(model, args.z0, strength=0.5)
        model.set_relation_input(bias)
        label = "With Relation"
    else:
        # 无认知关系，仅靠含噪感官输入
        model.set_relation_input(None)
        label = "No Relation (Ablated)"
        
    # 4. 运行仿真
    t_span = (0, 50)
    out = spint.solve_ivp(model.get_dudt, t_span, model.u, method="RK45")
    
    # 5. 分析结果
    final_u = out.y[:, -1]
    model.u = final_u
    final_pos = model.cm_of_u()
    error = np.abs(final_pos - args.z0)
    
    return error, final_u, label

def main():
    print("=== Running Ablation: Impact of Cognitive Relations ===")
    
    noise_level = 0.5 # 较大的噪声，凸显认知关系的作用
    
    # 运行两组实验
    err_no_rel, u_no_rel, label_no = run_cognitive_experiment(use_relation=False, noise_level=noise_level)
    err_rel, u_rel, label_rel = run_cognitive_experiment(use_relation=True, noise_level=noise_level)
    
    print(f"{label_no}: Error = {err_no_rel:.4f}")
    print(f"{label_rel}: Error = {err_rel:.4f}")
    print(f"Improvement: {(err_no_rel - err_rel)/err_no_rel * 100:.2f}%")
    
    # 绘图对比
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(u_no_rel, label=label_no, color='gray', linestyle='--')
    plt.plot(u_rel, label=label_rel, color='red')
    plt.title(f"Network Activity (Noise={noise_level})")
    plt.legend()
    plt.xlabel("Neuron Index")
    
    plt.subplot(1, 2, 2)
    plt.bar([label_no, label_rel], [err_no_rel, err_rel], color=['gray', 'red'])
    plt.ylabel("Tracking Error")
    plt.title("Ablation Study Result")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()