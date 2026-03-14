import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

# ==============================
# 0. NMI 投稿级风格配置
# ==============================
def set_nmi_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 14          # 基础字号适中，适合独立子图
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==============================
# 1. 数据生成逻辑 (保持一致)
# ==============================
def generate_spatial_population(n_neurons=100, x_range=(-5, 5)):
    x = np.linspace(x_range[0], x_range[1], 800)
    lambdas = np.linspace(0.05, 1.0, n_neurons)
    sigmas = 0.15 + lambdas * 3.0
    population_activity = np.zeros((n_neurons, len(x)))
    center = 0.0
    for i, sigma in enumerate(sigmas):
        rate = np.exp(-0.5 * ((x - center) / sigma)**2)
        noise = np.random.normal(0, 0.08, size=len(x)) * (rate > 0.1)
        rate = np.clip(rate + noise, 0, 1)
        population_activity[i, :] = rate
    return x, lambdas, population_activity

def simulate_lif_trace_complex(alpha, duration=2000, dt=1.0):
    t = np.arange(0, duration, dt)
    tau = 20 + alpha * 780
    np.random.seed(42) 
    I_input = np.random.normal(0, 0.5, size=len(t))
    I_input += 1.5 * np.sin(2 * np.pi * t / 150) 
    I_input[300:700] += 3.0
    I_input[1200:1500] += 3.0
    I_input[1700:1800] -= 2.0
    V = np.zeros_like(t)
    v_curr = 0
    for i in range(1, len(t)):
        dv = (-(v_curr) + I_input[i]) / tau * dt
        v_curr += dv
        V[i] = v_curr
    return t, V, tau, I_input

def generate_grid_map(scale_lambda, size=100):
    x = np.linspace(-size/2, size/2, 250)
    y = np.linspace(-size/2, size/2, 250)
    XX, YY = np.meshgrid(x, y)
    period = 18 + scale_lambda * 70 
    k = 4 * np.pi / (period * np.sqrt(3))
    rate_map = np.zeros_like(XX)
    thetas = [0, np.pi/3, 2*np.pi/3]
    for theta in thetas:
        rate_map += np.cos(k * np.cos(theta) * XX + k * np.sin(theta) * YY)
    rate_map = np.exp(0.8 * rate_map)
    rate_map = (rate_map - rate_map.min()) / (rate_map.max() - rate_map.min())
    noise = np.random.normal(0, 0.15, size=rate_map.shape)
    rate_map = np.clip(rate_map + noise, 0, 1)
    from scipy.ndimage import gaussian_filter
    rate_map = gaussian_filter(rate_map, sigma=1.5)
    return XX, YY, rate_map

# ==============================
# 2. 独立的绘图函数
# ==============================

def save_panel_a():
    """ 生成 Panel A: 空间尺度图 """
    set_nmi_style()
    # 宽长比画布，专门适合展示频谱
    fig = plt.figure(figsize=(12, 6), layout='constrained')
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    
    # --- 左侧: 热力图 ---
    ax1 = fig.add_subplot(gs[0])
    x, lambdas, activity = generate_spatial_population()
    im1 = ax1.imshow(activity, aspect='auto', cmap='magma', 
                     extent=[x[0], x[-1], 0, 1], origin='lower')
    
    ax1.set_title(r"Continuous Spatial Scale ($\lambda$)", loc='left', pad=10)
    ax1.set_ylabel(r"Scale Parameter $\lambda$", fontweight='bold')
    ax1.set_xlabel("Spatial Position ($x$)")
    
    # 标注放在最右侧 Y 轴旁边，绝对不遮挡数据
    ax1.text(1.05, 0.95, 'Global\n(Low Freq)', transform=ax1.transAxes, 
             ha='left', va='top', fontsize=12, color='#333333')
    ax1.text(1.05, 0.05, 'Local\n(High Freq)', transform=ax1.transAxes, 
             ha='left', va='bottom', fontsize=12, color='#333333')
    
    # 细长的 colorbar
    cbar = plt.colorbar(im1, ax=ax1, fraction=0.04, pad=0.02)
    cbar.set_label('Firing Rate')

    # --- 右侧: 切片图 ---
    ax2 = fig.add_subplot(gs[1])
    idx_fine = 5; idx_coarse = 90
    ax2.plot(x, activity[idx_fine], color='#e74c3c', lw=2.5, alpha=0.9, label=r'Fine ($\lambda \to 0$)')
    ax2.fill_between(x, activity[idx_fine], color='#e74c3c', alpha=0.1)
    ax2.plot(x, activity[idx_coarse], color='#3498db', lw=2.5, alpha=0.9, label=r'Coarse ($\lambda \to 1$)')
    ax2.fill_between(x, activity[idx_coarse], color='#3498db', alpha=0.1)
    
    ax2.set_title("Receptive Fields", pad=10)
    ax2.set_xlabel("Position")
    ax2.set_yticks([])
    # Legend 放在顶部外部
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
    
    outfile = './plot/Panel_A_Spatial.pdf'
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    plt.close()

def save_panel_b():
    """ 生成 Panel B: 时间动态图 """
    set_nmi_style()
    # 扁长的画布，专门适合时间序列
    fig, ax = plt.subplots(figsize=(14, 6), layout='constrained')
    
    alphas = [0.1, 0.5, 0.9]
    colors = ['#2ecc71', '#f1c40f', '#9b59b6'] 
    labels = ['Fast / Reactive', 'Balanced', 'Slow / Integrative']
    
    # 绘制背景参考
    t_dummy, _, _, I_source = simulate_lif_trace_complex(0.5)
    ax.plot(t_dummy, (I_source - I_source.mean())*0.5 - 9, color='gray', alpha=0.3, ls='--', lw=1.5, label='Stimulus')
    
    for i, alpha in enumerate(alphas):
        t, V, tau, _ = simulate_lif_trace_complex(alpha)
        V_norm = (V - V.mean()) / (V.std() + 1e-6) 
        offset = i * 8.0 # 垂直间距拉大
        V_shifted = V_norm + offset
        
        ax.plot(t, V_shifted, color=colors[i], linewidth=2.5, 
                label=r'$\alpha=%.1f$ ($\tau \approx %d$ms)' % (alpha, tau))
        ax.fill_between(t, V_shifted, offset-2.5, color=colors[i], alpha=0.1)
        
        # 核心技巧：把文字放在 2100ms 之后，这是数据的空白区
        ax.text(2100, V_shifted[-1], labels[i], color=colors[i], 
                fontweight='bold', va='center', fontsize=14, ha='left')

    # ax.set_title(r"$\bf{b}$ | Continuous Temporal Dynamics ($\alpha$): Smoothing & Memory", loc='left', pad=15)
    ax.set_xlabel("Time (ms)")
    ax.set_yticks([])
    ax.set_ylabel("Membrane Potential (Offset)")
    
    # Legend 横向排列在顶部
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False)
    
    # 强制扩展 X 轴，给右侧文字留出绝对安全的 500ms 空间
    ax.set_xlim(0, 2600) 
    ax.spines['left'].set_visible(False)
    
    outfile = './plot/Panel_B_Temporal.pdf'
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    plt.close()

def save_panel_c():
    """ 生成 Panel C: 网格场 """
    set_nmi_style()
    # 宽画布，容纳三个正方形子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    
    sample_lambdas = [0.1, 0.5, 0.9]
    
    for i, ax in enumerate(axes):
        lam = sample_lambdas[i]
        XX, YY, rmap = generate_grid_map(lam)
        
        im = ax.imshow(rmap, extent=[-50, 50, -50, 50], origin='lower', cmap='viridis')
        
        ax.set_title(r"Scale $\lambda=%.1f$"%lam, fontsize=16, fontweight='bold', pad=10)
        ax.axis('off')
        
        # 比例尺: 增加半透明黑底，防止背景干扰
        if i == 0:
            ax.plot([-45, -25], [-45, -45], color='white', linewidth=5)
            # 使用 bbox 参数给文字加底色
            ax.text(-35, -40, '20cm', color='white', ha='center', fontsize=14, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=2))

    # 使用 figure text 添加总标题
    # fig.text(0.01, 0.95, r"$\bf{c}$ | Emergence of Multi-Scale Grid Fields", fontsize=18, va='top', ha='left')
    
    outfile = './plot/Panel_C_Grid.pdf'
    plt.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    plt.close()

if __name__ == "__main__":
    ensure_dir('./plot')
    print("Starting generation of separate panels...")
    save_panel_a()
    save_panel_b()
    save_panel_c()
    print("All panels generated successfully in ./plot folder.")