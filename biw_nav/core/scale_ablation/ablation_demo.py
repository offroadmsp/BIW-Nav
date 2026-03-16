# main.py
import os
import torch
import numpy as np
from config.config import DATA_ROOT_DIR
# [修改] 导入正确的类名


from biw_nav.core.scale_ablation.dataset import VariableLengthTrajectoryDataset
from biw_nav.core.scale_ablation.runner import run_training, run_evaluation
from biw_nav.core.scale_ablation.model import cann_analysis 

def run_demo():
    # 1. 准备数据
    print(f"Loading dataset from {DATA_ROOT_DIR}...")
    
    # [修改] 根据 dataset.py 的 __init__ 参数进行实例化
    # __init__(self, root_dir, transform=None, use_yaw=True, min_len=3)
    if not os.path.exists(DATA_ROOT_DIR):
        print(f"❌ Error: Data directory not found: {DATA_ROOT_DIR}")
        return

    dataset = VariableLengthTrajectoryDataset(
        root_dir=DATA_ROOT_DIR, 
        use_yaw=True  # 确保包含 yaw 数据，匹配模型 feature_dim=3
    )
    
    print(f"✅ Dataset loaded. Total trajectories: {len(dataset)}")
    
    # ==========================================
    # Phase 0: CANN Theoretical Analysis (Fig 7)
    # ==========================================
    print("\n>>> Phase 0: Generating Theoretical Figure 7 (CANN)...")
    # 这一步是纯理论仿真，不依赖数据集
    # 为了演示，我们生成一条模拟轨迹来跑一下扫描
    dt = 0.1
    sim_traj = 2.0 * np.sin(0.5 * np.arange(500) * dt)
    # 简单调用一下确保模块无误 (绘图建议在 Notebook 中完成)
    _ = cann_analysis.run_alpha_sweep(np.linspace(0, 1, 5), sim_traj, tau0=20.0, dt=dt)
    print("✅ CANN simulation module verified.")
    
    # ==========================================
    # Phase 2: Evaluation & Plotting
    # ==========================================
    print("\n>>> Starting Evaluation Phase...")
    m_full = run_evaluation("Exp_Full_Fusion", dataset, {'use_visual': True, 'use_kinematics': True})
    m_vis  = run_evaluation("Exp_Visual_Only", dataset, {'use_visual': True, 'use_kinematics': False})
    m_kin  = run_evaluation("Exp_Kinematics_Only", dataset, {'use_visual': False, 'use_kinematics': True})

    # ==========================================
    # Phase 3: Final Table Print
    # ==========================================
    print("\n" + "="*40)
    print(">>> Table 1: Performance Comparison <<<")
    print("="*40)
    print(f"{'Model Variant':<20} | {'RMSE':<10} | {'R²':<10}")
    print("-" * 45)
    
    def pr(name, m):
        if m: print(f"{name:<20} | {m['RMSE']:.4f}     | {m['R2']:.4f}")
        else: print(f"{name:<20} | N/A        | N/A")
        
    pr("Full Fusion (Ours)", m_full)
    pr("Visual Only", m_vis)
    pr("Kinematics Only", m_kin)
    print("="*40)
