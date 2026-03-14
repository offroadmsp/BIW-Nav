# runner.py
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from config import TRAIN_CONFIG, MODEL_CONFIG, RESULTS_DIR, DEVICE

# 导入你的模块
from dataset import collate_variable_length
from model.visualNet import VisualTemporalNet_Optimized
from model import plotcell, evalcell, evaltraj

# NMI 评估器辅助类 (简单的静态方法封装)
class NMI_Evaluator:
    @staticmethod
    def calculate_metrics(targets, preds):
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(targets, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, preds)
        # NRMSE (Normalized by range)
        data_range = targets.max() - targets.min()
        nrmse = rmse / (data_range + 1e-6)
        return {"MSE": mse, "RMSE": rmse, "NRMSE": nrmse, "R2": r2}

    @staticmethod
    def print_summary(metrics, prefix="Model"):
        print(f"[{prefix} Summary] RMSE: {metrics['RMSE']:.4f} | NRMSE: {metrics['NRMSE']:.4f} | R²: {metrics['R2']:.4f}")

def get_exp_dir(exp_name):
    path = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(path, exist_ok=True)
    return path

# --- 核心函数 1: 训练 ---
def run_training(exp_name, dataset, ablation_config, train_cfg=TRAIN_CONFIG, model_cfg=MODEL_CONFIG):
    save_dir = get_exp_dir(exp_name)
    print(f"\n{'='*10} [TRAIN] START: {exp_name} {'='*10}")
    
    # 1. 初始化模型
    model = VisualTemporalNet_Optimized(
        feature_dim=model_cfg['feature_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        visual_dim=model_cfg['visual_dim'],
        output_dim=model_cfg['output_dim'],
        ablation_config=ablation_config
    ).to(train_cfg['device'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
    criterion = torch.nn.MSELoss(reduction='none') 
    
    # 2. 训练循环
    loader = DataLoader(dataset, batch_size=train_cfg['batch_size'], 
                        shuffle=True, collate_fn=collate_variable_length)
    
    model.train()
    for epoch in range(train_cfg['epochs']):
        total_loss = 0
        batch_count = 0
        
        for feats, imgs, _, mask in loader:
            if feats is None: continue 
            
            targets_seq = feats[:, :, :2].clone() # 自监督：预测自身轨迹
            
            feats = feats.to(train_cfg['device'])
            imgs = imgs.to(train_cfg['device'])
            targets_seq = targets_seq.to(train_cfg['device'])
            mask = mask.to(train_cfg['device'])
            
            optimizer.zero_grad()
            preds, _ = model(feats, imgs, mask)
            
            raw_loss = criterion(preds, targets_seq)
            masked_loss = (raw_loss.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-6)
            
            masked_loss.backward()
            optimizer.step()
            
            total_loss += masked_loss.item()
            batch_count += 1
            
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{train_cfg['epochs']} | Avg Loss: {total_loss/(batch_count+1e-6):.6f}")

    # 3. 保存
    model_path = os.path.join(save_dir, "model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to: {model_path}")
    return model_path

# --- 核心函数 2: 评估 (包含所有 NMI 图表) ---
def run_evaluation(exp_name, dataset, ablation_config, model_cfg=MODEL_CONFIG, device=DEVICE):
    save_dir = get_exp_dir(exp_name)
    model_path = os.path.join(save_dir, "model_final.pth")
    
    print(f"\n{'='*10} [EVAL] START: {exp_name} {'='*10}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return None

    # 1. 加载模型
    model = VisualTemporalNet_Optimized(
        feature_dim=model_cfg['feature_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        visual_dim=model_cfg['visual_dim'],
        output_dim=model_cfg['output_dim'],
        ablation_config=ablation_config
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_variable_length)

    # 2. 计算 Metrics
    print("Computing metrics...")
    all_preds, all_targets = [], []
    with torch.no_grad():
        for feats, imgs, _, mask in loader:
            if feats is None: continue
            targets_seq = feats[:, :, :2].clone()
            feats, imgs, targets_seq, mask = feats.to(device), imgs.to(device), targets_seq.to(device), mask.to(device)
            preds, _ = model(feats, imgs, mask)
            
            mask_bool = mask.bool().cpu().numpy().flatten()
            p = preds.cpu().numpy().reshape(-1, 2)
            t = targets_seq.cpu().numpy().reshape(-1, 2)
            all_preds.append(p[mask_bool]); all_targets.append(t[mask_bool])
            
    if len(all_preds) > 0:
        flat_preds = np.concatenate(all_preds, axis=0)
        flat_targets = np.concatenate(all_targets, axis=0)
        metrics = NMI_Evaluator.calculate_metrics(flat_targets, flat_preds)
        NMI_Evaluator.print_summary(metrics, prefix=exp_name)
    else:
        return None

    # 3. 生成图表
    print(f"Generating plots in {save_dir}...")
    cell_results = plotcell.analyze_time_cells(model, loader, device, save_dir=save_dir)
    
    if cell_results:
        # 认知结构分析 (Figure 6/7)
        evalcell.analyze_cognitive_structure(model, loader, cell_results, device, save_dir=save_dir)
        # 轨迹相关性
        evaltraj.evaluate_timecell_trajectory_correlation(model, loader, cell_results, device, save_dir=save_dir)
        # 推理消融
        evalcell.evaluate_time_scales_and_extensions(model, loader, cell_results, device, save_dir=save_dir)
    
    # 轨迹可视化
    evaltraj.evaluate_and_visualize(model, loader, device, model_path="", save_dir=save_dir, save_prefix="final", plot=True)
    
    return metrics