import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class NMI_Evaluator:
    @staticmethod
    def calculate_metrics(targets, preds, target_std=None):
        """
        计算 NMI 论文级别的详细指标
        targets, preds: [N, D] numpy arrays
        target_std: 用于计算 NRMSE 的目标标准差 (可选)
        """
        # 1. 基础 MSE
        mse = mean_squared_error(targets, preds)
        
        # 2. RMSE (米级误差，直观)
        rmse = np.sqrt(mse)
        
        # 3. Per-dimension RMSE (X轴, Y轴, Yaw 谁在拖后腿?)
        # 假设维度是 [x, y] 或 [x, y, yaw]
        diff = targets - preds
        rmse_per_dim = np.sqrt(np.mean(diff**2, axis=0))
        
        # 4. NRMSE (归一化误差，消除量纲影响)
        # 如果没有提供 std，就用当前 batch 的 std
        if target_std is None:
            target_std = np.std(targets, axis=0).mean() + 1e-6
        nrmse = rmse / target_std
        
        # 5. R2 Score (拟合优度)
        r2 = r2_score(targets, preds)
        
        return {
            "MSE": mse,
            "RMSE": rmse,
            "RMSE_x": rmse_per_dim[0],
            "RMSE_y": rmse_per_dim[1] if len(rmse_per_dim) > 1 else 0,
            "NRMSE": nrmse,
            "R2": r2
        }

    @staticmethod
    def print_summary(metrics, prefix="Eval"):
        print(f"[{prefix} Summary] "
              f"RMSE: {metrics['RMSE']:.4f} | "
              f"NRMSE: {metrics['NRMSE']:.4f} | "
              f"R²: {metrics['R2']:.4f}")
        print(f"    > Detail: RMSE_x={metrics['RMSE_x']:.4f}, RMSE_y={metrics['RMSE_y']:.4f}")