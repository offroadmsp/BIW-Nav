# config.py
import torch

# 1. 路径设置
DATA_ROOT_DIR = "/media/zhen/Data/Datasets/nomad_data/go_stanford" # 请确认你的路径
RESULTS_DIR = "./eval_results"

# 2. 硬件设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. 训练超参数
TRAIN_CONFIG = {
    'batch_size': 4,
    'lr': 1e-4,
    'epochs': 50,  # 建议根据实际情况调整
    'device': DEVICE
}

# 4. 模型默认参数
MODEL_CONFIG = {
    'feature_dim': 3,    # x, y, yaw
    'hidden_dim': 128,   # LSTM hidden
    'visual_dim': 256,   # CNN output
    'output_dim': 2      # x, y prediction
}