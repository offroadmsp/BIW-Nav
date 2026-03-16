# config.py
import torch

# 1. Path Settings
DATA_ROOT_DIR = "data/go_stanford" # Please verify that this is the actual path to the data.
RESULTS_DIR = "./results"

# 2. Hardware Settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Training Hyperparameters
TRAIN_CONFIG = {
    'batch_size': 4,
    'lr': 1e-4,
    'epochs': 50,  # Please adjust according to your specific use case
    'device': DEVICE
}

# 4. Model Default Parameters
MODEL_CONFIG = {
    'feature_dim': 3,    # x, y, yaw
    'hidden_dim': 128,   # LSTM hidden
    'visual_dim': 256,   # CNN output
    'output_dim': 2      # x, y prediction
}