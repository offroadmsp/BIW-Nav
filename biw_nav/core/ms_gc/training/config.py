import yaml
import torch

class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # 模型配置
        self.grid_scales = config_dict['model']['grid_scales']
        self.num_cells_per_scale = config_dict['model']['num_cells_per_scale']
        self.spatial_resolution = config_dict['model']['spatial_resolution']
        
        # 训练配置
        self.batch_size = config_dict['training']['batch_size']
        self.num_epochs = config_dict['training']['num_epochs']
        self.learning_rate = config_dict['training']['learning_rate']
        self.save_interval = config_dict['training']['save_interval']
        self.checkpoint_dir = config_dict['training']['checkpoint_dir']
        self.resume_from_checkpoint = config_dict['training']['resume_from_checkpoint']
        self.checkpoint_path = config_dict['training']['checkpoint_path']
        
        # 数据配置
        self.data_dir = config_dict['data']['data_dir']
        self.pics_dir = config_dict['data']['pics_dir']
        
        # 随机种子
        self.seed = config_dict['seed']