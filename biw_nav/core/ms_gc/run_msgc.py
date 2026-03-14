import argparse
import os
import torch
from training.config import Config
from training.trainer import Trainer
from data.dataset import PositionDataset
from torch.utils.data import DataLoader

def main(config):
    # 初始化数据集和数据加载器
    train_dataset = PositionDataset(config.data_dir, split='train')
    val_dataset = PositionDataset(config.data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 初始化训练器
    trainer = Trainer(config)
    
    # 开始训练
    trainer.train(train_loader, val_loader, config.num_epochs)
    
    # 可视化
    # 生成一些示例数据进行可视化
    example_positions = torch.rand((100, 2)) * 10  # 100个随机位置，范围[0, 10]
    example_positions = example_positions.to(trainer.device)
    
    # 计算网格活动和解码位置
    with torch.no_grad():
        grid_activities = trainer.grid_model(example_positions)
        decoded_positions = trainer.decoder(grid_activities)
    
    # 可视化
    trainer.visualize(example_positions, grid_activities, decoded_positions)

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Grid Cells Decoding')
    parser.add_argument('--config', type=str, default='ms_gc/configs/default.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    print(config)
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # 创建检查点目录
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    if not os.path.exists(config.pics_dir):
        os.makedirs(config.pics_dir)
    
    # 开始主程序
    main(config)