import os
from utils.visualization import plot_position_decoding, visualize_grid_activities, plot_position_decoding_with_error_1x3
import torch
import torch.optim as optim
from models.grid_cell_model import GridCellModel
from models.population_vector_decoder import PopulationVectorDecoder
from utils.training import train_one_epoch, validate, save_checkpoint, load_checkpoint
from utils.visualization import plot_loss_curve

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.grid_model = GridCellModel(
            grid_scales=config.grid_scales,
            num_cells_per_scale=config.num_cells_per_scale,
            spatial_resolution=config.spatial_resolution
        ).to(self.device)
        
        self.decoder = PopulationVectorDecoder(
            grid_scales=config.grid_scales,
            num_cells_per_scale=config.num_cells_per_scale
        ).to(self.device)
        
        # 定义优化器
        self.optimizer = optim.Adam(
            list(self.grid_model.parameters()) + list(self.decoder.parameters()),
            lr=config.learning_rate
        )
        
        # 定义损失函数
        self.criterion = torch.nn.MSELoss()
    
    def train(self, train_loader, val_loader, num_epochs):
        train_losses = []
        val_losses = []
        
        start_epoch = 0
        if self.config.resume_from_checkpoint:
            start_epoch = load_checkpoint(self.grid_model, self.decoder, self.optimizer, 
                                          self.config.checkpoint_dir, self.device)
        
        for epoch in range(start_epoch, num_epochs):
            # 训练一个epoch
            avg_train_loss = train_one_epoch(self, train_loader, epoch)
            train_losses.append(avg_train_loss)
            
            # 验证
            avg_val_loss = validate(self, val_loader)
            val_losses.append(avg_val_loss)
            
            # 保存模型
            if (epoch+1) % self.config.save_interval == 0:
                save_checkpoint(self.grid_model, self.decoder, self.optimizer, epoch, 
                                self.config.checkpoint_dir)
        
        # 绘制损失曲线
        plot_loss_curve(train_losses, val_losses, 
                        save_path=os.path.join(self.config.pics_dir, 'loss_curve.pdf'))
    
    def visualize(self, positions, grid_activities, decoded_positions):
        # 确保所有张量都在 CPU 上
        positions = positions.cpu()
        grid_activities = [activity.cpu() for activity in grid_activities]
        decoded_positions = [decoded.cpu() for decoded in decoded_positions]
        
        # 绘制位置解码结果
        # plot_position_decoding(positions.numpy(), decoded_positions, self.config.grid_scales,
        #                        save_path=os.path.join(self.config.pics_dir, 'position_decoding.pdf'))
        
        # 可视化网格细胞活动
        visualize_grid_activities(positions.numpy(), grid_activities, self.config.grid_scales,
                                  save_path=os.path.join(self.config.pics_dir, 'grid_activities.pdf'), color_style="plasma")
        
        plot_position_decoding_with_error_1x3(positions.numpy(), decoded_positions, self.config.grid_scales,
                                        save_path=os.path.join(self.config.pics_dir, 'position_decoding_with_error.pdf'))