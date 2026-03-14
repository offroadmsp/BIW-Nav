import torch
import torch.nn as nn
import torch.nn.functional as F

class GridCellModel(nn.Module):
    def __init__(self, grid_scales, num_cells_per_scale, spatial_resolution):
        super(GridCellModel, self).__init__()
        self.grid_scales = grid_scales
        self.num_cells_per_scale = num_cells_per_scale
        self.spatial_resolution = spatial_resolution
        
        # 初始化每个尺度的网格细胞
        self.grid_modules = nn.ModuleList([
            self._init_grid_module(scale, num_cells) 
            for scale, num_cells in zip(grid_scales, num_cells_per_scale)
        ])
    
    def _init_grid_module(self, scale, num_cells):
        # 每个模块包含一组具有相同尺度的网格细胞
        return GridModule(scale, num_cells, self.spatial_resolution)
    
    def forward(self, positions):
        # 计算每个模块的网格细胞活动
        activities = []
        for module in self.grid_modules:
            activities.append(module(positions))
        return activities

class GridModule(nn.Module):
    def __init__(self, scale, num_cells, spatial_resolution):
        super(GridModule, self).__init__()
        self.scale = scale
        self.num_cells = num_cells
        self.spatial_resolution = spatial_resolution
        
        # 初始化网格细胞的相位和权重
        self.phases = nn.Parameter(torch.randn((num_cells, 2)) * scale) # 确保使用元组
        self.weights = nn.Parameter(torch.randn(num_cells))
    
    def forward(self, positions):
        # 计算网格细胞的活动水平
        batch_size = positions.size(0)
        positions = positions.view(batch_size, 1, 2)  # (batch_size, 1, 2)
        phases = self.phases.view(1, self.num_cells, 2)  # (1, num_cells, 2)
        
        # 计算每个网格细胞的活动
        distance = torch.norm(positions - phases, dim=-1)  # (batch_size, num_cells)
        activity = torch.exp(-distance / self.scale) * self.weights
        return activity