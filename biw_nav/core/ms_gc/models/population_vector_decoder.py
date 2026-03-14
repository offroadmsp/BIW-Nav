import torch
import torch.nn as nn

class PopulationVectorDecoder(nn.Module):
    def __init__(self, grid_scales, num_cells_per_scale):
        super(PopulationVectorDecoder, self).__init__()
        self.grid_scales = grid_scales
        self.num_cells_per_scale = num_cells_per_scale
        
        # 初始化每个尺度的解码器
        self.decoders = nn.ModuleList([
            self._init_decoder(scale, num_cells) for scale, num_cells in zip(grid_scales, num_cells_per_scale)
        ])
    
    def _init_decoder(self, scale, num_cells):
        return ScaleDecoder(scale, num_cells)
    
    def forward(self, grid_activities):
        # 解码每个尺度的位置信息
        decoded_positions = []
        for decoder, activity in zip(self.decoders, grid_activities):
            decoded_positions.append(decoder(activity))
        return decoded_positions

class ScaleDecoder(nn.Module):
    def __init__(self, scale, num_cells):
        super(ScaleDecoder, self).__init__()
        self.scale = scale
        self.num_cells = num_cells
        
        # 解码权重
        self.weights = nn.Parameter(torch.randn(num_cells, 2))
    
    def forward(self, activity):
        # 使用加权平均解码位置
        activity = activity.unsqueeze(-1)  # (batch_size, num_cells, 1)
        decoded_position = torch.sum(activity * self.weights, dim=1)
        return decoded_position * self.scale