import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

# ==========================================
# 模块 1: Excel 数据读取适配器 (针对 .xlsx)
# ==========================================
class NatureExcelParser:
    """
    专门读取 Grieves et al. 2020 的原始 Excel 文件 (.xlsx)
    自动处理多 Sheet 结构
    """
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.params = {
            'grid_scale_mean': 30.0,  # 默认备用值
            'grid_orientation': [0, 60, 120] # 默认角度
        }
        self.sheet_names = []
        
        # 预加载 Excel 文件信息
        if os.path.exists(self.excel_path):
            try:
                xl = pd.ExcelFile(self.excel_path)
                self.sheet_names = xl.sheet_names
                print(f"[文件加载] 成功识别 Excel 文件，包含 Sheet: {self.sheet_names}")
            except Exception as e:
                print(f"[错误] Excel 文件损坏或无法读取: {e}")
        else:
            print(f"[错误] 找不到文件: {self.excel_path}")

    def _find_sheet(self, keywords):
        """模糊匹配 Sheet 名称"""
        for sheet in self.sheet_names:
            # 只要包含关键词（如 "Fig. 7" 或 "Fig 7"）就匹配
            if all(k in sheet for k in keywords):
                return sheet
        return None

    def parse_field_stats(self):
        """
        从 Fig. S11 对应的 Sheet 提取位置场大小
        """
        target_sheet = self._find_sheet(["Fig", "S11"]) # 搜索包含 Fig 和 S11 的标签
        
        if target_sheet:
            try:
                print(f"[数据读取] 正在读取 Sheet: '{target_sheet}'...")
                # 读取该 Sheet
                df = pd.read_excel(self.excel_path, sheet_name=target_sheet, header=None)
                
                # 尝试解析：通常 S11 的 Panel b 或 d 包含直径数据
                # 我们暴力转换所有数据为数值，取非空值的统计特征
                numeric_df = df.apply(pd.to_numeric, errors='coerce')
                
                # 假设直径数据在第一列或第二列 (Arena/Aligned)
                # 过滤掉小于 10cm 的噪声值（直径通常 > 20cm）
                valid_data = numeric_df[numeric_df > 10].stack()
                
                if not valid_data.empty:
                    mean_dia = valid_data.mean()
                    self.params['grid_scale_mean'] = mean_dia
                    print(f"  -> 提取到位置场平均直径: {mean_dia:.2f} cm")
                else:
                    print("  -> 未在 Sheet 中找到有效数值，使用默认值。")
            except Exception as e:
                print(f"  -> 读取 Sheet 失败: {e}")
        else:
            print("[警告] 未找到 Fig. S11 对应的 Sheet，使用默认直径 30.0cm")

    def parse_eigenvectors(self):
        """
        从 Fig. 7 对应的 Sheet 提取网格方向
        """
        target_sheet = self._find_sheet(["Fig", "7"]) # 搜索包含 Fig 和 7 的标签
        
        if target_sheet:
            try:
                print(f"[数据读取] 正在读取 Sheet: '{target_sheet}'...")
                # Fig 7 的数据通常有较多表头，跳过前 4 行尝试
                df = pd.read_excel(self.excel_path, sheet_name=target_sheet, header=None, skiprows=4)
                
                # Fig 7 特征向量通常在中间列。我们寻找包含数值密集的区域。
                # 简化策略：寻找数值列的均值
                numeric_df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
                
                if not numeric_df.empty:
                    # 假设前 3 个有效数值列对应 X, Y, Z 分量
                    # 这里做一个鲁棒性处理：取方差最大的几列作为特征向量候选
                    vec_data = numeric_df.iloc[:, 10:13].dropna() # 尝试定位第10-12列(Arena数据)
                    
                    if vec_data.empty:
                         vec_data = numeric_df.iloc[:, :3].dropna() # 如果失败，尝试前3列

                    if not vec_data.empty:
                        mean_vec = vec_data.mean().values
                        # 计算 2D 投影角度
                        angle_deg = np.degrees(np.arctan2(mean_vec[1], mean_vec[0]))
                        self.params['grid_orientation'] = [angle_deg, angle_deg+60, angle_deg+120]
                        print(f"  -> 提取到网格主方向: {angle_deg:.2f} 度")
                    else:
                        print("  -> 未提取到有效特征向量，使用默认方向。")
            except Exception as e:
                print(f"  -> 读取 Sheet 失败: {e}")
        else:
            print("[警告] 未找到 Fig. 7 对应的 Sheet，使用默认方向 [0, 60, 120]")

# ==========================================
# 模块 2: 核心算法 (保持不变，无需修改)
# ==========================================
def compute_ratemap(pos, spikes, bin_size=2.0, map_lims=None, sigma=1.5, min_dwell=0.1, srate=50.0):
    pos = np.array(pos)
    spikes = np.array(spikes)
    if map_lims is None:
        min_x, max_x = np.nanmin(pos[:,0]), np.nanmax(pos[:,0])
        min_y, max_y = np.nanmin(pos[:,1]), np.nanmax(pos[:,1])
    else:
        min_x, max_x, min_y, max_y = map_lims
    
    x_edges = np.arange(min_x, max_x + bin_size, bin_size)
    y_edges = np.arange(min_y, max_y + bin_size, bin_size)
    
    occupancy, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=[x_edges, y_edges])
    dwellmap = occupancy.T * (1.0 / srate) 
    spike_counts, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=[x_edges, y_edges], weights=spikes)
    spikemap = spike_counts.T
    
    smooth_dwell = gaussian_filter(dwellmap, sigma, mode='constant')
    smooth_spikes = gaussian_filter(spikemap, sigma, mode='constant')
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratemap = smooth_spikes / smooth_dwell
    ratemap[dwellmap < min_dwell] = np.nan
    
    return ratemap, dwellmap, x_edges, y_edges

def compute_spatial_info(ratemap, dwellmap):
    rate = ratemap.flatten()
    dwell = dwellmap.flatten()
    valid = ~np.isnan(rate) & ~np.isnan(dwell) & (dwell > 0)
    rate, dwell = rate[valid], dwell[valid]
    if len(rate) == 0 or np.sum(rate) == 0: return 0.0
    T = np.sum(dwell)
    P_i = dwell / T
    mean_rate = np.sum(P_i * rate)
    if mean_rate == 0: return 0.0
    info = P_i * (rate / mean_rate) * np.log2((rate / mean_rate) + 1e-10)
    return np.sum(info)

# ==========================================
# 模块 3: 主程序
# ==========================================
def run_experiment():
    # 文件名配置
    excel_file = '41467_2020_14611_MOESM9_ESM.xlsx'
    
    print("=== 初始化 Excel 解析器 ===")
    parser = NatureExcelParser(excel_file)
    
    # 解析数据
    parser.parse_field_stats()  # 找 Fig S11
    parser.parse_eigenvectors() # 找 Fig 7
    
    # 提取参数用于仿真
    grid_scale = parser.params['grid_scale_mean']
    grid_angles = parser.params['grid_orientation']
    
    print(f"\n=== 开始仿真 (基于真实参数) ===")
    print(f"  - Grid Scale: {grid_scale:.2f} cm")
    print(f"  - Grid Orientation: {grid_angles}")
    
    # 仿真设置
    T_steps = 20000
    dt = 0.02
    box_size = 100 
    
    # 轨迹生成
    pos = np.zeros((T_steps, 2))
    vel = np.zeros(2)
    for t in range(1, T_steps):
        vel = vel * 0.95 + np.random.randn(2) * 2.0
        pos[t] = pos[t-1] + vel * dt
        for d in range(2):
            if pos[t,d] < 0: pos[t,d]=0; vel[d]*=-1
            if pos[t,d] > box_size: pos[t,d]=box_size; vel[d]*=-1
            
    # 神经元激活 (Grid Cell)
    # 使用真实数据的 Scale 和 Orientation
    # 注意：网格间距 (Spacing) 约为 Field Diameter * 1.5 ~ 2.0
    # 这里我们做一个近似转换：Spacing = Diameter * 1.6
    spacing = grid_scale * 1.6
    k = 4 * np.pi / (spacing * np.sqrt(3))
    
    activation = np.zeros(T_steps)
    for deg in grid_angles:
        theta = np.radians(deg)
        proj = pos[:, 0] * np.cos(theta) + pos[:, 1] * np.sin(theta)
        activation += np.cos(k * proj)
        
    firing_rate = (activation + 1.5) * 5 
    spikes = np.random.poisson(firing_rate * dt)
    
    # 分析
    ratemap, dwellmap, _, _ = compute_ratemap(pos, spikes, bin_size=2.5, sigma=2.0)
    si = compute_spatial_info(ratemap, dwellmap)
    
    print(f"\n[结果] Spatial Information: {si:.4f} bits/spike")
    print(f"(如果 SI > 0.3，说明基于真实参数复现的网格细胞具有良好的空间选择性)")
    
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(pos[:,0], pos[:,1], 'k', alpha=0.1)
    plt.scatter(pos[spikes>0,0], pos[spikes>0,1], c='r', s=2, alpha=0.5)
    plt.title(f"Simulated Spikes\n(Scale={grid_scale:.1f}cm)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(ratemap, origin='lower', cmap='jet', interpolation='nearest', extent=[0,100,0,100])
    plt.colorbar(label='Hz')
    plt.title(f"Rate Map (SI={si:.2f})")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 检查依赖
    try:
        import openpyxl
        run_experiment()
    except ImportError:
        print("错误：缺少 'openpyxl' 库。请运行: pip install openpyxl")