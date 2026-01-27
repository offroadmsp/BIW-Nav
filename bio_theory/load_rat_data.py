import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 尝试导入必要的库
try:
    import mat73
    HAS_MAT73 = True
except ImportError:
    HAS_MAT73 = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    print("[提示] 建议安装 h5py 以获得最强的兼容性: pip install h5py")

# ==========================================
# 配置区域
# ==========================================
class Config:
    DATA_ROOT = "/media/zhen/Data/cellData/The_place_cell_representation_of_volumetric_space_in_rats/Summarydata"
    SUMMARY_FILE = "tabulatedDATA.mat"
    BIN_SIZE = 2.0
    SMOOTH_SIGMA = 1.5
    MIN_DWELL = 0.1

# ==========================================
# 通用数据容器 (兼容 Struct/Dict/HDF5)
# ==========================================
class DataStruct:
    """
    一个通用的数据包装器，让 dict 或 h5py group 都能像 object.attribute 一样访问
    """
    def __init__(self, data_source):
        self._data = data_source
        self._is_h5 = isinstance(data_source, h5py.Group) if HAS_H5PY else False
        self._is_dict = isinstance(data_source, dict)

    def __getattr__(self, name):
        # 1. 尝试从字典或 H5 Group 中获取
        if self._is_dict:
            if name in self._data:
                val = self._data[name]
                return self._wrap(val)
        elif self._is_h5:
            if name in self._data:
                val = self._data[name]
                return self._wrap_h5(val)
        
        # 2. 如果是 scipy struct (object)，尝试直接访问
        if hasattr(self._data, name):
            return getattr(self._data, name)
            
        raise AttributeError(f"数据结构中不存在属性 '{name}'")

    def _wrap(self, val):
        if isinstance(val, dict):
            return DataStruct(val)
        return val

    def _wrap_h5(self, val):
        # HDF5 数据如果是 Dataset，需要读取为 numpy array
        if isinstance(val, h5py.Dataset):
            # 处理 MATLAB 字符串 (通常存储为 ASCII 整数数组)
            if val.dtype.kind == 'S': 
                return np.array(val).astype(str)
            # 转置：MATLAB (MxN) -> Python (NxM)，通常需要转置回来
            # 但 h5py 读取时维度往往是反的，这取决于具体存储
            data = np.array(val)
            # 这里不做激进的转置，按需处理
            return data
        elif isinstance(val, h5py.Group):
            return DataStruct(val)
        return val
    
    def keys(self):
        if self._is_dict or self._is_h5:
            return list(self._data.keys())
        return dir(self._data)

# ==========================================
# 核心类：生物数据管理器 (V3 稳健版)
# ==========================================
class RatDataManager:
    def __init__(self, data_dir=Config.DATA_ROOT):
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"[严重错误] 数据目录不存在: {self.data_dir}")
        self.tdata = None

    def _load_mat_robust(self, filepath):
        """
        三级加载策略: scipy -> mat73 -> h5py
        """
        basename = os.path.basename(filepath)
        
        # --- 策略 1: Scipy (最快，支持旧格式) ---
        try:
            # print(f"[尝试] Scipy 加载 {basename} ...")
            mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
            return mat, 'scipy'
        except NotImplementedError:
            pass # 遇到 v7.3 格式，继续
        except Exception as e:
            print(f"[Scipy 失败] {e}")

        # --- 策略 2: Mat73 (v7.3 专用) ---
        if HAS_MAT73:
            try:
                # print(f"[尝试] Mat73 加载 {basename} ...")
                mat = mat73.loadmat(filepath)
                return mat, 'mat73'
            except Exception as e:
                print(f"[Mat73 失败] {e}")
        
        # --- 策略 3: H5PY (底层读取，最稳健) ---
        if HAS_H5PY:
            try:
                print(f"[尝试] H5PY 直接读取 {basename} ...")
                f = h5py.File(filepath, 'r')
                return f, 'h5py'
            except OSError as e:
                print(f"[H5PY 失败] 文件可能不是有效的 HDF5/MAT 格式: {e}")
        
        raise RuntimeError(f"所有加载方法均失败。请检查文件 {basename} 是否损坏。")

    def load_summary_index(self):
        summary_path = os.path.join(self.data_dir, Config.SUMMARY_FILE)
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"找不到索引文件: {summary_path}")
            
        print(f"[加载] 正在读取索引: {summary_path}")
        raw_data, engine = self._load_mat_robust(summary_path)
        
        # 使用通用包装器
        wrapped_data = DataStruct(raw_data)
        available_keys = wrapped_data.keys()
        
        # --- 智能寻根逻辑 ---
        # 情况 A: 标准结构，包含 'tdata'
        if 'tdata' in available_keys:
            print("[结构] 发现标准 'tdata' 结构")
            self.tdata = getattr(wrapped_data, 'tdata')
            
        # 情况 B: 扁平结构，变量直接在根目录 (rat, date, cell_type)
        elif 'rat' in available_keys and 'cell_type' in available_keys:
            print("[结构] 数据位于根目录 (无 tdata 包裹)，正在自动组装...")
            self.tdata = wrapped_data # 直接把根作为 tdata 用
            
        # 情况 C: 只有 'None' 或奇怪的 key (mat73 解析失败常见情况)
        elif 'None' in available_keys and len(available_keys) == 1:
             raise RuntimeError(f"Mat73 解析失败 (Key='None')。请尝试安装 h5py: `pip install h5py` 以启用底层读取。")
             
        else:
            # 打印前10个key帮助调试
            debug_keys = list(available_keys)[:10]
            raise KeyError(f"在文件中未找到 'tdata' 且未识别出扁平结构。可用变量: {debug_keys}")

        # 验证数据的可用性
        try:
            # 尝试访问关键字段以测试
            _ = self.tdata.rat
            _ = self.tdata.cell_type
            print(f"[成功] 索引加载完成。")
        except Exception as e:
            print(f"[警告] tdata 对象存在，但无法访问 'rat' 或 'cell_type' 字段。可能需要进一步调试结构。错误: {e}")

    def query_cell(self, cell_type=1, min_si=0.5):
        if self.tdata is None:
            self.load_summary_index()
            
        # 提取数据 (兼容 HDF5 Dataset 和 Numpy)
        # 注意：如果是 H5PY 读取的，数据可能是 (N,1) 而不是 (N,)，需要 flatten
        try:
            types = np.array(self.tdata.cell_type).flatten()
            si_scores = np.array(self.tdata.spatial_information).flatten()
            rat_ids = np.array(self.tdata.rat).flatten()
            dates = np.array(self.tdata.date).flatten()
        except Exception as e:
            raise RuntimeError(f"数据字段提取失败，文件结构可能过于复杂。错误: {e}")

        # 筛选
        valid_indices = np.where((types == cell_type) & (si_scores > min_si))[0]
        
        if len(valid_indices) == 0:
            print(f"[警告] 未找到 Type={cell_type}, SI>{min_si} 的细胞。")
            return None
            
        # 排序：找 SI 最高的
        best_local_idx = np.argmax(si_scores[valid_indices])
        best_idx = valid_indices[best_local_idx]
        
        rat_id = int(rat_ids[best_idx])
        date_raw = int(dates[best_idx])
        date_str = str(date_raw)
        
        print(f"\n[筛选结果] 最佳细胞 (Index {best_idx}):")
        print(f"  - Rat ID : {rat_id}")
        print(f"  - Date   : {date_str}")
        print(f"  - SI     : {si_scores[best_idx]:.4f}")
        
        return rat_id, date_str

    def get_session_data(self, rat_id, date_str):
        # 模糊匹配文件名
        target_part = f"{rat_id}"
        date_part = f"{date_str}"
        
        found_file = None
        # 遍历目录寻找匹配
        for fname in os.listdir(self.data_dir):
            if fname.startswith(target_part) and date_part in fname and fname.endswith(".mat"):
                found_file = fname
                break
        
        if not found_file:
            print(f"[错误] 找不到对应的原始文件: Rat {rat_id} Date {date_str}")
            return None, None

        file_path = os.path.join(self.data_dir, found_file)
        print(f"[加载] 原始文件: {found_file}")
        
        # 加载数据
        raw_data, engine = self._load_mat_robust(file_path)
        wrapped_data = DataStruct(raw_data)
        keys = wrapped_data.keys()
        
        # 智能提取 pos 和 spk
        pos = None
        spk = None
        
        # 1. 找位置数据
        for k in ['pos', 'position', 'XY', 'pos_online']:
            if k in keys:
                pos = np.array(getattr(wrapped_data, k))
                # 兼容 H5PY 转置问题: 轨迹通常是 (N, 3) 或 (N, 2)
                # 如果形状是 (3, N)，转置它
                if pos.shape[0] in [2, 3] and pos.shape[1] > 100:
                    pos = pos.T
                break
        
        # 2. 找脉冲数据
        for k in ['spk', 'spikes', 'cell_TS']:
            if k in keys:
                spk = np.array(getattr(wrapped_data, k)).flatten()
                break
                
        if pos is None:
            print(f"[错误] 文件中未找到位置变量。可用 Keys: {keys}")
            return None, None
            
        return pos, spk

# ==========================================
# 主程序
# ==========================================
def main():
    manager = RatDataManager()
    
    # 1. 筛选细胞
    result = manager.query_cell(cell_type=1, min_si=0.8)
    
    if result:
        rat_id, date_str = result
        
        # 2. 加载原始数据
        pos, spk = manager.get_session_data(rat_id, date_str)
        
        if pos is not None:
            # 3. 数据预处理
            # 检查 spk 是时间戳还是 spike train
            if len(spk) < len(pos) and (len(spk) == 0 or np.max(spk) > 1):
                print("[处理] 检测到 Spike Timestamps，转换为 Spike Train...")
                spk_train = np.zeros(len(pos))
                # 假设 50Hz (dt=0.02s)
                # 注意：具体采样率需参考论文，这里使用 50Hz 作为通用默认
                idx = (spk * 50).astype(int)
                idx = idx[idx < len(pos)]
                spk_train[idx] = 1
                spk_vector = spk_train
            else:
                spk_vector = spk

            # 4. 可视化
            print("[绘图] 生成轨迹图...")
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(pos[:,0], pos[:,1], 'gray', alpha=0.3, label='Path')
            
            # 绘制脉冲位置
            spk_indices = np.where(spk_vector > 0)[0]
            if len(spk_indices) > 0:
                plt.scatter(pos[spk_indices,0], pos[spk_indices,1], c='r', s=5, alpha=0.8, label='Spikes')
            
            plt.title(f"Rat {rat_id} - {date_str}\n(Real Data)")
            plt.legend()
            plt.axis('equal')
            
            plt.subplot(1, 2, 2)
            plt.axis('off')
            plt.text(0.5, 0.5, "Data Loaded Successfully!\nReady for Bio-Model Input", 
                     ha='center', fontsize=12)
            
            plt.show()

if __name__ == "__main__":
    main()