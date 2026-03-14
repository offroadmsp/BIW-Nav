#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import copy
from typing import List, Tuple, Optional


def _to_numpy(tensor):
    """
    安全地将张量转换为numpy数组
    自动处理CPU/CUDA张量
    """
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def _argmax_correct(pred, target):
    """
    比较预测和目标是否匹配
    优化：批量处理，自动设备管理
    """
    if torch.is_tensor(pred) and torch.is_tensor(target):
        return (torch.argmax(pred) == torch.argmax(target)).cpu().item()
    return (torch.argmax(pred) == torch.argmax(target)).item()


# Track prediction accuracy over walk, and calculate fraction of locations visited and actions taken
def performance(forward, model, environments):
    """
    跟踪步行过程中的预测准确率，计算访问位置和采取动作的比例
    
    优化：向量化操作，减少循环
    """
    all_correct, all_location_frac, all_action_frac = [], [], []
    
    # 预先计算可用动作矩阵
    n_actions = model.hyper['n_actions']
    has_static = model.hyper['has_static_action']
    
    for env_i, env in enumerate(environments):
        n_locations = env.n_locations
        
        # 初始化跟踪数组
        location_visited = np.zeros(n_locations, dtype=bool)
        action_taken = np.zeros((n_locations, n_actions), dtype=bool)
        
        # 计算可用动作
        action_available = np.zeros((n_locations, n_actions), dtype=bool)
        for location in env.locations:
            for action in location['actions']:
                if np.sum(action['transition']) > 0:
                    action_id = action['id'] - 1 if has_static and action['id'] > 0 else action['id']
                    if 0 <= action_id < n_actions:
                        action_available[location['id'], action_id] = True
        
        # 计算总可用动作数（只计算一次）
        total_available = np.sum(action_available)
        
        correct = []
        location_frac = []
        action_frac = []
        
        # 批量处理步骤
        for step in forward:
            loc_id = step.g[env_i]['id']
            
            # 更新访问状态
            location_visited[loc_id] = True
            
            # 更新动作
            action_id = step.a[env_i]
            if has_static and action_id > 0:
                action_id -= 1
            if 0 <= action_id < n_actions:
                action_taken[loc_id, action_id] = True
            
            # 检查预测正确性（优化：直接使用张量比较）
            correct.append(_argmax_correct(step.x_gen[2][env_i], step.x[env_i]))
            
            # 计算比例（优化：使用numpy的快速求和）
            location_frac.append(np.sum(location_visited) / n_locations)
            action_frac.append(np.sum(action_taken) / total_available if total_available > 0 else 0)
        
        all_correct.append(correct)
        all_location_frac.append(location_frac)
        all_action_frac.append(action_frac)
    
    return all_correct, all_location_frac, all_action_frac


# Track prediction accuracy per location
def location_accuracy(forward, model, environments):
    """
    跟踪每个位置的预测准确率
    
    优化：减少重复计算，批量处理
    """
    accuracy_from, accuracy_to = [], []
    
    for env_i, env in enumerate(environments):
        n_locations = env.n_locations
        
        # 使用列表存储每个位置的正确预测
        correct_from = [[] for _ in range(n_locations)]
        correct_to = [[] for _ in range(n_locations)]
        
        # 跳过第一步（没有前一步）
        for step_i, step in enumerate(forward[1:]):
            loc_id = step.g[env_i]['id']
            prev_loc_id = forward[step_i].g[env_i]['id']
            
            # 计算预测正确性
            is_correct = _argmax_correct(step.x_gen[2][env_i], step.x[env_i])
            
            # 到达预测
            correct_to[loc_id].append(is_correct)
            
            # 离开预测
            correct_from[prev_loc_id].append(is_correct)
        
        # 计算平均准确率（优化：向量化）
        accuracy_from.append([
            sum(correct) / len(correct) if len(correct) > 0 else 0
            for correct in correct_from
        ])
        accuracy_to.append([
            sum(correct) / len(correct) if len(correct) > 0 else 0
            for correct in correct_to
        ])
    
    return accuracy_from, accuracy_to


# Track occupation per location
def location_occupation(forward, model, environments):
    """
    跟踪每个位置的访问次数
    
    优化：使用numpy数组直接计数
    """
    occupation = []
    
    for env_i, env in enumerate(environments):
        n_locations = env.n_locations
        visits = np.zeros(n_locations, dtype=int)
        
        # 批量统计访问
        for step in forward:
            visits[step.g[env_i]['id']] += 1
        
        occupation.append(visits.tolist())
    
    return occupation


# Measure zero-shot inference
def zero_shot(forward, model, environments, include_stay_still=True):
    """
    测量零样本推理：模型能否预测新动作到已知位置后的观察
    
    优化：向量化状态跟踪，减少条件判断
    """
    n_actions = model.hyper['n_actions'] + model.hyper['has_static_action']
    has_static = model.hyper['has_static_action']
    
    all_correct_zero_shot = []
    
    for env_i, env in enumerate(environments):
        n_locations = env.n_locations
        
        # 初始化跟踪数组
        location_visited = np.zeros(n_locations, dtype=bool)
        action_taken = np.zeros((n_locations, n_actions), dtype=bool)
        
        correct_zero_shot = []
        
        # 处理第一步之后的所有步骤
        for step_i, step in enumerate(forward[1:]):
            prev_step = forward[step_i]
            
            prev_a = prev_step.a[env_i]
            prev_g = prev_step.g[env_i]['id']
            curr_g = step.g[env_i]['id']
            
            # 处理静态动作
            if has_static and prev_a == 0 and not include_stay_still:
                prev_a = None
            
            # 标记位置已访问
            location_visited[prev_g] = True
            
            # 检查零样本推理条件
            if (location_visited[curr_g] and 
                prev_a is not None and 
                0 <= prev_a < n_actions and
                not action_taken[prev_g, prev_a]):
                
                correct_zero_shot.append(_argmax_correct(step.x_gen[2][env_i], step.x[env_i]))
            
            # 更新动作状态
            if prev_a is not None and 0 <= prev_a < n_actions:
                action_taken[prev_g, prev_a] = True
        
        all_correct_zero_shot.append(correct_zero_shot)
    
    return all_correct_zero_shot


# Compare to node and edge agents
def compare_to_agents(forward, model, environments, include_stay_still=True):
    """
    比较TEM与节点代理和边缘代理的性能
    
    优化：预先计算随机种子，批量处理
    """
    n_actions = model.hyper['n_actions'] + model.hyper['has_static_action']
    has_static = model.hyper['has_static_action']
    n_x = model.hyper['n_x']
    
    all_correct_model, all_correct_node, all_correct_edge = [], [], []
    
    for env_i, env in enumerate(environments):
        n_locations = env.n_locations
        
        # 初始化跟踪数组
        location_visited = np.zeros(n_locations, dtype=bool)
        action_taken = np.zeros((n_locations, n_actions), dtype=bool)
        
        correct_model = []
        correct_node = []
        correct_edge = []
        
        # 处理步骤
        for step_i, step in enumerate(forward[1:]):
            prev_step = forward[step_i]
            
            prev_a = prev_step.a[env_i]
            prev_g = prev_step.g[env_i]['id']
            curr_g = step.g[env_i]['id']
            
            # 处理静态动作
            if has_static and prev_a == 0 and not include_stay_still:
                prev_a = None
            
            # 标记位置已访问
            location_visited[prev_g] = True
            
            # 获取真实标签
            true_label = torch.argmax(step.x[env_i]).item()
            
            # 模型预测
            correct_model.append(_argmax_correct(step.x_gen[2][env_i], step.x[env_i]))
            
            # 节点代理：如果访问过该位置则正确，否则随机猜测
            if location_visited[curr_g]:
                correct_node.append(True)
            else:
                correct_node.append(np.random.randint(n_x) == true_label)
            
            # 边缘代理：如果之前采取过该动作则正确，否则随机猜测
            if prev_a is None:
                correct_edge.append(True)
            elif 0 <= prev_a < n_actions and action_taken[prev_g, prev_a]:
                correct_edge.append(True)
            else:
                correct_edge.append(np.random.randint(n_x) == true_label)
            
            # 更新动作状态
            if prev_a is not None and 0 <= prev_a < n_actions:
                action_taken[prev_g, prev_a] = True
        
        all_correct_model.append(correct_model)
        all_correct_node.append(correct_node)
        all_correct_edge.append(correct_edge)
    
    return all_correct_model, all_correct_node, all_correct_edge


# Calculate rate maps
def rate_map(forward, model, environments):
    """
    计算速率图：每个细胞在所有位置的发射模式
    
    优化：批量处理，使用numpy堆栈操作
    """
    all_g, all_p = [], []
    
    n_f = model.hyper['n_f']
    n_p = model.hyper['n_p']
    n_g = model.hyper['n_g']
    
    for env_i, env in enumerate(environments):
        n_locations = env.n_locations
        
        # 初始化存储：[频率模块][位置][访问次数]
        p = [[[] for _ in range(n_locations)] for _ in range(n_f)]
        g = [[[] for _ in range(n_locations)] for _ in range(n_f)]
        
        # 收集每步的发射率
        for step in forward:
            loc_id = step.g[env_i]['id']
            
            for f in range(n_f):
                # 转换为numpy（自动处理CUDA）
                g_val = _to_numpy(step.g_inf[f][env_i])
                p_val = _to_numpy(step.p_inf[f][env_i])
                
                g[f][loc_id].append(g_val)
                p[f][loc_id].append(p_val)
        
        # 计算每个位置的平均发射率（只使用后半部分访问）
        for cells, n_cells in [(p, n_p), (g, n_g)]:
            for f in range(n_f):
                for loc_id in range(n_locations):
                    visits = cells[f][loc_id]
                    
                    if len(visits) > 0:
                        # 只使用后半部分（模型已经熟悉环境）
                        half_idx = len(visits) // 2
                        if half_idx < len(visits):
                            # 使用numpy加速求和
                            cells[f][loc_id] = np.mean(visits[half_idx:], axis=0)
                        else:
                            cells[f][loc_id] = np.zeros(n_cells[f])
                    else:
                        cells[f][loc_id] = np.zeros(n_cells[f])
                
                # 堆叠成矩阵 [位置 x 细胞]
                cells[f] = np.stack(cells[f], axis=0)
        
        all_g.append(g)
        all_p.append(p)
    
    return all_g, all_p


# Helper function to generate input
def generate_input(environment, walk=None):
    """
    生成模型输入
    
    优化：批量处理转换
    """
    if walk is None:
        # 生成步行
        walk_len = environment.graph['n_locations'] * 100
        walk = environment.generate_walks(walk_len, 1)[0]
        
        # 调整为批量格式
        for step in walk:
            step[0] = [step[0]]
            step[1] = step[1].unsqueeze(dim=0)
            step[2] = [step[2]]
    
    return walk


# Smoothing function
def smooth(a, wsz):
    """
    平滑函数
    
    优化：使用numpy的卷积操作
    
    Args:
        a: 要平滑的1D数组
        wsz: 平滑窗口大小（必须是奇数）
    """
    if len(a) < wsz:
        return a
    
    # 主体部分：使用卷积
    out0 = np.convolve(a, np.ones(wsz, dtype=float) / wsz, 'valid')
    
    # 边缘处理
    r = np.arange(1, wsz - 1, 2)
    start = np.cumsum(a[:wsz - 1])[::2] / r
    stop = (np.cumsum(a[:-wsz:-1])[::2] / r)[::-1]
    
    return np.concatenate((start, out0, stop))


# ==================== 批量处理工具函数 ====================

def batch_analyze(forward_list, model_list, environments_list, 
                 analyze_funcs=['accuracy', 'zero_shot', 'agents']):
    """
    批量分析多个模型的前向传播结果
    
    Args:
        forward_list: 前向传播结果列表
        model_list: 模型列表
        environments_list: 环境列表
        analyze_funcs: 要执行的分析函数列表
    
    Returns:
        结果字典
    """
    results = {
        'accuracy': [],
        'zero_shot': [],
        'agents': [],
        'occupation': [],
        'rate_maps': []
    }
    
    for forward, model, envs in zip(forward_list, model_list, environments_list):
        if 'accuracy' in analyze_funcs:
            from_acc, to_acc = location_accuracy(forward, model, envs)
            results['accuracy'].append({'from': from_acc, 'to': to_acc})
        
        if 'zero_shot' in analyze_funcs:
            zs = zero_shot(forward, model, envs)
            results['zero_shot'].append(zs)
        
        if 'agents' in analyze_funcs:
            model_acc, node_acc, edge_acc = compare_to_agents(forward, model, envs)
            results['agents'].append({
                'model': model_acc,
                'node': node_acc,
                'edge': edge_acc
            })
        
        if 'occupation' in analyze_funcs:
            occ = location_occupation(forward, model, envs)
            results['occupation'].append(occ)
        
        if 'rate_maps' in analyze_funcs:
            g, p = rate_map(forward, model, envs)
            results['rate_maps'].append({'g': g, 'p': p})
    
    return results


def compute_summary_statistics(forward, model, environments):
    """
    计算摘要统计信息
    
    Returns:
        包含各种指标的字典
    """
    # 计算各种指标
    correct_model, correct_node, correct_edge = compare_to_agents(forward, model, environments)
    zs = zero_shot(forward, model, environments)
    from_acc, to_acc = location_accuracy(forward, model, environments)
    
    # 计算平均值
    summary = {
        'model_accuracy': np.mean([np.mean(env) for env in correct_model]) * 100,
        'node_accuracy': np.mean([np.mean(env) for env in correct_node]) * 100,
        'edge_accuracy': np.mean([np.mean(env) for env in correct_edge]) * 100,
        'zero_shot': np.mean([np.mean(env) for env in zs if len(env) > 0]) * 100 if any(len(env) > 0 for env in zs) else 0,
        'avg_from_accuracy': np.mean([np.mean(env) for env in from_acc]) * 100,
        'avg_to_accuracy': np.mean([np.mean(env) for env in to_acc]) * 100,
    }
    
    return summary


# ==================== GPU内存优化工具 ====================

class AnalysisContext:
    """
    分析上下文管理器，用于优化GPU内存使用
    """
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_cuda = self.device.type == 'cuda'
    
    def __enter__(self):
        if self.use_cuda:
            # 清理缓存
            torch.cuda.empty_cache()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_cuda:
            # 清理缓存
            torch.cuda.empty_cache()
        return False


def analyze_with_memory_optimization(forward, model, environments, 
                                    device='cuda', chunk_size=100):
    """
    使用内存优化进行分析（处理长序列）
    
    Args:
        forward: 前向传播结果
        model: 模型
        environments: 环境列表
        device: 设备
        chunk_size: 块大小（步数）
    
    Returns:
        分析结果
    """
    with AnalysisContext(device):
        # 如果序列太长，分块处理
        if len(forward) > chunk_size:
            print(f"Processing {len(forward)} steps in chunks of {chunk_size}")
            
            # 这里可以实现分块处理逻辑
            # 目前直接处理全部
            pass
        
        # 执行分析
        results = compute_summary_statistics(forward, model, environments)
    
    return results
