#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Testing Script - Test multiple model checkpoints
支持批量测试和性能对比
"""

import numpy as np
import torch
import glob
import matplotlib.pyplot as plt
import importlib.util
import time
import os
from pathlib import Path
import argparse

import analyse
import world
import analyse_cpu
import plot


class ModelTester:
    """模型测试器类"""
    
    def __init__(self, date, run, index, device=None, verbose=True):
        """
        初始化测试器
        
        Args:
            date: 模型日期
            run: 运行编号
            index: 检查点索引
            device: 计算设备
            verbose: 是否打印详细信息
        """
        self.date = date
        self.run = run
        self.index = index
        self.verbose = verbose
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 构建路径
        self.summary_path = f'./Summaries/{date}/run{run}'
        self.script_path = f'{self.summary_path}/script'
        self.model_path = f'{self.summary_path}/model'
        self.envs_path = f'{self.script_path}/envs'
        
        # 加载模型
        self._load_model()
        
        # 结果存储
        self.results = {}
        
    def _log(self, message):
        """打印日志"""
        if self.verbose:
            print(message)
    
    def _load_model(self):
        """加载模型"""
        self._log(f"Loading model: {self.date}, run {self.run}, iteration {self.index}")
        
        # 动态导入模型
        model_spec = importlib.util.spec_from_file_location(
            "model", f'{self.script_path}/model.py'
        )
        model_module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
        
        # 加载参数
        self.params = torch.load(
            f'{self.model_path}/params_{self.index}.pt',
            map_location=self.device
        )
        
        # 移动超参数到设备
        self._move_params_to_device()
        
        # 创建模型
        self.model = model_module.Model(self.params).to(self.device)
        
        # 加载权重
        model_weights = torch.load(
            f'{self.model_path}/tem_{self.index}.pt',
            map_location=self.device
        )
        self.model.load_state_dict(model_weights)
        
        # 评估模式
        self.model.eval()
        
        self._log("Model loaded successfully")
    
    def _move_params_to_device(self):
        """移动张量超参数到设备"""
        tensor_params = ['W_tile', 'W_repeat', 'g_downsample', 'two_hot_table', 
                        'p_update_mask', 'p_retrieve_mask_inf', 'p_retrieve_mask_gen']
        
        for param_name in tensor_params:
            if param_name in self.params:
                param = self.params[param_name]
                if isinstance(param, list):
                    self.params[param_name] = [
                        p.to(self.device) if torch.is_tensor(p) else p 
                        for p in param
                    ]
                elif torch.is_tensor(param):
                    self.params[param_name] = param.to(self.device)
    
    def prepare_data(self, shiny_envs=None, walk_len_multiplier=50):
        """
        准备测试数据
        
        Args:
            shiny_envs: 哪些环境包含闪亮物体
            walk_len_multiplier: 步行长度乘数
        """
        if shiny_envs is None:
            shiny_envs = [False, False, True, True]
        
        self.shiny_envs = shiny_envs
        self._log(f"Preparing data with {len(shiny_envs)} environments...")
        
        # 加载环境
        envs = list(glob.iglob(f'{self.envs_path}/*'))
        
        # 创建环境
        self.environments = [
            world.World(
                np.random.choice(envs),
                randomise_observations=True,
                shiny=(self.params['shiny'] if shiny_envs[i] else None)
            )
            for i in range(len(shiny_envs))
        ]
        
        # 生成步行
        walk_len = int(np.median([env.n_locations * walk_len_multiplier 
                                   for env in self.environments]))
        self.walks = [env.generate_walks(walk_len, 1)[0] for env in self.environments]
        
        # 准备模型输入
        walk_len = len(self.walks[0])
        self.model_input = [
            [[self.walks[i][j][k] for i in range(len(self.walks))] for k in range(3)]
            for j in range(walk_len)
        ]
        
        # 移动到设备
        for i_step in range(len(self.model_input)):
            self.model_input[i_step][1] = torch.stack(
                self.model_input[i_step][1], dim=0
            ).to(self.device)
        
        self._log(f"Data prepared: {walk_len} steps")
    
    def run_forward(self):
        """运行前向传播"""
        self._log("Running forward pass...")
        
        start_time = time.time()
        with torch.no_grad():
            self.forward = self.model(self.model_input, prev_iter=None)
        
        self.forward_time = time.time() - start_time
        self._log(f"Forward pass: {self.forward_time:.2f}s "
                 f"({self.forward_time/len(self.model_input)*1000:.2f}ms/step)")
        
        return self.forward
    
    def analyze(self, include_stay_still=True):
        """分析结果"""
        self._log("Analyzing results...")
        
        # 代理比较
        correct_model, correct_node, correct_edge = analyse.compare_to_agents(
            self.forward, self.model, self.environments, 
            include_stay_still=include_stay_still
        )
        
        # 零样本推理
        zero_shot = analyse.zero_shot(
            self.forward, self.model, self.environments,
            include_stay_still=include_stay_still
        )
        
        # 占用
        occupation = analyse.location_occupation(
            self.forward, self.model, self.environments
        )
        
        # 速率图
        g, p = analyse.rate_map(self.forward, self.model, self.environments)
        
        # 位置准确率
        from_acc, to_acc = analyse.location_accuracy(
            self.forward, self.model, self.environments
        )
        
        # 存储结果
        self.results = {
            'correct_model': correct_model,
            'correct_node': correct_node,
            'correct_edge': correct_edge,
            'zero_shot': zero_shot,
            'occupation': occupation,
            'g': g,
            'p': p,
            'from_acc': from_acc,
            'to_acc': to_acc
        }
        
        # 计算汇总统计
        self.results['avg_model_acc'] = np.mean([np.mean(env) for env in correct_model])
        self.results['avg_zero_shot'] = np.mean([np.mean(env) for env in zero_shot])
        
        self._log(f"Model accuracy: {self.results['avg_model_acc']*100:.2f}%")
        self._log(f"Zero-shot: {self.results['avg_zero_shot']*100:.2f}%")
        
        return self.results
    
    def plot_results(self, output_dir='./Pics', env_to_plot=0):
        """绘制结果"""
        self._log("Plotting results...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 选择要平均的环境
        envs_to_avg = (self.shiny_envs if self.shiny_envs[env_to_plot] 
                      else [not s for s in self.shiny_envs])
        
        # 图1: 代理比较
        filt_size = 41
        fig = plt.figure(figsize=(3.5, 2.2), dpi=600)
        
        avg_model = np.mean([env for i, env in enumerate(self.results['correct_model']) 
                            if envs_to_avg[i]], 0)[1:]
        avg_node = np.mean([env for i, env in enumerate(self.results['correct_node']) 
                           if envs_to_avg[i]], 0)[1:]
        avg_edge = np.mean([env for i, env in enumerate(self.results['correct_edge']) 
                           if envs_to_avg[i]], 0)[1:]
        
        plt.plot(analyse.smooth(avg_model, filt_size), label='BIW', linewidth=1.5)
        plt.plot(analyse.smooth(avg_node, filt_size), label='node', linewidth=1.5)
        plt.plot(analyse.smooth(avg_edge, filt_size), label='edge', linewidth=1.5)
        plt.ylim(0, 1)
        plt.legend(frameon=False)
        plt.xlabel('Indexed step')
        plt.ylabel('Precision')
        
        zs_avg = self.results['avg_zero_shot'] * 100
        plt.title(f'Zero-shot: {zs_avg:.1f}%', fontdict={'family':'Arial', 'size':10})
        plt.tight_layout()
        
        filename = f'{output_dir}/iter_{self.index}_inference.png'
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()
        self._log(f"  Saved: {filename}")
        
        # 图2: 位置准确率
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3), dpi=600)
        
        plot.plot_map(self.environments[env_to_plot], 
                     np.array(self.results['to_acc'][env_to_plot]), ax1)
        ax1.set_title('Arriving', fontsize=9)
        
        plot.plot_map(self.environments[env_to_plot], 
                     np.array(self.results['from_acc'][env_to_plot]), ax2)
        ax2.set_title('Leaving', fontsize=9)
        
        fig.suptitle('Location Accuracy', fontdict={'family':'Arial', 'size':10})
        plt.tight_layout()
        
        filename = f'{output_dir}/iter_{self.index}_accuracy.png'
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()
        self._log(f"  Saved: {filename}")
    
    def get_summary(self):
        """获取结果摘要"""
        return {
            'date': self.date,
            'run': self.run,
            'index': self.index,
            'device': str(self.device),
            'forward_time': self.forward_time,
            'avg_model_acc': self.results.get('avg_model_acc', 0),
            'avg_zero_shot': self.results.get('avg_zero_shot', 0),
        }


def batch_test(date, run, indices, output_dir='./Pics/batch_test', device=None):
    """
    批量测试多个检查点
    
    Args:
        date: 模型日期
        run: 运行编号
        indices: 检查点索引列表
        output_dir: 输出目录
        device: 计算设备
    """
    print("=" * 60)
    print("Batch Testing Multiple Checkpoints")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = []
    
    for idx in indices:
        print(f"\n{'='*60}")
        print(f"Testing checkpoint {idx}")
        print(f"{'='*60}\n")
        
        try:
            # 创建测试器
            tester = ModelTester(date, run, str(idx), device=device)
            
            # 准备数据
            tester.prepare_data()
            
            # 运行前向传播
            tester.run_forward()
            
            # 分析
            tester.analyze()
            
            # 绘图
            tester.plot_results(output_dir=output_dir)
            
            # 保存摘要
            summary = tester.get_summary()
            results_summary.append(summary)
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error testing checkpoint {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # 绘制对比图
    if results_summary:
        plot_comparison(results_summary, output_dir)
    
    return results_summary


def plot_comparison(results_summary, output_dir):
    """绘制多个检查点的对比图"""
    print("\nPlotting comparison...")
    
    indices = [r['index'] for r in results_summary]
    accuracies = [r['avg_model_acc'] * 100 for r in results_summary]
    zero_shots = [r['avg_zero_shot'] * 100 for r in results_summary]
    times = [r['forward_time'] for r in results_summary]
    
    # 图1: 准确率对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=300)
    
    ax1.plot(indices, accuracies, 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('Checkpoint Iteration')
    ax1.set_ylabel('Model Accuracy (%)')
    ax1.set_title('Model Accuracy vs Training Progress')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(indices, zero_shots, 's-', linewidth=2, markersize=6, color='orange')
    ax2.set_xlabel('Checkpoint Iteration')
    ax2.set_ylabel('Zero-shot Inference (%)')
    ax2.set_title('Zero-shot Performance vs Training Progress')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/comparison_accuracy.png")
    
    # 图2: 性能对比
    fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
    ax.bar(range(len(indices)), times, tick_label=indices)
    ax.set_xlabel('Checkpoint Iteration')
    ax.set_ylabel('Forward Pass Time (s)')
    ax.set_title('Inference Speed Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_speed.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/comparison_speed.png")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test TEM model')
    parser.add_argument('--date', type=str, default='2024-12-22', help='Model date')
    parser.add_argument('--run', type=str, default='1', help='Run number')
    parser.add_argument('--index', type=str, default='39000', help='Checkpoint index')
    parser.add_argument('--batch', action='store_true', help='Batch test multiple checkpoints')
    parser.add_argument('--indices', type=str, nargs='+', 
                       help='Checkpoint indices for batch testing')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device (cuda/cpu/cuda:0)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.batch:
        # 批量测试
        if args.indices:
            indices = [int(idx) for idx in args.indices]
        else:
            # 默认测试这些检查点
            indices = [1000, 5000, 10000, 20000, 30000, 39000]
        
        batch_test(args.date, args.run, indices, device=device)
    else:
        # 单个测试
        tester = ModelTester(args.date, args.run, args.index, device=device)
        tester.prepare_data()
        tester.run_forward()
        tester.analyze()
        tester.plot_results()
        
        print("\n" + "=" * 60)
        print("Testing Summary")
        print("=" * 60)
        for key, value in tester.get_summary().items():
            print(f"{key}: {value}")
        print("=" * 60)


if __name__ == "__main__":
    # 可以直接运行或使用命令行参数
    # python batch_test.py --date 2024-12-22 --run 1 --index 39000
    # python batch_test.py --date 2024-12-22 --run 1 --batch --indices 1000 10000 39000
    
    main()
