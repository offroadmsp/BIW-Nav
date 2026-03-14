#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training configuration for TEM model
"""

import argparse
import torch


class TrainingConfig:
    """Configuration class for TEM training"""
    
    def __init__(self):
        # Device settings
        self.device = None  # Will auto-detect
        self.cuda_benchmark = False  # Set True for better performance (less reproducibility)
        
        # Model settings
        self.load_existing = False
        self.date = '2024-12-22'
        self.run = '1'
        self.start_iteration = 0
        
        # Training settings
        self.total_iterations = 40000
        self.log_interval = 10
        self.save_interval = 1000
        self.clear_cache_interval = 100
        
        # Optimization settings
        self.use_amp = False  # Automatic Mixed Precision
        self.gradient_clip = None  # Set to value like 1.0 to enable gradient clipping
        
        # Environment settings
        self.env_files = ['./envs/5x5.json']
        
        # Random seed
        self.seed = 0
        
        # Memory management
        self.pin_memory = True  # Faster data transfer to GPU
        self.non_blocking = True  # Asynchronous data transfer
        
    @classmethod
    def from_args(cls):
        """Create config from command line arguments"""
        parser = argparse.ArgumentParser(description='Train TEM model')
        
        # Device
        parser.add_argument('--device', type=str, default=None,
                          help='Device to use (cuda/cpu/cuda:0/etc)')
        parser.add_argument('--cuda-benchmark', action='store_true',
                          help='Enable CUDNN benchmark for better performance')
        
        # Model loading
        parser.add_argument('--load', action='store_true',
                          help='Load existing model')
        parser.add_argument('--date', type=str, default='2024-12-22',
                          help='Date of model to load')
        parser.add_argument('--run', type=str, default='1',
                          help='Run number to load')
        parser.add_argument('--start-iter', type=int, default=0,
                          help='Starting iteration')
        
        # Training
        parser.add_argument('--iterations', type=int, default=40000,
                          help='Total training iterations')
        parser.add_argument('--log-interval', type=int, default=10,
                          help='Logging interval')
        parser.add_argument('--save-interval', type=int, default=1000,
                          help='Model saving interval')
        
        # Optimization
        parser.add_argument('--amp', action='store_true',
                          help='Use automatic mixed precision')
        parser.add_argument('--grad-clip', type=float, default=None,
                          help='Gradient clipping value')
        
        # Environment
        parser.add_argument('--env-files', type=str, nargs='+',
                          default=['./envs/5x5.json'],
                          help='Environment files to use')
        
        # Misc
        parser.add_argument('--seed', type=int, default=0,
                          help='Random seed')
        
        args = parser.parse_args()
        
        # Create config
        config = cls()
        config.device = args.device
        config.cuda_benchmark = args.cuda_benchmark
        config.load_existing = args.load
        config.date = args.date
        config.run = args.run
        config.start_iteration = args.start_iter
        config.total_iterations = args.iterations
        config.log_interval = args.log_interval
        config.save_interval = args.save_interval
        config.use_amp = args.amp
        config.gradient_clip = args.grad_clip
        config.env_files = args.env_files
        config.seed = args.seed
        
        return config
    
    def get_device(self):
        """Get the appropriate device"""
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def print_config(self):
        """Print configuration"""
        print("=" * 60)
        print("Training Configuration")
        print("=" * 60)
        print(f"Device: {self.get_device()}")
        print(f"CUDA Benchmark: {self.cuda_benchmark}")
        print(f"Load Existing: {self.load_existing}")
        if self.load_existing:
            print(f"  Date: {self.date}")
            print(f"  Run: {self.run}")
            print(f"  Start Iteration: {self.start_iteration}")
        print(f"Total Iterations: {self.total_iterations}")
        print(f"Log Interval: {self.log_interval}")
        print(f"Save Interval: {self.save_interval}")
        print(f"Use AMP: {self.use_amp}")
        print(f"Gradient Clipping: {self.gradient_clip}")
        print(f"Environment Files: {self.env_files}")
        print(f"Random Seed: {self.seed}")
        print("=" * 60)


# Example usage configurations
def get_quick_test_config():
    """Get configuration for quick testing"""
    config = TrainingConfig()
    config.total_iterations = 100
    config.log_interval = 10
    config.save_interval = 50
    return config


def get_full_training_config():
    """Get configuration for full training"""
    config = TrainingConfig()
    config.total_iterations = 40000
    config.log_interval = 10
    config.save_interval = 1000
    config.use_amp = True  # Use mixed precision for faster training
    config.gradient_clip = 1.0  # Prevent exploding gradients
    return config


def get_continue_training_config(date, run, start_iter):
    """Get configuration for continuing training"""
    config = TrainingConfig()
    config.load_existing = True
    config.date = date
    config.run = run
    config.start_iteration = start_iter
    config.total_iterations = 80000  # Train longer
    config.use_amp = True
    config.gradient_clip = 1.0
    return config


if __name__ == "__main__":
    # Example: Parse from command line
    config = TrainingConfig.from_args()
    config.print_config()
    
    # Example: Use predefined configs
    # config = get_quick_test_config()
    # config = get_full_training_config()
    # config = get_continue_training_config('2024-12-22', '1', 1000)
