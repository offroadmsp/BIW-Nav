#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced TEM Trainer with AMP support and advanced features
"""

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import time
from pathlib import Path
from collections import defaultdict

from run_gpu import TEMTrainer


class EnhancedTEMTrainer(TEMTrainer):
    """Enhanced trainer with AMP, gradient clipping, and more features"""
    
    def __init__(self, config):
        """
        Initialize enhanced trainer
        
        Args:
            config: TrainingConfig object
        """
        self.config = config
        
        # Initialize parent
        super().__init__(
            load_existing=config.load_existing,
            date=config.date if config.load_existing else None,
            run=config.run if config.load_existing else None,
            i_start=config.start_iteration,
            device=config.get_device()
        )
        
        # Update training iterations
        self.params['train_it'] = config.total_iterations
        
        # Setup AMP if requested
        self.use_amp = config.use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("Using Automatic Mixed Precision (AMP)")
        
        # Gradient clipping
        self.gradient_clip = config.gradient_clip
        if self.gradient_clip:
            self.logger.info(f"Gradient clipping enabled: {self.gradient_clip}")
        
        # Training statistics
        self.stats = defaultdict(list)
        
        # Timing
        self.total_train_time = 0
        
    def _compute_loss_amp(self, forward, loss_weights):
        """Compute loss with AMP support"""
        loss = torch.tensor(0.0, device=self.device)
        plot_loss = np.zeros(8)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            for step in forward:
                step_loss = []
                
                for env_i, env_visited in enumerate(self.visited):
                    if env_visited[step.g[env_i]['id']]:
                        step_loss.append(
                            loss_weights.to(self.device) * torch.stack([l[env_i] for l in step.L])
                        )
                    else:
                        env_visited[step.g[env_i]['id']] = True
                
                if step_loss:
                    step_loss = torch.mean(torch.stack(step_loss, dim=0), dim=0)
                    plot_loss += step_loss.detach().cpu().numpy()
                    loss += torch.sum(step_loss)
        
        return loss, plot_loss
    
    def _backward_step(self, loss):
        """Perform backward pass with optional AMP and gradient clipping"""
        self.optimizer.zero_grad()
        
        if self.use_amp:
            # AMP backward
            self.scaler.scale(loss).backward(retain_graph=True)
            
            # Gradient clipping
            if self.gradient_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard backward
            loss.backward(retain_graph=True)
            
            # Gradient clipping
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
    
    def _update_stats(self, iteration, loss, plot_loss, accuracies, iteration_time):
        """Update training statistics"""
        self.stats['iteration'].append(iteration)
        self.stats['total_loss'].append(loss.item())
        self.stats['iteration_time'].append(iteration_time)
        
        # Individual losses
        loss_names = ['p_g', 'p_x', 'x_gen', 'x_g', 'x_p', 'g', 'reg_g', 'reg_p']
        for name, value in zip(loss_names, plot_loss):
            self.stats[f'loss_{name}'].append(value)
        
        # Accuracies
        self.stats['acc_p'].append(accuracies[0])
        self.stats['acc_g'].append(accuracies[1])
        self.stats['acc_gt'].append(accuracies[2])
        
        # GPU memory if available
        if torch.cuda.is_available():
            self.stats['gpu_memory_gb'].append(torch.cuda.memory_allocated() / 1e9)
    
    def _log_extended_stats(self, iteration):
        """Log extended statistics"""
        if len(self.stats['iteration']) < 2:
            return
        
        # Get recent statistics (last 100 iterations)
        recent_n = min(100, len(self.stats['iteration']))
        
        avg_time = np.mean(self.stats['iteration_time'][-recent_n:])
        avg_loss = np.mean(self.stats['total_loss'][-recent_n:])
        avg_acc_p = np.mean(self.stats['acc_p'][-recent_n:])
        avg_acc_g = np.mean(self.stats['acc_g'][-recent_n:])
        avg_acc_gt = np.mean(self.stats['acc_gt'][-recent_n:])
        
        # Estimate time to completion
        remaining_iters = self.params['train_it'] - iteration
        est_time_remaining = remaining_iters * avg_time / 3600  # in hours
        
        self.logger.info(
            f'Recent averages (last {recent_n}): '
            f'Loss={avg_loss:.2f}, '
            f'Acc_p={avg_acc_p:.1f}%, '
            f'Acc_g={avg_acc_g:.1f}%, '
            f'Acc_gt={avg_acc_gt:.1f}%, '
            f'Time={avg_time:.2f}s/iter'
        )
        self.logger.info(
            f'Estimated time to completion: {est_time_remaining:.2f} hours'
        )
        
        if torch.cuda.is_available() and len(self.stats['gpu_memory_gb']) > 0:
            avg_gpu_mem = np.mean(self.stats['gpu_memory_gb'][-recent_n:])
            max_gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            self.logger.info(
                f'GPU Memory: Avg={avg_gpu_mem:.2f}GB, Max={max_gpu_mem:.2f}GB'
            )
    
    def _save_training_stats(self):
        """Save training statistics"""
        stats_path = Path(self.save_path) / 'training_stats.npz'
        np.savez(stats_path, **self.stats)
        self.logger.info(f'Training statistics saved to {stats_path}')
    
    def train(self):
        """Enhanced training loop"""
        self.logger.info("Starting enhanced training")
        self.logger.info(f"Training from iteration {self.i_start} to {self.params['train_it']}")
        self.logger.info(f"Using AMP: {self.use_amp}")
        self.logger.info(f"Gradient clipping: {self.gradient_clip}")
        
        train_start_time = time.time()
        
        try:
            for i in range(self.i_start, self.params['train_it']):
                iter_start_time = time.time()
                
                # Get updated parameters
                eta_new, lambda_new, p2g_scale_offset, lr, walk_length_center, loss_weights = \
                    parameters.parameter_iteration(i, self.params)
                
                # Update model hyperparameters
                self.model.hyper['eta'] = eta_new
                self.model.hyper['lambda'] = lambda_new
                self.model.hyper['p2g_scale_offset'] = p2g_scale_offset
                
                # Update learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Prepare data chunk
                chunk = self._prepare_chunk(walk_length_center)
                
                # Forward pass with AMP
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    forward = self.model(chunk, self.prev_iter)
                
                # Compute loss
                loss, plot_loss = self._compute_loss_amp(forward, loss_weights)
                
                # Backward pass with AMP and gradient clipping
                self._backward_step(loss)
                
                # Update previous iteration
                self.prev_iter = [forward[-1].detach()]
                
                # Compute accuracies
                accuracies = self._compute_accuracy(forward)
                
                # Update statistics
                iter_time = time.time() - iter_start_time
                self._update_stats(i, loss, plot_loss, accuracies, iter_time)
                
                # Logging
                if i % self.config.log_interval == 0:
                    self._log_progress(i, loss, plot_loss, accuracies, iter_start_time)
                    self._log_tensorboard(i, loss, plot_loss, accuracies)
                    
                    # Extended stats every 100 iterations
                    if i % 100 == 0:
                        self._log_extended_stats(i)
                
                # Save checkpoint
                if i % self.config.save_interval == 0 and i > 0:
                    self._save_checkpoint(i)
                    self._save_training_stats()
                
                # Clear cache periodically
                if i % self.config.clear_cache_interval == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Training completed
            self.total_train_time = time.time() - train_start_time
            
            # Save final model and stats
            final_iter = self.params['train_it'] - 1
            self._save_checkpoint(final_iter)
            self._save_training_stats()
            
            self.logger.info("=" * 60)
            self.logger.info("Training completed successfully!")
            self.logger.info(f"Total training time: {self.total_train_time/3600:.2f} hours")
            self.logger.info(f"Average time per iteration: {self.total_train_time/(self.params['train_it']-self.i_start):.2f} seconds")
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}", exc_info=True)
            # Save emergency checkpoint
            self._save_checkpoint(i)
            self._save_training_stats()
            raise
        
        finally:
            self.writer.close()
            
            # Clean up
            if torch.cuda.is_available():
                final_mem = torch.cuda.memory_allocated() / 1e9
                max_mem = torch.cuda.max_memory_allocated() / 1e9
                self.logger.info(f"Final GPU memory: {final_mem:.2f} GB")
                self.logger.info(f"Peak GPU memory: {max_mem:.2f} GB")
                torch.cuda.empty_cache()


def main():
    """Main entry point with config support"""
    from training_config import TrainingConfig, get_full_training_config
    
    # Option 1: Parse from command line
    # config = TrainingConfig.from_args()
    
    # Option 2: Use predefined config
    config = get_full_training_config()
    config.load_existing = True
    config.date = '2024-12-22'
    config.run = '1'
    config.start_iteration = 0
    
    # Print configuration
    config.print_config()
    
    # Create trainer
    trainer = EnhancedTEMTrainer(config)
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.logger.info("Training interrupted by user")
        trainer._save_training_stats()
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
