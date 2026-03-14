#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized GPU Training Script for Tolman-Eichenbaum Machine
@author: zhensun
"""

# Standard library imports
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import glob
import os
import shutil
import importlib.util
import logging
from pathlib import Path

# Own module imports
import world
import utils
import parameters
import model_bak as model_bak


class TEMTrainer:
    """Trainer class for TEM model with CUDA support"""
    
    def __init__(self, load_existing=False, date=None, run=None, i_start=0, device=None):
        """
        Initialize trainer
        
        Args:
            load_existing: Whether to load existing model
            date: Date of existing run to load
            run: Run number to load
            i_start: Starting iteration
            device: torch device to use
        """
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Set random seeds
        self._set_random_seeds()
        
        # Initialize model and paths
        if load_existing:
            self._load_existing_model(date, run, i_start)
        else:
            self._initialize_new_model()
        
        # Setup tensorboard and logger
        self.writer = SummaryWriter(self.train_path)
        self.logger = self._setup_logger()
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr_max'])
        
        # Initialize training state
        self._initialize_training_state()
        
    def _set_random_seeds(self, seed=0):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For better reproducibility (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _load_existing_model(self, date, run, i_start):
        """Load existing trained model"""
        print(f"Loading model from {date}, run {run}, iteration {i_start}")
        
        # Set paths
        self.run_path, self.train_path, self.model_path, self.save_path, \
            self.script_path, self.envs_path = utils.set_directories(date, run)
        
        # Load model module
        model_spec = importlib.util.spec_from_file_location(
            "model", str(Path(self.script_path) / 'model.py')
        )
        model_module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
        
        # Load parameters
        self.params = torch.load(
            Path(self.model_path) / f'params_{i_start}.pt',
            map_location=self.device
        )
        
        # Update parameters if needed
        new_params = {'train_it': 40000}
        self.params.update(new_params)
        
        # Move hyperparameters to device
        self._move_params_to_device()
        
        # Create model
        self.model = model_module.Model(self.params).to(self.device)
        self.model.hyper['device'] = self.device
        
        # Load weights
        model_weights = torch.load(
            Path(self.model_path) / f'tem_{i_start}.pt',
            map_location=self.device
        )
        self.model.load_state_dict(model_weights)
        
        # Load environments
        self.envs = list(glob.iglob(str(Path(self.envs_path) / '*')))
        
        # Set starting iteration
        self.i_start = i_start + 1
        
        print(f"Model loaded successfully. Starting from iteration {self.i_start}")
    
    def _initialize_new_model(self):
        """Initialize new model from scratch"""
        print("Initializing new model")
        
        self.i_start = 0
        
        # Create directories
        self.run_path, self.train_path, self.model_path, self.save_path, \
            self.script_path, self.envs_path = utils.make_directories()
        
        # Save all python files
        self._save_scripts()
        
        # Initialize parameters
        self.params = parameters.parameters()
        np.save(Path(self.save_path) / 'params', self.params)
        
        # Move hyperparameters to device
        self._move_params_to_device()
        
        # Create model
        self.model = model_bak.Model(self.params).to(self.device)
        self.model.hyper['device'] = self.device
        
        # Setup environments
        self.envs = ['./envs/5x5.json']
        for env_file in set(self.envs):
            shutil.copy2(env_file, Path(self.envs_path) / Path(env_file).name)
        
        print("Model initialized successfully")
    
    def _move_params_to_device(self):
        """Move tensor hyperparameters to device"""
        tensor_params = ['W_tile', 'W_repeat', 'g_downsample', 'two_hot_table', 
                        'p_update_mask', 'p_retrieve_mask_inf', 'p_retrieve_mask_gen']
        
        for param_name in tensor_params:
            if param_name in self.params:
                param_value = self.params[param_name]
                if isinstance(param_value, list):
                    self.params[param_name] = [
                        p.to(self.device) if torch.is_tensor(p) else p 
                        for p in param_value
                    ]
                elif torch.is_tensor(param_value):
                    self.params[param_name] = param_value.to(self.device)
    
    def _save_scripts(self):
        """Save all Python files to script directory"""
        py_files = glob.iglob('./*.py')
        for file in py_files:
            if os.path.isfile(file):
                shutil.copy2(file, Path(self.script_path) / Path(file).name)
    
    def _setup_logger(self):
        """Setup logger for training"""
        logger = logging.getLogger('TEM_Training')
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(Path(self.run_path) / 'training.log')
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _initialize_training_state(self):
        """Initialize training state variables"""
        # Create environments
        self.environments = [
            world.World(
                np.random.choice(self.envs),
                randomise_observations=True,
                shiny=(self.params['shiny'] if np.random.rand() < self.params['shiny_rate'] else None)
            )
            for _ in range(self.params['batch_size'])
        ]
        
        # Initialize visited states
        self.visited = [
            [False] * env.n_locations 
            for env in self.environments
        ]
        
        # Generate initial walks
        self.walks = [
            env.generate_walks(
                self.params['n_rollout'] * np.random.randint(
                    self.params['walk_it_min'], 
                    self.params['walk_it_max']
                ),
                1
            )[0]
            for env in self.environments
        ]
        
        # Initialize previous iteration
        self.prev_iter = None
    
    '''
    def _prepare_chunk(self, walk_length_center):
        """Prepare a chunk of data for training"""
        chunk = []
        
        for env_i, walk in enumerate(self.walks):
            # Check if walk needs regeneration
            if len(walk) < self.params['n_rollout']:
                self._regenerate_walk(env_i, walk_length_center)
            
            # Pop steps from walk
            for step in range(self.params['n_rollout']):
                if len(chunk) < self.params['n_rollout']:
                    chunk.append([[comp] for comp in walk.pop(0)])
                else:
                    for comp_i, comp in enumerate(walk.pop(0)):
                        chunk[step][comp_i].append(comp)
        
        # Stack observations and move to device
        for i_step, step in enumerate(chunk):
            chunk[i_step][1] = torch.stack(step[1], dim=0).to(self.device)
        
        return chunk
        '''
    
    def _prepare_chunk(self, walk_length_center):
        """Prepare a chunk of data for training"""
        chunk = []

        # Iterate through each environment's walk
        for env_i, walk in enumerate(self.walks):
            # Keep processing steps until we have filled the chunk or determined we can't
            steps_added_to_chunk = 0
            while steps_added_to_chunk < self.params['n_rollout']:
                
                # Check if current walk needs regeneration (is empty or too short potentially)
                # It's safer to check if it's empty here before trying to pop
                if len(walk) == 0: 
                     # Even if len was >= n_rollout initially, popping in previous iterations might have emptied it
                     # Or it was empty to begin with after regeneration logic failed somehow.
                     self._regenerate_walk(env_i, walk_length_center)
                     walk = self.walks[env_i] # Update local reference after regeneration
                     # After regeneration, check again. If still empty, there's likely an issue in _regenerate_walk or params
                     if len(walk) == 0:
                         self.logger.warning(f"Walk for env {env_i} is still empty after regeneration. Skipping this env for now.")
                         break # Skip this environment for this chunk preparation cycle

                # Pop one step from the walk (since walk is confirmed non-empty)
                step_data = walk.pop(0) 

                # Add this step's data to the chunk structure
                if len(chunk) <= steps_added_to_chunk: 
                    # If chunk doesn't have a slot for this step yet, create it
                    chunk.append([[comp] for comp in step_data])
                else:
                    # Otherwise, append components of this step to the existing slot
                    for comp_i, comp in enumerate(step_data):
                        chunk[steps_added_to_chunk][comp_i].append(comp)
                
                steps_added_to_chunk += 1
                
                # Optional: Break early if walk becomes empty *during* this process
                # This handles cases where n_rollout > regenerated walk length more gracefully within the loop
                # The outer check will catch it next iteration, but this is cleaner.
                # if len(walk) == 0: 
                #     self.logger.debug(f"Walk for env {env_i} became empty during chunk prep after adding {steps_added_to_chunk} steps.")
                #     break


        # Stack observations (component index 1) and move to device
        # Ensure chunk has elements before proceeding
        if chunk: 
             for i_step, step_components in enumerate(chunk):
                 # Ensure component index 1 (observations) exists and has elements to stack
                 if len(step_components) > 1 and step_components[1]: 
                     try:
                         chunk[i_step][1] = torch.stack(step_components[1], dim=0).to(self.device)
                     except IndexError as e:
                          self.logger.error(f"Error stacking observations at chunk step {i_step}: {e}")
                          self.logger.error(f"Step components: {[type(c) for c in step_components]}")
                          raise
                 else:
                      # Handle case where observation component is missing or empty for some steps
                      # This might be an error condition depending on your data structure assumptions
                      self.logger.warning(f"Missing or empty observation list for chunk step {i_step}")
                      # You might want to handle this differently, e.g., create dummy tensors
                      # For now, leave it as is or set to None/empty tensor
                      # chunk[i_step][1] = torch.empty(...) or chunk[i_step][1] = None 
        else:
            # Handle completely empty chunk - maybe log a warning or error?
            self.logger.warning("_prepare_chunk resulted in an empty chunk.")

        return chunk

    
    def _regenerate_walk(self, env_i, walk_length_center):
        """Regenerate walk for an environment"""
        # Create new environment
        self.environments[env_i] = world.World(
            self.envs[np.random.randint(len(self.envs))],
            randomise_observations=True,
            shiny=(self.params['shiny'] if np.random.rand() < self.params['shiny_rate'] else None)
        )
        
        # Reset visited states
        self.visited[env_i] = [False] * self.environments[env_i].n_locations
        
        # Generate new walk
        walk_length = self.params['n_rollout'] * np.random.randint(
            walk_length_center - self.params['walk_it_window'] * 0.5,
            walk_length_center + self.params['walk_it_window'] * 0.5
        )
        walk = self.environments[env_i].generate_walks(walk_length, 1)[0]
        self.walks[env_i] = walk
        
        # Reset previous iteration action
        if self.prev_iter is not None:
            self.prev_iter[0].a[env_i] = None
        
        self.logger.info(
            f'Iteration: new walk of length {len(walk)} for batch entry {env_i}'
        )
    
    def _compute_loss(self, forward, loss_weights):
        """Compute loss from forward pass"""
        loss = torch.tensor(0.0, device=self.device)
        plot_loss = np.zeros(8)  # 8 loss components
        
        for step in forward:
            step_loss = []
            
            # Only include loss for visited locations
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
    
    def _compute_accuracy(self, forward):
        """Compute model accuracies"""
        accuracies = np.mean(
            [[np.mean(a) for a in step.correct()] for step in forward],
            axis=0
        )
        return [a * 100 for a in accuracies]
    
    def _log_progress(self, iteration, loss, plot_loss, accuracies, start_time):
        """Log training progress"""
        acc_p, acc_g, acc_gt = accuracies
        
        self.logger.info(
            f'Finished backprop iter {iteration} in {time.time()-start_time:.2f} seconds.'
        )
        self.logger.info(
            f'Loss: {loss.item():.2f}. '
            f'<p_g> {plot_loss[0]:.2f} <p_x> {plot_loss[1]:.2f} '
            f'<x_gen> {plot_loss[2]:.2f} <x_g> {plot_loss[3]:.2f} '
            f'<x_p> {plot_loss[4]:.2f} <g> {plot_loss[5]:.2f} '
            f'<reg_g> {plot_loss[6]:.2f} <reg_p> {plot_loss[7]:.2f}'
        )
        self.logger.info(
            f'Accuracy: <p> {acc_p:.2f}% <g> {acc_g:.2f}% <gt> {acc_gt:.2f}%'
        )
        
        if self.prev_iter is not None:
            max_hebb = np.max(np.abs(self.prev_iter[0].M[0].detach().cpu().numpy()))
            self.logger.info(
                f'Parameters: <max_hebb> {max_hebb:.2f} '
                f'<eta> {self.model.hyper["eta"]:.2f} '
                f'<lambda> {self.model.hyper["lambda"]:.2f} '
                f'<p2g_scale_offset> {self.model.hyper["p2g_scale_offset"]:.2f}'
            )
        
        self.logger.info('')
    
    def _log_tensorboard(self, iteration, loss, plot_loss, accuracies):
        """Log to tensorboard"""
        acc_p, acc_g, acc_gt = accuracies
        
        # Losses
        self.writer.add_scalar('Losses/Total', loss.item(), iteration)
        loss_names = ['p_g', 'p_x', 'x_gen', 'x_g', 'x_p', 'g', 'reg_g', 'reg_p']
        for name, value in zip(loss_names, plot_loss):
            self.writer.add_scalar(f'Losses/{name}', value, iteration)
        
        # Accuracies
        self.writer.add_scalar('Accuracies/p', acc_p, iteration)
        self.writer.add_scalar('Accuracies/g', acc_g, iteration)
        self.writer.add_scalar('Accuracies/gt', acc_gt, iteration)
        
        # Memory usage (if CUDA)
        if torch.cuda.is_available():
            self.writer.add_scalar(
                'System/GPU_Memory_Allocated_GB',
                torch.cuda.memory_allocated() / 1e9,
                iteration
            )
            self.writer.add_scalar(
                'System/GPU_Memory_Reserved_GB',
                torch.cuda.memory_reserved() / 1e9,
                iteration
            )
    
    def _save_checkpoint(self, iteration):
        """Save model checkpoint"""
        torch.save(
            self.model.state_dict(),
            Path(self.model_path) / f'tem_{iteration}.pt'
        )
        torch.save(
            self.model.hyper,
            Path(self.model_path) / f'params_{iteration}.pt'
        )
        self.logger.info(f'Checkpoint saved at iteration {iteration}')
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training")
        self.logger.info(f"Training from iteration {self.i_start} to {self.params['train_it']}")
        
        for i in range(self.i_start, self.params['train_it']):
            start_time = time.time()
            
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
            
            # Move previous iteration to device if exists
            if self.prev_iter is not None:
                # Note: detach() returns tensor on same device, no need to move
                pass
            
            # Forward pass
            forward = self.model(chunk, self.prev_iter)
            
            # Compute loss
            loss, plot_loss = self._compute_loss(forward, loss_weights)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            # Update previous iteration
            self.prev_iter = [forward[-1].detach()]
            
            # Compute accuracies
            accuracies = self._compute_accuracy(forward)
            
            # Logging
            if i % 10 == 0:
                self._log_progress(i, loss, plot_loss, accuracies, start_time)
                self._log_tensorboard(i, loss, plot_loss, accuracies)
            
            # Save checkpoint
            if i % 1000 == 0:
                self._save_checkpoint(i)
            
            # Clear cache periodically
            if i % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save final model
        self._save_checkpoint(self.params['train_it'] - 1)
        self.logger.info("Training completed")
        self.writer.close()


def main():
    """Main entry point"""
    # Configuration
    load_existing_model = True
    
    if load_existing_model:
        # Load existing model
        trainer = TEMTrainer(
            load_existing=True,
            date='2024-12-22',
            run='1',
            i_start=0
        )
    else:
        # Create new model
        trainer = TEMTrainer(load_existing=False)
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.logger.info("Training interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        trainer.logger.error(f"Error during training: {e}", exc_info=True)
        raise
    finally:
        if torch.cuda.is_available():
            print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
