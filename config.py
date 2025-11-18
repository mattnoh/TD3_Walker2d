"""Unified configuration settings for TD3 training"""
import torch
import numpy as np
import random
import time


def set_random_seeds(seed=None):
    """Set random seeds"""
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
    print(f"Using seed: {seed}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    return seed


def get_config(training_type='speed_walking'):
    """
    Get training configuration dictionary
    
    Args:
        training_type: Type of training - 'speed_walking' or 'gait_walking'
    """
    # Generate a new random seed each time config is created
    random_seed = int(time.time() * 1000) % (2**32)
    
    base_config = {
        # Training parameters
        'seed': random_seed,
        'max_episodes': 3000,
        'max_steps': 1000,
        'warmup_steps': 10000,
        
        # TD3 hyperparameters
        'buffer_size': 1_000_000,
        'batch_size': 256,
        'gamma': 0.99,
        'tau': 0.005,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'policy_freq': 2,
        
        # Exploration parameters
        'exploration_noise': 0.15,
        'exploration_noise_decay': 0.9998,
        'min_exploration_noise': 0.05,
        
        # Logging parameters
        'log_interval': 10,
        'eval_interval': 100,
        'save_interval': 100,
        
        # Training type identifier
        'training_type': training_type,
    }
    
    # Training type specific configurations
    if training_type == 'speed_walking':
        speed_config = {
            # Reward shaping parameters
            'velocity_weight': 2.0,
            'consistency_weight': 0.3,
            'sustained_velocity_window': 150,
            'sustained_velocity_bonus': 12.0,
        }
        return {**base_config, **speed_config}
    
    elif training_type == 'gait_walking':
        gait_config = {
            # Velocity parameters
            'velocity_weight': 2.0,
            'consistency_weight': 0.3,
            'sustained_velocity_window': 150,
            'sustained_velocity_bonus': 12.0,
            
            # Gait quality parameters
            'rom_weight': 1.5,
            'antiphase_weight': 2.0,
            'periodicity_weight': 1.5, # Reward for rhythmic gait
        }
        return {**base_config, **gait_config}
    
    else:
        raise ValueError(
            f"Unknown training type: {training_type}. "
            "Supported types: 'speed_walking', 'gait_walking'"
        )


def get_device():
    """Get compute device (CUDA if available, else CPU)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device