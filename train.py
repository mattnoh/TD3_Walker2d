"""
Unified training script for TD3 on Walker2d-v5
"""
import numpy as np
import gymnasium as gym
import argparse
import os
from config import get_config, get_device, set_random_seeds
from td3_agent import TD3Agent, ReplayBuffer
from logging_utils import create_logger, get_model_path
from utils_vel import compute_velocity_reward, evaluate_velocity_agent
from utils_sym import compute_gait_reward, evaluate_gait_agent, GaitTracker


def train_td3(
    training_type='speed_walking',
    env_name='Walker2d-v5',
    use_wandb=True,
    project_name=None,
    save_best=True,
    save_final=True):
    """
    Unified training function for both speed walking and gait walking
    
    Args:
        training_type: Type of training - 'speed_walking' or 'gait_walking'  
    Returns:
        Trained agent and training logger
    """
    # Get configuration and setup
    config = get_config(training_type)
    device = get_device()
    set_random_seeds(config['seed'])
    
    # Set default project name if not provided
    if project_name is None:
        project_name = f"td3-walker2d-{training_type}"
    
    # Create run name based on seed
    run_name = f"seed_{config['seed']}"
    
    # Initialize logger with run name
    logger = create_logger(training_type=training_type, use_wandb=use_wandb, config=config, run_name=run_name)
    
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    print(f"\n{'='*60}")
    print(f"{training_type.upper()} TRAINING SETUP")
    print(f"{'='*60}")
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    print(f"Device: {device}")
    print(f"Training Type: {training_type}")
    
    
    # Initialize agent and replay buffer
    agent = TD3Agent(state_dim, action_dim, max_action, config, device)
    replay_buffer = ReplayBuffer(state_dim, action_dim, config['buffer_size'], device)
    
    # Training state
    total_steps = 0
    current_exploration_noise = config['exploration_noise']
    best_metric = -float('inf')
    episode_rewards_window = []
    
    min_sustained_velocity_window = []
    gait_quality_window = []
    
    print(f"Starting training for {config['max_episodes']} episodes...")
    print(f"Warmup steps: {config['warmup_steps']}\n")
    
    for episode in range(config['max_episodes']):
        state, _ = env.reset(seed=config['seed'] + episode)
        episode_reward = 0
        episode_shaped_reward = 0
        episode_steps = 0
        episode_actions = []
        velocities = []  
        gait_tracker = GaitTracker()
                
        for step in range(config['max_steps']):
            # Select action
            if total_steps < config['warmup_steps']:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise=current_exploration_noise)
            
            # Track action magnitude
            episode_actions.append(np.abs(action))
            
            # Execute action
            next_state, base_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Always track velocity and gait
            current_velocity = state[8] if len(state) > 8 else 0.0
            velocities.append(current_velocity)
            gait_tracker.update(state)
            
            # Compute shaped reward based on training type
            if training_type == 'speed_walking':
                shaped_reward = compute_velocity_reward(
                    base_reward, velocities, current_velocity, episode_steps, config
                )
            else:  # gait_walking
                shaped_reward = compute_gait_reward(
                    base_reward, state, gait_tracker, velocities, config, episode_steps
                )
            
            # Store transition with shaped reward
            replay_buffer.add(state, action, shaped_reward, next_state, float(done))
            
            state = next_state
            episode_reward += base_reward
            episode_shaped_reward += shaped_reward
            episode_steps += 1
            total_steps += 1
            
            # Train agent after warmup
            if total_steps >= config['warmup_steps']:
                losses = agent.train(replay_buffer, config['batch_size'])
                
                # Log training metrics periodically
                if total_steps % 100 == 0:
                    log_dict = {
                        'train/critic_loss': losses['critic_loss'],
                        'train/exploration_noise': current_exploration_noise,
                    }
                    
                    # Only log actor loss when it's computed
                    if losses['actor_loss'] is not None:
                        log_dict['train/actor_loss'] = losses['actor_loss']
                    
                    logger.log_step(log_dict, total_steps)
            
            if done:
                break
        
        # Decay exploration noise
        if total_steps >= config['warmup_steps']:
            current_exploration_noise = max(
                config['min_exploration_noise'],
                current_exploration_noise * config['exploration_noise_decay']
            )
        
        # Calculate episode metrics
        episode_rewards_window.append(episode_reward)
        if len(episode_rewards_window) > 100:
            episode_rewards_window.pop(0)
        
        avg_reward_100 = np.mean(episode_rewards_window)
        avg_action_magnitude = np.mean([np.mean(a) for a in episode_actions])
        
        # Always compute gait metrics for both training types
        gait_metrics = gait_tracker.compute_gait_metrics()
        
        # Calculate velocity metrics (for both types)
        max_velocity = max(velocities) if velocities else 0
        window = config.get('sustained_velocity_window', 100)
        if len(velocities) >= window:
            min_sustained_velocity = np.min(velocities[-window:])
            avg_sustained_velocity = np.mean(velocities[-window:])
        else:
            min_sustained_velocity = None
            avg_sustained_velocity = None
        
        # Prepare episode metrics for logging
        episode_metrics = {
            'episode/reward': episode_reward,
            'episode/shaped_reward': episode_shaped_reward,
            'episode/length': episode_steps,
            'episode/avg_reward_100': avg_reward_100,
            'episode/avg_action_magnitude': avg_action_magnitude,
            'episode/max_velocity': max_velocity,
        }
        
        if min_sustained_velocity is not None:
            episode_metrics['episode/min_sustained_velocity'] = min_sustained_velocity
            episode_metrics['episode/avg_sustained_velocity'] = avg_sustained_velocity
        
        # Add gait metrics with appropriate prefixes
        if training_type == 'speed_walking':
            # Track min sustained velocity for model saving
            if min_sustained_velocity is not None:
                min_sustained_velocity_window.append(min_sustained_velocity)
                if len(min_sustained_velocity_window) > 100:
                    min_sustained_velocity_window.pop(0)
            
            episode_metrics.update({
                'comparison/gait_quality': gait_metrics.get('gait_quality', 0),
                'comparison/antiphase_score': gait_metrics.get('antiphase_score', 0),
                'comparison/rom_symmetry': gait_metrics.get('rom_symmetry', 0),
                'comparison/periodicity_score': gait_metrics.get('periodicity_score', 0),
                'comparison/avg_rom': gait_metrics.get('avg_rom', 0),
            })
                
        else:  # gait_walking
            # Track gait quality for model saving
            gait_quality = gait_metrics.get('gait_quality', 0)
            gait_quality_window.append(gait_quality)
            if len(gait_quality_window) > 100:
                gait_quality_window.pop(0)
            
            # Add gait metrics as primary metrics
            episode_metrics.update({
                'episode/gait_quality': gait_metrics.get('gait_quality', 0),
                'episode/antiphase_score': gait_metrics.get('antiphase_score', 0),
                'episode/rom_symmetry': gait_metrics.get('rom_symmetry', 0),
                'episode/periodicity_score': gait_metrics.get('periodicity_score', 0),
                'episode/avg_rom': gait_metrics.get('avg_rom', 0),
                'episode/vel_consistency': gait_metrics.get('vel_consistency', 0),
            })
        
        logger.log_episode(episode_metrics, episode + 1)

        if (episode + 1) % config['log_interval'] == 0:
            main_line = f"Episode {episode + 1}/{config['max_episodes']} | " \
                        f"Reward: {episode_reward:.2f} | " \
                        f"Steps: {episode_steps} | " \
                        f"MaxVel: {max_velocity:.2f} | " \
                        f"Total: {total_steps:,}"
            
            # Add sustained velocity if available
            if min_sustained_velocity is not None:
                main_line += f" | MinSust: {min_sustained_velocity:.2f} | AvgSust: {avg_sustained_velocity:.2f}"
            
            # Add gait metrics based on training type
            if training_type == 'speed_walking':
                main_line += f" | GaitQ: {episode_metrics['comparison/gait_quality']:.3f} | "
            else:  # gait_walking
                main_line += f" | GaitQ: {episode_metrics['episode/gait_quality']:.3f} | " \
                            f"AntiP: {episode_metrics['episode/antiphase_score']:.3f} | " \
                            f"Period: {episode_metrics['episode/periodicity_score']:.3f}"
            
            print(main_line, end='\r', flush=True)
            if episode + 1 == config['max_episodes']:
                print() 
        
        # Save best model based on appropriate metric
        if save_best and len(episode_rewards_window) >= 100:
            # Determine metric based on training type
            if training_type == 'speed_walking':
                # Use average of min sustained velocities over last 100 episodes
                current_metric = np.mean(min_sustained_velocity_window) if len(min_sustained_velocity_window) >= 100 else None
                metric_name = "avg min sustained velocity"
            else:  # gait_walking
                # Use gait quality instead of symmetry score
                current_metric = np.mean(gait_quality_window) if len(gait_quality_window) >= 100 else None
                metric_name = "avg gait quality"
            
            # Use get_model_path for consistent naming
            model_path = get_model_path(training_type, 'best', config['seed'])
            
            if current_metric is not None and current_metric > best_metric:
                best_metric = current_metric
                agent.save(model_path)
                print(f"  → New best model saved! {metric_name}: {best_metric:.4f}")
    
    env.close()
    
    # Save final model
    if save_final:
        model_path = get_model_path(training_type, 'final', config['seed'])
        agent.save(model_path)
        print(f"\nFinal model saved to: {model_path}")
    
    logger.print_statistics()
    logger.finish()
    
    print(f"\n{'='*60}")
    print(f"{training_type.upper()} TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"Total transitions: {total_steps:,}")
    if training_type == 'speed_walking':
        metric_name = "Best avg min sustained velocity"
    else:
        metric_name = "Best avg gait quality"
    print(f"{metric_name}: {best_metric:.4f}")
    print(f"{'='*60}\n")
    
    return agent, logger


def main():
    """Main entry point with simplified command line arguments"""
    parser = argparse.ArgumentParser(description='Train TD3 agent on Walker2d-v5')
    parser.add_argument('--training_type', type=str, default='speed_walking',
                       choices=['speed_walking', 'gait_walking'],
                       help='Type of training: speed_walking or gait_walking')
    
    args = parser.parse_args()
    
    os.makedirs('models/speed', exist_ok=True)
    os.makedirs('models/gait', exist_ok=True)
    
    # Train agent with default parameters
    agent, logger = train_td3(
        training_type=args.training_type,
    )

if __name__ == "__main__":
    main()