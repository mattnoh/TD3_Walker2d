"""
Logging system for TD3 training
"""
import numpy as np
import os
from collections import defaultdict

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")


class TrainingLogger:  
    def __init__(self, use_wandb=True, project_name="td3-walker2d", config=None, training_type="speed_walking", run_name=None):
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.training_type = training_type
        self.metrics = defaultdict(list)
        self.run_name = run_name or f"{training_type}_{wandb.util.generate_id() if WANDB_AVAILABLE else 'local'}"
        
        # Initialize WandB
        if self.use_wandb and config:
            try:
                wandb.init(project=project_name, config=config, name=self.run_name)
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                print("Continuing without wandb logging...")
                self.use_wandb = False
    
    def log_step(self, metrics_dict, step):
        """Log metrics for a single step"""
        # Store locally
        for key, value in metrics_dict.items():
            if value is not None:
                self.metrics[key].append(value)
        
        # Log to WandB
        if self.use_wandb:
            try:
                wandb_dict = {k: v for k, v in metrics_dict.items() if v is not None}
                wandb_dict['step'] = step
                wandb.log(wandb_dict)
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")
    
    def log_episode(self, metrics_dict, episode):
        """Log metrics for a complete episode"""
        # Store locally
        for key, value in metrics_dict.items():
            if value is not None:
                self.metrics[key].append(value)
        
        # Log to WandB
        if self.use_wandb:
            try:
                wandb_dict = {k: v for k, v in metrics_dict.items() if v is not None}
                wandb_dict['episode'] = episode
                wandb.log(wandb_dict)
            except Exception as e:
                print(f"Warning: Failed to log to wandb: {e}")
    
    def get_metric(self, key):
        """Get stored metric values"""
        return self.metrics.get(key, [])
    
    def get_latest_metric(self, key, window=100):
        """Get the latest values of a metric with optional windowing"""
        values = self.get_metric(key)
        if not values:
            return []
        return values[-window:] if len(values) > window else values
    
    def compute_statistics(self):
        """Compute comprehensive training statistics"""
        stats = {}
        
        episode_rewards = self.get_metric('episode/reward')
        episode_lengths = self.get_metric('episode/length')
        
        if episode_rewards and episode_lengths:
            # Basic statistics
            stats['total_episodes'] = len(episode_rewards)
            stats['total_transitions'] = sum(episode_lengths)
            stats['best_reward'] = max(episode_rewards)
            stats['best_episode'] = episode_rewards.index(stats['best_reward']) + 1
            
            # Last 100 episodes
            last_100_rewards = self.get_latest_metric('episode/reward', 100)
            last_100_lengths = self.get_latest_metric('episode/length', 100)
            last_100_action_magnitudes = self.get_latest_metric('episode/avg_action_magnitude', 100)
            
            stats['mean_reward_last_100'] = np.mean(last_100_rewards) if last_100_rewards else 0
            stats['mean_length_last_100'] = np.mean(last_100_lengths) if last_100_lengths else 0
            stats['mean_action_magnitude_last_100'] = np.mean(last_100_action_magnitudes) if last_100_action_magnitudes else 0
            
            # Training losses (from step logs)
            critic_losses = self.get_latest_metric('train/critic_loss', 100)
            actor_losses = [x for x in self.get_latest_metric('train/actor_loss', 100) if x is not None]
            
            if critic_losses:
                stats['mean_critic_loss_last_100'] = np.mean(critic_losses)
            if actor_losses:
                stats['mean_actor_loss_last_100'] = np.mean(actor_losses)
            
            # Training type specific statistics
            if self.training_type == 'speed_walking':
                max_vel = self.get_latest_metric('episode/max_velocity', 100)
                min_sustained = self.get_latest_metric('episode/min_sustained_velocity', 100)
                avg_sustained = self.get_latest_metric('episode/avg_sustained_velocity', 100)
                
                # CHANGED: Comparison gait metrics for speed walking (not used for training)
                comparison_gait_quality = self.get_latest_metric('comparison/gait_quality', 100)
                comparison_antiphase = self.get_latest_metric('comparison/antiphase_score', 100)
                comparison_rom = self.get_latest_metric('comparison/avg_rom', 100)
                comparison_periodicity = self.get_latest_metric('comparison/periodicity_score', 100)
                
                if max_vel:
                    stats['mean_max_velocity_last_100'] = np.mean(max_vel)
                if min_sustained:
                    stats['mean_min_sustained_velocity'] = np.mean(min_sustained)
                if avg_sustained:
                    stats['mean_avg_sustained_velocity'] = np.mean(avg_sustained)
                if comparison_gait_quality:
                    stats['mean_comparison_gait_quality_last_100'] = np.mean(comparison_gait_quality)
                if comparison_antiphase:
                    stats['mean_comparison_antiphase_last_100'] = np.mean(comparison_antiphase)
                if comparison_rom:
                    stats['mean_comparison_rom_last_100'] = np.mean(comparison_rom)
                if comparison_periodicity:
                    stats['mean_comparison_periodicity_last_100'] = np.mean(comparison_periodicity)
                    
            elif self.training_type == 'gait_walking':
                # CHANGED: Gait metrics for gait_walking
                gait_quality = self.get_latest_metric('episode/gait_quality', 100)
                antiphase = self.get_latest_metric('episode/antiphase_score', 100)
                rom_symmetry = self.get_latest_metric('episode/rom_symmetry', 100)
                periodicity = self.get_latest_metric('episode/periodicity_score', 100)
                avg_rom = self.get_latest_metric('episode/avg_rom', 100)
                vel_consistency = self.get_latest_metric('episode/vel_consistency', 100)
                
                # Also track velocity metrics
                max_vel = self.get_latest_metric('episode/max_velocity', 100)
                min_sustained = self.get_latest_metric('episode/min_sustained_velocity', 100)
                avg_sustained = self.get_latest_metric('episode/avg_sustained_velocity', 100)
                
                if gait_quality:
                    stats['best_gait_quality'] = max(self.get_metric('episode/gait_quality'))
                    stats['mean_gait_quality_last_100'] = np.mean(gait_quality)
                if antiphase:
                    stats['mean_antiphase_last_100'] = np.mean(antiphase)
                if rom_symmetry:
                    stats['mean_rom_symmetry_last_100'] = np.mean(rom_symmetry)
                if periodicity:
                    stats['mean_periodicity_last_100'] = np.mean(periodicity)
                if avg_rom:
                    stats['mean_avg_rom_last_100'] = np.mean(avg_rom)
                if vel_consistency:
                    stats['mean_vel_consistency_last_100'] = np.mean(vel_consistency)
                if max_vel:
                    stats['mean_max_velocity_last_100'] = np.mean(max_vel)
                if min_sustained:
                    stats['mean_min_sustained_velocity'] = np.mean(min_sustained)
                if avg_sustained:
                    stats['mean_avg_sustained_velocity'] = np.mean(avg_sustained)
        
        return stats
    
    def print_statistics(self, episode=None):
        """Print formatted training statistics"""
        stats = self.compute_statistics()
        
        print("\n" + "="*60)
        print(f"TRAINING STATISTICS - {self.training_type.upper()}")
        print("="*60)
        
        if episode is not None:
            print(f"Current Episode: {episode}")
        
        print(f"\nOverall Progress:")
        print(f"  Total Episodes: {stats.get('total_episodes', 0):,}")
        print(f"  Total Transitions: {stats.get('total_transitions', 0):,}")
        print(f"  Best Reward: {stats.get('best_reward', 0):.2f} (Episode {stats.get('best_episode', 0)})")
        
        print(f"\nLast 100 Episodes:")
        print(f"  Mean Reward: {stats.get('mean_reward_last_100', 0):.2f}")
        print(f"  Mean Length: {stats.get('mean_length_last_100', 0):.1f}")
        print(f"  Mean Action Magnitude: {stats.get('mean_action_magnitude_last_100', 0):.4f}")
        
        # Training losses
        if 'mean_critic_loss_last_100' in stats:
            print(f"  Mean Critic Loss: {stats['mean_critic_loss_last_100']:.4f}")
        if 'mean_actor_loss_last_100' in stats:
            print(f"  Mean Actor Loss: {stats['mean_actor_loss_last_100']:.4f}")
        
        if self.training_type == 'speed_walking':
            print(f"\nSpeed Metrics (last 100):")
            if 'mean_max_velocity_last_100' in stats:
                print(f"  Mean Max Velocity: {stats['mean_max_velocity_last_100']:.2f}")
            if 'mean_min_sustained_velocity' in stats:
                print(f"  Mean Min Sustained: {stats['mean_min_sustained_velocity']:.2f}")
            if 'mean_avg_sustained_velocity' in stats:
                print(f"  Mean Avg Sustained: {stats['mean_avg_sustained_velocity']:.2f}")
            
            # CHANGED: Print comparison gait metrics
            print(f"\nComparison Gait Metrics (last 100) - NOT USED FOR TRAINING:")
            if 'mean_comparison_gait_quality_last_100' in stats:
                print(f"  Mean Gait Quality: {stats['mean_comparison_gait_quality_last_100']:.4f}")
            if 'mean_comparison_antiphase_last_100' in stats:
                print(f"  Mean Anti-phase Score: {stats['mean_comparison_antiphase_last_100']:.4f}")
            if 'mean_comparison_rom_last_100' in stats:
                print(f"  Mean ROM: {stats['mean_comparison_rom_last_100']:.4f}")
            if 'mean_comparison_periodicity_last_100' in stats:
                print(f"  Mean Periodicity: {stats['mean_comparison_periodicity_last_100']:.4f}")
                
        elif self.training_type == 'gait_walking':
            # CHANGED: Print gait metrics
            print(f"\nGait Quality Metrics (last 100):")
            if 'best_gait_quality' in stats:
                print(f"  Best Gait Quality: {stats['best_gait_quality']:.4f}")
            if 'mean_gait_quality_last_100' in stats:
                print(f"  Mean Gait Quality: {stats['mean_gait_quality_last_100']:.4f}")
            if 'mean_antiphase_last_100' in stats:
                print(f"  Mean Anti-phase Score: {stats['mean_antiphase_last_100']:.4f}")
            if 'mean_rom_symmetry_last_100' in stats:
                print(f"  Mean ROM Symmetry: {stats['mean_rom_symmetry_last_100']:.4f}")
            if 'mean_avg_rom_last_100' in stats:
                print(f"  Mean Average ROM: {stats['mean_avg_rom_last_100']:.4f}")
            if 'mean_periodicity_last_100' in stats:
                print(f"  Mean Periodicity: {stats['mean_periodicity_last_100']:.4f}")
            if 'mean_vel_consistency_last_100' in stats:
                print(f"  Mean Velocity Consistency: {stats['mean_vel_consistency_last_100']:.4f}")
            
            # Also print velocity metrics for gait_walking
            print(f"\nSpeed Metrics (last 100):")
            if 'mean_max_velocity_last_100' in stats:
                print(f"  Mean Max Velocity: {stats['mean_max_velocity_last_100']:.2f}")
            if 'mean_min_sustained_velocity' in stats:
                print(f"  Mean Min Sustained: {stats['mean_min_sustained_velocity']:.2f}")
            if 'mean_avg_sustained_velocity' in stats:
                print(f"  Mean Avg Sustained: {stats['mean_avg_sustained_velocity']:.2f}")
        
        print("="*60)
        
    def log_config_stats(self, config):
        """Log configuration parameters to WandB"""
        if self.use_wandb:
            try:
                # Log reward weights based on training type
                if self.training_type == 'speed_walking':
                    wandb.log({
                        "config/velocity_weight": config.get('velocity_weight', 0),
                        "config/consistency_weight": config.get('consistency_weight', 0),
                        "config/sustained_velocity_bonus": config.get('sustained_velocity_bonus', 0)
                    })
                elif self.training_type == 'gait_walking':
                    # CHANGED: Log gait walking config
                    wandb.log({
                        "config/velocity_weight": config.get('velocity_weight', 0),
                        "config/consistency_weight": config.get('consistency_weight', 0),
                        "config/sustained_velocity_bonus": config.get('sustained_velocity_bonus', 0),
                        "config/rom_weight": config.get('rom_weight', 0),
                        "config/antiphase_weight": config.get('antiphase_weight', 0),
                        "config/periodicity_weight": config.get('periodicity_weight', 0)
                    })
            except Exception as e:
                print(f"Warning: Failed to log config to wandb: {e}")
    
    def get_progress_summary(self):
        """Get a concise progress summary for periodic logging"""
        stats = self.compute_statistics()
        
        summary = {
            'episodes': stats.get('total_episodes', 0),
            'transitions': stats.get('total_transitions', 0),
            'current_reward': stats.get('mean_reward_last_100', 0),
            'best_reward': stats.get('best_reward', 0),
            'action_magnitude': stats.get('mean_action_magnitude_last_100', 0)
        }
        
        if self.training_type == 'speed_walking':
            if 'mean_avg_sustained_velocity' in stats:
                summary['avg_velocity'] = stats['mean_avg_sustained_velocity']
            # CHANGED: Add comparison gait metrics to summary
            if 'mean_comparison_gait_quality_last_100' in stats:
                summary['comparison_gait_quality'] = stats['mean_comparison_gait_quality_last_100']
            if 'mean_comparison_antiphase_last_100' in stats:
                summary['comparison_antiphase'] = stats['mean_comparison_antiphase_last_100']
                
        elif self.training_type == 'gait_walking':
            # CHANGED: Add gait metrics to summary
            if 'mean_gait_quality_last_100' in stats:
                summary['gait_quality'] = stats['mean_gait_quality_last_100']
            if 'mean_antiphase_last_100' in stats:
                summary['antiphase'] = stats['mean_antiphase_last_100']
            if 'mean_avg_rom_last_100' in stats:
                summary['avg_rom'] = stats['mean_avg_rom_last_100']
            if 'mean_avg_sustained_velocity' in stats:
                summary['avg_velocity'] = stats['mean_avg_sustained_velocity']
        
        return summary
    
    def finish(self):
        """Finish logging session"""
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb session: {e}")


def create_logger(training_type='speed_walking', use_wandb=True, config=None, run_name=None):
    """Create a logger for the specified training type"""
    
    # CHANGED: Update project names
    project_names = {
        'speed_walking': 'td3-walker2d-speed',
        'gait_walking': 'td3-walker2d-gait'
    }
    
    project_name = project_names.get(training_type, 'td3-walker2d')
    
    return TrainingLogger(
        use_wandb=use_wandb,
        project_name=project_name,
        config=config,
        training_type=training_type,
        run_name=run_name
    )


def get_model_path(training_type, model_type, seed=None):
    """Generate model path with seed-specific directories"""
    # If seed is None or not provided, use a fallback
    if seed is None:
        # Try to extract seed from environment or use default
        seed = "default"
        print(f"Warning: No seed provided, using fallback: {seed}")
    
    # CHANGED: Update path generation for gait_walking
    if training_type == 'speed_walking':
        base_dir = f"models/speed/speed_walking_seed_{seed}"
        file_name = f"{model_type}.pth"
    elif training_type == 'gait_walking':
        base_dir = f"models/gait/gait_walking_seed_{seed}"
        file_name = f"{model_type}.pth"
    else:
        # Fallback for any other training type
        base_dir = f"models/{training_type}/seed_{seed}"
        file_name = f"{model_type}.pth"
    
    # Create directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    return f"{base_dir}/{file_name}"