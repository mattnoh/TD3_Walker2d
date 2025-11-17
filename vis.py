"""
Visualization script for trained TD3 Walker2d models - Updated for unified architecture
"""
import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, get_device
from td3_agent import TD3Agent
from utils_vel import compute_velocity_reward, evaluate_velocity_agent
from utils_sym import compute_gait_reward, evaluate_gait_agent, GaitTracker
from logging_utils import get_model_path

def extract_seed_from_path(model_path):
    """Extract seed from model path, return 'unknown' if not found"""
    try:
        # Look for patterns like "seed_42" or "seed42" in the path
        path_parts = model_path.split('/')
        for part in path_parts:
            if 'seed' in part.lower():
                # Extract numbers after 'seed'
                seed_part = part.lower().split('seed')[-1]
                # Remove any non-digit characters and underscores
                seed_digits = ''.join(filter(str.isdigit, seed_part))
                if seed_digits:
                    return seed_digits
    except:
        pass
    return "unknown"


def extract_training_type_from_path(model_path):
    """Extract training type from model path"""
    model_path_lower = model_path.lower()
    
    if 'gait' in model_path_lower or 'gait' in model_path_lower:
        return 'gait_walking'
    elif 'speed' in model_path_lower or 'velocity' in model_path_lower:
        return 'speed_walking'
    else:
        # Default to speed_walking if cannot determine
        return 'speed_walking'


def compute_sustained_velocity(velocities, window_size):
    """Compute minimum sustained velocity over the LAST window"""
    if len(velocities) < window_size:
        return 0.0
    
    return np.min(velocities[-window_size:])


def capture_freeze_frames(env, agent, num_frames=20, episode_length=1000):
    """Capture freeze frames at evenly spaced intervals"""
    frames = []
    capture_steps = np.linspace(0, episode_length - 1, num_frames, dtype=int)
    
    state, _ = env.reset()
    frames_captured = 0
    
    for step in range(episode_length):
        action = agent.select_action(state, noise=0.0)
        state, _, terminated, truncated, _ = env.step(action)
        
        # Capture frame if this is one of our target steps
        if step in capture_steps:
            frame = env.render()
            # Check if frame is not None and has the right shape
            if frame is not None and hasattr(frame, 'shape') and len(frame.shape) == 3:
                frames.append((step, frame))
                frames_captured += 1
            else:
                print(f"Warning: Failed to capture frame at step {step}, got: {type(frame)}")
        
        if terminated or truncated:
            break
    
    return frames


def create_progression_composite(frames, training_type, seed, output_dir="progression_composites"):
    """Create a side-by-side progression composite showing the walker at different stages"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not frames:
        print("No valid frames to create composite!")
        return None
    
    # Filter out any None frames
    valid_frames = [(step, frame) for step, frame in frames if frame is not None]
    if not valid_frames:
        print("No valid frames after filtering!")
        return None
    
    # Calculate grid dimensions (4x5 for 20 frames)
    rows, cols = 4, 5
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.ravel()
    
    for i, (step, frame) in enumerate(valid_frames):
        if i < len(axes):
            axes[i].imshow(frame)
            axes[i].set_title(f"Step {step}", fontsize=12, fontweight='bold')
            axes[i].axis('off')
    
    # Hide any unused subplots
    for i in range(len(valid_frames), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Use consistent naming format: (speed) walking seed (num)
    training_type_readable = training_type.replace('_', ' ').replace('symmetric', 'gait')
    filename = f"{training_type_readable} seed {seed} progression {int(time.time())}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=900, bbox_inches='tight', facecolor='white', pil_kwargs={'optimize': True})
    plt.close()
    print(f"Created progression composite: {filepath}")
    return filepath


def run_evaluation_episode(env, agent, training_type, config, episode_num, render=False):
    """Run a single evaluation episode and return comprehensive metrics"""
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    max_velocity = 0
    velocities = []
    gait_tracker = GaitTracker()
    done = False
    
    while not done and episode_steps < config['max_steps']:
        action = agent.select_action(state, noise=0.0)
        next_state, base_reward, terminated, truncated, _ = env.step(action)
        
        # Track velocity
        if len(state) > 8:
            current_velocity = abs(state[8])
            velocities.append(current_velocity)
            max_velocity = max(max_velocity, current_velocity)
        
        # Track gait metrics for both training types
        gait_tracker.update(state)
        
        state = next_state
        episode_reward += base_reward
        episode_steps += 1
        done = terminated or truncated
    
    # Calculate sustained velocity metrics
    sustained_window = config.get('sustained_velocity_window', 100)
    if len(velocities) >= sustained_window:
        min_sustained = np.min(velocities[-sustained_window:])
        avg_sustained = np.mean(velocities[-sustained_window:])
    else:
        min_sustained = 0.0
        avg_sustained = np.mean(velocities) if velocities else 0.0
    
    # Calculate gait metrics
    gait_metrics = gait_tracker.compute_gait_metrics()
    gait_quality = gait_metrics.get('gait_quality', 0.0)
    
    # Print episode results
    print(f"Episode {episode_num}: "
          f"Reward = {episode_reward:.2f}, "
          f"Steps = {episode_steps}, "
          f"Max Vel = {max_velocity:.2f}, "
          f"Min Sustained = {min_sustained:.2f}, "
          f"Gait Score = {gait_quality:.4f}")
    
    return {
        'reward': episode_reward,
        'steps': episode_steps,
        'max_velocity': max_velocity,
        'min_sustained_velocity': min_sustained,
        'avg_sustained_velocity': avg_sustained,
        'gait_quality': gait_quality,
        'gait_metrics': gait_metrics,
        'velocities': velocities
    }


def print_comprehensive_summary(results, training_type, model_path, config):
    """Print comprehensive performance summary for all training types"""
    print("=" * 60)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Training Type: {training_type}")
    print(f"Model: {model_path}")
    
    # Extract metrics
    rewards = [r['reward'] for r in results]
    steps = [r['steps'] for r in results]
    max_velocities = [r['max_velocity'] for r in results]
    min_sustained_velocities = [r['min_sustained_velocity'] for r in results]
    avg_sustained_velocities = [r['avg_sustained_velocity'] for r in results]
    gait_qualities = [r['gait_quality'] for r in results]
    
    # Common metrics for all training types
    print(f"\nBASE METRICS:")
    print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Min Reward: {np.min(rewards):.2f}")
    print(f"  Max Reward: {np.max(rewards):.2f}")
    print(f"  Mean Steps: {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    
    print(f"\nVELOCITY METRICS:")
    print(f"  Mean Max Velocity: {np.mean(max_velocities):.2f} ± {np.std(max_velocities):.2f}")
    print(f"  Mean Min Sustained Velocity: {np.mean(min_sustained_velocities):.2f} ± {np.std(min_sustained_velocities):.2f}")
    print(f"  Mean Average Sustained Velocity: {np.mean(avg_sustained_velocities):.2f} ± {np.std(avg_sustained_velocities):.2f}")
    
    # Training type specific metrics
    if training_type == 'speed_walking':
        sustained_window = config.get('sustained_velocity_window', 100)
        print(f"\nSPEED WALKING SPECIFIC METRICS:")
        best_min_sustained = np.max(min_sustained_velocities)
        worst_min_sustained = np.min(min_sustained_velocities)
        print(f"  Best Min Sustained: {best_min_sustained:.2f}")
        print(f"  Worst Min Sustained: {worst_min_sustained:.2f}")
        print(f"  Consistency (Min/Max Ratio): {best_min_sustained/np.max(max_velocities) if np.max(max_velocities) > 0 else 0:.2f}")
        
        # Gait metrics for speed walking (comparison only)
        print(f"\nCOMPARISON GAIT METRICS:")
        mean_gait_quality = np.mean(gait_qualities)
        print(f"  Mean Gait Quality: {mean_gait_quality:.4f}")
        
    elif training_type == 'symmetric_walking':
        print(f"\nGAIT WALKING SPECIFIC METRICS:")
        mean_gait_quality = np.mean(gait_qualities)
        best_gait_quality = np.max(gait_qualities)
        print(f"  Mean Gait Quality: {mean_gait_quality:.4f}")
        print(f"  Best Gait Quality: {best_gait_quality:.4f}")
        
        # Additional gait metrics
        antiphase_scores = [r['gait_metrics'].get('antiphase_score', 0) for r in results]
        periodicity_scores = [r['gait_metrics'].get('periodicity_score', 0) for r in results]
        rom_scores = [r['gait_metrics'].get('rom_symmetry', 0) for r in results]
        
        print(f"  Mean Anti-phase Score: {np.mean(antiphase_scores):.4f}")
        print(f"  Mean Periodicity Score: {np.mean(periodicity_scores):.4f}")
        print(f"  Mean ROM Symmetry: {np.mean(rom_scores):.4f}")
    
    print("=" * 60)


def create_custom_video_wrapper(env, training_type, seed, video_folder="./demo_videos"):
    """Create a custom video wrapper that uses consistent naming"""
    from gymnasium.wrappers import RecordVideo
    
    # Create video folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Use consistent naming format: (speed) walking seed (num) video
    training_type_readable = training_type.replace('_', ' ').replace('symmetric', 'gait')
    name_prefix = f"{training_type_readable} seed {seed} video"
    
    return RecordVideo(
        env, 
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda x: True
    )


def visualize_agent(model_path, training_type=None, num_episodes=5, render=False, record_video=False, capture_frames=False):
    
    # Get device and config
    device = get_device()
    
    # Extract seed and training type from model path if not provided
    seed = extract_seed_from_path(model_path)
    
    # Auto-detect training type from model path if not provided
    if training_type is None:
        training_type = extract_training_type_from_path(model_path)
        print(f"Auto-detected training type: {training_type} from model path")
    
    config = get_config(training_type)
    
    print(f"Using seed: {seed} (extracted from model path)")
    print(f"Using training type: {training_type}")
    
    env = None
    try:
        # Fix the environment creation logic
        if record_video:
            env = gym.make('Walker2d-v5', render_mode='rgb_array')
            # Use custom video wrapper with consistent naming
            env = create_custom_video_wrapper(env, training_type, seed)
            training_type_readable = training_type.replace('_', ' ').replace('symmetric', 'gait')
            print(f"Recording video with naming: {training_type_readable} seed {seed} video")
        elif render:
            # Only use human mode if explicitly requested with --render
            env = gym.make('Walker2d-v5', render_mode='human')
        elif capture_frames:
            # Use rgb_array for frame capture
            env = gym.make('Walker2d-v5', render_mode='rgb_array')
        else:
            # No rendering at all by default
            env = gym.make('Walker2d-v5')
        
        # Get environment dimensions
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        print(f"Environment: Walker2d-v5")
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
        print(f"Max action: {max_action}")
        print(f"Device: {device}")
        print(f"Loading model: {model_path}")
        
        # Initialize agent using your existing TD3Agent class
        agent = TD3Agent(state_dim, action_dim, max_action, config, device)
        
        # Load the model - using the load method from your td3_agent.py
        try:
            agent.load(model_path)
            print(f"Successfully loaded model from {model_path}")
            print(f"Training type: {training_type}")
            print(f"Actor parameters: {sum(p.numel() for p in agent.actor.parameters()):,}")
            print(f"Critic parameters: {sum(p.numel() for p in agent.critic.parameters()):,}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Available model files in 'models/' directory:")
            if os.path.exists('models'):
                for f in os.listdir('models'):
                    if f.endswith('.pth'):
                        print(f"  - models/{f}")
            return
        
        results = []
        
        print(f"\nRunning {num_episodes} episodes...")
        print("=" * 60)
        
        # Capture frames on first episode if requested
        if capture_frames:
            print("Capturing freeze frames for progression composite...")
            frames = capture_freeze_frames(env, agent, num_frames=20, episode_length=config['max_steps'])
            
            # Create progression composite with consistent naming
            if frames:
                progression_path = create_progression_composite(frames, training_type, seed)
                if progression_path:
                    print(f"Progression composite saved: {progression_path}")
            else:
                print("Warning: No frames were captured!")
            
            # If we're not rendering, we can break after capturing frames
            if not render and not record_video:
                print("Frame capture complete. Exiting.")
                return
        
        # Run evaluation episodes
        for ep in range(num_episodes):
            episode_result = run_evaluation_episode(env, agent, training_type, config, ep + 1, render)
            results.append(episode_result)
        
        # Print comprehensive summary
        print_comprehensive_summary(results, training_type, model_path, config)
        
        return {
            'results': results,
            'training_type': training_type,
            'model_path': model_path,
            'seed': seed
        }
    
    finally:
        # Always close the environment
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize trained TD3 Walker2d agent')
    parser.add_argument('--model', type=str, default='models/best.pth',
                       help='Path to the model file (default: models/best.pth)')
    parser.add_argument('--training_type', type=str, default=None,
                       choices=['speed_walking', 'symmetric_walking'],
                       help='Type of training the model was trained with. If not provided, will be auto-detected from model path.')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run (default: 5)')
    parser.add_argument('--render', action='store_true',
                       help='Render the environment (show visualization)')
    parser.add_argument('--record', action='store_true',
                       help='Record video of the episodes')
    parser.add_argument('--capture', action='store_true',
                       help='Capture 20 freeze frames and create progression composite')
    
    args = parser.parse_args()
    
    # Visualize single model
    results = visualize_agent(
        model_path=args.model,
        training_type=args.training_type,
        num_episodes=args.episodes,
        render=args.render,
        record_video=args.record,
        capture_frames=args.capture
    )


if __name__ == "__main__":
    main()