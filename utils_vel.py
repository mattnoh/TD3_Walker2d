"""
Utility functions for Velocity Walking
"""
import numpy as np
import gymnasium as gym

def compute_velocity_reward(base_reward, velocities, current_velocity, episode_steps, config):
    """
    Compute velocity-based reward shaping - OPTIMIZED FOR HIGHER PERFORMANCE
    """
    shaped_reward = base_reward
    
    velocity_weight = config['velocity_weight']
    consistency_weight = config['consistency_weight']
    window = config['sustained_velocity_window']
    bonus = config['sustained_velocity_bonus']
    
    # 1. Current velocity bonus (main driver)
    shaped_reward += current_velocity * velocity_weight
    
    # 2. Progressive sustained velocity rewards
    if len(velocities) >= window:
        recent_velocities = velocities[-window:]
        min_velocity = np.min(recent_velocities)
        velocity_std = np.std(recent_velocities)
        mean_velocity = np.mean(recent_velocities)
        
        if min_velocity > 3.0:
            min_velocity_bonus = min_velocity * velocity_weight * 1.5
        else:
            min_velocity_bonus = min_velocity * velocity_weight
        
        # Mean velocity bonus
        mean_velocity_bonus = mean_velocity * velocity_weight * 0.6
        
        # Consistency penalty (reduced impact)
        # consistency_bonus = -velocity_std * consistency_weight
        
        # Add bonuses
        # shaped_reward += min_velocity_bonus + mean_velocity_bonus + consistency_bonus
        shaped_reward += min_velocity_bonus + mean_velocity_bonus
        if min_velocity > 3.5 and mean_velocity > 4.0: 
            peak_bonus = (min_velocity - 3.5) * velocity_weight * 2.0
            shaped_reward += peak_bonus
    
    # 3. Staged episode completion bonus
    if episode_steps == 1000 and len(velocities) >= 1000:
        min_1000_velocity = np.min(velocities)
        mean_1000_velocity = np.mean(velocities)
        
        # Base completion bonus
        completion_bonus = min_1000_velocity * bonus
        
        # High-performance completion bonus
        if min_1000_velocity > 4.0:
            completion_bonus *= 1.5 
        
        shaped_reward += completion_bonus
    
    # 4. Acceleration bonus 
    if len(velocities) >= 10:
        recent_acceleration = np.mean(np.diff(velocities[-10:]))
        if recent_acceleration > 0.1:
            acceleration_bonus = recent_acceleration * velocity_weight * 0.3
            shaped_reward += acceleration_bonus
    
    return shaped_reward


def evaluate_velocity_agent(agent, env_name='Walker2d-v5', num_episodes=10, render=False, config=None):
    """
    Evaluate velocity trained agent
    """
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    
    rewards = []
    lengths = []
    max_velocities = []
    min_sustained_velocities = []
    avg_sustained_velocities = []
    
    window = config['sustained_velocity_window'] if config else 100
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        max_velocity = 0
        velocities = []
        done = False
        
        while not done and episode_length < 1000:
            action = agent.select_action(state, noise=0.0)  # No exploration
            state, reward, terminated, truncated, _ = env.step(action)
            
            # Track velocity (index 8 for Walker2d-v5)
            if len(state) > 8:
                velocity = state[8]
                velocities.append(velocity)
                max_velocity = max(max_velocity, abs(velocity))
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        # Calculate sustained velocity metrics
        if len(velocities) >= window:
            recent_velocities = velocities[-window:]
            min_sustained = np.min(recent_velocities)
            avg_sustained = np.mean(recent_velocities)
        else:
            min_sustained = 0
            avg_sustained = 0
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        max_velocities.append(max_velocity)
        min_sustained_velocities.append(min_sustained)
        avg_sustained_velocities.append(avg_sustained)
        
        print(f"Eval Episode {ep + 1}/{num_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={episode_length}, "
              f"MaxVel={max_velocity:.2f}, "
              f"MinSustained={min_sustained:.2f}, "
              f"AvgSustained={avg_sustained:.2f}")
    
    env.close()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Mean Max Velocity: {np.mean(max_velocities):.2f} ± {np.std(max_velocities):.2f}")
    
    if any(v > 0 for v in min_sustained_velocities):
        print(f"Mean Min Sustained Velocity: {np.mean(min_sustained_velocities):.2f} ± {np.std(min_sustained_velocities):.2f}")
        print(f"Mean Avg Sustained Velocity: {np.mean(avg_sustained_velocities):.2f} ± {np.std(avg_sustained_velocities):.2f}")
    
    print(f"Min Reward: {np.min(rewards):.2f}")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print(f"{'='*60}\n")
    
    return {
        'rewards': rewards,
        'lengths': lengths,
        'max_velocities': max_velocities,
        'min_sustained_velocities': min_sustained_velocities,
        'avg_sustained_velocities': avg_sustained_velocities,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'mean_max_velocity': np.mean(max_velocities),
    }