"""
Utility functions for Symmetric Walking (Phase-Shifted Gait)
Builds on velocity walking to encourage high-speed symmetric gait
"""
import numpy as np
import gymnasium as gym
from scipy import signal
from collections import deque


class GaitTracker:
    """Track gait patterns for phase-shifted symmetry"""
    
    def __init__(self, history_length=200):
        self.history_length = history_length
        self.right_angles = deque(maxlen=history_length)
        self.left_angles = deque(maxlen=history_length)
        self.velocities = deque(maxlen=history_length)
        
    def update(self, state):
        """Update tracker with current state"""
        # Extract joint angles (indices 2-7)
        right_ang = state[2:5]  # thigh, leg, foot
        left_ang = state[5:8]
        forward_vel = state[8]
        
        self.right_angles.append(right_ang)
        self.left_angles.append(left_ang)
        self.velocities.append(forward_vel)
    
    def compute_phase_shift_correlation(self):
        """
        Compute cross-correlation to find phase shift between legs.
        For symmetric walking, legs should be ~180 degrees out of phase.
        """
        if len(self.right_angles) < 50:
            return 0.0, 0
        
        right_arr = np.array(self.right_angles)
        left_arr = np.array(self.left_angles)
        
        # Use thigh angle as representative joint
        right_thigh = right_arr[:, 0]
        left_thigh = left_arr[:, 0]
        
        # Normalize
        right_norm = (right_thigh - np.mean(right_thigh)) / (np.std(right_thigh) + 1e-8)
        left_norm = (left_thigh - np.mean(left_thigh)) / (np.std(left_thigh) + 1e-8)
        
        # Cross-correlation
        correlation = np.correlate(right_norm, left_norm, mode='full')
        correlation = correlation / len(right_norm)
        
        # Find the lag (should be around half the gait cycle for good walking)
        center = len(correlation) // 2
        search_range = min(30, len(correlation) // 4)  # Search within reasonable range
        
        search_start = center - search_range
        search_end = center + search_range
        
        # For walking, we want NEGATIVE correlation at zero lag (legs doing opposite)
        # and POSITIVE correlation at half-cycle lag
        zero_lag_corr = correlation[center]
        
        # Find best negative correlation (anti-phase)
        if search_end <= len(correlation):
            max_corr_idx = np.argmax(np.abs(correlation[search_start:search_end]))
            max_corr = correlation[search_start + max_corr_idx]
            lag = max_corr_idx - search_range
        else:
            max_corr = 0.0
            lag = 0
        
        return zero_lag_corr, lag
    
    def compute_gait_metrics(self):
        """Compute comprehensive gait quality metrics"""
        if len(self.right_angles) < 50:
            return {}
        
        right_arr = np.array(self.right_angles)
        left_arr = np.array(self.left_angles)
        
        # 1. Range of Motion (how much joints move)
        right_rom = np.std(right_arr, axis=0)
        left_rom = np.std(left_arr, axis=0)
        avg_rom = (np.mean(right_rom) + np.mean(left_rom)) / 2
        rom_symmetry = 1.0 - np.abs(np.mean(right_rom) - np.mean(left_rom)) / (np.mean(right_rom) + np.mean(left_rom) + 1e-8)
        
        # 2. Phase shift (legs should be anti-phase)
        zero_lag_corr, lag = self.compute_phase_shift_correlation()
        # Good walking has negative correlation at zero lag
        antiphase_score = max(0, -zero_lag_corr)  # Higher is better
        
        # 3. Periodicity (gait should be rhythmic)
        right_thigh = right_arr[:, 0]
        autocorr = np.correlate(right_thigh - np.mean(right_thigh), 
                               right_thigh - np.mean(right_thigh), 
                               mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        
        # Find first significant peak (indicates period)
        periodicity_score = 0.0
        gait_period = 0
        if len(autocorr) > 20:
            peaks, properties = signal.find_peaks(autocorr[10:], height=0.2, distance=10)
            if len(peaks) > 0:
                periodicity_score = properties['peak_heights'][0]
                gait_period = peaks[0] + 10
        
        # 4. Velocity consistency
        vels = np.array(self.velocities)
        mean_vel = np.mean(vels)
        vel_std = np.std(vels)
        vel_consistency = max(0, 1.0 - vel_std / (abs(mean_vel) + 0.1))
        
        # 5. Overall gait quality score
        gait_quality = (
            0.25 * min(1.0, avg_rom / 0.5) +  # Normalized ROM
            0.35 * antiphase_score +           # Phase shift
            0.20 * periodicity_score +         # Rhythmicity
            0.20 * rom_symmetry                # ROM symmetry
        )
        
        return {
            'avg_rom': avg_rom,
            'rom_symmetry': rom_symmetry,
            'antiphase_score': antiphase_score,
            'zero_lag_corr': zero_lag_corr,
            'phase_lag': lag,
            'periodicity_score': periodicity_score,
            'gait_period': gait_period,
            'gait_quality': gait_quality,
            'mean_velocity': mean_vel,
            'vel_consistency': vel_consistency,
        }


def compute_gait_reward(base_reward, state, tracker, velocities, config, episode_step):
    """
    Compute gait-quality reward that encourages fast, symmetric walking.
    Builds on velocity reward structure.
    
    Args:
        base_reward: Original environment reward
        state: Current state observation
        tracker: GaitTracker instance
        velocities: List of historical velocities
        config: Configuration dictionary
        episode_step: Current step in episode
    
    Returns:
        shaped_reward: Reward encouraging high-speed symmetric gait
    """
    # Start with velocity-based reward (same as velocity walking)
    shaped_reward = base_reward
    
    current_velocity = state[8]
    velocity_weight = config.get('velocity_weight', 5.0)
    consistency_weight = config.get('consistency_weight', 0.5)
    
    # 1. Current velocity bonus (main driver)
    shaped_reward += current_velocity * velocity_weight
    
    # 2. Sustained velocity (from velocity walking)
    window = config.get('sustained_velocity_window', 100)
    if len(velocities) >= window:
        recent_velocities = velocities[-window:]
        min_velocity = np.min(recent_velocities)
        velocity_std = np.std(recent_velocities)
        
        min_velocity_bonus = min_velocity * velocity_weight
        consistency_bonus = -velocity_std * consistency_weight
        shaped_reward += min_velocity_bonus + consistency_bonus
    
    # 3. GAIT QUALITY BONUSES (added on top of velocity)
    # Only compute these periodically to save computation
    if episode_step > 100 and episode_step % 20 == 0:
        
        # Range of Motion (penalize if too small - means not walking)
        right_angles = state[2:5]
        left_angles = state[5:8]
        
        # Instantaneous ROM check (joints should be moving)
        if len(tracker.right_angles) >= 20:
            recent_right = np.array(list(tracker.right_angles)[-20:])
            recent_left = np.array(list(tracker.left_angles)[-20:])
            
            right_movement = np.std(recent_right[:, 0])  # Thigh movement
            left_movement = np.std(recent_left[:, 0])
            
            # Reward movement (penalize if ROM too low)
            rom_weight = config.get('rom_weight', 2.0)
            movement_score = (right_movement + left_movement) / 2
            rom_bonus = min(1.0, movement_score / 0.3) * rom_weight
            shaped_reward += rom_bonus
        
        # Anti-phase bonus (legs doing opposite things)
        if len(tracker.right_angles) >= 50:
            zero_lag_corr, _ = tracker.compute_phase_shift_correlation()
            
            # Reward negative correlation (anti-phase)
            antiphase_weight = config.get('antiphase_weight', 3.0)
            antiphase_bonus = max(0, -zero_lag_corr) * antiphase_weight
            shaped_reward += antiphase_bonus
    
    # 4. Periodicity bonus (every 100 steps)
    if episode_step > 100 and episode_step % 100 == 0:
        if len(tracker.right_angles) >= 100:
            recent_angles = np.array(list(tracker.right_angles)[-100:])
            right_thigh = recent_angles[:, 0]
            
            # Autocorrelation for periodicity
            autocorr = np.correlate(right_thigh - np.mean(right_thigh), 
                                   right_thigh - np.mean(right_thigh), 
                                   mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if autocorr[0] > 0:
                autocorr = autocorr / autocorr[0]
                
                # Find peaks
                if len(autocorr) > 20:
                    peaks, properties = signal.find_peaks(autocorr[10:], height=0.2)
                    if len(peaks) > 0:
                        periodicity_weight = config.get('periodicity_weight', 2.0)
                        periodicity_bonus = properties['peak_heights'][0] * periodicity_weight
                        shaped_reward += periodicity_bonus
    
    # 5. Episode completion bonus with good gait
    if episode_step == 1000 and len(velocities) >= 1000:
        bonus_weight = config.get('sustained_velocity_bonus', 10.0)
        min_1000_velocity = np.min(velocities)
        shaped_reward += min_1000_velocity * bonus_weight
    
    return shaped_reward


def evaluate_gait_agent(agent, env_name='Walker2d-v5', num_episodes=10, render=False, config=None):
    """
    Evaluate agent with both velocity and gait metrics
    """
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    
    rewards = []
    lengths = []
    all_gait_metrics = []
    max_velocities = []
    min_sustained_velocities = []
    
    window = config.get('sustained_velocity_window', 100) if config else 100
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        tracker = GaitTracker(history_length=200)
        velocities = []
        
        while not done and episode_length < 1000:
            action = agent.select_action(state, noise=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            tracker.update(state)
            velocity = state[8]
            velocities.append(velocity)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        # Compute gait metrics
        gait_metrics = tracker.compute_gait_metrics()
        
        # Velocity metrics
        max_vel = np.max(velocities) if velocities else 0
        if len(velocities) >= window:
            min_sustained = np.min(velocities[-window:])
        else:
            min_sustained = 0
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        all_gait_metrics.append(gait_metrics)
        max_velocities.append(max_vel)
        min_sustained_velocities.append(min_sustained)
        
        print(f"\nEval Episode {ep + 1}/{num_episodes}:")
        print(f"  Reward: {episode_reward:.2f}, Length: {episode_length}")
        print(f"  Max Velocity: {max_vel:.3f} m/s")
        print(f"  Min Sustained Velocity: {min_sustained:.3f} m/s")
        print(f"  Gait Quality: {gait_metrics.get('gait_quality', 0):.4f}")
        print(f"  Anti-phase Score: {gait_metrics.get('antiphase_score', 0):.4f}")
        print(f"  ROM Symmetry: {gait_metrics.get('rom_symmetry', 0):.4f}")
        print(f"  Periodicity: {gait_metrics.get('periodicity_score', 0):.4f}")
    
    env.close()
    
    # Aggregate metrics
    avg_gait_quality = np.mean([m['gait_quality'] for m in all_gait_metrics])
    avg_antiphase = np.mean([m['antiphase_score'] for m in all_gait_metrics])
    avg_periodicity = np.mean([m['periodicity_score'] for m in all_gait_metrics])
    
    print(f"\n{'='*60}")
    print("GAIT EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Mean Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"\nVelocity Metrics:")
    print(f"  Mean Max Velocity: {np.mean(max_velocities):.3f} m/s")
    print(f"  Mean Min Sustained: {np.mean(min_sustained_velocities):.3f} m/s")
    print(f"\nGait Quality Metrics:")
    print(f"  Overall Gait Quality: {avg_gait_quality:.4f} (0-1, higher is better)")
    print(f"  Anti-phase Score: {avg_antiphase:.4f} (legs alternating)")
    print(f"  Periodicity Score: {avg_periodicity:.4f} (rhythmic gait)")
    print(f"{'='*60}\n")
    
    return {
        'rewards': rewards,
        'lengths': lengths,
        'gait_metrics': all_gait_metrics,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_max_velocity': np.mean(max_velocities),
        'mean_min_sustained': np.mean(min_sustained_velocities),
        'avg_gait_quality': avg_gait_quality,
        'avg_antiphase_score': avg_antiphase,
        'avg_periodicity': avg_periodicity,
    }