# Walker2d-v5 TD3 with Reward Shaping

A Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation for training bipedal locomotion in the Walker2d-v5 environment with two distinct reward shaping strategies: **Speed Walking** and **Gait Walking**.

## Overview

This project implements TD3 reinforcement learning to train a 2D bipedal walker with customizable reward functions:

- **Speed Walking**: Optimizes for sustained high-speed locomotion
- **Gait Walking**: Balances velocity with biomechanical gait quality (symmetry, periodicity, range of motion)

## Features

- 🚀 Optimized TD3 implementation with experience replay
- 📊 Comprehensive logging with WandB integration
- 🎥 Video recording and frame capture utilities
- 📈 Detailed evaluation metrics for both velocity and gait quality
- 🔧 Modular reward shaping architecture

## Requirements

```bash
pip install gymnasium torch numpy scipy matplotlib pillow wandb
```

For rendering:
```bash
pip install gymnasium[mujoco]
```

## Project Structure

```
├── train.py              # Main training script
├── vis.py               # Visualization and evaluation script
├── td3_agent.py         # TD3 algorithm implementation
├── networks.py          # Actor and Critic networks
├── config.py            # Configuration and hyperparameters
├── utils_vel.py         # Speed walking reward utilities
├── utils_sym.py         # Gait walking reward utilities
├── logging_utils.py     # Logging and model management
└── models/              # Saved model checkpoints
    ├── speed/           # Speed walking models
    └── gait/            # Gait walking models
```

## Usage

### Training

Train a speed walking agent:
```bash
python train.py --training_type speed_walking
```

Train a gait walking agent:
```bash
python train.py --training_type gait_walking
```

**Training Details:**
- Default: 1000 episodes with max 1000 steps per episode
- Models are saved in `models/speed/` or `models/gait/` directories
- Best model saved based on 100-episode moving average of target metric
- WandB logging enabled by default (configurable in `config.py`)

### Evaluation and Visualization

**Run evaluation without rendering:**
```bash
python vis.py --model models/speed/seed_42_best.pth --episodes 10
```

**Watch the agent in real-time:**
```bash
python vis.py --model models/gait/seed_42_best.pth --render
```

**Record a video:**
```bash
python vis.py --model models/speed/seed_42_best.pth --record --episodes 1
```
Videos are saved in `demo_videos/` with format: `{training_type} seed {num} video-episode-{id}.mp4`

**Capture progression frames:**
```bash
python vis.py --model models/speed/seed_42_best.pth --capture
```
Creates a composite image showing 20 freeze frames throughout an episode, saved in `progression_composites/`

**Command Line Arguments:**
- `--model`: Path to model file (default: `models/best.pth`)
- `--training_type`: Override auto-detection (`speed_walking` or `symmetric_walking`)
- `--episodes`: Number of evaluation episodes (default: 5)
- `--render`: Display real-time visualization
- `--record`: Record video of episodes
- `--capture`: Capture 20 freeze frames for progression composite

### Configuration

Modify hyperparameters in `config.py`:

```python
def get_config(training_type='speed_walking'):
    config = {
        'seed': 42,
        'max_episodes': 1000,
        'max_steps': 1000,
        'batch_size': 256,
        'buffer_size': 1000000,
        'warmup_steps': 10000,
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        # ... training-specific parameters
    }
```

## Reward Shaping Strategies

### Speed Walking
Focuses on maximizing forward velocity with components:
- Current velocity bonus
- Sustained velocity rewards (min/mean over sliding window)
- Peak performance bonuses for high-speed maintenance
- Acceleration bonuses
- Episode completion rewards

### Gait Walking
Extends speed walking with biomechanical quality terms:
- **Range of Motion (ROM)**: Rewards active joint movement
- **Anti-Phase Coordination**: Encourages alternating leg motion
- **Periodicity**: Rewards rhythmic, consistent gait cycles
- **Composite Gait Quality**: Weighted combination of all metrics

## Evaluation Metrics

The evaluation script provides comprehensive metrics:

**Common Metrics:**
- Episode rewards and lengths
- Max velocity achieved
- Sustained velocity statistics

**Speed Walking Specific:**
- Minimum sustained velocity over window
- Velocity consistency ratio
- Comparison gait metrics (logged but not optimized)

**Gait Walking Specific:**
- Gait quality score (0-1)
- Anti-phase coordination score
- Periodicity score
- ROM symmetry

## Model Naming Convention

Models are automatically saved with consistent naming:
```
models/{training_type}/seed_{num}_{type}.pth
```

Examples:
- `models/speed/seed_42_best.pth`
- `models/gait/seed_123_final.pth`

## Output Examples

**Training Output:**
```
Episode 500/1000 | Reward: 2847.23 | Steps: 1000 | MaxVel: 4.52 | 
MinSust: 4.12 | AvgSust: 4.28 | GaitQ: 0.723
```

**Evaluation Summary:**
```
==============================================================
COMPREHENSIVE PERFORMANCE SUMMARY
==============================================================
Training Type: speed_walking
Model: models/speed/seed_42_best.pth

BASE METRICS:
  Mean Reward: 2847.23 ± 145.67
  Mean Steps: 1000.0 ± 0.0

VELOCITY METRICS:
  Mean Max Velocity: 4.52 ± 0.12
  Mean Min Sustained Velocity: 4.12 ± 0.08
==============================================================
```

## Tips

1. **For fastest training**: Use `speed_walking` configuration
2. **For natural-looking gaits**: Use `gait_walking` configuration
3. **Hyperparameter tuning**: Adjust reward weights in respective `utils_*.py` files
4. **Multiple seeds**: Modify `seed` in `config.py` for reproducibility experiments
5. **Monitor training**: Check WandB dashboard for live metrics

## Algorithm Details

This implementation uses TD3 with:
- Twin critic networks to reduce overestimation
- Delayed policy updates (every 2 critic steps)
- Target policy smoothing with clipped noise
- Polyak averaging for target network updates
- Experience replay with 1M capacity buffer
- Layer normalization for training stability

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fujimoto2018addressing,
  title={Addressing function approximation error in actor-critic methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  journal={International Conference on Machine Learning},
  year={2018}
}
```

## License

MIT License - feel free to use and modify for your research and projects.

## Troubleshooting

**Issue: "No module named 'mujoco'"**
- Solution: `pip install gymnasium[mujoco]`

**Issue: Training is unstable**
- Try reducing learning rates in `config.py`
- Increase warmup steps
- Check reward scaling in `utils_*.py`

**Issue: Videos not recording**
- Ensure `ffmpeg` is installed: `apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)

**Issue: Model not loading**
- Verify model path is correct
- Check that training_type matches the model's training configuration

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.