# Run: 2026-07-21_23-31-13

## Configuration

| Key | Value |
| - | - |
| Algorithm | MaskablePPO (MlpPolicy) |
| Timesteps | 5,000,000 |
| Observation | 6 features |
| Network | [256, 256] |
| Parallel Envs | 4 |
| n_steps | 4096 |
| batch_size | 2048 |
| n_epochs | 4 |
| gamma | 1.0 |
| ent_coef | 0.05 |
| learning_rate | 0.0001 |
| gae_lambda | 0.95 |
| clip_range | 0.2 |

## Results

| Metric | Value |
| - | - |
| Win Rate (rolling) | 41.60% |
| Mean Reward (smoothed) | -0.0707 |

## Files

```bash
train/
  models/best_model.zip   ← best checkpoint by eval reward
  models/last_model.zip   ← final checkpoint
  progress.csv            ← full tensorboard metrics
  charts/training_reward.png
  charts/training_win_rate.png
```

## Evaluate

```bash
python -m eval.main 2026-07-21_23-31-13
python -m eval.main 2026-07-21_23-31-13 --episodes 20000
```
