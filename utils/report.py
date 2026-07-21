import os


def generate_run_readme(
    base_dir: str, timestamp: str, timesteps: int, metrics: dict, config: dict
):
    win_rate = metrics.get("win_rate", "N/A")
    reward = metrics.get("reward_mean", "N/A")
    content = f"""# Run: {timestamp}

## Configuration

| Key | Value |
| - | - |
| Algorithm | MaskablePPO (MlpPolicy) |
| Timesteps | {timesteps:,} |
| Observation | 6 features |
| Network | {config.get("net_arch", "[64, 64]")} |
| Parallel Envs | 4 |
| n_steps | {config.get("n_steps")} |
| batch_size | {config.get("batch_size")} |
| n_epochs | {config.get("n_epochs")} |
| gamma | {config.get("gamma")} |
| ent_coef | {config.get("ent_coef")} |
| learning_rate | {config.get("learning_rate")} |
| gae_lambda | {config.get("gae_lambda")} |
| clip_range | {config.get("clip_range")} |

## Results

| Metric | Value |
| - | - |
| Win Rate (rolling) | {win_rate} |
| Mean Reward (smoothed) | {reward} |

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
python -m eval.main {timestamp}
python -m eval.main {timestamp} --episodes 20000
```
"""
    readme_path = os.path.join(base_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  README  → {readme_path}")
