import os
import argparse
import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from train.env import BlackjackEnv
from utils.config import load_config, apply_overrides
from utils.charts import (
    chart_cumulative_reward,
    chart_rolling_win_rate,
    chart_strategy_heatmap,
    chart_action_distribution,
)


def evaluate_model(
    timestamp: str,
    episodes: int = 10_000,
    config: dict | None = None,
    model_name: str = "best_model.zip",
):
    if config is None:
        config = {}
    artifact_dir = f"artifacts/{timestamp}"
    train_dir = os.path.join(artifact_dir, "train")
    models_dir = os.path.join(train_dir, "models")
    eval_dir = os.path.join(artifact_dir, "eval")
    charts_dir = os.path.join(eval_dir, "charts")

    model_path = os.path.join(models_dir, model_name)

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    for d in (eval_dir, charts_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\n{'━' * 50}")
    print("  Blackjack AI — Evaluation")
    print(f"  Model    : {model_path}")
    print(f"  Episodes : {episodes:,}")
    print(f"{'━' * 50}\n")

    model = MaskablePPO.load(model_path)
    env = BlackjackEnv(config=config)

    # Run episodes
    episode_records = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Hit / Stand / Double / Split

    for i in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_result = "tie"
        info: dict = {}

        while not done:
            action_masks = np.array(env.action_masks(), dtype=bool)
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            action_int = int(action)
            action_counts[action_int] += 1
            obs, reward, terminated, truncated, info = env.step(action_int)
            ep_reward += reward
            done = terminated or truncated

        if "result" in info:
            ep_result = info["result"]

        episode_records.append(
            {
                "episode": i,
                "result": ep_result,
                "reward": ep_reward,
            }
        )

        if (i + 1) % (episodes // 10) == 0:
            done_pct = (i + 1) / episodes * 100
            print(f"  Progress: {done_pct:.0f}%  ({i + 1:,}/{episodes:,})")

    # Aggregate metrics
    df = pd.DataFrame(episode_records)

    wins = (df["result"] == "win").sum()
    losses = (df["result"] == "loss").sum()
    ties = (df["result"] == "tie").sum()

    win_rate = wins / episodes
    loss_rate = losses / episodes
    tie_rate = ties / episodes
    edge = df["reward"].mean()

    # Save per-episode log
    per_ep_path = os.path.join(eval_dir, "per_episode.csv")
    df.to_csv(per_ep_path, index=False)
    print(f"\n  Per-episode log → {per_ep_path}")

    # Save summary
    summary = pd.DataFrame(
        [
            {
                "episodes": episodes,
                "wins": int(wins),
                "losses": int(losses),
                "ties": int(ties),
                "win_rate": win_rate,
                "loss_rate": loss_rate,
                "tie_rate": tie_rate,
                "edge": edge,
            }
        ]
    )
    summary_path = os.path.join(eval_dir, "results.csv")
    summary.to_csv(summary_path, index=False)

    total_actions = sum(action_counts.values())
    action_pcts = {k: v / total_actions * 100 for k, v in action_counts.items()}

    print(f"\n{'━' * 50}")
    print("  Results")
    print(f"{'━' * 50}")
    print(f"  Win Rate  : {win_rate:.2%}")
    print(f"  Loss Rate : {loss_rate:.2%}")
    print(f"  Tie Rate  : {tie_rate:.2%}")
    print(f"  Edge      : {edge:+.4f}  (avg reward per hand)")
    print(f"{'━' * 50}")
    print(
        f"  Actions   : Hit {action_pcts[0]:.1f}%  Stand {action_pcts[1]:.1f}%  "
        f"Double {action_pcts[2]:.1f}%  Split {action_pcts[3]:.1f}%"
    )
    print(f"{'━' * 50}\n")

    # Charts
    print("  Generating charts…")

    chart_cumulative_reward(df, os.path.join(charts_dir, "cumulative_reward.png"))
    print("  ✓ cumulative_reward.png")

    chart_rolling_win_rate(df, os.path.join(charts_dir, "win_rate_rolling.png"))
    print("  ✓ win_rate_rolling.png")

    chart_strategy_heatmap(
        model, os.path.join(charts_dir, "strategy_hard.png"), soft=False
    )
    print("  ✓ strategy_hard.png")

    chart_strategy_heatmap(
        model, os.path.join(charts_dir, "strategy_soft.png"), soft=True
    )
    print("  ✓ strategy_soft.png")

    chart_action_distribution(action_pcts, charts_dir)
    print("  ✓ action_distribution.png")

    print(f"\n  All charts → {charts_dir}")
    print("  Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Blackjack AI model."
    )
    parser.add_argument(
        "timestamp", type=str, help="Artifact folder name in artifacts/"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation episodes (overrides eval.episodes)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best_model.zip",
        help="Name of the model file to evaluate (default: best_model.zip)",
    )
    parser.add_argument(
        "--set",
        nargs="+",
        help="Override config values using dot notation (e.g. --set env.num_decks=1)",
    )
    args = parser.parse_args()

    # Load artifact config
    artifact_dir = f"artifacts/{args.timestamp}"
    config_path = os.path.join(artifact_dir, "config.json")

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Warning: config.json not found in {artifact_dir}. Using defaults.")
        config = {"eval": {}}

    # Apply dynamic overrides
    if args.set:
        apply_overrides(config, args.set)

    episodes = (
        args.episodes
        if args.episodes is not None
        else config.get("eval", {}).get("episodes", 10_000)
    )

    evaluate_model(args.timestamp, episodes, config, args.model)
