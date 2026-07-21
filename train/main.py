import os
import argparse
from collections import deque
from datetime import datetime
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from train.env import BlackjackEnv
from utils.config import load_config, save_config, apply_overrides
from utils.charts import generate_training_charts
from utils.report import generate_run_readme


class MetricsCallback(BaseCallback):
    """
    Tracks win/loss/tie rates using a rolling window.
    Logs to the SB3 logger every `log_interval` timesteps.
    """

    def __init__(self, window: int = 2000, log_interval: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self._results: deque[str] = deque(maxlen=window)
        self._log_interval = log_interval

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            result = info.get("result")
            if result:
                self._results.append(result)

        if self.num_timesteps % self._log_interval == 0 and self._results:
            total = len(self._results)
            wins = self._results.count("win")
            losses = self._results.count("loss")
            ties = self._results.count("tie")
            self.logger.record("custom/win_rate", wins / total)
            self.logger.record("custom/loss_rate", losses / total)
            self.logger.record("custom/tie_rate", ties / total)

        return True


def train_agent(config: dict, run_name: str | None = None):
    timestamp = run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = f"artifacts/{timestamp}"
    train_dir = os.path.join(base_dir, "train")
    models_dir = os.path.join(train_dir, "models")
    charts_dir = os.path.join(train_dir, "charts")

    os.makedirs(models_dir, exist_ok=True)

    # Save the full config used for this run
    save_config(config, os.path.join(base_dir, "config.json"))

    timesteps = config["train"].get("timesteps", 5_000_000)

    print(f"\n{'━' * 50}")
    print("  Blackjack AI — Training")
    print(f"  Artifact : {base_dir}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"{'━' * 50}\n")

    def mask_fn(env) -> np.ndarray:
        return np.array(env.unwrapped.action_masks(), dtype=bool)

    def wrap_env(env) -> ActionMasker:
        return ActionMasker(env, action_mask_fn=mask_fn)

    n_envs = config.get("train", {}).get("n_envs", 4)
    env_kwargs = {"config": config}
    env = make_vec_env(
        BlackjackEnv, n_envs=n_envs, env_kwargs=env_kwargs, wrapper_class=wrap_env
    )
    eval_env = make_vec_env(
        BlackjackEnv, n_envs=1, env_kwargs=env_kwargs, wrapper_class=wrap_env
    )

    train_cfg = config["train"]

    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=train_cfg.get("n_steps", 1024),
        batch_size=train_cfg.get("batch_size", 256),
        n_epochs=train_cfg.get("n_epochs", 4),
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        gamma=train_cfg.get("gamma", 1.0),
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
        clip_range=train_cfg.get("clip_range", 0.2),
        ent_coef=train_cfg.get("ent_coef", 0.05),
        verbose=0,
        policy_kwargs=dict(net_arch=train_cfg.get("net_arch", [64, 64])),
    )

    sb3_logger = configure(train_dir, ["stdout", "csv"])
    model.set_logger(sb3_logger)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=train_dir,
        eval_freq=10_000,
        n_eval_episodes=500,
        deterministic=True,
        render=False,
        verbose=0,
    )
    metrics_callback = MetricsCallback(window=2000, log_interval=2048)
    callbacks = CallbackList([eval_callback, metrics_callback])

    model.learn(total_timesteps=timesteps, callback=callbacks)

    # Save models
    last_model_path = os.path.join(models_dir, "last_model")
    model.save(last_model_path)
    print(f"\n  Final model → {last_model_path}.zip")

    csv_path = os.path.join(train_dir, "progress.csv")

    # Generate charts
    os.makedirs(charts_dir, exist_ok=True)
    metrics = generate_training_charts(csv_path, charts_dir)

    # README
    generate_run_readme(base_dir, timestamp, timesteps, metrics, train_cfg)

    print(f"\n{'━' * 50}")
    print(f"  Done. Artifact: artifacts/{timestamp}")
    print(f"{'━' * 50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Blackjack AI agent.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the base configuration JSON file (default: config.json)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional artifact folder name (e.g. 2026-07-21_18-00-00)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides train.timesteps)",
    )
    parser.add_argument(
        "--set",
        nargs="+",
        help="Override config values using dot notation (e.g. --set env.num_decks=1 train.batch_size=128)",
    )

    args = parser.parse_args()

    # Load base config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Warning: {args.config} not found, using empty config dictionary.")
        config = {"train": {}}

    # Apply dynamic overrides
    if args.set:
        apply_overrides(config, args.set)

    # Apply top-level convenience overrides
    if args.timesteps is not None:
        if "train" not in config:
            config["train"] = {}
        config["train"]["timesteps"] = args.timesteps

    train_agent(config, args.name)
