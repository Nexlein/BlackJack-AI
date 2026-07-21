import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from train.features import build_observation

plt.style.use("dark_background")
_PALETTE = {
    "win": "#00C49F",
    "loss": "#FF6B6B",
    "tie": "#FFB800",
    "reward": "#6C8EBF",
}

_ACTION_NAMES = {0: "Hit", 1: "Stand", 2: "Double", 3: "Split"}
_ACTION_COLORS = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#1A535C"]


def _smooth(series: pd.Series, window: int = 10) -> pd.Series:
    result = series.rolling(window=window, min_periods=1).mean()
    return result  # type: ignore[return-value]


def generate_training_charts(csv_path: str, output_dir: str) -> dict:
    """Generate training charts from progress.csv. Returns final metric values."""
    metrics = {}
    if not os.path.exists(csv_path):
        return metrics
    try:
        df = pd.read_csv(csv_path)
        if len(df) < 2:
            return metrics

        steps_col = "time/total_timesteps"

        # ── Reward chart ───────────────────────────────────────────────
        if "rollout/ep_rew_mean" in df.columns:
            raw: pd.Series = df[steps_col]  # type: ignore[assignment]
            rew: pd.Series = df["rollout/ep_rew_mean"]  # type: ignore[assignment]
            smoothed = _smooth(rew, window=10)

            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(
                raw,
                rew,
                color=_PALETTE["reward"],
                alpha=0.3,
                linewidth=1.2,
                label="Raw",
            )
            ax.plot(
                raw, smoothed, color=_PALETTE["reward"], linewidth=2.2, label="Smoothed"
            )
            ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)
            ax.set_title("Training — Mean Episode Reward", fontsize=15, pad=12)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Mean Reward")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"
                )
            )
            ax.legend(framealpha=0.3)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "training_reward.png"), dpi=150)
            plt.close(fig)

            metrics["reward_mean"] = f"{smoothed.iloc[-1]:.4f}"

        # ── Win/Loss/Tie rate chart ────────────────────────────────────
        rate_cols = {
            "custom/win_rate": ("Win Rate", _PALETTE["win"]),
            "custom/loss_rate": ("Loss Rate", _PALETTE["loss"]),
            "custom/tie_rate": ("Tie Rate", _PALETTE["tie"]),
        }
        available = {k: v for k, v in rate_cols.items() if k in df.columns}
        if available:
            df_clean = df.dropna(subset=list(available.keys()))
            fig, ax = plt.subplots(figsize=(11, 5))
            for col, (label, color) in available.items():
                vals = df_clean[col] * 100
                ax.plot(
                    df_clean[steps_col],
                    _smooth(vals, window=10),
                    color=color,
                    linewidth=2.2,
                    label=label,
                )
            ax.axhline(
                42.5,
                color="white",
                linewidth=1.0,
                linestyle="--",
                alpha=0.8,
                label="Optimal Basic Strategy (~42.5%)",
            )
            ax.set_title("Training — Win / Loss / Tie Rates", fontsize=15, pad=12)
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Rate (%)")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k"
                )
            )
            ax.legend(framealpha=0.3)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "training_win_rate.png"), dpi=150)
            plt.close(fig)

            if "custom/win_rate" in df_clean.columns:
                metrics["win_rate"] = f"{df_clean['custom/win_rate'].iloc[-1]:.2%}"

        print(f"  Charts saved → {output_dir}")
    except Exception as exc:
        print(f"  Chart generation failed: {exc}")

    return metrics


def chart_cumulative_reward(episode_df: pd.DataFrame, output_path: str):
    cum_series = episode_df["reward"].cumsum()
    cum_vals = cum_series.to_numpy(dtype=float)
    cum_idx = cum_series.index.to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum_idx, cum_vals, color=_PALETTE["reward"], linewidth=1.8)
    ax.axhline(0, color="white", linewidth=0.7, linestyle="--", alpha=0.5)
    ax.fill_between(
        cum_idx,
        cum_vals,
        0,
        where=list(cum_vals >= 0),
        alpha=0.15,
        color=_PALETTE["win"],
    )
    ax.fill_between(
        cum_idx,
        cum_vals,
        0,
        where=list(cum_vals < 0),
        alpha=0.15,
        color=_PALETTE["loss"],
    )
    ax.set_title("Evaluation — Cumulative Reward", fontsize=15, pad=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward (units)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def chart_rolling_win_rate(
    episode_df: pd.DataFrame, output_path: str, window: int = 500
):
    wins = (episode_df["result"] == "win").astype(float)
    rolling_mean_s = wins.rolling(window=window, min_periods=1).mean() * 100
    rolling_std_s = wins.rolling(window=window, min_periods=1).std().fillna(0) * 100

    x = episode_df.index.to_numpy()
    mean_arr = rolling_mean_s.to_numpy(dtype=float)
    std_arr = rolling_std_s.to_numpy(dtype=float)
    lower = np.clip(mean_arr - std_arr, 0, None)
    upper = np.clip(mean_arr + std_arr, None, 100)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        x,
        mean_arr,
        color=_PALETTE["win"],
        linewidth=2.2,
        label=f"Win Rate ({window}-hand rolling)",
    )
    ax.fill_between(x, lower, upper, alpha=0.2, color=_PALETTE["win"])
    ax.axhline(
        42.5,
        color="white",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
        label="Optimal Basic Strategy (~42.5%)",
    )
    ax.set_title("Evaluation — Rolling Win Rate", fontsize=15, pad=12)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend(framealpha=0.3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def chart_strategy_heatmap(model, output_path: str, soft: bool = False):
    """
    Plot the AI's learned strategy for all player totals vs dealer upcards.

    Hard mode: player_total 4–21, is_soft=False
    Soft mode: player_total 13–20 (A+2 through A+9), is_soft=True
    """
    if soft:
        totals = list(range(13, 21))  # A+2 (13) → A+9 (20)
        title = "AI Learned Strategy — Soft Totals (Ace + X)"
        y_labels = [f"A+{t - 11}" for t in totals]
    else:
        totals = list(range(4, 22))  # Hard 4 → 21
        title = "AI Learned Strategy — Hard Totals"
        y_labels = [str(t) for t in totals]

    dealers = list(range(2, 12))
    strategy = np.zeros((len(totals), len(dealers)), dtype=float)

    for i, p in enumerate(totals):
        for j, d in enumerate(dealers):
            obs = build_observation(p, d, soft, False)
            action_masks = np.array(
                [True, True, True, True], dtype=bool
            )  # all allowed for heatmap hypotheticals
            action, _ = model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            strategy[i, j] = int(action)

    cmap = ListedColormap(_ACTION_COLORS)
    fig, ax = plt.subplots(figsize=(12, len(totals) * 0.55 + 2))
    extent: tuple[float, float, float, float] = (1.5, 11.5, -0.5, len(totals) - 0.5)
    im = ax.imshow(
        strategy,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=-0.5,
        vmax=3.5,
        extent=extent,
    )

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([_ACTION_NAMES[i] for i in range(4)])

    ax.set_title(title, fontsize=15, pad=14)
    ax.set_xlabel("Dealer Upcard")
    ax.set_ylabel("Player Total")
    ax.set_xticks(dealers)
    ax.set_xticklabels([str(d) if d < 11 else "A" for d in dealers])
    ax.set_yticks(range(len(totals)))
    ax.set_yticklabels(y_labels)
    ax.grid(False)

    # Annotate cells
    for i in range(len(totals)):
        for j in range(len(dealers)):
            action_char = ["H", "S", "D", "P"][int(strategy[i, j])]
            ax.text(
                dealers[j],
                i,
                action_char,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return strategy


def chart_action_distribution(action_pcts: dict, charts_dir: str):
    action_labels = ["Hit", "Stand", "Double", "Split"]
    action_colors = _ACTION_COLORS
    action_vals = [action_pcts[k] for k in range(4)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(action_labels, action_vals, color=action_colors, width=0.55, zorder=3)
    for bar, val in zip(bars, action_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="white",
        )
    ax.set_title("Evaluation — Action Distribution", fontsize=15, pad=12)
    ax.set_ylabel("% of all decisions")
    ax.set_ylim(0, max(action_vals) * 1.2 + 2)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    fig.tight_layout()
    fig.savefig(os.path.join(charts_dir, "action_distribution.png"), dpi=150)
    plt.close(fig)
