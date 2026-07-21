"""
tui/main.py — Blackjack AI Control Panel (Textual TUI)

Layout (no tabs — always-visible split):
  ┌─────────┬───────┐
  │  TRAIN  │       │
  ├─────────│ WATCH │
  │  EVAL   │       │
  └─────────┴───────┘

Keybindings:
  t  — start training
  e  — evaluate latest run
  w  — start / stop watching AI play
  p  — pause / resume watch
  +  — speed up watch (less delay)
  -  — slow down watch (more delay)
  s  — stop all processes
  q  — quit
"""

from __future__ import annotations

import os
import sys
import time
import signal
import subprocess
import threading
from typing import Optional

# Ensure the project root is in sys.path so 'train', 'eval', 'utils' can be imported
# even if the script is run directly as `python tui/main.py` instead of `-m tui.main`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, RichLog, Label, Static
from textual.containers import Horizontal, Vertical
from textual.binding import Binding
from textual import work

from tui.format import (
    fmt_card,
    fmt_cards,
    _ACTION_COLOR,
    _ACTION_LABEL,
    _SPEEDS,
    _DEFAULT_SPEED_IDX,
)

# ──────────────────────────────────────────────────────────────────────
# TUI App
# ──────────────────────────────────────────────────────────────────────


class BlackjackTUI(App):
    """Keyboard-driven Textual App to manage the Blackjack AI project."""

    BINDINGS = [
        Binding("t", "train_ai", "Train"),
        Binding("e", "eval_ai", "Eval"),
        Binding("w", "toggle_watch", "Watch"),
        Binding("p", "pause_watch", "Pause"),
        Binding("+", "speed_up", "Faster"),
        Binding("-", "speed_down", "Slower"),
        Binding("s", "stop_all", "Stop All"),
        Binding("q", "quit", "Quit"),
    ]

    CSS_PATH = "styles.tcss"

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        # ── Top metrics bar ────────────────────────────────────────────
        yield Horizontal(
            Label("Reward: [cyan]N/A[/cyan]", id="lbl_reward", classes="metric"),
            Label("Steps: [cyan]—[/cyan]", id="lbl_steps", classes="metric"),
            Label("Win Rate: [cyan]—[/cyan]", id="lbl_winrate", classes="metric"),
            id="metrics_bar",
        )

        # ── 3-panel split ──────────────────────────────────────────────
        with Horizontal(id="main_area"):
            # Left column: Train (top) + Eval (bottom)
            with Vertical(id="left_col"):
                with Vertical(id="train_section"):
                    yield Static("TRAIN", classes="panel_header")
                    yield RichLog(id="train_log", markup=False)

                with Vertical(id="eval_section"):
                    yield Static("EVAL", classes="panel_header")
                    yield RichLog(id="eval_log", markup=False)

            # Right column: Watch (full height)
            with Vertical(id="right_col"):
                yield Static("WATCH", classes="panel_header")
                yield RichLog(id="watch_log", markup=True)
                yield Horizontal(
                    Label(
                        "Hand: [cyan]0[/cyan]", id="score_hand", classes="score_item"
                    ),
                    Label(
                        "Wins: [green]0[/green]", id="score_wins", classes="score_item"
                    ),
                    Label(
                        "Losses: [red]0[/red]", id="score_losses", classes="score_item"
                    ),
                    Label(
                        "Ties: [yellow]0[/yellow]",
                        id="score_ties",
                        classes="score_item",
                    ),
                    Label(
                        "W-Rate: [cyan]—[/cyan]", id="score_wrate", classes="score_item"
                    ),
                    id="watch_scoreboard",
                )
                yield Horizontal(
                    Label(
                        "[dim]w: Start/Stop  p: Pause  +/-: Speed[/dim]",
                        id="watch_hint",
                    ),
                    id="watch_controls",
                )

        yield Footer()

    # ------------------------------------------------------------------
    # Mount
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self.title = "Blackjack AI Control Panel"
        self.procs: list[subprocess.Popen] = []

        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.artifacts_dir = os.path.join(self.root_dir, "artifacts")

        # Watch state
        self._watching = False
        self._paused = False
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._speed_idx = _DEFAULT_SPEED_IDX
        self._hand_count = 0
        self._wins = 0
        self._losses = 0
        self._ties = 0

        for wid in ("train_log", "eval_log", "watch_log"):
            self.query_one(f"#{wid}", RichLog).can_focus = False

        self.query_one("#train_log", RichLog).write("Press t to start training.")
        self.query_one("#eval_log", RichLog).write("Press e to evaluate latest model.")
        self.query_one("#watch_log", RichLog).write(
            "[dim]Press [bold]w[/bold] to watch the AI play.[/dim]"
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _latest_artifact(self) -> Optional[str]:
        if not os.path.exists(self.artifacts_dir):
            return None
        folders = sorted(
            [
                f
                for f in os.listdir(self.artifacts_dir)
                if os.path.isdir(os.path.join(self.artifacts_dir, f))
            ],
            reverse=True,
        )
        return folders[0] if folders else None

    def _latest_model_path(self) -> Optional[str]:
        ts = self._latest_artifact()
        if ts is None:
            return None
        path = os.path.join(self.artifacts_dir, ts, "train", "models", "best_model.zip")
        return path if os.path.exists(path) else None

    @property
    def _watch_speed(self) -> float:
        return _SPEEDS[self._speed_idx]

    # ------------------------------------------------------------------
    # Metrics bar polling (during training)
    # ------------------------------------------------------------------

    def poll_training_metrics(self) -> None:
        import csv
        import json

        ts = self._latest_artifact()
        if ts is None:
            return
        artifact_path = os.path.join(self.artifacts_dir, ts)
        csv_path = os.path.join(artifact_path, "train", "progress.csv")
        if not os.path.exists(csv_path):
            return
        try:
            # Read total_timesteps from config.json
            total_steps = 1_000_000
            config_path = os.path.join(artifact_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as cf:
                    cfg = json.load(cf)
                total_steps = int(cfg.get("train", {}).get("timesteps", 1_000_000))

            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return
            last = rows[-1]
            steps = last.get("time/total_timesteps", "")
            reward = last.get("rollout/ep_rew_mean", "")
            winrate = last.get("custom/win_rate", "")

            steps_int = int(float(steps)) if steps else 0
            pct = steps_int / total_steps * 100

            if steps_int >= 1_000_000:
                steps_str = f"{steps_int / 1e6:.2f}M"
            elif steps_int >= 1_000:
                steps_str = f"{steps_int / 1e3:.0f}k"
            else:
                steps_str = str(steps_int)

            total_str = (
                f"{total_steps / 1e6:.1f}M"
                if total_steps >= 1_000_000
                else str(total_steps)
            )
            steps_label = f"{steps_str} / {total_str} ({pct:.0f}%)"

            if reward:
                reward = f"{float(reward):.3f}"
            if winrate:
                winrate = f"{float(winrate):.1%}"

            self.query_one("#lbl_steps", Label).update(
                f"Steps: [cyan]{steps_label}[/cyan]"
            )
            self.query_one("#lbl_reward", Label).update(
                f"Reward: [cyan]{reward or 'N/A'}[/cyan]"
            )
            self.query_one("#lbl_winrate", Label).update(
                f"Win Rate: [cyan]{winrate or '—'}[/cyan]"
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Watch display helpers (thread-safe via call_from_thread)
    # ------------------------------------------------------------------

    def _watch_write(self, msg: str) -> None:
        self.query_one("#watch_log", RichLog).write(msg)

    def _watch_update_scoreboard(self) -> None:
        total = self._hand_count
        wrate = f"{self._wins / total:.1%}" if total > 0 else "—"
        self.query_one("#score_hand", Label).update(f"Hand: [cyan]{total}[/cyan]")
        self.query_one("#score_wins", Label).update(
            f"Wins: [green]{self._wins}[/green]"
        )
        self.query_one("#score_losses", Label).update(
            f"Losses: [red]{self._losses}[/red]"
        )
        self.query_one("#score_ties", Label).update(
            f"Ties: [yellow]{self._ties}[/yellow]"
        )
        self.query_one("#score_wrate", Label).update(f"W-Rate: [cyan]{wrate}[/cyan]")

    # ------------------------------------------------------------------
    # Watch worker
    # ------------------------------------------------------------------

    @work(thread=True)
    def run_watch_loop(self) -> None:
        import numpy as np
        from sb3_contrib import MaskablePPO
        from train.env import BlackjackEnv

        model_path = self._latest_model_path()
        if model_path is None:
            self.call_from_thread(
                self._watch_write,
                "[bold red]No trained model found. Train first (press t).[/bold red]",
            )
            self._watching = False
            return

        self.call_from_thread(
            self._watch_write, f"[dim]Loading model: {model_path}[/dim]"
        )

        import json

        config = {}
        latest = self._latest_artifact()
        if latest is not None:
            artifact_dir = os.path.join(self.artifacts_dir, latest)
            config_path = os.path.join(artifact_dir, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

        model = MaskablePPO.load(model_path)
        env = BlackjackEnv(config=config)

        self.call_from_thread(
            self._watch_write,
            "[bold green]Model loaded. Watching AI play…[/bold green]\n",
        )

        while self._watching:
            self._pause_event.wait()
            if not self._watching:
                break

            obs, _ = env.reset()
            done = False
            hand_n = self._hand_count + 1

            self.call_from_thread(
                self._watch_write,
                f"[dim]{'─' * 44}[/dim]  Hand #{hand_n}",
            )

            dealer_cards_str = fmt_cards(
                [c.to_dict() for c in env.game.dealer_hand.cards], hide_first=True
            )
            player_cards_str = fmt_cards(
                [c.to_dict() for c in env.player.hands[0].cards]
            )
            self.call_from_thread(
                self._watch_write,
                f"  🃏  Dealer: {dealer_cards_str}  [dim](? hidden)[/dim]",
            )
            self.call_from_thread(
                self._watch_write,
                f"  🤖  AI:     {player_cards_str}  "
                f"Total: [bold]{env.player.hands[0].value}[/bold]",
            )

            time.sleep(self._watch_speed)

            while not done:
                self._pause_event.wait()
                if not self._watching:
                    break

                action_masks = np.array(env.action_masks(), dtype=bool)
                action, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=True
                )

                current_hand_idx = env._hand_idx

                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

                action_int = int(action)
                action_color = _ACTION_COLOR.get(action_int, "white")
                action_label = _ACTION_LABEL.get(action_int, "?")

                acted_hand = env.player.hands[current_hand_idx]

                if action_int in (0, 2):  # Hit or Double Down
                    new_card_str = (
                        fmt_card(acted_hand.cards[-1].to_dict())
                        if acted_hand.cards
                        else ""
                    )
                    status_str = (
                        " [red](Bust)[/red]"
                        if acted_hand.status.name == "BUSTED"
                        else ""
                    )
                    self.call_from_thread(
                        self._watch_write,
                        f"  [{action_color}]→ {action_label}[/{action_color}]  drew {new_card_str}  Total: [bold]{acted_hand.value}[/bold]{status_str}",
                    )
                elif action_int == 3:  # Split
                    self.call_from_thread(
                        self._watch_write,
                        f"  [{action_color}]→ {action_label}[/{action_color}]  (Hand split)",
                    )
                else:  # Stand
                    self.call_from_thread(
                        self._watch_write,
                        f"  [{action_color}]→ {action_label}[/{action_color}]  Total: [bold]{acted_hand.value}[/bold]",
                    )

                if done:
                    dealer_reveal = fmt_cards(
                        [c.to_dict() for c in env.game.dealer_hand.cards]
                    )
                    dealer_total = env.game.dealer_hand.value
                    self.call_from_thread(
                        self._watch_write,
                        f"  🃏  Dealer reveals: {dealer_reveal}  "
                        f"Total: [bold]{dealer_total}[/bold]",
                    )

                    result = info.get("result", "tie")
                    if result == "win":
                        res_str = "[bold green]✅ WIN[/bold green]"
                        self._wins += 1
                    elif result == "loss":
                        res_str = "[bold red]❌ LOSS[/bold red]"
                        self._losses += 1
                    else:
                        res_str = "[bold yellow]🤝 TIE[/bold yellow]"
                        self._ties += 1

                    profit = info.get("profit", 0.0)
                    profit_str = (
                        f"[green]+{profit:.1f}[/green]"
                        if profit > 0
                        else (
                            f"[red]{profit:.1f}[/red]"
                            if profit < 0
                            else "[yellow]0[/yellow]"
                        )
                    )
                    self.call_from_thread(
                        self._watch_write,
                        f"  Result: {res_str}  ({profit_str} units)",
                    )

                time.sleep(self._watch_speed)

            self._hand_count += 1
            self.call_from_thread(self._watch_update_scoreboard)

        self.call_from_thread(self._watch_write, "[dim]Watch stopped.[/dim]")
        self._watching = False

    # ------------------------------------------------------------------
    # Tail subprocess output
    # ------------------------------------------------------------------

    @work(thread=True)
    def tail_output(self, stream, log_widget: RichLog, log_file: str | None = None):
        f = None
        if log_file:
            f = open(log_file, "a", encoding="utf-8")
        for line in iter(stream.readline, ""):
            if line:
                try:
                    text = line.rstrip()
                    if f:
                        f.write(text + "\n")
                        f.flush()
                    self.call_from_thread(log_widget.write, text)
                except Exception:
                    pass
        if f:
            f.close()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_train_ai(self) -> None:
        self.notify("Starting AI Training…", title="Train")
        target = self.query_one("#train_log", RichLog)
        target.write("\n─── NEW TRAINING RUN ───")

        from datetime import datetime

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        artifact_train_dir = os.path.join(self.artifacts_dir, ts, "train")
        os.makedirs(artifact_train_dir, exist_ok=True)

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        p = subprocess.Popen(
            ["uv", "run", "python", "-m", "train.main", "--name", ts],
            cwd=self.root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        self.procs.append(p)
        self.tail_output(
            p.stdout, target, log_file=os.path.join(artifact_train_dir, "train.log")
        )

        if hasattr(self, "metrics_timer") and self.metrics_timer:
            self.metrics_timer.stop()
        self.metrics_timer = self.set_interval(2.0, self.poll_training_metrics)

    def action_eval_ai(self) -> None:
        ts = self._latest_artifact()
        if ts is None:
            self.notify(
                "No artifacts found — train first.", title="Eval", severity="warning"
            )
            return

        self.notify(f"Evaluating {ts}…", title="Eval")
        target = self.query_one("#eval_log", RichLog)
        target.write(f"\n─── EVAL: {ts} ───")

        artifact_eval_dir = os.path.join(self.artifacts_dir, ts, "eval")
        os.makedirs(artifact_eval_dir, exist_ok=True)

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        p = subprocess.Popen(
            ["uv", "run", "python", "-m", "eval.main", ts],
            cwd=self.root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            start_new_session=True,
        )
        self.procs.append(p)
        self.tail_output(
            p.stdout, target, log_file=os.path.join(artifact_eval_dir, "eval.log")
        )

    def action_toggle_watch(self) -> None:
        if self._watching:
            self._watching = False
            self._pause_event.set()
            self.notify("Watch stopped.", title="Watch")
            self.query_one("#watch_hint", Label).update(
                "[dim]w: Start/Stop  p: Pause  +/-: Speed[/dim]"
            )
        else:
            if self._latest_model_path() is None:
                self.notify(
                    "No model found — train first.", title="Watch", severity="warning"
                )
                return
            self._watching = True
            self._paused = False
            self._pause_event.set()
            speed = self._watch_speed
            self.notify(f"Watching at {speed:.2f}s/step", title="Watch")
            self.query_one("#watch_hint", Label).update(
                f"[green]▶ Playing[/green]  speed: {speed:.2f}s  "
                "[dim]p: Pause  +/-: Speed  w: Stop[/dim]"
            )
            self.run_watch_loop()

    def action_pause_watch(self) -> None:
        if not self._watching:
            return
        if self._paused:
            self._paused = False
            self._pause_event.set()
            speed = self._watch_speed
            self.notify("Resumed.", title="Watch")
            self.query_one("#watch_hint", Label).update(
                f"[green]▶ Playing[/green]  speed: {speed:.2f}s  "
                "[dim]p: Pause  +/-: Speed  w: Stop[/dim]"
            )
        else:
            self._paused = True
            self._pause_event.clear()
            self.notify("Paused.", title="Watch")
            self.query_one("#watch_hint", Label).update(
                "[yellow]⏸ Paused[/yellow]  [dim]p: Resume  w: Stop[/dim]"
            )

    def action_speed_up(self) -> None:
        if self._speed_idx > 0:
            self._speed_idx -= 1
        self.notify(f"Speed: {self._watch_speed:.2f}s/step", title="Watch")

    def action_speed_down(self) -> None:
        if self._speed_idx < len(_SPEEDS) - 1:
            self._speed_idx += 1
        self.notify(f"Speed: {self._watch_speed:.2f}s/step", title="Watch")

    def action_stop_all(self) -> None:
        self._watching = False
        self._pause_event.set()
        self.notify("Stopping all processes…", title="System", severity="warning")
        for p in self.procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        self.procs = []

    def on_unmount(self) -> None:
        self.action_stop_all()


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = BlackjackTUI()
    app.run()
