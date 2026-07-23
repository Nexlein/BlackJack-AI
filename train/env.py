import gymnasium as gym
from gymnasium import spaces
import numpy as np
from engine.game import BlackjackGame
from engine.player import Player
from engine.hand import HandStatus


class BlackjackEnv(gym.Env):
    """
    Blackjack Gymnasium Environment.

    Action Space (Discrete 4):
        0 — Hit
        1 — Stand
        2 — Double Down
        3 — Split

    Observation Space (Box 6):
        [player_total, dealer_upcard, is_soft, can_split, num_cards, can_double]

    Multi-hand (after split) is handled by sequentially playing each hand.
    Natural blackjacks are handled correctly — they terminate the episode with
    the proper +1.5× payout reward, giving the model a signal to expect them.
    """

    INITIAL_BET = 10.0

    def __init__(self, config: dict | None = None):
        super().__init__()

        self.config = config or {}
        self.env_cfg = self.config.get("env", {})
        self.num_decks = self.env_cfg.get("num_decks", 6)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

        self.player = Player("AI", 1000)
        self.game = BlackjackGame(
            self.player,
            num_decks=self.num_decks,
        )
        self._hand_idx = 0
        self._blackjack_pending = False  # Natural BJ — resolve on next step()

    def step(self, action: int):
        """
        Executes the given action in the environment.
        Advances hand logic, resolves dealer if all hands are done,
        calculates rewards, and returns the observation step tuple.
        """
        # ── Natural blackjack: resolve immediately regardless of action ──
        if self._blackjack_pending:
            self._blackjack_pending = False
            self.game.resolve_dealer()

            total_bet = sum(h.bet for h in self.player.hands)
            start_bankroll = self.player.bankroll + total_bet
            self.game.settle_bets()

            profit = self.player.bankroll - start_bankroll
            reward = profit / self.INITIAL_BET
            result = "win" if profit > 0 else ("loss" if profit < 0 else "tie")
            return (
                self._get_obs(),
                reward,
                True,
                False,
                {"result": result, "profit": profit},
            )

        reward = 0.0
        terminated = False

        hand = self.player.hands[self._hand_idx]

        if action == 0:  # Hit
            self.game.hit(self._hand_idx)

        elif action == 1:  # Stand
            self.game.stand(self._hand_idx)

        elif action == 2:  # Double Down
            if not self.game.double_down(self._hand_idx):
                self.game.hit(self._hand_idx)

        elif action == 3:  # Split
            if not self.game.split(self._hand_idx):
                self.game.hit(self._hand_idx)

        # Re-fetch hand (split may have appended a new hand)
        hand = self.player.hands[self._hand_idx]

        # Current hand finished — advance to next
        if hand.status != HandStatus.PLAYING:
            self._hand_idx += 1

        # All hands played — resolve dealer and settle
        if self._hand_idx >= len(self.player.hands):
            terminated = True

            any_alive = any(h.status != HandStatus.BUSTED for h in self.player.hands)
            if any_alive:
                self.game.resolve_dealer()

            total_bet = sum(h.bet for h in self.player.hands)
            start_bankroll = self.player.bankroll + total_bet
            self.game.settle_bets()

            profit = self.player.bankroll - start_bankroll
            reward += profit / self.INITIAL_BET

            result = "win" if profit > 0 else ("loss" if profit < 0 else "tie")
            info = {"result": result, "profit": profit}
        else:
            info = {}

        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        """
        Constructs the observation vector for the current state.
        See `build_observation` for feature scaling and representation.
        """
        from train.features import build_observation

        if not self.player.hands or not self.game.dealer_hand:
            return build_observation(0, 0, False, False)

        hand_idx = min(self._hand_idx, len(self.player.hands) - 1)
        hand = self.player.hands[hand_idx]

        player_total = hand.value
        dealer_upcard = (
            self.game.dealer_hand.cards[0].value if self.game.dealer_hand.cards else 0
        )
        is_soft = hand.is_soft
        can_split = hand.can_split()
        num_cards = len(hand.cards)
        can_double = len(hand.cards) == 2 and self.player.bankroll >= hand.bet

        return build_observation(
            player_total,
            dealer_upcard,
            is_soft,
            can_split,
            num_cards,
            can_double,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Resets the environment for a new episode.
        Restores bankroll and deals initial cards.
        """
        super().reset(seed=seed, options=options)

        self.player.bankroll = 1000.0
        self.player.reset_hands()
        self._hand_idx = 0
        self._blackjack_pending = False

        self.game.start_round(self.INITIAL_BET)

        # Natural blackjack — flag it, resolve via step() so reward is captured
        if self.player.hands[0].status == HandStatus.BLACKJACK:
            self._blackjack_pending = True

        return self._get_obs(), {}

    def action_masks(self) -> list[bool]:
        """
        Action mask for MaskablePPO.
        Returns a boolean list for [Hit, Stand, Double, Split].
        """
        if not self.player.hands or self._hand_idx >= len(self.player.hands):
            return [False, False, False, False]

        hand = self.player.hands[self._hand_idx]
        can_hit = hand.value < 21
        can_stand = True
        can_double = (
            len(hand.cards) == 2
            and hand.value < 21
            and self.player.bankroll >= hand.bet
        )
        can_split = (
            hand.can_split()
            and self.player.bankroll >= hand.bet
            and len(self.player.hands) < 4
        )

        return [can_hit, can_stand, can_double, can_split]
