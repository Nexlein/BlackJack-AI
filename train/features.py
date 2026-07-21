import numpy as np

MAX_HAND_VALUE = 30.0
MAX_UPCARD = 11.0
MAX_CARDS = 10.0


def build_observation(
    player_total: int,
    dealer_upcard: int,
    is_soft: bool,
    can_split: bool,
    num_cards: int = 2,
    can_double: bool = False,
) -> np.ndarray:
    """
    Build a 6-feature observation vector for the neural network.

    Features:
        0  player_total   — normalized hand value [0, 1]
        1  dealer_upcard  — normalized dealer upcard [0, 1]
        2  is_soft        — 1 if hand contains an active Ace counted as 11
        3  can_split      — 1 if hand is splittable
        4  num_cards      — normalized card count (capped at MAX_CARDS)
        5  can_double     — 1 if doubling is legal (2 cards + enough bankroll)
    """
    return np.array(
        [
            float(player_total) / MAX_HAND_VALUE,
            float(dealer_upcard) / MAX_UPCARD,
            1.0 if is_soft else 0.0,
            1.0 if can_split else 0.0,
            min(float(num_cards) / MAX_CARDS, 1.0),
            1.0 if can_double else 0.0,
        ],
        dtype=np.float32,
    )
