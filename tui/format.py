"""
tui/format.py — Card formatting helpers for the TUI.
"""

_SUIT_COLOR = {
    "Hearts": "bold red",
    "Diamonds": "bold red",
    "Clubs": "bold white",
    "Spades": "bold white",
}
_SUIT_SYMBOL = {
    "Hearts": "♥",
    "Diamonds": "♦",
    "Clubs": "♣",
    "Spades": "♠",
}
_ACTION_LABEL = {0: "HIT", 1: "STAND", 2: "DOUBLE", 3: "SPLIT"}
_ACTION_COLOR = {0: "yellow", 1: "cyan", 2: "magenta", 3: "blue"}

_SPEEDS = [2.0, 1.5, 1.0, 0.7, 0.4, 0.2, 0.1, 0.05]
_DEFAULT_SPEED_IDX = 4  # 0.4s


def fmt_card(card_dict: dict) -> str:
    """Format a single card dict into a colored string."""
    rank = card_dict["rank"].replace("_", "")
    suit = card_dict["suit"]
    sym = _SUIT_SYMBOL.get(suit, "?")
    col = _SUIT_COLOR.get(suit, "white")
    short = {
        "TWO": "2",
        "THREE": "3",
        "FOUR": "4",
        "FIVE": "5",
        "SIX": "6",
        "SEVEN": "7",
        "EIGHT": "8",
        "NINE": "9",
        "TEN": "10",
        "JACK": "J",
        "QUEEN": "Q",
        "KING": "K",
        "ACE": "A",
    }.get(rank, rank)
    return f"[{col}][{short}{sym}][/{col}]"


def fmt_hidden() -> str:
    """Return a hidden card representation."""
    return "[dim white][ ? ][/dim white]"


def fmt_cards(cards: list[dict], hide_first: bool = False) -> str:
    """Format a list of cards, optionally hiding the first one."""
    parts = []
    for i, c in enumerate(cards):
        parts.append(fmt_hidden() if i == 0 and hide_first else fmt_card(c))
    return "  ".join(parts)
