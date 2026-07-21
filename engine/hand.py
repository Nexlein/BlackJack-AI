from enum import Enum
from typing import List
from .card import Card, Rank


class HandStatus(Enum):
    PLAYING = "Playing"
    BUSTED = "Busted"
    STAND = "Stand"
    BLACKJACK = "Blackjack"


class Hand:
    """
    Represents a player's or dealer's hand of cards.
    Tracks cards, bet amount, total value (with Ace resolution), and status.
    """

    def __init__(self, bet: float = 0.0):
        self.cards: List[Card] = []
        self.bet = bet
        self.status = HandStatus.PLAYING
        self.is_split = False

    def add_card(self, card: Card):
        self.cards.append(card)
        if self.value > 21:
            self.status = HandStatus.BUSTED
        elif self.value == 21 and len(self.cards) == 2 and not self.is_split:
            self.status = HandStatus.BLACKJACK

    def _eval(self) -> tuple[int, bool]:
        total = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == Rank.ACE)
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        return total, aces > 0

    @property
    def value(self) -> int:
        return self._eval()[0]

    @property
    def is_soft(self) -> bool:
        return self._eval()[1]

    def can_split(self) -> bool:
        return len(self.cards) == 2 and self.cards[0].value == self.cards[1].value

    def to_dict(self):
        return {
            "cards": [c.to_dict() for c in self.cards],
            "value": self.value,
            "bet": self.bet,
            "status": self.status.value,
            "is_soft": self.is_soft,
            "can_split": self.can_split(),
        }
