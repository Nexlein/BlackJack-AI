import random
from enum import Enum


class Suit(Enum):
    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"
    SPADES = "Spades"


class Rank(Enum):
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


class Card:
    """
    Represents a single playing card with a Suit and Rank.
    """

    def __init__(self, rank: Rank, suit: Suit):
        self.rank = rank
        self.suit = suit

    @property
    def value(self) -> int:
        if self.rank in (Rank.JACK, Rank.QUEEN, Rank.KING):
            return 10
        elif self.rank == Rank.ACE:
            return 11
        else:
            return int(self.rank.value)

    def __repr__(self) -> str:
        return f"{self.rank.name} of {self.suit.value}"

    def to_dict(self):
        return {"rank": self.rank.name, "suit": self.suit.value, "value": self.value}


class Deck:
    """
    Represents a shoe of multiple playing card decks.
    Automatically reshuffles when penetration drops below 15 cards.
    """

    def __init__(self, num_decks: int = 6):
        self.num_decks = num_decks
        self.cards = []
        self._build()

    def _build(self):
        self.cards = [
            Card(rank, suit)
            for _ in range(self.num_decks)
            for suit in Suit
            for rank in Rank
        ]
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self) -> Card:
        if len(self.cards) < 15:  # Shoe penetration reshuffle point
            self._build()
        return self.cards.pop()

    @property
    def remaining(self) -> int:
        return len(self.cards)
