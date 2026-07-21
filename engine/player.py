from typing import List
import uuid
from .hand import Hand


class Player:
    """
    Represents a player in the Blackjack game.
    Manages bankroll and multiple hands (due to splitting).
    """

    def __init__(
        self,
        name: str,
        bankroll: float = 1000.0,
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.bankroll = bankroll
        self.hands: List[Hand] = []

    def reset_hands(self):
        self.hands = []

    def start_hand(self, bet: float) -> Hand:
        if bet > self.bankroll:
            raise ValueError("Insufficient funds")
        self.bankroll -= bet
        new_hand = Hand(bet)
        self.hands.append(new_hand)
        return new_hand

    def split_hand(self, hand_index: int) -> Hand:
        target_hand = self.hands[hand_index]
        if not target_hand.can_split():
            raise ValueError("Cannot split this hand")
        if target_hand.bet > self.bankroll:
            raise ValueError("Insufficient funds to split")

        self.bankroll -= target_hand.bet
        card2 = target_hand.cards.pop()

        new_hand = Hand(target_hand.bet)
        new_hand.add_card(card2)

        target_hand.is_split = True
        new_hand.is_split = True

        self.hands.append(new_hand)
        return new_hand

    def win_bet(self, hand: Hand, multiplier: float = 2.0):
        self.bankroll += hand.bet * multiplier

    def push_bet(self, hand: Hand):
        self.bankroll += hand.bet

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "bankroll": self.bankroll,
            "hands": [h.to_dict() for h in self.hands],
        }
