from .card import Deck
from .player import Player
from .hand import Hand, HandStatus


class BlackjackGame:
    """
    Simulates a standard casino Blackjack game.
    Handles dealing, player actions (hit, stand, double, split),
    dealer resolution (S17), and bet settlement (3:2 blackjack payout).
    """

    def __init__(
        self,
        player: Player,
        num_decks: int = 6,
    ):
        self.player = player
        self.deck = Deck(num_decks=num_decks)
        self.dealer_hand = Hand()

    def start_round(self, bet: float):
        """
        Starts a new round. Clears hands, places the initial bet,
        and deals 2 cards to the player and the dealer.
        """
        self.dealer_hand = Hand()
        self.player.reset_hands()
        self.player.start_hand(bet)

        # Deal 2 cards each
        for _ in range(2):
            for hand in self.player.hands:
                hand.add_card(self.deck.draw())
            self.dealer_hand.add_card(self.deck.draw())

    def hit(self, hand_index: int = 0):
        """Draws one card for the specified hand."""
        hand = self.player.hands[hand_index]
        if hand.status == HandStatus.PLAYING:
            hand.add_card(self.deck.draw())

    def stand(self, hand_index: int = 0):
        """Stands on the specified hand, ending its turn."""
        hand = self.player.hands[hand_index]
        if hand.status == HandStatus.PLAYING:
            hand.status = HandStatus.STAND

    def double_down(self, hand_index: int = 0) -> bool:
        """
        Doubles the bet, draws exactly one card, and stands.
        Returns True if successful, False if illegal (insufficient funds or >2 cards).
        """
        hand = self.player.hands[hand_index]
        if hand.status == HandStatus.PLAYING and len(hand.cards) == 2:
            if self.player.bankroll >= hand.bet:
                self.player.bankroll -= hand.bet
                hand.bet *= 2
                hand.add_card(self.deck.draw())
                if hand.status == HandStatus.PLAYING:
                    hand.status = HandStatus.STAND
                return True
        return False

    def split(self, hand_index: int = 0) -> bool:
        """
        Splits a hand into two separate hands.
        Returns True if successful, False if illegal.
        """
        try:
            self.player.split_hand(hand_index)
            self.player.hands[hand_index].add_card(self.deck.draw())
            self.player.hands[-1].add_card(self.deck.draw())
            return True
        except ValueError:
            return False

    def resolve_dealer(self):
        """
        Dealer plays out their hand. The dealer must hit until 17,
        and stands on all 17s (S17 rule).
        """
        # Dealer must hit until 17 (stands on all 17s)
        while self.dealer_hand.value < 17:
            self.dealer_hand.add_card(self.deck.draw())

    def settle_bets(self):
        """
        Compares player hands against the dealer's hand and updates the player's
        bankroll accordingly (win, loss, push, or blackjack 3:2 payout).
        """
        dealer_val = self.dealer_hand.value
        dealer_blackjack = self.dealer_hand.status == HandStatus.BLACKJACK

        for hand in self.player.hands:
            if hand.status == HandStatus.BUSTED:
                continue  # Lose

            if hand.status == HandStatus.BLACKJACK:
                if dealer_blackjack:
                    self.player.push_bet(hand)
                else:
                    self.player.win_bet(
                        hand, 2.5
                    )  # 1.0 (return bet) + 1.5x payout (3:2)
                continue

            if dealer_val > 21 or hand.value > dealer_val:
                self.player.win_bet(hand, 2.0)
            elif hand.value == dealer_val:
                self.player.push_bet(hand)
            # Else lose, bet is already deducted
