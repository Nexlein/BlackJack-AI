from engine.card import Card, Rank, Suit
from engine.hand import Hand, HandStatus
from engine.player import Player
from engine.game import BlackjackGame


def test_card_values():
    assert Card(Rank.ACE, Suit.HEARTS).value == 11
    assert Card(Rank.TEN, Suit.SPADES).value == 10
    assert Card(Rank.KING, Suit.DIAMONDS).value == 10


def test_hand_value():
    hand = Hand()
    hand.add_card(Card(Rank.ACE, Suit.HEARTS))
    hand.add_card(Card(Rank.NINE, Suit.CLUBS))
    assert hand.value == 20
    assert hand.is_soft

    hand.add_card(Card(Rank.FIVE, Suit.DIAMONDS))
    assert hand.value == 15  # 11 + 9 + 5 = 25 -> 1 + 9 + 5 = 15
    assert not hand.is_soft


def test_blackjack():
    hand = Hand()
    hand.add_card(Card(Rank.ACE, Suit.HEARTS))
    hand.add_card(Card(Rank.KING, Suit.CLUBS))
    assert hand.status == HandStatus.BLACKJACK


def test_split():
    player = Player("P1", 100)
    hand = player.start_hand(10)
    hand.add_card(Card(Rank.EIGHT, Suit.HEARTS))
    hand.add_card(Card(Rank.EIGHT, Suit.CLUBS))
    assert hand.can_split()

    player.split_hand(0)
    assert len(player.hands) == 2
    assert player.bankroll == 80  # 100 - 10 - 10
    assert len(player.hands[0].cards) == 1
    assert len(player.hands[1].cards) == 1
    assert player.hands[0].cards[0].value == 8


def test_game_settle():
    p1 = Player("P1", 100)
    game = BlackjackGame(p1)
    game.start_round(10)

    # Force hands and reset status in case start_round dealt a natural blackjack
    p1.hands[0].cards = [Card(Rank.TEN, Suit.HEARTS), Card(Rank.TEN, Suit.CLUBS)]  # 20
    p1.hands[0].status = HandStatus.STAND
    game.dealer_hand.cards = [
        Card(Rank.TEN, Suit.SPADES),
        Card(Rank.NINE, Suit.DIAMONDS),
    ]  # 19

    game.settle_bets()
    assert p1.bankroll == 110  # 90 + 20
