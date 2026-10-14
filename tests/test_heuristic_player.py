"""Tests for the rules-and-payoff heuristic player."""

from euchre.ai.ai_factory import AIFactory
from euchre.ai.base_ai_interface import BaseAIInterface
from euchre.ai.heuristic_player import HeuristicPlayer
from euchre.game import EuchreGame
from euchre.game_logic.trump_selection import TrumpSelectionManager
from euchre.models import Card, Player, Rank, Suit


def card(rank: Rank, suit: Suit) -> Card:
    """Build a card without repeating the constructor at every call."""
    return Card(rank, suit)


def seated(name: str, seat: int, dealer_seat: int, my_score: int = 0, opp_score: int = 0) -> HeuristicPlayer:
    """A heuristic player that knows the score and where the dealer sits."""
    player = HeuristicPlayer(name)
    player.begin_round(seat, dealer_seat, my_score, opp_score)
    return player


class TestHeuristicPlayer:
    """Bidding and card play from the rules, not from a trained model."""

    def test_extends_the_base_player(self) -> None:
        """The heuristic is a Player, created by the AI factory."""
        player = AIFactory.create_ai_player("Ada", "heuristic")
        assert isinstance(player, HeuristicPlayer)
        assert isinstance(player, Player)
        assert not isinstance(player, BaseAIInterface)

    def test_orders_up_both_bowers_and_passes_junk(self) -> None:
        """Both bowers make the contract. A hand with no trump does not."""
        strong = seated("Ada", seat=1, dealer_seat=0)
        strong.hand = [
            card(Rank.JACK, Suit.HEARTS),
            card(Rank.JACK, Suit.DIAMONDS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.TEN, Suit.CLUBS),
            card(Rank.NINE, Suit.SPADES),
        ]
        assert strong.should_order_up(card(Rank.QUEEN, Suit.HEARTS))

        weak = seated("Bea", seat=1, dealer_seat=0)
        weak.hand = [
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.TEN, Suit.CLUBS),
            card(Rank.NINE, Suit.SPADES),
            card(Rank.TEN, Suit.SPADES),
            card(Rank.QUEEN, Suit.CLUBS),
        ]
        assert not weak.should_order_up(card(Rank.ACE, Suit.HEARTS))

    def test_does_not_hand_an_opponent_the_right_bower(self) -> None:
        """Ordering the jack up gives that boss trump to the dealer."""
        player = seated("Ada", seat=1, dealer_seat=0)
        player.hand = [
            card(Rank.NINE, Suit.HEARTS),
            card(Rank.TEN, Suit.HEARTS),
            card(Rank.ACE, Suit.CLUBS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.NINE, Suit.SPADES),
        ]
        assert not player.should_order_up(card(Rank.JACK, Suit.HEARTS))

    def test_orders_up_when_partner_receives_the_right_bower(self) -> None:
        """The left bower plus an outside ace is a call when partner is dealt the right."""
        player = seated("Ada", seat=0, dealer_seat=2)
        player.hand = [
            card(Rank.JACK, Suit.DIAMONDS),
            card(Rank.ACE, Suit.CLUBS),
            card(Rank.KING, Suit.CLUBS),
            card(Rank.NINE, Suit.SPADES),
            card(Rank.TEN, Suit.SPADES),
        ]
        assert player.should_order_up(card(Rank.JACK, Suit.HEARTS))

    def test_lone_right_bower_passes_from_first_seat(self) -> None:
        """One boss trump is not enough tricks to accept a euchre at 0-0."""
        player = seated("Ada", seat=1, dealer_seat=0)
        player.hand = [
            card(Rank.JACK, Suit.HEARTS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.TEN, Suit.CLUBS),
            card(Rank.NINE, Suit.SPADES),
            card(Rank.TEN, Suit.SPADES),
        ]
        assert not player.should_order_up(card(Rank.ACE, Suit.HEARTS))

    def test_score_changes_whether_the_same_hand_calls(self) -> None:
        """At 9-0 a make wins and a euchre does not lose. At 9-8 a euchre loses the game."""
        cards = [
            card(Rank.JACK, Suit.HEARTS),
            card(Rank.ACE, Suit.CLUBS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.NINE, Suit.SPADES),
            card(Rank.TEN, Suit.SPADES),
        ]
        upcard = card(Rank.NINE, Suit.HEARTS)

        at_start = seated("Ada", 1, 0, 0, 0)
        at_start.hand = list(cards)
        ahead_and_safe = seated("Bea", 1, 0, 9, 0)
        ahead_and_safe.hand = list(cards)
        nine_all = seated("Cia", 1, 0, 9, 9)
        nine_all.hand = list(cards)
        euchre_loses = seated("Dee", 1, 0, 9, 8)
        euchre_loses.hand = list(cards)

        assert not at_start.should_order_up(upcard)
        assert ahead_and_safe.should_order_up(upcard)
        assert not euchre_loses.should_order_up(upcard)

        start_edge = at_start.evaluate_call(Suit.HEARTS, upcard, 1, "opponent")
        nine_edge = nine_all.evaluate_call(Suit.HEARTS, upcard, 1, "opponent")
        danger_edge = euchre_loses.evaluate_call(Suit.HEARTS, upcard, 1, "opponent")
        assert nine_edge.advantage > danger_edge.advantage
        assert start_edge.expected_tricks == nine_edge.expected_tricks

    def test_second_round_names_next_and_never_the_turned_suit(self) -> None:
        """Three diamond trump, with hearts turned down, is a call in diamonds."""
        player = seated("Ada", seat=1, dealer_seat=0)
        player.hand = [
            card(Rank.JACK, Suit.DIAMONDS),
            card(Rank.ACE, Suit.DIAMONDS),
            card(Rank.KING, Suit.DIAMONDS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.TEN, Suit.SPADES),
        ]
        named = player.choose_trump_suit(card(Rank.QUEEN, Suit.HEARTS))
        assert named == Suit.DIAMONDS

    def test_dealer_must_name_a_suit_and_a_weak_hand_may_pass(self) -> None:
        """Stick the dealer. Anyone else may pass a hand that cannot make three tricks."""
        junk = [
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.TEN, Suit.CLUBS),
            card(Rank.QUEEN, Suit.CLUBS),
            card(Rank.NINE, Suit.SPADES),
            card(Rank.TEN, Suit.SPADES),
        ]
        dealer = seated("Ada", seat=0, dealer_seat=0)
        dealer.hand = list(junk)
        named = dealer.choose_trump_suit(card(Rank.ACE, Suit.HEARTS))
        assert named is not None
        assert named != Suit.HEARTS

        passer = seated("Bea", seat=1, dealer_seat=0)
        passer.hand = list(junk)
        assert passer.choose_trump_suit(card(Rank.ACE, Suit.HEARTS)) is None

    def test_follows_suit_and_does_not_drop_the_card(self) -> None:
        """A led spade must be answered with a spade, and the hand stays intact."""
        player = seated("Ada", 0, 1)
        player.hand = [card(Rank.ACE, Suit.SPADES), card(Rank.ACE, Suit.HEARTS)]
        played = player.choose_card_to_play(Suit.SPADES, Suit.HEARTS)
        assert played.suit == Suit.SPADES
        assert len(player.hand) == 2
        assert played in player.hand

    def test_ducks_when_partner_is_winning(self) -> None:
        """The nine keeps the king for a later trick once partner's ace is high."""
        player = seated("Ada", seat=0, dealer_seat=1)
        player.note_trump_call(0, Suit.HEARTS)
        player.hand = [
            card(Rank.KING, Suit.SPADES),
            card(Rank.NINE, Suit.SPADES),
            card(Rank.ACE, Suit.HEARTS),
        ]
        player.note_card_played(2, card(Rank.ACE, Suit.SPADES))
        played = player.choose_card_to_play(Suit.SPADES, Suit.HEARTS)
        assert played.rank == Rank.NINE
        assert played.suit == Suit.SPADES

    def test_takes_the_trick_with_the_cheapest_winner(self) -> None:
        """The ace beats the king. The nine does not, so the ace is the card to spend."""
        player = seated("Ada", seat=0, dealer_seat=1)
        player.note_trump_call(1, Suit.HEARTS)
        player.hand = [card(Rank.ACE, Suit.SPADES), card(Rank.NINE, Suit.SPADES)]
        player.note_card_played(1, card(Rank.KING, Suit.SPADES))
        played = player.choose_card_to_play(Suit.SPADES, Suit.HEARTS)
        assert played.rank == Rank.ACE

    def test_does_not_trump_partners_ace(self) -> None:
        """A void is a discard when partner already holds the trick."""
        player = seated("Ada", seat=0, dealer_seat=1)
        player.note_trump_call(0, Suit.HEARTS)
        player.hand = [card(Rank.NINE, Suit.HEARTS), card(Rank.NINE, Suit.DIAMONDS)]
        player.note_card_played(2, card(Rank.ACE, Suit.CLUBS))
        played = player.choose_card_to_play(Suit.CLUBS, Suit.HEARTS)
        assert played.suit == Suit.DIAMONDS

    def test_trumps_an_opponents_winner_when_void(self) -> None:
        """The lowest trump beats a side ace once clubs cannot be followed."""
        player = seated("Ada", seat=0, dealer_seat=1)
        player.note_trump_call(1, Suit.HEARTS)
        player.hand = [card(Rank.NINE, Suit.HEARTS), card(Rank.TEN, Suit.DIAMONDS)]
        player.note_card_played(1, card(Rank.ACE, Suit.CLUBS))
        played = player.choose_card_to_play(Suit.CLUBS, Suit.HEARTS)
        assert played.suit == Suit.HEARTS
        assert played.rank == Rank.NINE

    def test_maker_leads_the_lower_boss_trump(self) -> None:
        """With both bowers, lead the left and keep the right bower."""
        player = seated("Ada", seat=0, dealer_seat=1)
        player.note_trump_call(0, Suit.HEARTS)
        player.hand = [
            card(Rank.JACK, Suit.HEARTS),
            card(Rank.JACK, Suit.DIAMONDS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.TEN, Suit.SPADES),
            card(Rank.NINE, Suit.SPADES),
        ]
        played = player.choose_card_to_play(None, Suit.HEARTS)
        assert played.rank == Rank.JACK
        assert played.suit == Suit.DIAMONDS

    def test_defender_cashes_an_off_ace(self) -> None:
        """Defenders lead an outside ace instead of opening a low trump."""
        player = seated("Ada", seat=0, dealer_seat=1)
        player.note_trump_call(1, Suit.HEARTS)
        player.hand = [
            card(Rank.NINE, Suit.HEARTS),
            card(Rank.ACE, Suit.CLUBS),
            card(Rank.TEN, Suit.SPADES),
            card(Rank.NINE, Suit.SPADES),
            card(Rank.TEN, Suit.DIAMONDS),
        ]
        played = player.choose_card_to_play(None, Suit.HEARTS)
        assert played.rank == Rank.ACE
        assert played.suit == Suit.CLUBS

    def test_dealer_discard_creates_a_void_and_keeps_trump(self) -> None:
        """Throw the bare nine. The bowers and the guarded clubs stay."""
        player = seated("Ada", seat=0, dealer_seat=0)
        player.hand = [
            card(Rank.JACK, Suit.HEARTS),
            card(Rank.ACE, Suit.HEARTS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.TEN, Suit.CLUBS),
            card(Rank.NINE, Suit.DIAMONDS),
            card(Rank.ACE, Suit.SPADES),
        ]
        thrown = player.choose_card_to_discard(Suit.HEARTS)
        assert thrown.suit == Suit.DIAMONDS
        assert thrown.rank == Rank.NINE
        assert len(player.hand) == 6

    def test_table_uses_the_heuristic_bid_and_discard(self) -> None:
        """The selection loop orders up for the partner who holds both bowers, and the dealer keeps that trump."""
        names = ["Ada", "Bea", "Cia", "Dee"]
        players = [HeuristicPlayer(name) for name in names]
        junk = [
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.TEN, Suit.SPADES),
            card(Rank.NINE, Suit.SPADES),
            card(Rank.TEN, Suit.DIAMONDS),
            card(Rank.QUEEN, Suit.CLUBS),
        ]
        for player in players:
            player.hand = list(junk)
        players[2].hand = [
            card(Rank.JACK, Suit.HEARTS),
            card(Rank.JACK, Suit.DIAMONDS),
            card(Rank.ACE, Suit.CLUBS),
            card(Rank.NINE, Suit.CLUBS),
            card(Rank.NINE, Suit.SPADES),
        ]
        upcard = card(Rank.QUEEN, Suit.HEARTS)
        suit, caller = TrumpSelectionManager().select_trump_suit(players, upcard, players[0])
        assert suit == Suit.HEARTS
        assert caller is players[2]

        game = EuchreGame(players=players, quiet_mode=True)
        game.game_state_manager.initialize_game(players)
        game.game_state_manager.set_dealer(players[0])
        game.round_number = 1
        dealer_before = list(players[0].hand)
        game._handle_dealer_pickup(players[0], upcard)
        assert len(players[0].hand) == 5
        assert upcard in players[0].hand
        assert any(card not in players[0].hand for card in dealer_before)

    def test_four_heuristics_finish_a_game(self) -> None:
        """A full game of four heuristic players reaches 10 without an illegal play."""
        game = EuchreGame(quiet_mode=True)
        for name in ["Ada", "Bea", "Cia", "Dee"]:
            game.add_ai_player(name, "heuristic")
        game.run_full_game()
        assert game.is_game_over()
        scores = game.game_state_manager.game_scores
        assert scores["Team 1"] >= 10 or scores["Team 2"] >= 10
