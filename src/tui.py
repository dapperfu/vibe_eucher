"""Text-based user interface for Euchre game."""

from typing import Dict, List, Optional, Tuple

from src.cards import Card, Suit
from src.players import Player


class TextTUI:
    """Text-based Terminal User Interface for the game."""

    def __init__(self) -> None:
        """Initialize the text UI."""
        self.players: Optional[List[Player]] = None
        self.current_trick_cards: List[Card] = []
        self.current_trick_player_ids: List[int] = []
        # Game log storage
        self.game_log: List[str] = []
        self.current_hand_log: List[str] = []
        self.initial_hands: Dict[int, List[Card]] = {}

    def start_new_hand(self) -> None:
        """Start logging a new hand."""
        self.current_hand_log = []
        self.initial_hands = {}

    def log_initial_hands(self, players: List[Player]) -> None:
        """
        Log the initial hands dealt to all players.

        Parameters
        ----------
        players : List[Player]
            List of all players with their initial hands.
        """
        self.current_hand_log.append("\n" + "=" * 70)
        self.current_hand_log.append("INITIAL DEAL")
        self.current_hand_log.append("=" * 70)
        for player in players:
            hand_str = ", ".join(str(card) for card in player.hand)
            self.current_hand_log.append(f"{player.name}: {hand_str}")
            self.initial_hands[player.player_id] = player.hand.copy()

    def log_turned_card(self, card: Card) -> None:
        """
        Log the turned up card.

        Parameters
        ----------
        card : Card
            The card that was turned up.
        """
        self.current_hand_log.append(f"\nTurned up card: {card}")

    def log_order_up_decision(self, player_name: str, decision: bool) -> None:
        """
        Log an order up decision.

        Parameters
        ----------
        player_name : str
            Name of the player making the decision.
        decision : bool
            True if ordered up, False if passed.
        """
        action = "Ordered up" if decision else "Passed"
        self.current_hand_log.append(f"{player_name}: {action}")

    def log_call_trump_decision(
        self, player_name: str, decision: Optional[Suit]
    ) -> None:
        """
        Log a call trump decision.

        Parameters
        ----------
        player_name : str
            Name of the player making the decision.
        decision : Optional[Suit]
            The suit called as trump, or None if passed.
        """
        if decision is None:
            self.current_hand_log.append(f"{player_name}: Passed")
        else:
            self.current_hand_log.append(f"{player_name}: Called {decision.value} as trump")

    def log_trump_selected(self, trump_suit: Suit, maker_name: str) -> None:
        """
        Log that trump was selected.

        Parameters
        ----------
        trump_suit : Suit
            The selected trump suit.
        maker_name : str
            Name of the player who made trump.
        """
        self.current_hand_log.append(f"\nTrump: {trump_suit.value} (made by {maker_name})")

    def log_trick(
        self,
        trick_num: int,
        played_cards: List[Card],
        player_ids: List[int],
        winner_id: int,
    ) -> None:
        """
        Log a trick with cards played and winner.

        Parameters
        ----------
        trick_num : int
            The trick number (1-5).
        played_cards : List[Card]
            Cards played in the trick.
        player_ids : List[int]
            Player IDs corresponding to each card.
        winner_id : int
            ID of the winning player.
        """
        if self.players is None:
            return

        self.current_hand_log.append(f"\nTrick {trick_num}:")
        for card, pid in zip(played_cards, player_ids):
            player_name = self.players[pid].name
            winner_marker = " *" if pid == winner_id else ""
            self.current_hand_log.append(f"  {player_name}: {card}{winner_marker}")

    def log_hand_score(
        self, tricks_won: List[int], scores: Tuple[int, int]
    ) -> None:
        """
        Log the hand score.

        Parameters
        ----------
        tricks_won : List[int]
            Tricks won by each team [team0, team1].
        scores : Tuple[int, int]
            Current scores (team0, team1).
        """
        self.current_hand_log.append(f"\nTricks won: Team 0: {tricks_won[0]}, Team 1: {tricks_won[1]}")
        self.current_hand_log.append(f"Scores: Team 0: {scores[0]}, Team 1: {scores[1]}")

    def display_game_log(self) -> None:
        """Display the complete game log."""
        print("\n" + "=" * 70)
        print("GAME LOG")
        print("=" * 70)
        for line in self.game_log:
            print(line)
        print("=" * 70)

    def display_hand_log(self) -> None:
        """Display the current hand log and add it to game log."""
        if self.current_hand_log:
            for line in self.current_hand_log:
                print(line)
                self.game_log.append(line)
            self.current_hand_log = []

    def set_players(self, players: List[Player]) -> None:
        """
        Set the list of all players for table display.

        Parameters
        ----------
        players : List[Player]
            List of all players in the game.
        """
        self.players = players

    def update_trick_state(
        self, played_cards: List[Card], player_ids: List[int]
    ) -> None:
        """
        Update the current trick state for table display.

        Parameters
        ----------
        played_cards : List[Card]
            Cards played in the current trick.
        player_ids : List[int]
            Player IDs corresponding to each card.
        """
        self.current_trick_cards = played_cards.copy()
        self.current_trick_player_ids = player_ids.copy()

    def display_hand(self, player: Player) -> None:
        """
        Display a player's hand.

        Parameters
        ----------
        player : Player
            The player whose hand to display.
        """
        print(f"\n{player.name}'s hand:")
        for i, card in enumerate(player.hand):
            print(f"  {i + 1}. {card}")

    def display_turned_card(self, card: Card) -> None:
        """
        Display the turned up card.

        Parameters
        ----------
        card : Card
            The turned card.
        """
        print(f"\nTurned card: {card}")

    def display_trump_selection(self, trump_suit: Optional[Suit]) -> None:
        """
        Display the selected trump suit.

        Parameters
        ----------
        trump_suit : Optional[Suit]
            The selected trump suit, or None.
        """
        if trump_suit is None:
            print("\nAll players passed. Redealing...")
        else:
            print(f"\nTrump suit: {trump_suit.value}")

    def display_trick(self, played_cards: List[Card], players: List[str]) -> None:
        """
        Display the current trick.

        Parameters
        ----------
        played_cards : List[Card]
            Cards played in the trick.
        players : List[str]
            Names of players who played each card.
        """
        print("\nCurrent trick:")
        for card, player_name in zip(played_cards, players):
            print(f"  {player_name}: {card}")

    def display_trick_winner(self, winner_name: str) -> None:
        """
        Display the trick winner.

        Parameters
        ----------
        winner_name : str
            Name of the winning player.
        """
        print(f"\n{winner_name} wins the trick!")

    def display_scores(self, team0_score: int, team1_score: int) -> None:
        """
        Display current scores.

        Parameters
        ----------
        team0_score : int
            Team 0's score.
        team1_score : int
            Team 1's score.
        """
        print(f"\nScores: Team 0: {team0_score}, Team 1: {team1_score}")

    def display_game_over(self, winning_team: int) -> None:
        """
        Display game over message.

        Parameters
        ----------
        winning_team : int
            The winning team ID (0 or 1).
        """
        print(f"\n{'=' * 50}")
        print(f"Game Over! Team {winning_team} wins!")
        print(f"{'=' * 50}")

    def get_order_up_decision(
        self, player: Player, turned_card: Card, dealer_id: int
    ) -> bool:
        """
        Get user input for ordering up.

        Parameters
        ----------
        player : HumanPlayer
            The human player making the decision.
        turned_card : Card
            The card that was turned up.
        dealer_id : int
            The dealer's ID.

        Returns
        -------
        bool
            True to order up, False to pass.
        """
        self.display_hand(player)
        print(f"\nTurned card: {turned_card}")
        print("Order up this card? (y/n): ", end="")
        response = input().strip().lower()
        return response == "y"

    def get_call_trump_decision(
        self, player: Player, turned_card: Card, must_choose: bool = False
    ) -> Optional[Suit]:
        """
        Get user input for calling trump.

        Parameters
        ----------
        player : HumanPlayer
            The human player making the decision.
        turned_card : Card
            The card that was turned up (cannot be chosen).

        Returns
        -------
        Optional[Suit]
            The suit to call as trump, or None to pass.
        """
        self.display_hand(player)
        print(f"\nTurned card: {turned_card} (cannot be chosen)")
        if must_choose:
            print("You MUST choose a suit (screw the dealer rule):")
        else:
            print("Call trump:")
        print("  1. Hearts")
        print("  2. Diamonds")
        print("  3. Clubs")
        print("  4. Spades")
        if not must_choose:
            print("  5. Pass")
        print("Choice (1-{}): ".format("4" if must_choose else "5"), end="")

        choice = input().strip()
        suit_map = {
            "1": Suit.HEARTS,
            "2": Suit.DIAMONDS,
            "3": Suit.CLUBS,
            "4": Suit.SPADES,
        }

        if not must_choose and choice == "5":
            return None
        if choice in suit_map:
            suit = suit_map[choice]
            if suit == turned_card.suit:
                if must_choose:
                    print("Cannot choose the turned card's suit. Choose another.")
                    # Recursively ask again if must choose
                    return self.get_call_trump_decision(player, turned_card, must_choose)
                print("Cannot choose the turned card's suit. Passing.")
                return None
            return suit

        if must_choose:
            print("Invalid choice. You must choose a suit.")
            return self.get_call_trump_decision(player, turned_card, must_choose)
        print("Invalid choice. Passing.")
        return None

    def get_discard_decision(self, player: Player) -> Card:
        """
        Get user input for discarding a card.

        Parameters
        ----------
        player : HumanPlayer
            The dealer player.

        Returns
        -------
        Card
            The card to discard.
        """
        self.display_hand(player)
        print("\nChoose a card to discard (1-6): ", end="")
        choice = input().strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(player.hand):
                return player.hand[idx]
            raise ValueError("Invalid index")
        except (ValueError, IndexError):
            print("Invalid choice. Discarding first card.")
            return player.hand[0]

    def display_table(
        self,
        current_player: Player,
        led_suit: Optional[Suit] = None,
        trump_suit: Optional[Suit] = None,
    ) -> None:
        """
        Display the table layout with players and cards in the middle.

        Parameters
        ----------
        current_player : Player
            The current human player viewing the table.
        led_suit : Optional[Suit]
            The suit that was led, if any.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        """
        if self.players is None:
            # Fallback if players not set
            return

        current_id = current_player.player_id
        partner_id = (current_id + 2) % 4
        left_opponent_id = (current_id + 1) % 4
        right_opponent_id = (current_id + 3) % 4

        partner = self.players[partner_id]
        left_opponent = self.players[left_opponent_id]
        right_opponent = self.players[right_opponent_id]

        # Create a mapping of player_id to card for this trick
        card_map: dict[int, Optional[Card]] = {
            current_id: None,
            partner_id: None,
            left_opponent_id: None,
            right_opponent_id: None,
        }

        # Use stored trick state if available
        played_cards = self.current_trick_cards
        player_ids = self.current_trick_player_ids

        for card, pid in zip(played_cards, player_ids):
            card_map[pid] = card

        # Build the table display
        print("\n" + "=" * 70)
        print("TABLE VIEW")
        print("=" * 70)

        # Top: Partner (across from you)
        partner_card = card_map[partner_id]
        partner_display = f"{partner.name} (Partner)"
        if partner_card:
            partner_display += f" - {partner_card}"
        print(f"\n{partner_display:^70}")

        # Middle section: Cards in the center, opponents on sides
        left_card = card_map[left_opponent_id]
        right_card = card_map[right_opponent_id]
        current_card = card_map[current_id]

        # Format cards in the middle (center of table)
        if played_cards:
            # Show cards prominently in the center
            card_strings = []
            for card, pid in zip(played_cards, player_ids):
                player_name = self.players[pid].name
                card_strings.append(f"  {player_name}: {card}")
            cards_display = "\n".join(card_strings)
        else:
            cards_display = "  [No cards played yet]"

        # Left opponent
        left_display = f"{left_opponent.name}"
        if left_card:
            left_display += f"\n{left_card}"

        # Right opponent
        right_display = f"{right_opponent.name}"
        if right_card:
            right_display += f"\n{right_card}"

        # Print middle row: Left | Cards (center) | Right
        # Split into lines for proper alignment
        left_lines = left_display.split("\n")
        right_lines = right_display.split("\n")
        card_lines = cards_display.split("\n")

        max_lines = max(len(left_lines), len(card_lines), len(right_lines))
        for i in range(max_lines):
            left_part = left_lines[i] if i < len(left_lines) else ""
            center_part = card_lines[i] if i < len(card_lines) else ""
            right_part = right_lines[i] if i < len(right_lines) else ""
            print(f"{left_part:<25} {center_part:^25} {right_part:>25}")

        # Bottom: Current player (You)
        you_display = f"{current_player.name} (You)"
        if current_card:
            you_display += f" - {current_card}"
        print(f"\n{you_display:^70}")

        # Display game info
        info_parts = []
        if led_suit:
            info_parts.append(f"Led: {led_suit.value}")
        if trump_suit:
            info_parts.append(f"Trump: {trump_suit.value}")
        if info_parts:
            print(f"\n{' | '.join(info_parts):^70}")

        print("=" * 70)

    def get_play_card_decision(
        self,
        player: Player,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
    ) -> Card:
        """
        Get user input for playing a card.

        Parameters
        ----------
        player : HumanPlayer
            The human player.
        led_suit : Optional[Suit]
            The suit that was led, if any.
        trump_suit : Optional[Suit]
            The current trump suit, if any.
        trick_cards : List[Card]
            Cards already played in the trick.

        Returns
        -------
        Card
            The card to play.
        """
        # Display table view (uses stored trick state)
        if self.players is not None:
            self.display_table(player, led_suit, trump_suit)

        # Display hand
        self.display_hand(player)

        print("\nChoose a card to play (1-{}): ".format(len(player.hand)), end="")
        choice = input().strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(player.hand):
                return player.hand[idx]
            raise ValueError("Invalid index")
        except (ValueError, IndexError):
            print("Invalid choice. Playing first card.")
            return player.hand[0]

