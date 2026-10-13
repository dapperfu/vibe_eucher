"""Text-based user interface for Euchre game."""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from eucher.cards import Card, Suit
from eucher.players import Player

try:
    from rich.table import Table
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if TYPE_CHECKING:
    from eucher.assistant import AssistantHelper
    from eucher.decision_playback import DecisionPlayback


class TextTUI:
    """Text-based Terminal User Interface for the game."""

    def __init__(
        self,
        assistant_helpers: Optional[List["AssistantHelper"]] = None,
        xray_mode: bool = False,
        decision_playback: Optional["DecisionPlayback"] = None,
    ) -> None:
        """
        Initialize the text UI.

        Parameters
        ----------
        assistant_helpers : Optional[List[AssistantHelper]]
            Optional list of assistant helpers for displaying bot recommendations.
            If None or empty, no assistant recommendations will be shown.
        xray_mode : bool
            If True, display all hidden information (all players' hands, kitty cards, etc.).
            Default is False.
        decision_playback : Optional[DecisionPlayback]
            Optional decision playback instance for recording/replaying decisions.
        """
        self.players: Optional[List[Player]] = None
        self.current_trick_cards: List[Card] = []
        self.current_trick_player_ids: List[int] = []
        # Game log storage
        self.game_log: List[str] = []
        self.current_hand_log: List[str] = []
        self.last_hand_log: List[str] = []  # Store last hand log for AI summaries
        self.initial_hands: Dict[int, List[Card]] = {}
        # Game state
        self.team_scores: Tuple[int, int] = (0, 0)
        self.current_trick_number: int = 0
        self.current_hand_number: int = 0
        self.dealer_id: Optional[int] = None
        self.trump_suit: Optional[Suit] = None
        # Trick tracking
        self.current_tricks_won: Tuple[int, int] = (0, 0)  # Team 0, Team 1
        self.hand_tricks: List[Dict] = []  # Store trick data for summary display
        self.making_team: Optional[int] = None  # Team that made trump
        self.final_tricks_won: Tuple[int, int] = (0, 0)  # Final tricks won for hand
        # Kitty tracking
        self.turned_card: Optional[Card] = None  # Turned card
        self.discarded_card: Optional[Card] = None  # Card discarded by dealer (if ordered up)
        self.kitty_cards: List[Card] = []  # The 3 remaining cards in deck
        self.trump_ordered_up: bool = False  # Whether trump was ordered up
        # Assistant helpers (list for multiple assistants)
        self.assistant_helpers = assistant_helpers if assistant_helpers else []
        # Backward compatibility: support single assistant_helper for old code
        self.assistant_helper = assistant_helpers[0] if assistant_helpers and len(assistant_helpers) == 1 else None
        # Xray mode
        self.xray_mode: bool = xray_mode
        # Decision playback
        self.decision_playback: Optional["DecisionPlayback"] = decision_playback
        # Going alone decision (set when ordering up with 'a' option)
        self.going_alone_decision: Optional[bool] = None

    def start_new_hand(self) -> None:
        """Start logging a new hand."""
        self.current_hand_log = []
        self.initial_hands = {}
        self.trump_suit = None
        self.current_hand_number += 1
        self.current_trick_number = 0  # Reset trick number for new hand
        self.current_tricks_won = (0, 0)  # Reset trick counts for new hand
        self.hand_tricks = []  # Reset trick tracking for new hand
        self.making_team = None  # Reset making team
        self.final_tricks_won = (0, 0)  # Reset final tricks won
        self.turned_card = None  # Reset turned card
        self.discarded_card = None  # Reset discarded card
        self.kitty_cards = []  # Reset kitty cards
        self.trump_ordered_up = False  # Reset trump ordered up flag
        self.going_alone_decision = None  # Reset going alone decision

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
            hand_str = "  ".join(repr(card) for card in player.hand)
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
        self.turned_card = card
        self.current_hand_log.append(f"\nTurned up card: {repr(card)}")
    
    def set_kitty_cards(self, kitty_cards: List[Card]) -> None:
        """
        Set the kitty cards (3 remaining cards in deck).

        Parameters
        ----------
        kitty_cards : List[Card]
            The 3 remaining cards in the deck.
        """
        self.kitty_cards = kitty_cards.copy()
    
    def set_discarded_card(self, discarded_card: Card) -> None:
        """
        Set the card that was discarded by the dealer.

        Parameters
        ----------
        discarded_card : Card
            The card that was discarded.
        """
        self.discarded_card = discarded_card
        self.trump_ordered_up = True

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
        self.trump_suit = trump_suit
        self.current_hand_log.append(f"\nTrump: {trump_suit.value} (made by {maker_name})")

    def log_trade_in_decision(self, player_name: str, decision: bool, eligible_cards: List[Card]) -> None:
        """
        Log a trade-in decision.

        Parameters
        ----------
        player_name : str
            Name of the player making the decision.
        decision : bool
            True if trading in, False if passing.
        eligible_cards : List[Card]
            The three cards eligible for trade-in.
        """
        cards_str = ", ".join(repr(card) for card in eligible_cards)
        action = "Trading in" if decision else "Passing on trade-in"
        self.current_hand_log.append(f"{player_name}: {action} ({cards_str})")

    def log_trade_in(self, player_name: str, trash_cards: List[Card], kitty_cards: List[Card]) -> None:
        """
        Log that a trade-in occurred.

        Parameters
        ----------
        player_name : str
            Name of the player who traded in.
        trash_cards : List[Card]
            The three cards that were discarded.
        kitty_cards : List[Card]
            The three cards received from the kitty.
        """
        trash_str = ", ".join(repr(card) for card in trash_cards)
        kitty_str = ", ".join(repr(card) for card in kitty_cards)
        self.current_hand_log.append(
            f"\n{player_name} traded in: {trash_str} -> received: {kitty_str}"
        )

    def log_trump_decision_history(self, order_up: list, call_trump: list) -> None:
        """
        Append a concise summary of all order-up and call-trump decisions for the hand.

        Parameters
        ----------
        order_up : list
            List of tuples `(player_name, bool)` representing order-up decisions.
        call_trump : list
            List of tuples `(player_name, Optional[Suit])` representing call-trump decisions.
        """
        if order_up:
            self.current_hand_log.append("\nOrder Up Decisions:")
            for name, decision in order_up:
                action = "Ordered up" if decision else "Passed"
                self.current_hand_log.append(f"  {name}: {action}")

        if call_trump:
            self.current_hand_log.append("\nCall Trump Decisions:")
            for name, decision in call_trump:
                if decision is None:
                    self.current_hand_log.append(f"  {name}: Passed")
                else:
                    self.current_hand_log.append(f"  {name}: Called {decision.value} as trump")

    def display_trump_decision_summary(
        self,
        order_up: list,
        call_trump: list,
        trump_suit: Optional[Suit],
        maker_name: Optional[str],
        turned_card: Optional["Card"] = None,
    ) -> None:
        """
        Display a summary of all trump decisions before gameplay starts.

        Parameters
        ----------
        order_up : list
            List of tuples `(player_name, bool)` representing order-up decisions.
        call_trump : list
            List of tuples `(player_name, Optional[Suit])` representing call-trump decisions.
        trump_suit : Optional[Suit]
            The selected trump suit.
        maker_name : Optional[str]
            Name of the player who made trump.
        turned_card : Optional[Card]
            The card that was turned up (for order-up phase).
        """
        self._clear_screen()
        
        # Get a player for the header (use first player if available)
        if self.players and len(self.players) > 0:
            self._display_gameboard_header(self.players[0])
        else:
            print("\n" + "=" * 70)
        
        print("\nTrump Selection Summary:")
        print("=" * 70)
        
        if order_up:
            print("\nOrder Up Decisions:")
            if turned_card is not None:
                print(f"  Turned up card: {repr(turned_card)}")
            for name, decision in order_up:
                action = "Ordered up" if decision else "Passed"
                print(f"  {name}: {action}")

        if call_trump:
            print("\nCall Trump Decisions:")
            for name, decision in call_trump:
                if decision is None:
                    print(f"  {name}: Passed")
                else:
                    print(f"  {name}: Called {decision.value} as trump")
        
        if trump_suit is not None:
            print(f"\nTrump: {trump_suit.value} {trump_suit.unicode_symbol()}")
            if maker_name:
                print(f"Made by: {maker_name}")
        
        # In xray mode, show kitty cards during trump selection
        if self.xray_mode and self.kitty_cards:
            print("\n" + "=" * 70)
            kitty_str = self._format_cards_xray(self.kitty_cards)
            print(f"[XRAY] Kitty cards: {kitty_str}")
            if self.discarded_card is not None:
                discarded_str = self._format_card_xray(self.discarded_card)
                print(f"[XRAY] Discarded card: {discarded_str}")
        
        print("\n" + "=" * 70)
        print("Press spacebar to start hand...")
        self._wait_for_spacebar()

    def log_trick(
        self,
        trick_num: int,
        played_cards: List[Card],
        player_ids: List[int],
        winner_id: int,
        renege_card: Optional[Card] = None,
        renege_player_id: Optional[int] = None,
        renege_team: Optional[int] = None,
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
        renege_card : Optional[Card]
            Card that was reneged, if any.
        renege_player_id : Optional[int]
            ID of player who reneged, if any.
        renege_team : Optional[int]
            Team of player who reneged, if any.
        """
        if self.players is None:
            return

        # Store trick data for summary display
        trick_data = {
            "trick_num": trick_num,
            "played_cards": played_cards.copy(),
            "player_ids": player_ids.copy(),
            "winner_id": winner_id,
            "renege_card": renege_card,
            "renege_player_id": renege_player_id,
            "renege_team": renege_team,
        }
        self.hand_tricks.append(trick_data)
    
    def update_trick_renege(
        self,
        trick_num: int,
        renege_card: Card,
        renege_player_id: int,
        renege_team: int,
    ) -> None:
        """
        Update a trick with renege information detected after hand completes.

        Parameters
        ----------
        trick_num : int
            The trick number (1-5) where renege occurred.
        renege_card : Card
            Card that was reneged.
        renege_player_id : int
            ID of player who reneged.
        renege_team : int
            Team of player who reneged.
        """
        # Find the trick in hand_tricks and update it
        for trick_data in self.hand_tricks:
            if trick_data["trick_num"] == trick_num:
                trick_data["renege_card"] = renege_card
                trick_data["renege_player_id"] = renege_player_id
                trick_data["renege_team"] = renege_team
                break
        
        # Mark the last trick (trick 5) as where the renege was discovered
        # (since reneges are discovered after the hand completes)
        for trick_data in self.hand_tricks:
            if trick_data["trick_num"] == 5:
                trick_data["renege_discovered"] = True
                break

        # Update hand log with renege information
        renege_trick_data = None
        for trick_data in self.hand_tricks:
            if trick_data["trick_num"] == trick_num:
                renege_trick_data = trick_data
                break
        
        if renege_trick_data:
            played_cards = renege_trick_data["played_cards"]
            player_ids = renege_trick_data["player_ids"]
            winner_id = renege_trick_data["winner_id"]
            self.current_hand_log.append(f"\nTrick {trick_num}:")
            for card, pid in zip(played_cards, player_ids):
                player_name = self.players[pid].name if pid < len(self.players) else f"Player {pid}"
                winner_marker = " *" if pid == winner_id else ""
                renege_marker = " [RENEGE]" if card == renege_card and pid == renege_player_id else ""
                self.current_hand_log.append(f"  {player_name}: {repr(card)}{winner_marker}{renege_marker}")

    def log_hand_score(
        self, tricks_won: List[int], scores: Tuple[int, int], making_team: Optional[int] = None
    ) -> None:
        """
        Log the hand score.

        Parameters
        ----------
        tricks_won : List[int]
            Tricks won by each team [team0, team1].
        scores : Tuple[int, int]
            Current scores (team0, team1).
        making_team : Optional[int]
            Team that made trump (0 or 1), if any.
        """
        self.final_tricks_won = (tricks_won[0], tricks_won[1])
        self.making_team = making_team
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

    def get_hand_log(self) -> List[str]:
        """
        Get a copy of the current hand log without clearing it.

        Returns
        -------
        List[str]
            Copy of the current hand log.
        """
        return self.current_hand_log.copy()

    def display_trick_summary(self) -> None:
        """
        Display a tabular summary of all tricks in the hand.
        
        Shows columns for each player, rows for each trick, with cards played and winner.
        """
        if not self.hand_tricks or self.players is None:
            return
        
        if RICH_AVAILABLE:
            # Use Rich table for proper alignment
            console = Console()
            table = Table(title="TRICK SUMMARY", show_header=True, header_style="bold")
            
            # Add columns: Trick number, then each player, then Winner
            table.add_column("Trick", justify="center", style="cyan", no_wrap=True)
            for player in self.players:
                table.add_column(player.name, justify="center", no_wrap=True)
            table.add_column("Winner", justify="center", style="bold red", no_wrap=True)
            
            # Track tricks won by each team to determine euchre point
            tricks_won_by_team = [0, 0]  # Team 0, Team 1
            euchre_trick_num: Optional[int] = None
            
            # Add rows for each trick
            for trick_idx, trick_data in enumerate(self.hand_tricks):
                trick_num = trick_data["trick_num"]
                played_cards = trick_data["played_cards"]
                player_ids = trick_data["player_ids"]
                winner_id = trick_data["winner_id"]
                renege_card = trick_data.get("renege_card")
                renege_player_id = trick_data.get("renege_player_id")
                
                # Update tricks won
                winner_team = self.players[winner_id].team if winner_id < len(self.players) else None
                if winner_team is not None:
                    tricks_won_by_team[winner_team] += 1
                
                # Check if making team was euchred at this trick
                if self.making_team is not None and euchre_trick_num is None:
                    making_tricks = tricks_won_by_team[self.making_team]
                    remaining_tricks = 5 - trick_num
                    # Euchred if: making team can't reach 3 tricks even if they win all remaining
                    if making_tricks + remaining_tricks < 3:
                        euchre_trick_num = trick_num
                
                # Find lead card (first card played)
                lead_player_id = player_ids[0] if player_ids else None
                
                # Create a mapping of player_id -> card for this trick
                # Use repr() to get compact format with suit emojis (e.g., "K♥" instead of "K of Hearts")
                trick_cards: Dict[int, str] = {}
                for card, pid in zip(played_cards, player_ids):
                    trick_cards[pid] = repr(card)
                
                # Build row: trick number, then each player's card, then winner
                trick_num_str = str(trick_num)
                renege_discovered = trick_data.get("renege_discovered", False)
                if euchre_trick_num == trick_num:
                    trick_num_str = f"[bold red]{trick_num} [EUCHRED][/bold red]"
                elif renege_discovered:
                    trick_num_str = f"[bold yellow]{trick_num} [RENEGE DISCOVERED][/bold yellow]"
                row_data = [trick_num_str]
                for i in range(len(self.players)):
                    card_str = trick_cards.get(i, "-")
                    if card_str == "-":
                        row_data.append("-")
                        continue
                    
                    # Build markers first
                    markers = []
                    
                    # Check if this is the lead card (first card played)
                    is_lead = (i == lead_player_id)
                    if is_lead:
                        markers.append("🎯")  # Lead card marker (target)
                    
                    # Check if this is the winning card
                    is_winner = (i == winner_id)
                    if is_winner:
                        markers.append("🏆")  # Winner marker (trophy)
                    
                    # Check for renege
                    is_renege = (renege_card is not None and renege_player_id == i)
                    if is_renege:
                        markers.append("[RENEGE]")
                    
                    # Build formatted card with markers
                    if is_winner:
                        # Winner gets red color, then markers
                        formatted_card = f"[red]{card_str}[/red]" + "".join(markers)
                    elif is_renege:
                        # Renege gets bold red, then markers
                        formatted_card = f"[bold red]{card_str}[/bold red]" + "".join(markers)
                    else:
                        # Regular card with markers
                        formatted_card = card_str + "".join(markers)
                    
                    row_data.append(formatted_card)
                
                # Add winner name
                winner_name = self.players[winner_id].name if winner_id < len(self.players) else f"Player {winner_id}"
                row_data.append(f"[red]{winner_name}[/red]")
                
                table.add_row(*row_data)
            
            console.print("\n")
            console.print(table)
            
            # Display if a team was set (euchred)
            if self.making_team is not None:
                making_tricks = self.final_tricks_won[self.making_team]
                if making_tricks < 3:
                    # Calling team was set (euchred)
                    defending_team = 1 - self.making_team
                    console.print(f"[bold red]Team {self.making_team} was SET (euchred)[/bold red] - Team {defending_team} gets 2 points")
                elif making_tricks == 5:
                    # Calling team swept
                    console.print(f"[bold green]Team {self.making_team} SWEPT (5 tricks)[/bold green]")
            
            # Display kitty
            kitty_display = []
            if self.kitty_cards:
                kitty_display.extend(self.kitty_cards)
            if self.trump_ordered_up and self.discarded_card is not None:
                # If ordered up, show discarded card
                kitty_display.append(self.discarded_card)
            elif self.turned_card is not None:
                # If not ordered up, show turned card
                kitty_display.append(self.turned_card)
            
            if kitty_display:
                kitty_str = ", ".join(str(card) for card in kitty_display)
                # Mark which card was discarded vs turned down
                if self.trump_ordered_up and self.discarded_card is not None:
                    # Find discarded card in the list and mark it
                    kitty_parts = []
                    for card in kitty_display:
                        if card == self.discarded_card:
                            kitty_parts.append(f"[bold red]{card} (discarded)[/bold red]")
                        else:
                            kitty_parts.append(str(card))
                    kitty_str = ", ".join(kitty_parts)
                elif self.turned_card is not None and not self.trump_ordered_up:
                    # Mark turned down card
                    kitty_parts = []
                    for card in kitty_display:
                        if card == self.turned_card:
                            kitty_parts.append(f"[dim]{card} (turned down)[/dim]")
                        else:
                            kitty_parts.append(str(card))
                    kitty_str = ", ".join(kitty_parts)
                
                console.print(f"\n[bold]Kitty:[/bold] {kitty_str}")
            
            console.print()
        else:
            # Fallback to simple text format if Rich is not available
            print("\n" + "=" * 80)
            print("TRICK SUMMARY")
            print("=" * 80)
            
            # Get player names
            player_names = [p.name for p in self.players]
            num_players = len(player_names)
            
            # Collect all card strings to calculate proper column widths
            # Use repr() to get compact format with suit emojis
            all_trick_cards: List[Dict[int, str]] = []
            for trick_data in self.hand_tricks:
                trick_cards: Dict[int, str] = {}
                for card, pid in zip(trick_data["played_cards"], trick_data["player_ids"]):
                    trick_cards[pid] = repr(card)
                all_trick_cards.append(trick_cards)
            
            # Calculate column widths: max of player name length and longest card string for that column
            # Account for markers: 🎯 (lead) and 🏆 (winner) - emojis are typically 2 display characters each
            col_widths = []
            for i in range(num_players):
                max_card_len = max(
                    (len(all_trick_cards[trick_idx].get(i, "-")) for trick_idx in range(len(all_trick_cards))),
                    default=0
                )
                # Add 4 for potential emoji markers (🎯 = 2 chars, 🏆 = 2 chars)
                col_widths.append(max(len(player_names[i]), max_card_len + 4, 8))
            
            # Calculate trick column width - account for markers
            trick_col_width = 8
            if self.making_team is not None:
                making_tricks = self.final_tricks_won[self.making_team]
                if making_tricks < 3:
                    # Account for "[EUCHRED]" marker (11 chars)
                    trick_col_width = max(trick_col_width, len("5 [EUCHRED]"))
            # Check if any trick has renege discovered marker
            for trick_data in self.hand_tricks:
                if trick_data.get("renege_discovered", False):
                    # Account for "[RENEGE DISCOVERED]" marker (20 chars)
                    trick_col_width = max(trick_col_width, len("5 [RENEGE DISCOVERED]"))
                    break
            winner_col_width = max(len(name) for name in player_names) + 5  # Extra space for winner column
            
            # Print header with proper alignment (matching data row format)
            header_parts = [f"{'Trick':<{trick_col_width}}"]
            for i, name in enumerate(player_names):
                header_parts.append(f"{name:<{col_widths[i]}}")
            header_parts.append(f"{'Winner':<{winner_col_width}}")
            header = " ".join(header_parts)
            print(header)
            # Calculate separator length: sum of column widths + spaces between columns
            separator_len = trick_col_width + sum(col_widths) + winner_col_width + (num_players + 1)
            print("-" * separator_len)
            
            # Track tricks won by each team to determine euchre point
            tricks_won_by_team = [0, 0]  # Team 0, Team 1
            euchre_trick_num: Optional[int] = None
            
            # Print each trick
            for trick_idx, trick_data in enumerate(self.hand_tricks):
                trick_num = trick_data["trick_num"]
                played_cards = trick_data["played_cards"]
                player_ids = trick_data["player_ids"]
                winner_id = trick_data["winner_id"]
                renege_card = trick_data.get("renege_card")
                renege_player_id = trick_data.get("renege_player_id")
                
                # Update tricks won
                winner_team = self.players[winner_id].team if winner_id < len(self.players) else None
                if winner_team is not None:
                    tricks_won_by_team[winner_team] += 1
                
                # Check if making team was euchred at this trick
                if self.making_team is not None and euchre_trick_num is None:
                    making_tricks = tricks_won_by_team[self.making_team]
                    remaining_tricks = 5 - trick_num
                    # Euchred if: making team can't reach 3 tricks even if they win all remaining
                    if making_tricks + remaining_tricks < 3:
                        euchre_trick_num = trick_num
                
                # Find lead card (first card played)
                lead_player_id = player_ids[0] if player_ids else None
                
                # Create a mapping of player_id -> card for this trick
                # Use repr() to get compact format with suit emojis (e.g., "K♥" instead of "K of Hearts")
                trick_cards: Dict[int, str] = {}
                for card, pid in zip(played_cards, player_ids):
                    trick_cards[pid] = repr(card)
                
                # Build row with proper alignment
                # Format trick number
                renege_discovered = trick_data.get("renege_discovered", False)
                if euchre_trick_num == trick_num:
                    trick_num_str = self._red_text(f"{trick_num} [EUCHRED]")
                    # Pad to account for ANSI escape codes (they don't count toward display width)
                    # "[EUCHRED]" is 11 chars, so total is len(str(trick_num)) + 11
                    padding_needed = trick_col_width - (len(str(trick_num)) + 11)
                    trick_num_str = trick_num_str + " " * max(0, padding_needed)
                elif renege_discovered:
                    # Mark trick where renege was discovered (after hand completes)
                    trick_num_str = f"\033[33m{trick_num} [RENEGE DISCOVERED]\033[0m"
                    # Pad to account for ANSI escape codes
                    # "[RENEGE DISCOVERED]" is 20 chars
                    padding_needed = trick_col_width - (len(str(trick_num)) + 20)
                    trick_num_str = trick_num_str + " " * max(0, padding_needed)
                else:
                    trick_num_str = f"{trick_num:<{trick_col_width}}"
                row_parts = [trick_num_str]
                
                # Format each player's card column
                for i in range(num_players):
                    card_str = trick_cards.get(i, "-")
                    if card_str == "-":
                        padding_needed = col_widths[i] - 1
                        row_parts.append("-" + " " * padding_needed)
                        continue
                    
                    # Build markers
                    markers = []
                    
                    # Check if this is the lead card (first card played)
                    if i == lead_player_id:
                        markers.append("🎯")  # Lead card marker (target)
                    
                    # Check if this is the winning card
                    if i == winner_id:
                        markers.append("🏆")  # Winner marker (trophy)
                    
                    # Check for renege
                    if renege_card is not None and renege_player_id == i:
                        markers.append(" [RENEGE]")
                    
                    # Combine card with markers
                    display_str = card_str + "".join(markers)
                    
                    # Calculate padding needed (account for emoji width - emojis are typically 2 display chars)
                    emoji_count = len([m for m in markers if m in ("🎯", "🏆")])
                    display_width = len(card_str) + emoji_count * 2 + len("".join([m for m in markers if m not in ("🎯", "🏆")]))
                    padding_needed = col_widths[i] - display_width
                    
                    # Apply formatting based on card role
                    if i == winner_id:
                        formatted_card = self._red_text(display_str) + " " * max(0, padding_needed)
                    elif renege_card is not None and renege_player_id == i:
                        formatted_card = self._red_text(display_str) + " " * max(0, padding_needed)
                    else:
                        formatted_card = display_str + " " * max(0, padding_needed)
                    row_parts.append(formatted_card)
                
                # Format winner column
                winner_name = self.players[winner_id].name if winner_id < len(self.players) else f"Player {winner_id}"
                winner_padding = winner_col_width - len(winner_name)
                formatted_winner = self._red_text(winner_name) + " " * winner_padding
                row_parts.append(formatted_winner)
                
                # Join with single space between columns
                print(" ".join(row_parts))
            
            # Display if a team was set (euchred)
            if self.making_team is not None:
                making_tricks = self.final_tricks_won[self.making_team]
                if making_tricks < 3:
                    # Calling team was set (euchred)
                    defending_team = 1 - self.making_team
                    print(f"\n{self._red_text('Team ' + str(self.making_team) + ' was SET (euchred)')} - Team {defending_team} gets 2 points")
                elif making_tricks == 5:
                    # Calling team swept
                    print(f"\nTeam {self.making_team} SWEPT (5 tricks)")
            
            # Display kitty
            kitty_display = []
            if self.kitty_cards:
                kitty_display.extend(self.kitty_cards)
            if self.trump_ordered_up and self.discarded_card is not None:
                # If ordered up, show discarded card
                kitty_display.append(self.discarded_card)
            elif self.turned_card is not None:
                # If not ordered up, show turned card
                kitty_display.append(self.turned_card)
            
            if kitty_display:
                kitty_parts = []
                for card in kitty_display:
                    if self.trump_ordered_up and self.discarded_card is not None and card == self.discarded_card:
                        kitty_parts.append(f"{self._red_text(str(card) + ' (discarded)')}")
                    elif self.turned_card is not None and not self.trump_ordered_up and card == self.turned_card:
                        kitty_parts.append(f"{str(card)} (turned down)")
                    else:
                        kitty_parts.append(str(card))
                kitty_str = ", ".join(kitty_parts)
                print(f"\nKitty: {kitty_str}")
            
            print("=" * 80)
    
    def display_hand_log(self) -> List[str]:
        """
        Display the current hand log and add it to game log.

        Returns
        -------
        List[str]
            Copy of the hand log before it was cleared.
        """
        hand_log_copy = self.current_hand_log.copy()
        if self.current_hand_log:
            for line in self.current_hand_log:
                print(line)
                self.game_log.append(line)
            # Store last hand log before clearing
            self.last_hand_log = hand_log_copy.copy()
            self.current_hand_log = []
        
        # Display trick summary after hand log
        self.display_trick_summary()
        
        # Pause so user can see the summary before continuing
        if self.hand_tricks:
            print("\nPress spacebar to continue...")
            self._wait_for_spacebar()
        
        return hand_log_copy

    def set_players(self, players: List[Player], dealer_id: Optional[int] = None) -> None:
        """
        Set the list of all players for table display.

        Parameters
        ----------
        players : List[Player]
            List of all players in the game.
        dealer_id : Optional[int]
            The dealer's player ID, if available.
        """
        self.players = players
        self.dealer_id = dealer_id

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
        # Sort hand like humans would: by suit, then by rank
        sorted_hand = self._sort_hand_human_style(player.hand)
        
        print(f"\n{player.name}'s hand ({len(player.hand)} cards):")
        for i, card in enumerate(sorted_hand):
            print(f"  {i + 1}. {card}")
    
    def _sort_hand_human_style(self, hand: List[Card]) -> List[Card]:
        """
        Sort a hand like humans would: cluster by suit, then by rank within each suit.
        
        Parameters
        ----------
        hand : List[Card]
            The hand to sort.
        
        Returns
        -------
        List[Card]
            The sorted hand.
        """
        # Sort by suit first, then by rank value within each suit
        return sorted(hand, key=self._get_card_sort_key)
    
    def _get_card_sort_key(self, card: Card) -> tuple[int, int]:
        """
        Get the sort key for a card (suit order, then rank value).
        
        Parameters
        ----------
        card : Card
            The card to get the sort key for.
        
        Returns
        -------
        tuple[int, int]
            Tuple of (suit_order, rank_value) for sorting.
        """
        # Define suit order (standard order)
        suit_order = {
            Suit.HEARTS: 0,
            Suit.DIAMONDS: 1,
            Suit.CLUBS: 2,
            Suit.SPADES: 3,
        }
        
        return (suit_order[card.suit], card.rank.value)
    
    def _clear_screen(self) -> None:
        """
        Clear the terminal screen.
        
        Uses ANSI escape codes for cross-platform compatibility.
        """
        print("\033[2J\033[H", end="")
    
    def _display_gameboard_header(self, current_player: Player) -> None:
        """
        Display gameboard header with scores and game state.
        
        Parameters
        ----------
        current_player : Player
            The current player viewing the gameboard.
        """
        if self.players is None:
            return
        
        # Display scores prominently at the top (no indentation)
        team0_score, team1_score = self.team_scores
        print(f"\nScore: {team0_score} - {team1_score}")
        
        # Display hand and trick indicators
        if self.current_hand_number > 0:
            hand_str = f"Hand: {self.current_hand_number}"
            if self.current_trick_number > 0:
                trick_str = f"Trick: {self.current_trick_number}/5"
                # Show current trick counts
                tricks_team0, tricks_team1 = self.current_tricks_won
                tricks_str = f"Tricks: Team 0: {tricks_team0}, Team 1: {tricks_team1}"
                print(f"{hand_str} | {trick_str} | {tricks_str}")
            else:
                print(hand_str)
        elif self.current_trick_number > 0:
            print(f"Trick: {self.current_trick_number}/5")
        
        if self.trump_suit is not None:
            print(f"Trump: {self.trump_suit.value} {self.trump_suit.unicode_symbol()}")
        
        # Show player positions around table
        current_id = current_player.player_id
        current_team = current_player.team
        
        # Collect player information first to calculate column widths
        from eucher.players.profiles import HumanProfile
        player_data = []
        for i in range(4):
            pid = (current_id + i) % 4
            player = self.players[pid]
            is_dealer = self.dealer_id is not None and pid == self.dealer_id
            dealer_indicator = "D" if is_dealer else " "
            
            # Get profile class name
            if isinstance(player.profile, HumanProfile):
                profile_class = "Human"
            else:
                profile_class = player.profile.__class__.__name__
            
            # Get markers
            markers = []
            if pid == current_id:
                markers.append(" ← YOU")
            elif hasattr(player, 'team') and player.team == current_team:
                markers.append(" *")
            marker_str = "".join(markers)
            
            team_marker = f" [Team {player.team}]" if hasattr(player, 'team') else ""
            
            player_data.append({
                'dealer': dealer_indicator,
                'pid': pid,
                'name': player.name,
                'team': team_marker,
                'profile': profile_class,
                'markers': marker_str
            })
        
        # Calculate column widths for alignment
        max_name_len = max(len(d['name']) for d in player_data)
        max_team_len = max(len(d['team']) for d in player_data)
        # Profile length includes brackets: [ProfileName]
        max_profile_len = max(len(f"[{d['profile']}]") for d in player_data)
        
        # Print table
        print("Players:")
        for data in player_data:
            dealer_col = data['dealer']
            player_col = f"Player {data['pid']}:"
            name_col = data['name'].ljust(max_name_len)
            team_col = data['team'].ljust(max_team_len)
            profile_col = f"[{data['profile']}]".ljust(max_profile_len)
            markers_col = data['markers']
            
            print(f"{dealer_col} {player_col} {name_col} {team_col} {profile_col}{markers_col}")
        print()
        
        # In xray mode, show all players' hands
        if self.xray_mode:
            self._display_all_hands(current_player)
        
        # In xray mode, show kitty cards only before gameplay begins (during trump selection)
        # Hide during actual trick play
        if self.xray_mode and self.kitty_cards and self.current_trick_number == 0:
            kitty_str = self._format_cards_xray(self.kitty_cards)
            print(f"[XRAY] Kitty: {kitty_str}")
            if self.turned_card is not None:
                turned_str = self._format_card_xray(self.turned_card)
                print(f"[XRAY] Turned card: {turned_str}")
            if self.discarded_card is not None:
                discarded_str = self._format_card_xray(self.discarded_card)
                print(f"[XRAY] Discarded card: {discarded_str}")
            print()

    def display_turned_card(self, card: Card) -> None:
        """
        Display the turned up card prominently after the deal.

        Parameters
        ----------
        card : Card
            The turned card.
        """
        if self.xray_mode:
            # In xray mode, use formatted display with unicode, colors, and bold
            card_str = self._format_card_xray(card)
            print(f"\nTurned card: {card_str} ({card.suit.value})\n")
        else:
            print(f"\nTurned card: {card} ({card.suit.value})\n")
        
        # In xray mode, also show the kitty cards
        if self.xray_mode and self.kitty_cards:
            kitty_str = self._format_cards_xray(self.kitty_cards)
            print(f"[XRAY] Kitty cards: {kitty_str}\n")

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
        Display the trick winner and wait for spacebar to continue.

        Parameters
        ----------
        winner_name : str
            Name of the winning player.
        """
        import sys
        
        # Display all cards played in the trick
        if self.current_trick_cards and self.current_trick_player_ids and self.players:
            print("\nCards played this trick:")
            # Find winner ID by name
            winner_id: Optional[int] = None
            for pid, player in enumerate(self.players):
                if player.name == winner_name:
                    winner_id = pid
                    break
            
            # Check if going alone (fewer than 4 cards means someone sat out)
            going_alone = len(self.current_trick_cards) < 4
            if going_alone:
                print("  [GOING ALONE - Only 3 players in this trick]")
            
            for card, pid in zip(self.current_trick_cards, self.current_trick_player_ids):
                player_name = self.players[pid].name
                winner_marker = " *" if winner_id is not None and pid == winner_id else ""
                print(f"  {player_name}: {card}{winner_marker}")
        
        # Display trick summary with current counts
        tricks_team0, tricks_team1 = self.current_tricks_won
        print(f"\n{winner_name} wins the trick!")
        print(f"\nTrick Summary:")
        print(f"  Team 0: {tricks_team0} tricks")
        print(f"  Team 1: {tricks_team1} tricks")
        print("Press spacebar to continue to next trick...")
        sys.stdout.flush()  # Ensure output is flushed before waiting for input
        self._wait_for_spacebar()
    
    def display_renege(
        self,
        renege_card: Card,
        renege_player_id: int,
        renege_team: int,
        points_awarded: Tuple[int, int],
    ) -> None:
        """
        Display renege message with card and points awarded.

        Parameters
        ----------
        renege_card : Card
            Card that was reneged.
        renege_player_id : int
            ID of player who reneged.
        renege_team : int
            Team of player who reneged.
        points_awarded : Tuple[int, int]
            Points awarded to each team (team0, team1) due to renege.
        """
        if self.players is None:
            return
        
        renege_player_name = self.players[renege_player_id].name if renege_player_id < len(self.players) else f"Player {renege_player_id}"
        
        if RICH_AVAILABLE:
            console = Console()
            console.print(f"\n[bold red]RENEGE[/bold red]")
            console.print(f"[red]{renege_player_name}[/red] played [bold]{renege_card}[/bold] (invalid play)")
            console.print(f"Points awarded: Team 0: {points_awarded[0]}, Team 1: {points_awarded[1]}")
        else:
            print(f"\n{self._red_text('RENEGE')}")
            print(f"{self._red_text(renege_player_name)} played {renege_card} (invalid play)")
            print(f"Points awarded: Team 0: {points_awarded[0]}, Team 1: {points_awarded[1]}")
    
    def _wait_for_spacebar(self) -> None:
        """
        Wait for user to press spacebar (or Enter as fallback).
        """
        import sys
        
        # Ensure all output is flushed before waiting for input
        sys.stdout.flush()
        sys.stderr.flush()
        
        try:
            import tty
            import termios
            
            # Try to use raw input for spacebar detection
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                while True:
                    char = sys.stdin.read(1)
                    if ord(char) == 32 or ord(char) == 13:  # Spacebar or Enter
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except (ImportError, OSError, AttributeError):
            # Fallback for systems without termios (e.g., Windows)
            # Just wait for Enter key
            input()
    
    def _red_text(self, text: str) -> str:
        """
        Format text in red using ANSI color codes.
        
        Parameters
        ----------
        text : str
            Text to format in red.
        
        Returns
        -------
        str
            Text with ANSI red color codes.
        """
        return f"\033[91m{text}\033[0m"
    
    def _display_all_hands(self, current_player: Player) -> None:
        """
        Display all players' hands (xray mode only).
        
        Parameters
        ----------
        current_player : Player
            The current player viewing the gameboard.
        """
        if self.players is None:
            return
        
        print("[XRAY] All Players' Hands:")
        print("-" * 70)
        current_id = current_player.player_id
        
        for i in range(4):
            pid = (current_id + i) % 4
            player = self.players[pid]
            # Format cards with unicode symbols, colors, and bold
            formatted_cards = [self._format_card_xray(card) for card in self._sort_hand_human_style(player.hand)]
            hand_str = "  ".join(formatted_cards)
            you_marker = " ← YOU" if pid == current_id else ""
            print(f"  Player {pid} ({player.name}){you_marker}: {hand_str}")
        print()
    
    def _format_card_xray(self, card: Card) -> str:
        """
        Format a card for xray display with unicode symbols, colors, and bold.
        
        Red suits (Hearts, Diamonds) are displayed in red.
        All cards are displayed in bold.
        
        Parameters
        ----------
        card : Card
            The card to format.
        
        Returns
        -------
        str
            Formatted card string with ANSI color codes.
        """
        # Use repr() to get unicode symbols (e.g., "K♥" instead of "K of Hearts")
        card_repr = repr(card)
        
        # Red suits: Hearts and Diamonds
        red_suits = {Suit.HEARTS, Suit.DIAMONDS}
        
        # Apply bold and color
        if card.suit in red_suits:
            # Red suit: bold + red color
            return f"\033[1m\033[91m{card_repr}\033[0m"
        else:
            # Black suit: bold only (default color is black)
            return f"\033[1m{card_repr}\033[0m"
    
    def _format_cards_xray(self, cards: List[Card]) -> str:
        """
        Format a list of cards for xray display.
        
        Parameters
        ----------
        cards : List[Card]
            List of cards to format.
        
        Returns
        -------
        str
            Comma-separated string of formatted cards.
        """
        formatted = [self._format_card_xray(card) for card in cards]
        return ", ".join(formatted)

    def display_scores(self, team0_score: int, team1_score: int) -> None:
        """
        Display current scores and update stored scores.

        Parameters
        ----------
        team0_score : int
            Team 0's score.
        team1_score : int
            Team 1's score.
        """
        self.team_scores = (team0_score, team1_score)
        # Score is now displayed in gameboard header, so we don't need a separate display here
    
    def update_trick_number(self, trick_number: int) -> None:
        """
        Update the current trick number.
        
        Parameters
        ----------
        trick_number : int
            The current trick number (0-4).
        """
        self.current_trick_number = trick_number
    
    def update_tricks_won(self, tricks_won: Tuple[int, int]) -> None:
        """
        Update the current tricks won by each team.
        
        Parameters
        ----------
        tricks_won : Tuple[int, int]
            Tricks won by Team 0 and Team 1.
        """
        self.current_tricks_won = tricks_won
    
    def update_dealer_id(self, dealer_id: int) -> None:
        """
        Update the dealer ID.
        
        Parameters
        ----------
        dealer_id : int
            The dealer's player ID.
        """
        self.dealer_id = dealer_id

    def display_game_over(self, winning_team: int) -> None:
        """
        Display game over message.

        Parameters
        ----------
        winning_team : int
            The winning team ID (0 or 1).
        """
        print(f"\nGame Over! Team {winning_team} wins!")

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
        # Check for playback decision first
        if self.decision_playback:
            from eucher.decision_playback import DecisionType

            playback_decision = self.decision_playback.get_playback_decision_deserialized(
                DecisionType.ORDER_UP
            )
            if playback_decision is not None:
                # Clear screen and show gameboard for context
                self._clear_screen()
                self._display_gameboard_header(player)
                print(f"\n[PLAYBACK] Order up: {playback_decision}")
                print("Press spacebar to continue...")
                self._wait_for_spacebar()
                self.decision_playback.record_decision(DecisionType.ORDER_UP, playback_decision)
                return playback_decision

        # Clear screen and show gameboard
        self._clear_screen()
        self._display_gameboard_header(player)
        
        # Show decisions made so far by other players
        if self.current_hand_log:
            decisions_shown = False
            for log_entry in self.current_hand_log:
                if log_entry.endswith(": Ordered up") or log_entry.endswith(": Passed"):
                    # Extract player name (everything before ": ")
                    if ": " in log_entry:
                        player_name = log_entry.rsplit(": ", 1)[0].strip()
                        if player_name != player.name:
                            if not decisions_shown:
                                print("Decisions so far:")
                                decisions_shown = True
                            print(f"  {log_entry}")
            if decisions_shown:
                print()
        
        # Show context
        dealer_name = self.players[dealer_id].name if self.players else f"Player {dealer_id}"
        print(f"Order up?")
        print(f"Dealer: {dealer_name}")
        print(f"Turned card: {turned_card}")
        print(f"If you order up, {dealer_name} will pick it up and discard a card.\n")
        
        # Display hand
        self.display_hand(player)
        
        # In xray mode, show kitty cards
        if self.xray_mode and self.kitty_cards:
            kitty_str = self._format_cards_xray(self.kitty_cards)
            print(f"\n[XRAY] Kitty cards: {kitty_str}")
        
        # Get assistant recommendations if available
        yes_probs: Dict[str, float] = {}  # assistant_name -> prob
        no_probs: Dict[str, float] = {}  # assistant_name -> prob
        hand_strength = None
        if self.assistant_helpers:
            for helper in self.assistant_helpers:
                try:
                    recommendations, hand_strength = helper.get_order_up_recommendation(
                        player.hand, turned_card, dealer_id
                    )
                    yes_probs[helper.bot_type] = recommendations.get(True, 0.0) * 100
                    no_probs[helper.bot_type] = recommendations.get(False, 0.0) * 100
                except Exception:
                    pass  # Silently fail if assistant fails
        
        # Display hand strength if available
        if hand_strength is not None:
            print(f"\n{self._red_text(f'Hand Strength: {hand_strength:.1f}')}")
        
        # Reset going alone decision for this order up decision
        self.going_alone_decision = None
        
        # Get decision
        while True:
            print("\nOrder up this card?")
            if yes_probs or no_probs:
                yes_parts = [f"{name}: {prob:.0f}%" for name, prob in yes_probs.items()]
                no_parts = [f"{name}: {prob:.0f}%" for name, prob in no_probs.items()]
                yes_str = self._red_text(f"({', '.join(yes_parts)})") if yes_parts else ""
                no_str = self._red_text(f"({', '.join(no_parts)})") if no_parts else ""
                print(f"  y) Yes {yes_str}")
                print(f"  a) Alone (order up and go alone)")
                print(f"  n) No {no_str}")
            else:
                print("  y) Yes")
                print("  a) Alone (order up and go alone)")
                print("  n) No")
            print("Choice (y/a/n): ", end="")
            response = input().strip().lower()
            if response in ("y", "yes"):
                decision = True
                # Don't set going_alone_decision - user will be asked later
            elif response in ("a", "alone"):
                decision = True
                self.going_alone_decision = True
            elif response in ("n", "no"):
                decision = False
                # Clear going_alone_decision since we're not ordering up
                self.going_alone_decision = None
            else:
                print("Invalid input. Please enter 'y' for yes, 'a' for alone, or 'n' for no.")
                continue

            # Record decision
            if self.decision_playback:
                from eucher.decision_playback import DecisionType

                self.decision_playback.record_decision(DecisionType.ORDER_UP, decision)
            return decision

    def get_going_alone_decision(self, player: Player, trump_suit: Suit) -> bool:
        """
        Get user input for going alone decision.

        If the player already decided to go alone when ordering up (via 'a' option),
        return that decision. Otherwise, prompt the user.

        Parameters
        ----------
        player : Player
            The human player making the decision.
        trump_suit : Suit
            The trump suit that was selected.

        Returns
        -------
        bool
            True to go alone, False to play with partner.
        """
        # If we already decided to go alone when ordering up, use that decision
        if self.going_alone_decision is not None:
            decision = self.going_alone_decision
            # Clear the decision so it doesn't affect future decisions
            self.going_alone_decision = None
            return decision

        # Otherwise, prompt the user
        # Check for playback decision first
        if self.decision_playback:
            from eucher.decision_playback import DecisionType

            playback_decision = self.decision_playback.get_playback_decision_deserialized(
                DecisionType.GOING_ALONE
            )
            if playback_decision is not None:
                # Clear screen and show gameboard for context
                self._clear_screen()
                self._display_gameboard_header(player)
                print(f"\n[PLAYBACK] Going alone: {playback_decision}")
                print("Press spacebar to continue...")
                self._wait_for_spacebar()
                self.decision_playback.record_decision(DecisionType.GOING_ALONE, playback_decision)
                return playback_decision

        # Clear screen and show gameboard
        self._clear_screen()
        self._display_gameboard_header(player)

        # Show context
        print(f"\nTrump suit: {trump_suit}")
        print("Do you want to go alone?")
        print("  (Going alone means your partner sits out and you play against both opponents)")
        print("  (If you win all 5 tricks alone, you get 4 points instead of 2)")

        # Display hand
        self.display_hand(player)

        # Get decision
        while True:
            print("\nGo alone?")
            print("  y) Yes")
            print("  n) No")
            print("Choice (y/n): ", end="")
            response = input().strip().lower()
            if response in ("y", "yes"):
                decision = True
            elif response in ("n", "no"):
                decision = False
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")
                continue

            # Record decision
            if self.decision_playback:
                from eucher.decision_playback import DecisionType

                self.decision_playback.record_decision(DecisionType.GOING_ALONE, decision)
            return decision

    def get_trade_in_decision(self, player: Player, eligible_cards: List[Card]) -> bool:
        """
        Get user input for trade-in decision.

        Parameters
        ----------
        player : Player
            The human player making the decision.
        eligible_cards : List[Card]
            The three cards eligible for trade-in.

        Returns
        -------
        bool
            True to trade-in, False to pass.
        """
        # Check for playback decision first
        if self.decision_playback:
            from eucher.decision_playback import DecisionType

            playback_decision = self.decision_playback.get_playback_decision_deserialized(
                DecisionType.TRADE_IN
            )
            if playback_decision is not None:
                # Clear screen and show gameboard for context
                self._clear_screen()
                self._display_gameboard_header(player)
                print(f"\n[PLAYBACK] Trade in: {playback_decision}")
                print("Press spacebar to continue...")
                self._wait_for_spacebar()
                self.decision_playback.record_decision(DecisionType.TRADE_IN, playback_decision)
                return playback_decision

        # Clear screen and show gameboard
        self._clear_screen()
        self._display_gameboard_header(player)
        
        # Show context
        print("Trade-in opportunity!")
        print("You have three cards of the same suit (all 9s or 10s):")
        for card in eligible_cards:
            print(f"  {card}")
        print("\nYou can trade these three cards for the three kitty cards.\n")
        
        # Display hand
        self.display_hand(player)
        
        # Get decision
        while True:
            print("\nTrade in these cards? (y/n): ", end="")
            response = input().strip().lower()
            if response in ("y", "yes"):
                decision = True
            elif response in ("n", "no"):
                decision = False
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")
                continue

            # Record decision
            if self.decision_playback:
                from eucher.decision_playback import DecisionType

                self.decision_playback.record_decision(DecisionType.TRADE_IN, decision)
            return decision

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
        # Check for playback decision first
        if self.decision_playback:
            from eucher.decision_playback import DecisionType

            playback_decision = self.decision_playback.get_playback_decision_deserialized(
                DecisionType.CALL_TRUMP
            )
            if playback_decision is not None:
                # Clear screen and show gameboard for context
                self._clear_screen()
                self._display_gameboard_header(player)
                decision_str = playback_decision.value if playback_decision else "Pass"
                print(f"\n[PLAYBACK] Call trump: {decision_str}")
                print("Press spacebar to continue...")
                self._wait_for_spacebar()
                self.decision_playback.record_decision(DecisionType.CALL_TRUMP, playback_decision)
                return playback_decision

        # Clear screen and show gameboard
        self._clear_screen()
        self._display_gameboard_header(player)
        
        # Show context
        print(f"Call trump")
        print(f"Turned card: {turned_card} (cannot be chosen as trump)")
        if must_choose:
            print("⚠️  SCREW THE DEALER: You MUST choose a suit!")
        else:
            print("All players passed on ordering up. Now choose a trump suit or pass.\n")
        
        # Display hand
        self.display_hand(player)
        
        # In xray mode, show kitty cards
        if self.xray_mode and self.kitty_cards:
            kitty_str = self._format_cards_xray(self.kitty_cards)
            print(f"\n[XRAY] Kitty cards: {kitty_str}")
        
        # Get assistant recommendations if available
        suit_probs: Dict[Suit, Dict[str, float]] = {}  # suit -> {assistant_name: prob}
        pass_probs: Dict[str, float] = {}  # assistant_name -> prob
        hand_strengths: Optional[Dict[Suit, float]] = None
        if self.assistant_helpers:
            for helper in self.assistant_helpers:
                try:
                    recommendations, hand_strengths = helper.get_call_trump_recommendation(
                        player.hand, turned_card, must_choose
                    )
                    for suit, prob in recommendations.items():
                        if suit is None:
                            pass_probs[helper.bot_type] = prob * 100
                        else:
                            if suit not in suit_probs:
                                suit_probs[suit] = {}
                            suit_probs[suit][helper.bot_type] = prob * 100
                except Exception:
                    pass  # Silently fail if assistant fails
        
        # Display hand strengths if available
        if hand_strengths is not None:
            print(f"\n{self._red_text('Hand Strength by Suit:')}")
            for suit, strength in hand_strengths.items():
                print(f"  {self._red_text(f'{suit.value}: {strength:.1f}')}")
        
        # Get decision
        while True:
            print("\nCall trump:")
            suit_map = {
                "1": Suit.HEARTS,
                "2": Suit.DIAMONDS,
                "3": Suit.CLUBS,
                "4": Suit.SPADES,
            }
            
            for choice, suit in suit_map.items():
                prob_str = ""
                if suit in suit_probs and suit_probs[suit]:
                    prob_parts = [f"{name}: {prob:.0f}%" for name, prob in suit_probs[suit].items()]
                    prob_str = f" {self._red_text('(' + ', '.join(prob_parts) + ')')}"
                print(f"  {choice}. {suit.value} {suit.unicode_symbol()}{prob_str}")
            
            if not must_choose:
                pass_str = ""
                if pass_probs:
                    prob_parts = [f"{name}: {prob:.0f}%" for name, prob in pass_probs.items()]
                    pass_str = f" {self._red_text('(' + ', '.join(prob_parts) + ')')}"
                print(f"  5. Pass{pass_str}")
            print(f"\nChoice (1-{'4' if must_choose else '5'}): ", end="")

            choice = input().strip()
            decision: Optional[Suit] = None
            
            if choice in suit_map:
                suit = suit_map[choice]
                if suit == turned_card.suit:
                    print(f"❌ Cannot choose {suit.value} (same as turned card).")
                    if must_choose:
                        print("Please choose another suit.")
                        continue
                    print("Passing.")
                    decision = None
                else:
                    decision = suit
            else:
                if must_choose:
                    print("❌ Invalid choice. You must choose a suit (1-4).")
                    continue
                print("❌ Invalid choice. Passing.")
                decision = None

            # Record decision
            if self.decision_playback:
                from eucher.decision_playback import DecisionType

                self.decision_playback.record_decision(DecisionType.CALL_TRUMP, decision)
            return decision

    def get_discard_decision(self, player: Player, turned_card: Optional[Card] = None, ordered_up_by: Optional[str] = None) -> Card:
        """
        Get user input for discarding a card.

        Parameters
        ----------
        player : HumanPlayer
            The dealer player.
        turned_card : Optional[Card]
            The card that was ordered up, if available.
        ordered_up_by : Optional[str]
            Name of the player who ordered up, if available.

        Returns
        -------
        Card
            The card to discard.
        """
        # Check for playback decision first
        if self.decision_playback:
            from eucher.decision_playback import DecisionType

            playback_decision = self.decision_playback.get_playback_decision_deserialized(
                DecisionType.DISCARD
            )
            if playback_decision is not None:
                # Clear screen and show gameboard for context
                self._clear_screen()
                self._display_gameboard_header(player)
                print(f"\n[PLAYBACK] Discard: {playback_decision}")
                print("Press spacebar to continue...")
                self._wait_for_spacebar()
                self.decision_playback.record_decision(DecisionType.DISCARD, playback_decision)
                return playback_decision

        # Clear screen and show gameboard
        self._clear_screen()
        self._display_gameboard_header(player)
        
        print(f"Discard card")
        if turned_card is not None:
            if ordered_up_by and ordered_up_by != player.name:
                print(f"{ordered_up_by} ordered up the {turned_card} ({turned_card.suit.value}).")
                print(f"You (the dealer) must pick it up. You now have 6 cards.")
            elif ordered_up_by is None:
                # Dealer ordered up themselves
                print(f"You ordered up the {turned_card} ({turned_card.suit.value}).")
                print(f"You now have 6 cards (including the {turned_card.suit.value} you picked up).")
            else:
                print(f"The {turned_card} ({turned_card.suit.value}) was ordered up. You now have 6 cards.")
        else:
            print("A card was ordered up. You now have 6 cards.")
        print("Choose one card to discard (you'll keep 5 cards).\n")
        
        # Display hand (sorted for human-style viewing)
        sorted_hand = self._sort_hand_human_style(player.hand)
        
        # Get assistant recommendations if available
        card_probs: Dict[Card, Dict[str, float]] = {}  # card -> {assistant_name: prob}
        if self.assistant_helpers:
            for helper in self.assistant_helpers:
                try:
                    recommendations = helper.get_discard_recommendation(
                        player.hand, turned_card, ordered_up_by
                    )
                    for card, prob in recommendations.items():
                        if card not in card_probs:
                            card_probs[card] = {}
                        card_probs[card][helper.bot_type] = prob * 100
                except Exception:
                    pass  # Silently fail if assistant fails
        
        print(f"\n{player.name}'s hand ({len(player.hand)} cards):")
        for i, card in enumerate(sorted_hand):
            prob_str = ""
            if card in card_probs and card_probs[card]:
                prob_parts = [f"{name}: {prob:.0f}%" for name, prob in card_probs[card].items()]
                prob_str = f" {self._red_text('(' + ', '.join(prob_parts) + ')')}"
            print(f"  {i + 1}. {card}{prob_str}")
        
        # Get decision
        while True:
            print(f"\nChoose a card to discard (1-{len(player.hand)}): ", end="")
            choice = input().strip()

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(sorted_hand):
                    selected_card = sorted_hand[idx]
                    print(f"Discarding: {selected_card}")
                    # Record decision
                    if self.decision_playback:
                        from eucher.decision_playback import DecisionType

                        self.decision_playback.record_decision(DecisionType.DISCARD, selected_card)
                    return selected_card
                print(f"❌ Invalid choice. Please enter a number between 1 and {len(player.hand)}.")
            except ValueError:
                print(f"❌ Invalid input. Please enter a number between 1 and {len(player.hand)}.")

    def display_table(
        self,
        current_player: Player,
        led_suit: Optional[Suit] = None,
        trump_suit: Optional[Suit] = None,
    ) -> None:
        """
        Display cards played and game info (led suit, trump suit).

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
            return

        # Use stored trick state if available
        played_cards = self.current_trick_cards
        player_ids = self.current_trick_player_ids

        # Only display if cards have been played
        if played_cards:
            print()
            print("Cards played this trick:")
            current_id = current_player.player_id
            
            # Determine current winning player if we have cards
            current_winner_id: Optional[int] = None
            # Use provided trump_suit or stored trump_suit
            effective_trump_suit = trump_suit if trump_suit is not None else self.trump_suit
            
            # Determine led_suit if not provided but cards have been played
            effective_led_suit = led_suit
            if effective_led_suit is None and played_cards:
                # Infer led suit from first card (simplified - doesn't handle bowers perfectly)
                effective_led_suit = played_cards[0].suit
            
            if effective_led_suit is not None and effective_trump_suit is not None:
                from eucher.rules import RulesEngine
                rules = RulesEngine()
                current_winner_id = rules.determine_trick_winner(
                    played_cards, player_ids, effective_led_suit, effective_trump_suit
                )
            
            for i, (card, pid) in enumerate(zip(played_cards, player_ids)):
                player_name = self.players[pid].name
                markers = []
                if pid == current_id:
                    markers.append(" ←")
                if current_winner_id is not None and pid == current_winner_id:
                    markers.append(" *")
                marker_str = "".join(markers)
                print(f"  {i+1}. {player_name}: {card}{marker_str}")

        # Display game info (led suit and trump suit) for decision making
        info_parts = []
        if led_suit:
            info_parts.append(f"Led: {led_suit.value} {led_suit.unicode_symbol()}")
        if trump_suit:
            info_parts.append(f"Trump: {trump_suit.value} {trump_suit.unicode_symbol()}")
        if info_parts:
            print(f"\n{' | '.join(info_parts):^70}")
        
        # In xray mode, show kitty cards only before gameplay begins (during trump selection)
        # Hide during actual trick play
        if self.xray_mode and self.kitty_cards and self.current_trick_number == 0:
            kitty_str = self._format_cards_xray(self.kitty_cards)
            print(f"\n[XRAY] Kitty: {kitty_str}")

    def get_play_card_decision(
        self,
        player: Player,
        led_suit: Optional[Suit],
        trump_suit: Optional[Suit],
        trick_cards: List[Card],
        trick_player_ids: List[int],
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
        trick_player_ids : List[int]
            Player IDs who played each card in trick_cards.

        Returns
        -------
        Card
            The card to play.
        """
        # Update stored trick state with the current trick state (includes all cards played so far)
        self.update_trick_state(trick_cards, trick_player_ids)
        
        # Check for playback decision first
        if self.decision_playback:
            from eucher.decision_playback import DecisionType

            playback_decision = self.decision_playback.get_playback_decision_deserialized(
                DecisionType.PLAY_CARD
            )
            if playback_decision is not None:
                # Clear screen and show gameboard for context
                self._clear_screen()
                self._display_gameboard_header(player)
                # Display table view to show current trick state
                if self.players is not None:
                    self.display_table(player, led_suit, trump_suit)
                print(f"\n[PLAYBACK] Play card: {playback_decision}")
                print("Press spacebar to continue...")
                self._wait_for_spacebar()
                self.decision_playback.record_decision(DecisionType.PLAY_CARD, playback_decision)
                return playback_decision

        # Clear screen and show gameboard
        self._clear_screen()
        self._display_gameboard_header(player)
        
        # Display table view (uses stored trick state, now updated with all cards played so far)
        if self.players is not None:
            self.display_table(player, led_suit, trump_suit)

        # Get valid cards to play
        from eucher.rules import RulesEngine
        rules = RulesEngine()
        valid_cards = rules.get_valid_plays(player.hand, led_suit, trump_suit)
        
        # If there are restrictions, only show playable cards
        if valid_cards and len(valid_cards) < len(player.hand):
            # Display only playable cards
            print(f"\n{player.name}'s playable cards ({len(valid_cards)} of {len(player.hand)} cards):")
            
            # Create mapping from displayed index to actual card index
            # First, collect valid cards with their original indices
            card_mapping: List[tuple[int, Card]] = []
            for i, card in enumerate(player.hand):
                if card in valid_cards:
                    card_mapping.append((i, card))
            
            # Sort the cards for display (human-style: by suit, then rank)
            sorted_mapping = sorted(
                card_mapping,
                key=lambda x: self._get_card_sort_key(x[1])
            )
            
            # Get assistant recommendations if available
            card_probs: Dict[Card, Dict[str, float]] = {}  # card -> {assistant_name: prob}
            if self.assistant_helpers:
                for helper in self.assistant_helpers:
                    try:
                        # Use stored trick player IDs if available, otherwise construct placeholder
                        trick_ids = self.current_trick_player_ids if len(self.current_trick_player_ids) == len(trick_cards) else list(range(len(trick_cards)))
                        recommendations = helper.get_play_card_recommendation(
                            player.hand, valid_cards, led_suit, trump_suit, trick_cards, trick_ids
                        )
                        for card, prob in recommendations.items():
                            if card not in card_probs:
                                card_probs[card] = {}
                            card_probs[card][helper.bot_type] = prob * 100
                    except Exception:
                        pass  # Silently fail if assistant fails
            
            # Display only playable cards (sorted)
            for display_idx, (actual_idx, card) in enumerate(sorted_mapping, start=1):
                prob_str = ""
                if card in card_probs and card_probs[card]:
                    prob_parts = [f"{name}: {prob:.0f}%" for name, prob in card_probs[card].items()]
                    prob_str = f" {self._red_text('(' + ', '.join(prob_parts) + ')')}"
                print(f"  {display_idx}. {card}{prob_str}")
            
            # Get decision
            while True:
                print(f"\nChoose a card to play (1-{len(valid_cards)}): ", end="")
                choice = input().strip()

                try:
                    display_idx = int(choice) - 1
                    if 0 <= display_idx < len(sorted_mapping):
                        actual_idx, selected_card = sorted_mapping[display_idx]
                        print(f"Playing: {selected_card}")
                        # Record decision
                        if self.decision_playback:
                            from eucher.decision_playback import DecisionType

                            self.decision_playback.record_decision(DecisionType.PLAY_CARD, selected_card)
                        return selected_card
                    print(f"❌ Invalid choice. Please enter a number between 1 and {len(valid_cards)}.")
                except ValueError:
                    print(f"❌ Invalid input. Please enter a number between 1 and {len(valid_cards)}.")
        else:
            # No restrictions - show all cards (sorted)
            sorted_hand = self._sort_hand_human_style(player.hand)
            
            # Get assistant recommendations if available
            card_probs: Dict[Card, Dict[str, float]] = {}  # card -> {assistant_name: prob}
            if self.assistant_helpers:
                for helper in self.assistant_helpers:
                    try:
                        # Use stored trick player IDs if available, otherwise construct placeholder
                        trick_ids = self.current_trick_player_ids if len(self.current_trick_player_ids) == len(trick_cards) else list(range(len(trick_cards)))
                        recommendations = helper.get_play_card_recommendation(
                            player.hand, valid_cards, led_suit, trump_suit, trick_cards, trick_ids
                        )
                        for card, prob in recommendations.items():
                            if card not in card_probs:
                                card_probs[card] = {}
                            card_probs[card][helper.bot_type] = prob * 100
                    except Exception:
                        pass  # Silently fail if assistant fails
            
            print(f"\n{player.name}'s hand ({len(player.hand)} cards):")
            for i, card in enumerate(sorted_hand):
                prob_str = ""
                if card in card_probs and card_probs[card]:
                    prob_parts = [f"{name}: {prob:.0f}%" for name, prob in card_probs[card].items()]
                    prob_str = f" {self._red_text('(' + ', '.join(prob_parts) + ')')}"
                print(f"  {i + 1}. {card}{prob_str}")
            
            # Get decision
            while True:
                print(f"\nChoose a card to play (1-{len(player.hand)}): ", end="")
                choice = input().strip()

                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(sorted_hand):
                        selected_card = sorted_hand[idx]
                        print(f"Playing: {selected_card}")
                        # Record decision
                        if self.decision_playback:
                            from eucher.decision_playback import DecisionType

                            self.decision_playback.record_decision(DecisionType.PLAY_CARD, selected_card)
                        return selected_card
                    print(f"❌ Invalid choice. Please enter a number between 1 and {len(player.hand)}.")
                except ValueError:
                    print(f"❌ Invalid input. Please enter a number between 1 and {len(player.hand)}.")

