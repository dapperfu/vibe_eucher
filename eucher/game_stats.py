"""Game statistics tracking and summary display."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from eucher.cards import Suit
from eucher.players import Player


class GameStatistics:
    """Tracks statistics throughout a game."""

    def __init__(self) -> None:
        """Initialize statistics tracker."""
        # Tricks won per player
        self.tricks_won_per_player: Dict[int, int] = defaultdict(int)
        
        # Tricks won per team
        self.tricks_won_per_team: Dict[int, int] = defaultdict(int)
        
        # Trump selections
        self.trump_selections: Dict[Optional[Suit], int] = defaultdict(int)
        
        # Points contributed per player (approximate - based on team points)
        self.points_contributed_per_player: Dict[int, int] = defaultdict(int)
        
        # Hands where player made trump
        self.trump_makers: Dict[int, int] = defaultdict(int)
        
        # Hands played
        self.hands_played: int = 0
        
        # Going alone attempts
        self.going_alone_attempts: Dict[int, int] = defaultdict(int)
        
        # Going alone successes
        self.going_alone_successes: Dict[int, int] = defaultdict(int)

    def record_trick_winner(self, player_id: int, team: int) -> None:
        """
        Record a trick win.

        Parameters
        ----------
        player_id : int
            ID of the player who won the trick.
        team : int
            Team ID (0 or 1) of the winning team.
        """
        self.tricks_won_per_player[player_id] += 1
        self.tricks_won_per_team[team] += 1

    def record_trump_selection(self, trump_suit: Optional[Suit], maker_id: Optional[int] = None) -> None:
        """
        Record a trump selection.

        Parameters
        ----------
        trump_suit : Optional[Suit]
            The trump suit selected, or None if no trump was selected.
        maker_id : Optional[int]
            ID of the player who made trump.
        """
        self.trump_selections[trump_suit] += 1
        if maker_id is not None:
            self.trump_makers[maker_id] += 1

    def record_hand_complete(
        self,
        tricks_won: List[int],
        points_awarded: List[int],
        players: List[Player],
        making_team: Optional[int] = None,
    ) -> None:
        """
        Record completion of a hand.

        Parameters
        ----------
        tricks_won : List[int]
            Tricks won by each team [team0, team1].
        points_awarded : List[int]
            Points awarded to each team [team0, team1].
        players : List[Player]
            List of all players.
        making_team : Optional[int]
            Team that made trump (for point attribution).
        """
        self.hands_played += 1
        
        # Distribute points to players on each team
        for team in range(2):
            points = points_awarded[team]
            if points > 0:
                # Distribute points equally among team members
                team_players = [p for p in players if p.team == team]
                points_per_player = points // len(team_players) if team_players else 0
                for player in team_players:
                    self.points_contributed_per_player[player.player_id] += points_per_player

    def record_going_alone(self, player_id: int, successful: bool) -> None:
        """
        Record a going alone attempt.

        Parameters
        ----------
        player_id : int
            ID of the player who went alone.
        successful : bool
            Whether the going alone attempt was successful (team won 5 tricks).
        """
        self.going_alone_attempts[player_id] += 1
        if successful:
            self.going_alone_successes[player_id] += 1

    def get_most_effective_player(self, players: List[Player]) -> Optional[Tuple[Player, Dict[str, float]]]:
        """
        Determine the most effective player based on multiple metrics.

        Parameters
        ----------
        players : List[Player]
            List of all players.

        Returns
        -------
        Optional[Tuple[Player, Dict[str, float]]]
            Tuple of (most effective player, stats dict) or None if no data.
        """
        if not players:
            return None

        # Calculate effectiveness score for each player
        player_scores: Dict[int, float] = {}
        
        for player in players:
            pid = player.player_id
            tricks = self.tricks_won_per_player[pid]
            points = self.points_contributed_per_player[pid]
            trump_makes = self.trump_makers[pid]
            
            # Weighted score: tricks (40%), points (40%), trump makes (20%)
            score = (tricks * 0.4) + (points * 0.4) + (trump_makes * 2.0 * 0.2)
            player_scores[pid] = score

        if not player_scores:
            return None

        # Find player with highest score
        best_player_id = max(player_scores.keys(), key=lambda pid: player_scores[pid])
        best_player = next(p for p in players if p.player_id == best_player_id)
        
        stats = {
            "tricks_won": self.tricks_won_per_player[best_player_id],
            "points_contributed": self.points_contributed_per_player[best_player_id],
            "trump_makes": self.trump_makers[best_player_id],
            "effectiveness_score": player_scores[best_player_id],
        }
        
        return (best_player, stats)

    def get_most_selected_trump(self) -> Optional[Tuple[Suit, int]]:
        """
        Get the most frequently selected trump suit.

        Returns
        -------
        Optional[Tuple[Suit, int]]
            Tuple of (trump suit, count) or None if no trump was selected.
        """
        # Filter out None (no trump selected)
        trump_counts = {suit: count for suit, count in self.trump_selections.items() if suit is not None}
        
        if not trump_counts:
            return None
        
        most_common = max(trump_counts.items(), key=lambda x: x[1])
        return most_common

    def get_summary_data(self, players: List[Player]) -> Dict:
        """
        Get comprehensive summary data.

        Parameters
        ----------
        players : List[Player]
            List of all players.

        Returns
        -------
        Dict
            Dictionary containing all summary statistics.
        """
        most_effective = self.get_most_effective_player(players)
        most_trump = self.get_most_selected_trump()
        
        return {
            "hands_played": self.hands_played,
            "tricks_won_per_player": {
                players[pid].name: self.tricks_won_per_player[pid]
                for pid in range(len(players))
            },
            "tricks_won_per_team": dict(self.tricks_won_per_team),
            "points_per_player": {
                players[pid].name: self.points_contributed_per_player[pid]
                for pid in range(len(players))
            },
            "trump_selections": {
                (suit.value if suit else "None"): count
                for suit, count in self.trump_selections.items()
            },
            "trump_makers": {
                players[pid].name: self.trump_makers[pid]
                for pid in range(len(players))
            },
            "most_effective_player": {
                "name": most_effective[0].name if most_effective else None,
                "stats": most_effective[1] if most_effective else None,
            },
            "most_selected_trump": {
                "suit": most_trump[0].value if most_trump else None,
                "count": most_trump[1] if most_trump else 0,
            },
            "going_alone_stats": {
                players[pid].name: {
                    "attempts": self.going_alone_attempts[pid],
                    "successes": self.going_alone_successes[pid],
                    "success_rate": (
                        self.going_alone_successes[pid] / self.going_alone_attempts[pid] * 100
                        if self.going_alone_attempts[pid] > 0
                        else 0.0
                    ),
                }
                for pid in range(len(players))
            },
        }


def create_bar_chart(data: Dict[str, int], max_width: int = 50, title: str = "") -> str:
    """
    Create a simple ASCII bar chart.

    Parameters
    ----------
    data : Dict[str, int]
        Dictionary mapping labels to values.
    max_width : int
        Maximum width of bars in characters.
    title : str
        Title for the chart.

    Returns
    -------
    str
        Formatted bar chart as string.
    """
    if not data:
        return ""

    lines: List[str] = []
    if title:
        lines.append(title)
        lines.append("=" * len(title))

    max_value = max(data.values()) if data.values() else 1
    
    for label, value in sorted(data.items(), key=lambda x: x[1], reverse=True):
        bar_width = int((value / max_value) * max_width) if max_value > 0 else 0
        bar = "█" * bar_width
        lines.append(f"{label:20s} │{bar} {value}")

    return "\n".join(lines)


def display_game_summary(stats: GameStatistics, players: List[Player], final_scores: Tuple[int, int], winner: Optional[int]) -> None:
    """
    Display a comprehensive game summary with bar charts.

    Parameters
    ----------
    stats : GameStatistics
        Game statistics object.
    players : List[Player]
        List of all players.
    final_scores : Tuple[int, int]
        Final scores (team0, team1).
    winner : Optional[int]
        Winning team (0 or 1) or None.
    """
    summary_data = stats.get_summary_data(players)
    
    print("\n" + "=" * 70)
    print("GAME SUMMARY")
    print("=" * 70)
    
    # Final scores
    print(f"\nFinal Score: Team 0: {final_scores[0]} - Team 1: {final_scores[1]}")
    if winner is not None:
        print(f"Winner: Team {winner}")
    print(f"Hands Played: {summary_data['hands_played']}")
    
    # Most effective player
    if summary_data["most_effective_player"]["name"]:
        mep = summary_data["most_effective_player"]
        print(f"\n🏆 Most Effective Player: {mep['name']}")
        if mep["stats"]:
            print(f"   Tricks Won: {mep['stats']['tricks_won']}")
            print(f"   Points Contributed: {mep['stats']['points_contributed']}")
            print(f"   Trump Makes: {mep['stats']['trump_makes']}")
    
    # Tricks won per player
    print("\n" + "-" * 70)
    print("Tricks Won Per Player")
    print("-" * 70)
    tricks_data = summary_data["tricks_won_per_player"]
    print(create_bar_chart(tricks_data, max_width=40, title=""))
    
    # Points per player
    print("\n" + "-" * 70)
    print("Points Contributed Per Player")
    print("-" * 70)
    points_data = summary_data["points_per_player"]
    print(create_bar_chart(points_data, max_width=40, title=""))
    
    # Trump selections
    print("\n" + "-" * 70)
    print("Trump Suit Selections")
    print("-" * 70)
    trump_data = summary_data["trump_selections"]
    if trump_data:
        print(create_bar_chart(trump_data, max_width=40, title=""))
        if summary_data["most_selected_trump"]["suit"]:
            print(f"\nMost Selected: {summary_data['most_selected_trump']['suit']} "
                  f"({summary_data['most_selected_trump']['count']} times)")
    
    # Trump makers
    print("\n" + "-" * 70)
    print("Times Made Trump")
    print("-" * 70)
    makers_data = summary_data["trump_makers"]
    if any(makers_data.values()):
        print(create_bar_chart(makers_data, max_width=40, title=""))
    
    # Going alone stats
    print("\n" + "-" * 70)
    print("Going Alone Statistics")
    print("-" * 70)
    for player_name, alone_stats in summary_data["going_alone_stats"].items():
        if alone_stats["attempts"] > 0:
            print(f"{player_name}: {alone_stats['attempts']} attempts, "
                  f"{alone_stats['successes']} successes "
                  f"({alone_stats['success_rate']:.1f}% success rate)")
    
    print("\n" + "=" * 70)

