"""Generate human-readable documentation of model inputs and decision weights."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.ml_features import GameStateEncoder


class InputDocumentationGenerator:
    """Generates documentation for model inputs and decision weights."""

    def __init__(self) -> None:
        """Initialize the documentation generator."""
        self.encoder = GameStateEncoder()

    def get_feature_descriptions(self) -> List[Dict[str, Any]]:
        """
        Get descriptions of all model input features.

        Returns
        -------
        List[Dict[str, Any]]
            List of feature descriptions with name, index, type, range, and description.
        """
        features: List[Dict[str, Any]] = []

        # Hand encoding (120 features: 5 cards × 24 one-hot)
        for card_pos in range(5):
            for card_idx in range(24):
                features.append(
                    {
                        "index": card_pos * 24 + card_idx,
                        "name": f"hand_card_{card_pos}_is_{self._card_index_to_name(card_idx)}",
                        "type": "binary",
                        "range": "[0, 1]",
                        "description": f"One-hot encoding: 1 if card position {card_pos} in hand is {self._card_index_to_name(card_idx)}, 0 otherwise",
                        "category": "hand_encoding",
                    }
                )

        # Turned card (24 features)
        for card_idx in range(24):
            features.append(
                {
                    "index": 120 + card_idx,
                    "name": f"turned_card_is_{self._card_index_to_name(card_idx)}",
                    "type": "binary",
                    "range": "[0, 1]",
                    "description": f"One-hot encoding: 1 if turned card is {self._card_index_to_name(card_idx)}, 0 otherwise",
                    "category": "turned_card",
                }
            )

        # Trump suit (4 features)
        suits = ["HEARTS", "DIAMONDS", "CLUBS", "SPADES"]
        for suit_idx, suit_name in enumerate(suits):
            features.append(
                {
                    "index": 144 + suit_idx,
                    "name": f"trump_suit_is_{suit_name}",
                    "type": "binary",
                    "range": "[0, 1]",
                    "description": f"One-hot encoding: 1 if trump suit is {suit_name}, 0 otherwise",
                    "category": "trump_suit",
                }
            )

        # Led suit (4 features)
        for suit_idx, suit_name in enumerate(suits):
            features.append(
                {
                    "index": 148 + suit_idx,
                    "name": f"led_suit_is_{suit_name}",
                    "type": "binary",
                    "range": "[0, 1]",
                    "description": f"One-hot encoding: 1 if led suit is {suit_name}, 0 otherwise",
                    "category": "led_suit",
                }
            )

        # Trick cards (72 features: 3 cards × 24 one-hot)
        for trick_pos in range(3):
            for card_idx in range(24):
                features.append(
                    {
                        "index": 152 + trick_pos * 24 + card_idx,
                        "name": f"trick_card_{trick_pos}_is_{self._card_index_to_name(card_idx)}",
                        "type": "binary",
                        "range": "[0, 1]",
                        "description": f"One-hot encoding: 1 if trick position {trick_pos} card is {self._card_index_to_name(card_idx)}, 0 otherwise",
                        "category": "trick_cards",
                    }
                )

        # Positional features (12 features)
        positional_features = [
            ("player_id_0", "Player ID one-hot: 1 if player is player 0"),
            ("player_id_1", "Player ID one-hot: 1 if player is player 1"),
            ("player_id_2", "Player ID one-hot: 1 if player is player 2"),
            ("player_id_3", "Player ID one-hot: 1 if player is player 3"),
            ("dealer_id_0", "Dealer ID one-hot: 1 if dealer is player 0"),
            ("dealer_id_1", "Dealer ID one-hot: 1 if dealer is player 1"),
            ("dealer_id_2", "Dealer ID one-hot: 1 if dealer is player 2"),
            ("dealer_id_3", "Dealer ID one-hot: 1 if dealer is player 3"),
            ("team", "Team binary: 0 for team 0, 1 for team 1"),
            ("trick_number_normalized", "Trick number normalized: current_trick / 4.0, range [0, 1]"),
            ("tricks_won_team0_normalized", "Tricks won by team 0 normalized: tricks_won / 5.0, range [0, 1]"),
            ("tricks_won_team1_normalized", "Tricks won by team 1 normalized: tricks_won / 5.0, range [0, 1]"),
        ]

        for idx, (name, description) in enumerate(positional_features):
            features.append(
                {
                    "index": 224 + idx,
                    "name": name,
                    "type": "binary" if idx < 9 else "continuous",
                    "range": "[0, 1]",
                    "description": description,
                    "category": "positional",
                }
            )

        return features

    def _card_index_to_name(self, card_idx: int) -> str:
        """
        Convert card index to human-readable name.

        Parameters
        ----------
        card_idx : int
            Card index (0-23).

        Returns
        -------
        str
            Card name (e.g., "9_HEARTS", "JACK_CLUBS").
        """
        from src.cards import Rank, Suit

        suits = list(Suit)
        ranks = list(Rank)

        suit_idx = card_idx // len(ranks)
        rank_idx = card_idx % len(ranks)

        suit = suits[suit_idx]
        rank = ranks[rank_idx]

        return f"{rank.name}_{suit.name}"

    def generate_markdown_docs(self, output_path: Path) -> None:
        """
        Generate markdown documentation of model inputs.

        Parameters
        ----------
        output_path : Path
            Path to save markdown documentation.
        """
        features = self.get_feature_descriptions()

        with open(output_path, "w") as f:
            f.write("# Model Input Features Documentation\n\n")
            f.write("This document describes all 260 input features used by the ML models.\n\n")
            f.write("## Feature Overview\n\n")
            f.write("| Category | Count | Description |\n")
            f.write("|----------|-------|-------------|\n")
            f.write("| Hand Encoding | 120 | 5 cards × 24 one-hot encodings |\n")
            f.write("| Turned Card | 24 | One-hot encoding of turned card |\n")
            f.write("| Trump Suit | 4 | One-hot encoding of trump suit |\n")
            f.write("| Led Suit | 4 | One-hot encoding of led suit |\n")
            f.write("| Trick Cards | 72 | 3 cards × 24 one-hot encodings |\n")
            f.write("| Played Cards | 24 | Binary encoding of cards played in previous tricks |\n")
            f.write("| Positional | 12 | Player position, dealer, team, trick info |\n")
            f.write("| **Total** | **260** | |\n\n")

            # Group by category
            categories: Dict[str, List[Dict[str, Any]]] = {}
            for feature in features:
                category = feature["category"]
                if category not in categories:
                    categories[category] = []
                categories[category].append(feature)

            for category, category_features in categories.items():
                f.write(f"## {category.replace('_', ' ').title()}\n\n")
                f.write("| Index | Name | Type | Range | Description |\n")
                f.write("|-------|------|------|-------|-------------|\n")

                for feature in category_features:
                    f.write(
                        f"| {feature['index']} | `{feature['name']}` | {feature['type']} | {feature['range']} | {feature['description']} |\n"
                    )
                f.write("\n")

            f.write("## Decision Weight System\n\n")
            f.write(
                "The ML models output decision weights (confidence scores) for each possible action.\n"
            )
            f.write("Temperature thresholds are applied to filter actions:\n\n")
            f.write("- **Low temperature (0.0-0.3)**: Conservative play (only high-confidence actions)\n")
            f.write("- **Medium temperature (0.3-0.7)**: Balanced play\n")
            f.write("- **High temperature (0.7-1.0)**: Risky play (accepts lower-confidence actions)\n\n")
            f.write(
                "Actions with weight >= temperature threshold are considered valid.\n"
            )
            f.write("Separate risk factors are used for:\n\n")
            f.write("- **Trump Selection**: `trump_selection_risk` (order up, call trump decisions)\n")
            f.write("- **Gameplay**: `gameplay_risk` (play card, discard decisions)\n\n")

    def generate_html_docs(self, output_path: Path) -> None:
        """
        Generate interactive HTML documentation of model inputs.

        Parameters
        ----------
        output_path : Path
            Path to save HTML documentation.
        """
        features = self.get_feature_descriptions()

        html = """<!DOCTYPE html>
<html>
<head>
    <title>Model Input Features Documentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .search-box { margin: 20px 0; padding: 10px; width: 300px; }
        .category-filter { margin: 10px 0; }
        .feature-index { font-weight: bold; color: #0066cc; }
    </style>
    <script>
        function filterTable() {
            const search = document.getElementById('search').value.toLowerCase();
            const category = document.getElementById('category').value;
            const rows = document.querySelectorAll('#features-table tr');
            
            rows.forEach((row, index) => {
                if (index === 0) return; // Skip header
                const text = row.textContent.toLowerCase();
                const rowCategory = row.getAttribute('data-category') || '';
                const matchSearch = text.includes(search);
                const matchCategory = !category || rowCategory === category;
                row.style.display = (matchSearch && matchCategory) ? '' : 'none';
            });
        }
    </script>
</head>
<body>
    <h1>Model Input Features Documentation</h1>
    <p>This interactive document describes all 260 input features used by the ML models.</p>
    
    <div>
        <input type="text" id="search" class="search-box" placeholder="Search features..." onkeyup="filterTable()">
        <select id="category" class="category-filter" onchange="filterTable()">
            <option value="">All Categories</option>
"""

        # Get unique categories
        categories = sorted(set(f["category"] for f in features))
        for category in categories:
            html += f'            <option value="{category}">{category.replace("_", " ").title()}</option>\n'

        html += """        </select>
    </div>
    
    <h2>Feature Overview</h2>
    <table>
        <tr>
            <th>Category</th>
            <th>Count</th>
            <th>Description</th>
        </tr>
        <tr>
            <td>Hand Encoding</td>
            <td>120</td>
            <td>5 cards × 24 one-hot encodings</td>
        </tr>
        <tr>
            <td>Turned Card</td>
            <td>24</td>
            <td>One-hot encoding of turned card</td>
        </tr>
        <tr>
            <td>Trump Suit</td>
            <td>4</td>
            <td>One-hot encoding of trump suit</td>
        </tr>
        <tr>
            <td>Led Suit</td>
            <td>4</td>
            <td>One-hot encoding of led suit</td>
        </tr>
        <tr>
            <td>Trick Cards</td>
            <td>72</td>
            <td>3 cards × 24 one-hot encodings</td>
        </tr>
        <tr>
            <td>Played Cards</td>
            <td>24</td>
            <td>Binary encoding of cards played in previous tricks</td>
        </tr>
        <tr>
            <td>Positional</td>
            <td>12</td>
            <td>Player position, dealer, team, trick info</td>
        </tr>
        <tr>
            <td><strong>Total</strong></td>
            <td><strong>260</strong></td>
            <td></td>
        </tr>
    </table>
    
    <h2>All Features</h2>
    <table id="features-table">
        <tr>
            <th>Index</th>
            <th>Name</th>
            <th>Type</th>
            <th>Range</th>
            <th>Description</th>
        </tr>
"""

        for feature in features:
            html += f"""
        <tr data-category="{feature['category']}">
            <td class="feature-index">{feature['index']}</td>
            <td><code>{feature['name']}</code></td>
            <td>{feature['type']}</td>
            <td>{feature['range']}</td>
            <td>{feature['description']}</td>
        </tr>
"""

        html += """    </table>
    
    <h2>Decision Weight System</h2>
    <p>The ML models output decision weights (confidence scores) for each possible action.</p>
    <p>Temperature thresholds are applied to filter actions:</p>
    <ul>
        <li><strong>Low temperature (0.0-0.3)</strong>: Conservative play (only high-confidence actions)</li>
        <li><strong>Medium temperature (0.3-0.7)</strong>: Balanced play</li>
        <li><strong>High temperature (0.7-1.0)</strong>: Risky play (accepts lower-confidence actions)</li>
    </ul>
    <p>Actions with weight >= temperature threshold are considered valid.</p>
    <p>Separate risk factors are used for:</p>
    <ul>
        <li><strong>Trump Selection</strong>: <code>trump_selection_risk</code> (order up, call trump decisions)</li>
        <li><strong>Gameplay</strong>: <code>gameplay_risk</code> (play card, discard decisions)</li>
    </ul>
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html)

