#!/usr/bin/env python3
"""
Create AI Demo Notebook

This script programmatically generates a Jupyter notebook demonstrating
the 'dumb' AI implementations with various scenarios and analysis.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

def create_notebook_cells() -> List[Dict[str, Any]]:
    """Create the notebook cells with AI demonstrations."""
    
    cells = []
    
    # Title and description
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Euchre AI Demonstration Notebook\n",
            "\n",
            "This notebook demonstrates how the 'dumb' AI players work in the Euchre game.\n",
            "We'll explore different AI profiles, risk ratios, and decision-making scenarios."
        ]
    })
    
    # Setup and imports
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Setup and Imports"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import sys\n",
            "import os\n",
            "from pathlib import Path\n",
            "import random\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Add the parent directory to the path to import euchre modules\n",
            "sys.path.append(str(Path.cwd().parent))\n",
            "\n",
            "from euchre.models import Card, Suit, Rank, Player, PlayerType\n",
            "from euchre.ai.ai_profiles import AggressiveAI, ConservativeAI, BalancedAI, OpportunisticAI\n",
            "from euchre.ai.base_ai import BaseAI\n",
            "\n",
            "print(\"✅ Imports successful\")\n",
            "print(f\"🎴 Available suits: {[s.name for s in Suit]}\")\n",
            "print(f\"🃏 Available ranks: {[r.name for r in Rank]}\")"
        ]
    })
    
    # AI Profile Overview
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## AI Profile Overview\n",
            "\n",
            "The Euchre system has four main AI profiles, each with different playing styles:"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create AI players with different profiles\n",
            "ai_profiles = {\n",
            "    'Aggressive': AggressiveAI('Aggressive', 0.8),\n",
            "    'Conservative': ConservativeAI('Conservative', 0.2),\n",
            "    'Balanced': BalancedAI('Balanced', 0.5),\n",
            "    'Opportunistic': OpportunisticAI('Opportunistic', 0.6)\n",
            "}\n",
            "\n",
            "print(\"🤖 AI Profiles Created:\")\n",
            "for name, ai in ai_profiles.items():\n",
            "    print(f\"  {name}: risk_ratio={ai.risk_ratio:.1f}\")\n",
            "\n",
            "# Show profile characteristics\n",
            "profile_info = {\n",
            "    'Profile': ['Aggressive', 'Conservative', 'Balanced', 'Opportunistic'],\n",
            "    'Risk Ratio': [0.8, 0.2, 0.5, 0.6],\n",
            "    'Style': ['High risk, high reward', 'Defensive, safe play', 'Moderate, balanced', 'Adaptive, situational'],\n",
            "    'Trump Calling': ['Frequent', 'Rare', 'Moderate', 'Context-dependent'],\n",
            "    'Card Play': ['High cards early', 'Save strong cards', 'Balanced approach', 'Situational']\n",
            "}\n",
            "\n",
            "df_profiles = pd.DataFrame(profile_info)\n",
            "display(df_profiles)"
        ]
    })
    
    # Scenario 1: Trump Calling Decision
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Scenario 1: Trump Calling Decision\n",
            "\n",
            "Let's see how different AI profiles decide whether to order up the top card."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create a test scenario for trump calling\n",
            "def test_trump_calling_scenario():\n",
            "    \"\"\"Test trump calling with different AI profiles.\"\"\"\n",
            "    \n",
            "    # Create a test hand\n",
            "    test_hand = [\n",
            "        Card(Suit.HEARTS, Rank.ACE),\n",
            "        Card(Suit.HEARTS, Rank.KING),\n",
            "        Card(Suit.DIAMONDS, Rank.JACK),  # Left bower for Hearts\n",
            "        Card(Suit.CLUBS, Rank.QUEEN),\n",
            "        Card(Suit.SPADES, Rank.TEN)\n",
            "    ]\n",
            "    \n",
            "    # Top card to potentially order up\n",
            "    top_card = Card(Suit.HEARTS, Rank.JACK)  # Right bower\n",
            "    \n",
    "    print(\"🎴 Test Scenario: Trump Calling Decision\")\n",
    "    print(\"=\" * 50)\n",
    "    print(f\"🃏 Top card: {top_card}\")\n",
    "    print(f\"👤 Player hand: {[str(card) for card in test_hand]}\")\n",
    "    print(f\"🎯 Potential trump suit: {top_card.suit.name}\")\n",
    "    print(f\"🃏 Left bower would be: {ai_profiles['Aggressive']._get_left_bower_suit(top_card.suit).name}\")\n",
    "    print()\n",
    "    \n",
    "    # Test each AI profile\n",
    "    results = []\n",
    "    for profile_name, ai in ai_profiles.items():\n",
    "        # Give the AI the test hand\n",
    "        ai.hand = test_hand.copy()\n",
    "        \n",
    "        # Evaluate hand strength\n",
    "        hand_strength = ai._evaluate_hand_for_trump(top_card.suit, top_card, False)\n",
    "        \n",
    "        # Get decision\n",
    "        should_order = ai.should_order_up(top_card, False)\n",
    "        \n",
    "        results.append({\n",
    "            'Profile': profile_name,\n",
    "            'Risk Ratio': ai.risk_ratio,\n",
    "            'Hand Strength': hand_strength,\n",
    "            'Should Order Up': should_order,\n",
    "            'Decision': '✅ Order Up' if should_order else '❌ Pass'\n",
    "        })\n",
    "        \n",
    "        print(f\"🤖 {profile_name} AI:\")\n",
    "        print(f\"  Risk ratio: {ai.risk_ratio:.1f}\")\n",
    "        print(f\"  Hand strength: {hand_strength:.1f}\")\n",
    "        print(f\"  Decision: {results[-1]['Decision']}\")\n",
    "        print()\n",
    "    \n",
    "    # Display results as DataFrame\n",
    "    df_results = pd.DataFrame(results)\n",
    "    display(df_results)\n",
    "    \n",
    "    return results\n",
    "\n",
    "# Run the scenario\n",
    "trump_results = test_trump_calling_scenario()"
        ]
    })
    
    # Scenario 2: Card Selection Logic
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Scenario 2: Card Selection Logic\n",
            "\n",
            "Now let's see how different AI profiles choose which card to play in different situations."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test card selection scenarios\n",
            "def test_card_selection_scenarios():\n",
            "    \"\"\"Test card selection in different game situations.\"\"\"\n",
    "    \n",
    "    print(\"🎮 Card Selection Scenarios\")\n",
    "    print(\"=\" * 50)\n",
    "    \n",
    "    # Scenario 2a: Leading a trick (no lead suit)\n",
    "    print(\"\\n🎯 Scenario 2a: Leading a Trick (No Lead Suit)\")\n",
    "    print(\"-\" * 40)\n",
    "    \n",
    "    test_hand = [\n",
    "        Card(Suit.HEARTS, Rank.ACE),\n",
    "        Card(Suit.HEARTS, Rank.KING),\n",
    "        Card(Suit.DIAMONDS, Rank.JACK),\n",
    "        Card(Suit.CLUBS, Rank.QUEEN),\n",
    "        Card(Suit.SPADES, Rank.TEN)\n",
    "    ]\n",
    "    \n",
    "    trump_suit = Suit.HEARTS\n",
    "    \n",
    "    print(f\"🃏 Hand: {[str(card) for card in test_hand]}\")\n",
    "    print(f\"🎯 Trump suit: {trump_suit.name}\")\n",
    "    print(f\"🎮 Situation: Leading a trick (no lead suit)\")\n",
    "    print()\n",
    "    \n",
    "    for profile_name, ai in ai_profiles.items():\n",
    "        ai.hand = test_hand.copy()\n",
    "        \n",
    "        # Simulate leading (no lead suit)\n",
    "        chosen_card = ai.choose_card_to_play(None, trump_suit)\n",
    "        \n",
    "        print(f\"🤖 {profile_name} AI chooses: {chosen_card}\")\n",
    "        \n",
    "        # Show reasoning\n",
    "        card_value = ai._card_value(chosen_card, trump_suit)\n",
    "        print(f\"  Card value: {card_value:.1f}\")\n",
    "        \n",
    "        if chosen_card.is_trump:\n",
    "            print(f\"  Strategy: Playing trump card (high value)\")\n",
    "        elif chosen_card.suit == ai._get_left_bower_suit(trump_suit) and chosen_card.rank == Rank.JACK:\n",
    "            print(f\"  Strategy: Playing left bower (second highest trump)\")\n",
    "        else:\n",
    "            print(f\"  Strategy: Playing high non-trump card\")\n",
    "        print()\n",
    "    \n",
    "    # Scenario 2b: Following suit\n",
    "    print(\"\\n🎯 Scenario 2b: Following Suit\")\n",
    "    print(\"-\" * 40)\n",
    "    \n",
    "    lead_suit = Suit.HEARTS\n",
    "    print(f\"🎯 Lead suit: {lead_suit.name}\")\n",
    "    print(f\"🎮 Situation: Must follow suit\")\n",
    "    print()\n",
    "    \n",
    "    for profile_name, ai in ai_profiles.items():\n",
    "        ai.hand = test_hand.copy()\n",
    "        \n",
    "        # Simulate following suit\n",
    "        chosen_card = ai.choose_card_to_play(lead_suit, trump_suit)\n",
    "        \n",
    "        print(f\"🤖 {profile_name} AI chooses: {chosen_card}\")\n",
    "        \n",
    "        # Show reasoning\n",
    "        if chosen_card.suit == lead_suit:\n",
    "            print(f\"  Strategy: Following suit with {chosen_card.rank.name}\")\n",
    "        else:\n",
    "            print(f\"  Strategy: Cannot follow suit, playing {chosen_card}\")\n",
    "        print()\n",
    "    \n",
    "    # Scenario 2c: Cannot follow suit\n",
    "    print(\"\\n🎯 Scenario 2c: Cannot Follow Suit\")\n",
    "    print(\"-\" * 40)\n",
    "    \n",
    "    # Remove hearts from hand\n",
    "    no_hearts_hand = [card for card in test_hand if card.suit != Suit.HEARTS]\n",
    "    \n",
    "    print(f\"🃏 Hand (no hearts): {[str(card) for card in no_hearts_hand]}\")\n",
    "    print(f\"🎯 Lead suit: {lead_suit.name}\")\n",
    "    print(f\"🎮 Situation: Cannot follow suit, can play any card\")\n",
    "    print()\n",
    "    \n",
    "    for profile_name, ai in ai_profiles.items():\n",
    "        ai.hand = no_hearts_hand.copy()\n",
    "        \n",
    "        # Simulate cannot follow suit\n",
    "        chosen_card = ai.choose_card_to_play(lead_suit, trump_suit)\n",
    "        \n",
    "        print(f\"🤖 {profile_name} AI chooses: {chosen_card}\")\n",
    "        \n",
    "        # Show reasoning\n",
    "        if chosen_card.is_trump:\n",
    "            print(f\"  Strategy: Playing trump to win trick\")\n",
    "        else:\n",
    "            print(f\"  Strategy: Playing high non-trump card\")\n",
    "        print()\n",
    "\n",
    "# Run the scenarios\n",
    "test_card_selection_scenarios()"
        ]
    })
    
    # Scenario 3: Trump Suit Selection
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Scenario 3: Trump Suit Selection\n",
            "\n",
            "When the top card is rejected, the dealer must choose a trump suit. Let's see how AIs make this decision."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test trump suit selection\n",
            "def test_trump_suit_selection():\n",
            "    \"\"\"Test how AIs choose trump suits when top card is rejected.\"\"\"\n",
    "    \n",
    "    print(\"🎯 Trump Suit Selection Scenario\")\n",
    "    print(\"=\" * 50)\n",
    "    \n",
    "    # Create a test hand with cards in multiple suits\n",
    "    test_hand = [\n",
    "        Card(Suit.HEARTS, Rank.ACE),\n",
    "        Card(Suit.HEARTS, Rank.KING),\n",
    "        Card(Suit.DIAMONDS, Rank.JACK),\n",
    "        Card(Suit.DIAMONDS, Rank.QUEEN),\n",
    "        Card(Suit.CLUBS, Rank.TEN)\n",
    "    ]\n",
    "    \n",
    "    # Top card that was rejected (cannot be chosen)\n",
    "    rejected_top_card = Card(Suit.SPADES, Rank.KING)\n",
    "    \n",
    "    print(f\"🃏 Hand: {[str(card) for card in test_hand]}\")\n",
    "    print(f\"❌ Rejected top card: {rejected_top_card}\")\n",
    "    print(f\"🎮 Situation: Dealer must choose trump suit (cannot be {rejected_top_card.suit.name})\")\n",
    "    print()\n",
    "    \n",
    "    # Count cards by suit\n",
    "    suit_counts = {}\n",
    "    for suit in Suit:\n",
    "        if suit != rejected_top_card.suit:  # Exclude rejected suit\n",
    "            count = len([card for card in test_hand if card.suit == suit])\n",
    "            suit_counts[suit] = count\n",
    "    \n",
    "    print(\"📊 Available suits and card counts:\")\n",
    "    for suit, count in suit_counts.items():\n",
    "        print(f\"  {suit.name}: {count} cards\")\n",
    "    print()\n",
    "    \n",
    "    # Test each AI profile\n",
    "    for profile_name, ai in ai_profiles.items():\n",
    "        ai.hand = test_hand.copy()\n",
    "        \n",
    "        # Get trump suit choice\n",
    "        chosen_trump = ai.choose_trump_suit(rejected_top_card)\n",
    "        \n",
    "        print(f\"🤖 {profile_name} AI chooses: {chosen_trump.name}\")\n",
    "        \n",
    "        # Show reasoning\n",
    "        card_count = suit_counts[chosen_trump]\n",
    "        print(f\"  Reasoning: {card_count} cards in {chosen_trump.name}\")\n",
    "        \n",
    "        # Check if this is the best choice\n",
    "        best_suit = max(suit_counts.keys(), key=lambda s: suit_counts[s])\n",
    "        if chosen_trump == best_suit:\n",
    "            print(f\"  ✅ Optimal choice (most cards)\")\n",
    "        else:\n",
    "            print(f\"  ⚠️  Suboptimal choice (best would be {best_suit.name} with {suit_counts[best_suit]} cards)\")\n",
    "        print()\n",
    "\n",
    "# Run the scenario\n",
    "test_trump_suit_selection()"
        ]
    })
    
    # Risk Ratio Analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Risk Ratio Analysis\n",
            "\n",
            "Let's analyze how different risk ratios affect AI decision-making across multiple scenarios."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Analyze risk ratio effects\n",
            "def analyze_risk_ratios():\n",
            "    \"\"\"Analyze how different risk ratios affect AI decisions.\"\"\"\n",
    "    \n",
    "    print(\"🎲 Risk Ratio Analysis\")\n",
    "    print(\"=\" * 50)\n",
    "    \n",
    "    # Test different risk ratios for each profile\n",
    "    risk_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]\n",
    "    \n",
    "    # Create test scenarios\n",
    "    test_scenarios = [\n",
    "        {\n",
    "            'name': 'Strong Hand',\n",
    "            'hand': [\n",
    "                Card(Suit.HEARTS, Rank.ACE),\n",
    "                Card(Suit.HEARTS, Rank.KING),\n",
    "                Card(Suit.HEARTS, Rank.QUEEN),\n",
    "                Card(Suit.DIAMONDS, Rank.JACK),\n",
    "                Card(Suit.CLUBS, Rank.TEN)\n",
    "            ],\n",
    "            'top_card': Card(Suit.HEARTS, Rank.JACK),\n",
    "            'expected': 'Order Up'\n",
    "        },\n",
    "        {\n",
    "            'name': 'Weak Hand',\n",
    "            'hand': [\n",
    "                Card(Suit.HEARTS, Rank.TEN),\n",
    "                Card(Suit.DIAMONDS, Rank.NINE),\n",
    "                Card(Suit.CLUBS, Rank.NINE),\n",
    "                Card(Suit.SPADES, Rank.TEN),\n",
    "                Card(Suit.HEARTS, Rank.NINE)\n",
    "            ],\n",
    "            'top_card': Card(Suit.HEARTS, Rank.JACK),\n",
    "            'expected': 'Pass'\n",
    "        },\n",
    "        {\n",
    "            'name': 'Marginal Hand',\n",
    "            'hand': [\n",
    "                Card(Suit.HEARTS, Rank.ACE),\n",
    "                Card(Suit.HEARTS, Rank.TEN),\n",
    "                Card(Suit.DIAMONDS, Rank.NINE),\n",
    "                Card(Suit.CLUBS, Rank.QUEEN),\n",
    "                Card(Suit.SPADES, Rank.KING)\n",
    "            ],\n",
    "            'top_card': Card(Suit.HEARTS, Rank.JACK),\n",
    "            'expected': 'Variable'\n",
    "        }\n",
    "    ]\n",
    "    \n",
    "    # Collect results\n",
    "    all_results = []\n",
    "    \n",
    "    for scenario in test_scenarios:\n",
    "        print(f\"\\n🎯 Scenario: {scenario['name']}\")\n",
    "        print(f\"🃏 Hand: {[str(card) for card in scenario['hand']]}\")\n",
    "        print(f\"🃏 Top card: {scenario['top_card']}\")\n",
    "        print(f\"📊 Expected: {scenario['expected']}\")\n",
    "        print(\"-\" * 40)\n",
    "        \n",
    "        for profile_name in ['Aggressive', 'Conservative', 'Balanced', 'Opportunistic']:\n",
    "            for risk_ratio in risk_ratios:\n",
    "                # Create AI with specific risk ratio\n",
    "                if profile_name == 'Aggressive':\n",
    "                    ai = AggressiveAI(f'{profile_name}_{risk_ratio}', risk_ratio)\n",
    "                elif profile_name == 'Conservative':\n",
    "                    ai = ConservativeAI(f'{profile_name}_{risk_ratio}', risk_ratio)\n",
    "                elif profile_name == 'Balanced':\n",
    "                    ai = BalancedAI(f'{profile_name}_{risk_ratio}', risk_ratio)\n",
    "                else:  # Opportunistic\n",
    "                    ai = OpportunisticAI(f'{profile_name}_{risk_ratio}', risk_ratio)\n",
    "                \n",
    "                # Give the AI the test hand\n",
    "                ai.hand = scenario['hand'].copy()\n",
    "                \n",
    "                # Get decision\n",
    "                should_order = ai.should_order_up(scenario['top_card'], False)\n",
    "                \n",
    "                # Store result\n",
    "                all_results.append({\n",
    "                    'Scenario': scenario['name'],\n",
    "                    'Profile': profile_name,\n",
    "                    'Risk Ratio': risk_ratio,\n",
    "                    'Decision': 'Order Up' if should_order else 'Pass',\n",
    "                    'Expected': scenario['expected']\n",
    "                })\n",
    "                \n",
    "                print(f\"  {profile_name} (risk={risk_ratio:.1f}): {'✅' if should_order else '❌'} {all_results[-1]['Decision']}\")\n",
    "        \n",
    "        print()\n",
    "    \n",
    "    # Create summary DataFrame\n",
    "    df_risk_analysis = pd.DataFrame(all_results)\n",
    "    \n",
    "    print(\"📊 Risk Ratio Analysis Summary\")\n",
    "    print(\"=\" * 50)\n",
    "    display(df_risk_analysis)\n",
    "    \n",
    "    # Create pivot table\n",
    "    pivot_table = df_risk_analysis.pivot_table(\n",
    "        index=['Profile', 'Risk Ratio'],\n",
    "        columns='Scenario',\n",
    "        values='Decision',\n",
    "        aggfunc='first'\n",
    "    )\n",
    "    \n",
    "    print(\"\\n📋 Decision Matrix by Profile and Risk Ratio\")\n",
    "    display(pivot_table)\n",
    "    \n",
    "    return df_risk_analysis\n",
    "\n",
    "# Run the analysis\n",
    "risk_analysis_results = analyze_risk_ratios()"
        ]
    })
    
    # Statistical Analysis
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Statistical Analysis\n",
            "\n",
            "Let's run multiple simulations to see how different AI profiles perform statistically."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run statistical analysis\n",
            "def run_statistical_analysis():\n",
            "    \"\"\"Run statistical analysis on AI decision-making.\"\"\"\n",
    "    \n",
    "    print(\"📊 Statistical Analysis of AI Decision-Making\")\n",
    "    print(\"=\" * 60)\n",
    "    \n",
    "    # Generate random hands for testing\n",
    "    def generate_random_hand() -> List[Card]:\n",
    "        \"\"\"Generate a random 5-card hand.\"\"\"\n",
    "        all_cards = []\n",
    "        for suit in Suit:\n",
    "            for rank in Rank:\n",
    "                all_cards.append(Card(suit, rank))\n",
    "        \n",
    "        return random.sample(all_cards, 5)\n",
    "    \n",
    "    def generate_random_top_card() -> Card:\n",
    "        \"\"\"Generate a random top card.\"\"\"\n",
    "        suit = random.choice(list(Suit))\n",
    "        rank = random.choice(list(Rank))\n",
    "        return Card(suit, rank)\n",
    "    \n",
    "    # Test parameters\n",
    "    num_simulations = 100\n",
    "    risk_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]\n",
    "    \n",
    "    # Collect data\n",
    "    simulation_data = []\n",
    "    \n",
    "    print(f\"🧪 Running {num_simulations} simulations for each profile and risk ratio...\")\n",
    "    \n",
    "    for profile_name in ['Aggressive', 'Conservative', 'Balanced', 'Opportunistic']:\n",
    "        for risk_ratio in risk_ratios:\n",
    "            print(f\"\\n🤖 Testing {profile_name} with risk ratio {risk_ratio:.1f}...\")\n",
    "            \n",
    "            # Create AI\n",
    "            if profile_name == 'Aggressive':\n",
    "                ai = AggressiveAI(f'{profile_name}_{risk_ratio}', risk_ratio)\n",
    "            elif profile_name == 'Conservative':\n",
    "                ai = ConservativeAI(f'{profile_name}_{risk_ratio}', risk_ratio)\n",
    "            elif profile_name == 'Balanced':\n",
    "                ai = BalancedAI(f'{profile_name}_{risk_ratio}', risk_ratio)\n",
    "            else:  # Opportunistic\n",
    "                ai = OpportunisticAI(f'{profile_name}_{risk_ratio}', risk_ratio)\n",
    "            \n",
    "            # Run simulations\n",
    "            order_up_count = 0\n",
    "            hand_strengths = []\n",
    "            \n",
    "            for sim in range(num_simulations):\n",
    "                # Generate random scenario\n",
    "                hand = generate_random_hand()\n",
    "                top_card = generate_random_top_card()\n",
    "                \n",
    "                # Give AI the hand\n",
    "                ai.hand = hand\n",
    "                \n",
    "                # Get decision and hand strength\n",
    "                should_order = ai.should_order_up(top_card, False)\n",
    "                hand_strength = ai._evaluate_hand_for_trump(top_card.suit, top_card, False)\n",
    "                \n",
    "                if should_order:\n",
    "                    order_up_count += 1\n",
    "                \n",
    "                hand_strengths.append(hand_strength)\n",
    "                \n",
    "                # Store data\n",
    "                simulation_data.append({\n",
    "                    'Profile': profile_name,\n",
    "                    'Risk Ratio': risk_ratio,\n",
    "                    'Simulation': sim,\n",
    "                    'Hand Strength': hand_strength,\n",
    "                    'Ordered Up': should_order,\n",
    "                    'Decision': 'Order Up' if should_order else 'Pass'\n",
    "                })\n",
    "            \n",
    "            # Calculate statistics\n",
    "            order_up_rate = order_up_count / num_simulations\n",
    "            avg_hand_strength = np.mean(hand_strengths)\n",
    "            \n",
    "            print(f\"  📊 Order up rate: {order_up_rate:.1%}\")\n",
    "            print(f\"  📊 Average hand strength: {avg_hand_strength:.2f}\")\n",
    "    \n",
    "    # Create DataFrame\n",
    "    df_simulations = pd.DataFrame(simulation_data)\n",
    "    \n",
    "    print(\"\\n📊 Simulation Results Summary\")\n",
    "    print(\"=\" * 50)\n",
    "    \n",
    "    # Summary statistics by profile and risk ratio\n",
    "    summary_stats = df_simulations.groupby(['Profile', 'Risk Ratio']).agg({\n",
    "        'Ordered Up': 'mean',\n",
    "        'Hand Strength': 'mean'\n",
    "    }).round(3)\n",
    "    \n",
    "    summary_stats.columns = ['Order Up Rate', 'Avg Hand Strength']\n",
    "    display(summary_stats)\n",
    "    \n",
    "    # Pivot table for order up rates\n",
    "    order_up_pivot = df_simulations.pivot_table(\n",
    "        index='Profile',\n",
    "        columns='Risk Ratio',\n",
    "        values='Ordered Up',\n",
    "        aggfunc='mean'\n",
    "    ).round(3)\n",
    "    \n",
    "    print(\"\\n📋 Order Up Rates by Profile and Risk Ratio\")\n",
    "    display(order_up_pivot)\n",
    "    \n",
    "    return df_simulations\n",
    "\n",
    "# Run the analysis\n",
    "simulation_results = run_statistical_analysis()"
        ]
    })
    
    # Visualization
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Visualization\n",
            "\n",
            "Let's create visualizations to better understand the AI behavior patterns."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create visualizations\n",
            "def create_visualizations(df_simulations: pd.DataFrame):\n",
            "    \"\"\"Create visualizations of AI behavior patterns.\"\"\"\n",
    "    \n",
    "    print(\"📈 Creating Visualizations\")\n",
    "    print(\"=\" * 50)\n",
    "    \n",
    "    # Set up the plotting style\n",
    "    plt.style.use('default')\n",
    "    fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n",
    "    fig.suptitle('Euchre AI Behavior Analysis', fontsize=16, fontweight='bold')\n",
    "    \n",
    "    # Plot 1: Order Up Rates by Profile and Risk Ratio\n",
    "    ax1 = axes[0, 0]\n",
    "    order_up_pivot = df_simulations.pivot_table(\n",
    "        index='Profile',\n",
    "        columns='Risk Ratio',\n",
    "        values='Ordered Up',\n",
    "        aggfunc='mean'\n",
    "    )\n",
    "    \n",
    "    order_up_pivot.plot(kind='bar', ax=ax1, width=0.8)\n",
    "    ax1.set_title('Order Up Rates by Profile and Risk Ratio')\n",
    "    ax1.set_ylabel('Order Up Rate')\n",
    "    ax1.set_xlabel('AI Profile')\n",
    "    ax1.legend(title='Risk Ratio', bbox_to_anchor=(1.05, 1), loc='upper left')\n",
    "    ax1.tick_params(axis='x', rotation=45)\n",
    "    \n",
    "    # Plot 2: Hand Strength Distribution\n",
    "    ax2 = axes[0, 1]\n",
    "    for profile in df_simulations['Profile'].unique():\n",
    "        profile_data = df_simulations[df_simulations['Profile'] == profile]\n",
    "        ax2.hist(profile_data['Hand Strength'], alpha=0.6, label=profile, bins=20)\n",
    "    \n",
    "    ax2.set_title('Hand Strength Distribution by Profile')\n",
    "    ax2.set_xlabel('Hand Strength')\n",
    "    ax2.set_ylabel('Frequency')\n",
    "    ax2.legend()\n",
    "    \n",
    "    # Plot 3: Risk Ratio vs Order Up Rate\n",
    "    ax3 = axes[1, 0]\n",
    "    for profile in df_simulations['Profile'].unique():\n",
    "        profile_data = df_simulations[df_simulations['Profile'] == profile]\n",
    "        risk_vs_order = profile_data.groupby('Risk Ratio')['Ordered Up'].mean()\n",
    "        ax3.plot(risk_vs_order.index, risk_vs_order.values, 'o-', label=profile, linewidth=2, markersize=6)\n",
    "    \n",
    "    ax3.set_title('Risk Ratio vs Order Up Rate')\n",
    "    ax3.set_xlabel('Risk Ratio')\n",
    "    ax3.set_ylabel('Order Up Rate')\n",
    "    ax3.legend()\n",
    "    ax3.grid(True, alpha=0.3)\n",
    "    \n",
    "    # Plot 4: Decision Heatmap\n",
    "    ax4 = axes[1, 1]\n",
    "    decision_pivot = df_simulations.pivot_table(\n",
    "        index='Profile',\n",
    "        columns='Risk Ratio',\n",
    "        values='Ordered Up',\n",
    "        aggfunc='mean'\n",
    "    )\n",
    "    \n",
    "    im = ax4.imshow(decision_pivot.values, cmap='RdYlBu_r', aspect='auto')\n",
    "    ax4.set_title('Decision Heatmap (Order Up Rate)')\n",
    "    ax4.set_xlabel('Risk Ratio')\n",
    "    ax4.set_ylabel('AI Profile')\n",
    "    ax4.set_xticks(range(len(decision_pivot.columns)))\n",
    "    ax4.set_xticklabels([f'{r:.1f}' for r in decision_pivot.columns])\n",
    "    ax4.set_yticks(range(len(decision_pivot.index)))\n",
    "    ax4.set_yticklabels(decision_pivot.index)\n",
    "    \n",
    "    # Add colorbar\n",
    "    cbar = plt.colorbar(im, ax=ax4)\n",
    "    cbar.set_label('Order Up Rate')\n",
    "    \n",
    "    # Add text annotations to heatmap\n",
    "    for i in range(len(decision_pivot.index)):\n",
    "        for j in range(len(decision_pivot.columns)):\n",
    "            text = ax4.text(j, i, f'{decision_pivot.iloc[i, j]:.2f}',\n",
    "                           ha='center', va='center', color='black', fontweight='bold')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    # Additional analysis: Correlation between hand strength and decisions\n",
    "    print(\"\\n📊 Correlation Analysis\")\n",
    "    print(\"-\" * 30)\n",
    "    \n",
    "    correlation_data = df_simulations.groupby(['Profile', 'Risk Ratio']).agg({\n",
    "        'Hand Strength': 'corr',\n",
    "        'Ordered Up': 'corr'\n",
    "    }).round(3)\n",
    "    \n",
    "    print(\"Correlation between Hand Strength and Order Up decisions:\")\n",
    "    display(correlation_data)\n",
    "\n",
    "# Create visualizations\n",
    "if 'simulation_results' in locals():\n",
    "    create_visualizations(simulation_results)\n",
    "else:\n",
    "    print(\"⚠️  Run the statistical analysis first to generate visualizations.\")"
        ]
    })
    
    # Conclusion
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Conclusion\n",
            "\n",
            "This notebook has demonstrated:\n",
            "\n",
            "1. **AI Profile Differences**: How Aggressive, Conservative, Balanced, and Opportunistic AIs make decisions\n",
            "2. **Risk Ratio Effects**: How varying risk ratios (0.1 to 0.9) affect decision-making\n",
            "3. **Decision Scenarios**: Trump calling, card selection, and trump suit selection\n",
            "4. **Statistical Analysis**: Performance patterns across multiple simulations\n",
            "5. **Visual Insights**: Charts showing behavior patterns and correlations\n",
            "\n",
            "The 'dumb' AIs use rule-based logic with configurable risk parameters to make strategic decisions in Euchre."
        ]
    })
    
    return cells

def create_notebook() -> Dict[str, Any]:
    """Create the complete notebook structure."""
    
    notebook = {
        "cells": create_notebook_cells(),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook

def main():
    """Main function to create and save the notebook."""
    
    print("🤖 Creating Euchre AI Demonstration Notebook...")
    
    # Create the notebook
    notebook = create_notebook()
    
    # Save to file
    output_file = "euchre_ai_demo.ipynb"
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Notebook created successfully: {output_file}")
    print(f"📊 Total cells: {len(notebook['cells'])}")
    print(f"🎯 Notebook includes:")
    print(f"  - AI profile demonstrations")
    print(f"  - Risk ratio analysis")
    print(f"  - Statistical simulations")
    print(f"  - Data visualizations")
    print(f"  - Decision-making scenarios")

if __name__ == "__main__":
    main() 