# Model Input Features Documentation

This document describes all 260 input features used by the ML models.

## Feature Overview

| Category | Count | Description |
|----------|-------|-------------|
| Hand Encoding | 120 | 5 cards × 24 one-hot encodings (player's own hand only) |
| Turned Card | 24 | One-hot encoding of turned card |
| Trump Suit | 4 | One-hot encoding of trump suit |
| Led Suit | 4 | One-hot encoding of led suit |
| Trick Cards | 72 | 3 cards × 24 one-hot encodings (current trick) |
| Played Cards | 24 | Binary encoding of cards played in previous tricks |
| Positional | 12 | Player position, dealer, team, trick info |
| **Total** | **260** | |

## Important Notes

- **Hand Encoding**: Only includes the player's own hand. The AI cannot see other players' hands during gameplay, so training data must only include information visible to the player making the decision.
- **Played Cards**: Tracks cards that have been played in previous tricks (excluding current trick and player's hand). This helps the model understand which cards are still available in the deck.

## Hand Encoding

**Note**: Only includes the player's own hand. The player cannot see other players' hands during gameplay.

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 0 | `hand_card_0_is_NINE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is NINE_HEARTS, 0 otherwise |
| 1 | `hand_card_0_is_TEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is TEN_HEARTS, 0 otherwise |
| 2 | `hand_card_0_is_JACK_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is JACK_HEARTS, 0 otherwise |
| 3 | `hand_card_0_is_QUEEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is QUEEN_HEARTS, 0 otherwise |
| 4 | `hand_card_0_is_KING_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is KING_HEARTS, 0 otherwise |
| 5 | `hand_card_0_is_ACE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is ACE_HEARTS, 0 otherwise |
| 6 | `hand_card_0_is_NINE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is NINE_DIAMONDS, 0 otherwise |
| 7 | `hand_card_0_is_TEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is TEN_DIAMONDS, 0 otherwise |
| 8 | `hand_card_0_is_JACK_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is JACK_DIAMONDS, 0 otherwise |
| 9 | `hand_card_0_is_QUEEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is QUEEN_DIAMONDS, 0 otherwise |
| 10 | `hand_card_0_is_KING_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is KING_DIAMONDS, 0 otherwise |
| 11 | `hand_card_0_is_ACE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is ACE_DIAMONDS, 0 otherwise |
| 12 | `hand_card_0_is_NINE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is NINE_CLUBS, 0 otherwise |
| 13 | `hand_card_0_is_TEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is TEN_CLUBS, 0 otherwise |
| 14 | `hand_card_0_is_JACK_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is JACK_CLUBS, 0 otherwise |
| 15 | `hand_card_0_is_QUEEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is QUEEN_CLUBS, 0 otherwise |
| 16 | `hand_card_0_is_KING_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is KING_CLUBS, 0 otherwise |
| 17 | `hand_card_0_is_ACE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is ACE_CLUBS, 0 otherwise |
| 18 | `hand_card_0_is_NINE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is NINE_SPADES, 0 otherwise |
| 19 | `hand_card_0_is_TEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is TEN_SPADES, 0 otherwise |
| 20 | `hand_card_0_is_JACK_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is JACK_SPADES, 0 otherwise |
| 21 | `hand_card_0_is_QUEEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is QUEEN_SPADES, 0 otherwise |
| 22 | `hand_card_0_is_KING_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is KING_SPADES, 0 otherwise |
| 23 | `hand_card_0_is_ACE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 0 in hand is ACE_SPADES, 0 otherwise |
| 24 | `hand_card_1_is_NINE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is NINE_HEARTS, 0 otherwise |
| 25 | `hand_card_1_is_TEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is TEN_HEARTS, 0 otherwise |
| 26 | `hand_card_1_is_JACK_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is JACK_HEARTS, 0 otherwise |
| 27 | `hand_card_1_is_QUEEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is QUEEN_HEARTS, 0 otherwise |
| 28 | `hand_card_1_is_KING_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is KING_HEARTS, 0 otherwise |
| 29 | `hand_card_1_is_ACE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is ACE_HEARTS, 0 otherwise |
| 30 | `hand_card_1_is_NINE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is NINE_DIAMONDS, 0 otherwise |
| 31 | `hand_card_1_is_TEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is TEN_DIAMONDS, 0 otherwise |
| 32 | `hand_card_1_is_JACK_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is JACK_DIAMONDS, 0 otherwise |
| 33 | `hand_card_1_is_QUEEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is QUEEN_DIAMONDS, 0 otherwise |
| 34 | `hand_card_1_is_KING_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is KING_DIAMONDS, 0 otherwise |
| 35 | `hand_card_1_is_ACE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is ACE_DIAMONDS, 0 otherwise |
| 36 | `hand_card_1_is_NINE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is NINE_CLUBS, 0 otherwise |
| 37 | `hand_card_1_is_TEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is TEN_CLUBS, 0 otherwise |
| 38 | `hand_card_1_is_JACK_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is JACK_CLUBS, 0 otherwise |
| 39 | `hand_card_1_is_QUEEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is QUEEN_CLUBS, 0 otherwise |
| 40 | `hand_card_1_is_KING_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is KING_CLUBS, 0 otherwise |
| 41 | `hand_card_1_is_ACE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is ACE_CLUBS, 0 otherwise |
| 42 | `hand_card_1_is_NINE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is NINE_SPADES, 0 otherwise |
| 43 | `hand_card_1_is_TEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is TEN_SPADES, 0 otherwise |
| 44 | `hand_card_1_is_JACK_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is JACK_SPADES, 0 otherwise |
| 45 | `hand_card_1_is_QUEEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is QUEEN_SPADES, 0 otherwise |
| 46 | `hand_card_1_is_KING_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is KING_SPADES, 0 otherwise |
| 47 | `hand_card_1_is_ACE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 1 in hand is ACE_SPADES, 0 otherwise |
| 48 | `hand_card_2_is_NINE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is NINE_HEARTS, 0 otherwise |
| 49 | `hand_card_2_is_TEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is TEN_HEARTS, 0 otherwise |
| 50 | `hand_card_2_is_JACK_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is JACK_HEARTS, 0 otherwise |
| 51 | `hand_card_2_is_QUEEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is QUEEN_HEARTS, 0 otherwise |
| 52 | `hand_card_2_is_KING_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is KING_HEARTS, 0 otherwise |
| 53 | `hand_card_2_is_ACE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is ACE_HEARTS, 0 otherwise |
| 54 | `hand_card_2_is_NINE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is NINE_DIAMONDS, 0 otherwise |
| 55 | `hand_card_2_is_TEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is TEN_DIAMONDS, 0 otherwise |
| 56 | `hand_card_2_is_JACK_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is JACK_DIAMONDS, 0 otherwise |
| 57 | `hand_card_2_is_QUEEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is QUEEN_DIAMONDS, 0 otherwise |
| 58 | `hand_card_2_is_KING_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is KING_DIAMONDS, 0 otherwise |
| 59 | `hand_card_2_is_ACE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is ACE_DIAMONDS, 0 otherwise |
| 60 | `hand_card_2_is_NINE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is NINE_CLUBS, 0 otherwise |
| 61 | `hand_card_2_is_TEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is TEN_CLUBS, 0 otherwise |
| 62 | `hand_card_2_is_JACK_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is JACK_CLUBS, 0 otherwise |
| 63 | `hand_card_2_is_QUEEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is QUEEN_CLUBS, 0 otherwise |
| 64 | `hand_card_2_is_KING_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is KING_CLUBS, 0 otherwise |
| 65 | `hand_card_2_is_ACE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is ACE_CLUBS, 0 otherwise |
| 66 | `hand_card_2_is_NINE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is NINE_SPADES, 0 otherwise |
| 67 | `hand_card_2_is_TEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is TEN_SPADES, 0 otherwise |
| 68 | `hand_card_2_is_JACK_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is JACK_SPADES, 0 otherwise |
| 69 | `hand_card_2_is_QUEEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is QUEEN_SPADES, 0 otherwise |
| 70 | `hand_card_2_is_KING_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is KING_SPADES, 0 otherwise |
| 71 | `hand_card_2_is_ACE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 2 in hand is ACE_SPADES, 0 otherwise |
| 72 | `hand_card_3_is_NINE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is NINE_HEARTS, 0 otherwise |
| 73 | `hand_card_3_is_TEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is TEN_HEARTS, 0 otherwise |
| 74 | `hand_card_3_is_JACK_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is JACK_HEARTS, 0 otherwise |
| 75 | `hand_card_3_is_QUEEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is QUEEN_HEARTS, 0 otherwise |
| 76 | `hand_card_3_is_KING_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is KING_HEARTS, 0 otherwise |
| 77 | `hand_card_3_is_ACE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is ACE_HEARTS, 0 otherwise |
| 78 | `hand_card_3_is_NINE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is NINE_DIAMONDS, 0 otherwise |
| 79 | `hand_card_3_is_TEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is TEN_DIAMONDS, 0 otherwise |
| 80 | `hand_card_3_is_JACK_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is JACK_DIAMONDS, 0 otherwise |
| 81 | `hand_card_3_is_QUEEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is QUEEN_DIAMONDS, 0 otherwise |
| 82 | `hand_card_3_is_KING_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is KING_DIAMONDS, 0 otherwise |
| 83 | `hand_card_3_is_ACE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is ACE_DIAMONDS, 0 otherwise |
| 84 | `hand_card_3_is_NINE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is NINE_CLUBS, 0 otherwise |
| 85 | `hand_card_3_is_TEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is TEN_CLUBS, 0 otherwise |
| 86 | `hand_card_3_is_JACK_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is JACK_CLUBS, 0 otherwise |
| 87 | `hand_card_3_is_QUEEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is QUEEN_CLUBS, 0 otherwise |
| 88 | `hand_card_3_is_KING_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is KING_CLUBS, 0 otherwise |
| 89 | `hand_card_3_is_ACE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is ACE_CLUBS, 0 otherwise |
| 90 | `hand_card_3_is_NINE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is NINE_SPADES, 0 otherwise |
| 91 | `hand_card_3_is_TEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is TEN_SPADES, 0 otherwise |
| 92 | `hand_card_3_is_JACK_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is JACK_SPADES, 0 otherwise |
| 93 | `hand_card_3_is_QUEEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is QUEEN_SPADES, 0 otherwise |
| 94 | `hand_card_3_is_KING_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is KING_SPADES, 0 otherwise |
| 95 | `hand_card_3_is_ACE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 3 in hand is ACE_SPADES, 0 otherwise |
| 96 | `hand_card_4_is_NINE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is NINE_HEARTS, 0 otherwise |
| 97 | `hand_card_4_is_TEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is TEN_HEARTS, 0 otherwise |
| 98 | `hand_card_4_is_JACK_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is JACK_HEARTS, 0 otherwise |
| 99 | `hand_card_4_is_QUEEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is QUEEN_HEARTS, 0 otherwise |
| 100 | `hand_card_4_is_KING_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is KING_HEARTS, 0 otherwise |
| 101 | `hand_card_4_is_ACE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is ACE_HEARTS, 0 otherwise |
| 102 | `hand_card_4_is_NINE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is NINE_DIAMONDS, 0 otherwise |
| 103 | `hand_card_4_is_TEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is TEN_DIAMONDS, 0 otherwise |
| 104 | `hand_card_4_is_JACK_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is JACK_DIAMONDS, 0 otherwise |
| 105 | `hand_card_4_is_QUEEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is QUEEN_DIAMONDS, 0 otherwise |
| 106 | `hand_card_4_is_KING_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is KING_DIAMONDS, 0 otherwise |
| 107 | `hand_card_4_is_ACE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is ACE_DIAMONDS, 0 otherwise |
| 108 | `hand_card_4_is_NINE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is NINE_CLUBS, 0 otherwise |
| 109 | `hand_card_4_is_TEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is TEN_CLUBS, 0 otherwise |
| 110 | `hand_card_4_is_JACK_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is JACK_CLUBS, 0 otherwise |
| 111 | `hand_card_4_is_QUEEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is QUEEN_CLUBS, 0 otherwise |
| 112 | `hand_card_4_is_KING_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is KING_CLUBS, 0 otherwise |
| 113 | `hand_card_4_is_ACE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is ACE_CLUBS, 0 otherwise |
| 114 | `hand_card_4_is_NINE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is NINE_SPADES, 0 otherwise |
| 115 | `hand_card_4_is_TEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is TEN_SPADES, 0 otherwise |
| 116 | `hand_card_4_is_JACK_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is JACK_SPADES, 0 otherwise |
| 117 | `hand_card_4_is_QUEEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is QUEEN_SPADES, 0 otherwise |
| 118 | `hand_card_4_is_KING_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is KING_SPADES, 0 otherwise |
| 119 | `hand_card_4_is_ACE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if card position 4 in hand is ACE_SPADES, 0 otherwise |

## Turned Card

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 120 | `turned_card_is_NINE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if turned card is NINE_HEARTS, 0 otherwise |
| 121 | `turned_card_is_TEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if turned card is TEN_HEARTS, 0 otherwise |
| 122 | `turned_card_is_JACK_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if turned card is JACK_HEARTS, 0 otherwise |
| 123 | `turned_card_is_QUEEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if turned card is QUEEN_HEARTS, 0 otherwise |
| 124 | `turned_card_is_KING_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if turned card is KING_HEARTS, 0 otherwise |
| 125 | `turned_card_is_ACE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if turned card is ACE_HEARTS, 0 otherwise |
| 126 | `turned_card_is_NINE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if turned card is NINE_DIAMONDS, 0 otherwise |
| 127 | `turned_card_is_TEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if turned card is TEN_DIAMONDS, 0 otherwise |
| 128 | `turned_card_is_JACK_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if turned card is JACK_DIAMONDS, 0 otherwise |
| 129 | `turned_card_is_QUEEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if turned card is QUEEN_DIAMONDS, 0 otherwise |
| 130 | `turned_card_is_KING_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if turned card is KING_DIAMONDS, 0 otherwise |
| 131 | `turned_card_is_ACE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if turned card is ACE_DIAMONDS, 0 otherwise |
| 132 | `turned_card_is_NINE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if turned card is NINE_CLUBS, 0 otherwise |
| 133 | `turned_card_is_TEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if turned card is TEN_CLUBS, 0 otherwise |
| 134 | `turned_card_is_JACK_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if turned card is JACK_CLUBS, 0 otherwise |
| 135 | `turned_card_is_QUEEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if turned card is QUEEN_CLUBS, 0 otherwise |
| 136 | `turned_card_is_KING_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if turned card is KING_CLUBS, 0 otherwise |
| 137 | `turned_card_is_ACE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if turned card is ACE_CLUBS, 0 otherwise |
| 138 | `turned_card_is_NINE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if turned card is NINE_SPADES, 0 otherwise |
| 139 | `turned_card_is_TEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if turned card is TEN_SPADES, 0 otherwise |
| 140 | `turned_card_is_JACK_SPADES` | binary | [0, 1] | One-hot encoding: 1 if turned card is JACK_SPADES, 0 otherwise |
| 141 | `turned_card_is_QUEEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if turned card is QUEEN_SPADES, 0 otherwise |
| 142 | `turned_card_is_KING_SPADES` | binary | [0, 1] | One-hot encoding: 1 if turned card is KING_SPADES, 0 otherwise |
| 143 | `turned_card_is_ACE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if turned card is ACE_SPADES, 0 otherwise |

## Trump Suit

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 144 | `trump_suit_is_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trump suit is HEARTS, 0 otherwise |
| 145 | `trump_suit_is_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trump suit is DIAMONDS, 0 otherwise |
| 146 | `trump_suit_is_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trump suit is CLUBS, 0 otherwise |
| 147 | `trump_suit_is_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trump suit is SPADES, 0 otherwise |

## Led Suit

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 148 | `led_suit_is_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if led suit is HEARTS, 0 otherwise |
| 149 | `led_suit_is_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if led suit is DIAMONDS, 0 otherwise |
| 150 | `led_suit_is_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if led suit is CLUBS, 0 otherwise |
| 151 | `led_suit_is_SPADES` | binary | [0, 1] | One-hot encoding: 1 if led suit is SPADES, 0 otherwise |

## Trick Cards

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 152 | `trick_card_0_is_NINE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is NINE_HEARTS, 0 otherwise |
| 153 | `trick_card_0_is_TEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is TEN_HEARTS, 0 otherwise |
| 154 | `trick_card_0_is_JACK_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is JACK_HEARTS, 0 otherwise |
| 155 | `trick_card_0_is_QUEEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is QUEEN_HEARTS, 0 otherwise |
| 156 | `trick_card_0_is_KING_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is KING_HEARTS, 0 otherwise |
| 157 | `trick_card_0_is_ACE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is ACE_HEARTS, 0 otherwise |
| 158 | `trick_card_0_is_NINE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is NINE_DIAMONDS, 0 otherwise |
| 159 | `trick_card_0_is_TEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is TEN_DIAMONDS, 0 otherwise |
| 160 | `trick_card_0_is_JACK_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is JACK_DIAMONDS, 0 otherwise |
| 161 | `trick_card_0_is_QUEEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is QUEEN_DIAMONDS, 0 otherwise |
| 162 | `trick_card_0_is_KING_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is KING_DIAMONDS, 0 otherwise |
| 163 | `trick_card_0_is_ACE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is ACE_DIAMONDS, 0 otherwise |
| 164 | `trick_card_0_is_NINE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is NINE_CLUBS, 0 otherwise |
| 165 | `trick_card_0_is_TEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is TEN_CLUBS, 0 otherwise |
| 166 | `trick_card_0_is_JACK_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is JACK_CLUBS, 0 otherwise |
| 167 | `trick_card_0_is_QUEEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is QUEEN_CLUBS, 0 otherwise |
| 168 | `trick_card_0_is_KING_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is KING_CLUBS, 0 otherwise |
| 169 | `trick_card_0_is_ACE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is ACE_CLUBS, 0 otherwise |
| 170 | `trick_card_0_is_NINE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is NINE_SPADES, 0 otherwise |
| 171 | `trick_card_0_is_TEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is TEN_SPADES, 0 otherwise |
| 172 | `trick_card_0_is_JACK_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is JACK_SPADES, 0 otherwise |
| 173 | `trick_card_0_is_QUEEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is QUEEN_SPADES, 0 otherwise |
| 174 | `trick_card_0_is_KING_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is KING_SPADES, 0 otherwise |
| 175 | `trick_card_0_is_ACE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 0 card is ACE_SPADES, 0 otherwise |
| 176 | `trick_card_1_is_NINE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is NINE_HEARTS, 0 otherwise |
| 177 | `trick_card_1_is_TEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is TEN_HEARTS, 0 otherwise |
| 178 | `trick_card_1_is_JACK_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is JACK_HEARTS, 0 otherwise |
| 179 | `trick_card_1_is_QUEEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is QUEEN_HEARTS, 0 otherwise |
| 180 | `trick_card_1_is_KING_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is KING_HEARTS, 0 otherwise |
| 181 | `trick_card_1_is_ACE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is ACE_HEARTS, 0 otherwise |
| 182 | `trick_card_1_is_NINE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is NINE_DIAMONDS, 0 otherwise |
| 183 | `trick_card_1_is_TEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is TEN_DIAMONDS, 0 otherwise |
| 184 | `trick_card_1_is_JACK_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is JACK_DIAMONDS, 0 otherwise |
| 185 | `trick_card_1_is_QUEEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is QUEEN_DIAMONDS, 0 otherwise |
| 186 | `trick_card_1_is_KING_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is KING_DIAMONDS, 0 otherwise |
| 187 | `trick_card_1_is_ACE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is ACE_DIAMONDS, 0 otherwise |
| 188 | `trick_card_1_is_NINE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is NINE_CLUBS, 0 otherwise |
| 189 | `trick_card_1_is_TEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is TEN_CLUBS, 0 otherwise |
| 190 | `trick_card_1_is_JACK_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is JACK_CLUBS, 0 otherwise |
| 191 | `trick_card_1_is_QUEEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is QUEEN_CLUBS, 0 otherwise |
| 192 | `trick_card_1_is_KING_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is KING_CLUBS, 0 otherwise |
| 193 | `trick_card_1_is_ACE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is ACE_CLUBS, 0 otherwise |
| 194 | `trick_card_1_is_NINE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is NINE_SPADES, 0 otherwise |
| 195 | `trick_card_1_is_TEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is TEN_SPADES, 0 otherwise |
| 196 | `trick_card_1_is_JACK_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is JACK_SPADES, 0 otherwise |
| 197 | `trick_card_1_is_QUEEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is QUEEN_SPADES, 0 otherwise |
| 198 | `trick_card_1_is_KING_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is KING_SPADES, 0 otherwise |
| 199 | `trick_card_1_is_ACE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 1 card is ACE_SPADES, 0 otherwise |
| 200 | `trick_card_2_is_NINE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is NINE_HEARTS, 0 otherwise |
| 201 | `trick_card_2_is_TEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is TEN_HEARTS, 0 otherwise |
| 202 | `trick_card_2_is_JACK_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is JACK_HEARTS, 0 otherwise |
| 203 | `trick_card_2_is_QUEEN_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is QUEEN_HEARTS, 0 otherwise |
| 204 | `trick_card_2_is_KING_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is KING_HEARTS, 0 otherwise |
| 205 | `trick_card_2_is_ACE_HEARTS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is ACE_HEARTS, 0 otherwise |
| 206 | `trick_card_2_is_NINE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is NINE_DIAMONDS, 0 otherwise |
| 207 | `trick_card_2_is_TEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is TEN_DIAMONDS, 0 otherwise |
| 208 | `trick_card_2_is_JACK_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is JACK_DIAMONDS, 0 otherwise |
| 209 | `trick_card_2_is_QUEEN_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is QUEEN_DIAMONDS, 0 otherwise |
| 210 | `trick_card_2_is_KING_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is KING_DIAMONDS, 0 otherwise |
| 211 | `trick_card_2_is_ACE_DIAMONDS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is ACE_DIAMONDS, 0 otherwise |
| 212 | `trick_card_2_is_NINE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is NINE_CLUBS, 0 otherwise |
| 213 | `trick_card_2_is_TEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is TEN_CLUBS, 0 otherwise |
| 214 | `trick_card_2_is_JACK_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is JACK_CLUBS, 0 otherwise |
| 215 | `trick_card_2_is_QUEEN_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is QUEEN_CLUBS, 0 otherwise |
| 216 | `trick_card_2_is_KING_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is KING_CLUBS, 0 otherwise |
| 217 | `trick_card_2_is_ACE_CLUBS` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is ACE_CLUBS, 0 otherwise |
| 218 | `trick_card_2_is_NINE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is NINE_SPADES, 0 otherwise |
| 219 | `trick_card_2_is_TEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is TEN_SPADES, 0 otherwise |
| 220 | `trick_card_2_is_JACK_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is JACK_SPADES, 0 otherwise |
| 221 | `trick_card_2_is_QUEEN_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is QUEEN_SPADES, 0 otherwise |
| 222 | `trick_card_2_is_KING_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is KING_SPADES, 0 otherwise |
| 223 | `trick_card_2_is_ACE_SPADES` | binary | [0, 1] | One-hot encoding: 1 if trick position 2 card is ACE_SPADES, 0 otherwise |

## Played Cards

Cards that have been played in previous tricks (excluding current trick and player's hand). This is a binary vector where 1 indicates the card was played in a previous trick.

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 224 | `played_card_is_NINE_HEARTS` | binary | [0, 1] | Binary: 1 if NINE_HEARTS was played in a previous trick, 0 otherwise |
| 225 | `played_card_is_TEN_HEARTS` | binary | [0, 1] | Binary: 1 if TEN_HEARTS was played in a previous trick, 0 otherwise |
| 226 | `played_card_is_JACK_HEARTS` | binary | [0, 1] | Binary: 1 if JACK_HEARTS was played in a previous trick, 0 otherwise |
| 227 | `played_card_is_QUEEN_HEARTS` | binary | [0, 1] | Binary: 1 if QUEEN_HEARTS was played in a previous trick, 0 otherwise |
| 228 | `played_card_is_KING_HEARTS` | binary | [0, 1] | Binary: 1 if KING_HEARTS was played in a previous trick, 0 otherwise |
| 229 | `played_card_is_ACE_HEARTS` | binary | [0, 1] | Binary: 1 if ACE_HEARTS was played in a previous trick, 0 otherwise |
| 230 | `played_card_is_NINE_DIAMONDS` | binary | [0, 1] | Binary: 1 if NINE_DIAMONDS was played in a previous trick, 0 otherwise |
| 231 | `played_card_is_TEN_DIAMONDS` | binary | [0, 1] | Binary: 1 if TEN_DIAMONDS was played in a previous trick, 0 otherwise |
| 232 | `played_card_is_JACK_DIAMONDS` | binary | [0, 1] | Binary: 1 if JACK_DIAMONDS was played in a previous trick, 0 otherwise |
| 233 | `played_card_is_QUEEN_DIAMONDS` | binary | [0, 1] | Binary: 1 if QUEEN_DIAMONDS was played in a previous trick, 0 otherwise |
| 234 | `played_card_is_KING_DIAMONDS` | binary | [0, 1] | Binary: 1 if KING_DIAMONDS was played in a previous trick, 0 otherwise |
| 235 | `played_card_is_ACE_DIAMONDS` | binary | [0, 1] | Binary: 1 if ACE_DIAMONDS was played in a previous trick, 0 otherwise |
| 236 | `played_card_is_NINE_CLUBS` | binary | [0, 1] | Binary: 1 if NINE_CLUBS was played in a previous trick, 0 otherwise |
| 237 | `played_card_is_TEN_CLUBS` | binary | [0, 1] | Binary: 1 if TEN_CLUBS was played in a previous trick, 0 otherwise |
| 238 | `played_card_is_JACK_CLUBS` | binary | [0, 1] | Binary: 1 if JACK_CLUBS was played in a previous trick, 0 otherwise |
| 239 | `played_card_is_QUEEN_CLUBS` | binary | [0, 1] | Binary: 1 if QUEEN_CLUBS was played in a previous trick, 0 otherwise |
| 240 | `played_card_is_KING_CLUBS` | binary | [0, 1] | Binary: 1 if KING_CLUBS was played in a previous trick, 0 otherwise |
| 241 | `played_card_is_ACE_CLUBS` | binary | [0, 1] | Binary: 1 if ACE_CLUBS was played in a previous trick, 0 otherwise |
| 242 | `played_card_is_NINE_SPADES` | binary | [0, 1] | Binary: 1 if NINE_SPADES was played in a previous trick, 0 otherwise |
| 243 | `played_card_is_TEN_SPADES` | binary | [0, 1] | Binary: 1 if TEN_SPADES was played in a previous trick, 0 otherwise |
| 244 | `played_card_is_JACK_SPADES` | binary | [0, 1] | Binary: 1 if JACK_SPADES was played in a previous trick, 0 otherwise |
| 245 | `played_card_is_QUEEN_SPADES` | binary | [0, 1] | Binary: 1 if QUEEN_SPADES was played in a previous trick, 0 otherwise |
| 246 | `played_card_is_KING_SPADES` | binary | [0, 1] | Binary: 1 if KING_SPADES was played in a previous trick, 0 otherwise |
| 247 | `played_card_is_ACE_SPADES` | binary | [0, 1] | Binary: 1 if ACE_SPADES was played in a previous trick, 0 otherwise |

## Positional

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 248 | `player_id_0` | binary | [0, 1] | Player ID one-hot: 1 if player is player 0 |
| 249 | `player_id_1` | binary | [0, 1] | Player ID one-hot: 1 if player is player 1 |
| 250 | `player_id_2` | binary | [0, 1] | Player ID one-hot: 1 if player is player 2 |
| 251 | `player_id_3` | binary | [0, 1] | Player ID one-hot: 1 if player is player 3 |
| 252 | `dealer_id_0` | binary | [0, 1] | Dealer ID one-hot: 1 if dealer is player 0 |
| 253 | `dealer_id_1` | binary | [0, 1] | Dealer ID one-hot: 1 if dealer is player 1 |
| 254 | `dealer_id_2` | binary | [0, 1] | Dealer ID one-hot: 1 if dealer is player 2 |
| 255 | `dealer_id_3` | binary | [0, 1] | Dealer ID one-hot: 1 if dealer is player 3 |
| 256 | `team` | binary | [0, 1] | Team binary: 0 for team 0, 1 for team 1 |
| 257 | `trick_number_normalized` | continuous | [0, 1] | Trick number normalized: current_trick / 4.0, range [0, 1] |
| 258 | `tricks_won_team0_normalized` | continuous | [0, 1] | Tricks won by team 0 normalized: tricks_won / 5.0, range [0, 1] |
| 259 | `tricks_won_team1_normalized` | continuous | [0, 1] | Tricks won by team 1 normalized: tricks_won / 5.0, range [0, 1] |

## Decision Weight System

The ML models output decision weights (confidence scores) for each possible action.
Temperature thresholds are applied to filter actions:

- **Low temperature (0.0-0.3)**: Conservative play (only high-confidence actions)
- **Medium temperature (0.3-0.7)**: Balanced play
- **High temperature (0.7-1.0)**: Risky play (accepts lower-confidence actions)

Actions with weight >= temperature threshold are considered valid.
Separate risk factors are used for:

- **Trump Selection**: `trump_selection_risk` (order up, call trump decisions)
- **Gameplay**: `gameplay_risk` (play card, discard decisions)

