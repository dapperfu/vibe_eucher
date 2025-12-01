"""Supervised learning models using sklearn for Euchre decision making."""

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from eucher.cards import Card, Suit
from ..ml_config import MLConfig


class OrderUpClassifier:
    """Binary classifier for deciding whether to order up the turned card."""

    def __init__(self, model_type: str = "random_forest") -> None:
        """
        Initialize the order up classifier.

        Parameters
        ----------
        model_type : str
            Type of model: "random_forest", "gradient_boosting", or "neural_network"
        """
        self.model_type = model_type
        self.model = self._create_model()

    def _create_model(self):
        """Create the appropriate sklearn model."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif self.model_type == "neural_network":
            return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Binary labels (1 = order up, 0 = pass).
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict whether to order up.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Binary predictions (1 = order up, 0 = pass).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Probability array of shape (n_samples, 2).
        """
        return self.model.predict_proba(X)

    def save(self, filepath: Path) -> None:
        """
        Save the model to disk.

        Parameters
        ----------
        filepath : Path
            Path to save the model.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: Path) -> None:
        """
        Load the model from disk.

        Parameters
        ----------
        filepath : Path
            Path to load the model from.
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)


class CallTrumpClassifier:
    """Multi-class classifier for deciding which suit to call as trump."""

    def __init__(self, model_type: str = "random_forest") -> None:
        """
        Initialize the call trump classifier.

        Parameters
        ----------
        model_type : str
            Type of model: "random_forest", "gradient_boosting", or "neural_network"
        """
        self.model_type = model_type
        self.model = self._create_model()
        # Class mapping: 0 = pass, 1 = HEARTS, 2 = DIAMONDS, 3 = CLUBS, 4 = SPADES
        self.suit_classes = [None, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES]

    def _create_model(self):
        """Create the appropriate sklearn model."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif self.model_type == "neural_network":
            return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Class labels (0 = pass, 1-4 = suit indices).
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict which suit to call (or pass).

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Class predictions (0 = pass, 1-4 = suit indices).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Probability array of shape (n_samples, 5).
        """
        return self.model.predict_proba(X)

    def predict_suit(self, X: np.ndarray, forbidden_suit: Optional[Suit] = None) -> Optional[Suit]:
        """
        Predict suit to call, excluding forbidden suit.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        forbidden_suit : Optional[Suit]
            Suit that cannot be chosen (turned card suit).

        Returns
        -------
        Optional[Suit]
            Predicted suit to call, or None to pass.
        """
        probs = self.predict_proba(X)
        class_idx = np.argmax(probs[0])

        if class_idx == 0:
            return None  # Pass

        predicted_suit = self.suit_classes[class_idx]
        if predicted_suit == forbidden_suit:
            # Find next best suit
            sorted_indices = np.argsort(probs[0])[::-1]
            for idx in sorted_indices:
                if idx == 0:
                    continue  # Skip pass
                suit = self.suit_classes[idx]
                if suit != forbidden_suit:
                    return suit
            return None  # All suits forbidden, pass

        return predicted_suit

    def save(self, filepath: Path) -> None:
        """
        Save the model to disk.

        Parameters
        ----------
        filepath : Path
            Path to save the model.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: Path) -> None:
        """
        Load the model from disk.

        Parameters
        ----------
        filepath : Path
            Path to load the model from.
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)


class PlayCardClassifier:
    """Multi-class classifier for deciding which card to play."""

    def __init__(self, model_type: str = "random_forest") -> None:
        """
        Initialize the play card classifier.

        Parameters
        ----------
        model_type : str
            Type of model: "random_forest", "gradient_boosting", or "neural_network"
        """
        self.model_type = model_type
        self.model = self._create_model()
        # 24 possible cards in Euchre deck
        self.num_cards = 24

    def _create_model(self):
        """Create the appropriate sklearn model."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=100, max_depth=7, random_state=42)
        elif self.model_type == "neural_network":
            return MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Card indices (0-23).
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict which card to play.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Card index predictions (0-23).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Probability array of shape (n_samples, 24).
        """
        return self.model.predict_proba(X)

    def predict_card_from_valid(
        self, X: np.ndarray, valid_card_indices: List[int]
    ) -> int:
        """
        Predict card to play from valid cards only.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        valid_card_indices : List[int]
            List of valid card indices.

        Returns
        -------
        int
            Predicted card index from valid cards.
        """
        probs = self.predict_proba(X)
        # Mask invalid cards
        masked_probs = np.full(self.num_cards, -np.inf)
        for idx in valid_card_indices:
            masked_probs[idx] = probs[0][idx]

        return int(np.argmax(masked_probs))

    def save(self, filepath: Path) -> None:
        """
        Save the model to disk.

        Parameters
        ----------
        filepath : Path
            Path to save the model.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: Path) -> None:
        """
        Load the model from disk.

        Parameters
        ----------
        filepath : Path
            Path to load the model from.
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)


class DiscardCardClassifier:
    """Multi-class classifier for deciding which card to discard."""

    def __init__(self, model_type: str = "random_forest") -> None:
        """
        Initialize the discard card classifier.

        Parameters
        ----------
        model_type : str
            Type of model: "random_forest", "gradient_boosting", or "neural_network"
        """
        self.model_type = model_type
        self.model = self._create_model()
        # 24 possible cards in Euchre deck
        self.num_cards = 24

    def _create_model(self):
        """Create the appropriate sklearn model."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        elif self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif self.model_type == "neural_network":
            return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the classifier.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Card indices (0-23).
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict which card to discard.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Card index predictions (0-23).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Probability array of shape (n_samples, 24).
        """
        return self.model.predict_proba(X)

    def predict_card_from_hand(self, X: np.ndarray, hand_card_indices: List[int]) -> int:
        """
        Predict card to discard from hand.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        hand_card_indices : List[int]
            List of card indices in hand.

        Returns
        -------
        int
            Predicted card index from hand.
        """
        probs = self.predict_proba(X)
        # Mask cards not in hand
        masked_probs = np.full(self.num_cards, -np.inf)
        for idx in hand_card_indices:
            masked_probs[idx] = probs[0][idx]

        return int(np.argmax(masked_probs))

    def save(self, filepath: Path) -> None:
        """
        Save the model to disk.

        Parameters
        ----------
        filepath : Path
            Path to save the model.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath: Path) -> None:
        """
        Load the model from disk.

        Parameters
        ----------
        filepath : Path
            Path to load the model from.
        """
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)

