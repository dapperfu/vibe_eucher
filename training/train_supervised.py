"""Train supervised learning models on collected game data."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from src.ml_config import MLConfig
from src.ml_models_supervised import (
    CallTrumpClassifier,
    DiscardCardClassifier,
    OrderUpClassifier,
    PlayCardClassifier,
)


def load_training_data(data_dir: Path, prefix: str = "training") -> Dict[str, np.ndarray]:
    """
    Load training data from JSON files.

    Parameters
    ----------
    data_dir : Path
        Directory containing training data files.
    prefix : str
        Prefix for data filenames.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with 'X' (features) and 'y' (labels) for each decision type.
    """
    order_up_file = data_dir / f"{prefix}_order_up.json"
    call_trump_file = data_dir / f"{prefix}_call_trump.json"
    play_card_file = data_dir / f"{prefix}_play_card.json"
    discard_file = data_dir / f"{prefix}_discard.json"

    data = {}

    # Load order up data
    if order_up_file.exists():
        with open(order_up_file, "r") as f:
            order_up_data = json.load(f)
        if order_up_data:
            X = np.array([item["features"] for item in order_up_data])
            y = np.array([item["decision"] for item in order_up_data])
            data["order_up"] = {"X": X, "y": y}

    # Load call trump data
    if call_trump_file.exists():
        with open(call_trump_file, "r") as f:
            call_trump_data = json.load(f)
        if call_trump_data:
            X = np.array([item["features"] for item in call_trump_data])
            y = np.array([item["decision"] for item in call_trump_data])
            data["call_trump"] = {"X": X, "y": y}

    # Load play card data
    if play_card_file.exists():
        with open(play_card_file, "r") as f:
            play_card_data = json.load(f)
        if play_card_data:
            X = np.array([item["features"] for item in play_card_data])
            y = np.array([item["decision"] for item in play_card_data])
            data["play_card"] = {"X": X, "y": y}

    # Load discard data
    if discard_file.exists():
        with open(discard_file, "r") as f:
            discard_data = json.load(f)
        if discard_data:
            X = np.array([item["features"] for item in discard_data])
            y = np.array([item["decision"] for item in discard_data])
            data["discard"] = {"X": X, "y": y}

    return data


def train_order_up_classifier(
    X: np.ndarray, y: np.ndarray, model_type: str = "random_forest", output_dir: Path = None
) -> OrderUpClassifier:
    """
    Train the order up classifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary labels.
    model_type : str
        Type of model to use.
    output_dir : Path
        Directory to save the trained model.

    Returns
    -------
    OrderUpClassifier
        Trained classifier.
    """
    print(f"Training OrderUpClassifier ({model_type})...")
    print(f"  Dataset size: {len(X)} samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    classifier = OrderUpClassifier(model_type)
    classifier.train(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Save model
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "order_up.pkl"
        classifier.save(model_path)
        print(f"  Saved model to {model_path}")

    return classifier


def train_call_trump_classifier(
    X: np.ndarray, y: np.ndarray, model_type: str = "random_forest", output_dir: Path = None
) -> CallTrumpClassifier:
    """
    Train the call trump classifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Class labels (0-4).
    model_type : str
        Type of model to use.
    output_dir : Path
        Directory to save the trained model.

    Returns
    -------
    CallTrumpClassifier
        Trained classifier.
    """
    print(f"Training CallTrumpClassifier ({model_type})...")
    print(f"  Dataset size: {len(X)} samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    classifier = CallTrumpClassifier(model_type)
    classifier.train(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "call_trump.pkl"
        classifier.save(model_path)
        print(f"  Saved model to {model_path}")

    return classifier


def train_play_card_classifier(
    X: np.ndarray, y: np.ndarray, model_type: str = "random_forest", output_dir: Path = None
) -> PlayCardClassifier:
    """
    Train the play card classifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Card index labels (0-23).
    model_type : str
        Type of model to use.
    output_dir : Path
        Directory to save the trained model.

    Returns
    -------
    PlayCardClassifier
        Trained classifier.
    """
    print(f"Training PlayCardClassifier ({model_type})...")
    print(f"  Dataset size: {len(X)} samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    classifier = PlayCardClassifier(model_type)
    classifier.train(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {accuracy:.4f}")

    # Save model
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "play_card.pkl"
        classifier.save(model_path)
        print(f"  Saved model to {model_path}")

    return classifier


def train_discard_classifier(
    X: np.ndarray, y: np.ndarray, model_type: str = "random_forest", output_dir: Path = None
) -> DiscardCardClassifier:
    """
    Train the discard card classifier.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Card index labels (0-23).
    model_type : str
        Type of model to use.
    output_dir : Path
        Directory to save the trained model.

    Returns
    -------
    DiscardCardClassifier
        Trained classifier.
    """
    print(f"Training DiscardCardClassifier ({model_type})...")
    print(f"  Dataset size: {len(X)} samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    classifier = DiscardCardClassifier(model_type)
    classifier.train(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {accuracy:.4f}")

    # Save model
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "discard.pkl"
        classifier.save(model_path)
        print(f"  Saved model to {model_path}")

    return classifier


def train_all_models(
    data_dir: Path,
    model_type: str = "random_forest",
    output_dir: Path = None,
    prefix: str = "training",
    show_progress: bool = True,
) -> Dict[str, any]:
    """
    Train all supervised learning models.

    Parameters
    ----------
    data_dir : Path
        Directory containing training data.
    model_type : str
        Type of model to use.
    output_dir : Path
        Directory to save trained models.
    prefix : str
        Prefix for data filenames.
    show_progress : bool
        Whether to show progress bars (default True).

    Returns
    -------
    Dict[str, any]
        Dictionary of trained classifiers.
    """
    if output_dir is None:
        config = MLConfig()
        output_dir = config.models_dir

    console = Console() if show_progress else None

    # Load data
    if console:
        console.print("[cyan]Loading training data...[/cyan]")
    else:
        print("Loading training data...")
    data = load_training_data(data_dir, prefix)

    classifiers = {}

    # Determine which models to train
    models_to_train = []
    if "order_up" in data:
        models_to_train.append(("order_up", "Order Up", train_order_up_classifier))
    if "call_trump" in data:
        models_to_train.append(("call_trump", "Call Trump", train_call_trump_classifier))
    if "play_card" in data:
        models_to_train.append(("play_card", "Play Card", train_play_card_classifier))
    if "discard" in data:
        models_to_train.append(("discard", "Discard", train_discard_classifier))

    # Train with progress bar
    if console and models_to_train:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Training models...", total=len(models_to_train))

            for model_key, model_name, train_func in models_to_train:
                progress.update(task, description=f"Training {model_name}...")
                classifiers[model_key] = train_func(
                    data[model_key]["X"], data[model_key]["y"], model_type, output_dir
                )
                progress.advance(task)
    else:
        # Train without progress bar
        for model_key, model_name, train_func in models_to_train:
            classifiers[model_key] = train_func(
                data[model_key]["X"], data[model_key]["y"], model_type, output_dir
            )

    if console:
        console.print("[green]Training complete![/green]")
    else:
        print("\nTraining complete!")
    return classifiers


if __name__ == "__main__":
    from pathlib import Path

    config = MLConfig()
    train_all_models(
        data_dir=config.training_data_dir,
        model_type="random_forest",
        output_dir=config.models_dir,
    )

