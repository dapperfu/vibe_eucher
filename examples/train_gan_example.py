"""Example script for training GAN models.

IMPORTANT: The GAN backend REQUIRES training before use. Without training,
the GAN model will use random weights and make essentially random decisions.
This script trains the GAN model on collected training data.
"""

from pathlib import Path

from src.ml_config import MLConfig
from src.training.train_gan import train_gan_for_decision_type


def main() -> None:
    """Example: Train GAN model for card play decisions.
    
    This script trains a GAN model on training data. The trained model
    is saved to models/gan_model.pth and will be automatically loaded
    when creating MLPlayer instances with backend="gan".
    
    Note: Training data must exist in the training_data/ directory
    (e.g., training_play_card.json) before running this script.
    """
    config = MLConfig()

    print("Training GAN model for play_card decisions...")
    print("=" * 60)

    model = train_gan_for_decision_type(
        data_dir=config.training_data_dir,
        decision_type="play_card",
        config=config,
        num_epochs=100,
        output_dir=config.models_dir,
        prefix="training",
    )

    print("\nGAN training complete!")
    print(f"Model saved to: {config.models_dir / 'gan_model.pth'}")


if __name__ == "__main__":
    main()

