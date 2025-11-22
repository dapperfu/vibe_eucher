"""Structured logging for training runs."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainingLogger:
    """Structured logger for training runs."""

    def __init__(self, log_dir: Optional[Path] = None, log_level: int = logging.INFO) -> None:
        """
        Initialize the training logger.

        Parameters
        ----------
        log_dir : Optional[Path]
            Directory for log files. If None, uses default.
        log_level : int
            Logging level (default INFO).
        """
        self.log_dir = log_dir or Path("training_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up Python logger
        self.logger = logging.getLogger("training")
        self.logger.setLevel(log_level)

        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        # Structured log storage
        self.structured_logs: List[Dict[str, Any]] = []

    def log(self, level: int, message: str, **kwargs: Any) -> None:
        """
        Log a message with optional structured data.

        Parameters
        ----------
        level : int
            Logging level (logging.INFO, logging.WARNING, etc.).
        message : str
            Log message.
        **kwargs : Any
            Additional structured data to include.
        """
        self.logger.log(level, message)

        # Store structured log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": logging.getLevelName(level),
            "message": message,
            **kwargs,
        }
        self.structured_logs.append(log_entry)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self.log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self.log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self.log(logging.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self.log(logging.DEBUG, message, **kwargs)

    def log_training_start(self, config: Dict[str, Any]) -> None:
        """
        Log training start with configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Training configuration.
        """
        self.info("Training started", event="training_start", config=config)

    def log_training_end(self, summary: Dict[str, Any]) -> None:
        """
        Log training end with summary.

        Parameters
        ----------
        summary : Dict[str, Any]
            Training summary.
        """
        self.info("Training completed", event="training_end", summary=summary)

    def log_game_result(self, game_id: int, won: bool, metrics: Dict[str, Any]) -> None:
        """
        Log game result.

        Parameters
        ----------
        game_id : int
            Game identifier.
        won : bool
            Whether model/player won.
        metrics : Dict[str, Any]
            Game metrics.
        """
        self.info(
            f"Game {game_id}: {'Won' if won else 'Lost'}",
            event="game_result",
            game_id=game_id,
            won=won,
            **metrics,
        )

    def log_checkpoint(self, checkpoint_path: Path, game_id: int) -> None:
        """
        Log checkpoint save.

        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file.
        game_id : int
            Current game number.
        """
        self.info(
            f"Checkpoint saved: {checkpoint_path}",
            event="checkpoint",
            checkpoint_path=str(checkpoint_path),
            game_id=game_id,
        )

    def save_structured_logs(self, filename: Optional[str] = None) -> Path:
        """
        Save structured logs to JSON file.

        Parameters
        ----------
        filename : Optional[str]
            Filename for logs. If None, uses timestamp.

        Returns
        -------
        Path
            Path to saved log file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"structured_logs_{timestamp}.json"

        filepath = self.log_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.structured_logs, f, indent=2, default=str)

        return filepath

