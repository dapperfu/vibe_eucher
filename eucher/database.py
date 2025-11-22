"""Database setup and connection for PonyORM."""

from pathlib import Path
from typing import Optional

from pony.orm import Database

# Global database instance
db = Database()


def init_database(database_path: Optional[str] = None, create_tables: bool = True) -> None:
    """
    Initialize the database connection.

    Parameters
    ----------
    database_path : Optional[str]
        Path to SQLite database file. If None, uses default "euchre.db" in current directory.
    create_tables : bool
        Whether to create tables if they don't exist. Default is True.
    """
    if database_path is None:
        database_path = "euchre.db"

    # Convert to Path for easier handling
    db_path = Path(database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Bind to SQLite database
    db.bind(
        provider="sqlite",
        filename=str(db_path),
        create_db=True,
    )

    # Import models to register them with the database
    from eucher.models import (  # noqa: F401
        CardPlay,
        Game,
        GamePlayer,
        Hand,
        Player,
        PlayerHand,
        Trick,
        TrumpDecision,
    )

    # Generate mapping and create tables
    db.generate_mapping(create_tables=create_tables)


def get_db() -> Database:
    """
    Get the global database instance.

    Returns
    -------
    Database
        The global PonyORM database instance.
    """
    return db


def close_database() -> None:
    """
    Close the database connection.

    This should be called when the application is shutting down.
    """
    db.disconnect()
