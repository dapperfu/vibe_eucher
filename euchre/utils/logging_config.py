"""
Logging configuration for the Euchre game.

This module provides centralized logging configuration with verbosity control
for different levels of detail in game output.
"""

import logging
import sys
from typing import Optional


class GameLogger:
    """Centralized logger for the Euchre game with verbosity control."""
    
    def __init__(self, verbose: bool = False, very_verbose: bool = False):
        """Initialize the game logger.
        
        Parameters
        ----------
        verbose : bool
            Enable verbose logging (INFO level)
        very_verbose : bool
            Enable very verbose logging (DEBUG level)
        """
        self.verbose = verbose
        self.very_verbose = very_verbose
        
        # Configure root logger
        self.logger = logging.getLogger('euchre')
        self.logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Set appropriate log level based on verbosity
        if very_verbose:
            console_handler.setLevel(logging.DEBUG)
        elif verbose:
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.WARNING)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
        
        # Create child loggers for different components
        self.game_logger = logging.getLogger('euchre.game')
        self.ai_logger = logging.getLogger('euchre.ai')
        self.core_logger = logging.getLogger('euchre.core')
        self.cli_logger = logging.getLogger('euchre.cli')
    
    def debug(self, message: str, component: str = 'game') -> None:
        """Log a debug message if very verbose mode is enabled.
        
        Parameters
        ----------
        message : str
            The debug message to log
        component : str
            The component logging the message ('game', 'ai', 'core', 'cli')
        """
        if self.very_verbose:
            if component == 'ai':
                self.ai_logger.debug(message)
            elif component == 'core':
                self.core_logger.debug(message)
            elif component == 'cli':
                self.cli_logger.debug(message)
            else:
                self.game_logger.debug(message)
    
    def info(self, message: str, component: str = 'game') -> None:
        """Log an info message if verbose mode is enabled.
        
        Parameters
        ----------
        message : str
            The info message to log
        component : str
            The component logging the message ('game', 'ai', 'core', 'cli')
        """
        if self.verbose or self.very_verbose:
            if component == 'ai':
                self.ai_logger.info(message)
            elif component == 'core':
                self.core_logger.info(message)
            elif component == 'cli':
                self.cli_logger.info(message)
            else:
                self.game_logger.info(message)
    
    def warning(self, message: str, component: str = 'game') -> None:
        """Log a warning message (always shown).
        
        Parameters
        ----------
        message : str
            The warning message to log
        component : str
            The component logging the message ('game', 'ai', 'core', 'cli')
        """
        if component == 'ai':
            self.ai_logger.warning(message)
        elif component == 'core':
            self.core_logger.warning(message)
        elif component == 'cli':
            self.cli_logger.warning(message)
        else:
            self.game_logger.warning(message)
    
    def error(self, message: str, component: str = 'game') -> None:
        """Log an error message (always shown).
        
        Parameters
        ----------
        message : str
            The error message to log
        component : str
            The component logging the message ('game', 'ai', 'core', 'cli')
        """
        if component == 'ai':
            self.ai_logger.error(message)
        elif component == 'core':
            self.core_logger.error(message)
        elif component == 'cli':
            self.cli_logger.error(message)
        else:
            self.game_logger.error(message)
    
    def game_info(self, message: str) -> None:
        """Log a game-specific info message.
        
        Parameters
        ----------
        message : str
            The game info message to log
        """
        self.info(message, 'game')
    
    def game_debug(self, message: str) -> None:
        """Log a game-specific debug message.
        
        Parameters
        ----------
        message : str
            The game debug message to log
        """
        self.debug(message, 'game')
    
    def ai_info(self, message: str) -> None:
        """Log an AI-specific info message.
        
        Parameters
        ----------
        message : str
            The AI info message to log
        """
        self.info(message, 'ai')
    
    def ai_debug(self, message: str) -> None:
        """Log an AI-specific debug message.
        
        Parameters
        ----------
        message : str
            The AI debug message to log
        """
        self.debug(message, 'ai')


def setup_logging(verbose: bool = False, very_verbose: bool = False) -> GameLogger:
    """Set up logging for the Euchre game.
    
    Parameters
    ----------
    verbose : bool
        Enable verbose logging (INFO level)
    very_verbose : bool
        Enable very verbose logging (DEBUG level)
        
    Returns
    -------
    GameLogger
        Configured game logger instance
    """
    return GameLogger(verbose, very_verbose) 