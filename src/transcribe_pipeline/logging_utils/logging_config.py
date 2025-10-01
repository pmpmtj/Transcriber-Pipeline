"""
Logging Configuration Module

This module provides centralized logging configuration for the transcription package.
It supports configurable log levels, file outputs, and console outputs.

The configuration allows each module/function to have its own logger with 
independent settings.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    # Main CLI logger - used by transcribe_cli.py
    "transcribe_cli": {
        "level": "INFO",  # Default level (can be overridden by --debug flag)
        "log_filename": "transcribe_cli.log",
        "console_output": True,
        "file_output": True,
    },
    
    # Core transcription logger
    "transcription": {
        "level": "DEBUG",
        "log_filename": "transcription.log",
        "console_output": True,
        "file_output": True,
    },
    
    # Language detection logger
    "language_detection": {
        "level": "DEBUG",
        "log_filename": "language_detection.log",
        "console_output": True,
        "file_output": True,
    },
}

# ============================================================================
# DEFAULT SETTINGS
# ============================================================================
DEFAULT_LOG_DIR = "logs"  # Default log directory (relative to CWD or configurable)
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    logger_name: str,
    log_dir: Optional[Path] = None,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
) -> logging.Logger:
    """
    Get or create a configured logger instance.
    
    This function creates a logger with both console and file handlers based on
    the configuration defined in LOGGING_CONFIG. If the logger already exists,
    it returns the existing instance.
    
    Args:
        logger_name: Name of the logger (must match a key in LOGGING_CONFIG)
        log_dir: Optional directory for log files (defaults to DEFAULT_LOG_DIR in CWD)
        console_level: Optional override for console log level ('DEBUG', 'INFO', etc.)
        file_level: Optional override for file log level
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = get_logger('transcribe_cli', console_level='DEBUG')
        >>> logger.info('Starting transcription...')
        >>> logger.debug('Detailed debug information')
    """
    # Get configuration for this logger
    config = LOGGING_CONFIG.get(logger_name, {})
    
    # Use provided levels or fall back to config
    default_level = config.get("level", "INFO")
    console_level = console_level or default_level
    file_level = file_level or default_level
    
    # Create logger
    logger = logging.getLogger(logger_name)
    
    # Only configure if not already configured (avoid duplicate handlers)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to allow handlers to filter
        logger.propagate = False  # Don't propagate to root logger
        
        # Create formatters
        formatter = logging.Formatter(
            fmt=DEFAULT_LOG_FORMAT,
            datefmt=DEFAULT_DATE_FORMAT
        )
        
        # Console handler (if enabled in config)
        if config.get("console_output", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, console_level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler (if enabled in config)
        if config.get("file_output", False):
            # Determine log directory
            if log_dir is None:
                log_dir = Path.cwd() / DEFAULT_LOG_DIR
            else:
                log_dir = Path(log_dir)
            
            # Create log directory if it doesn't exist
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Get log filename from config
            log_filename = config.get("log_filename", f"{logger_name}.log")
            log_file = log_dir / log_filename
            
            # Create file handler
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(getattr(logging, file_level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def set_console_level(logger: logging.Logger, level: str) -> None:
    """
    Update the console handler's log level for an existing logger.
    
    This is useful for dynamically changing log levels based on CLI flags
    (e.g., enabling DEBUG mode with --debug flag).
    
    Args:
        logger: Logger instance to update
        level: New log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        
    Example:
        >>> logger = get_logger('transcribe_cli')
        >>> set_console_level(logger, 'DEBUG')  # Enable debug output
    """
    log_level = getattr(logging, level.upper())
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handler.setLevel(log_level)
            break


def disable_file_logging(logger: logging.Logger) -> None:
    """
    Disable file logging for a logger (keeps console logging only).
    
    Args:
        logger: Logger instance to modify
    """
    for handler in logger.handlers[:]:  # Use slice to iterate over copy
        if isinstance(handler, logging.FileHandler):
            handler.close()
            logger.removeHandler(handler)

