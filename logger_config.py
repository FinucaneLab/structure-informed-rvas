"""
Centralized logging configuration for the structure-informed RVAS tool.
"""

import logging
import logging.handlers
import os
from typing import Optional


def setup_logger(
    name: str, 
    log_dir: str = "logs", 
    level: str = "INFO",
    console_level: str = "INFO"
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name (typically module name)
        log_dir: Directory to store log files
        level: Overall logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_level: Console-specific logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, console_level.upper()))
    
    # File handler with rotation for detailed logs
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        f"{log_dir}/{name}.log", 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Structured format with timestamps
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Simpler format for console to reduce clutter
    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    
    console_handler.setFormatter(console_formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return setup_logger(name)


def set_log_level(logger: logging.Logger, level: str) -> None:
    """
    Change the logging level for an existing logger.
    
    Args:
        logger: Logger instance
        level: New logging level
    """
    logger.setLevel(getattr(logging, level.upper()))
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            # This is the console handler
            handler.setLevel(getattr(logging, level.upper()))