"""
Centralized logging configuration.
"""
import logging
import sys
from pathlib import Path

def setup_logger(
    name: str,
    level: str = "INFO",
    log_format: str = "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
    log_file: Path = None
) -> logging.Logger:
    """
    Set up and return a configured logger.
    
    Parameters
    ----------
    name : str
        Name of the logger (typically __name__)
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_format : str
        Format string for log messages
    date_format : str
        Format string for timestamps
    log_file : Path, optional
        If provided, also log to this file
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger