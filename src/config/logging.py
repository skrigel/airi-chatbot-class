"""
Logging configuration for the AIRI chatbot application.
"""
import logging
import sys
from typing import Optional

def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    loggers_config = {
        'chromadb.telemetry.product.posthog': logging.WARNING,
        'werkzeug': logging.INFO,
        'urllib3.connectionpool': logging.WARNING,
    }
    
    for logger_name, logger_level in loggers_config.items():
        logging.getLogger(logger_name).setLevel(logger_level)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    Args:
        name: Name for the logger, typically __name__
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)