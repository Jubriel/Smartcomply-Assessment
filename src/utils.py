import logging
from src.config import config


def setup_logging() -> logging.Logger:
    """
    Configure logging with proper formatting
    Args:
        level: Logging level (defaults to config.LOG_LEVEL)
    Returns:
        Logger instance
    """

    logging.basicConfig(
        level=config.LOG_LEVEL,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        handlers=[logging.FileHandler(config.LOG_FILE),
                  logging.StreamHandler()]
    )
    return logging.getLogger(__name__)
