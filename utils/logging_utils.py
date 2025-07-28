"""
Utility functions for configuring loggers.
"""
import logging


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger that writes INFO-level messages to the console.

    :param name: Logger name.
    :return: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
