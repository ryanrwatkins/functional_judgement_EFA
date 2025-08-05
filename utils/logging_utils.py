"""Logging utilities for the EFA project."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

from config import LOG_FORMAT, LOG_LEVEL


def setup_logging(
    level: str = LOG_LEVEL,
    log_file: Optional[Union[str, Path]] = None,
    format_string: str = LOG_FORMAT
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Log format string
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log a function call with its arguments.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function arguments to log
    """
    logger = get_logger(__name__)
    args_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"Calling {func_name}({args_str})")


def log_processing_step(step_name: str, items_count: int) -> None:
    """
    Log a processing step with item count.
    
    Args:
        step_name: Description of the processing step
        items_count: Number of items being processed
    """
    logger = get_logger(__name__)
    logger.info(f"{step_name}: Processing {items_count} items")


def log_validation_result(
    validation_name: str, 
    passed: bool, 
    details: Optional[str] = None
) -> None:
    """
    Log validation results.
    
    Args:
        validation_name: Name of the validation
        passed: Whether validation passed
        details: Optional details about the validation
    """
    logger = get_logger(__name__)
    status = "PASSED" if passed else "FAILED"
    message = f"Validation {validation_name}: {status}"
    
    if details:
        message += f" - {details}"
    
    if passed:
        logger.info(message)
    else:
        logger.error(message)


def log_file_operation(operation: str, file_path: Union[str, Path]) -> None:
    """
    Log file operations.
    
    Args:
        operation: Type of operation (read, write, create, etc.)
        file_path: Path to the file
    """
    logger = get_logger(__name__)
    logger.info(f"File {operation}: {file_path}")


def log_error_with_context(
    error: Exception, 
    context: str, 
    additional_info: Optional[dict] = None
) -> None:
    """
    Log an error with contextual information.
    
    Args:
        error: The exception that occurred
        context: Description of what was being done when error occurred
        additional_info: Optional dictionary of additional information
    """
    logger = get_logger(__name__)
    
    error_msg = f"Error in {context}: {str(error)}"
    
    if additional_info:
        info_str = ", ".join([f"{k}={v}" for k, v in additional_info.items()])
        error_msg += f" | Additional info: {info_str}"
    
    logger.error(error_msg, exc_info=True)


def log_progress(current: int, total: int, step_name: str = "Processing") -> None:
    """
    Log progress of a long-running operation.
    
    Args:
        current: Current item number
        total: Total number of items
        step_name: Name of the operation
    """
    logger = get_logger(__name__)
    percentage = (current / total) * 100
    logger.info(f"{step_name}: {current}/{total} ({percentage:.1f}%)")


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)
    
    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def log_error(self, message: str, exc_info: bool = False) -> None:
        """Log an error message."""
        self.logger.error(message, exc_info=exc_info)
    
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def log_debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)