"""
Logging configuration and utilities for diamond segmentation pipeline.
Provides structured logging with file rotation and debug support.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
from datetime import datetime


# Log levels mapping
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        
        # Format the message
        result = super().format(record)
        
        # Reset levelname for next use
        record.levelname = levelname
        
        return result


def setup_logging(log_dir: Optional[Path] = None,
                  log_level: str = 'INFO',
                  log_to_file: bool = True,
                  log_to_console: bool = True,
                  use_colors: bool = True,
                  max_bytes: int = 10 * 1024 * 1024,  # 10 MB
                  backup_count: int = 5) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files (default: logs/)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        use_colors: Whether to use colored console output
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('diamond_segmentation')
    logger.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Add console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVELS.get(log_level.upper(), logging.INFO))
        
        if use_colors:
            console_handler.setFormatter(colored_formatter)
        else:
            console_handler.setFormatter(simple_formatter)
        
        logger.addHandler(console_handler)
    
    # Add file handler with rotation
    if log_to_file:
        if log_dir is None:
            log_dir = Path('logs')
        
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        log_file = log_dir / 'diamond_segmentation.log'
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error log file
        error_log_file = log_dir / 'errors.log'
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name (default: diamond_segmentation)
        
    Returns:
        Logger instance
    """
    if name is None:
        name = 'diamond_segmentation'
    
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set up default configuration
    if not logger.handlers:
        setup_logging()
    
    return logger


def set_log_level(level: str, logger_name: Optional[str] = None):
    """
    Change log level dynamically.
    
    Args:
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Optional logger name
    """
    logger = get_logger(logger_name)
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # Update handler levels
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
            handler.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))


def enable_debug_mode(logger_name: Optional[str] = None):
    """
    Enable debug mode with verbose logging.
    
    Args:
        logger_name: Optional logger name
    """
    set_log_level('DEBUG', logger_name)
    logger = get_logger(logger_name)
    logger.debug("Debug mode enabled")


def disable_debug_mode(logger_name: Optional[str] = None):
    """
    Disable debug mode and return to INFO level.
    
    Args:
        logger_name: Optional logger name
    """
    set_log_level('INFO', logger_name)
    logger = get_logger(logger_name)
    logger.info("Debug mode disabled")


class LogContext:
    """
    Context manager for temporary log level changes.
    """
    
    def __init__(self, level: str, logger_name: Optional[str] = None):
        """
        Initialize context.
        
        Args:
            level: Temporary log level
            logger_name: Optional logger name
        """
        self.level = level
        self.logger_name = logger_name
        self.original_level = None
        self.logger = None
    
    def __enter__(self):
        """Enter context and set log level."""
        self.logger = get_logger(self.logger_name)
        self.original_level = self.logger.level
        set_log_level(self.level, self.logger_name)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original log level."""
        self.logger.setLevel(self.original_level)


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
            raise
    
    return wrapper


def create_session_log(log_dir: Optional[Path] = None) -> Path:
    """
    Create a session-specific log file.
    
    Args:
        log_dir: Directory for log files
        
    Returns:
        Path to session log file
    """
    if log_dir is None:
        log_dir = Path('logs')
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create session log with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_log = log_dir / f'session_{timestamp}.log'
    
    # Add handler for this session
    logger = get_logger()
    
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    session_handler = logging.FileHandler(session_log)
    session_handler.setLevel(logging.DEBUG)
    session_handler.setFormatter(formatter)
    
    logger.addHandler(session_handler)
    logger.info(f"Session log created: {session_log}")
    
    return session_log


def log_system_info():
    """Log system and environment information."""
    import platform
    import psutil
    
    logger = get_logger()
    
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"Total memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    logger.info("=" * 60)


def log_processing_start(operation: str, **kwargs):
    """
    Log the start of a processing operation.
    
    Args:
        operation: Operation name
        **kwargs: Additional information to log
    """
    logger = get_logger()
    logger.info("=" * 60)
    logger.info(f"Starting: {operation}")
    
    for key, value in kwargs.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)


def log_processing_end(operation: str, success: bool = True, **kwargs):
    """
    Log the end of a processing operation.
    
    Args:
        operation: Operation name
        success: Whether operation succeeded
        **kwargs: Additional information to log
    """
    logger = get_logger()
    
    status = "✓ COMPLETED" if success else "✗ FAILED"
    logger.info("=" * 60)
    logger.info(f"{status}: {operation}")
    
    for key, value in kwargs.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)


def log_error(error: Exception, operation: Optional[str] = None, **context):
    """
    Log an error with context.
    
    Args:
        error: Exception that occurred
        operation: Optional operation name
        **context: Additional context information
    """
    logger = get_logger()
    
    logger.error("=" * 60)
    logger.error(f"ERROR: {type(error).__name__}")
    
    if operation:
        logger.error(f"Operation: {operation}")
    
    logger.error(f"Message: {str(error)}")
    
    if context:
        logger.error("Context:")
        for key, value in context.items():
            logger.error(f"  {key}: {value}")
    
    logger.error("=" * 60)
    logger.exception("Traceback:")


def clear_logs(log_dir: Optional[Path] = None, keep_latest: int = 0):
    """
    Clear old log files.
    
    Args:
        log_dir: Directory containing log files
        keep_latest: Number of latest files to keep
    """
    if log_dir is None:
        log_dir = Path('logs')
    
    log_dir = Path(log_dir)
    
    if not log_dir.exists():
        return
    
    # Get all log files
    log_files = sorted(log_dir.glob('*.log*'), key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Keep latest N files
    files_to_delete = log_files[keep_latest:]
    
    logger = get_logger()
    
    for log_file in files_to_delete:
        try:
            log_file.unlink()
            logger.info(f"Deleted old log file: {log_file}")
        except Exception as e:
            logger.warning(f"Failed to delete {log_file}: {e}")


# Initialize default logger on module import
_default_logger = None


def initialize_default_logger(log_dir: Optional[Path] = None,
                              log_level: str = 'INFO',
                              debug: bool = False):
    """
    Initialize the default logger with specified configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        debug: Whether to enable debug mode
    """
    global _default_logger
    
    level = 'DEBUG' if debug else log_level
    _default_logger = setup_logging(log_dir=log_dir, log_level=level)
    
    return _default_logger