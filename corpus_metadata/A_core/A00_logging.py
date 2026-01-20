# corpus_metadata/A_core/A00_logging.py
"""
Centralized logging configuration for the ESE pipeline.

Provides structured, consistent logging across all modules with:
- Colored console output for different log levels
- File logging with rotation for persistence
- Context managers for operation tracking
- Performance timing decorators
- Structured log formatting

Usage:
    from A_core.A00_logging import get_logger, timed, LogContext

    logger = get_logger(__name__)
    logger.info("Processing started")

    with LogContext(logger, "extraction"):
        # ... extraction code ...

    @timed(logger)
    def expensive_operation():
        # ... code ...
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Generator, Optional, TypeVar, Union

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])

# Default configuration
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5

# ANSI color codes for console output
COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m",      # Reset
    "BOLD": "\033[1m",       # Bold
    "DIM": "\033[2m",        # Dim
}


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console log output.

    Attributes:
        use_colors: Whether to apply ANSI color codes.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
    ) -> None:
        """
        Initialize the colored formatter.

        Args:
            fmt: Log message format string.
            datefmt: Date format string.
            use_colors: Whether to apply colors (disable for non-TTY output).
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with optional colors.

        Args:
            record: The log record to format.

        Returns:
            Formatted log string with optional ANSI colors.
        """
        if self.use_colors:
            level_color = COLORS.get(record.levelname, COLORS["RESET"])
            record.levelname = f"{level_color}{record.levelname}{COLORS['RESET']}"
            record.name = f"{COLORS['DIM']}{record.name}{COLORS['RESET']}"
        return super().format(record)


class PipelineLogger:
    """
    Singleton logger manager for the ESE pipeline.

    Provides centralized configuration and access to loggers
    with consistent formatting across all modules.

    Attributes:
        _instance: Singleton instance.
        _initialized: Whether the logger has been configured.
        _log_dir: Directory for log files.
        _log_level: Current log level.
    """

    _instance: Optional["PipelineLogger"] = None
    _initialized: bool = False

    def __new__(cls) -> "PipelineLogger":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the logger manager (only runs once)."""
        if PipelineLogger._initialized:
            return

        self._log_dir: Path = DEFAULT_LOG_DIR
        self._log_level: int = DEFAULT_LOG_LEVEL
        self._file_handler: Optional[RotatingFileHandler] = None
        self._console_handler: Optional[logging.StreamHandler] = None
        self._run_id: Optional[str] = None

        PipelineLogger._initialized = True

    def configure(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        log_level: int = DEFAULT_LOG_LEVEL,
        run_id: Optional[str] = None,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
    ) -> None:
        """
        Configure the logging system.

        Args:
            log_dir: Directory for log files. Created if doesn't exist.
            log_level: Minimum log level to capture.
            run_id: Unique identifier for the current run.
            enable_file_logging: Whether to write logs to file.
            enable_console_logging: Whether to output to console.
        """
        self._log_level = log_level
        self._run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        if log_dir:
            self._log_dir = Path(log_dir)

        # Create log directory
        if enable_file_logging:
            self._log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger("corpus_metadata")
        root_logger.setLevel(log_level)

        # Remove existing handlers
        root_logger.handlers.clear()

        # Console handler with colors
        if enable_console_logging:
            self._console_handler = logging.StreamHandler(sys.stdout)
            self._console_handler.setLevel(log_level)
            console_formatter = ColoredFormatter(
                fmt="%(levelname)-8s | %(message)s",
                datefmt=DEFAULT_DATE_FORMAT,
            )
            self._console_handler.setFormatter(console_formatter)
            root_logger.addHandler(self._console_handler)

        # File handler with rotation
        if enable_file_logging:
            log_file = self._log_dir / f"pipeline_{self._run_id}.log"
            self._file_handler = RotatingFileHandler(
                log_file,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            self._file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                fmt=DEFAULT_FORMAT,
                datefmt=DEFAULT_DATE_FORMAT,
            )
            self._file_handler.setFormatter(file_formatter)
            root_logger.addHandler(self._file_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance for the given module.

        Args:
            name: Module name (typically __name__).

        Returns:
            Configured logger instance.
        """
        # Ensure name is under corpus_metadata namespace
        if not name.startswith("corpus_metadata"):
            name = f"corpus_metadata.{name}"
        return logging.getLogger(name)

    @property
    def run_id(self) -> Optional[str]:
        """Get the current run ID."""
        return self._run_id


# Module-level singleton instance
_logger_manager = PipelineLogger()


def configure_logging(
    log_dir: Optional[Union[str, Path]] = None,
    log_level: int = DEFAULT_LOG_LEVEL,
    run_id: Optional[str] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
) -> None:
    """
    Configure the pipeline logging system.

    This should be called once at application startup.

    Args:
        log_dir: Directory for log files.
        log_level: Minimum log level (e.g., logging.INFO).
        run_id: Unique identifier for this run.
        enable_file_logging: Whether to write to log files.
        enable_console_logging: Whether to output to console.

    Example:
        >>> configure_logging(
        ...     log_dir="logs",
        ...     log_level=logging.DEBUG,
        ...     run_id="RUN_20260120_123456"
        ... )
    """
    _logger_manager.configure(
        log_dir=log_dir,
        log_level=log_level,
        run_id=run_id,
        enable_file_logging=enable_file_logging,
        enable_console_logging=enable_console_logging,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given module.

    Args:
        name: Module name, typically __name__.

    Returns:
        Configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return _logger_manager.get_logger(name)


@contextmanager
def LogContext(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
) -> Generator[None, None, None]:
    """
    Context manager for logging operation start/end with timing.

    Args:
        logger: Logger instance to use.
        operation: Description of the operation.
        level: Log level for messages.

    Yields:
        None

    Example:
        >>> with LogContext(logger, "PDF parsing"):
        ...     parse_pdf(file_path)
        INFO | Starting: PDF parsing
        INFO | Completed: PDF parsing (2.34s)
    """
    start_time = time.perf_counter()
    logger.log(level, f"Starting: {operation}")
    try:
        yield
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"Failed: {operation} ({elapsed:.2f}s) - {type(e).__name__}: {e}")
        raise
    else:
        elapsed = time.perf_counter() - start_time
        logger.log(level, f"Completed: {operation} ({elapsed:.2f}s)")


def timed(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
) -> Callable[[F], F]:
    """
    Decorator for timing function execution.

    Args:
        logger: Logger to use. If None, uses module logger.
        level: Log level for timing messages.

    Returns:
        Decorated function.

    Example:
        >>> @timed(logger)
        ... def expensive_operation(data):
        ...     # ... processing ...
        ...     return result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                logger.log(level, f"{func.__name__} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {e}")
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


class StepLogger:
    """
    Logger for pipeline steps with progress tracking.

    Provides consistent formatting for multi-step pipelines
    with step numbering and indentation.

    Attributes:
        logger: Underlying logger instance.
        total_steps: Total number of steps in the pipeline.
        current_step: Current step number.
    """

    def __init__(
        self,
        logger: logging.Logger,
        total_steps: int,
        prefix: str = "",
    ) -> None:
        """
        Initialize the step logger.

        Args:
            logger: Logger instance to use.
            total_steps: Total number of steps.
            prefix: Optional prefix for all messages.
        """
        self.logger = logger
        self.total_steps = total_steps
        self.current_step = 0
        self.prefix = prefix
        self._step_start_time: Optional[float] = None

    def step(self, description: str) -> "StepLogger":
        """
        Start a new step.

        Args:
            description: Description of the step.

        Returns:
            Self for chaining.
        """
        self.current_step += 1
        self._step_start_time = time.perf_counter()
        msg = f"[{self.current_step}/{self.total_steps}] {description}"
        if self.prefix:
            msg = f"{self.prefix} {msg}"
        self.logger.info(msg)
        return self

    def detail(self, message: str, indent: int = 2) -> None:
        """
        Log a detail within the current step.

        Args:
            message: Detail message.
            indent: Number of spaces for indentation.
        """
        self.logger.info(f"{' ' * indent}{message}")

    def complete(self, summary: Optional[str] = None) -> float:
        """
        Complete the current step.

        Args:
            summary: Optional completion summary.

        Returns:
            Elapsed time in seconds.
        """
        elapsed = 0.0
        if self._step_start_time:
            elapsed = time.perf_counter() - self._step_start_time

        if summary:
            self.detail(f"{summary} ({elapsed:.2f}s)")

        return elapsed


# Convenience aliases
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
