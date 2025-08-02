"""Functional logging utilities for the spreadsheet analyzer.

This module provides a functional approach to logging, avoiding
global state and mutable loggers where possible.

CLAUDE-KNOWLEDGE: Traditional Python logging uses global state.
This module provides a functional wrapper while still integrating
with the standard logging infrastructure for compatibility.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .errors import IOError as AppIOError
from .types import Result, err, ok


class LogLevel(Enum):
    """Log levels matching Python's standard levels."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass(frozen=True)
class LogEntry:
    """Immutable log entry."""

    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    extra: dict[str, Any]
    exception: Exception | None = None


@dataclass(frozen=True)
class LoggerConfig:
    """Configuration for a logger."""

    name: str
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: tuple[logging.Handler, ...] = ()


# Pure functions for log message formatting
def format_log_message(template: str, **kwargs: Any) -> str:
    """Format a log message with given parameters."""
    try:
        return template.format(**kwargs)
    except (KeyError, ValueError) as e:
        # Fallback for formatting errors
        return f"{template} (formatting error: {e})"


def create_log_entry(
    level: LogLevel,
    message: str,
    logger_name: str = "spreadsheet_analyzer",
    exception: Exception | None = None,
    **extra: Any,
) -> LogEntry:
    """Create an immutable log entry."""
    return LogEntry(
        timestamp=datetime.now(),
        level=level,
        message=message,
        logger_name=logger_name,
        extra=extra,
        exception=exception,
    )


# Logger wrapper for functional interface
class FunctionalLogger:
    """Functional wrapper around Python's logger.

    This provides a more functional interface while still
    integrating with Python's logging infrastructure.
    """

    def __init__(self, config: LoggerConfig):
        """Initialize with immutable configuration."""
        self._config = config
        self._logger = logging.getLogger(config.name)
        self._logger.setLevel(config.level.value)

        # Add handlers if not already configured
        if not self._logger.handlers and config.handlers:
            for handler in config.handlers:
                self._logger.addHandler(handler)

    def log(self, entry: LogEntry) -> None:
        """Log an entry. This is the only method with side effects."""
        extra_dict = {**entry.extra, "timestamp": entry.timestamp.isoformat()}

        self._logger.log(entry.level.value, entry.message, exc_info=entry.exception, extra=extra_dict)

    @property
    def config(self) -> LoggerConfig:
        """Get immutable configuration."""
        return self._config


# Pure functions for creating log entries
def debug(message: str, **extra: Any) -> LogEntry:
    """Create a debug log entry."""
    return create_log_entry(LogLevel.DEBUG, message, **extra)


def info(message: str, **extra: Any) -> LogEntry:
    """Create an info log entry."""
    return create_log_entry(LogLevel.INFO, message, **extra)


def warning(message: str, **extra: Any) -> LogEntry:
    """Create a warning log entry."""
    return create_log_entry(LogLevel.WARNING, message, **extra)


def error(message: str, exception: Exception | None = None, **extra: Any) -> LogEntry:
    """Create an error log entry."""
    return create_log_entry(LogLevel.ERROR, message, exception=exception, **extra)


def critical(message: str, exception: Exception | None = None, **extra: Any) -> LogEntry:
    """Create a critical log entry."""
    return create_log_entry(LogLevel.CRITICAL, message, exception=exception, **extra)


# Handler creation functions
def create_console_handler(
    level: LogLevel = LogLevel.INFO, format_string: str | None = None
) -> logging.StreamHandler[Any]:
    """Create a console handler."""
    handler = logging.StreamHandler()
    handler.setLevel(level.value)

    if format_string:
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)

    return handler


def create_file_handler(
    file_path: Path, level: LogLevel = LogLevel.INFO, format_string: str | None = None, mode: str = "a"
) -> Result[logging.FileHandler, AppIOError]:
    """Create a file handler, returning Result for error handling."""
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        handler = logging.FileHandler(str(file_path), mode=mode)
        handler.setLevel(level.value)

        if format_string:
            formatter = logging.Formatter(format_string)
            handler.setFormatter(formatter)

        return ok(handler)
    except Exception as e:
        return err(AppIOError(f"Failed to create file handler for {file_path}", path=str(file_path), cause=e))


# Logger factory functions
def create_logger(
    name: str, level: LogLevel = LogLevel.INFO, handlers: list[logging.Handler] | None = None
) -> FunctionalLogger:
    """Create a functional logger with given configuration."""
    if handlers is None:
        handlers = [create_console_handler(level)]

    config = LoggerConfig(name=name, level=level, handlers=tuple(handlers))

    return FunctionalLogger(config)


def create_module_logger(module_name: str) -> FunctionalLogger:
    """Create a logger for a specific module."""
    return create_logger(f"spreadsheet_analyzer.{module_name}")


# Structured logging utilities
def log_operation(logger: FunctionalLogger, operation: str, **details: Any) -> Callable[[Callable], Callable]:
    """Decorator for logging function operations."""

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Log start
            start_entry = info(f"Starting {operation}", operation=operation, function=func.__name__, **details)
            logger.log(start_entry)

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Log success
                success_entry = info(
                    f"Completed {operation}", operation=operation, function=func.__name__, status="success", **details
                )
                logger.log(success_entry)

                return result

            except Exception as e:
                # Log failure
                error_entry = error(
                    f"Failed {operation}",
                    exception=e,
                    operation=operation,
                    function=func.__name__,
                    status="error",
                    **details,
                )
                logger.log(error_entry)
                raise

        return wrapper

    return decorator


# Logging context for structured data
@dataclass(frozen=True)
class LogContext:
    """Immutable context for structured logging."""

    operation: str | None = None
    file_path: str | None = None
    sheet_name: str | None = None
    cell_reference: str | None = None
    agent_id: str | None = None
    extra: dict[str, Any] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary for logging."""
        result = {}

        if self.operation:
            result["operation"] = self.operation
        if self.file_path:
            result["file_path"] = self.file_path
        if self.sheet_name:
            result["sheet_name"] = self.sheet_name
        if self.cell_reference:
            result["cell_reference"] = self.cell_reference
        if self.agent_id:
            result["agent_id"] = self.agent_id
        if self.extra:
            result.update(self.extra)

        return result

    def with_extra(self, **kwargs: Any) -> "LogContext":
        """Create new context with additional fields."""
        from dataclasses import replace

        new_extra = {**(self.extra or {}), **kwargs}
        return replace(self, extra=new_extra)


def log_with_context(
    logger: FunctionalLogger, level: LogLevel, message: str, context: LogContext, exception: Exception | None = None
) -> None:
    """Log a message with structured context."""
    entry = create_log_entry(
        level=level, message=message, logger_name=logger.config.name, exception=exception, **context.to_dict()
    )
    logger.log(entry)


# Performance logging
@dataclass(frozen=True)
class PerformanceMetrics:
    """Immutable performance metrics."""

    operation: str
    duration_ms: float
    memory_mb: float | None = None
    items_processed: int | None = None
    throughput: float | None = None

    def to_log_entry(self, level: LogLevel = LogLevel.INFO) -> LogEntry:
        """Convert metrics to log entry."""
        message = f"Performance: {self.operation} took {self.duration_ms:.2f}ms"

        extra = {"operation": self.operation, "duration_ms": self.duration_ms, "performance_metric": True}

        if self.memory_mb is not None:
            extra["memory_mb"] = self.memory_mb
        if self.items_processed is not None:
            extra["items_processed"] = self.items_processed
        if self.throughput is not None:
            extra["throughput"] = self.throughput

        return create_log_entry(level, message, **extra)
