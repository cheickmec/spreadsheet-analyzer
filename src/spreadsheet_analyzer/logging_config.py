"""Structured logging configuration for the Spreadsheet Analyzer.

This module sets up structured logging using structlog for both CLI and future API
usage. It provides human-readable output for terminals and JSON output for
machine processing.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import structlog


def setup_logging(
    *, verbosity: int = 0, log_file: Path | None = None, json_logs: bool = False, no_color: bool = False
) -> None:
    """Configure structured logging for both CLI and future API.

    Args:
        verbosity: Logging verbosity level
            0 = WARNING and above (default)
            1 = INFO and above
            2 = DEBUG and above
        log_file: Optional file path to write logs to
        json_logs: If True, output JSON logs instead of console format
        no_color: If True, disable colored output even in TTY
    """
    # Map verbosity to log level
    log_levels = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    log_level = log_levels.get(verbosity, logging.DEBUG)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # Choose renderer based on output format
    if json_logs or log_file:
        # JSON output for files or when requested
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    elif sys.stdout.isatty() and not no_color:
        # Rich colored output for terminals
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                pad_event=True,
                exception_formatter=structlog.dev.RichTracebackFormatter(
                    show_locals=verbosity >= 2  # Show locals in DEBUG mode
                ),
            )
        )
    else:
        # Plain output for pipes or when colors disabled
        processors.append(structlog.dev.ConsoleRenderer(colors=False))

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set up file logging if requested
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        # File always gets JSON format
        file_processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(),
            foreign_pre_chain=file_processors,
        )
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Logger name, typically __name__. If None, uses calling module.

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def log_operation(operation: str, **context_vars: Any):
    """Create a logger with operation context.

    This is useful for tracking operations across multiple log entries,
    especially in async contexts like the future API.

    Args:
        operation: Name of the operation being performed
        **context_vars: Additional context variables

    Returns:
        Logger bound with context

    Example:
        logger = log_operation("analyze_file", file_path=str(path))
        logger.info("Starting analysis")
        # ... do work ...
        logger.info("Analysis complete", duration=elapsed)
    """
    return structlog.contextvars.bind_contextvars(operation=operation, **context_vars)
