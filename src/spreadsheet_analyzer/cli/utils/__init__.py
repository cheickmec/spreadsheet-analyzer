"""Utility functions for CLI operations.

This package contains pure functions for:
- File naming generation
- Markdown conversion
- Argument parsing
"""

from .markdown import (
    content_to_markdown,
    create_header,
    formulas_to_markdown,
    integrity_to_markdown,
    pipeline_to_markdown,
    security_to_markdown,
    should_show_security,
    structure_to_markdown,
)
from .naming import (
    FileNameConfig,
    generate_log_name,
    generate_notebook_name,
    generate_session_id,
    get_cost_tracking_path,
    sanitize_model_name,
    sanitize_sheet_name,
)
from .parsers import (
    CLIArguments,
    create_argument_parser,
    get_help_text,
    get_log_level,
    get_usage_text,
    parse_and_validate,
    parse_arguments,
    resolve_output_directory,
    should_enable_observability,
    should_track_costs,
    validate_arguments,
)

__all__ = [
    # Parser utilities
    "CLIArguments",
    "create_argument_parser",
    "get_help_text",
    "get_log_level",
    "get_usage_text",
    "parse_and_validate",
    "parse_arguments",
    "resolve_output_directory",
    "should_enable_observability",
    "should_track_costs",
    "validate_arguments",
    # Naming utilities
    "FileNameConfig",
    "generate_log_name",
    "generate_notebook_name",
    "generate_session_id",
    "get_cost_tracking_path",
    "sanitize_model_name",
    "sanitize_sheet_name",
    # Markdown utilities
    "content_to_markdown",
    "create_header",
    "formulas_to_markdown",
    "integrity_to_markdown",
    "pipeline_to_markdown",
    "security_to_markdown",
    "should_show_security",
    "structure_to_markdown",
]
