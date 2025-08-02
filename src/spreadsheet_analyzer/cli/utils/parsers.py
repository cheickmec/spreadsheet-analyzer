"""Pure functions for CLI argument parsing.

This module provides functional argument parsing without side effects,
returning Result types for error handling.

CLAUDE-KNOWLEDGE: Argument parsing is separated from validation to support
composable validation rules and better error messages.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from ...core.types import Result, err, ok


@dataclass(frozen=True)
class CLIArguments:
    """Immutable CLI arguments for notebook analysis."""

    excel_file: Path
    model: str = "claude-3-5-sonnet-20241022"
    max_rounds: int = 5
    sheet_index: int = 0
    output_dir: Path | None = None
    verbose: bool = False
    debug: bool = False
    no_cost_tracking: bool = False
    no_observability: bool = False
    custom_llm_api_base: str | None = None
    custom_llm_api_key: str | None = None
    system_prompt: str | None = None


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with all CLI options.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Analyze Excel files using LLM-powered insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze with default model
  uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx

  # Use specific model with custom rounds
  uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model gpt-4 --max-rounds 3

  # Analyze specific sheet
  uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --sheet-index 2

  # Use Ollama model
  uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model ollama/mistral:latest
""",
    )

    # Positional arguments
    parser.add_argument("excel_file", type=Path, help="Path to Excel file to analyze")

    # Model configuration
    parser.add_argument(
        "--model", default="claude-3-5-sonnet-20241022", help="LLM model to use (default: claude-3-5-sonnet-20241022)"
    )

    parser.add_argument("--max-rounds", type=int, default=5, help="Maximum analysis rounds (default: 5)")

    # Sheet selection
    parser.add_argument("--sheet-index", type=int, default=0, help="Sheet index to analyze (default: 0)")

    # Output configuration
    parser.add_argument("--output-dir", type=Path, help="Output directory for notebooks (default: auto-generated)")

    # Logging options
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Feature flags
    parser.add_argument("--no-cost-tracking", action="store_true", help="Disable cost tracking")

    parser.add_argument("--no-observability", action="store_true", help="Disable observability features")

    # Custom LLM configuration
    parser.add_argument("--custom-llm-api-base", help="Custom LLM API base URL")

    parser.add_argument("--custom-llm-api-key", help="Custom LLM API key")

    # Advanced options
    parser.add_argument("--system-prompt", help="Custom system prompt for analysis")

    return parser


def parse_arguments(args: list[str]) -> Result[CLIArguments, str]:
    """Parse CLI arguments into immutable config.

    Pure function that parses arguments without side effects.

    Args:
        args: List of command line arguments

    Returns:
        Result containing parsed arguments or error message
    """
    parser = create_argument_parser()

    try:
        # Parse arguments without exiting on error
        namespace = parser.parse_args(args)

        # Convert to immutable dataclass
        cli_args = CLIArguments(
            excel_file=namespace.excel_file,
            model=namespace.model,
            max_rounds=namespace.max_rounds,
            sheet_index=namespace.sheet_index,
            output_dir=namespace.output_dir,
            verbose=namespace.verbose,
            debug=namespace.debug,
            no_cost_tracking=namespace.no_cost_tracking,
            no_observability=namespace.no_observability,
            custom_llm_api_base=namespace.custom_llm_api_base,
            custom_llm_api_key=namespace.custom_llm_api_key,
            system_prompt=namespace.system_prompt,
        )

        return ok(cli_args)

    except SystemExit:
        # ArgumentParser calls sys.exit on error
        # Capture the error message instead
        return err("Invalid arguments provided")
    except Exception as e:
        return err(f"Failed to parse arguments: {e!s}")


def validate_arguments(args: CLIArguments) -> Result[None, str]:
    """Validate parsed CLI arguments.

    Performs validation checks on parsed arguments.

    Args:
        args: Parsed CLI arguments

    Returns:
        Ok(None) if valid, Err with error message if invalid
    """
    # Check file exists
    if not args.excel_file.exists():
        return err(f"Excel file not found: {args.excel_file}")

    # Check file is readable
    if not args.excel_file.is_file():
        return err(f"Not a file: {args.excel_file}")

    # Validate file extension
    valid_extensions = {".xlsx", ".xls", ".xlsm", ".xlsb"}
    if args.excel_file.suffix.lower() not in valid_extensions:
        return err(f"Invalid file type: {args.excel_file.suffix}")

    # Validate max rounds
    if args.max_rounds < 1:
        return err("Max rounds must be at least 1")

    if args.max_rounds > 20:
        return err("Max rounds cannot exceed 20")

    # Validate sheet index
    if args.sheet_index < 0:
        return err("Sheet index must be non-negative")

    # Validate output directory if provided
    if args.output_dir and args.output_dir.exists() and not args.output_dir.is_dir():
        return err(f"Output path exists but is not a directory: {args.output_dir}")

    # Validate model name format
    if not args.model or not args.model.strip():
        return err("Model name cannot be empty")

    # Validate custom LLM config
    if args.custom_llm_api_base and not args.custom_llm_api_key:
        return err("Custom LLM API key required when using custom API base")

    return ok(None)


def parse_and_validate(args: list[str]) -> Result[CLIArguments, str]:
    """Parse and validate CLI arguments in one step.

    Convenience function that combines parsing and validation.

    Args:
        args: List of command line arguments

    Returns:
        Result containing validated arguments or error message
    """
    # Parse arguments
    parse_result = parse_arguments(args)
    if parse_result.is_err():
        return parse_result

    cli_args = parse_result.unwrap()

    # Validate arguments
    validation = validate_arguments(cli_args)
    if validation.is_err():
        return err(validation.unwrap_err())

    return ok(cli_args)


def get_help_text() -> str:
    """Get help text for the CLI.

    Returns:
        Help text string
    """
    parser = create_argument_parser()
    return parser.format_help()


def get_usage_text() -> str:
    """Get usage text for the CLI.

    Returns:
        Usage text string
    """
    parser = create_argument_parser()
    return parser.format_usage()


# Helper functions for specific argument handling


def resolve_output_directory(args: CLIArguments) -> Path:
    """Resolve the output directory for notebooks.

    If no output directory specified, creates a default based on Excel filename.

    Args:
        args: CLI arguments

    Returns:
        Resolved output directory path
    """
    if args.output_dir:
        return args.output_dir

    # Default: excel_filename_analysis/
    base_name = args.excel_file.stem
    return Path(f"{base_name}_analysis")


def should_track_costs(args: CLIArguments) -> bool:
    """Determine if cost tracking should be enabled.

    Args:
        args: CLI arguments

    Returns:
        True if cost tracking should be enabled
    """
    return not args.no_cost_tracking


def should_enable_observability(args: CLIArguments) -> bool:
    """Determine if observability should be enabled.

    Args:
        args: CLI arguments

    Returns:
        True if observability should be enabled
    """
    return not args.no_observability


def get_log_level(args: CLIArguments) -> str:
    """Determine logging level from arguments.

    Args:
        args: CLI arguments

    Returns:
        Logging level string (DEBUG, INFO, WARNING, ERROR)
    """
    if args.debug:
        return "DEBUG"
    elif args.verbose:
        return "INFO"
    else:
        return "WARNING"
