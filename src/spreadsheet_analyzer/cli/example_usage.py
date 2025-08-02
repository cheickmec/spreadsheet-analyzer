"""Example of using the functional CLI utilities.

This file demonstrates how the pure functions in cli/utils can be
composed together for a functional CLI implementation.

CLAUDE-KNOWLEDGE: This is an example file showing functional patterns,
not part of the actual CLI implementation.
"""

import sys
from pathlib import Path

from ..core.types import Result
from .utils import (
    CLIArguments,
    FileNameConfig,
    generate_notebook_name,
    parse_and_validate,
    resolve_output_directory,
)


def process_excel_file(args: CLIArguments) -> Result[Path, str]:
    """Process an Excel file using functional approach.

    This demonstrates how to compose the pure functions.

    Args:
        args: Validated CLI arguments

    Returns:
        Result containing notebook path or error
    """
    # Create file naming config
    config = FileNameConfig(
        excel_file=args.excel_file, model=args.model, sheet_index=args.sheet_index, max_rounds=args.max_rounds
    )

    # Generate output filename
    notebook_name = generate_notebook_name(config)

    # Resolve output directory
    output_dir = resolve_output_directory(args)
    output_path = output_dir / notebook_name

    # Here you would:
    # 1. Run the pipeline analysis
    # 2. Convert results to markdown using pipeline_to_markdown()
    # 3. Create notebook with markdown cells
    # 4. Save to output_path

    return Result.ok(output_path)


def main(argv: list[str] | None = None) -> int:
    """Main entry point demonstrating functional CLI.

    Args:
        argv: Command line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Parse and validate arguments
    result = parse_and_validate(argv or sys.argv[1:])

    if result.is_err():
        print(f"Error: {result.unwrap_err()}", file=sys.stderr)
        return 1

    args = result.unwrap()

    # Process the file
    process_result = process_excel_file(args)

    if process_result.is_err():
        print(f"Processing failed: {process_result.unwrap_err()}", file=sys.stderr)
        return 1

    output_path = process_result.unwrap()
    print(f"Analysis complete! Notebook saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
