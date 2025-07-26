#!/usr/bin/env python3
"""
Unified spreadsheet analysis CLI.

Simple, focused, powerful: analyze any spreadsheet and get a Jupyter notebook.
Uses LLM enhancement by default, with option to disable for cost control or testing.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import NoReturn

from structlog import get_logger

from ..core_exec import KernelProfile
from ..plugins.spreadsheet import register_all_plugins as register_spreadsheet_plugins
from ..plugins.spreadsheet.io import list_sheets
from ..workflows import NotebookWorkflow, WorkflowConfig, WorkflowMode

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="analyze-spreadsheet",
        description="Analyze spreadsheets and generate Jupyter notebooks with LLM enhancement",
        epilog="""
Examples:
  %(prog)s sales_data.xlsx                                    # Full LLM analysis
  %(prog)s sales_data.xlsx --no-llm                          # Deterministic only
  %(prog)s sales_data.xlsx --sheet "Q4 Revenue"              # Specific sheet
  %(prog)s inventory.csv --tasks "profile,outliers"          # Custom tasks
  %(prog)s data.xlsx --model "gpt-4" --max-cost 1.0          # Custom LLM settings
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # The file to analyze
    parser.add_argument("file", type=Path, help="Excel or CSV file to analyze")

    # Optional sheet name
    parser.add_argument("--sheet", "-s", help="Sheet name (for Excel files, defaults to first sheet)")

    # LLM control - key change: LLM is ON by default
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM enhancement (deterministic analysis only)")

    parser.add_argument(
        "--model", "-m", default="claude-3-opus-20240229", help="LLM model to use (default: Claude 3 Opus)"
    )

    parser.add_argument(
        "--max-cost", type=float, default=0.50, help="Maximum cost in USD for LLM calls (default: $0.50)"
    )

    parser.add_argument(
        "--temperature", type=float, default=0.0, help="LLM temperature (0.0-1.0, default: 0.0 for consistency)"
    )

    # Task selection
    parser.add_argument("--tasks", "-t", help="Comma-separated tasks to run (default: auto-detect all available)")

    # Output control
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("analysis_results"),
        help="Output directory (default: ./analysis_results)",
    )

    # Execution control
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel to use (default: python3)")

    parser.add_argument("--timeout", type=int, default=300, help="Cell execution timeout in seconds (default: 300)")

    parser.add_argument("--no-quality-checks", action="store_true", help="Skip quality checks on generated notebook")

    # Utilities
    parser.add_argument("--list-sheets", action="store_true", help="List available sheets and exit")

    parser.add_argument("--list-tasks", action="store_true", help="List available analysis tasks and exit")

    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress and debug information")

    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")

    return parser


async def analyze_spreadsheet(
    file_path: Path,
    sheet_name: str | None = None,
    use_llm: bool = True,
    model: str = "claude-3-opus-20240229",
    max_cost: float = 0.50,
    temperature: float = 0.0,
    tasks: list[str] | None = None,
    output_dir: Path = Path("analysis_results"),
    kernel: str = "python3",
    timeout: int = 300,
    quality_checks: bool = True,
    verbose: bool = False,
) -> Path:
    """
    Analyze a spreadsheet and generate a Jupyter notebook.

    Args:
        file_path: Path to Excel/CSV file
        sheet_name: Specific sheet to analyze (None = auto-detect)
        use_llm: Whether to use LLM enhancement (default: True)
        model: LLM model identifier
        max_cost: Maximum cost for LLM calls
        temperature: LLM sampling temperature
        tasks: List of task names to run (None = auto-detect)
        output_dir: Where to save results
        kernel: Jupyter kernel to use
        timeout: Cell execution timeout
        quality_checks: Whether to run quality inspection
        verbose: Enable detailed logging

    Returns:
        Path to generated notebook

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If configuration is invalid
        RuntimeError: If analysis fails
    """
    # Validate input file
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() not in {".xlsx", ".xls", ".csv"}:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    # Register plugins - this discovers all available tasks
    register_spreadsheet_plugins()

    # Auto-detect sheet name for Excel files if not provided
    if sheet_name is None and file_path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            sheets = list_sheets(file_path)
            sheet_name = sheets[0] if sheets else "Sheet1"
            if verbose:
                logger.info(f"Auto-detected sheet: {sheet_name}")
        except Exception as e:
            logger.warning(f"Could not auto-detect sheets: {e}")
            sheet_name = "Sheet1"
    elif sheet_name is None:
        # CSV files don't have sheets
        sheet_name = file_path.stem

    # Build output path
    file_subdir = output_dir / file_path.stem
    file_subdir.mkdir(parents=True, exist_ok=True)

    output_name = sheet_name.replace(" ", "_").replace("/", "_")  # Safe filename
    output_path = file_subdir / f"{output_name}.ipynb"

    # Check API key if using LLM
    if use_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.warning("ANTHROPIC_API_KEY not set, falling back to deterministic mode")
        use_llm = False

    # Configure workflow
    config = WorkflowConfig(
        file_path=str(file_path),
        output_path=str(output_path),
        sheet_name=sheet_name,
        mode=WorkflowMode.BUILD_AND_EXECUTE if use_llm else WorkflowMode.BUILD_ONLY,
        tasks=tasks or [],  # Empty list means auto-detect all
        kernel_profile=KernelProfile(
            name=kernel,
            max_execution_time=timeout,
            idle_timeout_seconds=60,
        ),
        execute_timeout=timeout,
        quality_checks=quality_checks,
    )

    if verbose:
        logger.info("Starting workflow", file=str(file_path), sheet=sheet_name, use_llm=use_llm, mode=config.mode.value)

    # Run workflow
    workflow = NotebookWorkflow()
    result = await workflow.run(config)

    # Check for workflow errors and warnings
    if result.errors:
        for error in result.errors:
            logger.error(f"Workflow error: {error}")

    if result.warnings:
        for warning in result.warnings:
            logger.warning(f"Workflow warning: {warning}")

    # Check for issues
    if result.quality_metrics and result.quality_metrics.overall_score < 0.3:
        logger.warning(
            "Low quality score detected",
            score=result.quality_metrics.overall_score,
            issues=len(result.quality_metrics.issues),
        )

    if verbose and result.execution_stats:
        logger.info(
            "Execution complete",
            total_cells=result.execution_stats.total_cells,
            executed=result.execution_stats.executed_cells,
            errors=result.execution_stats.error_cells,
            duration=result.execution_stats.total_duration_seconds,
        )

    return output_path


def main(argv: list[str] | None = None) -> NoReturn:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    try:
        # Handle utility commands first
        if args.list_sheets:
            try:
                sheets = list_sheets(args.file)
                print(f"\n📊 Sheets in {args.file.name}:")
                for i, sheet in enumerate(sheets, 1):
                    print(f"  {i}. {sheet}")
                print()
            except Exception as e:
                print(f"❌ Error reading sheets: {e}", file=sys.stderr)
                sys.exit(1)
            sys.exit(0)

        if args.list_tasks:
            register_spreadsheet_plugins()
            from ..plugins.base import registry

            tasks = registry.list_tasks()
            print("\n🔧 Available analysis tasks:")
            for task in sorted(tasks, key=lambda t: t.name):
                print(f"  - {task.name}")
            print()
            sys.exit(0)

        # Parse task list
        tasks = None
        if args.tasks:
            tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

        # Determine LLM usage
        use_llm = not args.no_llm

        # Show analysis plan
        print(f"\n🔍 Analyzing {args.file.name}")
        if args.sheet:
            print(f"   📄 Sheet: {args.sheet}")
        if tasks:
            print(f"   🔧 Tasks: {', '.join(tasks)}")
        else:
            print("   🔧 Tasks: auto-detect all available")

        if use_llm:
            print(f"   🧠 LLM: {args.model} (max ${args.max_cost})")
        else:
            print("   📋 Mode: Deterministic only (no LLM)")

        print(f"   💾 Output: {args.output_dir}")

        if args.dry_run:
            print("\n🚀 DRY RUN - Would perform analysis with above settings")
            sys.exit(0)

        print()

        # Run analysis
        output_path = asyncio.run(
            analyze_spreadsheet(
                file_path=args.file,
                sheet_name=args.sheet,
                use_llm=use_llm,
                model=args.model,
                max_cost=args.max_cost,
                temperature=args.temperature,
                tasks=tasks,
                output_dir=args.output_dir,
                kernel=args.kernel,
                timeout=args.timeout,
                quality_checks=not args.no_quality_checks,
                verbose=args.verbose,
            )
        )

        # Success message
        print("✅ Analysis complete!")
        print(f"   📓 Notebook: {output_path}")
        print("\n🚀 Open in Jupyter:")
        print(f'   jupyter notebook "{output_path}"')
        print()

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n❌ File Error: {e}", file=sys.stderr)
        sys.exit(1)

    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}", file=sys.stderr)
        sys.exit(2)

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"\n❌ Analysis Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
