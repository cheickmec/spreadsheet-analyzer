"""Analyze command for single Excel file analysis."""

import asyncio
import json
from pathlib import Path

import click
import yaml

from spreadsheet_analyzer.cli.console import RichConsoleHandler
from spreadsheet_analyzer.logging_config import get_logger
from spreadsheet_analyzer.pipeline.types import AnalysisOptions, AnalysisResult
from spreadsheet_analyzer.services import AnalysisService

logger = get_logger(__name__)


@click.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["fast", "standard", "deep"]),
    default="standard",
    help="Analysis depth and thoroughness",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Save results to file (format auto-detected from extension)",
)
@click.option("--no-formulas", is_flag=True, help="Skip formula analysis")
@click.option("--no-security", is_flag=True, help="Skip security scanning")
@click.option("--no-content", is_flag=True, help="Skip content intelligence")
@click.pass_context
def analyze(
    ctx: click.Context, file: Path, mode: str, output: Path, no_formulas: bool, no_security: bool, no_content: bool
) -> None:
    """Analyze a single Excel file.

    This command performs comprehensive analysis on an Excel file including:

    - File integrity and format validation
    - Security scanning for macros and external links
    - Structural mapping of sheets and ranges
    - Formula dependency analysis
    - Content pattern detection

    Examples:

        # Basic analysis
        spreadsheet-analyzer analyze financial-model.xlsx

        # Fast mode with JSON output
        spreadsheet-analyzer analyze data.xlsx --mode fast -o results.json

        # Skip formula analysis for faster results
        spreadsheet-analyzer analyze large-file.xlsx --no-formulas
    """
    # Get settings from context
    verbosity = ctx.obj.get("verbosity", 0)
    output_format = ctx.obj.get("format", "table")
    quiet = ctx.obj.get("quiet", False)
    no_color = ctx.obj.get("no_color", False)

    # Log start
    logger.info("Starting analysis", file=str(file), mode=mode, verbosity=verbosity)

    # Create console handler
    console = RichConsoleHandler(no_color=no_color)

    # Show header unless quiet
    if not quiet:
        console.print_header(f"Analyzing {file.name}")

    # Create service and options
    service = AnalysisService()

    # Create progress callback for non-quiet mode
    progress_callback = console.create_progress_callback() if not quiet else None

    options = AnalysisOptions(
        mode=mode,
        include_formulas=not no_formulas,
        include_security=not no_security,
        include_content=not no_content,
        progress_callback=progress_callback,
    )

    try:
        # Run analysis
        result = asyncio.run(service.analyze_file(file, options))

        # Stop progress display
        console.stop_progress()

        # Display results based on format
        if not quiet:
            console.display_results(result, output_format)

        # Save to file if requested
        if output:
            _save_results(result, output, output_format)
            if not quiet:
                console.print_success(f"Results saved to {output}")

        # Exit with appropriate code
        exit_code = 0 if result.is_healthy else 1
        if not quiet and exit_code != 0:
            console.print_warning(f"Analysis found {len(result.issues)} issue(s) and {len(result.warnings)} warning(s)")

        ctx.exit(exit_code)

    except FileNotFoundError:
        console.stop_progress()
        logger.exception("File not found", file=str(file))
        console.print_error(f"File not found: {file}")
        ctx.exit(2)

    except click.exceptions.Exit:
        # Re-raise Click exits (normal behavior)
        raise

    except Exception as e:
        console.stop_progress()
        logger.exception("Analysis failed", file=str(file))
        console.print_error(f"Analysis failed: {e}")
        ctx.exit(3)


def _save_results(result: AnalysisResult, output_path: Path, format: str) -> None:
    """Save results to file."""
    # Auto-detect format from extension if not specified
    if output_path.suffix == ".json":
        format = "json"
    elif output_path.suffix in [".yml", ".yaml"]:
        format = "yaml"
    elif output_path.suffix == ".md":
        format = "markdown"

    # Convert to appropriate format
    if format == "json":
        content = json.dumps(result.to_dict(), indent=2)
    elif format == "yaml":
        content = yaml.dump(result.to_dict(), default_flow_style=False)
    else:
        # For table/markdown, capture the display output
        # This is a simple approach - could be improved
        content = f"Analysis Results for {result.file_path}\n"
        content += "=" * 50 + "\n"
        content += json.dumps(result.to_dict(), indent=2)

    # Write to file
    output_path.write_text(content, encoding="utf-8")
