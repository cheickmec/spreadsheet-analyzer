"""CLI interface for Spreadsheet Analyzer.

This module provides a rich terminal interface for Excel analysis with
comprehensive logging and progress tracking. Designed to be easily extensible
with future FastAPI endpoints.
"""

import click

from spreadsheet_analyzer.cli import commands

__version__ = "0.4.0"


@click.group()
@click.option("--verbose", "-v", count=True, help="Increase verbosity (-v, -vv, -vvv)")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
@click.option(
    "--format", type=click.Choice(["json", "yaml", "table", "markdown"]), default="table", help="Output format"
)
@click.option("--no-color", is_flag=True, help="Disable colored output")
@click.version_option(version=__version__, prog_name="spreadsheet-analyzer")
@click.pass_context
def cli(ctx: click.Context, verbose: int, quiet: bool, format: str, no_color: bool) -> None:
    """Spreadsheet Analyzer - Intelligent Excel Analysis.

    Analyze Excel files to reveal hidden structures, relationships, and potential
    issues using a combination of deterministic parsing and AI-powered insights.
    """
    # Ensure that ctx.obj exists
    ctx.ensure_object(dict)

    # Store configuration in context
    ctx.obj["verbosity"] = 0 if quiet else verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["format"] = format
    ctx.obj["no_color"] = no_color

    # Setup logging based on verbosity
    _setup_logging(verbose=ctx.obj["verbosity"], no_color=no_color)


def _setup_logging(*, verbose: int, no_color: bool) -> None:
    """Configure logging based on verbosity level."""
    from spreadsheet_analyzer.logging_config import setup_logging

    setup_logging(verbosity=verbose, no_color=no_color)


# Register commands
cli.add_command(commands.analyze)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
