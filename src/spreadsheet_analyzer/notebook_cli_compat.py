"""Backward compatibility module for notebook_cli.

This module provides the old class-based API for backward compatibility,
delegating to the new functional implementation.

CLAUDE-KNOWLEDGE: This allows existing code that imports NotebookCLI,
StructuredFileNameGenerator, or PipelineResultsToMarkdown to continue working.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from structlog import get_logger

from .cli.notebook_analysis import (
    AnalysisArtifacts,
    AnalysisConfig,
    run_notebook_analysis,
)
from .cli.utils.markdown import pipeline_to_markdown
from .cli.utils.naming import (
    FileNameConfig,
    generate_log_name,
    generate_notebook_name,
    generate_session_id,
    get_cost_tracking_path,
    sanitize_model_name,
    sanitize_sheet_name,
)
from .observability import PhoenixConfig
from .pipeline.types import PipelineResult


class StructuredFileNameGenerator:
    """Generate structured file names for notebooks and logs.

    Backward compatibility wrapper around functional naming module.
    """

    def __init__(
        self,
        excel_file: Path,
        model: str,
        sheet_index: int,
        sheet_name: str | None = None,
        max_rounds: int = 5,
        session_id: str | None = None,
    ):
        """Initialize the file name generator."""
        self.excel_file = excel_file
        self.model = sanitize_model_name(model)
        self.sheet_index = sheet_index
        self.sheet_name = sanitize_sheet_name(sheet_name) if sheet_name else None
        self.max_rounds = max_rounds
        self.session_id = session_id

        # Create immutable config for functional module
        self._config = FileNameConfig(
            excel_file=excel_file,
            model=model,
            sheet_index=sheet_index,
            sheet_name=sheet_name,
            max_rounds=max_rounds,
            session_id=session_id,
        )

    def _sanitize_model_name(self, model: str) -> str:
        """Sanitize model name for use in file names."""
        return sanitize_model_name(model)

    def _sanitize_sheet_name(self, sheet_name: str) -> str:
        """Sanitize sheet name for use in file names."""
        return sanitize_sheet_name(sheet_name)

    def generate_notebook_name(self, include_timestamp: bool = False) -> str:
        """Generate a structured notebook file name."""
        return generate_notebook_name(self._config, include_timestamp)

    def get_cost_tracking_path(self, output_dir: Path | None = None) -> Path:
        """Get the path for the cost tracking file."""
        return get_cost_tracking_path(self._config, output_dir)

    def generate_log_name(self, include_timestamp: bool = True) -> str:
        """Generate a structured log file name."""
        return generate_log_name(self._config, include_timestamp)


class PipelineResultsToMarkdown:
    """Convert pipeline results to markdown cells.

    Backward compatibility wrapper around functional markdown module.
    """

    def __init__(self, result: PipelineResult):
        """Initialize with pipeline result."""
        self.result = result

    def generate_cells(self) -> list[str]:
        """Generate markdown cells from pipeline results."""
        return pipeline_to_markdown(self.result)


class NotebookCLI:
    """CLI interface for automated Excel analysis.

    Backward compatibility wrapper around functional implementation.
    """

    def __init__(self):
        """Initialize the CLI."""
        self.parser = self._create_parser()

    def _create_parser(self):
        """Create argument parser."""
        # Duplicate the parser creation to avoid circular import
        parser = argparse.ArgumentParser(
            description="Automated Excel analysis using LLM function calling with Phoenix observability.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic analysis with default Claude model
  %(prog)s data.xlsx

  # Use GPT-4 instead of Claude
  %(prog)s data.xlsx --model gpt-4 --sheet-index 1

  # Use a specific Claude model
  %(prog)s data.xlsx --model claude-3-opus-20240229

  # Custom output location and session
  %(prog)s data.xlsx --output-dir results --session-id analysis-001

  # Configure Phoenix observability
  %(prog)s data.xlsx --phoenix-mode docker --phoenix-host localhost

  # Set cost limit
  %(prog)s data.xlsx --cost-limit 5.0
""",
        )

        # Positional arguments
        parser.add_argument("excel_file", type=Path, help="Path to the Excel file to analyze")

        # Model configuration
        parser.add_argument(
            "--model",
            type=str,
            default="claude-3-5-sonnet-20241022",
            help="LLM model to use for analysis (default: claude-3-5-sonnet-20241022)",
        )
        parser.add_argument(
            "--api-key",
            type=str,
            help="API key for the LLM provider (defaults to environment variable)",
        )

        # Sheet selection
        parser.add_argument(
            "--sheet-index",
            type=int,
            default=0,
            help="Index of the sheet to analyze (0-based, default: 0)",
        )
        parser.add_argument(
            "--sheet-name",
            type=str,
            help="Name of the sheet to analyze (overrides --sheet-index if provided)",
        )

        # Analysis configuration
        parser.add_argument(
            "--max-rounds",
            type=int,
            default=5,
            help="Maximum number of analysis rounds (default: 5)",
        )

        # Output configuration
        parser.add_argument(
            "--output-dir",
            type=Path,
            help="Directory to save output files (defaults to current directory)",
        )
        parser.add_argument(
            "--session-id",
            type=str,
            help="Custom session ID (defaults to auto-generated)",
        )
        parser.add_argument(
            "--notebook-path",
            type=Path,
            help="Custom path for the output notebook",
        )

        # Logging
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose logging",
        )

        # Phoenix observability configuration
        parser.add_argument(
            "--phoenix-mode",
            type=str,
            default="none",
            choices=["none", "docker", "cloud"],
            help="Phoenix observability mode (default: none)",
        )
        parser.add_argument(
            "--phoenix-host",
            type=str,
            default="localhost",
            help="Phoenix host for docker mode (default: localhost)",
        )
        parser.add_argument(
            "--phoenix-port",
            type=int,
            default=6006,
            help="Phoenix port for docker mode (default: 6006)",
        )
        parser.add_argument(
            "--phoenix-api-key",
            type=str,
            help="Phoenix API key for cloud mode (defaults to PHOENIX_API_KEY env var)",
        )
        parser.add_argument(
            "--phoenix-project",
            type=str,
            default="spreadsheet-analyzer",
            help="Phoenix project name (default: spreadsheet-analyzer)",
        )

        # Cost tracking
        parser.add_argument(
            "--cost-limit",
            type=float,
            help="Maximum cost limit in USD (analysis stops if exceeded)",
        )
        parser.add_argument(
            "--no-cost-tracking",
            action="store_true",
            help="Disable cost tracking (default: enabled)",
        )
        parser.add_argument(
            "--track-costs",
            dest="track_costs",
            action="store_true",
            default=True,
            help="Enable cost tracking (default: True)",
        )

        return parser

    async def run_analysis(self, args):
        """Run the automated analysis loop."""
        # Create configuration
        excel_path = args.excel_file.resolve()

        phoenix_config = None
        if args.phoenix_mode != "none":
            phoenix_config = PhoenixConfig(
                mode=args.phoenix_mode,
                host=args.phoenix_host,
                port=args.phoenix_port,
                api_key=args.phoenix_api_key or os.getenv("PHOENIX_API_KEY"),
                project_name=args.phoenix_project,
            )

        config = AnalysisConfig(
            excel_path=excel_path,
            sheet_index=args.sheet_index,
            sheet_name=args.sheet_name,
            model=args.model,
            api_key=args.api_key,
            max_rounds=args.max_rounds,
            output_dir=args.output_dir,
            session_id=args.session_id,
            notebook_path=args.notebook_path,
            verbose=args.verbose,
            phoenix_config=phoenix_config,
            track_costs=args.track_costs and not args.no_cost_tracking,
            cost_limit=args.cost_limit,
        )

        # Create artifacts
        file_config = FileNameConfig(
            excel_file=config.excel_path,
            model=config.model,
            sheet_index=config.sheet_index,
            sheet_name=config.sheet_name,
            max_rounds=config.max_rounds,
            session_id=config.session_id,
        )

        session_id = config.session_id or generate_session_id(file_config)

        if config.notebook_path:
            notebook_path = config.notebook_path
        else:
            output_dir = config.output_dir or Path.cwd()
            output_dir.mkdir(parents=True, exist_ok=True)
            notebook_name = generate_notebook_name(file_config, include_timestamp=True)
            notebook_path = output_dir / notebook_name

        log_dir = notebook_path.parent
        log_name = generate_log_name(file_config, include_timestamp=True)
        log_path = log_dir / log_name

        cost_tracking_path = get_cost_tracking_path(file_config, notebook_path.parent)

        artifacts = AnalysisArtifacts(
            session_id=session_id,
            notebook_path=notebook_path,
            log_path=log_path,
            cost_tracking_path=cost_tracking_path,
            file_config=file_config,
        )

        # Run analysis using functional module
        result = await run_notebook_analysis(config, artifacts)

        if result.is_err():
            raise RuntimeError(f"Analysis failed: {result.unwrap_err()}")

    def run(self):
        """Parse arguments and run the analysis."""
        args = self.parser.parse_args()

        # Setup basic logging
        log_level = logging.INFO if args.verbose else logging.WARNING
        logging.basicConfig(level=log_level, stream=sys.stdout)

        # Give a more specific logger name
        logger = get_logger(f"notebook_cli.{args.model}")

        asyncio.run(self.run_analysis(args))
