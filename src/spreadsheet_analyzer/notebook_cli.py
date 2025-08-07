#!/usr/bin/env python3
"""Command-line interface for notebook-based spreadsheet analysis.

This module provides the CLI entry point for notebook-based analysis,
delegating to functional modules for the actual implementation.

CLAUDE-KNOWLEDGE: This is now a thin wrapper around functional modules
following the principles in CONTRIBUTING.md.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from structlog import get_logger

from spreadsheet_analyzer.cli.model_registry import (
    AgentType,
    format_model_list,
    get_available_models,
    validate_model,
)
from spreadsheet_analyzer.cli.notebook_analysis import (
    AnalysisArtifacts,
    AnalysisConfig,
    run_notebook_analysis,
)
from spreadsheet_analyzer.cli.thinking_config import (
    ThinkingConfig,
    ThinkingMode,
)
from spreadsheet_analyzer.cli.utils.naming import (
    FileNameConfig,
    generate_log_name,
    generate_notebook_name,
    generate_session_id,
    get_cost_tracking_path,
    get_short_hash,
)
from spreadsheet_analyzer.observability import PhoenixConfig

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Automated Excel analysis using LLM function calling with Phoenix observability.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models with recommendations
  %(prog)s --list-models

  # Basic analysis with Claude Sonnet 3.5
  %(prog)s data.xlsx --model claude-3-5-sonnet-20241022

  # Multi-table detection workflow
  %(prog)s data.xlsx --model claude-3-5-sonnet-20241022 --multi-table

  # Use Gemini Pro 2.5 for complex analysis
  %(prog)s data.xlsx --model gemini-2.5-pro --sheet-index 1

  # Use o3 model for formula analysis
  %(prog)s data.xlsx --model o3

  # Custom output location and session
  %(prog)s data.xlsx --output-dir results --session-id analysis-001

  # Configure Phoenix observability
  %(prog)s data.xlsx --phoenix-mode docker --phoenix-host localhost

  # Set cost limit
  %(prog)s data.xlsx --cost-limit 5.0
""",
    )

    # Positional arguments (but make optional when listing models)
    parser.add_argument(
        "excel_file",
        type=Path,
        nargs="?",
        help="Path to the Excel file to analyze (not required when using --list-models)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        help="LLM model to use for analysis (required when not using --list-models)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models with agent-specific recommendations and exit",
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

    # Multi-table detection
    parser.add_argument(
        "--multi-table",
        action="store_true",
        help="Use multi-table detection workflow (experimental)",
    )

    # Detector-specific configuration
    parser.add_argument(
        "--detector-max-rounds",
        type=int,
        default=3,
        help="Maximum number of rounds for table detection agent (default: 3)",
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        help="Specific model to use for table detection (defaults to main --model)",
    )
    parser.add_argument(
        "--detector-only",
        action="store_true",
        help="Run only the table detection agent without the analyst (useful for testing)",
    )

    # Extended thinking configuration
    parser.add_argument(
        "--thinking-budget",
        type=int,
        help="Override default thinking budget in tokens (auto-determined by agent type if not specified)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Force disable extended thinking even for compatible models",
    )
    parser.add_argument(
        "--thinking-enabled",
        action="store_true",
        help="Force enable extended thinking (will warn if model doesn't support it)",
    )

    return parser


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = create_parser()
    args = parser.parse_args()

    # Handle --list-models command
    if getattr(args, "list_models", False):
        print(format_model_list())
        sys.exit(0)

    # Validate required arguments when not listing models
    if not args.excel_file:
        print("❌ Error: Excel file is required when not using --list-models")
        print("💡 Use --list-models to see available models, or provide an Excel file to analyze")
        sys.exit(1)

    # Validate model is provided when not listing models
    if not args.model:
        print("❌ Error: --model is required when not using --list-models")
        print("💡 Use --list-models to see all available models with agent-specific recommendations")
        print("\n🎯 Quick suggestions:")
        print("   For table detection: --model claude-3-5-haiku-20241022")
        print("   For general analysis: --model claude-3-5-sonnet-20241022")
        print("   For complex formulas: --model o3")
        print("   For cost efficiency: --model gpt-4.1-mini")
        sys.exit(1)

    # Validate model selection if provided
    if args.model:
        if not validate_model(args.model):
            available_models = get_available_models()
            print(f"❌ Error: Invalid model '{args.model}'")
            print("\n💡 Use --list-models to see all available models with recommendations")
            print("\n🎯 Quick suggestions:")
            print("   For table detection: claude-3-5-haiku-20241022")
            print("   For general analysis: claude-3-5-sonnet-20241022")
            print("   For complex formulas: o3")
            print("   For cost efficiency: gpt-4.1-mini")
            sys.exit(1)

    # Setup basic logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, stream=sys.stdout)

    return args


def create_analysis_config(args: argparse.Namespace) -> AnalysisConfig:
    """Create analysis configuration from parsed arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Analysis configuration
    """
    # Resolve the excel file path to be absolute
    excel_path = args.excel_file.resolve()

    # Handle sheet selection
    sheet_index = args.sheet_index
    sheet_name = args.sheet_name

    # If sheet name is provided, we'll need to find its index later
    # For now, keep the default index

    # Create Phoenix config if needed
    phoenix_config = None
    if args.phoenix_mode != "none":
        phoenix_config = PhoenixConfig(
            mode=args.phoenix_mode,
            host=args.phoenix_host,
            port=args.phoenix_port,
            api_key=args.phoenix_api_key or os.getenv("PHOENIX_API_KEY"),
            project_name=args.phoenix_project,
        )

    # Create thinking configuration
    thinking_config = None
    if hasattr(args, "no_thinking") and hasattr(args, "thinking_enabled"):
        # Determine thinking mode based on CLI arguments
        if args.no_thinking:
            thinking_mode = ThinkingMode.DISABLED
        elif args.thinking_enabled:
            thinking_mode = ThinkingMode.ENABLED
        else:
            thinking_mode = ThinkingMode.AUTO

        # Determine agent type based on workflow
        if args.detector_only:
            agent_type = AgentType.TABLE_DETECTOR
        else:
            # Default to data analyst for general analysis
            agent_type = AgentType.DATA_ANALYST

        # Create thinking config
        thinking_config = ThinkingConfig.create_for_agent(
            model_id=args.model, agent_type=agent_type, mode=thinking_mode, budget_override=args.thinking_budget
        )

        # Log thinking configuration
        if thinking_config.enabled:
            logger.info(f"🧠 Extended thinking enabled: {thinking_config.budget_tokens:,} tokens")
            if thinking_config.interleaved:
                logger.info("⚡ Interleaved thinking enabled for tool use")
        else:
            logger.info("❌ Extended thinking disabled")

    # Create the configuration
    return AnalysisConfig(
        excel_path=excel_path,
        sheet_index=sheet_index,
        sheet_name=sheet_name,
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
        detector_max_rounds=args.detector_max_rounds,
        detector_model=args.detector_model,
        detector_only=args.detector_only,
        thinking_config=thinking_config,
    )


def create_analysis_artifacts(config: AnalysisConfig) -> AnalysisArtifacts:
    """Create analysis artifacts (paths, IDs, etc.).

    Args:
        config: Analysis configuration

    Returns:
        Analysis artifacts
    """
    from spreadsheet_analyzer.prompts import get_prompt_definition

    # Determine which prompt will be used based on config
    # Note: table_boundaries is set later, so for main analysis we default to data_analyst_system
    prompt_name = (
        "table_aware_analyst_system"
        if hasattr(config, "table_boundaries") and config.table_boundaries
        else "data_analyst_system"
    )
    prompt_def = get_prompt_definition(prompt_name)
    prompt_hash = get_short_hash(prompt_def.content_hash) if prompt_def else None

    # Create file name config
    file_config = FileNameConfig(
        excel_file=config.excel_path,
        model=config.model,
        sheet_index=config.sheet_index,
        sheet_name=config.sheet_name,
        max_rounds=config.max_rounds,
        session_id=config.session_id,
        prompt_hash=prompt_hash,
    )

    # Generate session ID if not provided
    session_id = config.session_id or generate_session_id(file_config)

    # Generate notebook path if not provided
    if config.notebook_path:
        notebook_path = config.notebook_path
    else:
        output_dir = config.output_dir or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        notebook_name = generate_notebook_name(file_config, include_timestamp=True)
        notebook_path = output_dir / notebook_name

    # Generate log path
    log_dir = notebook_path.parent
    log_name = generate_log_name(file_config, include_timestamp=True)
    log_path = log_dir / log_name

    # Generate cost tracking path
    cost_tracking_path = get_cost_tracking_path(file_config, notebook_path.parent)

    return AnalysisArtifacts(
        session_id=session_id,
        notebook_path=notebook_path,
        log_path=log_path,
        cost_tracking_path=cost_tracking_path,
        file_config=file_config,
    )


async def main() -> None:
    """Main entry point for the CLI."""
    # Parse arguments
    args = parse_arguments()

    # Create configuration
    config = create_analysis_config(args)

    # Check if we should use multi-table workflow
    use_multi_table = getattr(args, "multi_table", False)

    # Log startup information
    logger.info("Starting notebook-based Excel analysis", model=config.model)
    logger.info(f"Excel file: {config.excel_path}")
    logger.info(f"Model: {config.model}")

    if not use_multi_table:
        # Only create artifacts for single-table workflow
        artifacts = create_analysis_artifacts(config)
        logger.info(f"Session ID: {artifacts.session_id}")
        logger.info(f"Output notebook: {artifacts.notebook_path}")

    if use_multi_table:
        logger.info("Using multi-table detection workflow")
        from spreadsheet_analyzer.workflows.multi_table_workflow import run_multi_table_analysis

        # Run multi-table analysis
        result = await run_multi_table_analysis(config.excel_path, sheet_index=config.sheet_index, config=config)

        if result.is_ok():
            analysis = result.unwrap()
            logger.info(f"Multi-table analysis complete: {analysis['tables_found']} tables found")
            if analysis["detection_notebook"]:
                logger.info(f"Detection notebook: {analysis['detection_notebook']}")
            if analysis["analysis_notebook"]:
                logger.info(f"Analysis notebook: {analysis['analysis_notebook']}")
            # Success - multi-table workflow handles everything
            return
        else:
            # Fall back to standard analysis
            logger.warning(f"Multi-table workflow failed: {result.unwrap_err()}")
            logger.info("Falling back to standard analysis")
            # Create artifacts for fallback
            artifacts = create_analysis_artifacts(config)
            logger.info(f"Session ID: {artifacts.session_id}")
            logger.info(f"Output notebook: {artifacts.notebook_path}")

    # Run the standard analysis using the functional orchestration (only if not multi-table or fallback)
    if not use_multi_table or "artifacts" in locals():
        result = await run_notebook_analysis(config, artifacts)

        if result.is_err():
            logger.error(f"Analysis failed: {result.unwrap_err()}")
            sys.exit(1)
        else:
            logger.info("Analysis completed successfully")


def run() -> None:
    """Run the CLI application."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
