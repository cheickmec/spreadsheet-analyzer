#!/usr/bin/env python3
"""
Notebook Tools CLI with Phoenix Observability

Automated Excel analysis using LLM function calling with the notebook tools interface,
enhanced with Phoenix tracing and LiteLLM cost tracking.
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from structlog import get_logger

from spreadsheet_analyzer.notebook_session import notebook_session

# Import observability components
from spreadsheet_analyzer.observability import (
    PhoenixConfig,
    get_cost_tracker,
    initialize_cost_tracker,
    initialize_phoenix,
    instrument_all,
)
from spreadsheet_analyzer.pipeline import DeterministicPipeline
from spreadsheet_analyzer.pipeline.types import (
    ContentAnalysis,
    FormulaAnalysis,
    IntegrityResult,
    PipelineResult,
    SecurityReport,
    WorkbookStructure,
)

logger = get_logger(__name__)


class StructuredFileNameGenerator:
    """Generate structured file names for notebooks and logs with all relevant parameters."""

    def __init__(
        self,
        excel_file: Path,
        model: str,
        sheet_index: int,
        sheet_name: str | None = None,
        max_rounds: int = 5,
        session_id: str | None = None,
    ):
        """
        Initialize the file name generator.

        Args:
            excel_file: Path to the Excel file being analyzed
            model: LLM model name being used
            sheet_index: Index of the sheet being analyzed
            sheet_name: Name of the sheet being analyzed (optional)
            max_rounds: Maximum number of analysis rounds
            session_id: Custom session ID (optional)
        """
        self.excel_file = excel_file
        self.model = self._sanitize_model_name(model)
        self.sheet_index = sheet_index
        self.sheet_name = self._sanitize_sheet_name(sheet_name) if sheet_name else None
        self.max_rounds = max_rounds
        self.session_id = session_id

    def _sanitize_model_name(self, model: str) -> str:
        """Sanitize model name for use in file names."""
        # Remove version suffixes and special characters
        model_clean = model.replace("-", "_").replace(".", "_")
        # Extract the main model name (e.g., "claude_3_5_sonnet" from "claude-3-5-sonnet-20241022")
        if "claude" in model_clean.lower():
            # Extract Claude model variant
            if "opus" in model_clean.lower():
                return "claude_opus"
            elif "sonnet" in model_clean.lower():
                return "claude_sonnet"
            elif "haiku" in model_clean.lower():
                return "claude_haiku"
            else:
                return "claude"
        elif "gpt" in model_clean.lower():
            # Extract GPT model variant
            if "4" in model_clean:
                return "gpt4"
            elif "3" in model_clean:
                return "gpt3"
            else:
                return "gpt"
        else:
            # For other models, use a simplified version
            return model_clean.split("_")[0] if "_" in model_clean else model_clean

    def _sanitize_sheet_name(self, sheet_name: str) -> str:
        """Sanitize sheet name for use in file names."""
        # Remove or replace characters that are problematic in file names
        # Replace spaces, special characters with underscores
        sanitized = re.sub(r"[^\w\-_.]", "_", sheet_name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        # Limit length to avoid filesystem issues
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        return sanitized or f"sheet_{self.sheet_index}"

    def generate_base_name(self) -> str:
        """Generate the base name for files with all parameters."""
        parts = [
            self.excel_file.stem,  # Original file name without extension
            self.model,  # Model identifier
            f"sheet{self.sheet_index}",  # Sheet index
        ]

        # Add sheet name if available and different from index
        if self.sheet_name:
            parts.append(self.sheet_name)

        # Add max rounds if not default
        if self.max_rounds != 5:
            parts.append(f"rounds{self.max_rounds}")

        # Add custom session ID if provided
        if self.session_id:
            parts.append(self.session_id)

        return "_".join(parts)

    def get_notebook_path(self, output_dir: Path | None = None) -> Path:
        """Get the path for the notebook file."""
        base_name = self.generate_base_name()
        notebook_name = f"{base_name}.ipynb"

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / notebook_name
        else:
            # Use analysis_results directory with date organization
            date_str = datetime.now().strftime("%Y%m%d")
            results_dir = Path("analysis_results") / date_str
            results_dir.mkdir(parents=True, exist_ok=True)
            return results_dir / notebook_name

    def get_log_path(self, output_dir: Path | None = None) -> Path:
        """Get the path for the LLM interaction log file."""
        base_name = self.generate_base_name()
        log_name = f"{base_name}_llm_log.txt"

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / log_name
        else:
            # Use logs directory with date organization
            date_str = datetime.now().strftime("%Y%m%d")
            logs_dir = Path("logs") / date_str
            logs_dir.mkdir(parents=True, exist_ok=True)
            return logs_dir / log_name

    def get_cost_tracking_path(self, output_dir: Path | None = None) -> Path:
        """Get the path for the cost tracking file."""
        base_name = self.generate_base_name()
        cost_name = f"{base_name}_cost_tracking.json"

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir / cost_name
        else:
            # Use logs directory with date organization
            date_str = datetime.now().strftime("%Y%m%d")
            logs_dir = Path("logs") / date_str
            logs_dir.mkdir(parents=True, exist_ok=True)
            return logs_dir / cost_name


def configure_logging(log_path: Path):
    """Configure logging to also write to file."""
    # Create a file handler for LLM interactions
    llm_logger = logging.getLogger("llm_interactions")
    llm_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    llm_logger.handlers = []

    # File handler with UTF-8 encoding for emojis
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(message)s")  # Simple format for LLM logs
    fh.setFormatter(formatter)

    # Add handler
    llm_logger.addHandler(fh)

    # Also log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    llm_logger.addHandler(ch)

    # Prevent propagation to root logger
    llm_logger.propagate = False

    return llm_logger


async def track_llm_usage(response: Any, model: str) -> None:
    """
    Track token usage from LLM response.

    Args:
        response: LLM response object
        model: Model name
    """
    try:
        # Extract usage metadata from response
        usage_metadata = None

        # Try different ways to get usage data based on provider
        if hasattr(response, "usage_metadata"):
            usage_metadata = response.usage_metadata
        elif hasattr(response, "usage"):
            usage_metadata = response.usage
        elif hasattr(response, "response_metadata"):
            usage_metadata = response.response_metadata.get("usage", {})

        if usage_metadata:
            input_tokens = (
                usage_metadata.get("input_tokens", 0)
                or usage_metadata.get("prompt_tokens", 0)
                or usage_metadata.get("total_tokens", 0) // 2  # Rough estimate
            )
            output_tokens = (
                usage_metadata.get("output_tokens", 0)
                or usage_metadata.get("completion_tokens", 0)
                or usage_metadata.get("total_tokens", 0) // 2  # Rough estimate
            )

            if input_tokens > 0 or output_tokens > 0:
                cost_tracker = get_cost_tracker()
                cost_tracker.track_usage(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    metadata={"source": "notebook_cli"},
                )

    except Exception as e:
        logger.debug(f"Failed to track LLM usage: {e}")


# Add this function to the original notebook_cli content
def format_security_section(security: SecurityReport) -> str:
    """Format security analysis for markdown display."""
    content = []

    # Risk level with emoji
    risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(security.risk_level, "⚪")
    content.append(f"**Risk Level**: {risk_emoji} {security.risk_level.upper()}")

    # Macro status
    if security.has_macros:
        macro_status = "⚠️ **Macros detected** - Review required"
    else:
        macro_status = "✅ No macros found"
    content.append(f"**Macro Status**: {macro_status}")

    # External links
    if security.external_links:
        content.append(f"\n**External Links** ({len(security.external_links)} found):")
        for link in security.external_links[:5]:  # Show first 5
            content.append(f"- `{link}`")
        if len(security.external_links) > 5:
            content.append(f"- ... and {len(security.external_links) - 5} more")
    else:
        content.append("\n**External Links**: None found")

    # Data validation
    if security.data_validation_rules:
        content.append(f"\n**Data Validation**: {security.data_validation_rules} rules configured")

    # Hidden elements
    if security.hidden_sheets or security.hidden_rows or security.hidden_columns:
        content.append("\n**Hidden Elements**:")
        if security.hidden_sheets:
            content.append(f"- Hidden sheets: {', '.join(security.hidden_sheets)}")
        if security.hidden_rows:
            content.append(f"- Hidden rows found in: {', '.join(security.hidden_rows.keys())}")
        if security.hidden_columns:
            content.append(f"- Hidden columns found in: {', '.join(security.hidden_columns.keys())}")

    return "\n".join(content)


def format_structure_section(structure: WorkbookStructure) -> str:
    """Format structure analysis for markdown display."""
    content = []

    content.append(f"**Total Sheets**: {structure.sheet_count}")
    content.append(f"**Total Cells**: {structure.total_cells:,}")
    content.append(f"**Total Formulas**: {structure.total_formulas:,}")
    content.append(f"**Complexity Score**: {structure.complexity_score}/100")

    # Sheet details
    content.append("\n**Sheet Details**:")
    for sheet in structure.sheets[:5]:  # Show first 5 sheets
        content.append(f"- **{sheet.name}**: {sheet.cell_count:,} cells, {sheet.formula_count} formulas")
        if sheet.has_charts:
            content.append("  - Has charts")
        if sheet.has_pivot_tables:
            content.append("  - Has pivot tables")
        if sheet.named_ranges:
            content.append(f"  - Named ranges: {len(sheet.named_ranges)}")

    if len(structure.sheets) > 5:
        content.append(f"... and {len(structure.sheets) - 5} more sheets")

    # Additional structural information
    if structure.has_vba_project:
        content.append("\n⚠️ **VBA Project detected**")
    if structure.has_external_links:
        content.append("⚠️ **External links detected**")
    if structure.named_ranges:
        content.append(f"\n**Named Ranges**: {len(structure.named_ranges)}")
        for nr in structure.named_ranges[:3]:
            content.append(f"- {nr}")
        if len(structure.named_ranges) > 3:
            content.append(f"- ... and {len(structure.named_ranges) - 3} more")

    return "\n".join(content)


def format_formulas_section(formulas: FormulaAnalysis) -> str:
    """Format formula analysis for markdown display."""
    content = []

    # Summary metrics
    total_formulas = len(formulas.dependency_graph)
    content.append(f"**Total Formulas**: {total_formulas:,}")
    content.append(f"**Volatile Formulas**: {len(formulas.volatile_formulas):,}")
    content.append(f"**Complexity Score**: {formulas.formula_complexity_score:.1f}")

    # Dependencies and issues
    if formulas.max_dependency_depth:
        content.append(f"**Max Dependency Depth**: {formulas.max_dependency_depth}")

    # External references
    if formulas.external_references:
        content.append(f"\n**External References**: {len(formulas.external_references)} found")
        for ref in list(formulas.external_references)[:3]:
            content.append(f"- {ref}")
        if len(formulas.external_references) > 3:
            content.append(f"- ... and {len(formulas.external_references) - 3} more")

    # Circular references
    if formulas.has_circular_references:
        content.append("\n⚠️ **Circular References Detected**")
        for cycle in list(formulas.circular_references)[:2]:
            cycle_str = " → ".join(list(cycle)[:5])
            if len(cycle) > 5:
                cycle_str += " → ..."
            content.append(f"- {cycle_str}")

    # Statistics
    if formulas.statistics:
        content.append("\n**Formula Statistics**:")
        for key, value in list(formulas.statistics.items())[:5]:
            content.append(f"- {key}: {value}")

    return "\n".join(content)


def format_content_section(content: ContentAnalysis) -> str:
    """Format content analysis for markdown display."""
    content_lines = []

    # Summary
    content_lines.append(f"**Total Cells with Data**: {content.total_cells:,}")
    content_lines.append(f"**Empty Cells**: {content.empty_cells:,}")
    content_lines.append(f"**Sheets Analyzed**: {content.sheets_analyzed}")

    # Content insights
    if content.insights:
        content_lines.append("\n**Key Insights**:")
        for insight in content.insights[:5]:
            content_lines.append(f"- {insight}")
        if len(content.insights) > 5:
            content_lines.append(f"- ... and {len(content.insights) - 5} more insights")

    # Content patterns
    if hasattr(content, "patterns_found") and content.patterns_found:
        content_lines.append("\n**Patterns Detected**:")
        for pattern in content.patterns_found[:3]:
            content_lines.append(f"- {pattern}")

    # Anomalies
    if hasattr(content, "anomalies") and content.anomalies:
        content_lines.append("\n**Anomalies Found**:")
        for anomaly in content.anomalies[:3]:
            content_lines.append(f"- {anomaly}")

    return "\n".join(content_lines)


def format_integrity_section(integrity: IntegrityResult) -> str:
    """Format integrity check results for markdown display."""
    content = []

    # Overall status
    status_emoji = "✅" if integrity.is_valid else "❌"
    content.append(f"**Overall Status**: {status_emoji} {'Valid' if integrity.is_valid else 'Issues Found'}")

    # Issues by severity
    if integrity.issues:
        issues_by_severity = {"error": [], "warning": [], "info": []}
        for issue in integrity.issues:
            severity = getattr(issue, "severity", "info")
            issues_by_severity[severity].append(issue)

        if issues_by_severity["error"]:
            content.append(f"\n**Errors** ({len(issues_by_severity['error'])}):")
            for issue in issues_by_severity["error"][:3]:
                content.append(f"- ❌ {issue}")
            if len(issues_by_severity["error"]) > 3:
                content.append(f"- ... and {len(issues_by_severity['error']) - 3} more errors")

        if issues_by_severity["warning"]:
            content.append(f"\n**Warnings** ({len(issues_by_severity['warning'])}):")
            for issue in issues_by_severity["warning"][:3]:
                content.append(f"- ⚠️ {issue}")
            if len(issues_by_severity["warning"]) > 3:
                content.append(f"- ... and {len(issues_by_severity['warning']) - 3} more warnings")

    else:
        content.append("\n✅ No integrity issues found")

    return "\n".join(content)


def create_pipeline_summary_tool(pipeline_result: PipelineResult):
    """Create a tool that provides the pipeline analysis summary."""

    @tool
    async def get_excel_analysis_summary() -> str:
        """Get the comprehensive Excel file analysis summary from the deterministic pipeline.

        Returns a detailed markdown report of the Excel file analysis including:
        - Security assessment
        - Workbook structure
        - Formula analysis
        - Content analysis
        - Data integrity checks
        """
        sections = []

        # Header
        sections.append("# 📊 Excel Analysis Summary\n")
        sections.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sections.append(f"**Analysis Status**: {'✅ Successful' if pipeline_result.success else '❌ Failed'}\n")

        if pipeline_result.success:
            # Security Analysis
            sections.append("## 🔒 Security Analysis")
            sections.append(format_security_section(pipeline_result.security))

            # Structure Analysis
            sections.append("\n## 📁 Structure Analysis")
            sections.append(format_structure_section(pipeline_result.structure))

            # Formula Analysis
            sections.append("\n## 🧮 Formula Analysis")
            sections.append(format_formulas_section(pipeline_result.formulas))

            # Content Analysis
            sections.append("\n## 📝 Content Analysis")
            sections.append(format_content_section(pipeline_result.content))

            # Integrity Check
            sections.append("\n## ✅ Integrity Check")
            sections.append(format_integrity_section(pipeline_result.integrity))

            # Summary statistics
            sections.append("\n## 📈 Summary Statistics")
            sections.append(f"- **Total Sheets**: {pipeline_result.structure.sheet_count}")
            sections.append(f"- **Total Cells**: {pipeline_result.structure.total_cells:,}")
            sections.append(f"- **Total Formulas**: {len(pipeline_result.formulas.dependency_graph):,}")
            sections.append(f"- **Formula Complexity**: {pipeline_result.formulas.formula_complexity_score:.1f}")
            sections.append(f"- **Security Risk**: {pipeline_result.security.risk_level}")

        else:
            sections.append("## ❌ Analysis Failed")
            sections.append(f"**Error**: {pipeline_result.error}")

        return "\n".join(sections)

    return get_excel_analysis_summary


class NotebookCLI:
    """CLI for notebook-based Excel analysis."""

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self):
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            description="Analyze Excel files using LLM-powered notebook interface with Phoenix observability",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic analysis with default model
  %(prog)s data.xlsx

  # Use specific model and sheet
  %(prog)s data.xlsx --model gpt-4 --sheet-index 1

  # Custom output location and session
  %(prog)s data.xlsx --output-dir results --session-id analysis-001

  # Set maximum analysis rounds
  %(prog)s data.xlsx --max-rounds 10

  # Configure Phoenix observability
  %(prog)s data.xlsx --phoenix-mode docker --phoenix-host localhost

  # Set cost limit
  %(prog)s data.xlsx --cost-limit 5.0
""",
        )

        parser.add_argument("excel_file", type=Path, help="Path to Excel file to analyze")

        parser.add_argument(
            "-m",
            "--model",
            default="claude-3-5-sonnet-20241022",
            help="LLM model to use (default: claude-3-5-sonnet-20241022)",
        )

        parser.add_argument(
            "-s",
            "--sheet-index",
            type=int,
            default=0,
            help="Sheet index to analyze (default: 0)",
        )

        parser.add_argument(
            "-o",
            "--output-dir",
            type=Path,
            help="Output directory for results (default: analysis_results/YYYYMMDD/)",
        )

        parser.add_argument(
            "-n",
            "--notebook-path",
            type=Path,
            help="Specific path for output notebook (overrides output-dir)",
        )

        parser.add_argument(
            "--session-id",
            help="Custom session ID for the analysis",
        )

        parser.add_argument(
            "--max-rounds",
            type=int,
            default=5,
            help="Maximum number of analysis rounds (default: 5)",
        )

        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

        # Phoenix observability options
        phoenix_group = parser.add_argument_group("Phoenix Observability")
        phoenix_group.add_argument(
            "--phoenix-mode",
            choices=["local", "cloud", "docker", "none"],
            default="local",
            help="Phoenix deployment mode (default: local)",
        )
        phoenix_group.add_argument(
            "--phoenix-host",
            default="localhost",
            help="Phoenix host for docker mode (default: localhost)",
        )
        phoenix_group.add_argument(
            "--phoenix-port",
            type=int,
            default=6006,
            help="Phoenix port for local mode (default: 6006)",
        )
        phoenix_group.add_argument(
            "--phoenix-api-key",
            help="Phoenix API key for cloud mode (or set PHOENIX_API_KEY env var)",
        )
        phoenix_group.add_argument(
            "--phoenix-project",
            default="spreadsheet-analyzer",
            help="Phoenix project name (default: spreadsheet-analyzer)",
        )

        # Cost tracking options
        cost_group = parser.add_argument_group("Cost Tracking")
        cost_group.add_argument(
            "--cost-limit",
            type=float,
            help="Set spending limit in USD",
        )
        cost_group.add_argument(
            "--track-costs",
            action="store_true",
            default=True,
            help="Enable cost tracking (default: True)",
        )

        return parser

    async def run_analysis(self, args):
        """Run the Excel analysis with notebook interface."""
        excel_file = args.excel_file
        if not excel_file.exists():
            logger.error(f"Excel file not found: {excel_file}")
            sys.exit(1)

        # Initialize Phoenix observability
        tracer_provider = None
        if args.phoenix_mode != "none":
            phoenix_config = PhoenixConfig(
                mode=args.phoenix_mode,
                host=args.phoenix_host,
                port=args.phoenix_port,
                api_key=args.phoenix_api_key or os.getenv("PHOENIX_API_KEY"),
                project_name=args.phoenix_project,
            )
            tracer_provider = initialize_phoenix(phoenix_config)

            if tracer_provider:
                # Instrument all providers
                results = instrument_all(tracer_provider)
                logger.info("Phoenix instrumentation complete", results=results)
            else:
                logger.warning("Phoenix initialization failed, continuing without observability")

        # Generate file paths
        name_generator = StructuredFileNameGenerator(
            excel_file=excel_file,
            model=args.model,
            sheet_index=args.sheet_index,
            sheet_name=None,  # Will be determined after loading the file
            max_rounds=args.max_rounds,
            session_id=args.session_id,
        )

        # Determine notebook path
        if args.notebook_path:
            notebook_path = args.notebook_path
        else:
            notebook_path = name_generator.get_notebook_path(args.output_dir)

        # Set up logging
        log_path = name_generator.get_log_path(args.output_dir)
        llm_logger = configure_logging(log_path)
        logger.info(f"📝 LLM interaction log: {log_path}")

        # Initialize cost tracking
        cost_tracking_path = name_generator.get_cost_tracking_path(args.output_dir)
        if args.track_costs:
            cost_tracker = initialize_cost_tracker(cost_limit=args.cost_limit, save_path=cost_tracking_path)
            logger.info(f"💰 Cost tracking enabled: {cost_tracking_path}")
            if args.cost_limit:
                logger.info(f"💰 Cost limit set: ${args.cost_limit:.2f}")

        # First, run the deterministic pipeline
        logger.info(f"🔍 Running deterministic analysis on: {excel_file}")
        pipeline = DeterministicPipeline()
        pipeline_result = pipeline.run(excel_file)  # Pass Path object directly

        if not pipeline_result.success:
            logger.error(f"❌ Pipeline analysis failed: {getattr(pipeline_result, 'error', 'Unknown error')}")
            if hasattr(pipeline_result, "errors") and pipeline_result.errors:
                for error in pipeline_result.errors:
                    logger.error(f"  - {error}")
            sys.exit(1)

        logger.info("✅ Deterministic analysis complete")

        # Display quick summary
        logger.info(f"📊 File: {excel_file.name}")
        logger.info(f"📑 Sheets: {pipeline_result.structure.sheet_count}")
        logger.info(f"🧮 Formulas: {len(pipeline_result.formulas.dependency_graph)}")
        logger.info(f"🔒 Security Risk: {pipeline_result.security.risk_level}")

        # Get sheet name from pipeline results
        sheet_name = None
        if pipeline_result.structure.sheets and args.sheet_index < len(pipeline_result.structure.sheets):
            sheet_name = pipeline_result.structure.sheets[args.sheet_index].name
            name_generator.sheet_name = name_generator._sanitize_sheet_name(sheet_name)
            # Regenerate paths with sheet name
            if not args.notebook_path:
                notebook_path = name_generator.get_notebook_path(args.output_dir)

        # Save formula analysis to pickle cache if available
        formula_cache_path = None
        if pipeline_result.formulas:
            import pickle

            cache_dir = Path(".pipeline_cache")
            cache_dir.mkdir(exist_ok=True)

            # Create a unique filename based on the Excel file
            cache_filename = f"{excel_file.stem}_formula_analysis.pkl"
            formula_cache_path = cache_dir / cache_filename

            with open(formula_cache_path, "wb") as f:
                pickle.dump(pipeline_result.formulas, f)

            logger.info(f"Saved formula analysis to cache: {formula_cache_path}")

        # Initialize enhanced query interface for formulas
        logger.info("🔗 Initializing formula graph database...")
        enhanced_interface = None
        if pipeline_result.formulas.dependency_graph:
            from spreadsheet_analyzer.graph_db.query_interface import create_enhanced_query_interface

            enhanced_interface = create_enhanced_query_interface(
                formula_analysis=pipeline_result.formulas,
                # Neo4j parameters not provided - will use in-memory NetworkX only
            )

        # Create session with enhanced tools
        session_id = args.session_id or f"excel_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"🚀 Starting notebook session: {session_id}")
        logger.info(f"📓 Output notebook: {notebook_path}")

        async with notebook_session(session_id, notebook_path) as session:
            # Create initial analysis context cell
            context_md = f"""# Excel Analysis Session

**File**: `{excel_file.name}`
**Model**: `{args.model}`
**Sheet**: {args.sheet_index}
**Session**: `{session_id}`
**Started**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Analysis Overview

The deterministic pipeline has completed the initial analysis. Key findings:

- **Security Risk**: {pipeline_result.security.risk_level}
- **Total Sheets**: {pipeline_result.structure.sheet_count}
- **Total Formulas**: {len(pipeline_result.formulas.dependency_graph)}
- **Formula Complexity**: {pipeline_result.formulas.formula_complexity_score:.1f}
- **Has Circular References**: {pipeline_result.formulas.has_circular_references}
"""
            await session.toolkit.render_markdown(context_md)

            # Execute data loading
            load_code = f"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load the Excel file
excel_path = r"{excel_file}"
sheet_index = {args.sheet_index}

# Load with pandas
df = pd.read_excel(excel_path, sheet_name=sheet_index)
print(f"Loaded sheet at index {{sheet_index}}")
print(f"Shape: {{df.shape}}")
print(f"Columns: {{', '.join(df.columns)}}")
"""
            result = await session.toolkit.execute_code(load_code)
            if result.is_err():
                logger.error(f"Failed to load data: {result.err_value}")

            # Load sheet names
            sheet_code = """
# Get all sheet names
xl_file = pd.ExcelFile(excel_path)
sheet_names = xl_file.sheet_names
print(f"Available sheets: {sheet_names}")
print(f"Analyzing sheet: '{sheet_names[sheet_index]}' (index {sheet_index})")
"""
            await session.toolkit.execute_code(sheet_code)

            # Add graph query interface tools if formula analysis succeeded
            if enhanced_interface and pipeline_result and pipeline_result.formulas:
                logger.info("Adding graph query interface tools...")

                # Add markdown documentation for graph queries
                graph_tools_doc = """## 🔍 Formula Analysis Tools

You have TWO approaches available for formula analysis:

### 1️⃣ Graph-Based Dependency Analysis (Recommended for Complex Files)
The deterministic pipeline has analyzed all formulas and created a dependency graph. These tools are robust and handle complex Excel files:

- **get_cell_dependencies** - Analyze what a cell depends on and what depends on it
- **find_cells_affecting_range** - Find all cells that affect a specific range
- **find_empty_cells_in_formula_ranges** - Find gaps in data that formulas reference
- **get_formula_statistics** - Get overall statistics about formulas
- **find_circular_references** - Find all circular reference chains

### 2️⃣ Formulas Library for Advanced Formula Evaluation (Recommended)
Robust formula evaluation using the 'formulas' library that handles complex Excel files:

- **load_excel_with_formulas** - Load Excel file for formula evaluation
- **evaluate_cell** - Get calculated cell values and formulas
- **set_cell_and_recalculate** - What-if analysis with recalculation
- **get_cell_dependencies_formulas** - Track formula dependencies
- **export_formulas_model** - Export model to JSON
- **get_formulas_help** - Get detailed help

✅ **Recommended**: The formulas library handles complex Excel files much better than other alternatives.

### Usage:
All tools are available through the tool-calling interface. Use graph-based analysis for quick dependency queries, and the formulas library for accurate formula evaluation and what-if analysis.
"""

                result = await session.toolkit.render_markdown(graph_tools_doc)
                if result.is_err():
                    logger.warning(f"Failed to add graph tools documentation: {result.err_value}")

                logger.info("Graph query tools are available via tool-calling interface")

            # Initialize tools for LangChain
            from spreadsheet_analyzer.notebook_llm_interface import get_session_manager

            # Set the current session
            session_manager = get_session_manager()
            session_manager._sessions["default_session"] = session

            # Now create the LLM analysis context
            try:
                from spreadsheet_analyzer.notebook_llm_interface import get_notebook_tools

                # Get all notebook tools
                tools = list(get_notebook_tools())

                # Add pipeline summary tool
                tools.append(create_pipeline_summary_tool(pipeline_result))

                # Import LangChain components
                try:
                    from langchain_anthropic import ChatAnthropic
                    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
                    from langchain_openai import ChatOpenAI
                except ImportError as e:
                    logger.error(f"Failed to import LangChain components: {e}")
                    return

                # Initialize the appropriate LLM based on model selection
                llm = None
                if "claude" in args.model.lower():
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    if not api_key:
                        logger.error("ANTHROPIC_API_KEY environment variable not set")
                        llm = None
                    else:
                        llm = ChatAnthropic(
                            model_name=args.model,
                            api_key=api_key,
                            max_tokens=4096,
                        )
                elif "gpt" in args.model.lower():
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        logger.error("OPENAI_API_KEY environment variable not set")
                        llm = None
                    else:
                        llm = ChatOpenAI(
                            model_name=args.model,
                            api_key=api_key,
                            temperature=0.1,
                        )
                else:
                    logger.error(f"Unsupported model: {args.model}")
                    llm = None

                if llm:
                    # Bind tools to the LLM
                    llm_with_tools = llm.bind_tools(tools)

                    # Get current notebook state in py:percent format
                    try:
                        notebook_state = session.toolkit.export_to_percent_format()
                    except Exception as e:
                        logger.error(f"Failed to export notebook state: {e}")
                        import traceback

                        traceback.print_exc()
                        notebook_state = "# Failed to export notebook state"

                    # Create initial prompt with notebook context
                    sheet_info = f" (sheet index {args.sheet_index})" if args.sheet_index != 0 else ""
                    initial_prompt = f"""I've loaded the Excel file '{excel_file.name}'{sheet_info} into a Jupyter notebook.

## Current Notebook State:
```python
{notebook_state}
```

The notebook already contains:
- Pipeline analysis results (Security, Structure, Formula Analysis)
- Data loaded into DataFrame 'df' with initial exploration showing shape and first rows
{"- Query interface for formula dependencies (graph-based analysis)" if formula_cache_path and formula_cache_path.exists() else ""}
- Formula analysis tools: graph-based for dependencies, formulas library for evaluation

Please continue the analysis from where it left off. **DO NOT re-execute cells that already have output.**

You can:
1. Execute NEW Python code to explore the data further
2. Use pandas operations to analyze patterns
3. Create visualizations if helpful
4. {"Query the formula dependency graph using graph-based tools" if formula_cache_path and formula_cache_path.exists() else "Look for data quality issues"}
5. Use formulas library tools for formula evaluation and what-if analysis
6. Provide insights and recommendations

Focus on deeper analysis that builds upon what's already been done.

IMPORTANT: Track your progress against the completion criteria and create a final summary when done."""

                    # Import session tracking
                    from spreadsheet_analyzer.observability import add_session_metadata, phoenix_session

                    messages = [
                        SystemMessage(
                            f"""You are an autonomous data analyst AI conducting comprehensive spreadsheet analysis.

CRITICAL CONSTRAINT - NO VISUAL ACCESS:
- Cannot see images, plots, charts, or visualizations
- Must extract insights through textual methods only
- All analysis based on numerical summaries and textual descriptions

CONTEXT:
- Analyzing Excel file: {excel_file.name}
- Sheet index: {args.sheet_index}
- Sheet name: {sheet_name}

CURRENT NOTEBOOK STATE:
```python
{notebook_state}
```

AUTONOMOUS ANALYSIS PROTOCOL:
1. Conduct systematic analysis without seeking user approval
2. Back all assumptions with evidence from the data
3. Reach solid conclusions based on thorough investigation
4. Document reasoning in code comments
5. Use available tools to execute code and explore data
6. Focus on actionable insights and recommendations

MULTI-TABLE DETECTION - CRITICAL FIRST STEP:
Before analyzing data, ALWAYS check if the sheet contains multiple tables using BOTH mechanical AND semantic analysis:

1. **Initial Structure Scan**
   ```python
   # First, examine the raw structure with MORE context
   print(f"Sheet dimensions: {{df.shape}}")
   print("\\n--- First 30 rows ---")
   print(df.head(30))

   # Look at ALL columns including unnamed ones
   print("\\n--- Column overview ---")
   for col in df.columns:
       non_null = df[col].notna().sum()
       print(f"{{col}}: {{non_null}} non-null values, dtype: {{df[col].dtype}}")
   ```

2. **Mechanical Detection** (empty rows/columns)
   ```python
   # Check for empty row patterns that separate tables
   empty_rows = df.isnull().all(axis=1)
   empty_row_groups = empty_rows.groupby((~empty_rows).cumsum()).sum()
   print(f"\\nEmpty row blocks: {{empty_row_groups[empty_row_groups > 0].to_dict()}}")

   # Check for empty columns (potential horizontal separators)
   empty_cols = df.isnull().all(axis=0)
   print(f"Empty columns: {{list(df.columns[empty_cols])}}")
   ```

3. **Semantic Detection** (USE HUMAN JUDGMENT)
   ```python
   # Ask yourself: What does each row represent?
   # Example: If row 1 = "Product ABC, $50" and row 100 = "John Smith, Developer"
   # These are clearly different entity types!

   # Check for semantic shifts in data
   print("\\n--- Checking for semantic table boundaries ---")

   # 1. Analyze column groupings - do they describe the same type of thing?
   # Example: ['Customer', 'Address', 'Phone'] vs ['Date', 'Amount', 'Transaction ID']

   # 2. Look for granularity changes
   # Example: 5 summary rows followed by 1000 detail rows

   # 3. Check if column names suggest different purposes
   # Example: Financial columns vs HR columns in same sheet
   ```

4. **Common Multi-Table Patterns to Recognize**
   - **Master/Detail**: Header info (few rows) + Line items (many rows)
   - **Summary/Breakdown**: Totals followed by individual components
   - **Different Domains**: Unrelated business data side-by-side
   - **Time Periods**: Current data adjacent to historical data

   Ask: "Would these naturally be separate tables in a database?"

5. **Decision Framework**
   Even WITHOUT empty rows/columns, declare multiple tables if:
   - Rows represent fundamentally different entity types
   - Column sets serve different business purposes
   - There's a clear shift in data granularity
   - A business analyst would logically separate them

6. **Multi-Table Handling Strategy**
   If multiple tables detected:
   - Document the table boundaries and what each represents
   - Analyze each table's purpose separately
   - Use `.iloc[start:end, start_col:end_col]` for extraction
   - Focus on the most relevant table(s) for insights

COMPLETION PROTOCOL - CRITICAL:
- FIRST: Always perform multi-table detection using the empty row analysis code
- Complete ALL analysis steps autonomously without asking for user input
- NEVER ask "Would you like me to..." or "Let me know if..." or "Do you need..."
- When analysis is complete, provide a final summary and STOP
- If errors occur, implement workarounds or fix code, re-run it and continue analysis
- End with definitive conclusions
- DO NOT offer to perform additional analysis - just complete what's needed

TEXTUAL DATA EXPLORATION TECHNIQUES:
- `.iloc[start:end]` or `.loc[condition]` to examine specific data regions
- `.sample(n)` for random sampling
- `.groupby()` for categorical analysis
- `.value_counts()` for frequency distributions
- `.describe()` for statistical summaries
- `.corr()` for correlation analysis
- `.isnull().sum()` for missing data analysis

TEXTUAL VISUALIZATION ALTERNATIVES (NO IMAGES):
**Distributions & Patterns:**
- `.value_counts().head(10)` to show frequency distribution
- `.describe()` to show quartiles, mean, std, min/max
- `.quantile([0.1, 0.25, 0.5, 0.75, 0.9])` for detailed percentiles
- `.hist(bins=20).value_counts()` for histogram-like data

**Trends & Relationships:**
- `.groupby().agg(['mean', 'std', 'count'])` to show patterns by category
- `.corr().round(3)` to show correlation matrix numerically
- `.pivot_table()` to show cross-tabulations
- `.rolling(window=5).mean()` for moving averages

**Outliers & Anomalies:**
- IQR method: Q1, Q3 = df.quantile([0.25, 0.75]); IQR = Q3 - Q1
- `.quantile([0.01, 0.99])` to show extreme values
- `(df > df.quantile(0.99)) | (df < df.quantile(0.01))` to identify outliers
- `.std()` and z-score calculations for statistical outliers

**Missing Data Patterns:**
- `.isnull().sum()` for column-wise missing counts
- `.isnull().sum(axis=1)` for row-wise missing patterns
- `.isnull().groupby(df['category']).sum()` for missing by category

**Data Quality Assessment:**
- `.dtypes` to check data types
- `.nunique()` to check cardinality
- `.duplicated().sum()` to find duplicates
- `.apply(lambda x: x.astype(str).str.len().max())` for string length analysis

ERROR VALIDATION REQUIREMENTS:
- **Data Type Validation**: Check for mixed data types in columns
- **Range Validation**: Identify values outside expected ranges
- **Formula Verification**: If formulas exist, verify calculations manually
- **Consistency Checks**: Look for inconsistent naming, formatting, or values
- **Missing Data Patterns**: Analyze if missing data follows patterns
- **Duplicate Detection**: Check for duplicate rows or suspicious duplicates
- **Business Logic Validation**: Verify data makes business sense
- **Cross-Reference Validation**: Check relationships between columns

EVIDENCE-BASED ANALYSIS:
- Never assume - always verify with data
- Show calculations - don't just state conclusions
- Provide confidence levels - indicate uncertainty
- Cross-validate findings - use multiple methods
- Document assumptions - clearly state what you're assuming

OUTPUT REQUIREMENTS:
- All outputs truncated at 1000 characters
- Use ONLY textual summaries and numerical descriptions
- Provide specific data examples to support conclusions
- Include error detection findings in analysis
- Reach definitive, evidence-based conclusions
- Describe patterns, trends, and relationships in words

BEST PRACTICES:
- Include reasoning in code comments
- Document analysis approach and findings
- Build upon existing analysis
- Provide clear, actionable recommendations
- Always validate findings with multiple approaches
- Use descriptive statistics to paint a picture of the data

ANALYSIS COMPLETION CRITERIA:
Mark analysis as COMPLETE when ALL of the following are achieved:
1. ✓ Multi-table detection performed (empty row analysis to identify table boundaries)
2. ✓ Data quality assessment completed (missing data, duplicates, anomalies)
3. ✓ Statistical analysis performed (distributions, correlations, patterns)
4. ✓ Business logic validated (calculations, relationships, consistency)
5. ✓ Key findings documented in markdown cells
6. ✓ Actionable recommendations provided
7. ✓ Final comprehensive analysis report created in markdown cell with title "## 📊 Analysis Complete"

When these criteria are met, create a final comprehensive analysis report in a markdown cell:

**Report Structure Required:**
# 📊 Analysis Complete

## Executive Summary
- Brief overview of the analysis performed
- Most important findings in 2-3 sentences

## Data Overview
- Dataset characteristics (size, timeframe, scope)
- Multi-table detection results
- Data quality summary

## Key Findings
1. **Finding 1**: [Detailed description with supporting evidence]
2. **Finding 2**: [Detailed description with supporting evidence]
3. **Finding 3**: [Detailed description with supporting evidence]
(Include 3-5 major findings)

## Data Quality Issues
- Missing data patterns
- Anomalies detected
- Validation concerns

## Statistical Insights
- Key distributions and patterns
- Significant correlations
- Trend analysis results

## Business Implications
- What these findings mean for business operations
- Risk factors identified
- Opportunities discovered

## Recommendations
1. **Immediate Actions**: [What should be done right away]
2. **Short-term Improvements**: [1-3 month timeline]
3. **Long-term Considerations**: [Strategic recommendations]

## Technical Notes
- Analysis methodology used
- Assumptions made
- Limitations of the analysis

Then STOP the analysis - do not ask for further instructions."""
                        ),
                        HumanMessage(content=initial_prompt),
                    ]

                    # Wrap analysis in session tracking
                    with phoenix_session(
                        session_id=session_id,
                        user_id=None,  # Could be set from args or env
                    ):
                        # Add session metadata
                        add_session_metadata(
                            session_id,
                            {
                                "excel_file": excel_file.name,
                                "sheet_index": args.sheet_index,
                                "sheet_name": sheet_name or f"Sheet {args.sheet_index}",
                                "model": args.model,
                                "max_rounds": args.max_rounds,
                                "cost_limit": args.cost_limit if args.track_costs else None,
                            },
                        )

                        # Analysis loop with round tracking
                        for round_num in range(1, args.max_rounds + 1):
                            try:
                                # Log round start
                                llm_logger.info(f"\n{'=' * 60}")
                                llm_logger.info(f"ROUND {round_num}/{args.max_rounds}")
                                llm_logger.info(f"{'=' * 60}\n")

                                # Log the message being sent
                                llm_logger.info(f"📤 Sending to LLM ({args.model}):")
                                llm_logger.info(f"{messages[-1].content}\n")

                                # Call LLM
                                response = await llm_with_tools.ainvoke(messages)

                                # Track token usage
                                await track_llm_usage(response, args.model)

                                # Log response
                                llm_logger.info("📥 LLM Response:")
                                if hasattr(response, "content") and response.content:
                                    llm_logger.info(f"{response.content}")

                                messages.append(response)

                                # Process tool calls if any
                                if hasattr(response, "tool_calls") and response.tool_calls:
                                    llm_logger.info(f"\n🔧 Tool Calls: {len(response.tool_calls)}")
                                for tool_call in response.tool_calls:
                                    llm_logger.info(f"\n  Tool: {tool_call['name']}")
                                    llm_logger.info(f"  Args: {tool_call['args']}")

                                    # Find and execute the tool
                                    tool_fn = None
                                    for tool in tools:
                                        if tool.name == tool_call["name"]:
                                            tool_fn = tool
                                            break

                                    if tool_fn:
                                        try:
                                            # Execute tool
                                            tool_result = await tool_fn.ainvoke(tool_call["args"])
                                            llm_logger.info(f"  Result: {tool_result[:500]}...")  # Log first 500 chars

                                            # Add tool result to messages
                                            tool_message = ToolMessage(
                                                content=str(tool_result),
                                                tool_call_id=tool_call["id"],
                                            )
                                            messages.append(tool_message)
                                        except Exception as e:
                                            error_msg = f"Tool execution failed: {e!s}"
                                            logger.error(error_msg)
                                            llm_logger.info(f"  Error: {error_msg}")
                                            tool_message = ToolMessage(
                                                content=error_msg,
                                                tool_call_id=tool_call["id"],
                                            )
                                            messages.append(tool_message)
                                    else:
                                        logger.warning(f"Tool not found: {tool_call['name']}")
                                else:
                                    # No tool calls - check if analysis is complete or if asking for input
                                    if hasattr(response, "content") and response.content:
                                        # Check for patterns that indicate the LLM is asking for user input
                                        forbidden_patterns = [
                                            "what would you like",
                                            "what do you want",
                                            "would you prefer",
                                            "shall i",
                                            "should i proceed",
                                            "let me know",
                                            "please specify",
                                            "please tell me",
                                            "any specific",
                                            "how would you like",
                                            "do you want me to",
                                            "would you like me to",
                                        ]

                                        # Check if response.content is a string (not a list of tool calls)
                                        if isinstance(response.content, str):
                                            response_lower = response.content.lower()
                                            if any(pattern in response_lower for pattern in forbidden_patterns):
                                                logger.warning(
                                                    "LLM attempted to ask for user input - enforcing autonomous completion"
                                                )
                                                # Add a system message to remind the LLM to complete autonomously
                                                messages.append(response)
                                                messages.append(
                                                    SystemMessage(
                                                        content="""
REMINDER: You must complete the analysis autonomously.
- Create a final comprehensive analysis report in markdown with "## 📊 Analysis Complete"
- Follow the required report structure (Executive Summary, Data Overview, Key Findings, etc.)
- Include all sections: findings, data quality, statistical insights, business implications, recommendations
- Then STOP - do not ask for further instructions
Complete the analysis now."""
                                                    )
                                                )
                                                continue  # Continue to next round instead of breaking

                                        # Check if analysis is complete
                                        if isinstance(response.content, str) and (
                                            "analysis complete" in response.content.lower()
                                            or "📊 analysis complete" in response.content.lower()
                                        ):
                                            logger.info("Analysis marked as complete by LLM")
                                            # Log round completion
                                        llm_logger.info(f"\n{'✅' * 40}")
                                        llm_logger.info(f"{'✅' * 15} ROUND {round_num} Complete {'✅' * 15}")
                                        llm_logger.info(f"{'✅' * 40}\n")
                                        break

                                        # If no tool calls and not asking for input, the LLM is done
                                        # Log round completion
                                        llm_logger.info(f"\n{'✅' * 40}")
                                        llm_logger.info(f"{'✅' * 15} ROUND {round_num} Complete {'✅' * 15}")
                                        llm_logger.info(f"{'✅' * 40}\n")
                                        break
                                    else:
                                        logger.warning("LLM response was empty.")
                                        # Log round completion
                                        llm_logger.info(f"\n{'✅' * 40}")
                                        llm_logger.info(f"{'✅' * 15} ROUND {round_num} Complete {'✅' * 15}")
                                        llm_logger.info(f"{'✅' * 40}\n")
                                        break

                            except Exception as e:
                                logger.error(f"Error in round {round_num}: {e}", exc_info=True)
                                llm_logger.info(f"\n❌ Error in round {round_num}: {e}")

                                # Try a simple fallback for specific errors
                                if "Connection error" in str(e) or "rate_limit" in str(e):
                                    logger.info("Attempting to recover from connection/rate limit error...")
                                    await asyncio.sleep(5)  # Wait before retry
                                    continue

                                # For tool not found errors, add a helpful message and continue
                                if "Unknown tool" in str(e):
                                    messages.append(
                                        SystemMessage(
                                            content=f"The tool you tried to use is not available. Available tools are: {', '.join(t.name for t in tools)}. Please continue with the analysis using the available tools."
                                        )
                                    )
                                continue

                            # Try fallback if we haven't already
                            if not isinstance(llm, ChatOpenAI):
                                try:
                                    logger.info("Switching to fallback model: gpt-4")
                                    llm = ChatOpenAI(model_name="gpt-4")
                                    llm_with_tools = llm.bind_tools(tools)
                                    response = await llm_with_tools.ainvoke(messages)
                                    messages.append(response)
                                    continue
                                except Exception as fallback_error:
                                    logger.error(f"Fallback also failed: {fallback_error}")

                            # If we can't recover, break the loop
                            break

                    # Log final summary
                    llm_logger.info(f"\n{'=' * 60}")
                    llm_logger.info("ANALYSIS COMPLETE")
                    llm_logger.info(f"Total rounds: {round_num}/{args.max_rounds}")
                    llm_logger.info(f"{'=' * 60}\n")

            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")

            # Log cost summary if tracking enabled
            if args.track_costs:
                cost_summary = cost_tracker.get_summary()
                logger.info("\n💰 Cost Summary:")
                logger.info(f"  Total Cost: ${cost_summary['total_cost_usd']:.4f}")
                logger.info(f"  Total Tokens: {cost_summary['total_tokens']['total']:,}")
                if cost_summary["cost_by_model"]:
                    logger.info("  Cost by Model:")
                    for model, cost in cost_summary["cost_by_model"].items():
                        logger.info(f"    {model}: ${cost:.4f}")
                if args.cost_limit:
                    logger.info(
                        f"  Budget Status: {'✅ Within' if cost_summary['within_budget'] else '❌ Exceeded'} limit (${args.cost_limit:.2f})"
                    )

            # Ensure notebook is saved at the end
            logger.info("Analysis complete. Saving notebook...")
            save_result = session.toolkit.save_notebook(notebook_path, overwrite=True)
            if save_result.is_ok():
                logger.info(f"✅ Notebook saved successfully to: {save_result.ok_value}")
            else:
                logger.error(f"❌ Failed to save notebook: {save_result.err_value}")

            logger.info("Analysis session completed.")

    def run(self):
        """Parse arguments and run the analysis."""
        args = self.parser.parse_args()
        # Setup basic logging
        log_level = logging.INFO if args.verbose else logging.WARNING
        logging.basicConfig(level=log_level, stream=sys.stdout)

        # Give a more specific logger name
        global logger
        logger = get_logger(f"notebook_cli.{args.model}")

        asyncio.run(self.run_analysis(args))


if __name__ == "__main__":
    NotebookCLI().run()
