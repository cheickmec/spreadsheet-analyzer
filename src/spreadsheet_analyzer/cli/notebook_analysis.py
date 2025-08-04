"""Functional notebook analysis orchestration.

This module provides the core analysis logic separated from CLI concerns,
following functional programming principles.

CLAUDE-KNOWLEDGE: Analysis is broken into discrete steps that can be
composed and tested independently.
"""

import asyncio
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from structlog import get_logger

from ..core.types import Result, err, ok
from ..graph_db.query_interface import create_enhanced_query_interface
from ..notebook_session import NotebookSession, notebook_session
from ..observability import (
    PhoenixConfig,
    initialize_cost_tracker,
    initialize_phoenix,
    instrument_all,
)
from ..pipeline import DeterministicPipeline
from ..pipeline.types import PipelineResult
from .utils.markdown import (
    create_header,
    formulas_to_markdown,
    security_to_markdown,
    structure_to_markdown,
)
from .utils.naming import FileNameConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class AnalysisConfig:
    """Immutable configuration for analysis run."""

    excel_path: Path
    sheet_index: int = 0
    sheet_name: str | None = None
    model: str = "claude-3-5-sonnet-20241022"
    api_key: str | None = None
    max_rounds: int = 5
    output_dir: Path | None = None
    session_id: str | None = None
    notebook_path: Path | None = None
    verbose: bool = False
    # Phoenix observability
    phoenix_config: PhoenixConfig | None = None
    # Cost tracking
    track_costs: bool = True
    cost_limit: float | None = None
    # Auto-save configuration
    auto_save_rounds: bool = True  # Save after each LLM round
    auto_save_frequency: int = 0  # Save after N tool calls (0 = disabled)
    # Context compression configuration
    enable_compression: bool = True  # Enable progressive context compression
    compression_verbose: bool = False  # Log detailed compression info


@dataclass(frozen=True)
class AnalysisArtifacts:
    """Immutable container for analysis artifacts."""

    session_id: str
    notebook_path: Path
    log_path: Path
    cost_tracking_path: Path
    file_config: FileNameConfig


@dataclass(frozen=True)
class PipelineResults:
    """Results from pipeline analysis phase."""

    pipeline_result: PipelineResult | None = None
    query_interface: Any | None = None
    formula_cache_path: Path | None = None


@dataclass
class AnalysisState:
    """Mutable state container for analysis progress."""

    config: AnalysisConfig
    artifacts: AnalysisArtifacts
    pipeline_results: PipelineResults = field(default_factory=lambda: PipelineResults())
    cost_tracker: Any | None = None
    tracer_provider: Any | None = None
    llm_logger: logging.Logger | None = None


# Pure functions for each analysis step


def initialize_observability(config: AnalysisConfig) -> Result[Any | None, str]:
    """Initialize Phoenix observability if configured.

    Args:
        config: Analysis configuration

    Returns:
        Result containing tracer provider or None
    """
    if not config.phoenix_config or config.phoenix_config.mode == "none":
        return ok(None)

    try:
        tracer_provider = initialize_phoenix(config.phoenix_config)
        if tracer_provider:
            results = instrument_all(tracer_provider)
            logger.info("Phoenix instrumentation complete", results=results)
            return ok(tracer_provider)
        else:
            logger.warning("Phoenix initialization failed, continuing without observability")
            return ok(None)
    except Exception as e:
        return err(f"Failed to initialize Phoenix: {e}")


def setup_llm_logging(log_path: Path) -> Result[logging.Logger, str]:
    """Set up LLM message logging to file.

    Args:
        log_path: Path for log file

    Returns:
        Result containing configured logger
    """
    try:
        llm_logger = logging.getLogger("llm_messages")
        llm_logger.setLevel(logging.DEBUG)

        # Remove existing handlers to avoid duplicates
        llm_logger.handlers.clear()

        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        llm_logger.addHandler(file_handler)

        llm_logger.info("Starting LLM message logging")
        return ok(llm_logger)
    except Exception as e:
        return err(f"Failed to setup LLM logging: {e}")


def setup_cost_tracking(config: AnalysisConfig, cost_tracking_path: Path) -> Result[Any | None, str]:
    """Initialize cost tracking if enabled.

    Args:
        config: Analysis configuration
        cost_tracking_path: Path for cost tracking data

    Returns:
        Result containing cost tracker or None
    """
    if not config.track_costs:
        return ok(None)

    try:
        cost_tracker = initialize_cost_tracker(cost_limit=config.cost_limit, save_path=cost_tracking_path)
        logger.info(f"💰 Cost tracking enabled: {cost_tracking_path}")
        if config.cost_limit:
            logger.info(f"💰 Cost limit set: ${config.cost_limit:.2f}")
        return ok(cost_tracker)
    except Exception as e:
        return err(f"Failed to setup cost tracking: {e}")


async def run_pipeline_analysis(excel_path: Path) -> PipelineResults:
    """Run deterministic pipeline analysis.

    Args:
        excel_path: Path to Excel file

    Returns:
        Pipeline analysis results
    """
    logger.info("Running deterministic pipeline analysis...")

    try:
        pipeline = DeterministicPipeline()
        pipeline_result = await asyncio.to_thread(pipeline.run, excel_path)

        if pipeline_result.success:
            logger.info(f"✅ Pipeline analysis completed in {pipeline_result.execution_time:.2f} seconds")

            # Save formula analysis to cache if available
            if pipeline_result.formulas:
                formula_cache_path = await cache_formula_analysis(excel_path, pipeline_result.formulas)

                # Create query interface
                query_interface = create_enhanced_query_interface(pipeline_result.formulas)
                logger.info("Created enhanced query interface for formula analysis")

                return PipelineResults(
                    pipeline_result=pipeline_result,
                    query_interface=query_interface,
                    formula_cache_path=formula_cache_path,
                )

            return PipelineResults(pipeline_result=pipeline_result)
        else:
            logger.error("Pipeline analysis failed:")
            for error in pipeline_result.errors:
                logger.error(f"  - {error}")
            return PipelineResults(pipeline_result=pipeline_result)

    except Exception:
        logger.exception("Error running deterministic pipeline")
        # Return empty results - we can still create a notebook
        return PipelineResults()


async def cache_formula_analysis(excel_path: Path, formulas: Any) -> Path | None:
    """Cache formula analysis results.

    Args:
        excel_path: Path to Excel file
        formulas: Formula analysis data

    Returns:
        Path to cache file if successful
    """
    try:
        cache_dir = Path(".pipeline_cache")
        cache_dir.mkdir(exist_ok=True)

        cache_filename = f"{excel_path.stem}_formula_analysis.pkl"
        cache_path = cache_dir / cache_filename

        with cache_path.open("wb") as f:
            pickle.dump(formulas, f)

        logger.info(f"Saved formula analysis to cache: {cache_path}")
        return cache_path
    except Exception as e:
        logger.error(f"Failed to cache formula analysis: {e}")
        return None


async def add_pipeline_results_to_notebook(
    session: NotebookSession, pipeline_result: PipelineResult
) -> Result[None, str]:
    """Add pipeline analysis results to notebook as markdown cells.

    Args:
        session: Active notebook session
        pipeline_result: Pipeline analysis results

    Returns:
        Result indicating success or failure
    """
    logger.info("Adding pipeline results to notebook...")
    toolkit = session.toolkit

    try:
        # Add header
        header_md = create_header(pipeline_result)
        result = await toolkit.render_markdown(header_md)
        if result.is_err():
            logger.warning(f"Failed to add header: {result.err_value}")

        # Add security analysis if relevant
        if pipeline_result.security:
            security_md = security_to_markdown(pipeline_result.security)
            if security_md:  # Only add if not empty
                result = await toolkit.render_markdown(security_md)
                if result.is_err():
                    logger.warning(f"Failed to add security analysis: {result.err_value}")

        # Add structure analysis
        if pipeline_result.structure:
            structure_md = structure_to_markdown(pipeline_result.structure, pipeline_result.security)
            result = await toolkit.render_markdown(structure_md)
            if result.is_err():
                logger.warning(f"Failed to add structure analysis: {result.err_value}")

        # Add formula analysis
        if pipeline_result.formulas:
            formulas_md = formulas_to_markdown(pipeline_result.formulas)
            if formulas_md:  # Only add if not empty
                result = await toolkit.render_markdown(formulas_md)
                if result.is_err():
                    logger.warning(f"Failed to add formula analysis: {result.err_value}")

        logger.info("Pipeline results added to notebook")
        return ok(None)

    except Exception as e:
        return err(f"Failed to add pipeline results: {e}")


def generate_data_loading_code(config: AnalysisConfig, excel_path: Path) -> str:
    """Generate Python code for loading Excel data.

    Args:
        config: Analysis configuration
        excel_path: Path to Excel file

    Returns:
        Python code string
    """
    # Calculate relative path from repo root
    try:
        repo_root = Path.cwd()
        relative_path = excel_path.relative_to(repo_root)
        path_str = f'"{relative_path}"'
    except ValueError:
        # If file is outside repo, use absolute path
        path_str = f'r"{excel_path}"'

    # CLAUDE-GOTCHA: F-strings within f-strings require careful escaping
    return f"""import pandas as pd
from pathlib import Path

# Load the Excel file (path relative to repo root)
excel_path = Path({path_str})
sheet_index = {config.sheet_index}

print(f"Loading data from: {{excel_path}}")
print(f"Loading sheet at index: {{sheet_index}}")

try:
    # First, get information about all sheets
    xl_file = pd.ExcelFile(excel_path)
    sheet_names = xl_file.sheet_names
    print(f"\\nAvailable sheets: {{sheet_names}}")

    if sheet_index >= len(sheet_names):
        print(f"\\nError: Sheet index {{sheet_index}} is out of range. File has {{len(sheet_names)}} sheets.")
        print("Please use --sheet-index with a value between 0 and {{}}")
        df = None
    else:
        # Load the specified sheet
        selected_sheet = sheet_names[sheet_index]
        print(f"\\nLoading sheet: '{{selected_sheet}}'")

        df = pd.read_excel(excel_path, sheet_name=sheet_index)
        print(f"\\nLoaded data with shape: {{df.shape}}")
        print(f"Columns: {{list(df.columns)}}")

        # Display first few rows
        print("\\nFirst 5 rows:")
        display(df.head())

        # Basic info
        print("\\nData types:")
        print(df.dtypes)

except Exception as e:
    print(f"Error loading Excel file: {{e}}")
    df = None
"""


def generate_query_interface_code(cache_path: Path) -> str:
    """Generate code for loading query interface.

    Args:
        cache_path: Path to formula cache

    Returns:
        Python code string
    """
    # Calculate relative path for the cache file
    try:
        repo_root = Path.cwd()
        relative_cache_path = cache_path.relative_to(repo_root)
        cache_path_str = f'"{relative_cache_path}"'
    except ValueError:
        # If cache is outside repo, use absolute path
        cache_path_str = f'r"{cache_path}"'

    # CLAUDE-GOTCHA: F-strings within f-strings require careful escaping
    return f'''# Query interface for formula dependency analysis
# The pipeline has already analyzed all formulas and cached the results

import pickle
from pathlib import Path
from spreadsheet_analyzer.graph_db.query_interface import create_enhanced_query_interface

# Load cached formula analysis
cache_file = Path({cache_path_str})
with open(cache_file, 'rb') as f:
    formula_analysis = pickle.load(f)

query_interface = create_enhanced_query_interface(formula_analysis)

# Convenience functions for graph queries
def get_cell_dependencies(sheet, cell_ref):
    """Get complete dependency information for a specific cell."""
    result = query_interface.get_cell_dependencies(sheet, cell_ref)
    print(f"\\nCell {{sheet}}!{{cell_ref}}:")
    print(f"  Has formula: {{result.has_formula}}")
    if result.formula:
        print(f"  Formula: {{result.formula}}")
    if result.direct_dependencies:
        print(f"  Direct dependencies: {{', '.join(result.direct_dependencies[:5])}}")
        if len(result.direct_dependencies) > 5:
            print(f"    ...and {{len(result.direct_dependencies) - 5}} more")
    if result.direct_dependents:
        print(f"  Cells that depend on this: {{', '.join(result.direct_dependents[:5])}}")
        if len(result.direct_dependents) > 5:
            print(f"    ...and {{len(result.direct_dependents) - 5}} more")
    return result

def find_cells_affecting_range(sheet, start_cell, end_cell):
    """Find all cells that affect any cell within the specified range."""
    result = query_interface.find_cells_affecting_range(sheet, start_cell, end_cell)
    print(f"\\nCells affecting range {{sheet}}!{{start_cell}}:{{end_cell}}:")
    for cell, deps in list(result.items())[:5]:
        print(f"  {{cell}} depends on: {{', '.join(deps[:3])}}")
        if len(deps) > 3:
            print(f"    ...and {{len(deps) - 3}} more")
    if len(result) > 5:
        print(f"  ...and {{len(result) - 5}} more cells")
    return result

def get_formula_statistics():
    """Get comprehensive statistics about formulas in the workbook."""
    stats = query_interface.get_formula_statistics_with_ranges()
    print("\\nFormula Statistics:")
    print(f"  Total formulas: {{stats['total_formulas']:,}}")
    print(f"  Formulas with dependencies: {{stats['formulas_with_dependencies']:,}}")
    print(f"  Unique cells referenced: {{stats['unique_cells_referenced']:,}}")
    print(f"  Max dependency depth: {{stats['max_dependency_depth']}} levels")
    print(f"  Circular references: {{stats['circular_reference_chains']}}")
    print(f"  Formula complexity score: {{stats['complexity_score']}}/100")
    return stats

def find_empty_cells_in_formula_ranges(sheet):
    """Find empty cells that are part of formula ranges."""
    result = query_interface.find_empty_cells_in_formula_ranges(sheet)
    print(f"\\nEmpty cells in formula ranges for sheet '{{sheet}}':")
    if result:
        print(f"  Found {{len(result)}} empty cells")
        # Group by rows for display
        rows = {{}}
        for cell in list(result)[:20]:
            row_num = ''.join(filter(str.isdigit, cell))
            if row_num not in rows:
                rows[row_num] = []
            rows[row_num].append(cell)
        for row, cells in list(rows.items())[:5]:
            print(f"  Row {{row}}: {{', '.join(cells)}}")
        if len(result) > 20:
            print(f"  ...and {{len(result) - 20}} more")
    else:
        print("  No empty cells found in formula ranges")
    return result
'''


async def setup_notebook_basics(session: NotebookSession, state: AnalysisState) -> Result[None, str]:
    """Set up basic notebook cells for data loading and query interface.

    Args:
        session: Active notebook session
        state: Analysis state

    Returns:
        Result indicating success or failure
    """
    toolkit = session.toolkit

    try:
        # Add data loading cell
        logger.info("Adding data loading cell...")
        load_code = generate_data_loading_code(state.config, state.config.excel_path)

        result = await toolkit.execute_code(load_code)
        if result.is_ok():
            logger.info(f"✅ Data loading cell executed: {result.ok_value.cell_id}")
        else:
            logger.error(f"❌ Failed to execute data loading cell: {result.err_value}")

        # Add query interface if formula cache exists
        if state.pipeline_results.formula_cache_path and state.pipeline_results.formula_cache_path.exists():
            logger.info("Adding query interface loading cell...")
            query_code = generate_query_interface_code(state.pipeline_results.formula_cache_path)

            result = await toolkit.execute_code(query_code)
            if result.is_ok():
                logger.info(f"✅ Query interface loading cell executed: {result.ok_value.cell_id}")
            else:
                logger.error(f"❌ Failed to execute query interface loading cell: {result.err_value}")

        # Add documentation about available tools
        if (
            state.pipeline_results.query_interface
            and state.pipeline_results.pipeline_result
            and state.pipeline_results.pipeline_result.formulas
        ):
            logger.info("Adding graph query interface tools documentation...")

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

            result = await toolkit.render_markdown(graph_tools_doc)
            if result.is_err():
                logger.warning(f"Failed to add graph tools documentation: {result.err_value}")

            logger.info("Graph query tools are available via tool-calling interface")

            # Add marker to delineate LLM analysis region
            marker_result = await toolkit.render_markdown("## --- LLM Analysis Start ---")
            if marker_result.is_err():
                logger.warning(f"Failed to add LLM analysis marker: {marker_result.err_value}")
            else:
                logger.info("Added LLM analysis boundary marker")

        return ok(None)

    except Exception as e:
        return err(f"Failed to setup notebook basics: {e}")


async def save_notebook(session: NotebookSession, notebook_path: Path) -> Result[Path, str]:
    """Save the notebook to disk.

    Args:
        session: Active notebook session
        notebook_path: Path to save notebook

    Returns:
        Result containing saved path or error
    """
    logger.info("Saving notebook...")
    save_result = session.toolkit.save_notebook(notebook_path, overwrite=True)

    if save_result.is_ok():
        logger.info(f"✅ Notebook saved successfully to: {save_result.ok_value}")
        return ok(save_result.ok_value)
    else:
        logger.error(f"❌ Failed to save notebook: {save_result.err_value}")
        return err(f"Failed to save notebook: {save_result.err_value}")


def log_cost_summary(cost_tracker: Any, cost_limit: float | None) -> None:
    """Log cost tracking summary.

    Args:
        cost_tracker: Cost tracking instance
        cost_limit: Optional spending limit
    """
    try:
        cost_summary = cost_tracker.get_summary()
        logger.info("\n💰 Cost Summary:")
        logger.info(f"  Total Cost: ${cost_summary['total_cost_usd']:.4f}")
        logger.info(f"  Total Tokens: {cost_summary['total_tokens']['total']:,}")

        if cost_summary["cost_by_model"]:
            logger.info("  Cost by Model:")
            for model, cost in cost_summary["cost_by_model"].items():
                logger.info(f"    {model}: ${cost:.4f}")

        if cost_limit:
            status = "✅ Within" if cost_summary["within_budget"] else "❌ Exceeded"
            logger.info(f"  Budget Status: {status} limit (${cost_limit:.2f})")
    except Exception as e:
        logger.error(f"Failed to log cost summary: {e}")


# Main orchestration function


async def run_notebook_analysis(config: AnalysisConfig, artifacts: AnalysisArtifacts) -> Result[None, str]:
    """Run the complete notebook analysis workflow.

    This is the main orchestration function that coordinates all analysis steps.

    Args:
        config: Analysis configuration
        artifacts: Analysis artifacts (paths, IDs)

    Returns:
        Result indicating success or failure
    """
    # Initialize state
    state = AnalysisState(config=config, artifacts=artifacts)

    # Step 1: Initialize observability
    obs_result = initialize_observability(config)
    if obs_result.is_err():
        logger.warning(f"Observability setup failed: {obs_result.unwrap_err()}")
    else:
        state.tracer_provider = obs_result.unwrap()

    # Step 2: Setup logging
    log_result = setup_llm_logging(artifacts.log_path)
    if log_result.is_err():
        return err(f"Failed to setup logging: {log_result.unwrap_err()}")
    state.llm_logger = log_result.unwrap()

    # Step 3: Setup cost tracking
    cost_result = setup_cost_tracking(config, artifacts.cost_tracking_path)
    if cost_result.is_err():
        logger.warning(f"Cost tracking setup failed: {cost_result.unwrap_err()}")
    else:
        state.cost_tracker = cost_result.unwrap()

    # Step 4: Check if file exists
    if not config.excel_path.exists():
        return err(f"Excel file not found: {config.excel_path}")

    logger.info(f"Starting analysis of: {config.excel_path}")
    logger.info(f"Session ID: {artifacts.session_id}")
    logger.info(f"Notebook Path: {artifacts.notebook_path}")
    logger.info(f"LLM message log: {artifacts.log_path}")

    # Step 5: Run pipeline analysis
    state.pipeline_results = await run_pipeline_analysis(config.excel_path)

    # Step 6: Create notebook and add results
    async with notebook_session(artifacts.session_id, artifacts.notebook_path) as session:
        # Add query interface to session if available
        if state.pipeline_results.query_interface:
            session.query_interface = state.pipeline_results.query_interface

        # Register session for tools access
        from ..notebook_llm_interface import get_session_manager

        session_manager = get_session_manager()
        session_manager._sessions["default_session"] = session

        # Add pipeline results if available
        if state.pipeline_results.pipeline_result:
            add_result = await add_pipeline_results_to_notebook(session, state.pipeline_results.pipeline_result)
            if add_result.is_err():
                logger.warning(f"Failed to add pipeline results: {add_result.unwrap_err()}")

        # Setup basic notebook cells
        setup_result = await setup_notebook_basics(session, state)
        if setup_result.is_err():
            logger.warning(f"Failed to setup notebook basics: {setup_result.unwrap_err()}")

        # Step 7: Run LLM analysis if requested
        if config.max_rounds > 0:
            # Import and run LLM analysis
            from .llm_interaction import run_llm_analysis

            llm_result = await run_llm_analysis(
                session=session,
                config=config,
                state=state,
                excel_path=config.excel_path,
                sheet_name=config.sheet_name,
                notebook_path=artifacts.notebook_path,
            )

            if llm_result.is_err():
                logger.error(f"LLM analysis failed: {llm_result.unwrap_err()}")

        # Step 8: Save notebook
        save_result = await save_notebook(session, artifacts.notebook_path)
        if save_result.is_err():
            logger.error(save_result.unwrap_err())
        else:
            # Clean up checkpoint files after successful completion
            # CLAUDE-KNOWLEDGE: Remove checkpoints only after successful save
            if config.auto_save_rounds:
                checkpoint_pattern = f"{artifacts.notebook_path.stem}_checkpoint_round*.ipynb"
                checkpoint_files = list(artifacts.notebook_path.parent.glob(checkpoint_pattern))
                for checkpoint in checkpoint_files:
                    try:
                        checkpoint.unlink()
                        logger.info(f"Removed checkpoint: {checkpoint}")
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint {checkpoint}: {e}")

    # Step 9: Log cost summary
    if state.cost_tracker and config.track_costs:
        log_cost_summary(state.cost_tracker, config.cost_limit)

    logger.info("Analysis session completed.")
    return ok(None)
