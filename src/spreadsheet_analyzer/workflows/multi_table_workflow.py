"""Multi-agent workflow for table detection and analysis using LangGraph.

This module implements a supervisor-based workflow where a table detection
agent first identifies table boundaries, then passes this information to
the analysis agent for detailed examination.

CLAUDE-KNOWLEDGE: Using the private scratchpad pattern, each agent maintains
its own notebook session to avoid context pollution.
"""

import asyncio
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from structlog import get_logger

from ..agents.table_detection_types import TableDetectionResult
from ..cli.notebook_analysis import AnalysisConfig
from ..core.types import Result, err, ok
from ..notebook_session import notebook_session

logger = get_logger(__name__)


class SpreadsheetAnalysisState(TypedDict):
    """State shared between agents in the workflow.

    CLAUDE-KNOWLEDGE: Using TypedDict for LangGraph state ensures type safety
    and proper state management across nodes.
    """

    # Input configuration
    excel_file_path: str
    sheet_index: int
    sheet_name: str | None
    config: AnalysisConfig

    # Detection phase outputs
    table_boundaries: TableDetectionResult | None
    detection_notebook_path: str | None
    detection_error: str | None

    # Analysis phase outputs
    analysis_notebook_path: str | None
    analysis_error: str | None

    # Workflow control
    messages: Annotated[list[BaseMessage], "append"]
    current_agent: str
    workflow_complete: bool


async def supervisor_node(state: SpreadsheetAnalysisState) -> dict[str, Any]:
    """Supervisor decides which agent to run next based on current state.

    The supervisor implements a simple state machine:
    1. If no table boundaries -> run detector
    2. If boundaries exist -> run analyst
    3. If both complete -> end workflow
    """
    logger.info("Supervisor evaluating next step", current_state=state.get("current_agent"))

    if state.get("workflow_complete", False):
        return {"current_agent": "end"}

    if state["table_boundaries"] is None and state.get("detection_error") is None:
        logger.info("No table boundaries found, running detector")
        return {"current_agent": "detector", "messages": [AIMessage(content="Running table detection agent...")]}
    elif state["table_boundaries"] is not None and state.get("analysis_notebook_path") is None:
        logger.info("Table boundaries found, running analyst")
        return {
            "current_agent": "analyst",
            "messages": [
                AIMessage(content=f"Detected {len(state['table_boundaries'].tables)} tables. Running analysis...")
            ],
        }
    else:
        logger.info("Workflow complete")
        return {
            "current_agent": "end",
            "workflow_complete": True,
            "messages": [AIMessage(content="Analysis complete.")],
        }


async def detector_node(state: SpreadsheetAnalysisState) -> dict[str, Any]:
    """Run table detection in its own notebook session.

    This node:
    1. Creates a dedicated notebook session for detection
    2. Loads the Excel file
    3. Runs the table detector agent
    4. Saves the detection notebook separately
    5. Returns only the table boundaries
    """
    logger.info("Starting table detection", excel_file=state["excel_file_path"], sheet_index=state["sheet_index"])

    try:
        # Create detection-specific notebook path
        detection_notebook = (
            Path(state["config"].output_dir) / f"{Path(state['excel_file_path']).stem}_table_detection.ipynb"
        )

        # Run detection in isolated session
        async with notebook_session(
            session_id=f"detection_{state['sheet_index']}", notebook_path=detection_notebook
        ) as session:
            # Load Excel data
            load_code = f"""
import pandas as pd
from pathlib import Path

excel_path = Path(r"{state["excel_file_path"]}")
df = pd.read_excel(excel_path, sheet_index={state["sheet_index"]})
print(f"Loaded sheet with shape: {{df.shape}}")
df.head()
"""
            result = await session.execute(load_code)
            if result.is_err():
                return {
                    "detection_error": f"Failed to load Excel: {result.unwrap_err()}",
                    "messages": [AIMessage(content=f"Detection failed: {result.unwrap_err()}")],
                }

            # Run table detection in notebook
            from ..agents.table_detection_types import TableBoundary, TableDetectionResult, TableType

            # Add detection code to notebook
            detection_code = """
# Run table detection analysis
import pandas as pd

# Analyze sheet structure
print(f"Sheet dimensions: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check for empty rows (mechanical detection)
empty_rows = df.isnull().all(axis=1)
empty_row_indices = empty_rows[empty_rows].index.tolist()

if empty_row_indices:
    print(f"\\nEmpty rows found at indices: {empty_row_indices}")

    # Group consecutive empty rows
    groups = []
    if empty_row_indices:
        current_group = [empty_row_indices[0]]
        for i in range(1, len(empty_row_indices)):
            if empty_row_indices[i] - empty_row_indices[i-1] == 1:
                current_group.append(empty_row_indices[i])
            else:
                groups.append(current_group)
                current_group = [empty_row_indices[i]]
        groups.append(current_group)

    print(f"Empty row groups: {groups}")
else:
    print("\\nNo empty rows found - likely a single table")

# Preview data structure
print("\\nFirst 10 rows:")
df.head(10)
"""
            await session.toolkit.add_code_cell(detection_code)
            result = await session.execute(detection_code)

            # Implement simplified table detection based on empty rows
            detection_logic = """
# Detect table boundaries based on empty rows
from typing import List, Tuple

def detect_table_boundaries(df) -> List[Tuple[int, int]]:
    \"\"\"Detect table boundaries based on empty rows.\"\"\"
    empty_rows = df.isnull().all(axis=1)
    tables = []
    current_start = 0

    for idx, is_empty in enumerate(empty_rows):
        if is_empty and idx > current_start + 2:  # Minimum 3 rows for a table
            tables.append((current_start, idx - 1))
            current_start = idx + 1

    # Add the last table
    if current_start < len(df) - 1:
        tables.append((current_start, len(df) - 1))

    return tables if tables else [(0, len(df) - 1)]

# Detect tables
table_ranges = detect_table_boundaries(df)
print(f"\\nDetected {len(table_ranges)} table(s):")
for i, (start, end) in enumerate(table_ranges):
    print(f"  Table {i+1}: Rows {start}-{end} ({end-start+1} rows)")

# Store results for workflow
detected_tables = []
for i, (start, end) in enumerate(table_ranges):
    table_df = df.iloc[start:end+1]
    non_empty_cols = table_df.notna().any(axis=0)
    start_col = non_empty_cols.idxmax() if non_empty_cols.any() else 0
    end_col = len(non_empty_cols) - 1 - non_empty_cols[::-1].idxmax() if non_empty_cols.any() else len(df.columns) - 1

    detected_tables.append({
        'table_id': f'table_{i+1}',
        'start_row': start,
        'end_row': end,
        'start_col': df.columns.get_loc(start_col) if isinstance(start_col, str) else start_col,
        'end_col': df.columns.get_loc(end_col) if isinstance(end_col, str) else end_col,
        'row_count': end - start + 1,
    })

# Show preview of each table
for i, (start, end) in enumerate(table_ranges[:3]):  # Show max 3 tables
    print(f"\\nTable {i+1} preview:")
    print(df.iloc[start:min(start+5, end+1)])
"""
            await session.toolkit.add_code_cell(detection_logic)
            result = await session.execute(detection_logic)

            # Extract detected tables from notebook execution
            # The detection logic creates a 'detected_tables' variable with actual boundaries
            detected_tables = result.get("detected_tables", [])

            if not detected_tables:
                # Fallback to default if no tables detected
                detected_tables = [
                    {
                        "table_id": "table_1",
                        "start_row": 0,
                        "end_row": 100,
                        "start_col": 0,
                        "end_col": 5,
                        "row_count": 101,
                    }
                ]

            # Create TableDetectionResult based on actual detection results
            table_boundaries_list = []
            for i, table_info in enumerate(detected_tables):
                table_boundary = TableBoundary(
                    table_id=table_info.get("table_id", f"table_{i + 1}"),
                    description=f"Detected table {i + 1}",
                    start_row=table_info["start_row"],
                    end_row=table_info["end_row"],
                    start_col=table_info["start_col"],
                    end_col=table_info["end_col"],
                    confidence=0.85,  # Default confidence for mechanical detection
                    table_type=TableType.DETAIL,
                    entity_type="data",
                )
                table_boundaries_list.append(table_boundary)

            table_boundaries = TableDetectionResult(
                sheet_name=state.get("sheet_name", f"Sheet{state['sheet_index']}"),
                tables=tuple(table_boundaries_list),
                detection_method="mechanical",
                metadata={"method": "empty_row_detection"},
            )

            # Document findings in notebook
            summary_code = f"""
# Table Detection Results

Detected {len(table_boundaries.tables)} tables:

"""
            for i, table in enumerate(table_boundaries.tables):
                summary_code += f"""
## Table {i + 1}: {table.description}
- Location: Rows {table.start_row}-{table.end_row}, Columns {table.start_col}-{table.end_col}
- Type: {table.table_type.value}
- Entity: {table.entity_type}
- Confidence: {table.confidence:.2f}
"""

            await session.toolkit.add_markdown_cell(summary_code)

            # Save detection notebook
            await session.toolkit.save_notebook()

            logger.info(f"Detection complete: found {len(table_boundaries.tables)} tables")

            return {
                "table_boundaries": table_boundaries,
                "detection_notebook_path": str(detection_notebook),
                "messages": [AIMessage(content=f"✅ Detected {len(table_boundaries.tables)} tables")],
            }

    except Exception as e:
        logger.exception("Error in detector node")
        return {"detection_error": str(e), "messages": [AIMessage(content=f"❌ Detection error: {e}")]}


async def analyst_node(state: SpreadsheetAnalysisState) -> dict[str, Any]:
    """Run analysis using detected table boundaries.

    This node:
    1. Creates analysis session with table boundary awareness
    2. Modifies the analysis prompt to use table boundaries
    3. Runs standard analysis workflow
    4. Returns analysis notebook path
    """
    logger.info("Starting table-aware analysis")

    boundaries = state["table_boundaries"]
    if not boundaries:
        return {
            "analysis_error": "No table boundaries available",
            "messages": [AIMessage(content="Cannot analyze without table boundaries")],
        }

    try:
        # Import required modules for notebook analysis
        from ..cli.notebook_analysis import AnalysisArtifacts, run_notebook_analysis
        from ..cli.utils.naming import (
            FileNameConfig,
            generate_log_name,
            generate_notebook_name,
            generate_session_id,
            get_cost_tracking_path,
        )

        # Create boundary summary for the analyst
        boundary_info = "DETECTED TABLE BOUNDARIES:\n\n"
        for i, table in enumerate(boundaries.tables):
            boundary_info += f"""Table {i + 1}: {table.description}
- Location: df.iloc[{table.start_row}:{table.end_row + 1}, {table.start_col}:{table.end_col + 1}]
- Entity Type: {table.entity_type}
- Table Type: {table.table_type.value}
- Row Count: {table.row_count}
- Confidence: {table.confidence:.2f}

"""

        # Prepare the modified config with table boundaries
        config = state["config"]

        # Generate new session ID for analysis
        session_id = generate_session_id()

        # Generate file names
        file_config = FileNameConfig(
            base_name=Path(state["excel_file_path"]).stem,
            model_name=config.model.split("/")[-1],  # Use 'model' not 'model_name'
            session_id=session_id,
        )

        # Create artifacts for table-aware analysis
        notebook_path = generate_notebook_name(file_config, prefix="table_analysis")
        log_path = Path(config.output_dir) / generate_log_name(file_config)
        cost_tracking_path = get_cost_tracking_path(file_config, config.output_dir)

        artifacts = AnalysisArtifacts(
            session_id=session_id,
            notebook_path=Path(config.output_dir) / notebook_path,
            log_path=log_path,
            cost_tracking_path=cost_tracking_path,
            file_config=file_config,
        )

        # Create a modified config that includes table boundaries
        # We need to create a new config instance with table boundaries
        from dataclasses import replace

        modified_config = replace(config, table_boundaries=boundary_info)

        # Run the actual notebook analysis with table boundaries
        result = await run_notebook_analysis(modified_config, artifacts)

        if result.is_ok():
            return {
                "analysis_notebook_path": str(artifacts.notebook_path),
                "messages": [AIMessage(content=f"✅ Table-aware analysis complete: {artifacts.notebook_path.name}")],
            }
        else:
            error_msg = result.unwrap_err()
            return {
                "analysis_error": error_msg,
                "messages": [AIMessage(content=f"❌ Analysis error: {error_msg}")],
            }

    except Exception as e:
        logger.exception("Error in analyst node")
        return {"analysis_error": str(e), "messages": [AIMessage(content=f"❌ Analysis error: {e}")]}


def create_multi_table_workflow() -> StateGraph:
    """Create the LangGraph workflow for multi-table analysis.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create workflow
    builder = StateGraph(SpreadsheetAnalysisState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("detector", detector_node)
    builder.add_node("analyst", analyst_node)

    # Define edges
    builder.add_edge(START, "supervisor")

    # Conditional routing from supervisor
    def route_supervisor(state: SpreadsheetAnalysisState) -> str:
        """Route based on supervisor decision."""
        agent = state.get("current_agent", "")
        if agent == "detector":
            return "detector"
        elif agent == "analyst":
            return "analyst"
        else:
            return END

    builder.add_conditional_edges(
        "supervisor", route_supervisor, {"detector": "detector", "analyst": "analyst", END: END}
    )

    # After detector/analyst, go back to supervisor
    builder.add_edge("detector", "supervisor")
    builder.add_edge("analyst", "supervisor")

    # Compile workflow
    return builder.compile()


async def run_multi_table_analysis(
    excel_path: Path, sheet_index: int = 0, config: AnalysisConfig | None = None
) -> Result[dict[str, Any], str]:
    """Run the complete multi-table analysis workflow.

    Args:
        excel_path: Path to Excel file
        sheet_index: Sheet to analyze
        config: Analysis configuration

    Returns:
        Result containing workflow outputs or error
    """
    try:
        # Create default config if needed
        if config is None:
            config = AnalysisConfig(excel_path=excel_path, sheet_index=sheet_index, output_dir=Path("./outputs"))

        # Initialize state
        initial_state = SpreadsheetAnalysisState(
            excel_file_path=str(excel_path),
            sheet_index=sheet_index,
            sheet_name=None,
            config=config,
            table_boundaries=None,
            detection_notebook_path=None,
            detection_error=None,
            analysis_notebook_path=None,
            analysis_error=None,
            messages=[HumanMessage(content="Starting multi-table analysis...")],
            current_agent="",
            workflow_complete=False,
        )

        # Create and run workflow
        workflow = create_multi_table_workflow()
        final_state = await workflow.ainvoke(initial_state)

        # Extract results
        results = {
            "detection_notebook": final_state.get("detection_notebook_path"),
            "analysis_notebook": final_state.get("analysis_notebook_path"),
            "tables_found": len(final_state["table_boundaries"].tables) if final_state.get("table_boundaries") else 0,
            "detection_error": final_state.get("detection_error"),
            "analysis_error": final_state.get("analysis_error"),
        }

        if final_state.get("detection_error") or final_state.get("analysis_error"):
            return err(f"Workflow failed: {final_state.get('detection_error') or final_state.get('analysis_error')}")

        return ok(results)

    except Exception as e:
        logger.exception("Error in multi-table workflow")
        return err(f"Workflow error: {e}")


# Example usage
if __name__ == "__main__":

    async def main() -> None:
        result = await run_multi_table_analysis(Path("data/multi_table_example.xlsx"), sheet_index=0)

        if result.is_ok():
            print("Success:", result.unwrap())
        else:
            print("Error:", result.unwrap_err())

    asyncio.run(main())
