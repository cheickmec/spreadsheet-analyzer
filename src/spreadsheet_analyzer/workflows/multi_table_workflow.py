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

            # For now, create mock detection result
            # TODO: Integrate actual detector with notebook execution
            from ..agents.table_detection_types import TableBoundary, TableType

            # Mock detection - in real implementation, would run detector
            mock_boundaries = TableDetectionResult(
                sheet_name=state.get("sheet_name", f"Sheet{state['sheet_index']}"),
                tables=(
                    TableBoundary(
                        table_id="table_1",
                        description="Main data table",
                        start_row=0,
                        end_row=100,
                        start_col=0,
                        end_col=5,
                        confidence=0.9,
                        table_type=TableType.DETAIL,
                        entity_type="data",
                    ),
                ),
                detection_method="mechanical",
                metadata={"mock": True},
            )

            table_boundaries = mock_boundaries

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
        # Modify config to include table boundary information
        config = state["config"]

        # Create boundary summary for the analyst
        boundary_info = "DETECTED TABLE BOUNDARIES:\n\n"
        for i, table in enumerate(boundaries.tables):
            boundary_info += f"""Table {i + 1}: {table.description}
- Location: df.iloc[{table.start_row}:{table.end_row + 1}, {table.start_col}:{table.end_col + 1}]
- Entity Type: {table.entity_type}
- Row Count: {table.row_count}

"""

        # For now, return a placeholder
        # TODO: Integrate with actual analysis pipeline
        logger.info("Analysis with table boundaries not yet implemented")

        # Create a placeholder notebook path
        analysis_notebook = (
            Path(config.output_dir) / f"{Path(state['excel_file_path']).stem}_analysis_with_tables.ipynb"
        )

        return {
            "analysis_notebook_path": str(analysis_notebook),
            "messages": [AIMessage(content="✅ Analysis complete (placeholder)")],
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
