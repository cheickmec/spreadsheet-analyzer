"""Multi-agent workflow for table detection and analysis using LangGraph.

This module implements a supervisor-based workflow where a table detection
agent first identifies table boundaries, then passes this information to
the analysis agent for detailed examination.

CLAUDE-KNOWLEDGE: Using the private scratchpad pattern, each agent maintains
its own notebook session to avoid context pollution.
"""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from structlog import get_logger

from ..agents.table_detection_types import (
    TableBoundary,
    TableDetectionResult,
    TableType,
)
from ..cli.llm_interaction import create_llm_instance, track_llm_usage
from ..cli.notebook_analysis import (
    AnalysisArtifacts,
    AnalysisConfig,
    run_notebook_analysis,
)
from ..cli.utils.naming import (
    FileNameConfig,
    generate_log_name,
    generate_notebook_name,
    generate_session_id,
    get_cost_tracking_path,
    get_short_hash,
)
from ..core.types import Result, err, ok
from ..notebook_llm_interface import get_session_manager
from ..notebook_session import notebook_session
from ..observability import add_session_metadata, get_cost_tracker, phoenix_session
from ..prompts import get_prompt_definition, load_prompt

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
    elif state["table_boundaries"] is not None and state["config"].detector_only:
        logger.info("Detector-only mode: skipping analyst")
        return {
            "current_agent": "end",
            "workflow_complete": True,
            "messages": [
                AIMessage(content=f"Detection complete. Found {len(state['table_boundaries'].tables)} tables.")
            ],
        }
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
    """Run table detection in its own notebook session using LLM.

    This node:
    1. Creates a dedicated notebook session for detection
    2. Loads the Excel file
    3. Runs LLM-based table detection with iterative refinement
    4. Saves the detection notebook separately
    5. Returns only the table boundaries
    """
    logger.info(
        "Starting LLM-based table detection", excel_file=state["excel_file_path"], sheet_index=state["sheet_index"]
    )

    try:
        # Create detection-specific notebook path using same naming convention as analysis
        # Imports moved to top of file

        detector_model = state["config"].detector_model or state["config"].model

        # Get prompt hash for tracking
        detector_prompt_def = get_prompt_definition("table_detector_system")
        prompt_hash = get_short_hash(detector_prompt_def.content_hash) if detector_prompt_def else None

        # Create file config for detector
        detector_file_config = FileNameConfig(
            excel_file=Path(state["excel_file_path"]),
            model=detector_model,
            sheet_index=state["sheet_index"],
            sheet_name=state.get("sheet_name"),
            max_rounds=state["config"].detector_max_rounds,
            session_id=f"detection_{state['sheet_index']}",
            prompt_hash=prompt_hash,
        )

        output_dir = Path(state["config"].output_dir if state["config"].output_dir else "./outputs")
        detector_output_dir = output_dir / "detector"
        detector_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate detector notebook name with model info
        detection_notebook_name = generate_notebook_name(detector_file_config, include_timestamp=True)
        # Replace "analysis" with "detection" in the filename
        detection_notebook_name = detection_notebook_name.replace("_analysis_", "_detection_")
        detection_notebook = detector_output_dir / detection_notebook_name

        # Run detection in isolated session
        async with notebook_session(
            session_id=f"detection_{state['sheet_index']}", notebook_path=detection_notebook
        ) as session:
            # Load Excel data first
            load_code = f"""
import pandas as pd
from pathlib import Path

excel_path = Path(r"{state["excel_file_path"]}")
df = pd.read_excel(excel_path, sheet_name={state["sheet_index"]})
print(f"Loaded sheet with shape: {{df.shape}}")

# Quick preview for LLM context
print("\\nFirst 10 rows:")
print(df.head(10).to_string())
print("\\nLast 5 rows:")
print(df.tail(5).to_string())

# Get basic info
sheet_name = "{state.get("sheet_name", f"Sheet{state['sheet_index']}")}"
sheet_dimensions = f"{{df.shape[0]}} rows x {{df.shape[1]}} columns"
"""
            result = await session.execute(load_code)
            if result.is_err():
                return {
                    "detection_error": f"Failed to load Excel: {result.unwrap_err()}",
                    "messages": [AIMessage(content=f"Detection failed: {result.unwrap_err()}")],
                }

            # Use imported modules for LLM-based detection

            # Register session for tools access
            session_manager = get_session_manager()
            session_manager._sessions["detector_session"] = session

            # Load detector prompt with hash validation
            prompt_result = load_prompt("table_detector_system")
            if prompt_result.is_err():
                logger.error(f"Failed to load detector prompt: {prompt_result.err_value}")
                # Use a fallback prompt if loading fails
                prompt_data = {
                    "template": "You are a table detection agent. Find tables in the Excel sheet.",
                    "input_variables": ["excel_file_name", "sheet_name", "sheet_dimensions"],
                }
            else:
                prompt_data = prompt_result.unwrap()

            # Format the system prompt with actual data
            # Use imported PromptTemplate

            system_template = PromptTemplate(
                template=prompt_data["template"], input_variables=prompt_data["input_variables"]
            )

            # Get sheet info from notebook
            info_code = """
excel_file_name = excel_path.name
sheet_dimensions = f"{df.shape[0]} rows x {df.shape[1]} columns"
sheet_name
"""
            await session.execute(info_code)  # Execute to ensure variables are set

            system_prompt = system_template.format(
                excel_file_name=Path(state["excel_file_path"]).name,
                sheet_name=state.get("sheet_name", f"Sheet{state['sheet_index']}"),
                sheet_dimensions="Shape from notebook execution",
            )

            # Create LLM instance for detection with thinking support
            # Use detector-specific model if provided, otherwise fall back to main model
            detector_model = state["config"].detector_model or state["config"].model

            # Pass thinking config if available
            thinking_config = getattr(state["config"], "thinking_config", None)
            llm_result = create_llm_instance(detector_model, state["config"].api_key, thinking_config)
            if llm_result.is_err():
                return {
                    "detection_error": f"Failed to create LLM: {llm_result.unwrap_err()}",
                    "messages": [AIMessage(content=f"Detection failed: {llm_result.unwrap_err()}")],
                }

            llm = llm_result.unwrap()

            # Execute exploration code directly and provide results to LLM
            exploration_code = """
# Table Detection Analysis
print("=== DATA STRUCTURE ANALYSIS ===")
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\\n=== FIRST 10 ROWS ===")
print(df.head(10).to_string())
print("\\n=== LAST 5 ROWS ===")
print(df.tail(5).to_string())

print("\\n=== COLUMN ANALYSIS ===")
# Check for empty columns that might separate tables
for i, col in enumerate(df.columns):
    non_null_count = df[col].notna().sum()
    print(f"Column {i} ('{col}'): {non_null_count}/{len(df)} non-null values")

print("\\n=== EMPTY ROW ANALYSIS ===")
# Check for empty rows that might separate tables
empty_rows = df.isnull().all(axis=1)
if empty_rows.any():
    empty_row_indices = empty_rows[empty_rows].index.tolist()
    print(f"Empty rows found at indices: {empty_row_indices[:10]}")
else:
    print("No completely empty rows found")

print("\\n=== SIDE-BY-SIDE ANALYSIS ===")
# Check for potential side-by-side tables by examining column groups
left_cols = df.iloc[:, 0:7]  # First 7 columns
right_cols = df.iloc[:, 7:]  # Remaining columns
print(f"Left section shape: {left_cols.shape}")
print(f"Right section shape: {right_cols.shape}")
print(f"Left section non-null density: {left_cols.notna().sum().sum() / (left_cols.shape[0] * left_cols.shape[1]):.2f}")
print(f"Right section non-null density: {right_cols.notna().sum().sum() / (right_cols.shape[0] * right_cols.shape[1]):.2f}")
"""

            exploration_result = await session.execute(exploration_code)
            if exploration_result.is_ok():
                analysis_output = (
                    exploration_result.unwrap().outputs[-1].content
                    if exploration_result.unwrap().outputs
                    else "No output"
                )
            else:
                analysis_output = "Analysis failed"

            # Create messages for detection
            # Use imported message types and tool decorator

            human_message = f"""Based on the analysis results below, identify all table boundaries and create the detected_tables variable.

ANALYSIS RESULTS:
{analysis_output}

Your task:
1. Analyze the data structure shown above
2. Identify distinct tables (look for side-by-side patterns, different entity types, etc.)
3. Create a Python list called 'detected_tables' with your findings
4. Each table should specify exact boundaries (start_row, end_row, start_col, end_col)

Create the detected_tables variable now using the execute_code_detector tool."""

            # Create a simple function to execute code in the session
            async def execute_detector_code(code: str) -> str:
                """Execute Python code in the detector session."""
                try:
                    result = await session.execute(code)
                    if result.is_ok():
                        outputs = result.unwrap().outputs
                        return outputs[-1].content if outputs else "Code executed successfully"
                    else:
                        return f"Error: {result.unwrap_err()}"
                except Exception as e:
                    return f"Execution failed: {e!s}"

            # Create tool for LLM binding
            @tool
            async def execute_code_detector(code: str) -> str:
                """Execute Python code in the detector session."""
                return await execute_detector_code(code)

            # CLAUDE-GOTCHA: Gemini has a known bug where bind_tools() only recognizes
            # the first tool (GitHub langchain-google #369). Use bind(functions=) workaround.
            # CLAUDE-TEST-WORKAROUND: Different binding method for Gemini vs other models
            if "gemini" in detector_model.lower():
                try:
                    # Try the workaround first
                    llm_with_tools = llm.bind(functions=[execute_code_detector])
                    logger.debug("Using bind(functions=) workaround for Gemini detector")
                except Exception:
                    # Fallback to standard method if workaround fails
                    llm_with_tools = llm.bind_tools([execute_code_detector])
                    logger.debug("Fallback to bind_tools() for Gemini detector")
            else:
                llm_with_tools = llm.bind_tools([execute_code_detector])

            # Add Gemini-specific instructions if using a Gemini model
            if "gemini" in detector_model.lower():
                # CLAUDE-KNOWLEDGE: Gemini 2.X models need very explicit examples to use tools correctly
                # Concrete code examples significantly improve success rate (based on research)
                gemini_instructions = SystemMessage(
                    content="""
CRITICAL GEMINI-SPECIFIC INSTRUCTIONS FOR TABLE DETECTION:

1. DO NOT call pandas DataFrame methods as tools. These are NOT tools:
   - to_markdown, tolist, head, tail, describe, info, etc.

2. To run ANY Python code, you MUST use the execute_code_detector tool:
   CORRECT: execute_code_detector(code="print(df.shape)")
   WRONG: df.shape or print(df.shape) as direct calls

3. CONCRETE EXAMPLE - Create detected_tables like this:
   execute_code_detector(code=\"\"\"
# Detect tables based on analysis
detected_tables = [
    {
        'table_id': 'table_1',
        'description': 'Financial transactions with dates and amounts',
        'start_row': 0,
        'end_row': 100,
        'start_col': 0,
        'end_col': 5,
        'confidence': 0.95,
        'table_type': 'DETAIL',
        'entity_type': 'transactions'
    }
]
print(f"Created {len(detected_tables)} table definitions")
\"\"\")

4. ALWAYS use execute_code_detector for ALL Python operations
5. MUST create a variable named 'detected_tables' containing the list of tables

NEVER attempt to call DataFrame methods directly as tools!
"""
                )
                messages = [
                    SystemMessage(content=system_prompt),
                    gemini_instructions,
                    HumanMessage(content=human_message),
                ]
            else:
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=human_message),
                ]

            # Run detection with limited rounds (default 3 for detection)
            max_detector_rounds = state["config"].detector_max_rounds

            # Generate session ID for detector
            # Use imported naming utilities
            file_config = FileNameConfig(
                excel_file=Path(state["excel_file_path"]),
                model=detector_model,
                sheet_index=state["sheet_index"],
                sheet_name=state.get("sheet_name"),
                max_rounds=max_detector_rounds,
                session_id=None,
                prompt_hash=prompt_hash,  # Use the same hash from earlier
            )
            detector_session_id = f"detector_{generate_session_id(file_config)}"

            # Wrap detector execution in Phoenix session tracking
            with phoenix_session(
                session_id=detector_session_id,
                user_id=None,
            ):
                # Add session metadata
                add_session_metadata(
                    detector_session_id,
                    {
                        "excel_file": Path(state["excel_file_path"]).name,
                        "sheet_index": state["sheet_index"],
                        "sheet_name": state.get("sheet_name", f"Sheet{state['sheet_index']}"),
                        "model": detector_model,
                        "max_rounds": max_detector_rounds,
                        "agent_type": "TABLE_DETECTOR",
                        "cost_limit": state["config"].cost_limit if state["config"].track_costs else None,
                    },
                )

                for round_num in range(max_detector_rounds):
                    logger.info(f"Detector round {round_num + 1}/{max_detector_rounds}")

                    # Get LLM response
                    try:
                        response = await llm_with_tools.ainvoke(messages)
                        logger.debug(f"LLM response type: {type(response)}")
                        logger.debug(
                            f"LLM response content: {response.content[:200] if response.content else 'None'}..."
                        )
                        messages.append(response)

                        # Track token usage for cost tracking
                        if state["config"].track_costs:
                            await track_llm_usage(response, detector_model)

                        # Process any tool calls
                        # CLAUDE-GOTCHA: Gemini sometimes has empty tool_calls or puts them in unexpected places
                        tool_calls_found = hasattr(response, "tool_calls") and response.tool_calls

                        if tool_calls_found:
                            logger.debug(f"Found {len(response.tool_calls)} tool calls")
                        else:
                            logger.warning("No tool calls found in LLM response")
                            logger.debug(f"Response attributes: {dir(response)}")

                            # CLAUDE-PERFORMANCE: For Gemini, add explicit retry prompt if no tools used
                            if "gemini" in detector_model.lower() and round_num < max_detector_rounds - 1:
                                logger.info("Adding explicit tool usage instruction for Gemini")
                                retry_msg = HumanMessage(
                                    content="Please use the execute_code_detector tool NOW to create the detected_tables variable. "
                                    "Example: execute_code_detector(code=\"detected_tables = [{'table_id': 'table_1', ...}]\")"
                                )
                                messages.append(retry_msg)
                                continue  # Skip to next round

                        if hasattr(response, "tool_calls") and response.tool_calls:
                            for tool_call in response.tool_calls:
                                tool_name = tool_call["name"]
                                tool_args = tool_call["args"]
                                tool_id = tool_call["id"]

                                # Execute the detector tool directly
                                if tool_name == "execute_code_detector":
                                    try:
                                        # Extract code from tool args
                                        code = tool_args.get("code", "")
                                        tool_result = await execute_detector_code(code)
                                        messages.append(
                                            ToolMessage(content=str(tool_result), tool_call_id=tool_id, name=tool_name)
                                        )
                                    except Exception as e:
                                        logger.exception("Tool execution error")
                                        messages.append(
                                            ToolMessage(content=f"Error: {e!s}", tool_call_id=tool_id, name=tool_name)
                                        )
                                else:
                                    logger.warning(f"Unknown tool: {tool_name}")
                                    messages.append(
                                        ToolMessage(
                                            content=f"Unknown tool: {tool_name}", tool_call_id=tool_id, name=tool_name
                                        )
                                    )

                        # Check if detection is complete
                        if (
                            response.content
                            and isinstance(response.content, str)
                            and "detection complete" in response.content.lower()
                        ):
                            break

                    except Exception:
                        logger.exception("Error in detector LLM call")
                        break

                # Log cost summary if tracking enabled
                if state["config"].track_costs:
                    try:
                        cost_tracker = get_cost_tracker()
                        cost_summary = cost_tracker.get_summary()
                        logger.info("💰 Detector Cost Summary:")
                        logger.info(f"  Total Cost: ${cost_summary['total_cost_usd']:.4f}")
                        logger.info(f"  Total Tokens: {cost_summary['total_tokens']['total']:,}")
                    except Exception:
                        logger.debug("Failed to log detector cost summary")

            # Extract detection results from notebook
            # CLAUDE-IMPORTANT: The LLM MUST have created a variable called 'detected_tables'
            # This is the contract between the detector and the extraction logic
            extract_code = """
# Extract detection results created by LLM
# The detector should have created a 'detected_tables' variable with the results

# CLAUDE-TEST-WORKAROUND: Validate that detected_tables exists and is properly formatted
if 'detected_tables' in globals():
    # Validate it's a list
    if isinstance(detected_tables, list):
        detection_results = detected_tables
        print(f"✅ Found {len(detection_results)} tables from LLM detection")
        # Validate first table has required fields (if any tables exist)
        if detection_results:
            required_fields = ['table_id', 'description', 'start_row', 'end_row', 'start_col', 'end_col']
            first_table = detection_results[0]
            missing_fields = [f for f in required_fields if f not in first_table]
            if missing_fields:
                print(f"⚠️ Warning: First table missing fields: {missing_fields}")
    else:
        print(f"❌ Error: detected_tables is not a list, it's a {type(detected_tables)}")
        detection_results = []
else:
    # CLAUDE-GOTCHA: Gemini sometimes fails to create the variable even after multiple prompts
    print("❌ No 'detected_tables' variable found - LLM failed to complete detection")
    detection_results = []

detection_results
"""
            extract_result = await session.execute(extract_code)

            # Import types for result processing
            # Use imported table detection types

            if extract_result.is_ok():
                exec_output = extract_result.unwrap()
                logger.info(f"LLM detection result type: {type(exec_output)}")

                # Extract the actual result from CellExecution if needed
                actual_result = None
                if hasattr(exec_output, "outputs") and exec_output.outputs:
                    # Try to get the result from the last output
                    try:
                        import ast

                        last_output = exec_output.outputs[-1].content
                        # Try to parse it as a Python literal
                        actual_result = ast.literal_eval(last_output)
                    except (ValueError, SyntaxError, IndexError):
                        # If that doesn't work, the actual result might be the variable directly
                        actual_result = None
                elif isinstance(exec_output, list):
                    actual_result = exec_output

                # Check if we got the detected tables list
                if isinstance(actual_result, list) and len(actual_result) > 0:
                    # Convert LLM detection results to TableBoundary objects
                    table_boundaries_list: list[TableBoundary] = []

                    for table_info in actual_result:
                        if isinstance(table_info, dict):
                            # Parse table type from string
                            type_str = table_info.get("table_type", "DETAIL").upper()
                            if type_str == "SUMMARY":
                                table_type = TableType.SUMMARY
                            elif type_str == "HEADER":
                                table_type = TableType.HEADER
                            elif type_str == "PIVOT":
                                table_type = TableType.PIVOT
                            elif type_str == "LOOKUP":
                                table_type = TableType.LOOKUP
                            else:
                                table_type = TableType.DETAIL

                            table_boundaries_list.append(
                                TableBoundary(
                                    table_id=table_info.get("table_id", f"table_{len(table_boundaries_list) + 1}"),
                                    description=table_info.get("description", "LLM-detected table"),
                                    start_row=table_info.get("start_row", 0),
                                    end_row=table_info.get("end_row", 0),
                                    start_col=table_info.get("start_col", 0),
                                    end_col=table_info.get("end_col", 0),
                                    confidence=float(table_info.get("confidence", 0.8)),
                                    table_type=table_type,
                                    entity_type=table_info.get("entity_type", "data"),
                                )
                            )

                    if table_boundaries_list:
                        table_boundaries = TableDetectionResult(
                            sheet_name=state.get("sheet_name", f"Sheet{state['sheet_index']}"),
                            tables=tuple(table_boundaries_list),
                            metadata={
                                "method": "llm_detection",
                                "model": detector_model,
                                "rounds": min(round_num + 1, max_detector_rounds),
                                "max_rounds": max_detector_rounds,
                            },
                        )
                    else:
                        logger.warning("No valid tables found in LLM detection results, using fallback")
                        # Get dimensions for fallback
                        dim_result = await session.execute("df_shape = df.shape; df_shape")
                        if dim_result.is_ok():
                            shape_output = dim_result.unwrap()
                            # Extract the actual shape tuple from the output
                            if hasattr(shape_output, "outputs") and shape_output.outputs:
                                shape_str = shape_output.outputs[-1].content
                                # Parse (rows, cols) from string
                                import re

                                match = re.match(r"\((\d+),\s*(\d+)\)", shape_str)
                                if match:
                                    df_rows, df_cols = int(match.group(1)), int(match.group(2))
                                else:
                                    df_rows, df_cols = 100, 10
                            else:
                                df_rows, df_cols = 100, 10
                        else:
                            df_rows, df_cols = 100, 10

                        table_boundaries = TableDetectionResult(
                            sheet_name=state.get("sheet_name", f"Sheet{state['sheet_index']}"),
                            tables=(
                                TableBoundary(
                                    table_id="table_1",
                                    description="Full spreadsheet (LLM detection found no boundaries)",
                                    start_row=0,
                                    end_row=df_rows - 1,
                                    start_col=0,
                                    end_col=df_cols - 1,
                                    confidence=0.5,
                                    table_type=TableType.DETAIL,
                                    entity_type="data",
                                ),
                            ),
                            metadata={"fallback": True, "reason": "no_boundaries_detected"},
                        )
                else:
                    logger.warning(f"Unexpected LLM result type: {type(exec_output)}")
                    # Get dimensions for fallback
                    dim_result = await session.execute("df_shape = df.shape; df_shape")
                    if dim_result.is_ok():
                        shape_output = dim_result.unwrap()
                        # Extract the actual shape tuple from the output
                        if hasattr(shape_output, "outputs") and shape_output.outputs:
                            shape_str = shape_output.outputs[-1].content
                            # Parse (rows, cols) from string
                            import re

                            match = re.match(r"\((\d+),\s*(\d+)\)", shape_str)
                            if match:
                                df_rows, df_cols = int(match.group(1)), int(match.group(2))
                            else:
                                df_rows, df_cols = 100, 10
                        else:
                            df_rows, df_cols = 100, 10
                    else:
                        df_rows, df_cols = 100, 10

                    table_boundaries = TableDetectionResult(
                        sheet_name=state.get("sheet_name", f"Sheet{state['sheet_index']}"),
                        tables=(
                            TableBoundary(
                                table_id="table_1",
                                description="Full spreadsheet (fallback)",
                                start_row=0,
                                end_row=df_rows - 1,
                                start_col=0,
                                end_col=df_cols - 1,
                                confidence=0.5,
                                table_type=TableType.DETAIL,
                                entity_type="data",
                            ),
                        ),
                        metadata={"fallback": True, "reason": "unexpected_result_type"},
                    )
            else:
                logger.error(f"Failed to extract detection results: {extract_result.unwrap_err()}")
                # Get dimensions for fallback
                dim_result = await session.execute("df_shape = df.shape; df_shape")
                if dim_result.is_ok():
                    shape_output = dim_result.unwrap()
                    # Extract the actual shape tuple from the output
                    if hasattr(shape_output, "outputs") and shape_output.outputs:
                        shape_str = shape_output.outputs[-1].content
                        # Parse (rows, cols) from string
                        import re

                        match = re.match(r"\((\d+),\s*(\d+)\)", shape_str)
                        if match:
                            df_rows, df_cols = int(match.group(1)), int(match.group(2))
                        else:
                            df_rows, df_cols = 100, 10
                    else:
                        df_rows, df_cols = 100, 10
                else:
                    df_rows, df_cols = 100, 10

                # Fallback to single table
                table_boundaries = TableDetectionResult(
                    sheet_name=state.get("sheet_name", f"Sheet{state['sheet_index']}"),
                    tables=(
                        TableBoundary(
                            table_id="table_1",
                            description="Full spreadsheet (fallback)",
                            start_row=0,
                            end_row=df_rows - 1,
                            start_col=0,
                            end_col=df_cols - 1,
                            confidence=0.5,
                            table_type=TableType.DETAIL,
                            entity_type="data",
                        ),
                    ),
                    metadata={"fallback": True, "reason": "execution_error"},
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

            # Add markdown cell with summary

            # Get the notebook toolkit from session
            toolkit = session.toolkit
            if hasattr(toolkit, "_notebook_builder"):
                toolkit._notebook_builder.add_markdown_cell(summary_code)
            else:
                # Fallback: execute as markdown magic
                await session.execute(f"# Detection Summary\n\n{summary_code}")

            # Save detection notebook
            session.toolkit.save_notebook()

            logger.info(f"Detection complete: found {len(table_boundaries.tables)} tables")
            logger.info(f"Detection notebook saved to: {detection_notebook}")

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
        # Use imported analysis and naming utilities

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

        # Determine which prompt will be used and get its hash
        analyst_prompt_name = "table_aware_analyst_system" if boundaries else "data_analyst_system"
        analyst_prompt_def = get_prompt_definition(analyst_prompt_name)
        analyst_hash = get_short_hash(analyst_prompt_def.content_hash) if analyst_prompt_def else None

        # Generate file names config first
        file_config = FileNameConfig(
            excel_file=Path(state["excel_file_path"]),
            model=config.model,
            sheet_index=state["sheet_index"],
            sheet_name=state.get("sheet_name"),
            prompt_hash=analyst_hash,
        )

        # Generate new session ID for analysis
        session_id = generate_session_id(file_config)

        # Create artifacts for table-aware analysis
        notebook_path = generate_notebook_name(file_config)
        output_dir = Path(config.output_dir if config.output_dir else "./outputs")
        analyst_output_dir = output_dir / "analyst"
        analyst_output_dir.mkdir(parents=True, exist_ok=True)
        log_path = analyst_output_dir / generate_log_name(file_config)
        cost_tracking_path = get_cost_tracking_path(file_config, analyst_output_dir)

        artifacts = AnalysisArtifacts(
            session_id=session_id,
            notebook_path=analyst_output_dir / notebook_path,
            log_path=log_path,
            cost_tracking_path=cost_tracking_path,
            file_config=file_config,
        )

        # Create a modified config that includes table boundaries
        # We need to create a new config instance with table boundaries
        # Use imported replace from dataclasses

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
            return "__end__"  # END is a sentinel value, return its string representation

    builder.add_conditional_edges(
        "supervisor", route_supervisor, {"detector": "detector", "analyst": "analyst", "__end__": END}
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
