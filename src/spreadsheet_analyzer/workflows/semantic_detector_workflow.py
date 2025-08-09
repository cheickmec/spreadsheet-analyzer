"""Semantic layout detection workflow for spreadsheets.

This module implements semantic zone detection that creates navigational guides
for spreadsheet analysis rather than rigid table boundaries.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from structlog import get_logger

from ..cli.llm_interaction import create_llm_instance, track_llm_usage
from ..cli.notebook_analysis import AnalysisConfig
from ..cli.utils.naming import FileNameConfig, generate_session_id, get_short_hash
from ..core.types import Result, err, ok
from ..models.layout_detection import (
    ZONE_TYPE_DEFINITIONS,
    LayoutRegion,
    NavigationGuide,
    RegionCoordinates,
    SpreadsheetLayout,
    ZoneType,
)
from ..notebook_llm_interface import get_session_manager
from ..notebook_session import notebook_session
from ..observability import add_session_metadata, get_cost_tracker, phoenix_session
from ..prompts import get_prompt_definition, load_prompt

logger = get_logger(__name__)


class SemanticDetectionState(TypedDict):
    """State for semantic detection workflow."""

    # Input configuration
    excel_file_path: str
    sheet_index: int
    sheet_name: str | None
    config: AnalysisConfig

    # Detection outputs
    layout: SpreadsheetLayout | None
    detection_notebook_path: str | None
    detection_error: str | None

    # Workflow control
    messages: list[BaseMessage]
    workflow_complete: bool


async def detect_semantic_layout(state: SemanticDetectionState) -> dict[str, Any]:
    """Detect semantic zones in a spreadsheet.

    This function:
    1. Creates a notebook session for detection
    2. Loads the Excel file
    3. Runs semantic zone detection with LLM
    4. Returns a SpreadsheetLayout object
    """
    logger.info(
        "Starting semantic layout detection",
        excel_file=state["excel_file_path"],
        sheet_index=state["sheet_index"],
    )

    try:
        detector_model = state["config"].detector_model or state["config"].model

        # Get prompt hash for tracking
        detector_prompt_def = get_prompt_definition("layout_comprehension_system")
        prompt_hash = get_short_hash(detector_prompt_def.content_hash) if detector_prompt_def else None

        # Create file config
        file_config = FileNameConfig(
            excel_file=Path(state["excel_file_path"]),
            model=detector_model,
            sheet_index=state["sheet_index"],
            sheet_name=state.get("sheet_name"),
            max_rounds=state["config"].detector_max_rounds,
            session_id=f"semantic_{state['sheet_index']}",
            prompt_hash=prompt_hash,
        )

        # Create output directory
        output_dir = Path(state["config"].output_dir if state["config"].output_dir else "./outputs")
        detector_output_dir = output_dir / "semantic_detector"
        detector_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate notebook name
        from ..cli.utils.naming import generate_notebook_name

        detection_notebook_name = generate_notebook_name(file_config, include_timestamp=True)
        detection_notebook_name = detection_notebook_name.replace("_analysis_", "_semantic_")
        detection_notebook = detector_output_dir / detection_notebook_name

        # Run detection in isolated session
        async with notebook_session(
            session_id=f"semantic_{state['sheet_index']}",
            notebook_path=detection_notebook,
        ) as session:
            # Load Excel data
            load_code = f"""
import pandas as pd
import numpy as np
from pathlib import Path

excel_path = Path(r"{state["excel_file_path"]}")
df = pd.read_excel(excel_path, sheet_name={state["sheet_index"]})
sheet_rows, sheet_cols = df.shape
print(f"Loaded sheet with shape: {{sheet_rows}} rows × {{sheet_cols}} columns")

# Get sheet name
import pandas as pd
excel_file = pd.ExcelFile(excel_path)
sheet_names = excel_file.sheet_names
sheet_name = sheet_names[{state["sheet_index"]}]
print(f"Sheet name: {{sheet_name}}")
"""
            result = await session.execute(load_code)
            if result.is_err():
                return {
                    "detection_error": f"Failed to load Excel: {result.unwrap_err()}",
                    "messages": [AIMessage(content=f"Detection failed: {result.unwrap_err()}")],
                }

            # Analyze sheet structure for semantic zones
            analysis_code = """
print("=== SEMANTIC ZONE ANALYSIS ===\\n")

# 1. Check for header zones
print("1. POTENTIAL HEADER ZONES:")
for i in range(min(5, len(df))):
    row_data = df.iloc[i]
    non_null = row_data.notna().sum()
    text_count = sum(isinstance(v, str) for v in row_data if pd.notna(v))
    if text_count > len(row_data) * 0.5:  # More than 50% text
        print(f"  Row {i}: Likely header (text ratio: {text_count/non_null:.2f})")

# 2. Identify data zones
print("\\n2. DATA ZONE PATTERNS:")
# Look for consistent data patterns
data_start = -1
data_end = -1
for i in range(len(df)):
    row_density = df.iloc[i].notna().sum() / len(df.columns)
    if row_density > 0.3:  # At least 30% filled
        if data_start == -1:
            data_start = i
        data_end = i
print(f"  Main data region: rows {data_start} to {data_end}")

# 3. Check for empty navigation zones
print("\\n3. NAVIGATION ZONES (Empty Separators):")
# Check column headers first
for i, col in enumerate(df.columns):
    print(f"  Column {i}: '{col}'")

empty_cols = df.isnull().all(axis=0)
empty_col_indices = [i for i, col in enumerate(df.columns) if empty_cols[col]]
if empty_col_indices:
    print(f"  Columns with all empty values: {empty_col_indices}")
    # But check if they have headers
    for idx in empty_col_indices:
        col_name = df.columns[idx]
        if col_name and not pd.isna(col_name) and str(col_name).strip():
            print(f"    Column {idx} has header '{col_name}' - NOT a navigation zone, just empty data")
    # Check if these separate regions
    if empty_col_indices:
        regions = []
        start = 0
        for empty_col in empty_col_indices + [len(df.columns)]:
            if empty_col > start:
                region_df = df.iloc[:, start:empty_col]
                density = region_df.notna().sum().sum() / (region_df.shape[0] * region_df.shape[1])
                if density > 0.1:
                    regions.append((start, empty_col - 1, density))
            start = empty_col + 1
        if len(regions) > 1:
            print(f"  Found {len(regions)} regions separated by empty columns:")
            for i, (s, e, d) in enumerate(regions):
                print(f"    Region {i+1}: Columns {s}-{e}, density: {d:.2%}")

# 4. Check for summary zones
print("\\n4. SUMMARY/TOTAL ZONES:")
# Look for totals keywords
for i, col in enumerate(df.columns):
    col_str = str(col).lower()
    if any(kw in col_str for kw in ['total', 'sum', 'revenue', 'expense']):
        print(f"  Column {i} ('{col}'): Contains summary keyword")

# 5. Check for formula patterns
print("\\n5. FORMULA ZONE INDICATORS:")
# Check for calculated-looking values
for i, col in enumerate(df.columns):
    if df[col].dtype in ['float64', 'int64']:
        non_null = df[col].notna().sum()
        if non_null > 0:
            unique_ratio = df[col].nunique() / non_null
            if unique_ratio > 0.9:
                print(f"  Column {i}: High uniqueness ({unique_ratio:.2f}) - possible formulas")

# 6. Analyze the specific embedded totals issue
print("\\n6. EMBEDDED STRUCTURES:")
# Check first row for embedded totals
first_row = df.iloc[0] if len(df) > 0 else []
print(f"  First row values: {list(first_row)}")
# Check if totals are embedded in header
header_has_totals = any('total' in str(col).lower() for col in df.columns)
if header_has_totals:
    print("  ⚠️ TOTALS EMBEDDED IN HEADER ROW!")

print("\\n=== FULL DATA PREVIEW ===")
print(df.to_string())
"""

            analysis_result = await session.execute(analysis_code)
            if analysis_result.is_ok():
                analysis_output = (
                    analysis_result.unwrap().outputs[-1].content if analysis_result.unwrap().outputs else "No output"
                )
            else:
                analysis_output = "Analysis failed"

            # Register session for tools
            session_manager = get_session_manager()
            session_manager._sessions["semantic_session"] = session

            # Load semantic detection prompt
            prompt_result = load_prompt("layout_comprehension_system")
            if prompt_result.is_err():
                logger.error(f"Failed to load semantic prompt: {prompt_result.err_value}")
                prompt_data = {
                    "template": ZONE_TYPE_DEFINITIONS,
                    "input_variables": ["excel_file_name", "sheet_name", "sheet_dimensions"],
                }
            else:
                prompt_data = prompt_result.unwrap()

            # Format system prompt
            system_template = PromptTemplate(
                template=prompt_data["template"],
                input_variables=prompt_data["input_variables"],
            )

            system_prompt = system_template.format(
                excel_file_name=Path(state["excel_file_path"]).name,
                sheet_name=state.get("sheet_name", f"Sheet{state['sheet_index']}"),
                sheet_dimensions="Shape from analysis",
            )

            # Create LLM instance
            llm_result = create_llm_instance(detector_model, state["config"].api_key)
            if llm_result.is_err():
                return {
                    "detection_error": f"Failed to create LLM: {llm_result.unwrap_err()}",
                    "messages": [AIMessage(content=f"Detection failed: {llm_result.unwrap_err()}")],
                }

            llm = llm_result.unwrap()

            # Create the human message with clear instructions
            human_message = f"""Based on the analysis below, identify all semantic zones in this spreadsheet.

ANALYSIS RESULTS:
{analysis_output}

CRITICAL OBSERVATIONS:
- Sheet 0 has TOTALS EMBEDDED IN THE HEADER ROW (columns 5-8)
- Main transaction data table spans columns 0-4 (Date, Description, USD Amount, Transaction type, Category)
- Column 4 (Category) has a header but all data values are empty - it's still part of the data table
- Columns 5-8 contain summary values but only in the header row
- There is NO separator between the data table and the embedded totals

Your task:
1. Identify ALL semantic zones including the embedded totals
2. Create a Python dictionary called 'detected_layout' with your findings
3. Use the execute_code_semantic tool to create the variable

You MUST create detected_layout with this exact structure:

Create a dictionary with:
- layout_type: string describing the overall layout
- total_rows: use the sheet_rows variable
- total_cols: use the sheet_cols variable
- regions: list of region dictionaries, each with:
  - region_id: unique identifier
  - zone_type: one of (header, data, formula, summary, metadata, navigation, annotation, other)
  - semantic_role: what it represents
  - coordinates: dict with start_row, end_row, start_col, end_col (all 0-indexed)
  - relationships: list of related region IDs
  - navigation_hints: list of guidance strings
  - confidence: float between 0 and 1
  - description: brief description
- navigation_guide: dict with:
  - suggested_flow: list of region IDs in analysis order
  - key_insights: list of important observations
  - analysis_recommendations: list of analysis tips
  - warnings: list of any concerns

You MUST detect:
- The header zone for the main table (columns 0-4) as zone_type "header"
- The main data zone (transactions in columns 0-4, rows 1-9) as zone_type "data"
- The embedded summary zone (totals in header row, columns 5-8) as zone_type "summary"
- Do NOT mark column 4 as navigation - it has a "Category" header and is part of the data table

Execute code to create detected_layout now using the EXACT format shown above."""

            # Create tool for code execution
            @tool
            async def execute_code_semantic(code: str) -> str:
                """Execute Python code in the semantic detection session."""
                try:
                    result = await session.execute(code)
                    if result.is_ok():
                        outputs = result.unwrap().outputs
                        return outputs[-1].content if outputs else "Code executed successfully"
                    else:
                        return f"Error: {result.unwrap_err()}"
                except Exception as e:
                    return f"Execution failed: {e!s}"

            # Bind tools to LLM
            llm_with_tools = llm.bind_tools([execute_code_semantic])

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_message),
            ]

            # Track costs if enabled
            max_rounds = state["config"].detector_max_rounds

            # Wrap in Phoenix session if tracking
            session_id = f"semantic_{generate_session_id(file_config)}"

            with phoenix_session(session_id=session_id, user_id=None):
                add_session_metadata(
                    session_id,
                    {
                        "excel_file": Path(state["excel_file_path"]).name,
                        "sheet_index": state["sheet_index"],
                        "model": detector_model,
                        "agent_type": "SEMANTIC_DETECTOR",
                    },
                )

                # Run detection rounds
                for round_num in range(max_rounds):
                    logger.info(f"Semantic detection round {round_num + 1}/{max_rounds}")

                    try:
                        response = await llm_with_tools.ainvoke(messages)
                        messages.append(response)

                        # Track usage
                        if state["config"].track_costs:
                            await track_llm_usage(response, detector_model)

                        # Process tool calls
                        if hasattr(response, "tool_calls") and response.tool_calls:
                            for tool_call in response.tool_calls:
                                if tool_call["name"] == "execute_code_semantic":
                                    code = tool_call["args"].get("code", "")
                                    tool_result = await execute_code_semantic.ainvoke({"code": code})
                                    messages.append(
                                        ToolMessage(
                                            content=str(tool_result),
                                            tool_call_id=tool_call["id"],
                                            name=tool_call["name"],
                                        )
                                    )

                        # Check if complete
                        if response.content and "complete" in response.content.lower():
                            break

                    except Exception:
                        logger.exception("Error in semantic LLM call")
                        break

                # Log costs
                if state["config"].track_costs:
                    cost_tracker = get_cost_tracker()
                    cost_summary = cost_tracker.get_summary()
                    logger.info(f"Semantic detection cost: ${cost_summary['total_cost_usd']:.4f}")

            # Extract detected layout
            extract_code = """
# Extract the semantic layout
if 'detected_layout' in globals():
    layout_dict = detected_layout
    print(f"✅ Semantic layout detected with {len(layout_dict.get('regions', []))} regions")
    # Store it in a way we can extract
    _semantic_result = layout_dict
    _semantic_result
else:
    print("❌ No detected_layout variable found")
    _semantic_result = None
    None
"""
            extract_result = await session.execute(extract_code)

            # Try to get the layout directly from the session's namespace
            layout_dict = None
            try:
                # Check if we can access the kernel's namespace directly
                ns_code = """
import json
if 'detected_layout' in globals() and detected_layout:
    print(json.dumps(detected_layout))
else:
    print("null")
"""
                ns_result = await session.execute(ns_code)
                if ns_result.is_ok() and ns_result.unwrap().outputs:
                    import json

                    json_str = ns_result.unwrap().outputs[-1].content
                    logger.debug(f"JSON extraction result: {json_str[:100] if json_str else 'None'}")
                    if json_str and json_str != "null":
                        layout_dict = json.loads(json_str)
                        logger.info(f"Successfully extracted layout with {len(layout_dict.get('regions', []))} regions")
            except Exception as e:
                logger.warning(f"Failed to extract via namespace: {e}")

            if extract_result.is_err() and not layout_dict:
                # Fallback to single zone
                layout = SpreadsheetLayout(
                    layout_type="unknown",
                    regions=[
                        LayoutRegion(
                            region_id="full_sheet",
                            zone_type=ZoneType.OTHER,
                            semantic_role="full_spreadsheet",
                            coordinates=RegionCoordinates(
                                start_row=0,
                                end_row=9,
                                start_col=0,
                                end_col=8,
                            ),
                            confidence=0.5,
                            description="Fallback - full sheet as single zone",
                            navigation_hints=["Analyze entire sheet as one unit"],
                        )
                    ],
                    navigation_guide=NavigationGuide(
                        suggested_flow=["full_sheet"],
                        warnings=["Semantic detection failed - using fallback"],
                    ),
                    total_rows=10,
                    total_cols=9,
                )
            elif not layout_dict:
                # If we didn't get it via namespace, try parsing the extract result
                exec_output = extract_result.unwrap()

                if hasattr(exec_output, "outputs") and exec_output.outputs:
                    try:
                        last_output = exec_output.outputs[-1].content
                        # Try to parse as JSON string first (our new approach)
                        if isinstance(last_output, str):
                            import ast
                            import json

                            # Try JSON first (cleanest)
                            try:
                                layout_dict = json.loads(last_output)
                            except json.JSONDecodeError:
                                # Fall back to ast.literal_eval for Python dict strings
                                if last_output.strip().startswith("{"):
                                    layout_dict = ast.literal_eval(last_output)
                                # Handle the case where output contains the dict after some text
                                elif "✅" in last_output and "{" in last_output:
                                    # Extract just the dict part
                                    dict_start = last_output.index("{")
                                    dict_str = last_output[dict_start:]
                                    layout_dict = ast.literal_eval(dict_str)
                        # If output is already a dict (shouldn't happen but handle it)
                        elif isinstance(last_output, dict):
                            layout_dict = last_output
                    except (ValueError, SyntaxError, json.JSONDecodeError) as e:
                        logger.warning(f"Failed to parse layout dict: {e}, output type: {type(last_output)}")

            if layout_dict and isinstance(layout_dict, dict):
                # Convert to SpreadsheetLayout
                regions = []
                for region_data in layout_dict.get("regions", []):
                    coords = region_data.get("coordinates", {})

                    # Map zone type string to enum
                    zone_type_str = region_data.get("zone_type", "other").lower()
                    zone_type = ZoneType.OTHER
                    for zt in ZoneType:
                        if zt.value == zone_type_str:
                            zone_type = zt
                            break

                    region = LayoutRegion(
                        region_id=region_data.get("region_id", f"region_{len(regions)}"),
                        zone_type=zone_type,
                        semantic_role=region_data.get("semantic_role", "unknown"),
                        coordinates=RegionCoordinates(
                            start_row=coords.get("start_row", 0),
                            end_row=coords.get("end_row", 0),
                            start_col=coords.get("start_col", 0),
                            end_col=coords.get("end_col", 0),
                        ),
                        relationships=region_data.get("relationships", []),
                        navigation_hints=region_data.get("navigation_hints", []),
                        confidence=region_data.get("confidence", 0.8),
                        description=region_data.get("description"),
                        sample_values=region_data.get("sample_values"),
                    )
                    regions.append(region)

                nav_guide_data = layout_dict.get("navigation_guide", {})
                navigation_guide = NavigationGuide(
                    suggested_flow=nav_guide_data.get("suggested_flow", []),
                    key_insights=nav_guide_data.get("key_insights", []),
                    analysis_recommendations=nav_guide_data.get("analysis_recommendations", []),
                    warnings=nav_guide_data.get("warnings", []),
                )

                layout = SpreadsheetLayout(
                    layout_type=layout_dict.get("layout_type", "unknown"),
                    regions=regions,
                    navigation_guide=navigation_guide,
                    total_rows=layout_dict.get("total_rows", 10),
                    total_cols=layout_dict.get("total_cols", 9),
                    complexity_score=layout_dict.get("complexity_score"),
                    has_multiple_tables=len([r for r in regions if r.zone_type == ZoneType.DATA]) > 1,
                    has_formulas=any(r.zone_type == ZoneType.FORMULA for r in regions),
                    has_summaries=any(r.zone_type == ZoneType.SUMMARY for r in regions),
                )
            else:
                # Fallback
                layout = SpreadsheetLayout(
                    layout_type="unknown",
                    regions=[
                        LayoutRegion(
                            region_id="full_sheet",
                            zone_type=ZoneType.OTHER,
                            semantic_role="full_spreadsheet",
                            coordinates=RegionCoordinates(
                                start_row=0,
                                end_row=9,
                                start_col=0,
                                end_col=8,
                            ),
                            confidence=0.5,
                            description="Fallback - full sheet as single zone",
                            navigation_hints=["Analyze entire sheet as one unit"],
                        )
                    ],
                    navigation_guide=NavigationGuide(
                        suggested_flow=["full_sheet"],
                        warnings=["Failed to create proper layout - using fallback"],
                    ),
                    total_rows=10,
                    total_cols=9,
                )

            # Save summary to notebook
            summary_md = f"""# Semantic Layout Detection Results

{layout.to_detection_summary()}
"""

            # Add summary to notebook
            toolkit = session.toolkit
            if hasattr(toolkit, "_notebook_builder"):
                toolkit._notebook_builder.add_markdown_cell(summary_md)

            # Save notebook
            session.toolkit.save_notebook()
            logger.info(f"Semantic detection notebook saved to: {detection_notebook}")

            # Log results for benchmarking
            if state["config"].track_costs:
                try:
                    benchmark_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "model": detector_model,
                        "excel_file": Path(state["excel_file_path"]).name,
                        "sheet_index": state["sheet_index"],
                        "prompt_hash": prompt_hash or "unknown",
                        "layout_type": layout.layout_type,
                        "regions_detected": len(layout.regions),
                        "zone_types": {zt.value: len(layout.get_regions_by_type(zt)) for zt in ZoneType},
                        "has_navigation_guide": bool(layout.navigation_guide.suggested_flow),
                        "complexity_score": layout.complexity_score,
                        "cost_usd": cost_summary["total_cost_usd"] if "cost_summary" in locals() else 0,
                        "notebook_path": str(detection_notebook),
                    }

                    # Save to results file
                    results_file = Path("outputs/benchmarks/semantic_detection_results.json")
                    results_file.parent.mkdir(parents=True, exist_ok=True)

                    if results_file.exists():
                        with open(results_file) as f:
                            results = json.load(f)
                    else:
                        results = []

                    results.append(benchmark_entry)

                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)

                    logger.info(f"Semantic results logged to {results_file}")

                except Exception as e:
                    logger.warning(f"Failed to log semantic results: {e}")

            return {
                "layout": layout,
                "detection_notebook_path": str(detection_notebook),
                "messages": [AIMessage(content=f"✅ Detected {len(layout.regions)} semantic zones")],
            }

    except Exception as e:
        logger.exception("Error in semantic detector")
        return {
            "detection_error": str(e),
            "messages": [AIMessage(content=f"❌ Detection error: {e}")],
        }


def create_semantic_workflow() -> StateGraph:
    """Create workflow for semantic layout detection."""
    builder = StateGraph(SemanticDetectionState)

    # Single node workflow
    builder.add_node("detector", detect_semantic_layout)

    # Simple flow
    builder.add_edge(START, "detector")
    builder.add_edge("detector", END)

    return builder.compile()


async def run_semantic_detection(
    excel_path: Path,
    sheet_index: int = 0,
    config: AnalysisConfig | None = None,
) -> Result[SpreadsheetLayout, str]:
    """Run semantic layout detection on a spreadsheet.

    Args:
        excel_path: Path to Excel file
        sheet_index: Sheet to analyze
        config: Analysis configuration

    Returns:
        Result containing SpreadsheetLayout or error
    """
    try:
        if config is None:
            from ..cli.notebook_analysis import AnalysisConfig

            config = AnalysisConfig(
                excel_path=excel_path,
                sheet_index=sheet_index,
                output_dir=Path("./outputs"),
            )

        # Initialize state
        initial_state = SemanticDetectionState(
            excel_file_path=str(excel_path),
            sheet_index=sheet_index,
            sheet_name=None,
            config=config,
            layout=None,
            detection_notebook_path=None,
            detection_error=None,
            messages=[HumanMessage(content="Starting semantic layout detection...")],
            workflow_complete=False,
        )

        # Create and run workflow
        workflow = create_semantic_workflow()
        final_state = await workflow.ainvoke(initial_state)

        if final_state.get("detection_error"):
            return err(f"Detection failed: {final_state['detection_error']}")

        if final_state.get("layout"):
            return ok(final_state["layout"])
        else:
            return err("No layout detected")

    except Exception as e:
        logger.exception("Error in semantic detection workflow")
        return err(f"Workflow error: {e!s}")
