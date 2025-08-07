"""Layout comprehension workflow for semantic understanding of spreadsheet structure.

This module implements a semantic layout detector that identifies zones and relationships
in spreadsheets, providing navigational guides for analysis rather than rigid table boundaries.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from structlog import get_logger

from ..cli.llm_interaction import create_llm_instance, track_llm_usage
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
from ..prompts import get_prompt_definition

logger = get_logger(__name__)


class LayoutComprehensionConfig:
    """Configuration for layout comprehension."""

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        api_key: str | None = None,
        max_rounds: int = 5,
        confidence_threshold: float = 0.7,
        track_costs: bool = True,
        output_dir: Path = Path("./outputs"),
        verbose: bool = True,
    ):
        self.model = model
        self.api_key = api_key
        self.max_rounds = max_rounds
        self.confidence_threshold = confidence_threshold
        self.track_costs = track_costs
        self.output_dir = output_dir
        self.verbose = verbose


async def detect_layout_zones(
    excel_path: Path,
    sheet_index: int,
    config: LayoutComprehensionConfig,
) -> Result[SpreadsheetLayout, str]:
    """Detect semantic zones in a spreadsheet and create layout comprehension.

    Args:
        excel_path: Path to Excel file
        sheet_index: Sheet to analyze
        config: Configuration for layout detection

    Returns:
        Result containing SpreadsheetLayout or error message
    """
    logger.info(
        "Starting layout comprehension",
        excel_file=excel_path,
        sheet_index=sheet_index,
        model=config.model,
    )

    try:
        # Get prompt hash for tracking
        prompt_def = get_prompt_definition("layout_comprehension_system")
        prompt_hash = get_short_hash(prompt_def.content_hash) if prompt_def else None

        # Create file config
        file_config = FileNameConfig(
            excel_file=excel_path,
            model=config.model,
            sheet_index=sheet_index,
            sheet_name=None,
            max_rounds=config.max_rounds,
            session_id=f"layout_{sheet_index}",
            prompt_hash=prompt_hash,
        )

        # Create output directory
        detector_output_dir = config.output_dir / "layout_detector"
        detector_output_dir.mkdir(parents=True, exist_ok=True)

        # Create notebook path
        notebook_name = f"{excel_path.stem}_sheet{sheet_index}_layout_{config.model.replace('.', '_')}_{prompt_hash or 'nohash'}.ipynb"
        notebook_path = detector_output_dir / notebook_name

        # Run detection in isolated session
        async with notebook_session(
            session_id=f"layout_{sheet_index}",
            notebook_path=notebook_path,
        ) as session:
            # Load Excel data
            load_code = f"""
import pandas as pd
import numpy as np
from pathlib import Path

excel_path = Path(r"{excel_path}")
df = pd.read_excel(excel_path, sheet_name={sheet_index})
print(f"Loaded sheet with shape: {{df.shape}}")

# Store sheet metadata
sheet_rows, sheet_cols = df.shape
sheet_name = pd.ExcelFile(excel_path).sheet_names[{sheet_index}]

print(f"Sheet: {{sheet_name}}")
print(f"Dimensions: {{sheet_rows}} rows × {{sheet_cols}} columns")
"""
            result = await session.execute(load_code)
            if result.is_err():
                return err(f"Failed to load Excel: {result.unwrap_err()}")

            # Analyze sheet structure
            analysis_code = """
# Comprehensive sheet structure analysis for layout comprehension

print("=== ZONE DETECTION ANALYSIS ===\\n")

# 1. Identify potential header zones
print("1. HEADER ZONE ANALYSIS:")
# Check first few rows for header patterns
for i in range(min(5, len(df))):
    row_data = df.iloc[i]
    non_null = row_data.notna().sum()
    is_text = sum(isinstance(v, str) for v in row_data if pd.notna(v))
    print(f"  Row {i}: {non_null}/{len(row_data)} non-null, {is_text} text values")
    if i < 3:  # Show first 3 rows content
        print(f"    Content: {list(row_data.dropna().head(5))[:50]}")

# 2. Identify data zones by density
print("\\n2. DATA ZONE DENSITY ANALYSIS:")
# Calculate row density (non-null percentage)
row_density = df.notna().sum(axis=1) / len(df.columns)
high_density_rows = row_density[row_density > 0.5].index.tolist()
print(f"  High density rows (>50% filled): {len(high_density_rows)} rows")
if high_density_rows:
    print(f"  Range: rows {min(high_density_rows)} to {max(high_density_rows)}")

# 3. Identify empty navigation zones
print("\\n3. NAVIGATION ZONE DETECTION (Empty Separators):")
empty_rows = df.isnull().all(axis=1)
empty_row_indices = empty_rows[empty_rows].index.tolist()
if empty_row_indices:
    print(f"  Empty rows found at: {empty_row_indices[:20]}")
else:
    print("  No completely empty rows")

empty_cols = df.isnull().all(axis=0)
empty_col_indices = [i for i, col in enumerate(df.columns) if empty_cols[col]]
if empty_col_indices:
    print(f"  Empty columns at indices: {empty_col_indices}")
else:
    print("  No completely empty columns")

# 4. Detect potential summary zones (bottom rows)
print("\\n4. SUMMARY ZONE DETECTION:")
if len(df) > 10:
    last_rows = df.tail(10)
    for i in range(len(last_rows)):
        row_idx = len(df) - len(last_rows) + i
        row_data = last_rows.iloc[i]
        # Check for keywords that indicate summaries
        row_str = ' '.join(str(v) for v in row_data if pd.notna(v))
        if any(keyword in row_str.lower() for keyword in ['total', 'sum', 'average', 'mean', 'count']):
            print(f"  Row {row_idx}: Potential summary - contains '{row_str[:100]}'")

# 5. Detect formula patterns (if values look calculated)
print("\\n5. FORMULA ZONE PATTERNS:")
# Check for columns with all numeric values that might be calculated
for col_idx, col in enumerate(df.columns):
    if df[col].dtype in ['float64', 'int64']:
        # Check if values look like they could be formulas (e.g., sums of other columns)
        non_null_count = df[col].notna().sum()
        if non_null_count > 0:
            unique_ratio = df[col].nunique() / non_null_count
            if unique_ratio > 0.9:  # High uniqueness might indicate calculated values
                print(f"  Column {col_idx} ('{col}'): Possible formula column (high uniqueness)")

# 6. Side-by-side region detection
print("\\n6. SIDE-BY-SIDE REGION ANALYSIS:")
# Check if there are distinct regions separated by empty columns
if empty_col_indices:
    regions = []
    start_col = 0
    for empty_col in empty_col_indices + [len(df.columns)]:
        if empty_col > start_col:
            region_df = df.iloc[:, start_col:empty_col]
            non_empty_pct = region_df.notna().sum().sum() / (region_df.shape[0] * region_df.shape[1])
            if non_empty_pct > 0.1:  # At least 10% filled
                regions.append((start_col, empty_col - 1, non_empty_pct))
        start_col = empty_col + 1

    if len(regions) > 1:
        print(f"  Found {len(regions)} potential side-by-side regions:")
        for i, (start, end, density) in enumerate(regions):
            print(f"    Region {i + 1}: Columns {start}-{end}, density: {density:.2%}")

# 7. Metadata detection (top/corner cells)
print("\\n7. METADATA ZONE DETECTION:")
# Check corners and top rows for metadata patterns
corner_cells = []
# Top-left corner
if pd.notna(df.iloc[0, 0]):
    val = str(df.iloc[0, 0])
    if any(pattern in val.lower() for pattern in ['report', 'date', 'generated', 'department', 'company']):
        print(f"  Top-left metadata: '{val}'")

# Store analysis results for LLM
analysis_complete = True
"""

            analysis_result = await session.execute(analysis_code)
            if analysis_result.is_err():
                logger.warning(f"Analysis code failed: {analysis_result.unwrap_err()}")
                analysis_output = "Analysis failed"
            else:
                analysis_output = (
                    analysis_result.unwrap().outputs[-1].content if analysis_result.unwrap().outputs else "No output"
                )

            # Create LLM instance
            llm_result = create_llm_instance(config.model, config.api_key)
            if llm_result.is_err():
                return err(f"Failed to create LLM: {llm_result.unwrap_err()}")

            llm = llm_result.unwrap()

            # Register session for tools
            session_manager = get_session_manager()
            session_manager._sessions["layout_session"] = session

            # Create layout comprehension prompt
            system_prompt = f"""You are a spreadsheet layout comprehension expert. Your task is to analyze the structure
of a spreadsheet and identify semantic zones that will help guide analysis.

{ZONE_TYPE_DEFINITIONS}

## Your Task

Based on the analysis results provided, identify all semantic zones in the spreadsheet and create a
comprehensive layout description. Focus on:

1. Identifying the type of each zone (using the 8 zone types defined above)
2. Determining the semantic role/purpose of each zone
3. Finding relationships between zones (e.g., headers linked to data)
4. Providing navigation hints for analysis
5. Using "OTHER" for ambiguous regions rather than forcing incorrect classification

Create a Python variable called 'detected_layout' with your findings.

## Output Format

You must create a dictionary called 'detected_layout' with this structure:

```python
detected_layout = {{
    "layout_type": "financial_report",  # or "inventory", "dashboard", etc.
    "total_rows": sheet_rows,  # Use the actual sheet_rows variable
    "total_cols": sheet_cols,  # Use the actual sheet_cols variable
    "regions": [
        {{
            "region_id": "header_main",
            "zone_type": "header",  # One of: header, data, formula, summary, metadata, navigation, annotation, other
            "semantic_role": "transaction_headers",
            "coordinates": {{
                "start_row": 0,
                "end_row": 0,
                "start_col": 0,
                "end_col": 5
            }},
            "relationships": ["data_transactions"],  # IDs of related regions
            "navigation_hints": ["Column headers for main transaction data"],
            "confidence": 0.95,
            "description": "Main column headers",
            "sample_values": ["Date", "Description", "Amount"]
        }},
        # ... more regions
    ],
    "navigation_guide": {{
        "suggested_flow": ["header_main", "data_transactions", "summary_totals"],
        "key_insights": ["Sheet has embedded totals in header row"],
        "analysis_recommendations": ["Focus on transaction patterns", "Check formula dependencies"],
        "warnings": ["Complex layout with side-by-side tables"]
    }}
}}
```

Remember:
- Use confidence scores to indicate certainty (1.0 = very certain, < 0.7 = uncertain)
- Use "other" zone_type when a region doesn't clearly fit defined categories
- Provide helpful navigation_hints for each region
- Include relationships between regions to show connections
"""

            # Create tool for code execution
            @tool
            async def execute_layout_code(code: str) -> str:
                """Execute Python code in the layout detection session."""
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
            if "gemini" in config.model.lower():
                # Special handling for Gemini
                try:
                    llm_with_tools = llm.bind(functions=[execute_layout_code])
                except Exception:
                    llm_with_tools = llm.bind_tools([execute_layout_code])

                # Add Gemini-specific instructions
                gemini_instructions = SystemMessage(
                    content="""CRITICAL: To execute Python code, use the execute_layout_code tool.

Example usage:
execute_layout_code(code=\"\"\"
detected_layout = {
    "layout_type": "transaction_log",
    "total_rows": 100,
    "total_cols": 10,
    "regions": [...],
    "navigation_guide": {...}
}
print("Layout detection complete")
\"\"\")

NEVER call DataFrame methods directly. ALWAYS use the tool."""
                )

                messages = [
                    SystemMessage(content=system_prompt),
                    gemini_instructions,
                    HumanMessage(
                        content=f"""Analyze this spreadsheet structure and create the detected_layout variable:

ANALYSIS RESULTS:
{analysis_output}

Sheet: {excel_path.name} - Sheet {sheet_index}

IMPORTANT: You MUST use the execute_layout_code tool to create the detected_layout dictionary.

Example of what you need to do:
execute_layout_code(code=\"\"\"
# Create the layout detection result
detected_layout = {{
    "layout_type": "transaction_log",
    "total_rows": sheet_rows,
    "total_cols": sheet_cols,
    "regions": [...],
    "navigation_guide": {{...}}
}}
print("Created detected_layout with", len(detected_layout["regions"]), "regions")
\"\"\")

Now create the actual detected_layout for this sheet."""
                    ),
                ]
            else:
                llm_with_tools = llm.bind_tools([execute_layout_code])
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=f"""Analyze this spreadsheet structure and create the detected_layout variable:

ANALYSIS RESULTS:
{analysis_output}

Sheet: {excel_path.name} - Sheet {sheet_index}

IMPORTANT: You MUST use the execute_layout_code tool to create the detected_layout dictionary.

Example of what you need to do:
execute_layout_code(code=\"\"\"
# Create the layout detection result
detected_layout = {{
    "layout_type": "transaction_log",
    "total_rows": sheet_rows,
    "total_cols": sheet_cols,
    "regions": [...],
    "navigation_guide": {{...}}
}}
print("Created detected_layout with", len(detected_layout["regions"]), "regions")
\"\"\")

Now create the actual detected_layout for this sheet."""
                    ),
                ]

            # Track costs if enabled
            if config.track_costs:
                session_id = f"layout_{generate_session_id(file_config)}"
                with phoenix_session(session_id=session_id, user_id=None):
                    add_session_metadata(
                        session_id,
                        {
                            "excel_file": excel_path.name,
                            "sheet_index": sheet_index,
                            "model": config.model,
                            "agent_type": "LAYOUT_DETECTOR",
                        },
                    )

                    # Run detection rounds
                    for round_num in range(config.max_rounds):
                        if config.verbose:
                            logger.info(f"Layout detection round {round_num + 1}/{config.max_rounds}")

                        try:
                            response = await llm_with_tools.ainvoke(messages)
                            messages.append(response)

                            # Track usage
                            if config.track_costs:
                                await track_llm_usage(response, config.model)

                            # Process tool calls
                            if hasattr(response, "tool_calls") and response.tool_calls:
                                for tool_call in response.tool_calls:
                                    if tool_call["name"] == "execute_layout_code":
                                        code = tool_call["args"].get("code", "")
                                        tool_result = await execute_layout_code.ainvoke({"code": code})
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
                            logger.exception("Error in layout LLM call")
                            break

                    # Log costs
                    if config.track_costs:
                        cost_tracker = get_cost_tracker()
                        cost_summary = cost_tracker.get_summary()
                        logger.info(f"Layout detection cost: ${cost_summary['total_cost_usd']:.4f}")

            # Extract detected layout
            extract_code = """
# Extract the detected layout
print("Checking for detected_layout variable...")
print(f"Global variables: {list(globals().keys())[-10:]}")  # Show last 10 globals

if 'detected_layout' in globals():
    layout_dict = detected_layout
    print(f"✅ Layout detected with {len(layout_dict.get('regions', []))} regions")
    layout_dict
else:
    print("❌ No detected_layout variable found")
    # Try to find any layout-related variables
    for var_name in globals():
        if 'layout' in var_name.lower() or 'detect' in var_name.lower():
            print(f"  Found related variable: {var_name}")
    None
"""
            extract_result = await session.execute(extract_code)

            if extract_result.is_err():
                return err(f"Failed to extract layout: {extract_result.unwrap_err()}")

            # Parse the layout dictionary
            exec_output = extract_result.unwrap()
            layout_dict = None

            if hasattr(exec_output, "outputs") and exec_output.outputs:
                try:
                    import ast

                    last_output = exec_output.outputs[-1].content
                    layout_dict = ast.literal_eval(last_output)
                except (ValueError, SyntaxError):
                    pass

            if not layout_dict:
                return err("Failed to detect layout - no valid layout structure created")

            # Convert dictionary to SpreadsheetLayout model
            regions = []
            for region_data in layout_dict.get("regions", []):
                coords = region_data.get("coordinates", {})

                # Map zone type string to enum
                zone_type_str = region_data.get("zone_type", "other").lower()
                zone_type = ZoneType.OTHER  # Default
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
                    metadata=region_data.get("metadata", {}),
                    description=region_data.get("description"),
                    sample_values=region_data.get("sample_values"),
                )
                regions.append(region)

            # Create navigation guide
            nav_guide_data = layout_dict.get("navigation_guide", {})
            navigation_guide = NavigationGuide(
                suggested_flow=nav_guide_data.get("suggested_flow", []),
                key_insights=nav_guide_data.get("key_insights", []),
                analysis_recommendations=nav_guide_data.get("analysis_recommendations", []),
                warnings=nav_guide_data.get("warnings", []),
            )

            # Create final layout
            layout = SpreadsheetLayout(
                layout_type=layout_dict.get("layout_type", "unknown"),
                regions=regions,
                navigation_guide=navigation_guide,
                total_rows=layout_dict.get("total_rows", 0),
                total_cols=layout_dict.get("total_cols", 0),
                complexity_score=layout_dict.get("complexity_score"),
                has_multiple_tables=len([r for r in regions if r.zone_type == ZoneType.DATA]) > 1,
                has_formulas=any(r.zone_type == ZoneType.FORMULA for r in regions),
                has_summaries=any(r.zone_type == ZoneType.SUMMARY for r in regions),
            )

            # Save detection summary to notebook
            summary_md = f"""# Layout Comprehension Results

{layout.to_detection_summary()}
"""

            # Add summary to notebook
            toolkit = session.toolkit
            if hasattr(toolkit, "_notebook_builder"):
                toolkit._notebook_builder.add_markdown_cell(summary_md)

            # Save notebook
            session.toolkit.save_notebook()
            logger.info(f"Layout notebook saved to: {notebook_path}")

            # Log results for benchmarking
            if config.track_costs:
                try:
                    benchmark_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "model": config.model,
                        "excel_file": excel_path.name,
                        "sheet_index": sheet_index,
                        "prompt_hash": prompt_hash or "unknown",
                        "layout_type": layout.layout_type,
                        "regions_detected": len(layout.regions),
                        "zone_types": {zt.value: len(layout.get_regions_by_type(zt)) for zt in ZoneType},
                        "has_navigation_guide": bool(layout.navigation_guide.suggested_flow),
                        "complexity_score": layout.complexity_score,
                        "cost_usd": cost_summary["total_cost_usd"],
                        "notebook_path": str(notebook_path),
                    }

                    # Save to results file
                    results_file = config.output_dir / "benchmarks" / "layout_detection_results.json"
                    results_file.parent.mkdir(parents=True, exist_ok=True)

                    if results_file.exists():
                        with open(results_file) as f:
                            results = json.load(f)
                    else:
                        results = []

                    results.append(benchmark_entry)

                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=2)

                    logger.info(f"Layout results logged to {results_file}")

                except Exception as e:
                    logger.warning(f"Failed to log layout results: {e}")

            return ok(layout)

    except Exception as e:
        logger.exception("Error in layout comprehension")
        return err(f"Layout detection error: {e!s}")


# Example usage
if __name__ == "__main__":

    async def main():
        config = LayoutComprehensionConfig(
            model="gpt-4.1-nano",
            max_rounds=3,
            verbose=True,
        )

        result = await detect_layout_zones(
            Path("test_assets/collection/business-accounting/Business Accounting.xlsx"),
            sheet_index=0,
            config=config,
        )

        if result.is_ok():
            layout = result.unwrap()
            print("\n" + "=" * 50)
            print("LAYOUT COMPREHENSION SUCCESS")
            print("=" * 50)
            print(layout.to_detection_summary())
        else:
            print(f"Error: {result.unwrap_err()}")

    asyncio.run(main())
