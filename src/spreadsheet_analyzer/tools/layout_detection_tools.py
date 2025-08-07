"""Tools for semantic layout detection and pattern recognition in spreadsheets."""

import pandas as pd
from langchain_core.tools import tool


@tool
async def analyze_zone_patterns(
    df: pd.DataFrame,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
) -> dict[str, any]:
    """Analyze patterns within a specific zone to determine its type.

    Args:
        df: The DataFrame to analyze
        start_row: Starting row index (inclusive)
        end_row: Ending row index (inclusive)
        start_col: Starting column index (inclusive)
        end_col: Ending column index (inclusive)

    Returns:
        Dictionary with pattern analysis results
    """
    # Extract the zone
    zone = df.iloc[start_row : end_row + 1, start_col : end_col + 1]

    analysis = {
        "dimensions": f"{zone.shape[0]}x{zone.shape[1]}",
        "density": zone.notna().sum().sum() / (zone.shape[0] * zone.shape[1]),
        "text_ratio": 0,
        "numeric_ratio": 0,
        "has_formulas": False,
        "has_totals_keywords": False,
        "is_mostly_empty": False,
        "unique_values_ratio": 0,
        "likely_zone_type": "other",
    }

    # Calculate content types
    total_non_null = zone.notna().sum().sum()
    if total_non_null > 0:
        text_count = 0
        numeric_count = 0

        for col in zone.columns:
            for val in zone[col].dropna():
                if isinstance(val, str):
                    text_count += 1
                    # Check for formula indicators
                    if val.startswith("="):
                        analysis["has_formulas"] = True
                    # Check for summary keywords
                    if any(kw in val.lower() for kw in ["total", "sum", "average", "count"]):
                        analysis["has_totals_keywords"] = True
                elif isinstance(val, (int, float)):
                    numeric_count += 1

        analysis["text_ratio"] = text_count / total_non_null
        analysis["numeric_ratio"] = numeric_count / total_non_null

        # Calculate uniqueness
        all_values = zone.values.flatten()
        non_null_values = [v for v in all_values if pd.notna(v)]
        if non_null_values:
            analysis["unique_values_ratio"] = len(set(map(str, non_null_values))) / len(non_null_values)

    # Determine if mostly empty
    analysis["is_mostly_empty"] = analysis["density"] < 0.2

    # Suggest zone type based on patterns
    if analysis["is_mostly_empty"]:
        analysis["likely_zone_type"] = "navigation"
    elif start_row <= 2 and analysis["text_ratio"] > 0.7:
        analysis["likely_zone_type"] = "header"
    elif analysis["has_totals_keywords"]:
        analysis["likely_zone_type"] = "summary"
    elif analysis["has_formulas"]:
        analysis["likely_zone_type"] = "formula"
    elif analysis["numeric_ratio"] > 0.5 and zone.shape[0] > 5:
        analysis["likely_zone_type"] = "data"
    elif analysis["text_ratio"] > 0.8 and zone.shape[0] <= 3:
        analysis["likely_zone_type"] = "metadata"
    else:
        analysis["likely_zone_type"] = "other"

    return analysis


@tool
async def detect_region_relationships(
    df: pd.DataFrame,
    regions: list[dict[str, any]],
) -> list[tuple[str, str, str]]:
    """Detect relationships between identified regions.

    Args:
        df: The DataFrame being analyzed
        regions: List of region dictionaries with coordinates and IDs

    Returns:
        List of relationships as (region1_id, region2_id, relationship_type) tuples
    """
    relationships = []

    for i, region1 in enumerate(regions):
        for j, region2 in enumerate(regions):
            if i >= j:
                continue

            r1_id = region1.get("region_id")
            r2_id = region2.get("region_id")

            # Check for vertical relationship (header above data)
            if (
                region1["end_row"] < region2["start_row"]
                and region1["start_col"] == region2["start_col"]
                and region1["end_col"] == region2["end_col"]
            ):
                if region1.get("zone_type") == "header" and region2.get("zone_type") == "data":
                    relationships.append((r1_id, r2_id, "header_for_data"))

            # Check for horizontal relationship (side-by-side)
            if region1["end_col"] < region2["start_col"] and abs(region1["start_row"] - region2["start_row"]) <= 2:
                relationships.append((r1_id, r2_id, "side_by_side"))

            # Check for summary relationship (summary below data)
            if (
                region1["end_row"] < region2["start_row"]
                and region1.get("zone_type") == "data"
                and region2.get("zone_type") == "summary"
            ):
                relationships.append((r1_id, r2_id, "data_with_summary"))

            # Check for formula dependency
            if region1.get("zone_type") == "data" and region2.get("zone_type") == "formula":
                # Check if formula region is to the right of data
                if region1["end_col"] < region2["start_col"]:
                    relationships.append((r1_id, r2_id, "data_feeds_formula"))

    return relationships


@tool
async def find_natural_boundaries(df: pd.DataFrame) -> dict[str, list[int]]:
    """Find natural boundaries in the spreadsheet based on empty rows/columns.

    Args:
        df: The DataFrame to analyze

    Returns:
        Dictionary with lists of empty row and column indices
    """
    boundaries = {
        "empty_rows": [],
        "empty_columns": [],
        "sparse_rows": [],  # Rows with < 20% data
        "sparse_columns": [],  # Columns with < 20% data
        "potential_separators": [],
    }

    # Find completely empty rows
    empty_rows = df.isnull().all(axis=1)
    boundaries["empty_rows"] = empty_rows[empty_rows].index.tolist()

    # Find completely empty columns
    empty_cols = df.isnull().all(axis=0)
    boundaries["empty_columns"] = [i for i, col in enumerate(df.columns) if empty_cols[col]]

    # Find sparse rows (< 20% filled)
    row_density = df.notna().sum(axis=1) / len(df.columns)
    boundaries["sparse_rows"] = row_density[row_density < 0.2].index.tolist()

    # Find sparse columns (< 20% filled)
    col_density = df.notna().sum(axis=0) / len(df)
    sparse_cols = col_density[col_density < 0.2]
    boundaries["sparse_columns"] = [i for i, col in enumerate(df.columns) if col in sparse_cols.index]

    # Identify potential separator patterns
    # Look for consecutive empty or sparse rows/columns
    if len(boundaries["empty_rows"]) > 0:
        prev_row = boundaries["empty_rows"][0]
        separator_group = [prev_row]
        for row in boundaries["empty_rows"][1:]:
            if row == prev_row + 1:
                separator_group.append(row)
            else:
                if len(separator_group) >= 2:
                    boundaries["potential_separators"].append(
                        {
                            "type": "row_separator",
                            "indices": separator_group,
                        }
                    )
                separator_group = [row]
            prev_row = row

        if len(separator_group) >= 2:
            boundaries["potential_separators"].append(
                {
                    "type": "row_separator",
                    "indices": separator_group,
                }
            )

    return boundaries


@tool
async def suggest_navigation_flow(
    regions: list[dict[str, any]],
    relationships: list[tuple[str, str, str]],
) -> dict[str, any]:
    """Suggest a navigation flow through the identified regions.

    Args:
        regions: List of region dictionaries
        relationships: List of relationship tuples

    Returns:
        Navigation flow suggestions
    """
    flow = {
        "suggested_order": [],
        "entry_points": [],
        "key_regions": [],
        "analysis_tips": [],
    }

    # Find entry points (headers and metadata)
    for region in regions:
        if region.get("zone_type") in ["header", "metadata"]:
            flow["entry_points"].append(region["region_id"])

    # Build suggested order based on typical analysis flow
    zone_priority = {
        "metadata": 1,
        "header": 2,
        "data": 3,
        "formula": 4,
        "summary": 5,
        "annotation": 6,
        "navigation": 7,
        "other": 8,
    }

    sorted_regions = sorted(
        regions,
        key=lambda r: (
            zone_priority.get(r.get("zone_type", "other"), 9),
            r.get("start_row", 0),
            r.get("start_col", 0),
        ),
    )

    flow["suggested_order"] = [r["region_id"] for r in sorted_regions]

    # Identify key regions (data and summary zones)
    for region in regions:
        if region.get("zone_type") in ["data", "summary"]:
            flow["key_regions"].append(region["region_id"])

    # Generate analysis tips based on patterns
    if any(r[2] == "side_by_side" for r in relationships):
        flow["analysis_tips"].append("Compare side-by-side regions for relationships")

    if any(r.get("zone_type") == "formula" for r in regions):
        flow["analysis_tips"].append("Trace formula dependencies to understand calculations")

    if any(r.get("zone_type") == "summary" for r in regions):
        flow["analysis_tips"].append("Validate summary totals against detail data")

    # Check for complex layouts
    if len(regions) > 5:
        flow["analysis_tips"].append("Complex layout - process regions systematically")

    return flow


@tool
async def classify_layout_pattern(
    regions: list[dict[str, any]],
    df_shape: tuple[int, int],
) -> dict[str, any]:
    """Classify the overall layout pattern of the spreadsheet.

    Args:
        regions: List of detected regions
        df_shape: Shape of the DataFrame (rows, cols)

    Returns:
        Layout classification and characteristics
    """
    classification = {
        "layout_type": "unknown",
        "complexity": "simple",
        "characteristics": [],
        "recommended_approach": "",
    }

    # Count zone types
    zone_counts = {}
    for region in regions:
        zone_type = region.get("zone_type", "other")
        zone_counts[zone_type] = zone_counts.get(zone_type, 0) + 1

    # Determine layout type based on patterns
    data_regions = zone_counts.get("data", 0)
    summary_regions = zone_counts.get("summary", 0)
    formula_regions = zone_counts.get("formula", 0)

    if data_regions == 1 and summary_regions == 0:
        classification["layout_type"] = "simple_table"
        classification["characteristics"].append("Single data table")
    elif data_regions == 1 and summary_regions >= 1:
        classification["layout_type"] = "table_with_summary"
        classification["characteristics"].append("Data table with summaries")
    elif data_regions > 1:
        classification["layout_type"] = "multi_table"
        classification["characteristics"].append("Multiple data regions")
    elif formula_regions > 2:
        classification["layout_type"] = "calculation_model"
        classification["characteristics"].append("Formula-heavy calculation model")
    elif zone_counts.get("metadata", 0) > 2:
        classification["layout_type"] = "report"
        classification["characteristics"].append("Report with metadata")
    else:
        classification["layout_type"] = "mixed"
        classification["characteristics"].append("Mixed content layout")

    # Determine complexity
    total_regions = len(regions)
    if total_regions <= 3:
        classification["complexity"] = "simple"
    elif total_regions <= 6:
        classification["complexity"] = "moderate"
    else:
        classification["complexity"] = "complex"

    # Add characteristics based on features
    if zone_counts.get("navigation", 0) > 0:
        classification["characteristics"].append("Visual separators present")

    if any(r.get("confidence", 1.0) < 0.7 for r in regions):
        classification["characteristics"].append("Contains ambiguous regions")

    # Recommend analysis approach
    if classification["layout_type"] == "simple_table":
        classification["recommended_approach"] = "Direct analysis of single table"
    elif classification["layout_type"] == "multi_table":
        classification["recommended_approach"] = "Analyze each table separately, then relationships"
    elif classification["layout_type"] == "calculation_model":
        classification["recommended_approach"] = "Trace formulas and dependencies first"
    else:
        classification["recommended_approach"] = "Start with metadata, then systematic region analysis"

    return classification


@tool
async def detect_embedded_structures(
    df: pd.DataFrame,
    region: dict[str, any],
) -> list[dict[str, any]]:
    """Detect embedded structures within a region (e.g., totals within headers).

    Args:
        df: The DataFrame to analyze
        region: Region dictionary with coordinates

    Returns:
        List of embedded structures found
    """
    embedded = []

    # Extract the region
    zone = df.iloc[
        region["start_row"] : region["end_row"] + 1,
        region["start_col"] : region["end_col"] + 1,
    ]

    # Check for embedded totals in headers
    if region.get("zone_type") == "header":
        for i, row in enumerate(zone.itertuples(index=False)):
            for j, val in enumerate(row):
                if pd.notna(val) and isinstance(val, str):
                    if any(kw in str(val).lower() for kw in ["total", "sum", "revenue", "expense"]):
                        embedded.append(
                            {
                                "type": "embedded_summary",
                                "location": f"Row {region['start_row'] + i}, Col {region['start_col'] + j}",
                                "value": str(val),
                                "context": "Summary value embedded in header zone",
                            }
                        )

    # Check for mixed content in data zones
    if region.get("zone_type") == "data":
        # Look for sudden changes in pattern
        for i in range(len(zone) - 1):
            row1_nulls = zone.iloc[i].isnull().sum()
            row2_nulls = zone.iloc[i + 1].isnull().sum()

            # Significant change in null pattern might indicate embedded structure
            if abs(row1_nulls - row2_nulls) > len(zone.columns) * 0.5:
                embedded.append(
                    {
                        "type": "pattern_break",
                        "location": f"Between rows {region['start_row'] + i} and {region['start_row'] + i + 1}",
                        "context": "Potential embedded separator or sub-structure",
                    }
                )

    return embedded
