"""Table detection agent for identifying table boundaries in spreadsheets.

This module provides a specialized agent that detects table boundaries
using both mechanical (empty rows/columns) and semantic (entity type changes)
analysis.

CLAUDE-KNOWLEDGE: Excel sheets often contain multiple logical tables that
need to be analyzed separately. Accurate detection improves analysis quality.
"""

import re
from typing import Any

import pandas as pd

from ..core.errors import AgentError
from ..core.types import Result, err, ok
from .core import FunctionalAgent, create_simple_agent
from .table_detection_types import (
    EMPTY_ROW_THRESHOLD,
    HEADER_PATTERN_CONFIDENCE,
    MIN_TABLE_ROWS,
    SEMANTIC_CONFIDENCE_THRESHOLD,
    DetectionMetrics,
    TableBoundary,
    TableDetectionResult,
    TableType,
)
from .types import AgentCapability, AgentMessage, AgentState


def create_table_detector() -> FunctionalAgent:
    """Create a table detection agent following functional pattern.

    Returns:
        FunctionalAgent configured for table boundary detection
    """

    def detect_tables(message: AgentMessage, state: AgentState) -> Result[AgentMessage, AgentError]:
        """Pure function for table detection.

        Args:
            message: Input message containing dataframe and metadata
            state: Current agent state

        Returns:
            Result containing detection results or error
        """
        try:
            excel_data = message.content
            df = excel_data.get("dataframe")
            sheet_name = excel_data.get("sheet_name", "Sheet")

            if df is None:
                return err(AgentError("No dataframe provided in message content"))

            # Mechanical detection (empty rows/columns)
            mechanical_boundaries = _detect_mechanical_boundaries(df)

            # Semantic detection (entity type changes)
            semantic_boundaries = _detect_semantic_boundaries(df)

            # Header pattern detection
            header_boundaries = _detect_header_patterns(df)

            # Merge and validate boundaries
            final_tables = _merge_boundaries(mechanical_boundaries, semantic_boundaries, header_boundaries, df)

            # Determine primary detection method
            detection_method = _determine_detection_method(mechanical_boundaries, semantic_boundaries, final_tables)

            # Create metrics
            metrics = DetectionMetrics(
                empty_row_blocks=_find_empty_row_blocks(df),
                empty_col_blocks=_find_empty_col_blocks(df),
                header_rows_detected=tuple(h["row"] for h in header_boundaries),
                semantic_shifts=tuple(s["row"] for s in semantic_boundaries),
                confidence_scores={
                    "mechanical": len(mechanical_boundaries) * 0.3,
                    "semantic": len(semantic_boundaries) * 0.3,
                    "header": len(header_boundaries) * 0.3,
                },
            )

            result = TableDetectionResult(
                sheet_name=sheet_name,
                tables=tuple(final_tables),
                metadata={
                    "total_rows": len(df),
                    "total_cols": len(df.columns),
                    "detection_confidence": sum(t.confidence for t in final_tables) / len(final_tables)
                    if final_tables
                    else 0,
                },
                metrics=metrics,
            )

            response = AgentMessage.create(
                sender=state.agent_id,
                receiver=message.sender,
                content=result,
                reply_to=message.id,
                correlation_id=message.correlation_id,
            )

            return ok(response)

        except Exception as e:
            return err(AgentError(f"Table detection failed: {e!s}"))

    capabilities = [
        AgentCapability(
            name="detect_tables",
            description="Detect table boundaries in spreadsheet using mechanical and semantic analysis",
            input_type=dict,
            output_type=TableDetectionResult,
        )
    ]

    return create_simple_agent("table_detector", capabilities, detect_tables)


def _detect_mechanical_boundaries(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect table boundaries based on empty rows and columns.

    Args:
        df: DataFrame to analyze

    Returns:
        List of boundary dictionaries with mechanical detection info
    """
    boundaries = []

    # Find empty rows
    empty_rows = df.isnull().all(axis=1)
    empty_row_indices = empty_rows[empty_rows].index.tolist()

    # Group consecutive empty rows
    if empty_row_indices:
        groups = []
        current_group = [empty_row_indices[0]]

        for i in range(1, len(empty_row_indices)):
            if empty_row_indices[i] - empty_row_indices[i - 1] == 1:
                current_group.append(empty_row_indices[i])
            else:
                groups.append(current_group)
                current_group = [empty_row_indices[i]]
        groups.append(current_group)

        # Find boundaries based on empty row groups
        current_start = 0
        for group in groups:
            if len(group) >= EMPTY_ROW_THRESHOLD:
                # Table ends before this empty group
                if group[0] - current_start >= MIN_TABLE_ROWS:
                    boundaries.append(
                        {"start_row": current_start, "end_row": group[0] - 1, "reason": "empty_rows", "confidence": 0.9}
                    )
                current_start = group[-1] + 1

        # Handle last table
        if len(df) - current_start >= MIN_TABLE_ROWS:
            boundaries.append(
                {"start_row": current_start, "end_row": len(df) - 1, "reason": "empty_rows", "confidence": 0.9}
            )
    else:
        # No empty rows found
        if len(df) >= MIN_TABLE_ROWS:
            boundaries.append({"start_row": 0, "end_row": len(df) - 1, "reason": "no_separators", "confidence": 0.7})

    return boundaries


def _detect_semantic_boundaries(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect table boundaries based on semantic content changes.

    Args:
        df: DataFrame to analyze

    Returns:
        List of boundary dictionaries with semantic detection info
    """
    boundaries = []

    # Look for major semantic shifts by analyzing ID patterns and column content
    window_size = 5  # Check patterns over windows

    for i in range(window_size, len(df) - window_size):
        # Get windows before and after current position
        before_window = df.iloc[i - window_size : i]
        after_window = df.iloc[i : i + window_size]

        # Check first column for ID pattern changes
        if len(df.columns) > 0:
            before_patterns = []
            after_patterns = []

            # Analyze first column patterns
            for val in before_window.iloc[:, 0]:
                if pd.notna(val) and _is_code_pattern(str(val)):
                    # Extract prefix from code pattern
                    prefix = re.match(r"^([A-Z]+)", str(val))
                    if prefix:
                        before_patterns.append(prefix.group(1))

            for val in after_window.iloc[:, 0]:
                if pd.notna(val) and _is_code_pattern(str(val)):
                    # Extract prefix from code pattern
                    prefix = re.match(r"^([A-Z]+)", str(val))
                    if prefix:
                        after_patterns.append(prefix.group(1))

            # Check if patterns changed significantly
            if before_patterns and after_patterns:
                before_common = max(set(before_patterns), key=before_patterns.count) if before_patterns else None
                after_common = max(set(after_patterns), key=after_patterns.count) if after_patterns else None

                if before_common and after_common and before_common != after_common:
                    # Found a semantic boundary
                    boundaries.append(
                        {"row": i, "reason": "semantic_shift", "confidence": SEMANTIC_CONFIDENCE_THRESHOLD}
                    )

    return boundaries


def _detect_header_patterns(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Detect potential header rows based on content patterns.

    Args:
        df: DataFrame to analyze

    Returns:
        List of potential header rows
    """
    headers = []

    for idx, row in df.iterrows():
        # Count non-null text values
        text_count = sum(
            1 for val in row if pd.notna(val) and isinstance(val, str) and not _is_numeric_pattern(str(val))
        )

        # Header heuristics
        if text_count >= len(row) * 0.5 and idx < len(df) - 1:  # At least 50% text and not last row
            next_row = df.iloc[idx + 1]
            next_numeric_count = sum(1 for val in next_row if pd.notna(val) and _is_numeric_pattern(str(val)))

            if next_numeric_count > text_count:
                headers.append({"row": idx, "confidence": HEADER_PATTERN_CONFIDENCE})

    return headers


def _merge_boundaries(
    mechanical: list[dict], semantic: list[dict], headers: list[dict], df: pd.DataFrame
) -> list[TableBoundary]:
    """Merge different boundary detection results into final table list.

    Args:
        mechanical: Boundaries from empty row detection
        semantic: Boundaries from semantic shifts
        headers: Detected header rows
        df: Original dataframe

    Returns:
        List of final TableBoundary objects
    """
    final_tables = []
    table_counter = 1

    # Start with mechanical boundaries as base
    for mech in mechanical:
        start = mech["start_row"]
        end = mech["end_row"]

        # Refine with semantic boundaries
        for sem in semantic:
            sem_row = sem["row"]
            if start < sem_row < end and sem_row - start >= MIN_TABLE_ROWS:
                # Split table at semantic boundary
                table = _create_table_boundary(
                    df, start, sem_row - 1, table_counter, confidence=max(mech["confidence"], sem["confidence"])
                )
                final_tables.append(table)
                table_counter += 1
                start = sem_row

        # Add remaining part
        if end - start >= MIN_TABLE_ROWS:
            table = _create_table_boundary(df, start, end, table_counter, confidence=mech["confidence"])
            final_tables.append(table)
            table_counter += 1

    return final_tables


def _create_table_boundary(
    df: pd.DataFrame, start_row: int, end_row: int, table_num: int, confidence: float
) -> TableBoundary:
    """Create a TableBoundary object for detected table.

    Args:
        df: DataFrame containing the table
        start_row: Starting row index
        end_row: Ending row index
        table_num: Table number for ID
        confidence: Detection confidence

    Returns:
        TableBoundary object
    """
    # Find non-empty columns
    table_df = df.iloc[start_row : end_row + 1]
    non_empty_cols = table_df.notna().any(axis=0)

    # Get numeric indices for columns
    if non_empty_cols.any():
        # Find first and last non-empty columns using numpy for efficiency
        non_empty_indices = non_empty_cols.values.nonzero()[0]
        start_col = non_empty_indices[0]
        end_col = non_empty_indices[-1]
    else:
        start_col = 0
        end_col = len(df.columns) - 1

    # Determine table type and entity
    table_type, entity_type = _classify_table(table_df)

    # Generate description
    description = _generate_table_description(table_df, table_type, entity_type)

    return TableBoundary(
        table_id=f"table_{table_num}",
        description=description,
        start_row=int(start_row),
        end_row=int(end_row),
        start_col=int(start_col),
        end_col=int(end_col),
        confidence=confidence,
        table_type=table_type,
        entity_type=entity_type,
    )


def _classify_table(df: pd.DataFrame) -> tuple[TableType, str]:
    """Classify table type and entity type based on content.

    Args:
        df: Table dataframe

    Returns:
        Tuple of (TableType, entity_type_string)
    """
    # Simple heuristics for classification
    row_count = len(df)

    # Check for summary patterns
    if row_count < 10:
        # Look for aggregation keywords
        text_values = df.select_dtypes(include=["object"]).values.flatten()
        text_values = [str(v).lower() for v in text_values if pd.notna(v)]

        summary_keywords = ["total", "sum", "average", "count", "summary", "region", "category"]
        if any(keyword in " ".join(text_values) for keyword in summary_keywords):
            return TableType.SUMMARY, "summary"

    # Check for header/metadata pattern
    if row_count < 6 and len(df.columns) >= 2:
        # Look for key-value pattern
        first_col = df.iloc[:, 0]
        if all(pd.notna(v) and ":" in str(v) for v in first_col[:3]):
            return TableType.HEADER, "metadata"

    # Default to detail table
    # Try to identify entity type from content
    entity_type = _detect_entity_type(df)

    return TableType.DETAIL, entity_type


def _detect_entity_type(df: pd.DataFrame) -> str:
    """Detect the business entity type from table content.

    Args:
        df: Table dataframe

    Returns:
        Entity type string
    """
    # Check column names if available
    col_names = [str(col).lower() for col in df.columns]

    # Entity patterns
    patterns = {
        "orders": ["order", "customer", "amount", "quantity"],
        "products": ["product", "price", "stock", "sku"],
        "employees": ["employee", "name", "department", "salary"],
        "customers": ["customer", "name", "email", "phone"],
        "invoices": ["invoice", "date", "total", "payment"],
        "items": ["item", "description", "quantity", "price"],
    }

    for entity, keywords in patterns.items():
        matches = sum(1 for keyword in keywords if any(keyword in col for col in col_names))
        if matches >= 2:
            return entity

    # Check first column values (limit to first 5 rows to avoid headers)
    if len(df) > 0:
        max_rows = min(5, len(df))
        first_col_values = df.iloc[:max_rows, 0].astype(str)
        for val in first_col_values:
            val_lower = val.lower()
            for entity, keywords in patterns.items():
                if any(keyword in val_lower for keyword in keywords):
                    return entity

    return "data"


def _generate_table_description(df: pd.DataFrame, table_type: TableType, entity_type: str) -> str:
    """Generate human-readable description of table.

    Args:
        df: Table dataframe
        table_type: Classified table type
        entity_type: Detected entity type

    Returns:
        Description string
    """
    row_count = len(df)
    col_count = len(df.columns)

    if table_type == TableType.SUMMARY:
        return f"Summary table with {row_count} rows showing {entity_type} aggregations"
    elif table_type == TableType.HEADER:
        return f"Header section containing {entity_type} information"
    else:
        # Try to get more specific for detail tables
        if entity_type == "orders":
            return f"Customer orders table with {row_count} order records"
        elif entity_type == "products":
            return f"Product inventory table with {row_count} products"
        elif entity_type == "employees":
            return f"Employee list with {row_count} employee records"
        else:
            return f"{entity_type.capitalize()} table with {row_count} rows and {col_count} columns"


def _determine_detection_method(mechanical: list[dict], semantic: list[dict], final_tables: list[TableBoundary]) -> str:
    """Determine primary detection method used.

    Args:
        mechanical: Mechanical detection results
        semantic: Semantic detection results
        final_tables: Final merged tables

    Returns:
        Detection method string
    """
    if not final_tables:
        return "mechanical"

    mech_count = len(mechanical)
    sem_count = len(semantic)

    if mech_count > 0 and sem_count > 0:
        return "hybrid"
    elif mech_count > 0:
        return "mechanical"
    else:
        return "semantic"


def _find_empty_row_blocks(df: pd.DataFrame) -> tuple[tuple[int, int], ...]:
    """Find blocks of consecutive empty rows.

    Args:
        df: DataFrame to analyze

    Returns:
        Tuple of (start, end) row indices for empty blocks
    """
    empty_rows = df.isnull().all(axis=1)
    blocks = []

    in_block = False
    start = 0

    for i, is_empty in enumerate(empty_rows):
        if is_empty and not in_block:
            start = i
            in_block = True
        elif not is_empty and in_block:
            blocks.append((start, i - 1))
            in_block = False

    if in_block:
        blocks.append((start, len(df) - 1))

    return tuple(blocks)


def _find_empty_col_blocks(df: pd.DataFrame) -> tuple[tuple[int, int], ...]:
    """Find blocks of consecutive empty columns.

    Args:
        df: DataFrame to analyze

    Returns:
        Tuple of (start, end) column indices for empty blocks
    """
    empty_cols = df.isnull().all(axis=0)
    blocks = []

    in_block = False
    start = 0

    for i, is_empty in enumerate(empty_cols):
        if is_empty and not in_block:
            start = i
            in_block = True
        elif not is_empty and in_block:
            blocks.append((start, i - 1))
            in_block = False

    if in_block:
        blocks.append((start, len(df.columns) - 1))

    return tuple(blocks)


# Helper functions


def _is_numeric_pattern(value: str) -> bool:
    """Check if value matches numeric pattern."""
    try:
        float(value.replace(",", "").replace("$", "").replace("%", ""))
    except ValueError:
        return False
    else:
        return True


def _is_code_pattern(value: str) -> bool:
    """Check if value matches a code/ID pattern."""
    # Simple patterns for codes
    patterns = [
        r"^[A-Z]{2,}-\d+",  # XX-123
        r"^[A-Z]+\d{3,}",  # ABC123
        r"^\d{4,}$",  # 1234
    ]

    return any(re.match(pattern, value) for pattern in patterns)
