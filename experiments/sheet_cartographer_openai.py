#!/usr/bin/env python3
"""
Sheet-Cartographer Agent (OpenAI Implementation) - Version 0.9

An LLM-assisted agent that maps Excel worksheet topology without reading
the entire sheet at once. This implementation uses OpenAI's function calling
API to invoke tools for incremental sheet exploration.

Hypothesis:
Using a sliding window approach with selective cell probing can efficiently
map large spreadsheets while maintaining token efficiency (≤2000 tokens per
10000 cells).

Expected Outcomes:
- Block recall ≥ 95% of ground-truth blocks
- Type accuracy ≥ 90%
- Runtime ≤ 20s per 100K cells

Implementation Approach:
1. Orient - Get sheet dimensions and metadata
2. Sweep - Slide windows across sheet recording density
3. Cluster - Merge adjacent dense tiles into blocks
4. Probe - Classify blocks via selective sampling
5. Overlay - Add named ranges, charts, pivots
6. Emit - Return cartographer_map JSON

Provider: OpenAI
Models Supported: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
Author: Sheet-Cartographer v0.9 (OpenAI)
Date: 2025-08-07
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from scipy import stats

# Add parent directory for utils import
sys.path.append(str(Path(__file__).parent))
# Import Excel and OpenAI libraries
import openpyxl
from openai import OpenAI
from openpyxl.utils import column_index_from_string, get_column_letter
from openpyxl.utils.cell import coordinate_from_string
from openpyxl.worksheet.worksheet import Worksheet

# Import new semantic classification system
from semantic_classification import DataCharacteristics, StructuralType
from utils import ExperimentLogger

# Constants
DEFAULT_MODEL: Final[str] = "gpt-4o-mini"
MAX_WINDOW_HEIGHT: Final[int] = 50
MAX_WINDOW_WIDTH: Final[int] = 50
MAX_CELL_META_CALLS: Final[int] = 100
MAX_RANGE_AREA: Final[int] = 10000  # Increased for larger samples
MAX_LLM_TOKENS: Final[int] = 32000
MAX_RUNTIME_SECONDS: Final[int] = 30
DENSITY_THRESHOLD: Final[float] = 0.05  # 5% threshold to catch even sparse summary regions
HIGH_DENSITY_THRESHOLD: Final[float] = 0.8  # 80% density reduces window size
BLANK_TAIL_THRESHOLD: Final[float] = 0.3  # 30% blank tail triggers split, not penalty
HOMOGENEITY_THRESHOLD: Final[float] = 0.8  # 80% homogeneity required to classify (reduced for better accuracy)
ENTROPY_THRESHOLD: Final[float] = 0.1  # Max entropy for homogeneous region

# Block type confidence thresholds
FACT_TABLE_CONFIDENCE: Final[float] = 0.75
HEADER_BANNER_CONFIDENCE: Final[float] = 0.70
KPI_SUMMARY_CONFIDENCE: Final[float] = 0.65
PIVOT_CACHE_CONFIDENCE: Final[float] = 0.60
CHART_ANCHOR_CONFIDENCE: Final[float] = 1.00
SUMMARY_CONFIDENCE: Final[float] = 0.80  # For totals sections
UNKNOWN_CONFIDENCE: Final[float] = 0.20

# Type mapping from string to enum
TYPE_MAP: Final[dict[str, StructuralType]] = {
    "DataTable": StructuralType.DATA_TABLE,
    "HeaderZone": StructuralType.HEADER_ZONE,
    "AggregationZone": StructuralType.AGGREGATION_ZONE,
    "MetadataZone": StructuralType.METADATA_ZONE,
    "VisualizationAnchor": StructuralType.VISUALIZATION_ANCHOR,
    "FormInputZone": StructuralType.FORM_INPUT_ZONE,
    "NavigationZone": StructuralType.NAVIGATION_ZONE,
    "UnstructuredText": StructuralType.UNSTRUCTURED_TEXT,
    "EmptyPadding": StructuralType.EMPTY_PADDING,
}


@dataclass
class SheetInfo:
    """Sheet metadata returned by sheet-info tool."""

    rows: int
    cols: int
    used_range: str
    hidden_rows: list[int] = field(default_factory=list)
    hidden_cols: list[int] = field(default_factory=list)


@dataclass
class CellMeta:
    """Cell metadata returned by cell-meta tool."""

    value_type: Literal["num", "text", "date", "blank", "error"]
    is_formula: bool
    is_merged: bool
    style_flags: list[str] = field(default_factory=list)


@dataclass
class Block:
    """Detected block in the spreadsheet with semantic classification."""

    id: str
    range: str
    structural_type: StructuralType  # New enum-based type
    semantic_description: str  # Natural language description
    confidence: float
    classification_reasoning: str = ""  # Why this classification
    data_characteristics: DataCharacteristics = field(default_factory=DataCharacteristics)
    suggested_operations: list[str] = field(default_factory=list)
    named_range: str | None = None


@dataclass
class ChartObject:
    """Chart or pivot object in the spreadsheet."""

    id: str
    anchor: str
    type: str
    linked_block: str | None = None


class ExcelToolHandler:
    """Handles tool execution for Excel operations."""

    def __init__(self, worksheet: Worksheet, logger: ExperimentLogger, excel_path: Path):
        self.worksheet = worksheet
        self.logger = logger
        self.excel_path = excel_path
        self.tool_call_count = defaultdict(int)
        self.total_tokens_used = 0

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool and return its result."""
        self.tool_call_count[tool_name] += 1
        self.logger.main.debug(f"🔧 Executing tool: {tool_name} with args: {arguments}")

        if tool_name == "sheet_info":
            return self._sheet_info()
        elif tool_name == "window_grab":
            return self._window_grab(**arguments)
        elif tool_name == "range_peek":
            return self._range_peek(**arguments)
        elif tool_name == "cell_meta":
            return self._cell_meta(**arguments)
        elif tool_name == "merged_list":
            return self._merged_list()
        elif tool_name == "names_list":
            return self._names_list()
        elif tool_name == "object_scan":
            return self._object_scan(**arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _sheet_info(self) -> dict[str, Any]:
        """Get sheet dimensions and metadata."""
        min_row = self.worksheet.min_row or 1
        max_row = self.worksheet.max_row or 1
        min_col = self.worksheet.min_column or 1
        max_col = self.worksheet.max_column or 1

        used_range = f"{get_column_letter(min_col)}{min_row}:{get_column_letter(max_col)}{max_row}"

        # Excel doesn't track hidden rows/cols in openpyxl easily, return empty for now
        return {"rows": max_row, "cols": max_col, "usedRange": used_range, "hiddenRows": [], "hiddenCols": []}

    def _safe_str(self, value: Any, max_chars: int = 25) -> str:
        """Smart truncation that preserves semantic information."""
        if value is None:
            return ""
        s = str(value)
        if len(s) <= max_chars:
            return s
        if len(s) <= 80:
            return s[:max_chars] + "↯"
        # For very long strings, keep beginning and end
        return f"{s[:max_chars]}↯{s[-10:]}"

    def _window_grab(self, top: int, left: int, height: int, width: int, format: str = "markdown") -> str:
        """Grab a window of cells from the sheet with smart truncation."""
        # Validate bounds
        height = min(height, MAX_WINDOW_HEIGHT)
        width = min(width, MAX_WINDOW_WIDTH)

        lines = []
        lines.append(f"# viewport: top={top} left={left} height={height} width={width}")

        # Column headers
        col_headers = ["# cols:"]
        for col in range(left, left + width):
            col_headers.append(f"{get_column_letter(col):>8}")
        lines.append(" | ".join(col_headers))

        # Data rows with smart truncation
        for row in range(top, top + height):
            row_data = [f"{row:3d}"]
            for col in range(left, left + width):
                cell = self.worksheet.cell(row=row, column=col)
                # Use smart truncation
                str_value = self._safe_str(cell.value)
                # Handle empty cells with NULL symbol for better compression
                if str_value == "":
                    str_value = "␀"  # Unicode NULL symbol for empty cells
                row_data.append(f"{str_value:>8}")
            lines.append(" | ".join(row_data))

        return "\n".join(lines)

    def _range_peek(self, range: str, format: str = "markdown") -> str:
        """Peek at a specific range of cells."""
        # Parse range (e.g., "A1:D20")
        parts = range.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid range: {range}")

        start_col, start_row = coordinate_from_string(parts[0])
        end_col, end_row = coordinate_from_string(parts[1])

        start_col_idx = column_index_from_string(start_col)
        end_col_idx = column_index_from_string(end_col)

        # Validate area
        area = (end_row - start_row + 1) * (end_col_idx - start_col_idx + 1)
        if area > MAX_RANGE_AREA:
            raise ValueError(f"Range area {area} exceeds maximum {MAX_RANGE_AREA}")

        return self._window_grab(
            top=start_row,
            left=start_col_idx,
            height=end_row - start_row + 1,
            width=end_col_idx - start_col_idx + 1,
            format=format,
        )

    def _cell_meta(self, row: int, col: int) -> dict[str, Any]:
        """Get metadata for a specific cell."""
        if self.tool_call_count["cell_meta"] > MAX_CELL_META_CALLS:
            raise ValueError(f"Exceeded maximum cell_meta calls ({MAX_CELL_META_CALLS})")

        cell = self.worksheet.cell(row=row, column=col)

        # Determine value type
        value_type = "blank"
        if cell.value is not None:
            if isinstance(cell.value, (int, float)):
                value_type = "num"
            elif isinstance(cell.value, str):
                if cell.value.startswith("="):
                    value_type = "num"  # Formula result
                else:
                    value_type = "text"
            elif hasattr(cell.value, "date"):
                value_type = "date"
            elif cell.value == "#ERROR":
                value_type = "error"

        # Check if formula - use non-read-only workbook for accurate detection
        is_formula = False
        try:
            # Load a small non-read-only handle for formula detection on first use
            # This works around data_only=True hiding formulas
            if not hasattr(self, "_formula_workbook"):
                # Re-open the same file without data_only for formula detection
                self._formula_workbook = openpyxl.load_workbook(self.excel_path, read_only=False, data_only=False)

            # Get the corresponding worksheet by index
            sheet_index = self.worksheet.parent.worksheets.index(self.worksheet)
            formula_worksheet = self._formula_workbook.worksheets[sheet_index]
            formula_cell = formula_worksheet.cell(row=row, column=col)

            # Check if the cell has a formula (starts with =)
            if (
                hasattr(formula_cell, "value")
                and isinstance(formula_cell.value, str)
                and formula_cell.value.startswith("=")
            ):
                is_formula = True
        except Exception as e:
            # Fallback to the data_only check (will miss formulas but won't crash)
            self.logger.main.debug(f"Formula detection fallback for {row},{col}: {e}")
            if isinstance(cell.value, str) and cell.value.startswith("="):
                is_formula = True

        # Check if merged (simplified check)
        is_merged = False
        # merged_cells not available in read-only mode
        if hasattr(self.worksheet, "merged_cells"):
            for merged_range in self.worksheet.merged_cells.ranges:
                if cell.coordinate in merged_range:
                    is_merged = True
                    break

        # Style flags (simplified)
        style_flags = []
        if cell.font and cell.font.bold:
            style_flags.append("bold")
        if cell.alignment and cell.alignment.horizontal == "center":
            style_flags.append("center")

        return {"valueType": value_type, "isFormula": is_formula, "isMerged": is_merged, "styleFlags": style_flags}

    def _merged_list(self) -> list[dict[str, str]]:
        """Get list of merged cell ranges."""
        merged = []
        # merged_cells not available in read-only mode
        if hasattr(self.worksheet, "merged_cells"):
            for merged_range in self.worksheet.merged_cells.ranges:
                merged.append({"range": str(merged_range)})
                if len(merged) >= 2000:  # Budget limit
                    break
        return merged

    def _names_list(self) -> list[dict[str, str]]:
        """Get list of named ranges."""
        # Note: Named ranges are workbook-level in openpyxl
        named = []
        try:
            if hasattr(self.worksheet.parent, "defined_names"):
                dn = self.worksheet.parent.defined_names
                # Handle different openpyxl versions
                if hasattr(dn, "items"):
                    iterable = dn.items()
                elif hasattr(dn, "definedName"):
                    # Older versions have a list-like structure
                    iterable = [(n.name, n) for n in dn.definedName]
                else:
                    # Fallback: try to iterate directly
                    iterable = [(getattr(n, "name", str(n)), n) for n in dn]

                for name, defn in iterable:
                    if len(named) >= 1000:  # Budget limit
                        break
                    named.append(
                        {"name": name, "range": str(defn.attr_text) if hasattr(defn, "attr_text") else str(defn)}
                    )
        except Exception as e:
            self.logger.main.debug(f"Could not retrieve named ranges: {e}")
        return named

    def _object_scan(self, type: str) -> list[dict[str, Any]]:
        """Scan for charts or pivot tables."""
        objects = []

        if type == "chart":
            # Check for charts in the worksheet
            try:
                if hasattr(self.worksheet, "_charts"):
                    for chart in self.worksheet._charts:
                        # Get chart anchor (top-left cell)
                        anchor = "A1"  # Default
                        if hasattr(chart, "anchor"):
                            if hasattr(chart.anchor, "_from"):
                                # Extract cell from anchor
                                from_marker = chart.anchor._from
                                if hasattr(from_marker, "col") and hasattr(from_marker, "row"):
                                    col_letter = get_column_letter(from_marker.col + 1)
                                    anchor = f"{col_letter}{from_marker.row + 1}"
                            elif hasattr(chart.anchor, "ref"):
                                # Alternative anchor format
                                anchor = (
                                    str(chart.anchor.ref).split(":")[0]
                                    if ":" in str(chart.anchor.ref)
                                    else str(chart.anchor.ref)
                                )

                        objects.append(
                            {
                                "type": "chart",
                                "anchor": anchor,
                                "title": chart.title if hasattr(chart, "title") else "Untitled Chart",
                            }
                        )

                        if len(objects) >= 50:  # Limit to prevent excessive scanning
                            break
            except Exception as e:
                self.logger.main.debug(f"Error scanning for charts: {e}")

        elif type == "pivot":
            # Pivot tables require parsing the workbook XML which is complex
            # For now, we can check for pivot cache definitions
            try:
                if hasattr(self.worksheet.parent, "pivots"):
                    for pivot in self.worksheet.parent.pivots:
                        objects.append(
                            {
                                "type": "pivot",
                                "anchor": "A1",  # Would need to parse XML for actual location
                                "name": pivot.name if hasattr(pivot, "name") else "Pivot Table",
                            }
                        )

                        if len(objects) >= 20:  # Limit pivots
                            break
            except Exception as e:
                self.logger.main.debug(f"Error scanning for pivots: {e}")

        return objects

    def calculate_layout_homogeneity(self, range_text: str) -> float:
        """Calculate layout homogeneity based on row AND column patterns.

        Returns a value between 0 (heterogeneous) and 1 (homogeneous).
        Combines entropy of row densities and column patterns.
        """
        lines = range_text.split("\n")[2:]  # Skip viewport header and column headers
        if not lines:
            return 1.0

        # Parse the data into a 2D array
        data_matrix = []
        for line in lines:
            cells = line.split("|")[1:]  # Skip row number
            if cells:
                data_matrix.append([cell.strip() for cell in cells])

        if not data_matrix:
            return 1.0

        # Check for merged cells spanning multiple columns (header/banner indicator)
        first_row = data_matrix[0] if data_matrix else []
        if first_row:
            # Count consecutive identical non-empty values (likely merged)
            consecutive_count = 1
            max_consecutive = 1
            for i in range(1, len(first_row)):
                if first_row[i] == first_row[i - 1] and first_row[i] not in ["", "␀"]:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 1

            # If more than 50% of columns seem merged, it's likely a banner
            if max_consecutive > len(first_row) * 0.5:
                return 0.3  # Low homogeneity for banner-like structures

        # Calculate ROW entropy
        row_densities = []
        for row in data_matrix:
            non_empty = sum(1 for cell in row if cell and cell != "␀")
            density = non_empty / len(row) if row else 0
            row_densities.append(density)

        # Calculate COLUMN entropy
        num_cols = max(len(row) for row in data_matrix) if data_matrix else 0
        col_densities = []
        for col_idx in range(num_cols):
            non_empty = 0
            total = 0
            for row in data_matrix:
                if col_idx < len(row):
                    total += 1
                    if row[col_idx] and row[col_idx] != "␀":
                        non_empty += 1
            density = non_empty / total if total > 0 else 0
            col_densities.append(density)

        # Calculate entropy for rows
        row_homogeneity = 1.0
        if len(set(row_densities)) > 1:
            bins = np.linspace(0, 1, 11)  # 10 bins
            digitized = np.digitize(row_densities, bins)
            _, counts = np.unique(digitized, return_counts=True)
            probabilities = counts / len(digitized)
            entropy = stats.entropy(probabilities, base=2)
            max_entropy = np.log2(len(bins))
            row_homogeneity = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)

        # Calculate entropy for columns
        col_homogeneity = 1.0
        if len(set(col_densities)) > 1:
            bins = np.linspace(0, 1, 11)  # 10 bins
            digitized = np.digitize(col_densities, bins)
            _, counts = np.unique(digitized, return_counts=True)
            probabilities = counts / len(digitized)
            entropy = stats.entropy(probabilities, base=2)
            max_entropy = np.log2(len(bins))
            col_homogeneity = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)

        # Combine both measures (weighted average)
        # Give more weight to row homogeneity as it's usually more important
        homogeneity = 0.7 * row_homogeneity + 0.3 * col_homogeneity

        return homogeneity


class SheetCartographer:
    """Main agent that orchestrates sheet mapping."""

    def __init__(self, excel_path: Path, sheet_index: int, model: str, logger: ExperimentLogger):
        self.excel_path = excel_path
        self.sheet_index = sheet_index
        self.model = model
        self.logger = logger
        self.start_time = time.time()

        # Load worksheet
        self.logger.main.info(f"📂 Loading Excel file: {excel_path}")
        # Use read_only=True for better performance with large files
        self.workbook = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
        self.worksheet = self.workbook.worksheets[sheet_index]
        self.logger.main.info(f"📊 Loaded sheet {sheet_index}: {self.worksheet.title}")

        # Initialize components
        self.tool_handler = ExcelToolHandler(self.worksheet, logger, self.excel_path)
        self.client = OpenAI()  # Requires OPENAI_API_KEY env var
        self.total_tokens = 0

        # State tracking
        self.blocks: list[Block] = []
        self.objects: list[ChartObject] = []
        self.unresolved: list[dict[str, str]] = []

        # Sheet metadata for global context (will be populated in _orient)
        self.sheet_metadata: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Execute the cartographer workflow."""
        self.logger.main.info("🚀 Starting Sheet-Cartographer analysis")

        try:
            # S0: Orient
            sheet_info = self._orient()

            # S1: Sweep
            density_grid = self._sweep(sheet_info)

            # S2: Cluster
            candidate_blocks = self._cluster(density_grid, sheet_info)

            # S3: Probe
            self._probe(candidate_blocks)

            # S4: Overlay
            self._overlay()

            # S5: Emit
            cartographer_map = self._emit()

            # Log metrics
            elapsed = time.time() - self.start_time
            self.logger.log_metrics(
                {
                    "runtime_seconds": elapsed,
                    "total_tokens": self.total_tokens,
                    "blocks_detected": len(self.blocks),
                    "tool_calls": dict(self.tool_handler.tool_call_count),
                    "token_efficiency": self.total_tokens / (sheet_info["rows"] * sheet_info["cols"]),
                }
            )

            return cartographer_map

        except Exception as e:
            self.logger.error.error(f"Cartographer failed: {e}", exc_info=True)
            raise

    def _orient(self) -> dict[str, Any]:
        """S0: Get sheet dimensions and metadata."""
        self.logger.main.info("📍 S0: ORIENT - Getting sheet dimensions")
        sheet_info = self.tool_handler.execute_tool("sheet_info", {})
        self.logger.main.info(f"   Sheet size: {sheet_info['rows']}x{sheet_info['cols']} ({sheet_info['usedRange']})")

        # Store metadata for global LLM context
        self.sheet_metadata = sheet_info
        self.sheet_metadata["sheet_name"] = self.worksheet.title

        # Create system metadata string for LLM
        self.system_metadata = self._create_system_metadata(sheet_info)
        self.logger.main.debug(f"System metadata for LLM:\n{self.system_metadata}")

        return sheet_info

    def _create_system_metadata(self, sheet_info: dict[str, Any]) -> str:
        """Create concise sheet metadata for system prompt.

        This provides global spatial context to improve LLM reasoning about
        where viewports sit within the full sheet grid.
        """
        rows = sheet_info["rows"]
        cols = sheet_info["cols"]
        used_range = sheet_info["usedRange"]

        # Get column letters for better readability
        max_col_letter = get_column_letter(cols)

        metadata_parts = [
            "You are the Sheet-Cartographer analyzing spreadsheet structure.",
            "",
            "SHEET METADATA:",
            f"• Sheet name: {self.worksheet.title}",
            f"• Dimensions: {rows} rows × {cols} columns (A–{max_col_letter})",
            f"• Used range: {used_range}",
        ]

        # Add hidden rows/cols if any
        if sheet_info.get("hiddenRows"):
            metadata_parts.append(f"• Hidden rows: {sheet_info['hiddenRows']}")
        if sheet_info.get("hiddenCols"):
            metadata_parts.append(f"• Hidden columns: {sheet_info['hiddenCols']}")

        metadata_parts.extend(
            [
                "",
                "Always treat any viewport or range as a sub-region of this full grid.",
                "This helps you understand where data likely continues beyond the current view.",
            ]
        )

        return "\n".join(metadata_parts)

    def _sweep(self, sheet_info: dict[str, Any]) -> list[list[float]]:
        """S1: Sweep across sheet recording density."""
        self.logger.main.info("🔍 S1: SWEEP - Scanning sheet density")

        # Initialize density grid
        rows = sheet_info["rows"]
        cols = sheet_info["cols"]

        # Use adaptive window size
        window_height = 40
        window_width = 20

        # Store step sizes for later use in _cluster
        self.sweep_step_h = window_height // 2
        self.sweep_step_w = window_width // 2

        density_grid = []

        # Sweep row-major with sliding windows
        for row_start in range(1, rows + 1, self.sweep_step_h):
            # Check runtime cap
            if time.time() - self.start_time > MAX_RUNTIME_SECONDS:
                windows_processed = len(density_map)
                total_windows = ((rows + self.sweep_step_h - 1) // self.sweep_step_h) * (
                    (cols + self.sweep_step_w - 1) // self.sweep_step_w
                )
                self.logger.main.warning(
                    f"Runtime cap hit during sweep; processed {windows_processed}/{total_windows} windows"
                )
                break

            density_row = []
            for col_start in range(1, cols + 1, self.sweep_step_w):
                # Use local window dimensions to avoid mutation
                current_height = min(window_height, rows - row_start + 1)
                current_width = min(window_width, cols - col_start + 1)

                # Grab window
                window_text = self.tool_handler.execute_tool(
                    "window_grab",
                    {
                        "top": row_start,
                        "left": col_start,
                        "height": current_height,
                        "width": current_width,
                        "format": "markdown",
                    },
                )

                # Calculate density (count cells with actual values)
                lines = window_text.split("\n")[2:]  # Skip viewport header and column headers
                non_empty = 0
                total = 0
                for line in lines:
                    cells = line.split("|")[1:]  # Skip row number
                    for cell in cells:
                        total += 1
                        # Check if cell has content (not just whitespace or NULL symbol)
                        cell_text = cell.strip()
                        # Count as non-empty if not empty string and not NULL symbol
                        if cell_text and cell_text not in ["", "␀"]:
                            non_empty += 1

                density = non_empty / total if total > 0 else 0
                density_row.append(density)

                # Debug logging for all windows
                self.logger.main.debug(
                    f"   Window [{row_start}:{min(row_start + current_height - 1, rows)}, "
                    f"{col_start}:{min(col_start + current_width - 1, cols)}] "
                    f"non_empty: {non_empty}/{total}, density: {density:.3f}"
                )

                # Note: Could implement adaptive re-grab here if density is high
                # For now, just log that this window is dense
                if density > HIGH_DENSITY_THRESHOLD:
                    self.logger.main.debug(f"   Dense window detected at [{row_start}, {col_start}]")

            density_grid.append(density_row)

        self.logger.main.info(
            f"   Density grid size: {len(density_grid)}x{len(density_grid[0]) if density_grid else 0}"
        )
        return density_grid

    def _cluster(self, density_grid: list[list[float]], sheet_info: dict[str, Any]) -> list[dict[str, Any]]:
        """S2: Cluster adjacent dense tiles into candidate blocks."""
        self.logger.main.info("🔗 S2: CLUSTER - Merging dense regions")

        candidate_blocks = []

        # Simple clustering: find contiguous regions above threshold
        visited = set()

        for i, row in enumerate(density_grid):
            for j, density in enumerate(row):
                if density > DENSITY_THRESHOLD and (i, j) not in visited:
                    # Start a new cluster
                    cluster = self._flood_fill(density_grid, i, j, visited)

                    # Convert cluster to range
                    min_i = min(c[0] for c in cluster)
                    max_i = max(c[0] for c in cluster)
                    min_j = min(c[1] for c in cluster)
                    max_j = max(c[1] for c in cluster)

                    # Convert grid coordinates to sheet coordinates using actual sweep steps
                    start_row = min_i * self.sweep_step_h + 1
                    # Extend to actual data boundaries, not just detected density
                    end_row = min((max_i + 2) * self.sweep_step_h, sheet_info["rows"])  # More generous boundaries
                    start_col = min_j * self.sweep_step_w + 1
                    end_col = min((max_j + 2) * self.sweep_step_w, sheet_info["cols"])  # More generous boundaries

                    block_range = f"{get_column_letter(start_col)}{start_row}:{get_column_letter(end_col)}{end_row}"

                    candidate_blocks.append(
                        {"range": block_range, "density": sum(density_grid[c[0]][c[1]] for c in cluster) / len(cluster)}
                    )

        self.logger.main.info(f"   Found {len(candidate_blocks)} candidate blocks")
        return candidate_blocks

    def _flood_fill(self, grid: list[list[float]], i: int, j: int, visited: set) -> list[tuple[int, int]]:
        """Flood fill to find connected dense regions."""
        cluster = []
        stack = [(i, j)]

        while stack:
            ci, cj = stack.pop()
            if (ci, cj) in visited:
                continue
            if ci < 0 or ci >= len(grid) or cj < 0 or cj >= len(grid[0]):
                continue
            if grid[ci][cj] <= DENSITY_THRESHOLD:
                continue

            visited.add((ci, cj))
            cluster.append((ci, cj))

            # Add neighbors
            stack.extend([(ci - 1, cj), (ci + 1, cj), (ci, cj - 1), (ci, cj + 1)])

        return cluster

    def _split_until_homogeneous(self, range_str: str, depth: int = 0, max_depth: int = 3) -> list[dict[str, Any]]:
        """Recursively split a range until homogeneous sub-regions are found.

        Args:
            range_str: A1-style range string
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            List of homogeneous sub-ranges with their metadata
        """
        # Prevent infinite recursion
        if depth >= max_depth:
            return [{"range": range_str, "homogeneity": 0.5, "split": False}]

        # Parse range
        parts = range_str.split(":")
        if len(parts) != 2:
            return [{"range": range_str, "homogeneity": 0.5, "split": False}]

        start_col, start_row = coordinate_from_string(parts[0])
        end_col, end_row = coordinate_from_string(parts[1])
        start_col_idx = column_index_from_string(start_col)
        end_col_idx = column_index_from_string(end_col)

        # Check area constraint
        area = (end_row - start_row + 1) * (end_col_idx - start_col_idx + 1)
        if area > MAX_RANGE_AREA:
            # Force split for large areas
            self.logger.main.debug(f"   Area {area} exceeds max, forcing split")
            mid_row = (start_row + end_row) // 2
            upper_range = f"{start_col}{start_row}:{end_col}{mid_row}"
            lower_range = f"{start_col}{mid_row + 1}:{end_col}{end_row}"

            upper_splits = self._split_until_homogeneous(upper_range, depth + 1, max_depth)
            lower_splits = self._split_until_homogeneous(lower_range, depth + 1, max_depth)
            return upper_splits + lower_splits

        # Get range content
        range_text = self.tool_handler.execute_tool("range_peek", {"range": range_str, "format": "markdown"})

        # Calculate homogeneity
        homogeneity = self.tool_handler.calculate_layout_homogeneity(range_text)

        self.logger.main.debug(f"   {'  ' * depth}Range {range_str}: homogeneity={homogeneity:.3f}")

        # If homogeneous enough, return as is
        if homogeneity >= HOMOGENEITY_THRESHOLD:
            return [{"range": range_str, "homogeneity": homogeneity, "split": False, "text": range_text}]

        # Otherwise, split and recurse
        total_rows = end_row - start_row + 1
        total_cols = end_col_idx - start_col_idx + 1

        sub_ranges = []

        # Decide split direction based on dimensions
        if total_rows > total_cols * 2:
            # Split horizontally (by rows)
            mid_row = (start_row + end_row) // 2
            upper_range = f"{start_col}{start_row}:{end_col}{mid_row}"
            lower_range = f"{start_col}{mid_row + 1}:{end_col}{end_row}"

            self.logger.main.debug(f"   {'  ' * depth}Splitting horizontally at row {mid_row}")

            sub_ranges.extend(self._split_until_homogeneous(upper_range, depth + 1, max_depth))
            sub_ranges.extend(self._split_until_homogeneous(lower_range, depth + 1, max_depth))

        elif total_cols > total_rows * 2:
            # Split vertically (by columns)
            mid_col_idx = (start_col_idx + end_col_idx) // 2
            mid_col = get_column_letter(mid_col_idx)

            left_range = f"{start_col}{start_row}:{mid_col}{end_row}"
            right_range = f"{get_column_letter(mid_col_idx + 1)}{start_row}:{end_col}{end_row}"

            self.logger.main.debug(f"   {'  ' * depth}Splitting vertically at column {mid_col}")

            sub_ranges.extend(self._split_until_homogeneous(left_range, depth + 1, max_depth))
            sub_ranges.extend(self._split_until_homogeneous(right_range, depth + 1, max_depth))

        else:
            # Split into quadrants
            mid_row = (start_row + end_row) // 2
            mid_col_idx = (start_col_idx + end_col_idx) // 2
            mid_col = get_column_letter(mid_col_idx)

            self.logger.main.debug(f"   {'  ' * depth}Splitting into quadrants at {mid_col}{mid_row}")

            # Top-left
            sub_ranges.extend(
                self._split_until_homogeneous(f"{start_col}{start_row}:{mid_col}{mid_row}", depth + 1, max_depth)
            )
            # Top-right
            sub_ranges.extend(
                self._split_until_homogeneous(
                    f"{get_column_letter(mid_col_idx + 1)}{start_row}:{end_col}{mid_row}", depth + 1, max_depth
                )
            )
            # Bottom-left
            sub_ranges.extend(
                self._split_until_homogeneous(f"{start_col}{mid_row + 1}:{mid_col}{end_row}", depth + 1, max_depth)
            )
            # Bottom-right
            sub_ranges.extend(
                self._split_until_homogeneous(
                    f"{get_column_letter(mid_col_idx + 1)}{mid_row + 1}:{end_col}{end_row}", depth + 1, max_depth
                )
            )

        return sub_ranges

    def _probe(self, candidate_blocks: list[dict[str, Any]]) -> None:
        """S3: Probe and classify candidate blocks with recursive segmentation."""
        self.logger.main.info("🔬 S3: PROBE - Segmenting and classifying blocks")

        block_id = 1
        for idx, candidate in enumerate(candidate_blocks):
            # Check runtime cap
            if time.time() - self.start_time > MAX_RUNTIME_SECONDS:
                regions_processed = idx
                total_regions = len(candidate_blocks)
                self.logger.main.warning(
                    f"Runtime cap hit during probe; processed {regions_processed}/{total_regions} regions"
                )
                break

            self.logger.main.info(f"   Processing candidate {idx + 1}/{len(candidate_blocks)}: {candidate['range']}")

            # First, recursively split until homogeneous
            sub_regions = self._split_until_homogeneous(candidate["range"])

            self.logger.main.info(f"   Split into {len(sub_regions)} homogeneous regions")

            # Now classify each homogeneous sub-region
            for sub_region in sub_regions:
                # Skip very sparse regions
                if "text" in sub_region:
                    lines = sub_region["text"].split("\n")[2:]  # Skip viewport line + column headers
                    total_cells = sum(len(line.split("|")[1:]) for line in lines)
                    non_empty = sum(
                        1 for line in lines for cell in line.split("|")[1:] if cell.strip() and cell.strip() != "␀"
                    )
                    density = non_empty / total_cells if total_cells > 0 else 0

                    if density < 0.05:  # Skip regions with < 5% density
                        self.logger.main.debug(
                            f"   Skipping sparse region {sub_region['range']} (density: {density:.3f})"
                        )
                        continue

                # Get full content if not already available
                if "text" not in sub_region:
                    sub_region["text"] = self.tool_handler.execute_tool(
                        "range_peek", {"range": sub_region["range"], "format": "markdown"}
                    )

                # Check for embedded totals in the text that suggest column-wise splitting
                text_lower = sub_region["text"].lower()
                # Check for various forms of "total" keywords that indicate mixed regions
                has_embedded_totals = any(
                    keyword in text_lower
                    for keyword in [
                        "total revenue",
                        "total revenues",
                        "total expense",
                        "total expenses",
                        "total income",
                        "total cost",
                        "summary",
                        "subtotal",
                        "grand total",
                    ]
                )

                # Initial classification with new extended format
                result = self._classify_block_with_llm(sub_region["text"], sub_region["range"])

                # Unpack all fields from extended classification
                (
                    block_type,
                    semantic_desc,
                    confidence,
                    reasoning,
                    open_questions,
                    peek_requests,
                    suggested_ops,
                    agg_subtype,
                ) = result

                # Force low confidence if embedded totals detected
                if has_embedded_totals and confidence > 0.5:
                    self.logger.main.info(f"   Detected embedded totals in {sub_region['range']} - forcing split")
                    confidence = 0.4  # Force splitting

                # Iterative refinement based on open questions and peek requests
                refinement_count = 0
                max_refinements = 3
                # Keep initial peek_requests from first classification
                peeked_ranges = set()  # Avoid duplicate peeks
                prev_confidence = confidence  # Track confidence changes
                prev_block_type = block_type  # Track type stability
                all_peeked_text = []  # Collect all peeked content for re-evaluation
                # Initialize running operations list for accumulation
                running_ops = list(suggested_ops)

                while (open_questions or peek_requests) and refinement_count < max_refinements:
                    self.logger.main.debug(f"   Refinement {refinement_count + 1} for {sub_region['range']}")
                    self.logger.main.debug(f"   Open questions: {open_questions}")

                    additional_context_parts = []

                    # Process any peek requests from LLM
                    if peek_requests:
                        self.logger.main.debug(f"   Processing {len(peek_requests)} peek requests")
                        for peek_req in peek_requests:
                            peek_range = peek_req.get("range", "")
                            peek_reason = peek_req.get("reason", "")

                            # Skip if already peeked
                            if peek_range in peeked_ranges:
                                continue
                            peeked_ranges.add(peek_range)

                            try:
                                # Execute the peek
                                peek_text = self.tool_handler.execute_tool(
                                    "range_peek", {"range": peek_range, "format": "markdown"}
                                )
                                additional_context_parts.append(
                                    f"Peek at {peek_range} (requested: {peek_reason}):\n{peek_text}"
                                )
                                # Store for homogeneity re-evaluation
                                all_peeked_text.append(peek_text)
                            except Exception as e:
                                self.logger.main.debug(f"   Failed to peek at {peek_range}: {e}")

                    # Address open questions with targeted probing
                    if open_questions:
                        heuristic_context = self._address_open_questions(
                            open_questions, sub_region["range"], sub_region["text"]
                        )
                        if heuristic_context and heuristic_context != "No additional context gathered":
                            additional_context_parts.append(heuristic_context)

                    additional_context = "\n\n".join(additional_context_parts) if additional_context_parts else None

                    # Re-classify with additional context
                    result = self._classify_block_with_llm(
                        sub_region["text"], sub_region["range"], additional_context=additional_context
                    )

                    # Unpack all fields from extended classification
                    (
                        new_block_type,
                        new_semantic_desc,
                        new_confidence,
                        new_reasoning,
                        open_questions,
                        peek_requests,
                        suggested_ops,
                        new_agg_subtype,
                    ) = result

                    # Check for confidence plateau (no meaningful improvement)
                    confidence_change = abs(new_confidence - prev_confidence)
                    if new_block_type == prev_block_type and confidence_change < 0.05 and refinement_count > 0:
                        self.logger.main.info(
                            f"   Confidence plateau detected ({prev_confidence:.2f} → {new_confidence:.2f}), forcing split"
                        )
                        # Force low confidence to trigger splitting
                        new_confidence = 0.4
                        break  # Exit refinement loop

                    # Recompute homogeneity if we have peeked additional content
                    if all_peeked_text:
                        # Concatenate original and peeked content
                        combined_text = sub_region["text"] + "\n" + "\n".join(all_peeked_text)
                        new_homogeneity = self.tool_handler.calculate_layout_homogeneity(combined_text)

                        if new_homogeneity < HOMOGENEITY_THRESHOLD:
                            self.logger.main.info(
                                f"   Homogeneity dropped to {new_homogeneity:.2f} after peeking - forcing re-split"
                            )
                            # Force low confidence to trigger splitting
                            new_confidence = 0.4
                            break

                    # Update for next iteration
                    block_type = new_block_type
                    semantic_desc = new_semantic_desc
                    confidence = new_confidence
                    reasoning = new_reasoning
                    # Accumulate suggested operations (de-dupe while preserving order)
                    running_ops = list(dict.fromkeys(running_ops + suggested_ops))
                    # Update aggregation subtype only if new one is specified
                    if new_agg_subtype and new_agg_subtype != "none":
                        agg_subtype = new_agg_subtype
                    prev_confidence = new_confidence
                    prev_block_type = new_block_type

                    refinement_count += 1

                # If confidence is still low OR embedded totals detected, force deeper splitting
                if confidence < 0.5 or has_embedded_totals:
                    self.logger.main.info(
                        f"   Low confidence {confidence:.2f} or embedded totals detected - forcing deeper split of {sub_region['range']}"
                    )
                    # Force split even if homogeneous
                    deeper_regions = self._force_split_region(sub_region["range"])

                    # Recursively process the split regions
                    for deep_region in deeper_regions:
                        if "text" not in deep_region:
                            deep_region["text"] = self.tool_handler.execute_tool(
                                "range_peek", {"range": deep_region["range"], "format": "markdown"}
                            )

                        deep_result = self._classify_block_with_llm(deep_region["text"], deep_region["range"])
                        # Extract ALL fields from the extended result tuple
                        (
                            deep_type,
                            deep_semantic,
                            deep_conf,
                            deep_reasoning,
                            deep_open_questions,
                            deep_peek_requests,
                            deep_suggested_ops,
                            deep_agg_subtype,
                        ) = deep_result

                        # Convert string type to enum for deep regions
                        struct_type = TYPE_MAP.get(deep_type, StructuralType.UNSTRUCTURED_TEXT)

                        # Build data characteristics for deep splits
                        deep_data_chars = self._infer_characteristics(deep_region["text"], deep_region["range"])
                        if deep_agg_subtype and deep_agg_subtype != "none":
                            deep_data_chars.aggregation_zone_subtype = deep_agg_subtype

                        # Combine suggested operations with default operations for this type
                        deep_default_ops = self._get_default_operations(struct_type)
                        deep_final_ops = list(dict.fromkeys(deep_suggested_ops + deep_default_ops))

                        self.blocks.append(
                            Block(
                                id=f"blk_{block_id:02d}",
                                range=deep_region["range"],
                                structural_type=struct_type,
                                semantic_description=deep_semantic or "Split region from mixed semantics",
                                confidence=deep_conf,
                                classification_reasoning=deep_reasoning or "Forced split due to mixed semantics",
                                data_characteristics=deep_data_chars,
                                suggested_operations=deep_final_ops,
                            )
                        )
                        self.logger.main.info(
                            f"   Block {block_id}: {deep_region['range']} -> {deep_type} (conf: {deep_conf:.2f})"
                        )
                        block_id += 1
                else:
                    # Add the classified block
                    # Convert string type to enum using centralized mapping
                    struct_type = TYPE_MAP.get(block_type, StructuralType.UNSTRUCTURED_TEXT)

                    # Build data characteristics - infer from text content
                    data_chars = self._infer_characteristics(sub_region["text"], sub_region["range"])
                    # Add classification-specific fields
                    if agg_subtype and agg_subtype != "none":
                        data_chars.aggregation_zone_subtype = agg_subtype

                    # Combine running operations with default operations for this type
                    default_ops = self._get_default_operations(struct_type)
                    final_ops = list(dict.fromkeys(running_ops + default_ops))  # De-dupe preserving order

                    self.blocks.append(
                        Block(
                            id=f"blk_{block_id:02d}",
                            range=sub_region["range"],
                            structural_type=struct_type,
                            semantic_description=semantic_desc or f"Region classified as {block_type}",
                            confidence=confidence,
                            classification_reasoning=reasoning,
                            data_characteristics=data_chars,
                            suggested_operations=final_ops,
                        )
                    )

                    self.logger.main.info(
                        f"   Block {block_id}: {sub_region['range']} -> "
                        f"{block_type} (conf: {confidence:.2f}, homogeneity: {sub_region.get('homogeneity', 0):.2f})"
                    )

                    block_id += 1

    def _force_split_region(self, range_str: str) -> list[dict[str, Any]]:
        """Force split a region into smaller parts when mixed semantics are detected.

        This is used when the LLM detects heterogeneous content even in
        seemingly homogeneous regions.
        """
        # Parse range
        parts = range_str.split(":")
        if len(parts) != 2:
            return [{"range": range_str, "homogeneity": 0.5}]

        start_col, start_row = coordinate_from_string(parts[0])
        end_col, end_row = coordinate_from_string(parts[1])
        start_col_idx = column_index_from_string(start_col)
        end_col_idx = column_index_from_string(end_col)

        total_rows = end_row - start_row + 1
        total_cols = end_col_idx - start_col_idx + 1

        sub_regions = []

        # PRIORITY 1: Check if first row is a header/summary row that should be separated
        if total_rows >= 2:
            # Peek at the first two rows to check for header/summary patterns
            try:
                first_row_range = f"{start_col}{start_row}:{end_col}{start_row}"
                second_row_range = f"{start_col}{start_row + 1}:{end_col}{start_row + 1}"

                first_row_text = self.tool_handler.execute_tool(
                    "range_peek", {"range": first_row_range, "format": "markdown"}
                )
                second_row_text = self.tool_handler.execute_tool(
                    "range_peek", {"range": second_row_range, "format": "markdown"}
                )

                # Extract cell values from markdown
                first_row_cells = []
                second_row_cells = []

                for line in first_row_text.split("\n")[2:]:  # Skip viewport line + column headers
                    cells = line.split("|")[1:]  # Skip row number
                    first_row_cells.extend([c.strip() for c in cells])

                for line in second_row_text.split("\n")[2:]:  # Skip viewport line + column headers
                    cells = line.split("|")[1:]  # Skip row number
                    second_row_cells.extend([c.strip() for c in cells])

                # Count totals keywords and numeric cells in first row
                first_row_lower = " ".join(first_row_cells).lower()
                totals_count = sum(
                    1 for keyword in ["total", "sum", "revenue", "expense"] if keyword in first_row_lower
                )

                # Count numeric vs text cells
                first_row_numeric = sum(
                    1
                    for cell in first_row_cells
                    if cell
                    and cell != "␀"
                    and (
                        cell.replace(".", "").replace("-", "").replace(",", "").isdigit()
                        or cell.startswith("$")
                        or cell.startswith("-$")
                    )
                )
                second_row_numeric = sum(
                    1
                    for cell in second_row_cells
                    if cell
                    and cell != "␀"
                    and (
                        cell.replace(".", "").replace("-", "").replace(",", "").isdigit()
                        or cell.startswith("$")
                        or cell.startswith("-$")
                    )
                )

                # Decision logic: Split off first row if it looks like header/summary
                should_split_first_row = False

                # Check 1: Contains multiple "total" keywords
                if totals_count >= 2:
                    should_split_first_row = True
                    self.logger.main.debug(f"   First row contains {totals_count} total keywords - splitting off")

                # Check 2: First row is >50% numeric while second row is mostly text (headers pattern)
                elif (
                    first_row_numeric > len(first_row_cells) * 0.5 and second_row_numeric < len(second_row_cells) * 0.3
                ):
                    should_split_first_row = True
                    self.logger.main.debug(
                        "   First row is numeric summary, second row appears to be headers - splitting"
                    )

                if should_split_first_row:
                    # Split off the first row as a separate region
                    header_range = f"{start_col}{start_row}:{end_col}{start_row}"
                    rest_range = f"{start_col}{start_row + 1}:{end_col}{end_row}"

                    self.logger.main.debug(f"   Splitting off row {start_row} as header/summary")

                    sub_regions.append({"range": header_range, "homogeneity": 0.9})  # High confidence for single row
                    sub_regions.append({"range": rest_range, "homogeneity": 0.8})

                    return sub_regions

            except Exception as e:
                self.logger.main.debug(f"   Error analyzing first row pattern: {e}")

        # PRIORITY 2: Try to split by columns (often separates data from summary)
        if total_cols >= 3:
            # Smart column splitting based on typical patterns
            # If we have 9 columns (A-I), split between F and G (columns 6 and 7)
            # This often separates transaction data from summary columns
            if total_cols == 9:
                split_col = start_col_idx + 5  # Split after column F (6th column)
            else:
                # Default: split at 2/3 point which often separates main data from summaries
                split_col = start_col_idx + (total_cols * 2 // 3)

            left_range = f"{start_col}{start_row}:{get_column_letter(split_col)}{end_row}"
            right_range = f"{get_column_letter(split_col + 1)}{start_row}:{end_col}{end_row}"

            self.logger.main.debug(
                f"   Force splitting columns at {get_column_letter(split_col)} (after column {split_col - start_col_idx + 1} of {total_cols})"
            )

            sub_regions.append({"range": left_range, "homogeneity": 0.8})
            sub_regions.append({"range": right_range, "homogeneity": 0.8})

        # If can't split by columns, try rows
        elif total_rows >= 2:
            mid_row = (start_row + end_row) // 2

            upper_range = f"{start_col}{start_row}:{end_col}{mid_row}"
            lower_range = f"{start_col}{mid_row + 1}:{end_col}{end_row}"

            self.logger.main.debug(f"   Force splitting rows at {mid_row}")

            sub_regions.append({"range": upper_range, "homogeneity": 0.8})
            sub_regions.append({"range": lower_range, "homogeneity": 0.8})
        else:
            # Can't split further
            sub_regions.append({"range": range_str, "homogeneity": 0.5})

        return sub_regions

    def _address_open_questions(self, questions: list[str], range_str: str, range_text: str) -> str:
        """Address open questions with targeted probing.

        Args:
            questions: List of open questions from LLM
            range_str: The range being analyzed
            range_text: The current text of the range

        Returns:
            Additional context string to help answer the questions
        """
        context_parts = []

        # Parse range
        parts = range_str.split(":")
        if len(parts) != 2:
            return ""

        start_col, start_row = coordinate_from_string(parts[0])
        end_col, end_row = coordinate_from_string(parts[1])
        start_col_idx = column_index_from_string(start_col)
        end_col_idx = column_index_from_string(end_col)

        for question in questions:
            question_lower = question.lower()

            # Check for header-related questions
            if "header" in question_lower or "row 1" in question_lower or "row 2" in question_lower:
                # Get metadata for first few cells
                sample_cells = []
                for col in range(start_col_idx, min(start_col_idx + 3, end_col_idx + 1)):
                    meta = self.tool_handler.execute_tool("cell_meta", {"row": start_row, "col": col})
                    if meta.get("styleFlags") and "bold" in meta["styleFlags"]:
                        sample_cells.append("bold")
                    else:
                        sample_cells.append("normal")

                if any(s == "bold" for s in sample_cells):
                    context_parts.append(f"Row {start_row} contains bold text, likely headers")
                else:
                    context_parts.append(f"Row {start_row} has normal formatting, not headers")

            # Check for totals-related questions
            elif "total" in question_lower or "sum" in question_lower:
                # Look for formula cells in the last few rows
                has_formulas = False

                for row in range(max(end_row - 2, start_row), end_row + 1):
                    for col in range(start_col_idx, min(start_col_idx + 3, end_col_idx + 1)):
                        meta = self.tool_handler.execute_tool("cell_meta", {"row": row, "col": col})
                        if meta.get("isFormula"):
                            has_formulas = True
                            break
                    if has_formulas:
                        break

                if has_formulas:
                    context_parts.append(f"Rows near {end_row} contain formulas, likely summary calculations")

                # Also check for "total" keywords
                if "total" in range_text.lower():
                    context_parts.append("Text contains 'total' keyword, confirming summary section")

            # Check for sparse/empty area questions
            elif "sparse" in question_lower or "empty" in question_lower or "blank" in question_lower:
                lines = range_text.split("\n")[2:]  # Skip viewport header and column headers
                last_quarter = lines[-(len(lines) // 4) :] if len(lines) > 4 else lines

                blank_rows = sum(
                    1 for line in last_quarter if all(cell.strip() in ["", "␀"] for cell in line.split("|")[1:])
                )

                if blank_rows > len(last_quarter) * 0.5:
                    context_parts.append("Bottom portion is mostly blank, indicating end of data")
                else:
                    context_parts.append("Bottom portion contains data, not just padding")

        return " | ".join(context_parts) if context_parts else "No additional context gathered"

    def _create_range_breadcrumb(self, range_str: str) -> str:
        """Create a breadcrumb context for where the range sits in the full sheet.

        This helps the LLM understand spatial positioning within the full grid.
        """
        # Parse range
        parts = range_str.split(":")
        if len(parts) != 2:
            return f"📍 Context: Examining range {range_str} within full sheet ({self.sheet_metadata.get('rows', '?')}×{self.sheet_metadata.get('cols', '?')})"

        start_col, start_row = coordinate_from_string(parts[0])
        end_col, end_row = coordinate_from_string(parts[1])

        total_rows = self.sheet_metadata.get("rows", 1)
        total_cols = self.sheet_metadata.get("cols", 1)

        # Determine relative position
        position_hints = []

        # Vertical position
        if start_row <= 5:
            position_hints.append("near top")
        elif end_row >= total_rows - 5:
            position_hints.append("near bottom")
        else:
            position_hints.append(f"middle region (row {start_row}/{total_rows})")

        # Horizontal position
        start_col_idx = column_index_from_string(start_col)
        end_col_idx = column_index_from_string(end_col)

        if start_col_idx <= 3:
            position_hints.append("left side")
        elif end_col_idx >= total_cols - 3:
            position_hints.append("right side")
        else:
            position_hints.append("center columns")

        # Check if it spans full width or height
        if end_col_idx - start_col_idx + 1 == total_cols:
            position_hints.append("full width")
        if end_row - start_row + 1 == total_rows:
            position_hints.append("full height")

        position_str = ", ".join(position_hints)

        return f"📍 Context: Examining {range_str} ({position_str}) within full sheet ({total_rows}×{total_cols})"

    def _infer_characteristics(self, viewport_text: str, range_str: str) -> DataCharacteristics:
        """Infer data characteristics from viewport text for enrichment.

        Analyzes the actual content to determine:
        - Primary data type (financial, temporal, categorical)
        - Header presence and structure
        - Formula density (via cell_meta sampling)
        - Numeric ratio
        - Time series patterns
        """
        chars = DataCharacteristics()

        # Sample cells for formula detection since we opened with data_only=True
        parts = range_str.split(":")
        if len(parts) == 2:
            start_col, start_row = coordinate_from_string(parts[0])
            end_col, end_row = coordinate_from_string(parts[1])
            start_col_idx = column_index_from_string(start_col)
            end_col_idx = column_index_from_string(end_col)

            # Sample first 2 header cells and last 2 body cells
            sample_coords = []
            # First two cells of first row
            for col in range(start_col_idx, min(start_col_idx + 2, end_col_idx + 1)):
                sample_coords.append((start_row, col))
            # Last two cells of last row
            for col in range(max(start_col_idx, end_col_idx - 1), end_col_idx + 1):
                sample_coords.append((end_row, col))

            formula_hits = 0
            samples_taken = 0
            for row, col in sample_coords[:4]:  # Limit to 4 samples
                try:
                    meta = self.tool_handler.execute_tool("cell_meta", {"row": row, "col": col})
                    samples_taken += 1
                    if meta.get("isFormula"):
                        formula_hits += 1
                except:
                    pass  # Skip if cell_meta fails

            if samples_taken > 0:
                chars.formula_density = formula_hits / samples_taken

        # Parse the markdown table
        lines = viewport_text.split("\n")
        if len(lines) < 3:  # Need at least viewport header + column headers + 1 data row
            return chars

        # Skip markdown header rows (first 2 lines: viewport info + column headers)
        data_lines = lines[2:]
        if not data_lines:
            return chars

        # Extract cells from each row
        rows = []
        for line in data_lines:
            # Don't slice off the end - lines may not have trailing |
            parts = line.split("|")
            if len(parts) > 1:
                cells = [cell.strip() for cell in parts[1:]]
                # Filter out any trailing empty strings from the split
                while cells and cells[-1] == "":
                    cells.pop()
                if cells:
                    rows.append(cells)

        if not rows:
            return chars

        # Analyze first row for headers
        first_row = rows[0]
        has_numeric_first_row = any(self._is_numeric(cell) for cell in first_row if cell and cell != "␀")
        has_text_first_row = any(not self._is_numeric(cell) and cell and cell != "␀" for cell in first_row)

        # If first row is mostly text and subsequent rows have numbers, likely headers
        if len(rows) > 1 and has_text_first_row and not has_numeric_first_row:
            chars.has_headers = True
            chars.header_rows = 1

        # Calculate numeric ratio
        total_cells = 0
        numeric_cells = 0
        date_cells = 0
        formula_cells = 0
        empty_cells = 0

        for row in rows[chars.header_rows :]:
            for cell in row:
                if not cell or cell == "␀":
                    empty_cells += 1
                    continue

                total_cells += 1

                # Check for formulas (simplified - starts with =)
                if cell.startswith("="):
                    formula_cells += 1
                    numeric_cells += 1  # Formulas usually produce numbers
                elif self._is_numeric(cell):
                    numeric_cells += 1
                elif self._is_date(cell):
                    date_cells += 1

        # Calculate ratios
        if total_cells > 0:
            chars.numeric_ratio = numeric_cells / total_cells
            # Keep the higher signal between sampled and computed formula density
            sampled_formula_density = chars.formula_density  # from cell_meta sampling above
            computed_formula_density = formula_cells / total_cells
            chars.formula_density = max(sampled_formula_density, computed_formula_density)
            chars.has_formulas = chars.formula_density > 0
            chars.data_density = total_cells / (total_cells + empty_cells) if (total_cells + empty_cells) > 0 else 0

            # Determine primary data type
            if chars.numeric_ratio > 0.5:
                # Check if financial (currency symbols, decimals)
                has_currency = any(
                    "$" in str(cell) or "€" in str(cell) or "," in str(cell) for row in rows for cell in row
                )
                if has_currency or chars.formula_density > 0.2:
                    chars.primary_data_type = "financial"
                else:
                    chars.primary_data_type = "numeric"
            elif date_cells / total_cells > 0.2:
                chars.primary_data_type = "temporal"
                chars.is_time_series = True
            else:
                chars.primary_data_type = "categorical"

        # Check for totals row/column patterns
        if rows:
            last_row = rows[-1]
            # Check if last row contains "total" keyword
            if any("total" in str(cell).lower() for cell in last_row):
                chars.has_totals_row = True

        return chars

    def _is_numeric(self, value: str) -> bool:
        """Check if a string value represents a number."""
        if not value or value == "␀":
            return False
        # Remove common formatting
        cleaned = value.strip().replace(",", "").replace("$", "").replace("%", "")
        # Handle accounting-style negatives: (1,234.56) → -1234.56
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            float(cleaned)
            return True
        except:
            return False

    def _is_date(self, value: str) -> bool:
        """Check if a string value represents a date."""
        if not value or value == "␀":
            return False
        # Simple date patterns
        date_patterns = [
            r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",  # MM/DD/YYYY or DD-MM-YY
            r"\d{4}[/-]\d{1,2}[/-]\d{1,2}",  # YYYY-MM-DD
            r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}",  # Month DD, YYYY
        ]
        import re

        return any(re.match(pattern, value.strip()) for pattern in date_patterns)

    def _get_default_operations(self, structural_type: StructuralType) -> list[str]:
        """Get default suggested operations based on structural type."""
        defaults = {
            StructuralType.DATA_TABLE: ["add_filters", "check_duplicates", "profile_columns"],
            StructuralType.HEADER_ZONE: ["extract_column_names", "validate_references"],
            StructuralType.AGGREGATION_ZONE: ["verify_calculations", "extract_summary_values"],
            StructuralType.METADATA_ZONE: ["extract_metadata", "parse_report_info"],
            StructuralType.VISUALIZATION_ANCHOR: ["extract_chart_data", "analyze_visualization"],
            StructuralType.FORM_INPUT_ZONE: ["validate_inputs", "extract_form_schema"],
            StructuralType.NAVIGATION_ZONE: ["extract_links", "map_navigation"],
            StructuralType.UNSTRUCTURED_TEXT: ["extract_text", "parse_notes"],
            StructuralType.EMPTY_PADDING: [],  # No operations for padding
        }
        return defaults.get(structural_type, [])

    def _strip_sheet_prefix(self, range_str: str) -> str:
        """Strip sheet prefix and quotes from range string.

        Converts 'Sheet Name'!$A$1:$E$20 to A1:E20
        """
        if not range_str:
            return range_str

        # Remove sheet prefix if present (everything before and including !)
        if "!" in range_str:
            range_str = range_str.split("!", 1)[1]

        # Remove $ signs from absolute references
        range_str = range_str.replace("$", "")

        return range_str

    def _classify_block_with_llm(
        self, range_text: str, range_str: str, additional_context: str | None = None
    ) -> tuple[str, str, float, str, list[str], list[dict], list[str], str]:
        """Use LLM to classify a block based on its content.

        Returns:
            Tuple of (block_type, semantic_description, confidence, reasoning,
                     open_questions, peek_requests, suggested_operations, aggregation_subtype)
        """

        # Define tools for classification with open questions AND peek requests
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "classify_block",
                    "description": "Classify the type of spreadsheet block with open questions for refinement",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "block_type": {
                                "type": "string",
                                "enum": [
                                    "DataTable",
                                    "HeaderZone",
                                    "AggregationZone",
                                    "MetadataZone",
                                    "VisualizationAnchor",
                                    "FormInputZone",
                                    "NavigationZone",
                                    "UnstructuredText",
                                    "EmptyPadding",
                                ],
                                "description": "Structural type based on layout pattern (DECO-based classification)",
                            },
                            "confidence": {"type": "number", "description": "Confidence score between 0 and 1"},
                            "semantic_description": {
                                "type": "string",
                                "description": "One sentence describing what this region semantically represents",
                            },
                            "reasoning": {"type": "string", "description": "Brief explanation for the classification"},
                            "aggregation_subtype": {
                                "type": "string",
                                "enum": ["kpi", "subtotal", "grand_total", "none"],
                                "description": "If AggregationZone, specify the subtype",
                            },
                            "suggested_operations": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Recommended downstream processing steps",
                            },
                            "open_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Questions that need answers to improve classification",
                            },
                            "peek_requests": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "range": {
                                            "type": "string",
                                            "description": "A1-style range to peek at (e.g., 'A11:I15')",
                                        },
                                        "reason": {
                                            "type": "string",
                                            "description": "Why this range would help classification",
                                        },
                                    },
                                    "required": ["range", "reason"],
                                },
                                "maxItems": 3,
                                "description": "Additional ranges to peek at for better classification (max 3)",
                            },
                        },
                        "required": ["block_type", "semantic_description", "confidence", "reasoning"],
                    },
                },
            }
        ]

        # Combine global metadata with classification instructions
        system_prompt = f"""{self.system_metadata}

CLASSIFICATION INSTRUCTIONS:

STEP 1 - Choose ONE structural type:
- DataTable: Regular rows/columns with consistent schema (>70% Data cells per DECO)
- HeaderZone: Column/row headers that are referenced by formulas
- AggregationZone: Summaries, totals, KPIs (specify subtype: kpi/subtotal/grand_total)
- MetadataZone: Titles, descriptions NOT referenced by formulas
- VisualizationAnchor: Charts, sparklines, conditional formatting
- FormInputZone: Data entry areas with validation/dropdowns
- NavigationZone: Links, buttons, menus (requires HYPERLINK evidence)
- UnstructuredText: Free-form notes without tabular structure
- EmptyPadding: Intentional whitespace for layout

STEP 2 - Write semantic description:
One sentence describing the business/domain meaning, not just structure.

STEP 3 - Provide additional analysis:
Include confidence, reasoning, suggested operations

IMPORTANT: If you detect mixed semantic types in the range, return confidence < 0.5 to trigger further splitting.
Never combine different semantic regions - each should be its own block.
Analyze the structure, not the content. Focus on layout patterns.
Consider where this block sits within the full sheet when making your classification.

When uncertain, provide specific open questions that would help clarify the classification.
For example:
- "Are rows 1-2 headers for the data below?"
- "Is row 10 a totals row?"
- "Does the sparse area at bottom indicate end of data?"
"""

        # Add breadcrumb about current position in sheet
        breadcrumb = self._create_range_breadcrumb(range_str)
        user_content = f"{breadcrumb}\n\nClassify this spreadsheet block at range {range_str}:\n\n{range_text}"

        if additional_context:
            user_content += f"\n\nAdditional context from refinement:\n{additional_context}"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

        try:
            # Check token budget before making LLM call
            if self.total_tokens > MAX_LLM_TOKENS:
                self.logger.error.error(f"Token budget exceeded: {self.total_tokens} > {MAX_LLM_TOKENS}")
                raise RuntimeError(f"Token budget exceeded: {self.total_tokens} > {MAX_LLM_TOKENS}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "classify_block"}},
                temperature=0.1,  # Lower temperature for more consistent classification
                max_tokens=500,
            )

            # Capture actual model from response
            actual_model = getattr(response, "model", None)

            # Update logger with actual model name on first call
            if actual_model and not hasattr(self, "_model_updated"):
                self.logger.update_model_actual(actual_model)
                self._model_updated = True

            # Track tokens
            if hasattr(response, "usage"):
                tokens_used = response.usage.total_tokens
                self.total_tokens += tokens_used
                self.logger.log_llm_interaction(
                    model=self.model,
                    prompt=messages[-1]["content"],
                    response=str(response.choices[0].message),
                    tokens={
                        "input": response.usage.prompt_tokens,
                        "output": response.usage.completion_tokens,
                        "total": tokens_used,
                    },
                    actual_model=actual_model,
                )

            # Extract classification with new fields
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)

                # Build enriched result tuple
                block_type = args["block_type"]
                semantic_desc = args.get("semantic_description", "")
                confidence = args["confidence"]
                reasoning = args.get("reasoning", "")
                open_questions = args.get("open_questions", [])
                peek_requests = args.get("peek_requests", [])
                suggested_ops = args.get("suggested_operations", [])
                agg_subtype = args.get("aggregation_subtype", "none")

                # Return extended tuple with all classification data
                return (
                    block_type,
                    semantic_desc,
                    confidence,
                    reasoning,
                    open_questions,
                    peek_requests,
                    suggested_ops,
                    agg_subtype,
                )

        except Exception as e:
            self.logger.error.error(f"LLM classification failed: {e}")

        # Fallback classification with all required fields
        return (
            "UnstructuredText",
            "Unable to classify region",
            UNKNOWN_CONFIDENCE,
            "Classification failed",
            [],
            [],
            [],
            "none",
        )

    def _overlay(self) -> None:
        """S4: Add merged ranges, named ranges, charts/pivots."""
        self.logger.main.info("🎨 S4: OVERLAY - Adding special elements")

        # Get merged ranges
        merged = self.tool_handler.execute_tool("merged_list", {})
        self.logger.main.info(f"   Found {len(merged)} merged ranges")

        # Get named ranges
        named = self.tool_handler.execute_tool("names_list", {})
        self.logger.main.info(f"   Found {len(named)} named ranges")

        # Match named ranges to blocks (sheet-aware)
        for name_info in named:
            range_str = name_info.get("range", "")
            if not range_str:
                continue

            # Resolve named range to concrete ranges on this worksheet only
            concrete_ranges = self._resolve_named_range_to_current_sheet(range_str)

            for concrete_range in concrete_ranges:
                for block in self.blocks:
                    if self._ranges_overlap(concrete_range, block.range):
                        block.named_range = name_info["name"]
                        break

        # Get charts/pivots
        charts = self.tool_handler.execute_tool("object_scan", {"type": "chart"})
        for idx, chart in enumerate(charts):
            self.objects.append(
                ChartObject(id=f"cht_{idx + 1:02d}", anchor=chart.get("anchor", ""), type="chart", linked_block=None)
            )

    def _split_named_range_areas(self, range_str: str) -> list[str]:
        """Split named range string into areas, handling quoted sheet names properly.

        Example: "'Financial Model'!A1:B2 'Other Sheet'!C1:D2" → ["'Financial Model'!A1:B2", "'Other Sheet'!C1:D2"]
        """
        areas = []
        i = 0
        current_area = ""

        while i < len(range_str):
            char = range_str[i]

            if char == "'":
                # Start of quoted sheet name - find the closing quote after !
                quote_end = range_str.find("'!", i + 1)
                if quote_end != -1:
                    # Find the end of this range (next space or end of string)
                    space_after = range_str.find(" ", quote_end)
                    if space_after == -1:
                        # Last area
                        current_area = range_str[i:].strip()
                        i = len(range_str)
                    else:
                        # Area ends at space
                        current_area = range_str[i:space_after].strip()
                        i = space_after + 1

                    if current_area:
                        areas.append(current_area)
                        current_area = ""
                else:
                    # Malformed quote - skip this character
                    i += 1
            elif char == " ":
                # Regular space - end current area if we have one
                if current_area.strip():
                    areas.append(current_area.strip())
                    current_area = ""
                i += 1
            else:
                # Regular character
                current_area += char
                i += 1

        # Add final area if exists
        if current_area.strip():
            areas.append(current_area.strip())

        return areas

    def _resolve_named_range_to_current_sheet(self, range_str: str) -> list[str]:
        """Resolve named range to concrete ranges on current worksheet only.

        Args:
            range_str: Named range definition which may include:
                - Sheet references: 'Sheet1'!A1:B2
                - External references (skip these)
                - Multi-area ranges: A1:A3 C1:C3

        Returns:
            List of concrete range strings for current sheet only
        """
        if not range_str:
            return []

        # Skip external workbook references (contain [])
        if "[" in range_str and "]" in range_str:
            return []

        current_sheet_name = self.worksheet.title
        concrete_ranges = []

        # Handle multi-area ranges - need to split carefully around quoted sheet names
        areas = self._split_named_range_areas(range_str)

        for area in areas:
            # Check if this area targets current sheet or has no sheet prefix
            if "!" in area:
                # Extract sheet name (handle quoted names properly)
                if area.startswith("'"):
                    # Find the closing quote before !
                    quote_end = area.find("'!")
                    if quote_end != -1:
                        sheet_part = area[1:quote_end]  # Remove outer quotes
                    else:
                        continue  # Malformed quoted range
                else:
                    # No quotes - split normally
                    sheet_part = area.split("!", 1)[0]

                # Skip if not current sheet
                if sheet_part != current_sheet_name:
                    continue

                # Extract range part after !
                range_part = area.split("!", 1)[1]
            else:
                # No sheet prefix - assume current sheet
                range_part = area

            # Clean the range (remove $ signs)
            clean_range = range_part.replace("$", "")

            # Validate it's a proper range format
            if self._is_valid_range_format(clean_range):
                concrete_ranges.append(clean_range)

        return concrete_ranges

    def _is_valid_range_format(self, range_str: str) -> bool:
        """Check if string is a valid Excel range format (A1 or A1:B2)."""
        try:
            if ":" in range_str:
                # Range format A1:B2
                parts = range_str.split(":")
                if len(parts) != 2:
                    return False
                coordinate_from_string(parts[0])
                coordinate_from_string(parts[1])
            else:
                # Single cell A1
                coordinate_from_string(range_str)
            return True
        except:
            return False

    def _ranges_overlap(self, range1: str, range2: str) -> bool:
        """Check if two A1-style ranges overlap.

        Two ranges overlap if their row intervals AND column intervals both intersect.
        Handles sheet prefixes like 'Sheet Name'!$A$1:$E$20.
        """
        try:
            # Strip sheet prefixes and quotes from both ranges
            clean_range1 = self._strip_sheet_prefix(range1)
            clean_range2 = self._strip_sheet_prefix(range2)

            # Parse first range
            if ":" in clean_range1:
                parts1 = clean_range1.split(":")
                start1_col, start1_row = coordinate_from_string(parts1[0])
                end1_col, end1_row = coordinate_from_string(parts1[1])
            else:
                # Single cell
                start1_col, start1_row = coordinate_from_string(clean_range1)
                end1_col, end1_row = start1_col, start1_row

            # Parse second range
            if ":" in clean_range2:
                parts2 = clean_range2.split(":")
                start2_col, start2_row = coordinate_from_string(parts2[0])
                end2_col, end2_row = coordinate_from_string(parts2[1])
            else:
                # Single cell
                start2_col, start2_row = coordinate_from_string(clean_range2)
                end2_col, end2_row = start2_col, start2_row

            # Convert column letters to indices
            start1_col_idx = column_index_from_string(start1_col)
            end1_col_idx = column_index_from_string(end1_col)
            start2_col_idx = column_index_from_string(start2_col)
            end2_col_idx = column_index_from_string(end2_col)

            # Check row interval overlap
            rows_overlap = not (end1_row < start2_row or end2_row < start1_row)

            # Check column interval overlap
            cols_overlap = not (end1_col_idx < start2_col_idx or end2_col_idx < start1_col_idx)

            # Both intervals must overlap
            return rows_overlap and cols_overlap

        except Exception as e:
            self.logger.main.debug(f"Error checking range overlap: {e}")
            return False

    def _emit(self) -> dict[str, Any]:
        """S5: Emit the final cartographer map."""
        self.logger.main.info("📤 S5: EMIT - Generating cartographer map")

        cartographer_map = {
            "sheet": self.worksheet.title,
            "analyzedAt": datetime.now().isoformat() + "Z",
            "blocks": [asdict(block) for block in self.blocks],
            "objects": [asdict(obj) for obj in self.objects],
            "unresolved": self.unresolved,
        }

        # Save results
        self.logger.save_results(cartographer_map)

        return cartographer_map


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sheet-Cartographer Agent (OpenAI) - Map Excel worksheet topology using OpenAI models"
    )
    parser.add_argument("excel_path", type=Path, help="Path to Excel file")
    parser.add_argument("--sheet-index", type=int, default=0, help="Sheet index to analyze (default: 0)")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Validate file exists
    if not args.excel_path.exists():
        print(f"❌ Error: File not found: {args.excel_path}")
        sys.exit(1)

    # Initialize logger with hierarchical output support
    logger = ExperimentLogger(
        module_path=__file__, model_name=args.model, excel_path=args.excel_path, sheet_index=args.sheet_index
    )

    # Log configuration
    logger.main.info("🎯 CONFIGURATION:")
    logger.main.info(f"   Excel file: {args.excel_path}")
    logger.main.info(f"   Sheet index: {args.sheet_index}")
    logger.main.info(f"   Model: {args.model}")
    logger.main.info(f"   Verbose: {args.verbose}")

    try:
        # Create and run cartographer
        cartographer = SheetCartographer(
            excel_path=args.excel_path, sheet_index=args.sheet_index, model=args.model, logger=logger
        )

        result = cartographer.run()

        # Print summary
        print("\n✅ ANALYSIS COMPLETE")
        print(f"📊 Detected {len(result['blocks'])} blocks")
        print(f"🎯 Detected {len(result['objects'])} objects")

        if args.verbose:
            print("\n📋 BLOCKS DETECTED:")
            for block in result["blocks"]:
                # Handle both old and new field names for transition
                block_type = block.get("structural_type", block.get("type", "Unknown"))
                # Extract enum value if it's a dict
                if isinstance(block_type, dict):
                    block_type = block_type.get("value", str(block_type))
                semantic = block.get("semantic_description", "")
                print(f"  - {block['id']}: {block_type} at {block['range']} (conf: {block['confidence']:.2f})")
                if semantic:
                    print(f"    → {semantic}")

        print(f"\n📁 Results saved to: {logger.outputs_dir}")

    except Exception as e:
        logger.error.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)

    finally:
        logger.finalize()


if __name__ == "__main__":
    main()
