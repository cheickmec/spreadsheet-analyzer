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
    """Detected block in the spreadsheet."""

    id: str
    range: str
    type: str
    confidence: float
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

    def __init__(self, worksheet: Worksheet, logger: ExperimentLogger):
        self.worksheet = worksheet
        self.logger = logger
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

        # Check if formula
        is_formula = False
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
                # In newer openpyxl versions, defined_names is a dict-like object
                for name, defn in self.worksheet.parent.defined_names.items():
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
        lines = range_text.split("\n")[3:]  # Skip headers
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
        self.tool_handler = ExcelToolHandler(self.worksheet, logger)
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
            density_row = []
            for col_start in range(1, cols + 1, self.sweep_step_w):
                # Grab window
                window_text = self.tool_handler.execute_tool(
                    "window_grab",
                    {
                        "top": row_start,
                        "left": col_start,
                        "height": min(window_height, rows - row_start + 1),
                        "width": min(window_width, cols - col_start + 1),
                        "format": "markdown",
                    },
                )

                # Calculate density (count cells with actual values)
                lines = window_text.split("\n")[3:]  # Skip headers
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
                    f"   Window [{row_start}:{min(row_start + window_height - 1, rows)}, "
                    f"{col_start}:{min(col_start + window_width - 1, cols)}] "
                    f"non_empty: {non_empty}/{total}, density: {density:.3f}"
                )

                # Adaptive window sizing
                if density > HIGH_DENSITY_THRESHOLD:
                    window_height = 20
                    window_width = 10

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
            self.logger.main.info(f"   Processing candidate {idx + 1}/{len(candidate_blocks)}: {candidate['range']}")

            # First, recursively split until homogeneous
            sub_regions = self._split_until_homogeneous(candidate["range"])

            self.logger.main.info(f"   Split into {len(sub_regions)} homogeneous regions")

            # Now classify each homogeneous sub-region
            for sub_region in sub_regions:
                # Skip very sparse regions
                if "text" in sub_region:
                    lines = sub_region["text"].split("\n")[3:]  # Skip headers
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

                # Initial classification
                result = self._classify_block_with_llm(sub_region["text"], sub_region["range"])

                # Handle the extended return value that may include peek requests
                if len(result) == 4:  # New format with peek_requests
                    block_type, confidence, open_questions, peek_requests = result
                else:  # Old format without peek_requests
                    block_type, confidence, open_questions = result
                    peek_requests = []

                # Force low confidence if embedded totals detected
                if has_embedded_totals and confidence > 0.5:
                    self.logger.main.info(f"   Detected embedded totals in {sub_region['range']} - forcing split")
                    confidence = 0.4  # Force splitting

                # Iterative refinement based on open questions and peek requests
                refinement_count = 0
                max_refinements = 3
                peek_requests = []  # Track initial peek requests
                peeked_ranges = set()  # Avoid duplicate peeks

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

                    # Handle the extended return value that may include peek requests
                    if len(result) == 4:  # New format with peek_requests
                        block_type, confidence, open_questions, peek_requests = result
                    else:  # Old format without peek_requests
                        block_type, confidence, open_questions = result
                        peek_requests = []

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
                        # Extract just type and confidence, ignore open questions
                        deep_type = deep_result[0]
                        deep_conf = deep_result[1]

                        self.blocks.append(
                            Block(
                                id=f"blk_{block_id:02d}",
                                range=deep_region["range"],
                                type=deep_type,
                                confidence=deep_conf,
                            )
                        )
                        self.logger.main.info(
                            f"   Block {block_id}: {deep_region['range']} -> {deep_type} (conf: {deep_conf:.2f})"
                        )
                        block_id += 1
                else:
                    # Add the classified block
                    self.blocks.append(
                        Block(
                            id=f"blk_{block_id:02d}", range=sub_region["range"], type=block_type, confidence=confidence
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

        # Try to split by columns first (often separates data from summary)
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
                lines = range_text.split("\n")[3:]  # Skip headers
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

    def _analyze_probes(
        self, probes: list[tuple[str, str, str]], full_range: str, total_rows: int
    ) -> tuple[str, float]:
        """Analyze multiple probes to determine block type with consensus and heuristics."""

        classifications = []
        has_totals = False
        blank_tail_ratio = 0.0

        for probe_position, probe_text, probe_range in probes:
            # Check for totals keywords
            probe_lower = probe_text.lower()
            if any(keyword in probe_lower for keyword in ["total", "grand total", "subtotal", "sum"]):
                has_totals = True

            # Count blank rows in tail probe
            if probe_position == "tail":
                lines = probe_text.split("\n")[3:]  # Skip headers
                blank_rows = sum(1 for line in lines if all(cell.strip() in ["", "␀"] for cell in line.split("|")[1:]))
                blank_tail_ratio = blank_rows / len(lines) if lines else 0

            # Classify this probe
            block_type, confidence = self._classify_block_with_llm(probe_text, probe_range)
            classifications.append((probe_position, block_type, confidence))

        # Determine consensus
        types_found = {cls[1] for cls in classifications}

        # If we found totals, prioritize SUMMARY classification
        if has_totals:
            # Check if it's a pure summary section or mixed
            if len(probes) == 1 or all(c[1] in ["HeaderBanner", "KPISummary", "Other"] for c in classifications):
                return "Summary", SUMMARY_CONFIDENCE
            else:
                # Mixed region - return low confidence to trigger splitting
                return "Other", 0.4

        # If multiple different types detected, return low confidence to trigger splitting
        if len(types_found) > 1:
            self.logger.main.debug(f"Multiple types detected: {types_found} - triggering split")
            return "Other", 0.4

        # Get the unanimous type
        unanimous_type = classifications[0][1] if classifications else "Other"
        base_confidence = (
            sum(c[2] for c in classifications) / len(classifications) if classifications else UNKNOWN_CONFIDENCE
        )

        # Apply blank-tail penalty for FactTable
        if unanimous_type == "FactTable" and blank_tail_ratio > BLANK_TAIL_THRESHOLD:
            self.logger.main.debug(f"Applying blank-tail penalty: {blank_tail_ratio:.2f}")
            base_confidence = max(base_confidence - 0.2, 0.5)

        return unanimous_type, base_confidence

    def _classify_block_with_llm(
        self, range_text: str, range_str: str, additional_context: str | None = None
    ) -> tuple[str, float, list[str]]:
        """Use LLM to classify a block based on its content.

        Returns:
            Tuple of (block_type, confidence, open_questions)
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
                                    "FactTable",
                                    "HeaderBanner",
                                    "KPISummary",
                                    "PivotCache",
                                    "ChartAnchor",
                                    "Summary",
                                    "Other",
                                ],
                                "description": "The type of block detected (if heterogeneous, split further)",
                            },
                            "confidence": {"type": "number", "description": "Confidence score between 0 and 1"},
                            "reasoning": {"type": "string", "description": "Brief explanation for the classification"},
                            "open_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Questions that need answers to improve classification (e.g., 'Are rows 1-3 headers?', 'Is row 10 a totals row?')",
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
                        "required": ["block_type", "confidence", "reasoning", "open_questions"],
                    },
                },
            }
        ]

        # Combine global metadata with classification instructions
        system_prompt = f"""{self.system_metadata}

CLASSIFICATION TYPES:
- FactTable: Has header rows (often bold), multiple data rows, numeric columns, >60% density
- HeaderBanner: Single row, often merged across columns, bold/centered text
- KPISummary: Small (≤5 rows), text+number pairs, surrounded by whitespace
- PivotCache: Contains "Grand Total" or uniform formulas
- ChartAnchor: Range containing chart objects
- Summary: Contains totals, aggregations, or summary statistics (look for keywords: total, sum, subtotal)
- Other: Doesn't clearly fit other categories

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
                )

            # Extract classification
            if response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)

                # Return with peek_requests if available
                peek_requests = args.get("peek_requests", [])
                if peek_requests:
                    return (args["block_type"], args["confidence"], args.get("open_questions", []), peek_requests)
                else:
                    return (args["block_type"], args["confidence"], args.get("open_questions", []))

        except Exception as e:
            self.logger.error.error(f"LLM classification failed: {e}")

        # Fallback classification
        return "Other", UNKNOWN_CONFIDENCE, []

    def _overlay(self) -> None:
        """S4: Add merged ranges, named ranges, charts/pivots."""
        self.logger.main.info("🎨 S4: OVERLAY - Adding special elements")

        # Get merged ranges
        merged = self.tool_handler.execute_tool("merged_list", {})
        self.logger.main.info(f"   Found {len(merged)} merged ranges")

        # Get named ranges
        named = self.tool_handler.execute_tool("names_list", {})
        self.logger.main.info(f"   Found {len(named)} named ranges")

        # Match named ranges to blocks
        for name_info in named:
            for block in self.blocks:
                if self._ranges_overlap(name_info.get("range", ""), block.range):
                    block.named_range = name_info["name"]
                    break

        # Get charts/pivots
        charts = self.tool_handler.execute_tool("object_scan", {"type": "chart"})
        for idx, chart in enumerate(charts):
            self.objects.append(
                ChartObject(id=f"cht_{idx + 1:02d}", anchor=chart.get("anchor", ""), type="chart", linked_block=None)
            )

    def _ranges_overlap(self, range1: str, range2: str) -> bool:
        """Check if two A1-style ranges overlap.

        Two ranges overlap if their row intervals AND column intervals both intersect.
        """
        try:
            # Parse first range
            if ":" in range1:
                parts1 = range1.split(":")
                start1_col, start1_row = coordinate_from_string(parts1[0])
                end1_col, end1_row = coordinate_from_string(parts1[1])
            else:
                # Single cell
                start1_col, start1_row = coordinate_from_string(range1)
                end1_col, end1_row = start1_col, start1_row

            # Parse second range
            if ":" in range2:
                parts2 = range2.split(":")
                start2_col, start2_row = coordinate_from_string(parts2[0])
                end2_col, end2_row = coordinate_from_string(parts2[1])
            else:
                # Single cell
                start2_col, start2_row = coordinate_from_string(range2)
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

    # Initialize logger
    logger = ExperimentLogger(module_path=__file__)

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
                print(f"  - {block['id']}: {block['type']} at {block['range']} (confidence: {block['confidence']:.2f})")

        print(f"\n📁 Results saved to: {logger.outputs_dir}")

    except Exception as e:
        logger.error.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)

    finally:
        logger.finalize()


if __name__ == "__main__":
    main()
