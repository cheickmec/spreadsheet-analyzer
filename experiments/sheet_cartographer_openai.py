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
DENSITY_THRESHOLD: Final[float] = 0.10  # 10% threshold to catch sparse regions
HIGH_DENSITY_THRESHOLD: Final[float] = 0.8  # 80% density reduces window size
BLANK_TAIL_THRESHOLD: Final[float] = 0.3  # 30% blank tail triggers split, not penalty
HOMOGENEITY_THRESHOLD: Final[float] = 0.9  # 90% homogeneity required to classify
ENTROPY_THRESHOLD: Final[float] = 0.1  # Max entropy for homogeneous region

# Block type confidence thresholds
FACT_TABLE_CONFIDENCE: Final[float] = 0.75
HEADER_BANNER_CONFIDENCE: Final[float] = 0.70
KPI_SUMMARY_CONFIDENCE: Final[float] = 0.65
PIVOT_CACHE_CONFIDENCE: Final[float] = 0.60
CHART_ANCHOR_CONFIDENCE: Final[float] = 1.00
SUMMARY_CONFIDENCE: Final[float] = 0.80  # For totals sections
COMPOSITE_CONFIDENCE: Final[float] = 0.50  # For mixed regions
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
        # Simplified - openpyxl doesn't easily expose charts/pivots
        # Return empty for now
        return []

    def calculate_layout_homogeneity(self, range_text: str) -> float:
        """Calculate layout homogeneity based on row patterns.

        Returns a value between 0 (heterogeneous) and 1 (homogeneous).
        Based on the entropy of non-blank cell distribution per row.
        """
        lines = range_text.split("\n")[3:]  # Skip headers
        if not lines:
            return 1.0

        row_densities = []
        for line in lines:
            cells = line.split("|")[1:]  # Skip row number
            if not cells:
                continue
            non_empty = sum(1 for cell in cells if cell.strip() and cell.strip() != "␀")
            density = non_empty / len(cells) if cells else 0
            row_densities.append(density)

        if not row_densities:
            return 1.0

        # Calculate entropy of row densities
        # Lower entropy = more homogeneous
        if len(set(row_densities)) == 1:
            return 1.0  # All rows have same density

        # Normalize densities to discrete bins
        bins = np.linspace(0, 1, 11)  # 10 bins
        digitized = np.digitize(row_densities, bins)

        # Calculate entropy
        _, counts = np.unique(digitized, return_counts=True)
        probabilities = counts / len(digitized)
        entropy = stats.entropy(probabilities, base=2)
        max_entropy = np.log2(len(bins))

        # Convert entropy to homogeneity (inverse relationship)
        homogeneity = 1.0 - (entropy / max_entropy if max_entropy > 0 else 0)

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
        self.workbook = openpyxl.load_workbook(excel_path, read_only=False, data_only=True)
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
        return sheet_info

    def _sweep(self, sheet_info: dict[str, Any]) -> list[list[float]]:
        """S1: Sweep across sheet recording density."""
        self.logger.main.info("🔍 S1: SWEEP - Scanning sheet density")

        # Initialize density grid
        rows = sheet_info["rows"]
        cols = sheet_info["cols"]

        # Use adaptive window size
        window_height = 40
        window_width = 20

        density_grid = []

        # Sweep row-major with sliding windows
        for row_start in range(1, rows + 1, window_height // 2):
            density_row = []
            for col_start in range(1, cols + 1, window_width // 2):
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

                    # Convert grid coordinates to sheet coordinates
                    window_height = 20  # Simplified
                    window_width = 10

                    start_row = min_i * (window_height // 2) + 1
                    end_row = min((max_i + 1) * (window_height // 2) + window_height // 2, sheet_info["rows"])
                    start_col = min_j * (window_width // 2) + 1
                    end_col = min((max_j + 1) * (window_width // 2) + window_width // 2, sheet_info["cols"])

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

                # Initial classification
                block_type, confidence, open_questions = self._classify_block_with_llm(
                    sub_region["text"], sub_region["range"]
                )

                # Iterative refinement based on open questions
                refinement_count = 0
                max_refinements = 3

                while open_questions and refinement_count < max_refinements:
                    self.logger.main.debug(f"   Refinement {refinement_count + 1} for {sub_region['range']}")
                    self.logger.main.debug(f"   Open questions: {open_questions}")

                    # Address open questions with targeted probing
                    additional_context = self._address_open_questions(
                        open_questions, sub_region["range"], sub_region["text"]
                    )

                    # Re-classify with additional context
                    block_type, confidence, open_questions = self._classify_block_with_llm(
                        sub_region["text"], sub_region["range"], additional_context=additional_context
                    )

                    refinement_count += 1

                # Add the classified block
                self.blocks.append(
                    Block(id=f"blk_{block_id:02d}", range=sub_region["range"], type=block_type, confidence=confidence)
                )

                self.logger.main.info(
                    f"   Block {block_id}: {sub_region['range']} -> "
                    f"{block_type} (conf: {confidence:.2f}, homogeneity: {sub_region.get('homogeneity', 0):.2f})"
                )

                block_id += 1

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
                return "Composite", COMPOSITE_CONFIDENCE

        # If multiple different types detected, mark as Composite
        if len(types_found) > 1:
            self.logger.main.debug(f"Multiple types detected: {types_found}")
            return "Composite", COMPOSITE_CONFIDENCE

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

        # Define tools for classification with open questions
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
                                    "Composite",
                                    "Other",
                                ],
                                "description": "The type of block detected",
                            },
                            "confidence": {"type": "number", "description": "Confidence score between 0 and 1"},
                            "reasoning": {"type": "string", "description": "Brief explanation for the classification"},
                            "open_questions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Questions that need answers to improve classification (e.g., 'Are rows 1-3 headers?', 'Is row 10 a totals row?')",
                            },
                        },
                        "required": ["block_type", "confidence", "reasoning", "open_questions"],
                    },
                },
            }
        ]

        system_prompt = """You are a spreadsheet structure analyzer. Classify the given block of cells into one of these types:

- FactTable: Has header rows (often bold), multiple data rows, numeric columns, >60% density
- HeaderBanner: Single row, often merged across columns, bold/centered text
- KPISummary: Small (≤5 rows), text+number pairs, surrounded by whitespace
- PivotCache: Contains "Grand Total" or uniform formulas
- ChartAnchor: Range containing chart objects
- Summary: Contains totals, aggregations, or summary statistics (look for keywords: total, sum, subtotal)
- Composite: Mixed region with multiple semantic types
- Other: Doesn't clearly fit other categories

Analyze the structure, not the content. Focus on layout patterns.

When uncertain, provide specific open questions that would help clarify the classification.
For example:
- "Are rows 1-2 headers for the data below?"
- "Is row 10 a totals row?"
- "Does the sparse area at bottom indicate end of data?"
"""

        user_content = f"Classify this spreadsheet block at range {range_str}:\n\n{range_text}"

        if additional_context:
            user_content += f"\n\nAdditional context from refinement:\n{additional_context}"

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]

        try:
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
        """Check if two ranges overlap (simplified)."""
        # Very simplified overlap check
        return False  # TODO: Implement proper range overlap

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
