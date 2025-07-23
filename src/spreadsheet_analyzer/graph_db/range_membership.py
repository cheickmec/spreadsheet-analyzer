"""
Range membership tracking for efficient cell-to-range queries.

This module provides efficient tracking of which cells belong to which ranges,
enabling queries like "Is cell B500 part of any formula range?" even when
B500 is empty.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class RangeBounds:
    """Represents the boundaries of a cell range."""

    sheet: str
    start_row: int
    start_col: int
    end_row: int
    end_col: int

    def contains(self, row: int, col: int) -> bool:
        """Check if a cell is within this range."""
        return self.start_row <= row <= self.end_row and self.start_col <= col <= self.end_col

    def intersects(self, other: "RangeBounds") -> bool:
        """Check if two ranges intersect."""
        return not (
            self.end_row < other.start_row
            or self.start_row > other.end_row
            or self.end_col < other.start_col
            or self.start_col > other.end_col
        )


@dataclass
class RangeReference:
    """A range reference with its associated formula."""

    range_ref: str
    formula_key: str
    bounds: RangeBounds
    total_cells: int


def col_to_num(col: str) -> int:
    """Convert Excel column letter to number (A=1, B=2, etc)."""
    result = 0
    for char in col:
        result = result * 26 + (ord(char.upper()) - ord("A") + 1)
    return result


def num_to_col(num: int) -> str:
    """Convert column number to Excel letter (1=A, 2=B, etc)."""
    result = ""
    while num > 0:
        num -= 1
        result = chr(num % 26 + ord("A")) + result
        num //= 26
    return result


def parse_cell_ref(cell_ref: str) -> tuple[str, int, int]:
    """Parse a cell reference like 'Sheet1!A1' or 'A1'."""
    # Handle sheet references
    sheet = ""
    if "!" in cell_ref:
        sheet, cell_ref = cell_ref.split("!", 1)
        # Remove quotes if present
        sheet = sheet.strip("'")

    # Parse column and row
    match = re.match(r"^([A-Z]+)(\d+)$", cell_ref.upper())
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")

    col_str, row_str = match.groups()
    return sheet, int(row_str), col_to_num(col_str)


def parse_range_ref(range_ref: str) -> tuple[str, RangeBounds]:
    """Parse a range reference like 'Sheet1!A1:B10' or 'A1:B10'."""
    # Handle sheet references
    sheet = ""
    if "!" in range_ref:
        sheet, range_ref = range_ref.split("!", 1)
        sheet = sheet.strip("'")

    # Split range
    if ":" not in range_ref:
        raise ValueError(f"Not a range reference: {range_ref}")

    start_cell, end_cell = range_ref.split(":", 1)

    # Parse start and end cells
    _, start_row, start_col = parse_cell_ref(start_cell)
    _, end_row, end_col = parse_cell_ref(end_cell)

    # Ensure proper ordering
    if start_row > end_row:
        start_row, end_row = end_row, start_row
    if start_col > end_col:
        start_col, end_col = end_col, start_col

    bounds = RangeBounds(sheet=sheet, start_row=start_row, start_col=start_col, end_row=end_row, end_col=end_col)

    return sheet, bounds


class RangeMembershipIndex:
    """
    Efficiently track which ranges contain which cells.

    Uses a grid-based approach for fast lookups while maintaining
    reasonable memory usage.
    """

    def __init__(self) -> None:
        """Initialize the range membership index."""
        # Map from sheet to list of range references
        self.ranges_by_sheet: dict[str, list[RangeReference]] = defaultdict(list)

        # Grid-based index for faster lookups (sheet -> row -> col -> set of range indices)
        # This is populated lazily
        self.grid_index: dict[str, dict[int, dict[int, set[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(set))
        )

        # Track if grid index needs rebuilding
        self.grid_dirty: set[str] = set()

    def add_range(self, range_ref: str, formula_key: str, sheet: str | None = None) -> None:
        """Add a range reference to the index."""
        # Parse the range
        parsed_sheet, bounds = parse_range_ref(range_ref)

        # Use provided sheet if range doesn't specify one
        if not parsed_sheet and sheet:
            parsed_sheet = sheet
            bounds.sheet = sheet
        elif not parsed_sheet:
            raise ValueError(f"No sheet specified for range: {range_ref}")

        # Calculate total cells
        total_cells = (bounds.end_row - bounds.start_row + 1) * (bounds.end_col - bounds.start_col + 1)

        # Create range reference
        range_obj = RangeReference(range_ref=range_ref, formula_key=formula_key, bounds=bounds, total_cells=total_cells)

        # Add to sheet's range list
        self.ranges_by_sheet[parsed_sheet].append(range_obj)

        # Mark grid as dirty for this sheet
        self.grid_dirty.add(parsed_sheet)

    def _rebuild_grid_index(self, sheet: str) -> None:
        """Rebuild the grid index for a sheet."""
        if sheet not in self.grid_dirty:
            return

        # Clear existing grid for this sheet
        self.grid_index[sheet].clear()

        # Rebuild from ranges
        for idx, range_ref in enumerate(self.ranges_by_sheet[sheet]):
            bounds = range_ref.bounds
            for row in range(bounds.start_row, bounds.end_row + 1):
                for col in range(bounds.start_col, bounds.end_col + 1):
                    self.grid_index[sheet][row][col].add(idx)

        # Mark as clean
        self.grid_dirty.discard(sheet)

    def get_ranges_containing_cell(self, sheet: str, cell_ref: str) -> list[str]:
        """Find all formula keys for ranges containing a specific cell."""
        # Parse cell reference
        _, row, col = parse_cell_ref(cell_ref)

        # Ensure grid index is up to date
        self._rebuild_grid_index(sheet)

        # Get range indices from grid
        range_indices = self.grid_index[sheet][row][col]

        # Convert indices to formula keys
        formula_keys = []
        for idx in range_indices:
            if idx < len(self.ranges_by_sheet[sheet]):
                formula_keys.append(self.ranges_by_sheet[sheet][idx].formula_key)

        return formula_keys

    def is_cell_in_any_range(self, sheet: str, cell_ref: str) -> bool:
        """Check if a cell is part of any formula range."""
        return len(self.get_ranges_containing_cell(sheet, cell_ref)) > 0

    def get_ranges_intersecting_range(self, sheet: str, start_cell: str, end_cell: str) -> list[tuple[str, str]]:
        """Find all ranges that intersect with a given range."""
        # Parse the query range
        _, query_bounds = parse_range_ref(f"{start_cell}:{end_cell}")
        query_bounds.sheet = sheet

        # Find intersecting ranges
        intersecting = []
        for range_ref in self.ranges_by_sheet[sheet]:
            if range_ref.bounds.intersects(query_bounds):
                intersecting.append((range_ref.formula_key, range_ref.range_ref))

        return intersecting

    def get_all_ranges(self) -> dict[str, list[dict[str, Any]]]:
        """Get all indexed ranges for debugging/inspection."""
        result = {}
        for sheet, ranges in self.ranges_by_sheet.items():
            result[sheet] = [
                {
                    "range_ref": r.range_ref,
                    "formula_key": r.formula_key,
                    "start": f"{num_to_col(r.bounds.start_col)}{r.bounds.start_row}",
                    "end": f"{num_to_col(r.bounds.end_col)}{r.bounds.end_row}",
                    "total_cells": r.total_cells,
                }
                for r in ranges
            ]
        return result
