"""Type definitions for table detection agent.

This module defines immutable types for table boundary detection,
following functional programming principles.

CLAUDE-KNOWLEDGE: Tables in Excel can be separated by empty rows/columns
(mechanical) or by changes in entity types (semantic). Both methods are
important for accurate detection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

# Constants following PLR2004 pattern (avoid magic numbers)
EMPTY_ROW_THRESHOLD: Final[int] = 2  # Number of consecutive empty rows to consider as table separator
MIN_TABLE_ROWS: Final[int] = 3  # Minimum rows to consider as a valid table
SEMANTIC_CONFIDENCE_THRESHOLD: Final[float] = 0.8  # Confidence threshold for semantic detection
HEADER_PATTERN_CONFIDENCE: Final[float] = 0.7  # Confidence for header pattern matching
MAX_EMPTY_CELL_RATIO: Final[float] = 0.9  # Maximum ratio of empty cells in a valid table


class TableType(Enum):
    """Types of tables that can be detected in spreadsheets."""

    DETAIL = "detail"  # Detailed data rows (orders, transactions, etc.)
    SUMMARY = "summary"  # Aggregated or summary data
    HEADER = "header"  # Header/metadata section (invoice headers, etc.)
    PIVOT = "pivot"  # Pivot table structure
    LOOKUP = "lookup"  # Reference/lookup table
    UNKNOWN = "unknown"  # Cannot determine type


@dataclass(frozen=True)
class TableBoundary:
    """Immutable representation of a detected table's boundaries.

    This class defines the exact location and characteristics of a table
    within a spreadsheet.

    Attributes:
        table_id: Unique identifier for the table
        description: Human-readable description of table contents
        start_row: Zero-indexed starting row
        end_row: Zero-indexed ending row (inclusive)
        start_col: Zero-indexed starting column
        end_col: Zero-indexed ending column (inclusive)
        confidence: Detection confidence score (0.0 to 1.0)
        table_type: Categorization of the table
        entity_type: Business entity type (e.g., "orders", "customers")

    Example:
        >>> boundary = TableBoundary(
        ...     table_id="table_1",
        ...     description="Customer orders from Q1 2024",
        ...     start_row=0,
        ...     end_row=99,
        ...     start_col=0,
        ...     end_col=5,
        ...     confidence=0.95,
        ...     table_type=TableType.DETAIL,
        ...     entity_type="orders"
        ... )
    """

    table_id: str
    description: str
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    confidence: float
    table_type: TableType
    entity_type: str

    def __post_init__(self) -> None:
        """Validate boundary constraints."""
        if self.start_row < 0 or self.start_col < 0:
            raise ValueError("Table boundaries cannot be negative")
        if self.end_row < self.start_row:
            raise ValueError("End row must be >= start row")
        if self.end_col < self.start_col:
            raise ValueError("End column must be >= start column")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1")

    @property
    def row_count(self) -> int:
        """Number of rows in the table."""
        return self.end_row - self.start_row + 1

    @property
    def col_count(self) -> int:
        """Number of columns in the table."""
        return self.end_col - self.start_col + 1

    @property
    def cell_count(self) -> int:
        """Total number of cells in the table."""
        return self.row_count * self.col_count


@dataclass(frozen=True)
class DetectionMetrics:
    """Metrics from the detection process."""

    empty_row_blocks: tuple[tuple[int, int], ...]  # (start_row, end_row) pairs
    empty_col_blocks: tuple[tuple[int, int], ...]  # (start_col, end_col) pairs
    header_rows_detected: tuple[int, ...]  # Row indices identified as headers
    semantic_shifts: tuple[int, ...]  # Row indices where entity type changes
    confidence_scores: dict[str, float]  # Confidence for different detection methods


@dataclass(frozen=True)
class TableDetectionResult:
    """Immutable result from table detection process.

    This class contains all detected tables and metadata about the
    detection process.

    Attributes:
        sheet_name: Name of the analyzed sheet
        tables: Tuple of detected table boundaries
        detection_method: Method used ("mechanical", "semantic", "hybrid")
        metadata: Additional information about detection process
        metrics: Detailed metrics from detection

    CLAUDE-KNOWLEDGE: The detection_method indicates which approach was
    primary in identifying tables. "hybrid" means both mechanical and
    semantic detection contributed significantly.
    """

    sheet_name: str
    tables: tuple[TableBoundary, ...]
    detection_method: str
    metadata: dict[str, Any]
    metrics: DetectionMetrics | None = None

    def __post_init__(self) -> None:
        """Validate detection result."""
        if self.detection_method not in ["mechanical", "semantic", "hybrid"]:
            raise ValueError(f"Invalid detection method: {self.detection_method}")

        # Check for overlapping tables
        for i, table1 in enumerate(self.tables):
            for table2 in self.tables[i + 1 :]:
                if self._tables_overlap(table1, table2):
                    raise ValueError(f"Tables {table1.table_id} and {table2.table_id} overlap")

    @staticmethod
    def _tables_overlap(t1: TableBoundary, t2: TableBoundary) -> bool:
        """Check if two tables have overlapping boundaries."""
        row_overlap = not (t1.end_row < t2.start_row or t2.end_row < t1.start_row)
        col_overlap = not (t1.end_col < t2.start_col or t2.end_col < t1.start_col)
        return row_overlap and col_overlap

    def get_table_by_id(self, table_id: str) -> TableBoundary | None:
        """Find a table by its ID."""
        for table in self.tables:
            if table.table_id == table_id:
                return table
        return None

    def get_tables_by_type(self, table_type: TableType) -> tuple[TableBoundary, ...]:
        """Get all tables of a specific type."""
        return tuple(t for t in self.tables if t.table_type == table_type)

    def get_primary_table(self) -> TableBoundary | None:
        """Get the largest table by cell count (likely the main data table)."""
        if not self.tables:
            return None
        return max(self.tables, key=lambda t: t.cell_count)
