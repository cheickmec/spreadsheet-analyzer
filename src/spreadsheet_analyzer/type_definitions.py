"""
Type definitions and protocols for the spreadsheet analyzer.

This module provides type aliases, protocols, and type guards to improve
type safety throughout the codebase with stricter mypy settings.

CLAUDE-KNOWLEDGE: Using protocols instead of concrete types allows for
more flexible and testable code while maintaining type safety.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Final,
    Generic,
    Literal,
    NewType,
    NotRequired,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeGuard,
    TypeVar,
    overload,
    runtime_checkable,
)

# ============================================================================
# TYPE VARIABLES
# ============================================================================

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

# ============================================================================
# BASIC TYPE ALIASES
# ============================================================================

# Excel-specific types
CellAddress = str  # e.g., "A1", "B2"
SheetName = str
Formula = str
CellRange = str  # e.g., "A1:B10"
WorkbookPath = Path

# Numeric types with meaning
RowIndex = NewType("RowIndex", int)  # 1-based row index
ColumnIndex = NewType("ColumnIndex", int)  # 1-based column index
CellCount = NewType("CellCount", int)
FormulaDepth = NewType("FormulaDepth", int)

# Risk and scoring types
RiskScore = NewType("RiskScore", int)  # 0-100
ComplexityScore = NewType("ComplexityScore", float)
QualityScore = NewType("QualityScore", int)  # 0-100

# ============================================================================
# TYPED DICTIONARIES
# ============================================================================


class CellMetadata(TypedDict):
    """Metadata for a spreadsheet cell."""

    row: int
    column: int
    data_type: str
    number_format: NotRequired[str]
    has_formula: bool
    has_comment: NotRequired[bool]
    is_merged: NotRequired[bool]


class FormulaStatistics(TypedDict):
    """Statistics about formula analysis."""

    total_formulas: int
    processed_cells: int
    skipped_ranges: int
    circular_reference_count: int
    volatile_formula_count: int
    external_reference_count: int
    unique_dependencies: int
    parser_cache_hits: NotRequired[int]
    parser_cache_misses: NotRequired[int]


class RangeMetadata(TypedDict):
    """Metadata for a cell range."""

    start_row: int
    start_col: int
    end_row: int
    end_col: int
    cell_count: int
    is_full_row: bool
    is_full_column: bool
    sheets: NotRequired[list[str]]


# ============================================================================
# PROTOCOLS
# ============================================================================


@runtime_checkable
class SupportsProgress(Protocol):
    """Protocol for objects that support progress reporting."""

    def report_progress(self, stage: str, progress: float, message: str, details: dict[str, Any] | None = None) -> None:
        """Report progress update."""
        ...


@runtime_checkable
class ExcelReadable(Protocol):
    """Protocol for Excel-like workbook objects."""

    @property
    def sheetnames(self) -> list[str]:
        """List of sheet names."""
        ...

    def __getitem__(self, key: str) -> Any:
        """Get worksheet by name."""
        ...

    def close(self) -> None:
        """Close the workbook."""
        ...


class CellProtocol(Protocol):
    """Protocol for Excel cell objects."""

    @property
    def value(self) -> Any:
        """Cell value."""
        ...

    @property
    def coordinate(self) -> str:
        """Cell coordinate (e.g., 'A1')."""
        ...

    @property
    def data_type(self) -> str:
        """Cell data type."""
        ...

    @property
    def row(self) -> int:
        """Row number (1-based)."""
        ...

    @property
    def column(self) -> int:
        """Column number (1-based)."""
        ...


class WorksheetProtocol(Protocol):
    """Protocol for Excel worksheet objects."""

    @property
    def title(self) -> str:
        """Worksheet title."""
        ...

    @property
    def max_row(self) -> int | None:
        """Maximum row with data."""
        ...

    @property
    def max_column(self) -> int | None:
        """Maximum column with data."""
        ...

    def iter_rows(
        self,
        min_row: int | None = None,
        max_row: int | None = None,
        min_col: int | None = None,
        max_col: int | None = None,
    ) -> Iterable[tuple[CellProtocol, ...]]:
        """Iterate over rows."""
        ...


@runtime_checkable
class Analyzer(Protocol[T_co]):
    """Protocol for analyzer classes."""

    def analyze(self, input_data: Any) -> T_co:
        """Perform analysis."""
        ...


# ============================================================================
# TYPE GUARDS
# ============================================================================


def is_valid_cell_address(value: str) -> TypeGuard[CellAddress]:
    """Check if string is a valid cell address."""
    import re

    pattern = r"^[A-Z]+[1-9]\d*$"
    return bool(re.match(pattern, value.upper()))


def is_valid_range(value: str) -> TypeGuard[CellRange]:
    """Check if string is a valid cell range."""
    import re

    pattern = r"^[A-Z]+[1-9]\d*:[A-Z]+[1-9]\d*$"
    return bool(re.match(pattern, value.upper()))


def is_risk_score(value: int) -> TypeGuard[RiskScore]:
    """Check if value is a valid risk score."""
    return 0 <= value <= 100


def is_formula(value: str) -> TypeGuard[Formula]:
    """Check if string is likely an Excel formula."""
    return value.startswith("=") and len(value) > 1


# ============================================================================
# FUNCTION SIGNATURES
# ============================================================================

ProgressCallback: TypeAlias = Callable[[str, float, str], None]
ValidationFunction: TypeAlias = Callable[[Any], bool]
TransformFunction: TypeAlias = Callable[[T], T]
FilterPredicate: TypeAlias = Callable[[T], bool]


# Result type for operations that can fail
@dataclass(frozen=True)
class Success(Generic[T]):
    """Successful result."""

    value: T


@dataclass(frozen=True)
class Failure:
    """Failed result."""

    error: str
    details: dict[str, Any] | None = None


Result: TypeAlias = Success[T] | Failure

# ============================================================================
# OVERLOADED FUNCTIONS
# ============================================================================


@overload
def get_cell_value(cell: CellProtocol, default: None = None) -> Any | None: ...


@overload
def get_cell_value(cell: CellProtocol, default: T) -> Any | T: ...


def get_cell_value(cell: CellProtocol, default: T | None = None) -> Any | T | None:
    """Get cell value with optional default."""
    return cell.value if cell.value is not None else default


# ============================================================================
# LITERAL TYPES
# ============================================================================

ProcessingStatus: TypeAlias = Literal["pending", "processing", "completed", "failed"]
RiskLevel: TypeAlias = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
EdgeType: TypeAlias = Literal[
    "DEPENDS_ON", "SUMS_OVER", "AVERAGES_OVER", "LOOKS_UP_IN", "CONDITIONALLY_USES", "REFERENCES"
]
NodeType: TypeAlias = Literal["cell", "range", "named_range", "external"]

# ============================================================================
# COMPLEX TYPE ALIASES
# ============================================================================

# Dependency graph types
DependencyGraph: TypeAlias = Mapping[str, Sequence[str]]
AdjacencyList: TypeAlias = dict[str, set[str]]
EdgeMetadataMap: TypeAlias = dict[tuple[str, str], dict[str, Any]]

# Analysis result types
AnalysisResult: TypeAlias = Result[Mapping[str, Any]]
ValidationResult: TypeAlias = Result[None]

# Configuration types
StageConfig: TypeAlias = Mapping[str, Any]
PipelineConfig: TypeAlias = Mapping[str, StageConfig]

# ============================================================================
# CONSTANTS WITH TYPES
# ============================================================================

MAX_EXCEL_ROWS: Final[int] = 1_048_576
MAX_EXCEL_COLUMNS: Final[int] = 16_384
MAX_SHEET_NAME_LENGTH: Final[int] = 31
EXCEL_EPOCH: Final[str] = "1900-01-01"

# Valid Excel file extensions
EXCEL_EXTENSIONS: Final[frozenset[str]] = frozenset({".xlsx", ".xlsm", ".xls", ".xlsb"})

# Volatile Excel functions
VOLATILE_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"NOW", "TODAY", "RAND", "RANDBETWEEN", "OFFSET", "INDIRECT", "INFO", "CELL"}
)

# ============================================================================
# UTILITY FUNCTIONS WITH PRECISE TYPES
# ============================================================================


def validate_row_index(row: int) -> RowIndex:
    """Validate and convert to RowIndex."""
    if not 1 <= row <= MAX_EXCEL_ROWS:
        raise ValueError(f"Invalid row index: {row}")
    return RowIndex(row)


def validate_column_index(col: int) -> ColumnIndex:
    """Validate and convert to ColumnIndex."""
    if not 1 <= col <= MAX_EXCEL_COLUMNS:
        raise ValueError(f"Invalid column index: {col}")
    return ColumnIndex(col)


def combine_results(results: Sequence[Result[T]]) -> Result[list[T]]:
    """Combine multiple results into a single result."""
    values: list[T] = []
    errors: list[str] = []

    for result in results:
        if isinstance(result, Success):
            values.append(result.value)
        else:
            errors.append(result.error)

    if errors:
        return Failure(error=f"Multiple failures: {'; '.join(errors)}", details={"errors": errors})

    return Success(values)
