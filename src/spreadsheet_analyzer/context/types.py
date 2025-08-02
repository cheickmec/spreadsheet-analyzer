"""Type definitions for context management.

This module defines protocols and types for the context management
system, enabling composable strategies for handling large contexts.

CLAUDE-KNOWLEDGE: Context management is critical for LLMs with
limited context windows. These types enable flexible composition
of different strategies.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.errors import ContextError
from ..core.types import Err, Result, ok


@dataclass(frozen=True)
class ContextCell:
    """Immutable representation of a spreadsheet cell in context."""

    location: str  # e.g., "Sheet1!A1"
    content: Any
    cell_type: str  # "value", "formula", "empty", etc.
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Estimate tokens for this cell."""
        # Simplified estimation
        content_str = str(self.content) if self.content else ""
        metadata_str = str(self.metadata) if self.metadata else ""
        return len(f"{self.location}: {content_str} {metadata_str}") // 4


@dataclass(frozen=True)
class ContextPackage:
    """Immutable package of context information."""

    cells: tuple[ContextCell, ...]
    metadata: dict[str, Any]
    focus_hints: tuple[str, ...]
    token_count: int
    compression_method: str | None = None

    @classmethod
    def create(
        cls,
        cells: list[ContextCell],
        metadata: dict[str, Any] | None = None,
        focus_hints: list[str] | None = None,
        compression_method: str | None = None,
    ) -> "ContextPackage":
        """Create a context package with calculated token count."""
        cells_tuple = tuple(cells)
        token_count = sum(cell.token_estimate for cell in cells_tuple)

        # Add metadata tokens
        if metadata:
            token_count += len(str(metadata)) // 4

        return cls(
            cells=cells_tuple,
            metadata=metadata or {},
            focus_hints=tuple(focus_hints or []),
            token_count=token_count,
            compression_method=compression_method,
        )

    def with_cells(self, cells: list[ContextCell]) -> "ContextPackage":
        """Create new package with different cells."""
        return ContextPackage.create(
            cells=cells,
            metadata=self.metadata,
            focus_hints=list(self.focus_hints),
            compression_method=self.compression_method,
        )

    def with_metadata(self, **kwargs: Any) -> "ContextPackage":
        """Create new package with updated metadata."""
        from dataclasses import replace

        new_metadata = {**self.metadata, **kwargs}
        return replace(self, metadata=new_metadata)


class ContextStrategy(Protocol):
    """Protocol for context management strategies.

    Strategies transform context packages to optimize them
    for LLM consumption while preserving important information.
    """

    @property
    def name(self) -> str:
        """Get the strategy name."""
        ...

    def apply(self, package: ContextPackage, token_budget: int) -> Result[ContextPackage, ContextError]:
        """Apply the strategy to a context package.

        This should be a pure function that transforms the package
        without side effects.
        """
        ...


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for a context strategy."""

    name: str
    enabled: bool = True
    priority: int = 0  # Higher priority strategies run first
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressionMetrics:
    """Metrics about context compression."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    cells_processed: int
    patterns_detected: int
    time_elapsed_ms: float
    strategies_applied: tuple[str, ...]

    @property
    def reduction_percentage(self) -> float:
        """Calculate percentage reduction in tokens."""
        if self.original_tokens == 0:
            return 0.0
        return (1 - self.compressed_tokens / self.original_tokens) * 100


@dataclass(frozen=True)
class ContextQuery:
    """Query for relevant context."""

    query_text: str
    sheet_names: tuple[str, ...] | None = None
    cell_ranges: tuple[str, ...] | None = None
    include_formulas: bool = True
    include_values: bool = True
    max_cells: int | None = None
    relevance_threshold: float = 0.5


@dataclass(frozen=True)
class PatternInfo:
    """Information about a detected pattern."""

    pattern_type: str  # "formula", "value", "format", etc.
    pattern_value: str
    locations: tuple[str, ...]
    frequency: int
    importance: float

    @property
    def is_significant(self) -> bool:
        """Check if pattern is significant enough to compress."""
        return self.frequency >= 3 and self.importance > 0.5


@dataclass(frozen=True)
class RangeInfo:
    """Information about a cell range."""

    start_cell: str
    end_cell: str
    cell_count: int
    range_type: str  # "data", "formula", "empty", "mixed"
    summary: str
    samples: tuple[Any, ...]

    @property
    def is_compressible(self) -> bool:
        """Check if range can be compressed."""
        return self.cell_count > 5 and self.range_type != "mixed"


# Strategy composition types
@dataclass(frozen=True)
class StrategyChain:
    """Chain of strategies to apply in sequence."""

    strategies: tuple[ContextStrategy, ...]

    def apply(self, package: ContextPackage, token_budget: int) -> Result[ContextPackage, ContextError]:
        """Apply all strategies in sequence."""
        current_package = package

        for strategy in self.strategies:
            result = strategy.apply(current_package, token_budget)
            if isinstance(result, Err):
                return result
            current_package = result.value

            # Stop if we're within budget
            if current_package.token_count <= token_budget:
                break

        return ok(current_package)


class ContextManager(Protocol):
    """Protocol for context management systems."""

    def prepare_context(
        self, cells: list[ContextCell], query: ContextQuery, token_budget: int
    ) -> Result[ContextPackage, ContextError]:
        """Prepare optimized context for LLM."""
        ...

    def get_metrics(self) -> CompressionMetrics:
        """Get metrics from last compression."""
        ...
