"""Functional context management strategies.

This module implements pure functional strategies for managing and optimizing
context for LLM consumption.

CLAUDE-KNOWLEDGE: Different strategies are optimal for different scenarios:
- Sliding window: Good for sequential data analysis
- Summary-based: Good for large datasets with patterns
- Hybrid: Balances detail and overview
"""

from collections.abc import Callable
from dataclasses import dataclass

from ..core.errors import ContextError
from ..core.types import Result, ok
from .types import (
    ContextCell,
    ContextPackage,
    ContextStrategy,
    PatternInfo,
    RangeInfo,
)

# Strategy implementations


@dataclass(frozen=True)
class SlidingWindowStrategy:
    """Keep most recent/relevant cells within token budget.

    This strategy prioritizes cells by importance and recency,
    keeping as many as possible within the token budget.
    """

    name: str = "sliding_window"
    importance_threshold: float = 0.0
    prefer_recent: bool = True

    def apply(self, package: ContextPackage, token_budget: int) -> Result[ContextPackage, ContextError]:
        """Apply sliding window selection."""
        if package.token_count <= token_budget:
            return ok(package)

        # Sort cells by importance and recency
        sorted_cells = sorted(package.cells, key=lambda c: (c.importance, self._get_recency_score(c)), reverse=True)

        # Keep cells within budget
        selected_cells = []
        current_tokens = 0

        for cell in sorted_cells:
            if cell.importance < self.importance_threshold:
                continue

            if current_tokens + cell.token_estimate <= token_budget:
                selected_cells.append(cell)
                current_tokens += cell.token_estimate
            else:
                # Stop when budget exceeded
                break

        return ok(package.with_cells(selected_cells))

    def _get_recency_score(self, cell: ContextCell) -> float:
        """Get recency score from cell metadata."""
        if not self.prefer_recent:
            return 0.0

        # Extract row number from location (e.g., "Sheet1!A10" -> 10)
        try:
            location = cell.location.split("!")[-1]
            # Simple heuristic: extract number from location
            import re

            match = re.search(r"\d+", location)
            if match:
                row = int(match.group())
                # Normalize to 0-1 range (assuming max 10000 rows)
                return min(row / 10000.0, 1.0)
        except Exception:
            pass

        return 0.5  # Default middle score


@dataclass(frozen=True)
class PatternCompressionStrategy:
    """Compress repeated patterns to save tokens.

    Detects and compresses patterns like:
    - Repeated formulas
    - Sequential values
    - Consistent formatting
    """

    name: str = "pattern_compression"
    min_pattern_frequency: int = 3
    pattern_detectors: tuple[Callable, ...] = ()

    def apply(self, package: ContextPackage, token_budget: int) -> Result[ContextPackage, ContextError]:
        """Apply pattern compression."""
        if package.token_count <= token_budget:
            return ok(package)

        # Detect patterns
        patterns = self._detect_patterns(package.cells)

        # Group cells by pattern
        pattern_groups = self._group_by_patterns(package.cells, patterns)

        # Compress each pattern group
        compressed_cells = []

        for pattern, cells in pattern_groups.items():
            if pattern and len(cells) >= self.min_pattern_frequency:
                # Create compressed representation
                compressed = self._compress_pattern(pattern, cells)
                compressed_cells.append(compressed)
            else:
                # Keep individual cells if no significant pattern
                compressed_cells.extend(cells)

        # Create new package with compressed cells
        new_package = package.with_cells(compressed_cells)

        # Add compression metadata
        new_package = new_package.with_metadata(
            compression_patterns=len(patterns),
            original_cell_count=len(package.cells),
            compressed_cell_count=len(compressed_cells),
        )

        return ok(new_package)

    def _detect_patterns(self, cells: tuple[ContextCell, ...]) -> list[PatternInfo]:
        """Detect patterns in cells."""
        patterns = []

        # Formula patterns
        formula_patterns: dict[str, list[str]] = {}
        for cell in cells:
            if cell.cell_type == "formula":
                # Normalize formula (replace cell refs with placeholders)
                normalized = self._normalize_formula(str(cell.content))
                if normalized not in formula_patterns:
                    formula_patterns[normalized] = []
                formula_patterns[normalized].append(cell.location)

        # Create PatternInfo for significant patterns
        for pattern, locations in formula_patterns.items():
            if len(locations) >= self.min_pattern_frequency:
                patterns.append(
                    PatternInfo(
                        pattern_type="formula",
                        pattern_value=pattern,
                        locations=tuple(locations),
                        frequency=len(locations),
                        importance=0.8,
                    )
                )

        # Apply custom pattern detectors
        for detector in self.pattern_detectors:
            detected = detector(cells)
            patterns.extend(detected)

        return patterns

    def _normalize_formula(self, formula: str) -> str:
        """Normalize formula by replacing cell references."""
        import re

        # Replace cell references with placeholder
        return re.sub(r"[A-Z]+\d+", "CELL", formula)

    def _group_by_patterns(
        self, cells: tuple[ContextCell, ...], patterns: list[PatternInfo]
    ) -> dict[PatternInfo | None, list[ContextCell]]:
        """Group cells by their patterns."""
        groups: dict[PatternInfo | None, list[ContextCell]] = {None: []}  # None key for cells without patterns
        location_to_pattern: dict[str, PatternInfo] = {}

        # Map locations to patterns
        for pattern in patterns:
            for location in pattern.locations:
                location_to_pattern[location] = pattern
                if pattern not in groups:
                    groups[pattern] = []

        # Group cells
        for cell in cells:
            pattern = location_to_pattern.get(cell.location)
            if pattern and pattern in groups:
                groups[pattern].append(cell)
            else:
                groups[None].append(cell)

        return groups

    def _compress_pattern(self, pattern: PatternInfo, cells: list[ContextCell]) -> ContextCell:
        """Create compressed representation of pattern."""
        # Create summary cell
        locations_summary = f"{cells[0].location}:{cells[-1].location}"

        content = f"Pattern: {pattern.pattern_value} in {len(cells)} cells"

        return ContextCell(
            location=locations_summary,
            content=content,
            cell_type="pattern_summary",
            importance=max(c.importance for c in cells),
            metadata={
                "pattern_type": pattern.pattern_type,
                "cell_count": len(cells),
                "sample_values": [cells[0].content, cells[-1].content] if cells else [],
            },
        )


@dataclass(frozen=True)
class RangeAggregationStrategy:
    """Aggregate cell ranges into summaries.

    Compresses contiguous ranges of similar cells into
    summary representations.
    """

    name: str = "range_aggregation"
    min_range_size: int = 5
    aggregation_functions: dict[str, Callable] | None = None

    def __post_init__(self) -> None:
        if self.aggregation_functions is None:
            # Default aggregation functions
            object.__setattr__(
                self,
                "aggregation_functions",
                {
                    "numeric": lambda cells: {
                        "min": min(c.content for c in cells if isinstance(c.content, int | float)),
                        "max": max(c.content for c in cells if isinstance(c.content, int | float)),
                        "avg": sum(c.content for c in cells if isinstance(c.content, int | float)) / len(cells),
                    },
                    "text": lambda cells: {
                        "unique_values": len(set(c.content for c in cells)),
                        "sample": cells[0].content if cells else None,
                    },
                    "empty": lambda cells: {"count": len(cells)},
                },
            )

    def apply(self, package: ContextPackage, token_budget: int) -> Result[ContextPackage, ContextError]:
        """Apply range aggregation."""
        if package.token_count <= token_budget:
            return ok(package)

        # Detect ranges
        ranges = self._detect_ranges(package.cells)

        # Aggregate ranges
        aggregated_cells = []
        processed_locations = set()

        for range_info in ranges:
            if range_info.cell_count >= self.min_range_size:
                # Create aggregated cell
                aggregated = self._aggregate_range(range_info, package.cells)
                aggregated_cells.append(aggregated)

                # Mark cells as processed
                for cell in package.cells:
                    if self._in_range(cell.location, range_info):
                        processed_locations.add(cell.location)

        # Add non-aggregated cells
        for cell in package.cells:
            if cell.location not in processed_locations:
                aggregated_cells.append(cell)

        return ok(package.with_cells(aggregated_cells))

    def _detect_ranges(self, cells: tuple[ContextCell, ...]) -> list[RangeInfo]:
        """Detect contiguous ranges of similar cells."""
        # Sort cells by location for range detection
        sorted_cells = sorted(
            cells, key=lambda c: (self._get_sheet(c.location), self._get_row(c.location), self._get_col(c.location))
        )

        ranges = []
        current_range: list[ContextCell] = []
        current_type = None

        for cell in sorted_cells:
            cell_type = self._get_cell_type_category(cell)

            if not current_range or (cell_type == current_type and self._is_adjacent(current_range[-1], cell)):
                current_range.append(cell)
                current_type = cell_type
            else:
                # End current range and start new one
                if len(current_range) >= self.min_range_size:
                    ranges.append(self._create_range_info(current_range))

                current_range = [cell]
                current_type = cell_type

        # Don't forget last range
        if len(current_range) >= self.min_range_size:
            ranges.append(self._create_range_info(current_range))

        return ranges

    def _get_sheet(self, location: str) -> str:
        """Extract sheet name from location."""
        return location.split("!")[0] if "!" in location else "Sheet1"

    def _get_row(self, location: str) -> int:
        """Extract row number from location."""
        import re

        match = re.search(r"\d+", location.split("!")[-1])
        return int(match.group()) if match else 0

    def _get_col(self, location: str) -> str:
        """Extract column letter from location."""
        import re

        match = re.search(r"[A-Z]+", location.split("!")[-1])
        return match.group() if match else "A"

    def _get_cell_type_category(self, cell: ContextCell) -> str:
        """Categorize cell type for range detection."""
        if cell.cell_type == "empty":
            return "empty"
        elif cell.cell_type == "formula":
            return "formula"
        elif isinstance(cell.content, int | float):
            return "numeric"
        else:
            return "text"

    def _is_adjacent(self, cell1: ContextCell, cell2: ContextCell) -> bool:
        """Check if two cells are adjacent."""
        sheet1 = self._get_sheet(cell1.location)
        sheet2 = self._get_sheet(cell2.location)

        if sheet1 != sheet2:
            return False

        row1 = self._get_row(cell1.location)
        row2 = self._get_row(cell2.location)
        col1 = self._get_col(cell1.location)
        col2 = self._get_col(cell2.location)

        # Check if adjacent (same column, consecutive rows for now)
        return col1 == col2 and abs(row2 - row1) == 1

    def _create_range_info(self, cells: list[ContextCell]) -> RangeInfo:
        """Create RangeInfo from cell list."""
        range_type = self._get_cell_type_category(cells[0])

        # Get appropriate aggregation function
        agg_func = self.aggregation_functions.get(range_type, lambda x: {})
        summary_data = agg_func(cells)

        return RangeInfo(
            start_cell=cells[0].location,
            end_cell=cells[-1].location,
            cell_count=len(cells),
            range_type=range_type,
            summary=str(summary_data),
            samples=tuple(c.content for c in cells[:3]),  # First 3 samples
        )

    def _in_range(self, location: str, range_info: RangeInfo) -> bool:
        """Check if location is within range."""
        # Simplified check - would need proper range parsing in production
        return location >= range_info.start_cell and location <= range_info.end_cell

    def _aggregate_range(self, range_info: RangeInfo, cells: tuple[ContextCell, ...]) -> ContextCell:
        """Create aggregated cell for range."""
        return ContextCell(
            location=f"{range_info.start_cell}:{range_info.end_cell}",
            content=range_info.summary,
            cell_type="range_summary",
            importance=0.6,  # Medium importance for summaries
            metadata={
                "range_type": range_info.range_type,
                "cell_count": range_info.cell_count,
                "samples": list(range_info.samples),
            },
        )


@dataclass(frozen=True)
class HybridStrategy:
    """Combine multiple strategies for optimal compression.

    Applies strategies in sequence until token budget is met.
    """

    name: str = "hybrid"
    strategies: tuple[ContextStrategy, ...] = ()

    def apply(self, package: ContextPackage, token_budget: int) -> Result[ContextPackage, ContextError]:
        """Apply strategies in sequence."""
        current_package = package

        for strategy in self.strategies:
            if current_package.token_count <= token_budget:
                # Already within budget
                break

            result = strategy.apply(current_package, token_budget)
            if result.is_err():
                return result

            current_package = result.unwrap()

        return ok(current_package)


# Factory functions for creating strategies


def create_sliding_window(importance_threshold: float = 0.0, prefer_recent: bool = True) -> SlidingWindowStrategy:
    """Create a sliding window strategy."""
    return SlidingWindowStrategy(importance_threshold=importance_threshold, prefer_recent=prefer_recent)


def create_pattern_compression(min_frequency: int = 3) -> PatternCompressionStrategy:
    """Create a pattern compression strategy."""
    return PatternCompressionStrategy(min_pattern_frequency=min_frequency)


def create_range_aggregation(min_size: int = 5) -> RangeAggregationStrategy:
    """Create a range aggregation strategy."""
    return RangeAggregationStrategy(min_range_size=min_size)


def create_hybrid(*strategies: ContextStrategy) -> HybridStrategy:
    """Create a hybrid strategy combining multiple strategies."""
    return HybridStrategy(strategies=strategies)
