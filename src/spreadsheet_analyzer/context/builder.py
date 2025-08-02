"""Functional context package builder.

This module provides pure functions for building optimized context
packages from spreadsheet data.

CLAUDE-KNOWLEDGE: Context building is a critical step that determines
what information the LLM sees. Good context = good analysis.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.errors import ContextError
from ..core.types import Result, err, ok
from .strategies import (
    create_hybrid,
    create_pattern_compression,
    create_range_aggregation,
    create_sliding_window,
)
from .token_management import (
    TokenBudget,
    estimate_tokens,
)
from .types import (
    ContextCell,
    ContextPackage,
    ContextQuery,
    ContextStrategy,
)


@dataclass(frozen=True)
class ContextBuilder:
    """Immutable context builder configuration."""

    default_strategy: ContextStrategy
    model: str = "gpt-4"
    include_metadata: bool = True
    metadata_builders: tuple[Callable[..., dict[str, Any]], ...] = ()

    def build(
        self, cells: list[dict[str, Any]], query: ContextQuery, budget: TokenBudget
    ) -> Result[ContextPackage, ContextError]:
        """Build optimized context package.

        Args:
            cells: Raw cell data from spreadsheet
            query: Context requirements
            budget: Token budget

        Returns:
            Optimized context package or error
        """
        # Convert to ContextCells
        context_cells_result = self._create_context_cells(cells, query)
        if context_cells_result.is_err():
            return context_cells_result

        context_cells = context_cells_result.unwrap()

        # Build metadata
        metadata = self._build_metadata(cells, query) if self.include_metadata else {}

        # Create initial package
        initial_package = ContextPackage.create(
            cells=context_cells, metadata=metadata, focus_hints=self._generate_focus_hints(query)
        )

        # Apply optimization strategy
        optimized_result = self.default_strategy.apply(initial_package, budget.context)
        if optimized_result.is_err():
            return optimized_result

        optimized_package = optimized_result.unwrap()

        # Validate result
        if optimized_package.token_count > budget.context:
            return err(ContextError(f"Context exceeds budget: {optimized_package.token_count} > {budget.context}"))

        return ok(optimized_package)

    def _create_context_cells(
        self, cells: list[dict[str, Any]], query: ContextQuery
    ) -> Result[list[ContextCell], ContextError]:
        """Convert raw cells to ContextCells."""
        context_cells = []

        for cell_data in cells:
            # Apply query filters
            if not self._matches_query(cell_data, query):
                continue

            # Calculate importance based on query
            importance = self._calculate_importance(cell_data, query)

            if importance < query.relevance_threshold:
                continue

            # Create ContextCell
            try:
                context_cell = ContextCell(
                    location=cell_data.get("location", ""),
                    content=cell_data.get("content"),
                    cell_type=cell_data.get("type", "value"),
                    importance=importance,
                    metadata=cell_data.get("metadata", {}),
                )
                context_cells.append(context_cell)
            except Exception as e:
                return err(ContextError(f"Failed to create context cell: {e}"))

        # Apply max_cells limit if specified
        if query.max_cells and len(context_cells) > query.max_cells:
            # Sort by importance and take top N
            context_cells.sort(key=lambda c: c.importance, reverse=True)
            context_cells = context_cells[: query.max_cells]

        return ok(context_cells)

    def _matches_query(self, cell_data: dict[str, Any], query: ContextQuery) -> bool:
        """Check if cell matches query criteria."""
        # Check sheet filter
        if query.sheet_names:
            location = cell_data.get("location", "")
            sheet = location.split("!")[0] if "!" in location else ""
            if sheet not in query.sheet_names:
                return False

        # Check cell type filters
        cell_type = cell_data.get("type", "value")
        if cell_type == "formula" and not query.include_formulas:
            return False
        if cell_type == "value" and not query.include_values:
            return False

        # Check range filters
        if query.cell_ranges:
            # Simplified range check - production would need proper range parsing
            location = cell_data.get("location", "")
            in_range = any(self._in_range(location, range_str) for range_str in query.cell_ranges)
            if not in_range:
                return False

        return True

    def _in_range(self, location: str, range_str: str) -> bool:
        """Check if location is in range (simplified)."""
        # This is a simplified implementation
        # Production would need proper Excel range parsing
        return range_str in location or location in range_str

    def _calculate_importance(self, cell_data: dict[str, Any], query: ContextQuery) -> float:
        """Calculate cell importance based on query."""
        importance = 0.5  # Base importance

        # Boost for query text matches
        if query.query_text:
            content_str = str(cell_data.get("content", "")).lower()
            query_lower = query.query_text.lower()

            if query_lower in content_str:
                importance += 0.3

            # Partial word matches
            query_words = query_lower.split()
            matches = sum(1 for word in query_words if word in content_str)
            importance += (matches / len(query_words)) * 0.2 if query_words else 0

        # Boost for formulas if they're included
        if cell_data.get("type") == "formula" and query.include_formulas:
            importance += 0.1

        # Boost for non-empty cells
        if cell_data.get("content") not in (None, "", 0):
            importance += 0.1

        return min(importance, 1.0)  # Cap at 1.0

    def _build_metadata(self, cells: list[dict[str, Any]], query: ContextQuery) -> dict[str, Any]:
        """Build metadata for context package."""
        metadata = {
            "total_cells": len(cells),
            "query": query.query_text or "general",
            "filters_applied": {
                "sheet_names": query.sheet_names is not None,
                "cell_ranges": query.cell_ranges is not None,
                "max_cells": query.max_cells is not None,
            },
        }

        # Apply custom metadata builders
        for builder in self.metadata_builders:
            try:
                custom_metadata = builder(cells, query)
                metadata.update(custom_metadata)
            except Exception:
                # Skip failed builders - they're optional
                continue

        return metadata

    def _generate_focus_hints(self, query: ContextQuery) -> list[str]:
        """Generate focus hints from query."""
        hints = []

        if query.query_text:
            hints.append(f"Focus on: {query.query_text}")

        if query.sheet_names:
            hints.append(f"Sheets: {', '.join(query.sheet_names)}")

        if query.cell_ranges:
            hints.append(f"Ranges: {', '.join(query.cell_ranges)}")

        return hints


def create_default_builder(model: str = "gpt-4") -> ContextBuilder:
    """Create a context builder with default configuration."""
    # Default strategy: hybrid with multiple compression techniques
    default_strategy = create_hybrid(
        create_sliding_window(importance_threshold=0.3),
        create_pattern_compression(min_frequency=3),
        create_range_aggregation(min_size=5),
    )

    return ContextBuilder(default_strategy=default_strategy, model=model, include_metadata=True)


def create_minimal_builder(model: str = "gpt-4") -> ContextBuilder:
    """Create a minimal context builder for tight token budgets."""
    # Aggressive compression strategy
    minimal_strategy = create_hybrid(
        create_sliding_window(importance_threshold=0.7, prefer_recent=False),
        create_pattern_compression(min_frequency=2),
        create_range_aggregation(min_size=3),
    )

    return ContextBuilder(
        default_strategy=minimal_strategy,
        model=model,
        include_metadata=False,  # Skip metadata to save tokens
    )


def create_analysis_builder(focus: str, model: str = "gpt-4") -> ContextBuilder:
    """Create a context builder optimized for specific analysis."""
    if focus == "formulas":
        # Preserve formula details
        strategy = create_hybrid(
            create_sliding_window(importance_threshold=0.2),
            create_pattern_compression(min_frequency=5),  # Less aggressive
        )
    elif focus == "structure":
        # Focus on structure overview
        strategy = create_hybrid(create_range_aggregation(min_size=3), create_pattern_compression(min_frequency=3))
    else:
        # General analysis
        return create_default_builder(model)

    return ContextBuilder(default_strategy=strategy, model=model, include_metadata=True)


def build_context_from_cells(
    cells: list[dict[str, Any]],
    query_text: str = "",
    token_budget: int = 4000,
    model: str = "gpt-4",
    builder: ContextBuilder | None = None,
) -> Result[ContextPackage, ContextError]:
    """Convenience function to build context from cells.

    Args:
        cells: Raw cell data
        query_text: Query for context relevance
        token_budget: Maximum tokens for context
        model: Model name
        builder: Optional custom builder

    Returns:
        Context package or error
    """
    if builder is None:
        builder = create_default_builder(model)

    query = ContextQuery(query_text=query_text, include_formulas=True, include_values=True)

    budget = TokenBudget(
        total=token_budget,
        context=token_budget,  # All tokens for context in this simple case
    )

    return builder.build(cells, query, budget)


def estimate_context_size(cells: list[dict[str, Any]], model: str = "gpt-4") -> int:
    """Estimate token size for cells without building full context.

    Args:
        cells: Raw cell data
        model: Model for token estimation

    Returns:
        Estimated token count
    """
    total = 0

    for cell in cells:
        # Estimate tokens for each cell
        location = cell.get("location", "")
        content = str(cell.get("content", ""))
        cell_type = cell.get("type", "")

        cell_text = f"{location}: {content}"
        if cell_type == "formula":
            cell_text = f"{location}: ={content}"

        total += estimate_tokens(cell_text, model)

    return total
