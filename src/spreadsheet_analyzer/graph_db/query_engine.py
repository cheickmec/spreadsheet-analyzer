"""
Flexible query engine for spreadsheet dependency analysis.

This module provides a high-level, intuitive interface for querying
spreadsheet dependencies and relationships, suitable for use by LLMs,
APIs, or any client that needs to analyze Excel formula dependencies.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from spreadsheet_analyzer.pipeline.types import FormulaAnalysis

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Supported query types for the LLM interface."""

    DEPENDENCIES = "dependencies"  # What does this cell depend on?
    DEPENDENTS = "dependents"  # What depends on this cell?
    PATH = "path"  # Path between two cells
    IMPACT = "impact"  # What cells are affected by changes to this cell?
    SOURCES = "sources"  # What are the ultimate data sources for this cell?
    CIRCULAR = "circular"  # Is this cell part of a circular reference?
    NEIGHBORS = "neighbors"  # Direct connections (both directions)
    FORMULA = "formula"  # Get formula for a cell
    EXISTS = "exists"  # Does this cell exist in the graph?
    STATS = "stats"  # Statistics about a cell's position in the graph


@dataclass
class CellInfo:
    """Complete information about a cell."""

    ref: str  # Cell reference like "B500"
    sheet: str
    formula: str | None
    is_empty: bool  # True if cell has no formula/value
    in_ranges: list[str]  # Ranges this cell is part of


@dataclass
class DependencyInfo:
    """Information about a dependency relationship."""

    cell: CellInfo
    relationship: Literal["uses", "used_by", "in_range", "range_contains"]
    via_range: str | None  # If dependency is through a range, which one


@dataclass
class QueryResult:
    """Flexible result structure for all query types."""

    query_type: QueryType
    success: bool
    data: dict[str, Any]
    cells: list[CellInfo]
    relationships: list[DependencyInfo]
    explanation: str


class SpreadsheetQueryEngine:
    """
    Flexible query engine for spreadsheet dependencies.

    CLAUDE-KNOWLEDGE: This interface unifies cell and range dependencies
    into a single, intuitive API that LLMs can use without needing to
    understand the distinction between "direct" and "range" dependencies.
    """

    def __init__(self, formula_analysis: FormulaAnalysis):
        """Initialize with formula analysis results."""
        self.analysis = formula_analysis
        self.graph = formula_analysis.dependency_graph
        self.range_index = formula_analysis.range_membership_index

    def query(self, query_type: QueryType | str, **params) -> QueryResult:
        """
        Execute a flexible query against the spreadsheet graph.

        Examples:
            # What does B500 depend on?
            engine.query("dependencies", sheet="Sheet1", cell="B500")

            # What cells would be affected if A1 changes?
            engine.query("impact", sheet="Sheet1", cell="A1", depth=3)

            # Is there a calculation path from A1 to Z100?
            engine.query("path", from_sheet="Sheet1", from_cell="A1",
                        to_sheet="Sheet1", to_cell="Z100")

            # What are the ultimate data sources for this cell?
            engine.query("sources", sheet="Sheet1", cell="D10")
        """
        # Convert string to enum if needed
        if isinstance(query_type, str):
            try:
                query_type = QueryType(query_type)
            except ValueError:
                return QueryResult(
                    query_type=QueryType.DEPENDENCIES,
                    success=False,
                    data={},
                    cells=[],
                    relationships=[],
                    explanation=f"Unknown query type: {query_type}",
                )

        # Route to appropriate handler
        handlers = {
            QueryType.DEPENDENCIES: self._query_dependencies,
            QueryType.DEPENDENTS: self._query_dependents,
            QueryType.PATH: self._query_path,
            QueryType.IMPACT: self._query_impact,
            QueryType.SOURCES: self._query_sources,
            QueryType.CIRCULAR: self._query_circular,
            QueryType.NEIGHBORS: self._query_neighbors,
            QueryType.FORMULA: self._query_formula,
            QueryType.EXISTS: self._query_exists,
            QueryType.STATS: self._query_stats,
        }

        handler = handlers.get(query_type)
        if not handler:
            return QueryResult(
                query_type=query_type,
                success=False,
                data={},
                cells=[],
                relationships=[],
                explanation=f"Handler not implemented for {query_type}",
            )

        return handler(**params)

    def _get_cell_info(self, sheet: str, cell_ref: str) -> CellInfo:
        """Get complete information about a cell."""
        cell_key = f"{sheet}!{cell_ref}"
        node = self.graph.get(cell_key)

        # Check which ranges contain this cell
        in_ranges = []
        if self.range_index:
            range_keys = self.range_index.get_ranges_containing_cell(sheet, cell_ref)
            for _range_key in range_keys:
                # Extract the range reference from formulas that use it
                for formula_key in range_keys:
                    if formula_node := self.graph.get(formula_key):
                        for dep in formula_node.dependencies:
                            if dep.is_range and self._cell_in_range(sheet, cell_ref, dep.cell):
                                in_ranges.append(f"{dep.sheet}!{dep.cell}")

        return CellInfo(
            ref=cell_ref,
            sheet=sheet,
            formula=node.formula if node else None,
            is_empty=node is None,
            in_ranges=in_ranges,
        )

    def _cell_in_range(self, sheet: str, cell_ref: str, range_ref: str) -> bool:  # noqa: ARG002
        """Check if a cell is within a range."""
        import re

        from spreadsheet_analyzer.graph_db.range_membership import col_to_num

        # Parse cell
        cell_match = re.match(r"^([A-Z]+)(\d+)$", cell_ref.upper())
        if not cell_match:
            return False

        cell_col, cell_row = cell_match.groups()
        cell_col_num = col_to_num(cell_col)
        cell_row_num = int(cell_row)

        # Parse range
        range_match = re.match(r"^([A-Z]+)(\d+):([A-Z]+)(\d+)$", range_ref.upper())
        if not range_match:
            return False

        start_col, start_row, end_col, end_row = range_match.groups()
        start_col_num = col_to_num(start_col)
        end_col_num = col_to_num(end_col)
        start_row_num = int(start_row)
        end_row_num = int(end_row)

        return start_col_num <= cell_col_num <= end_col_num and start_row_num <= cell_row_num <= end_row_num

    def _query_dependencies(self, sheet: str, cell: str, **kwargs) -> QueryResult:
        """What does this cell depend on?"""
        cell_info = self._get_cell_info(sheet, cell)
        relationships = []
        cells = [cell_info]

        cell_key = f"{sheet}!{cell}"

        # Get formula dependencies
        if node := self.graph.get(cell_key):
            for dep in node.dependencies:
                dep_info = self._get_cell_info(dep.sheet, dep.cell)
                cells.append(dep_info)

                rel_type = "uses"
                via_range = dep.cell if dep.is_range else None

                relationships.append(DependencyInfo(cell=dep_info, relationship=rel_type, via_range=via_range))

        # Check if this cell is part of any ranges
        if cell_info.in_ranges:
            explanation = (
                f"{cell} depends on {len(relationships)} cells and is part of {len(cell_info.in_ranges)} ranges"
            )
        else:
            explanation = f"{cell} depends on {len(relationships)} cells"

        return QueryResult(
            query_type=QueryType.DEPENDENCIES,
            success=True,
            data={"total_dependencies": len(relationships)},
            cells=cells,
            relationships=relationships,
            explanation=explanation,
        )

    def _query_dependents(self, sheet: str, cell: str, **kwargs) -> QueryResult:
        """What depends on this cell?"""
        cell_info = self._get_cell_info(sheet, cell)
        relationships = []
        cells = [cell_info]

        # Find formulas that reference this cell directly
        for key, node in self.graph.items():
            for dep in node.dependencies:
                if dep.sheet == sheet and dep.cell == cell and not dep.is_range:
                    dep_sheet, dep_cell = key.split("!", 1)
                    dep_info = self._get_cell_info(dep_sheet, dep_cell)
                    cells.append(dep_info)

                    relationships.append(DependencyInfo(cell=dep_info, relationship="used_by", via_range=None))

        # Find formulas that reference this cell through ranges
        if self.range_index:
            range_formulas = self.range_index.get_ranges_containing_cell(sheet, cell)
            for formula_key in range_formulas:
                if node := self.graph.get(formula_key):
                    formula_sheet, formula_cell = formula_key.split("!", 1)
                    formula_info = self._get_cell_info(formula_sheet, formula_cell)

                    # Find which range includes this cell
                    via_range = None
                    for dep in node.dependencies:
                        if dep.is_range and self._cell_in_range(sheet, cell, dep.cell):
                            via_range = f"{dep.sheet}!{dep.cell}"
                            break

                    cells.append(formula_info)
                    relationships.append(DependencyInfo(cell=formula_info, relationship="used_by", via_range=via_range))

        explanation = f"{len(relationships)} cells depend on {cell}"

        return QueryResult(
            query_type=QueryType.DEPENDENTS,
            success=True,
            data={"total_dependents": len(relationships)},
            cells=cells,
            relationships=relationships,
            explanation=explanation,
        )

    def _query_impact(self, sheet: str, cell: str, depth: int = 2, **kwargs) -> QueryResult:
        """What cells would be affected by changes to this cell?"""
        # This is similar to dependents but follows the chain
        impacted = set()
        relationships = []
        cells = []

        # BFS to find all impacted cells
        queue = [(f"{sheet}!{cell}", 0)]
        visited = set()

        while queue:
            current_key, current_depth = queue.pop(0)

            if current_key in visited or current_depth > depth:
                continue

            visited.add(current_key)

            # Add to results
            if current_depth > 0:  # Don't include the starting cell
                impacted.add(current_key)
                current_sheet, current_cell = current_key.split("!", 1)
                cell_info = self._get_cell_info(current_sheet, current_cell)
                cells.append(cell_info)

            # Find all cells that depend on this one
            current_sheet, current_cell = current_key.split("!", 1)

            # Direct dependencies
            for key, node in self.graph.items():
                for dep in node.dependencies:
                    dep_key = f"{dep.sheet}!{dep.cell}"
                    if dep_key == current_key and not dep.is_range:
                        queue.append((key, current_depth + 1))

            # Range dependencies
            if self.range_index:
                range_formulas = self.range_index.get_ranges_containing_cell(current_sheet, current_cell)
                for formula_key in range_formulas:
                    if formula_key not in visited:
                        queue.append((formula_key, current_depth + 1))

        explanation = f"Changes to {cell} would impact {len(impacted)} cells within {depth} steps"

        return QueryResult(
            query_type=QueryType.IMPACT,
            success=True,
            data={"impacted_cells": len(impacted), "max_depth": depth},
            cells=cells,
            relationships=relationships,
            explanation=explanation,
        )

    def _query_sources(self, sheet: str, cell: str, **kwargs) -> QueryResult:
        """Find ultimate data sources (cells with no dependencies)."""
        sources = set()
        cells = []

        # BFS to find all source cells
        queue = [f"{sheet}!{cell}"]
        visited = set()

        while queue:
            current_key = queue.pop(0)

            if current_key in visited:
                continue

            visited.add(current_key)

            if node := self.graph.get(current_key):
                if not node.dependencies:
                    # This is a source cell
                    sources.add(current_key)
                    current_sheet, current_cell = current_key.split("!", 1)
                    cell_info = self._get_cell_info(current_sheet, current_cell)
                    cells.append(cell_info)
                else:
                    # Add dependencies to queue
                    for dep in node.dependencies:
                        dep_key = f"{dep.sheet}!{dep.cell}"
                        if dep_key not in visited:
                            queue.append(dep_key)
            else:
                # No formula = data source
                sources.add(current_key)
                if "!" in current_key:
                    current_sheet, current_cell = current_key.split("!", 1)
                    cell_info = self._get_cell_info(current_sheet, current_cell)
                    cells.append(cell_info)

        explanation = f"{cell} ultimately depends on {len(sources)} source cells"

        return QueryResult(
            query_type=QueryType.SOURCES,
            success=True,
            data={"source_count": len(sources)},
            cells=cells,
            relationships=[],
            explanation=explanation,
        )

    def _query_path(self, from_sheet: str, from_cell: str, to_sheet: str, to_cell: str, **kwargs) -> QueryResult:
        """Find calculation path between two cells."""
        from_key = f"{from_sheet}!{from_cell}"
        to_key = f"{to_sheet}!{to_cell}"

        # BFS to find shortest path
        queue = [(from_key, [from_key])]
        visited = set()

        while queue:
            current_key, path = queue.pop(0)

            if current_key in visited:
                continue

            visited.add(current_key)

            if current_key == to_key:
                # Found path!
                cells = []
                for key in path:
                    key_sheet, key_cell = key.split("!", 1)
                    cells.append(self._get_cell_info(key_sheet, key_cell))

                explanation = f"Found path from {from_cell} to {to_cell} with {len(path)} steps"

                return QueryResult(
                    query_type=QueryType.PATH,
                    success=True,
                    data={"path_length": len(path), "path": path},
                    cells=cells,
                    relationships=[],
                    explanation=explanation,
                )

            # Add dependencies to queue
            if node := self.graph.get(current_key):
                for dep in node.dependencies:
                    dep_key = f"{dep.sheet}!{dep.cell}"
                    if dep_key not in visited:
                        queue.append((dep_key, path + [dep_key]))

        # No path found
        return QueryResult(
            query_type=QueryType.PATH,
            success=True,
            data={"path_exists": False},
            cells=[],
            relationships=[],
            explanation=f"No calculation path from {from_cell} to {to_cell}",
        )

    def _query_circular(self, sheet: str, cell: str, **kwargs) -> QueryResult:
        """Check if cell is part of circular reference."""
        cell_key = f"{sheet}!{cell}"

        # Check all circular reference chains
        for chain in self.analysis.circular_references:
            if cell_key in chain:
                cells = []
                for key in chain:
                    key_sheet, key_cell = key.split("!", 1)
                    cells.append(self._get_cell_info(key_sheet, key_cell))

                return QueryResult(
                    query_type=QueryType.CIRCULAR,
                    success=True,
                    data={"is_circular": True, "chain_length": len(chain), "chain": list(chain)},
                    cells=cells,
                    relationships=[],
                    explanation=f"{cell} is part of a circular reference chain with {len(chain)} cells",
                )

        return QueryResult(
            query_type=QueryType.CIRCULAR,
            success=True,
            data={"is_circular": False},
            cells=[self._get_cell_info(sheet, cell)],
            relationships=[],
            explanation=f"{cell} is not part of any circular reference",
        )

    def _query_neighbors(self, sheet: str, cell: str, **kwargs) -> QueryResult:
        """Get all direct connections (dependencies and dependents)."""
        deps_result = self._query_dependencies(sheet, cell)
        dependents_result = self._query_dependents(sheet, cell)

        # Combine results
        all_cells = deps_result.cells + dependents_result.cells[1:]  # Skip duplicate of main cell
        all_relationships = deps_result.relationships + dependents_result.relationships

        return QueryResult(
            query_type=QueryType.NEIGHBORS,
            success=True,
            data={
                "total_neighbors": len(all_relationships),
                "dependencies": len(deps_result.relationships),
                "dependents": len(dependents_result.relationships),
            },
            cells=all_cells,
            relationships=all_relationships,
            explanation=f"{cell} has {len(all_relationships)} direct connections",
        )

    def _query_formula(self, sheet: str, cell: str, **kwargs) -> QueryResult:
        """Get formula for a specific cell."""
        cell_info = self._get_cell_info(sheet, cell)

        if cell_info.formula:
            return QueryResult(
                query_type=QueryType.FORMULA,
                success=True,
                data={"has_formula": True, "formula": cell_info.formula},
                cells=[cell_info],
                relationships=[],
                explanation=f"{cell} contains formula: {cell_info.formula}",
            )
        else:
            return QueryResult(
                query_type=QueryType.FORMULA,
                success=True,
                data={"has_formula": False},
                cells=[cell_info],
                relationships=[],
                explanation=f"{cell} has no formula",
            )

    def _query_exists(self, sheet: str, cell: str, **kwargs) -> QueryResult:
        """Check if cell exists in the dependency graph."""
        cell_key = f"{sheet}!{cell}"
        exists_as_formula = cell_key in self.graph

        # Check if it's referenced by any formula
        referenced = False
        for node in self.graph.values():
            for dep in node.dependencies:
                if dep.sheet == sheet and dep.cell == cell:
                    referenced = True
                    break
            if referenced:
                break

        # Check if in any ranges
        in_ranges = []
        if self.range_index:
            range_keys = self.range_index.get_ranges_containing_cell(sheet, cell)
            in_ranges = list(range_keys)

        exists = exists_as_formula or referenced or bool(in_ranges)

        return QueryResult(
            query_type=QueryType.EXISTS,
            success=True,
            data={
                "exists": exists,
                "has_formula": exists_as_formula,
                "is_referenced": referenced,
                "in_ranges": bool(in_ranges),
            },
            cells=[self._get_cell_info(sheet, cell)],
            relationships=[],
            explanation=f"{cell} {'exists' if exists else 'does not exist'} in the dependency graph",
        )

    def _query_stats(self, sheet: str, cell: str, **kwargs) -> QueryResult:
        """Get statistics about a cell's position in the graph."""
        cell_info = self._get_cell_info(sheet, cell)
        cell_key = f"{sheet}!{cell}"

        # Calculate stats
        stats = {
            "has_formula": bool(cell_info.formula),
            "dependency_count": 0,
            "dependent_count": 0,
            "depth": 0,
            "in_range_count": len(cell_info.in_ranges),
            "is_volatile": cell_key in self.analysis.volatile_formulas,
            "has_external_ref": cell_key in self.analysis.external_references,
            "is_circular": any(cell_key in chain for chain in self.analysis.circular_references),
        }

        # Get dependency counts
        if node := self.graph.get(cell_key):
            stats["dependency_count"] = len(node.dependencies)
            stats["depth"] = node.depth

        # Count dependents
        deps_result = self._query_dependents(sheet, cell)
        stats["dependent_count"] = len(deps_result.relationships)

        return QueryResult(
            query_type=QueryType.STATS,
            success=True,
            data=stats,
            cells=[cell_info],
            relationships=[],
            explanation=f"Statistics for {cell}: {stats['dependent_count']} dependents, "
            f"{stats['dependency_count']} dependencies, depth {stats['depth']}",
        )


def create_query_engine(formula_analysis: FormulaAnalysis) -> SpreadsheetQueryEngine:
    """
    Factory function to create a spreadsheet query engine.

    This creates a high-level interface for querying spreadsheet
    dependencies, suitable for use by LLMs, APIs, or interactive tools.
    """
    return SpreadsheetQueryEngine(formula_analysis)
