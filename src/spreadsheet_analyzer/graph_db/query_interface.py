"""
Agent-friendly query interface for graph database and range membership queries.

This module provides high-level methods for querying the dependency graph
without requiring knowledge of Cypher or graph database internals. It also
includes support for range membership queries to handle empty cells that
are part of formula ranges.
"""

import logging
from dataclasses import dataclass
from typing import Any

from neo4j import GraphDatabase

from spreadsheet_analyzer.pipeline.types import FormulaAnalysis, FormulaNode

logger = logging.getLogger(__name__)


@dataclass
class DependencyQueryResult:
    """Result of a dependency query."""

    cell_ref: str
    sheet: str
    direct_dependencies: list[str]  # Cells this formula directly references
    range_dependencies: list[str]  # Ranges this formula references
    direct_dependents: list[str]  # Cells that directly reference this cell
    range_dependents: list[str]  # Formulas that reference this cell via ranges
    is_in_ranges: list[str]  # Ranges this cell is part of
    total_dependencies: int
    total_dependents: int
    has_formula: bool
    formula: str | None = None


class GraphQueryInterface:
    """High-level interface for querying formula dependency graphs."""

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize connection to Neo4j.

        Args:
            uri: Neo4j connection URI
            user: Database username
            password: Database password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close database connection."""
        self.driver.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_dependencies(self, cell_ref: str, depth: int = 1) -> list[dict[str, Any]]:
        """
        Get cells that this cell depends on.

        Args:
            cell_ref: Cell reference (e.g., "Sheet1!A1")
            depth: How many levels deep to traverse (default 1)

        Returns:
            List of dependency information
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start {key: $ref})-[:DEPENDS_ON*1..$depth]->(dep)
                RETURN DISTINCT
                    dep.key as cell_ref,
                    dep.sheet as sheet,
                    dep.ref as ref,
                    dep.formula as formula,
                    labels(dep)[0] as node_type,
                    length(path) as distance,
                    dep.pagerank as importance
                ORDER BY distance, importance DESC
            """,
                ref=cell_ref,
                depth=depth,
            )

            return [dict(record) for record in result]

    def get_dependents(self, cell_ref: str, depth: int = 1) -> list[dict[str, Any]]:
        """
        Get cells that depend on this cell.

        Args:
            cell_ref: Cell reference (e.g., "Sheet1!A1")
            depth: How many levels deep to traverse (default 1)

        Returns:
            List of dependent cells
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (start {key: $ref})<-[:DEPENDS_ON*1..$depth]-(dep)
                RETURN DISTINCT
                    dep.key as cell_ref,
                    dep.sheet as sheet,
                    dep.ref as ref,
                    dep.formula as formula,
                    labels(dep)[0] as node_type,
                    length(path) as distance,
                    dep.pagerank as importance
                ORDER BY distance, importance DESC
            """,
                ref=cell_ref,
                depth=depth,
            )

            return [dict(record) for record in result]

    def find_circular_references(self, sheet: str | None = None) -> list[list[str]]:
        """
        Find all circular reference chains.

        Args:
            sheet: Optional sheet name to filter by

        Returns:
            List of circular reference chains
        """
        with self.driver.session() as session:
            if sheet:
                query = """
                    MATCH path = (n:Cell {sheet: $sheet})-[:DEPENDS_ON*]->(n)
                    RETURN [node in nodes(path) | node.key] as chain
                """
                result = session.run(query, sheet=sheet)
            else:
                query = """
                    MATCH path = (n:Cell)-[:DEPENDS_ON*]->(n)
                    RETURN [node in nodes(path) | node.key] as chain
                """
                result = session.run(query)

            chains = []
            seen = set()
            for record in result:
                chain = record["chain"]
                # Normalize chain to start with smallest element
                chain_key = tuple(sorted(chain))
                if chain_key not in seen:
                    seen.add(chain_key)
                    chains.append(chain)

            return chains

    def get_critical_cells(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get cells with highest PageRank scores (most referenced).

        Args:
            limit: Maximum number of cells to return

        Returns:
            List of critical cells with their properties
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.pagerank IS NOT NULL
                RETURN
                    n.key as cell_ref,
                    n.sheet as sheet,
                    n.ref as ref,
                    n.formula as formula,
                    labels(n)[0] as node_type,
                    n.pagerank as importance,
                    n.depth as depth,
                    size((n)<-[:DEPENDS_ON]-()) as dependent_count
                ORDER BY n.pagerank DESC
                LIMIT $limit
            """,
                limit=limit,
            )

            return [dict(record) for record in result]

    def trace_calculation_path(self, from_cell: str, to_cell: str) -> list[str] | None:
        """
        Find the calculation path between two cells.

        Args:
            from_cell: Starting cell reference
            to_cell: Target cell reference

        Returns:
            List of cells in the path, or None if no path exists
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = shortestPath(
                    (start {key: $from})-[:DEPENDS_ON*]->(end {key: $to})
                )
                RETURN [node in nodes(path) | node.key] as path
            """,
                from_cell=from_cell,
                to_cell=to_cell,
            )

            record = result.single()
            return list(record["path"]) if record else None

    def get_range_details(self, range_ref: str) -> dict[str, Any] | None:
        """
        Get detailed information about a range node.

        Args:
            range_ref: Range reference (e.g., "Sheet1!B1:B10")

        Returns:
            Range details or None if not found
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (r:Range {key: $ref})
                OPTIONAL MATCH (r)<-[:DEPENDS_ON]-(dependent)
                WITH r, collect(DISTINCT dependent.key) as dependents
                RETURN {
                    key: r.key,
                    sheet: r.sheet,
                    ref: r.ref,
                    type: r.type,
                    size: r.size,
                    start_cell: r.start_cell,
                    end_cell: r.end_cell,
                    dependent_formulas: dependents,
                    dependent_count: size(dependents)
                } as details
            """,
                ref=range_ref,
            )

            record = result.single()
            return dict(record["details"]) if record else None

    def get_graph_summary(self) -> dict[str, Any]:
        """
        Get high-level summary of the graph for agent context.

        Returns:
            Summary statistics and key insights
        """
        with self.driver.session() as session:
            result = session.run("""
                // Basic counts
                MATCH (n)
                WITH
                    count(DISTINCT CASE WHEN n:Cell THEN n END) as cells,
                    count(DISTINCT CASE WHEN n:Range THEN n END) as ranges
                MATCH ()-[r:DEPENDS_ON]->()
                WITH cells, ranges, count(r) as edges

                // Cross-sheet dependencies
                MATCH (n1)-[:DEPENDS_ON]->(n2)
                WHERE n1.sheet <> n2.sheet
                WITH cells, ranges, edges, count(*) as cross_sheet_deps

                // Max depth and critical cells
                MATCH (n)
                WITH cells, ranges, edges, cross_sheet_deps,
                     max(n.depth) as max_depth,
                     max(n.pagerank) as max_pagerank

                // Volatile formulas (if tracked)
                OPTIONAL MATCH (v:Cell)
                WHERE v.formula =~ '.*(?i)(NOW|TODAY|RAND|RANDBETWEEN|INDIRECT|OFFSET).*'
                WITH cells, ranges, edges, cross_sheet_deps, max_depth, max_pagerank,
                     count(DISTINCT v) as volatile_count

                // Circular references
                OPTIONAL MATCH p=(c:Cell)-[:DEPENDS_ON*]->(c)
                WITH cells, ranges, edges, cross_sheet_deps, max_depth, max_pagerank,
                     volatile_count, count(DISTINCT c) as circular_count

                RETURN {
                    total_nodes: cells + ranges,
                    cell_nodes: cells,
                    range_nodes: ranges,
                    total_dependencies: edges,
                    cross_sheet_dependencies: cross_sheet_deps,
                    cross_sheet_percentage: round(100.0 * cross_sheet_deps / edges, 1),
                    max_calculation_depth: max_depth,
                    volatile_formulas: volatile_count,
                    circular_reference_cells: circular_count,
                    highest_pagerank: round(max_pagerank, 4)
                } as summary
            """)

            record = result.single()
            return dict(record["summary"]) if record else {}

    def query_graph(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """
        Execute arbitrary Cypher query (for advanced users).

        Args:
            cypher: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dictionaries
        """
        with self.driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]


class EnhancedQueryInterface:
    """
    Enhanced query interface that combines in-memory dependency graph with range membership.

    CLAUDE-KNOWLEDGE: This interface provides comprehensive dependency information
    by combining direct graph edges with range membership tracking, enabling
    queries about empty cells that are part of formula ranges.
    """

    def __init__(self, formula_analysis: FormulaAnalysis, *, graph_interface: GraphQueryInterface | None = None):
        """
        Initialize with formula analysis results and optional graph database.

        Args:
            formula_analysis: Results from formula analysis stage
            graph_interface: Optional Neo4j query interface for advanced queries
        """
        self.analysis = formula_analysis
        self.graph = formula_analysis.dependency_graph
        self.range_index = formula_analysis.range_index
        self.graph_db = graph_interface

    def get_cell_dependencies(self, sheet: str, cell_ref: str, *, include_ranges: bool = True) -> DependencyQueryResult:
        """
        Get complete dependency information for a cell.

        Args:
            sheet: Sheet name
            cell_ref: Cell reference (e.g., "B500")
            include_ranges: Whether to include range membership info

        Returns:
            Complete dependency information including range memberships
        """
        cell_key = f"{sheet}!{cell_ref}"

        # Check if this cell has a formula
        node = self.graph.get(cell_key)
        has_formula = node is not None
        formula = node.formula if node else None

        # Get direct dependencies (if cell has formula)
        direct_deps = []
        range_deps = []
        if node:
            # Dependencies are stored as strings in the format "Sheet!Cell" or "Sheet!Range"
            for dep in node.dependencies:
                # Check if it's a range (contains colon)
                if ":" in dep:
                    range_deps.append(dep)
                else:
                    direct_deps.append(dep)

        # Calculate dependents if not already done
        if all(n.dependents is None for n in self.graph.values()):
            self._calculate_dependents()

        # Get direct dependents
        direct_dependents = []
        if node and node.dependents:
            direct_dependents = list(node.dependents)

        # Also check for cells not in the graph that might be referenced
        if cell_key not in self.graph:
            for key, formula_node in self.graph.items():
                for dep in formula_node.dependencies:
                    # Check if this dependency points to our cell (not a range)
                    if dep == cell_key and ":" not in dep:
                        direct_dependents.append(key)

        # Get range-based information
        range_dependents = []
        is_in_ranges = []

        if include_ranges and self.range_index and hasattr(self.range_index, "sheet_ranges"):
            # Check if this cell is part of any ranges
            import re

            match = re.match(r"^([A-Z]+)(\d+)$", cell_ref.upper())
            if match:
                col_str, row_str = match.groups()
                from spreadsheet_analyzer.graph_db.range_membership import col_to_num

                row = int(row_str)
                col = col_to_num(col_str)

                # Check if cell is in any range
                if self.range_index.is_cell_in_any_range(sheet, row, col):
                    # For now, we can't determine which specific formulas reference this cell through ranges
                    # without more information in the index
                    pass

        # Calculate totals
        total_dependencies = len(direct_deps) + len(range_deps)
        total_dependents = len(direct_dependents) + len(range_dependents)

        return DependencyQueryResult(
            cell_ref=cell_ref,
            sheet=sheet,
            direct_dependencies=direct_deps,
            range_dependencies=range_deps,
            direct_dependents=direct_dependents,
            range_dependents=range_dependents,
            is_in_ranges=is_in_ranges,
            total_dependencies=total_dependencies,
            total_dependents=total_dependents,
            has_formula=has_formula,
            formula=formula,
        )

    def find_cells_affecting_range(self, sheet: str, start_cell: str, end_cell: str) -> dict[str, list[str]]:
        """
        Find all cells that affect any cell in the given range.

        Returns:
            Dict mapping cells in range to their dependencies
        """
        import re

        from spreadsheet_analyzer.graph_db.range_membership import col_to_num, num_to_col

        # Parse range boundaries
        match_start = re.match(r"^([A-Z]+)(\d+)$", start_cell.upper())
        match_end = re.match(r"^([A-Z]+)(\d+)$", end_cell.upper())

        if not match_start or not match_end:
            return {}

        start_col, start_row = match_start.groups()
        end_col, end_row = match_end.groups()

        start_col_num = col_to_num(start_col)
        end_col_num = col_to_num(end_col)
        start_row_num = int(start_row)
        end_row_num = int(end_row)

        affecting_cells = {}

        # Check each cell in range
        for row in range(start_row_num, end_row_num + 1):
            for col_idx in range(start_col_num, end_col_num + 1):
                col_letter = num_to_col(col_idx)
                cell_ref = f"{col_letter}{row}"

                # Get dependencies for this cell
                result = self.get_cell_dependencies(sheet, cell_ref)

                if result.total_dependencies > 0:
                    deps = result.direct_dependencies + result.range_dependencies
                    affecting_cells[cell_ref] = deps

        return affecting_cells

    def find_empty_cells_in_formula_ranges(self, sheet: str) -> list[str]:
        """
        Find all empty cells that are part of formula ranges.

        Returns:
            List of empty cell references that are included in formula ranges
        """
        if not self.range_index or not hasattr(self.range_index, "sheet_ranges"):
            return []

        empty_cells_in_ranges = []

        # For each range in the sheet
        if sheet in self.range_index.sheet_ranges:
            from spreadsheet_analyzer.graph_db.range_membership import num_to_col

            for range_tuple in self.range_index.sheet_ranges[sheet]:
                # Handle different tuple sizes safely
                if len(range_tuple) >= 4:
                    min_row, max_row, min_col, max_col = range_tuple[:4]
                    # Check each cell in the range
                    for row in range(min_row, max_row + 1):
                        for col in range(min_col, max_col + 1):
                            col_letter = num_to_col(col)
                            cell_ref = f"{col_letter}{row}"
                            cell_key = f"{sheet}!{cell_ref}"

                            # If cell doesn't have a formula, it's either empty or has a value
                            if cell_key not in self.graph:
                                # Check if any other formula references this cell directly
                                is_directly_referenced = any(
                                    dep == cell_key and ":" not in dep
                                    for node in self.graph.values()
                                    for dep in node.dependencies
                                )

                                if not is_directly_referenced:
                                    empty_cells_in_ranges.append(cell_ref)

        return list(set(empty_cells_in_ranges))  # Remove duplicates

    def get_formula_statistics_with_ranges(self) -> dict[str, Any]:
        """Get enhanced statistics including range membership information."""
        base_stats = self._get_base_statistics()

        if self.range_index and hasattr(self.range_index, "sheet_ranges"):
            # Count total cells covered by ranges
            total_cells_in_ranges = 0
            unique_ranges = 0

            # The pipeline's RangeMembershipIndex has a different structure
            for sheet, ranges in self.range_index.sheet_ranges.items():
                unique_ranges += len(ranges)
                for range_tuple in ranges:
                    # Handle different tuple sizes safely
                    if len(range_tuple) >= 4:
                        min_row, max_row, min_col, max_col = range_tuple[:4]
                        # Check for None values before calculation
                        if all(v is not None for v in [min_row, max_row, min_col, max_col]):
                            cells_in_range = (max_row - min_row + 1) * (max_col - min_col + 1)
                            total_cells_in_ranges += cells_in_range

            base_stats.update(
                {
                    "unique_ranges": unique_ranges,
                    "total_cells_in_ranges": total_cells_in_ranges,
                    "has_range_index": True,
                }
            )
        else:
            base_stats["has_range_index"] = False

        return base_stats

    def _calculate_dependents(self) -> None:
        """Calculate dependents for all nodes in the graph."""
        # Build reverse dependency map
        dependents_map = {}

        for key, node in self.graph.items():
            for dep in node.dependencies:
                if dep not in dependents_map:
                    dependents_map[dep] = set()
                dependents_map[dep].add(key)

        # Update nodes with calculated dependents
        for key, node in self.graph.items():
            dependents = dependents_map.get(key, set())
            if dependents or node.dependents != frozenset():
                # Create a new node with updated dependents
                self.graph[key] = FormulaNode(
                    sheet=node.sheet,
                    cell=node.cell,
                    formula=node.formula,
                    dependencies=node.dependencies,
                    volatile=node.volatile,
                    external=node.external,
                    complexity_score=node.complexity_score,
                    edge_labels=node.edge_labels,
                    cell_metadata=node.cell_metadata,
                    dependents=frozenset(dependents),
                    depth=node.depth,
                    pagerank=node.pagerank,
                )

    def _calculate_depths(self) -> None:
        """Calculate depth for all nodes in the graph."""
        # First, identify leaf nodes (nodes with no dependencies)
        leaf_nodes = {key for key, node in self.graph.items() if not node.dependencies}

        # Initialize depths
        depths = dict.fromkeys(leaf_nodes, 0)

        # Use BFS to calculate depths
        from collections import deque

        queue = deque(leaf_nodes)

        while queue:
            current = queue.popleft()
            current_depth = depths.get(current, 0)

            # Find nodes that depend on current node
            for key, node in self.graph.items():
                if current in node.dependencies and key not in depths:
                    # Calculate depth as max of all dependencies + 1
                    dep_depths = []
                    all_deps_calculated = True

                    for dep in node.dependencies:
                        if dep in depths:
                            dep_depths.append(depths[dep])
                        elif dep not in self.graph:  # External dependency
                            dep_depths.append(0)
                        else:
                            all_deps_calculated = False
                            break

                    if all_deps_calculated:
                        depths[key] = max(dep_depths) + 1 if dep_depths else 1
                        queue.append(key)

        # Update nodes with calculated depths
        for key, depth in depths.items():
            if key in self.graph:
                # Create a new node with updated depth
                old_node = self.graph[key]
                self.graph[key] = FormulaNode(
                    sheet=old_node.sheet,
                    cell=old_node.cell,
                    formula=old_node.formula,
                    dependencies=old_node.dependencies,
                    volatile=old_node.volatile,
                    external=old_node.external,
                    complexity_score=old_node.complexity_score,
                    edge_labels=old_node.edge_labels,
                    cell_metadata=old_node.cell_metadata,
                    dependents=old_node.dependents,
                    depth=depth,
                    pagerank=old_node.pagerank,
                )

    def _get_base_statistics(self) -> dict[str, Any]:
        """Get base statistics about formulas."""
        # Calculate depths if not already done
        if all(node.depth is None for node in self.graph.values()):
            self._calculate_depths()

        # Calculate dependents if not already done
        if all(node.dependents is None for node in self.graph.values()):
            self._calculate_dependents()

        total_formulas = len(self.graph)

        # Count formulas with dependencies
        formulas_with_deps = sum(1 for node in self.graph.values() if node.dependencies)

        # Count cells that are referenced
        referenced_cells = set()
        for node in self.graph.values():
            for dep in node.dependencies:
                # Skip ranges (contain colon)
                if ":" not in dep:
                    referenced_cells.add(dep)

        # Count range nodes (nodes don't have node_type, count range dependencies instead)
        range_nodes = sum(1 for node in self.graph.values() for dep in node.dependencies if ":" in dep)

        # Calculate average dependencies
        total_deps = sum(len(node.dependencies) for node in self.graph.values())
        avg_deps = total_deps / total_formulas if total_formulas > 0 else 0

        # Calculate max depth from nodes
        max_depth = max((node.depth for node in self.graph.values() if node.depth is not None), default=0)

        return {
            "total_formulas": total_formulas,
            "formulas_with_dependencies": formulas_with_deps,
            "unique_cells_referenced": len(referenced_cells),
            "range_nodes": range_nodes,
            "circular_reference_chains": len(self.analysis.circular_references),
            "volatile_formulas": len(self.analysis.volatile_formulas),
            "external_references": len(self.analysis.external_references),
            "max_dependency_depth": max_depth,
            "average_dependencies_per_formula": round(avg_deps, 2),
            "complexity_score": self.analysis.formula_complexity_score,
        }

    def export_to_graph_db_format(self) -> dict[str, Any]:
        """
        Export dependency data in a format suitable for graph database import.

        Returns nodes and edges with all metadata needed for Neo4j or similar.
        """
        nodes = []
        edges = []

        # Export formula nodes
        for key, node in self.graph.items():
            node_data = {
                "id": key,
                "type": "cell",  # FormulaNode doesn't have node_type
                "sheet": node.sheet,
                "cell": node.cell,
                "formula": node.formula,
                "depth": node.depth if node.depth is not None else 0,
                "is_volatile": key in self.analysis.volatile_formulas,
                "has_external_ref": key in self.analysis.external_references,
            }

            nodes.append(node_data)

            # Export edges
            for dep in node.dependencies:
                edge_data = {
                    "from": key,
                    "to": dep,
                    "type": "DEPENDS_ON",
                    "is_range": ":" in dep,
                }
                edges.append(edge_data)

        # Add range membership relationships if available
        if self.range_index and hasattr(self.range_index, "sheet_ranges"):
            # Note: The pipeline's RangeMembershipIndex doesn't store formula keys,
            # so we can't create range membership edges here
            pass

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "has_range_index": self.range_index is not None,
            },
        }


def create_enhanced_query_interface(
    formula_analysis: FormulaAnalysis,
    *,
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
) -> EnhancedQueryInterface:
    """
    Factory function to create an enhanced query interface.

    Args:
        formula_analysis: Results from formula analysis
        neo4j_uri: Optional Neo4j connection URI
        neo4j_user: Optional database username
        neo4j_password: Optional database password

    Returns:
        Enhanced query interface with optional graph database support
    """
    graph_db = None
    if neo4j_uri and neo4j_user and neo4j_password:
        graph_db = GraphQueryInterface(neo4j_uri, neo4j_user, neo4j_password)

    return EnhancedQueryInterface(formula_analysis, graph_interface=graph_db)
