"""Graph-based compression strategy using PROMPT-SAW approach.

This strategy implements the PROMPT-SAW (PageRank-Ordered Multi-Path Traversal
with Semantic Augmented Weighting) approach for compressing Excel dependency
graphs into LLM context.

Key Features:
1. Dependency graph construction from Excel formulas
2. PageRank-based importance scoring
3. Subgraph selection within token budget
4. Semantic edge addition for richer relationships
5. Graph serialization optimized for LLM understanding
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import networkx as nx

from spreadsheet_analyzer.notebook_llm.nap.protocols import CellType, NotebookDocument
from spreadsheet_analyzer.pipeline.types import FormulaNode

from .base import (
    AnalysisFocus,
    AnalysisTask,
    BaseStrategy,
    ContextPackage,
    ResponseFormat,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GraphCompressionConfig:
    """Configuration for graph-based compression."""

    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 100
    pagerank_tol: float = 1e-6

    # Subgraph selection parameters
    min_pagerank_threshold: float = 0.001
    max_nodes_per_component: int = 50
    preserve_circular_refs: bool = True

    # Semantic enrichment
    add_semantic_edges: bool = True
    semantic_similarity_threshold: float = 0.7

    # Token allocation
    graph_token_ratio: float = 0.7  # 70% for graph, 30% for metadata
    max_edge_labels: int = 3  # Max semantic labels per edge


class GraphBasedStrategy(BaseStrategy):
    """Strategy for graph-based compression using PROMPT-SAW approach.

    This strategy is particularly effective for:
    - Understanding complex formula dependencies
    - Identifying critical calculation paths
    - Finding formula bottlenecks and key aggregation points
    - Analyzing circular references and their impact
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the graph-based strategy.

        Args:
            config: Configuration options including PageRank parameters,
                    subgraph selection criteria, and semantic enrichment settings
        """
        super().__init__(config)

        # Load configuration with defaults
        cfg = self.config
        self.compression_config = GraphCompressionConfig(
            pagerank_alpha=cfg.get("pagerank_alpha", 0.85),
            pagerank_max_iter=cfg.get("pagerank_max_iter", 100),
            pagerank_tol=cfg.get("pagerank_tol", 1e-6),
            min_pagerank_threshold=cfg.get("min_pagerank_threshold", 0.001),
            max_nodes_per_component=cfg.get("max_nodes_per_component", 50),
            preserve_circular_refs=cfg.get("preserve_circular_refs", True),
            add_semantic_edges=cfg.get("add_semantic_edges", True),
            semantic_similarity_threshold=cfg.get("semantic_similarity_threshold", 0.7),
            graph_token_ratio=cfg.get("graph_token_ratio", 0.7),
            max_edge_labels=cfg.get("max_edge_labels", 3),
        )

    def prepare_context(self, notebook: NotebookDocument, focus: AnalysisFocus, token_budget: int) -> ContextPackage:
        """Prepare graph-based context using PROMPT-SAW approach.

        Args:
            notebook: The notebook to analyze
            focus: The analysis focus area
            token_budget: Maximum tokens to use

        Returns:
            Context package with compressed graph representation
        """
        logger.info(
            "Preparing graph-based context for notebook with %d cells, focus: %s, budget: %d tokens",
            len(notebook.cells),
            focus.value,
            token_budget,
        )

        # Step 1: Build dependency graph from formulas
        formula_graph = self._build_dependency_graph(notebook)

        if not formula_graph:
            # No formulas found, fall back to basic context
            return self._prepare_fallback_context(notebook, focus, token_budget)

        # Step 2: Compute PageRank scores
        pagerank_scores = self._compute_pagerank(formula_graph)

        # Step 3: Add semantic edges if enabled
        if self.compression_config.add_semantic_edges:
            formula_graph = self._add_semantic_edges(formula_graph, notebook)

        # Step 4: Select subgraph within token budget
        graph_budget = int(token_budget * self.compression_config.graph_token_ratio)
        selected_subgraph = self._select_subgraph(formula_graph, pagerank_scores, graph_budget, focus)

        # Step 5: Serialize graph for LLM
        serialized_graph = self._serialize_graph(selected_subgraph)

        # Step 6: Extract metadata
        metadata = self._extract_graph_metadata(formula_graph, selected_subgraph)

        # Step 7: Generate focus hints
        focus_hints = self._generate_graph_hints(focus, selected_subgraph, metadata)

        # Estimate token count
        token_count = self._estimate_tokens(serialized_graph, metadata)

        return ContextPackage(
            cells=serialized_graph["nodes"],
            metadata=metadata,
            focus_hints=focus_hints,
            token_count=token_count,
            compression_method="graph_based_prompt_saw",
            additional_data={
                "graph": serialized_graph,
                "pagerank_scores": pagerank_scores,
                "subgraph_stats": {
                    "total_nodes": len(formula_graph.nodes),
                    "selected_nodes": len(selected_subgraph.nodes),
                    "total_edges": len(formula_graph.edges),
                    "selected_edges": len(selected_subgraph.edges),
                },
            },
        )

    def format_prompt(self, context: ContextPackage, task: AnalysisTask) -> str:
        """Format graph-based prompt for LLM.

        Args:
            context: Prepared context package
            task: Analysis task to perform

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        # System context
        prompt_parts.append(
            "You are analyzing an Excel spreadsheet's formula dependency graph. "
            "This graph shows how cells reference each other through formulas, "
            "with PageRank scores indicating the most important cells in the calculation flow."
        )

        # Task description
        prompt_parts.append(f"\nTask: {task.description}")
        if task.focus_area:
            prompt_parts.append(f"Focus Area: {task.focus_area}")

        # Graph overview
        graph_data = context.additional_data.get("graph", {})
        stats = context.additional_data.get("subgraph_stats", {})

        prompt_parts.append("\n## Dependency Graph Overview")
        prompt_parts.append(f"- Total formula cells: {stats.get('total_nodes', 0)}")
        prompt_parts.append(f"- Shown in context: {stats.get('selected_nodes', 0)} most important cells")
        prompt_parts.append(f"- Total dependencies: {stats.get('total_edges', 0)}")

        # Critical paths and patterns
        if "critical_paths" in context.metadata:
            prompt_parts.append("\n## Critical Calculation Paths")
            for i, path in enumerate(context.metadata["critical_paths"][:3], 1):
                prompt_parts.append(f"\n### Path {i}")
                prompt_parts.append(" → ".join(path))

        # Circular references
        if "circular_references" in context.metadata:
            prompt_parts.append("\n## Circular References Detected")
            for cycle in context.metadata["circular_references"]:
                prompt_parts.append(f"- Cycle: {' → '.join(cycle)}")

        # Graph representation
        prompt_parts.append("\n## Dependency Graph (PROMPT-SAW Compressed)")
        prompt_parts.append(
            "Nodes show formula cells with their PageRank importance scores. "
            "Edges show dependencies with semantic labels where available."
        )

        # Node listing with formulas
        nodes = graph_data.get("nodes", [])
        if nodes:
            prompt_parts.append("\n### Key Formula Cells (by importance)")
            for node in nodes[:20]:  # Top 20 nodes
                cell_ref = f"{node.get('sheet', '?')}.{node.get('cell', '?')}"
                pagerank = node.get("pagerank", 0)
                formula = node.get("formula", "")

                prompt_parts.append(f"\n**{cell_ref}** (importance: {pagerank:.4f})")
                prompt_parts.append(f"```\n{formula}\n```")

                # Show key dependencies
                deps = node.get("dependencies", [])
                if deps:
                    prompt_parts.append(f"Dependencies: {', '.join(deps[:5])}")
                    if len(deps) > 5:
                        prompt_parts.append(f"... and {len(deps) - 5} more")

        # Edge patterns
        edges = graph_data.get("edges", [])
        if edges and self.compression_config.add_semantic_edges:
            prompt_parts.append("\n### Key Relationships")
            semantic_edges = [e for e in edges if e.get("semantic_type")]
            for edge in semantic_edges[:10]:
                source = edge.get("source", "?")
                target = edge.get("target", "?")
                edge_type = edge.get("semantic_type", "DEPENDS_ON")
                prompt_parts.append(f"- {source} {edge_type} {target}")

        # Focus hints
        if context.focus_hints:
            prompt_parts.append("\n## Analysis Hints")
            for hint in context.focus_hints:
                prompt_parts.append(f"- {hint}")

        # Expected output format
        prompt_parts.append(f"\n## Expected Output Format: {task.expected_format.value}")

        if task.expected_format == ResponseFormat.JSON:
            prompt_parts.append(
                "\nProvide your analysis as a JSON object with the following structure:\n"
                "```json\n"
                "{\n"
                '  "key_calculation_flows": [{"path": [...], "purpose": "...", "complexity": "high/medium/low"}],\n'
                '  "bottleneck_cells": [{"cell": "...", "reason": "...", "impact": "..."}],\n'
                '  "formula_patterns": [{"pattern": "...", "cells": [...], "implications": "..."}],\n'
                '  "improvement_opportunities": [{"issue": "...", "cells": [...], "suggestion": "..."}],\n'
                '  "risk_assessment": {"circular_refs": [...], "volatile_formulas": [...], "external_deps": [...]}\n'
                "}\n"
                "```"
            )

        return "\n".join(prompt_parts)

    def _build_dependency_graph(self, notebook: NotebookDocument) -> nx.DiGraph:
        """Build dependency graph from notebook formulas.

        Args:
            notebook: The notebook to analyze

        Returns:
            NetworkX directed graph of formula dependencies
        """
        graph = nx.DiGraph()

        # Extract formula information from code cells
        for cell_idx, cell in enumerate(notebook.cells):
            if cell.cell_type != CellType.CODE:
                continue

            # Look for formula analysis results in cell
            source = cell.source
            if "formula_analysis" not in source and "dependency_graph" not in source:
                continue

            # Parse formula nodes from cell output
            # This is simplified - in production would parse actual execution results
            formula_nodes = self._extract_formula_nodes(cell, cell_idx)

            for node in formula_nodes:
                node_id = f"{node.sheet}.{node.cell}"

                # Add node with attributes
                graph.add_node(
                    node_id,
                    sheet=node.sheet,
                    cell=node.cell,
                    formula=node.formula,
                    volatile=node.volatile,
                    external=node.external,
                    complexity_score=node.complexity_score,
                    cell_metadata=node.cell_metadata or {},
                )

                # Add edges with metadata
                for dep in node.dependencies:
                    edge_data: dict[str, Any] = {"weight": 1.0}

                    # Add semantic edge data if available
                    if node.edge_labels and dep in node.edge_labels:
                        edge_meta = node.edge_labels[dep]
                        edge_data.update(
                            {
                                "edge_type": edge_meta.edge_type,
                                "function_name": edge_meta.function_name,
                                "argument_position": edge_meta.argument_position,
                                "weight": edge_meta.weight,
                                "metadata": edge_meta.metadata,
                            }
                        )

                    graph.add_edge(node_id, dep, **edge_data)

        logger.info(
            "Built dependency graph with %d nodes and %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        return graph

    def _extract_formula_nodes(self, cell: Any, cell_idx: int) -> list[FormulaNode]:  # noqa: ARG002
        """Extract FormulaNode objects from a notebook cell.

        This is a simplified implementation that looks for formula patterns
        in the cell source. In production, this would parse actual cell outputs.

        Args:
            cell: The notebook cell
            cell_idx: Cell index in notebook

        Returns:
            List of FormulaNode objects found in cell
        """
        nodes = []

        # Simple pattern matching for formula definitions
        # In production, would parse structured output
        lines = cell.source.split("\n")
        current_sheet = "Sheet1"  # Default

        for line in lines:
            # Look for sheet indicators
            if "sheet:" in line.lower() or "worksheet:" in line.lower():
                parts = line.split(":")
                if len(parts) > 1:
                    current_sheet = parts[1].strip().strip("'\"")

            # Look for formula definitions
            if "=" in line and any(func in line.upper() for func in ["SUM", "VLOOKUP", "IF", "COUNT"]):
                # Extract cell reference (simplified)
                parts = line.split("=", 1)
                if len(parts) == 2:
                    cell_ref = parts[0].strip().upper()
                    formula = parts[1].strip()

                    # Extract dependencies (simplified - looks for cell patterns)
                    deps = self._extract_dependencies(formula)

                    # Create formula node
                    node = FormulaNode(
                        sheet=current_sheet,
                        cell=cell_ref,
                        formula=formula,
                        dependencies=frozenset(f"{current_sheet}.{d}" for d in deps),
                        volatile="RAND" in formula.upper() or "NOW" in formula.upper(),
                        external="[" in formula,
                        complexity_score=len(deps) + formula.count("("),
                    )
                    nodes.append(node)

        return nodes

    def _extract_dependencies(self, formula: str) -> list[str]:
        """Extract cell references from a formula string.

        Simplified implementation - in production would use proper formula parser.

        Args:
            formula: Excel formula string

        Returns:
            List of cell references found
        """
        import re

        # Simple regex for cell references
        cell_pattern = r"\b[A-Z]+\d+\b"
        return re.findall(cell_pattern, formula)

    def _compute_pagerank(self, graph: nx.DiGraph) -> dict[str, float]:
        """Compute PageRank scores for all nodes in the graph.

        Args:
            graph: The dependency graph

        Returns:
            Dictionary mapping node IDs to PageRank scores
        """
        if graph.number_of_nodes() == 0:
            return {}

        try:
            # Compute PageRank with configured parameters
            pagerank = nx.pagerank(
                graph,
                alpha=self.compression_config.pagerank_alpha,
                max_iter=self.compression_config.pagerank_max_iter,
                tol=self.compression_config.pagerank_tol,
            )

            # Normalize scores
            max_score = max(pagerank.values()) if pagerank else 1.0
            if max_score > 0:
                pagerank = {k: v / max_score for k, v in pagerank.items()}

            logger.info(
                "Computed PageRank for %d nodes, top score: %.4f",
                len(pagerank),
                max(pagerank.values()) if pagerank else 0,
            )

        except Exception as e:
            logger.warning("Failed to compute PageRank: %s", e)
            # Return uniform scores as fallback
            uniform_score = 1.0 / graph.number_of_nodes()
            return dict.fromkeys(graph.nodes(), uniform_score)
        else:
            return pagerank  # type: ignore[no-any-return]

    def _add_semantic_edges(self, graph: nx.DiGraph, notebook: NotebookDocument) -> nx.DiGraph:  # noqa: ARG002
        """Add semantic edges to enrich the graph representation.

        Args:
            graph: The dependency graph
            notebook: The notebook for additional context

        Returns:
            Graph with semantic edges added
        """
        # Create a copy to avoid modifying original
        enriched_graph = graph.copy()

        # Group nodes by sheet for efficiency
        sheet_nodes: defaultdict[str, list[str]] = defaultdict(list)
        for node in graph.nodes():
            sheet = graph.nodes[node].get("sheet", "")
            sheet_nodes[sheet].append(node)

        # Add semantic edges within sheets
        for _sheet, nodes in sheet_nodes.items():
            # Look for patterns that suggest relationships
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i + 1 :]:
                    # Check if nodes are in proximity (simplified)
                    cell1 = graph.nodes[node1].get("cell", "")
                    cell2 = graph.nodes[node2].get("cell", "")

                    if (
                        self._are_cells_adjacent(cell1, cell2)
                        and not enriched_graph.has_edge(node1, node2)
                        and not enriched_graph.has_edge(node2, node1)
                    ):
                        enriched_graph.add_edge(
                            node1,
                            node2,
                            edge_type="ADJACENT_TO",
                            semantic=True,
                            weight=0.5,
                        )

                    # Check for similar formulas
                    formula1 = graph.nodes[node1].get("formula", "")
                    formula2 = graph.nodes[node2].get("formula", "")

                    similarity = self._calculate_formula_similarity(formula1, formula2)
                    if (
                        similarity > self.compression_config.semantic_similarity_threshold
                        and not enriched_graph.has_edge(node1, node2)
                        and not enriched_graph.has_edge(node2, node1)
                    ):
                        enriched_graph.add_edge(
                            node1,
                            node2,
                            edge_type="SIMILAR_FORMULA",
                            semantic=True,
                            weight=similarity,
                            similarity=similarity,
                        )

        return enriched_graph

    def _are_cells_adjacent(self, cell1: str, cell2: str) -> bool:
        """Check if two cell references are adjacent.

        Args:
            cell1: First cell reference (e.g., "A1")
            cell2: Second cell reference (e.g., "A2")

        Returns:
            True if cells are adjacent
        """
        import re

        # Parse cell references
        match1 = re.match(r"([A-Z]+)(\d+)", cell1)
        match2 = re.match(r"([A-Z]+)(\d+)", cell2)

        if not match1 or not match2:
            return False

        col1, row1 = match1.groups()
        col2, row2 = match2.groups()

        row1, row2 = int(row1), int(row2)

        # Check if adjacent (same column, adjacent rows or same row, adjacent columns)
        if col1 == col2 and abs(row1 - row2) == 1:
            return True

        # Check column adjacency (simplified - only works for single letters)
        if row1 == row2 and len(col1) == 1 and len(col2) == 1:
            return abs(ord(col1) - ord(col2)) == 1

        return False

    def _calculate_formula_similarity(self, formula1: str, formula2: str) -> float:
        """Calculate similarity between two formulas.

        Simplified implementation using string similarity.

        Args:
            formula1: First formula
            formula2: Second formula

        Returns:
            Similarity score between 0 and 1
        """
        if not formula1 or not formula2:
            return 0.0

        # Normalize formulas
        f1 = formula1.upper().replace(" ", "")
        f2 = formula2.upper().replace(" ", "")

        # Extract function names
        import re

        funcs1 = set(re.findall(r"[A-Z]+(?=\()", f1))
        funcs2 = set(re.findall(r"[A-Z]+(?=\()", f2))

        if funcs1 and funcs2:
            # Jaccard similarity of functions
            intersection = len(funcs1 & funcs2)
            union = len(funcs1 | funcs2)
            return intersection / union if union > 0 else 0.0

        # Simple character-based similarity as fallback
        common = sum(c1 == c2 for c1, c2 in zip(f1, f2, strict=False))
        return common / max(len(f1), len(f2))

    def _select_subgraph(
        self, graph: nx.DiGraph, pagerank_scores: dict[str, float], token_budget: int, focus: AnalysisFocus
    ) -> nx.DiGraph:
        """Select subgraph within token budget using PROMPT-SAW approach.

        Args:
            graph: Full dependency graph
            pagerank_scores: PageRank scores for all nodes
            token_budget: Token budget for graph representation
            focus: Analysis focus for prioritization

        Returns:
            Selected subgraph
        """
        selected_nodes = set()

        # Priority 1: Preserve circular references if configured
        if self.compression_config.preserve_circular_refs:
            try:
                cycles = list(nx.simple_cycles(graph))
                for cycle in cycles[:5]:  # Limit to first 5 cycles
                    selected_nodes.update(cycle)
                    if len(selected_nodes) * 50 > token_budget:  # Rough estimate
                        break
            except Exception as e:
                logger.debug("No cycles found: %s", e)

        # Priority 2: Select high PageRank nodes
        sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

        for node, score in sorted_nodes:
            if score < self.compression_config.min_pagerank_threshold:
                break

            selected_nodes.add(node)

            # Add immediate neighbors for context
            selected_nodes.update(graph.predecessors(node))
            selected_nodes.update(graph.successors(node))

            # Check budget (rough estimate: 50 tokens per node)
            if len(selected_nodes) * 50 > token_budget:
                break

        # Priority 3: Focus-specific selection
        if focus == AnalysisFocus.FORMULAS:
            # Add nodes with complex formulas
            complex_nodes = [n for n in graph.nodes() if graph.nodes[n].get("complexity_score", 0) > 5]
            for node in complex_nodes[:10]:
                if len(selected_nodes) * 50 < token_budget:
                    selected_nodes.add(node)

        # Create subgraph
        subgraph = graph.subgraph(selected_nodes).copy()

        logger.info(
            "Selected subgraph with %d nodes (%.1f%% of total)",
            len(subgraph.nodes()),
            100 * len(subgraph.nodes()) / max(len(graph.nodes()), 1),
        )

        return subgraph

    def _serialize_graph(self, graph: nx.DiGraph) -> dict[str, Any]:
        """Serialize graph for LLM consumption.

        Args:
            graph: The graph to serialize

        Returns:
            Serialized graph representation
        """
        serialized = {
            "nodes": [],
            "edges": [],
            "stats": {
                "node_count": graph.number_of_nodes(),
                "edge_count": graph.number_of_edges(),
                "density": nx.density(graph),
            },
        }

        # Serialize nodes with attributes
        nodes_list: list[dict[str, Any]] = []
        for node in graph.nodes():
            node_data = graph.nodes[node]
            nodes_list.append(
                {
                    "id": node,
                    "sheet": node_data.get("sheet", ""),
                    "cell": node_data.get("cell", ""),
                    "formula": node_data.get("formula", ""),
                    "pagerank": node_data.get("pagerank", 0),
                    "volatile": node_data.get("volatile", False),
                    "external": node_data.get("external", False),
                    "complexity_score": node_data.get("complexity_score", 0),
                    "dependencies": list(graph.predecessors(node)),
                    "dependents": list(graph.successors(node)),
                }
            )
        serialized["nodes"] = nodes_list

        # Serialize edges with semantic information
        for source, target, data in graph.edges(data=True):
            edge_info = {
                "source": source,
                "target": target,
                "weight": data.get("weight", 1.0),
            }

            # Add semantic information if available
            if data.get("semantic", False):
                edge_info["semantic_type"] = data.get("edge_type", "UNKNOWN")
                edge_info["semantic"] = True

                if "similarity" in data:
                    edge_info["similarity"] = data["similarity"]

            serialized["edges"].append(edge_info)

        return serialized

    def _extract_graph_metadata(self, full_graph: nx.DiGraph, subgraph: nx.DiGraph) -> dict[str, Any]:
        """Extract metadata about the graph structure.

        Args:
            full_graph: Complete dependency graph
            subgraph: Selected subgraph

        Returns:
            Graph metadata dictionary
        """
        metadata = {
            "graph_stats": {
                "total_nodes": full_graph.number_of_nodes(),
                "total_edges": full_graph.number_of_edges(),
                "selected_nodes": subgraph.number_of_nodes(),
                "selected_edges": subgraph.number_of_edges(),
                "coverage": subgraph.number_of_nodes() / max(full_graph.number_of_nodes(), 1),
            }
        }

        # Find critical paths (longest paths in DAG components)
        critical_paths = []
        try:
            # Find DAG components
            if nx.is_directed_acyclic_graph(subgraph):
                dag = subgraph
            else:
                # Remove cycles for path analysis
                dag = subgraph.copy()
                cycles = list(nx.simple_cycles(dag))
                for cycle in cycles:
                    if len(cycle) > 1:
                        dag.remove_edge(cycle[-1], cycle[0])

            # Find longest paths from sources
            for source in [n for n in dag.nodes() if dag.in_degree(n) == 0]:
                for target in [n for n in dag.nodes() if dag.out_degree(n) == 0]:
                    try:
                        paths = list(nx.all_simple_paths(dag, source, target, cutoff=10))
                        if paths:
                            longest = max(paths, key=len)
                            if len(longest) > 2:
                                critical_paths.append(longest)
                    except nx.NetworkXNoPath:
                        continue

            # Sort by length and keep top paths
            critical_paths.sort(key=len, reverse=True)
            metadata["critical_paths"] = critical_paths[:5]

        except Exception as e:
            logger.debug("Error finding critical paths: %s", e)

        # Detect circular references
        try:
            cycles = list(nx.simple_cycles(subgraph))
            if cycles:
                metadata["circular_references"] = [list(cycle) for cycle in cycles[:5]]
        except Exception as e:
            logger.debug("Error detecting cycles: %s", e)

        # Identify hub nodes (high degree)
        degree_centrality = nx.degree_centrality(subgraph)
        hubs = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        metadata["hub_nodes"] = [{"node": node, "centrality": cent} for node, cent in hubs]

        # Component analysis
        if subgraph.number_of_nodes() > 0:
            weak_components = list(nx.weakly_connected_components(subgraph))
            metadata["component_count"] = len(weak_components)
            metadata["largest_component_size"] = len(max(weak_components, key=len))

        return metadata

    def _generate_graph_hints(self, focus: AnalysisFocus, subgraph: nx.DiGraph, metadata: dict[str, Any]) -> list[str]:  # noqa: ARG002
        """Generate analysis hints based on graph structure.

        Args:
            focus: The analysis focus
            subgraph: Selected subgraph
            metadata: Graph metadata

        Returns:
            List of analysis hints
        """
        hints = []

        # Focus-specific hints
        if focus == AnalysisFocus.FORMULAS:
            hints.append("Analyze the formula dependency chains to understand calculation flow")
            hints.append("Look for complex formulas that might be simplified or optimized")
            hints.append("Identify formula patterns that could be replaced with more efficient approaches")

            if metadata.get("circular_references"):
                hints.append("Pay special attention to circular references and their impact")

        elif focus == AnalysisFocus.RELATIONSHIPS:
            hints.append("Map the data flow through the formula network")
            hints.append("Identify key aggregation points where multiple calculations converge")
            hints.append("Look for isolated components that might represent independent analyses")

        elif focus == AnalysisFocus.VALIDATION:
            hints.append("Check for formulas that might produce errors under certain conditions")
            hints.append("Identify fragile dependency chains that could break easily")
            hints.append("Look for external references that might not always be available")

        # Structure-based hints
        if metadata.get("hub_nodes"):
            hints.append("Focus on hub cells with high connectivity - they are critical to the calculation flow")

        if metadata.get("component_count", 0) > 1:
            hints.append(
                f"The graph has {metadata['component_count']} disconnected components - "
                "consider if they should be connected"
            )

        coverage = metadata.get("graph_stats", {}).get("coverage", 0)
        if coverage < 0.5:
            hints.append(f"Showing {coverage:.0%} of all formula cells - the most important ones based on PageRank")

        return hints

    def _prepare_fallback_context(
        self,
        notebook: NotebookDocument,
        focus: AnalysisFocus,
        token_budget: int,  # noqa: ARG002
    ) -> ContextPackage:
        """Prepare fallback context when no formulas are found.

        Args:
            notebook: The notebook to analyze
            focus: The analysis focus
            token_budget: Token budget

        Returns:
            Basic context package
        """
        logger.info("No formulas found, preparing fallback context")

        # Extract basic cell information
        cells = []
        for i, cell in enumerate(notebook.cells[:20]):  # Limit to first 20 cells
            cells.append(
                {
                    "id": i,
                    "cell_type": cell.cell_type.value,
                    "source": cell.source[:200],  # Truncate
                }
            )

        return ContextPackage(
            cells=cells,
            metadata={
                "total_cells": len(notebook.cells),
                "cell_types": {
                    "code": sum(1 for c in notebook.cells if c.cell_type == CellType.CODE),
                    "markdown": sum(1 for c in notebook.cells if c.cell_type == CellType.MARKDOWN),
                },
            },
            focus_hints=["No formulas detected - analyze general notebook structure instead"],
            token_count=token_budget // 2,  # Conservative estimate
            compression_method="fallback",
        )

    def _estimate_tokens(self, serialized_graph: dict[str, Any], metadata: dict[str, Any]) -> int:
        """Estimate token count for serialized graph.

        Args:
            serialized_graph: Serialized graph data
            metadata: Graph metadata

        Returns:
            Estimated token count
        """
        # Rough estimation: 4 characters per token
        char_count = 0

        # Count node data
        for node in serialized_graph.get("nodes", []):
            char_count += len(str(node))

        # Count edge data
        for edge in serialized_graph.get("edges", []):
            char_count += len(str(edge))

        # Count metadata
        char_count += len(str(metadata))

        return char_count // 4
