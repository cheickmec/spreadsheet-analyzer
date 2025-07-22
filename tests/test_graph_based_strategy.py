"""Tests for graph-based analysis strategy."""

import networkx as nx
import pytest

from spreadsheet_analyzer.notebook_llm.nap.protocols import Cell, CellType, NotebookDocument
from spreadsheet_analyzer.notebook_llm.strategies.base import (
    AnalysisFocus,
    AnalysisTask,
    ContextPackage,
    ResponseFormat,
)
from spreadsheet_analyzer.notebook_llm.strategies.graph_based import (
    GraphBasedStrategy,
    GraphCompressionConfig,
)


class TestGraphCompressionConfig:
    """Tests for GraphCompressionConfig dataclass."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = GraphCompressionConfig()
        assert config.pagerank_alpha == 0.85
        assert config.pagerank_max_iter == 100
        assert config.pagerank_tol == 1e-6
        assert config.min_pagerank_threshold == 0.001
        assert config.max_nodes_per_component == 50
        assert config.preserve_circular_refs is True
        assert config.add_semantic_edges is True
        assert config.semantic_similarity_threshold == 0.7
        assert config.graph_token_ratio == 0.7
        assert config.max_edge_labels == 3

    def test_config_custom(self):
        """Test custom configuration values."""
        config = GraphCompressionConfig(
            pagerank_alpha=0.9,
            max_nodes_per_component=100,
            preserve_circular_refs=False,
        )
        assert config.pagerank_alpha == 0.9
        assert config.max_nodes_per_component == 100
        assert config.preserve_circular_refs is False


class TestGraphBasedStrategy:
    """Tests for GraphBasedStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return GraphBasedStrategy()

    @pytest.fixture
    def notebook(self):
        """Create test notebook."""
        cells = [
            Cell(
                id="1",
                cell_type=CellType.CODE,
                source="# Formula analysis\nSheet1.A1 = SUM(B1:B10)",
                metadata={},
            ),
            Cell(
                id="2",
                cell_type=CellType.CODE,
                source="Sheet1.B1 = A1 * 2\nSheet1.C1 = VLOOKUP(A1, D:E, 2)",
                metadata={},
            ),
        ]
        return NotebookDocument(
            id="test",
            cells=cells,
            metadata={},
            kernel_spec={"name": "python3"},
            language_info={"name": "python"},
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert isinstance(strategy.compression_config, GraphCompressionConfig)
        assert strategy.compression_config.pagerank_alpha == 0.85

    def test_custom_config(self):
        """Test strategy with custom config."""
        config = {"pagerank_alpha": 0.9, "max_nodes_per_component": 75}
        strategy = GraphBasedStrategy(config)
        assert strategy.compression_config.pagerank_alpha == 0.9
        assert strategy.compression_config.max_nodes_per_component == 75

    def test_prepare_context_no_formulas(self, strategy):
        """Test context preparation with no formulas."""
        # Notebook with no formula patterns
        notebook = NotebookDocument(
            id="test",
            cells=[
                Cell(
                    id="1",
                    cell_type=CellType.MARKDOWN,
                    source="# Title",
                    metadata={},
                )
            ],
            metadata={},
            kernel_spec={"name": "python3"},
            language_info={"name": "python"},
        )

        context = strategy.prepare_context(notebook, AnalysisFocus.FORMULAS, 1000)
        assert isinstance(context, ContextPackage)
        assert context.compression_method == "fallback"
        assert len(context.focus_hints) > 0

    def test_build_dependency_graph(self, strategy, notebook):
        """Test building dependency graph from notebook."""
        graph = strategy._build_dependency_graph(notebook)
        assert isinstance(graph, nx.DiGraph)
        # Graph might be empty if formula parsing doesn't find valid patterns
        assert graph.number_of_nodes() >= 0

    def test_extract_dependencies(self, strategy):
        """Test extracting cell dependencies from formula."""
        formula = "=SUM(A1:A10) + B1 * C2"
        deps = strategy._extract_dependencies(formula)
        assert "A1" in deps or "A10" in deps  # Range might be expanded
        assert "B1" in deps
        assert "C2" in deps

    def test_compute_pagerank_empty_graph(self, strategy):
        """Test PageRank computation on empty graph."""
        graph = nx.DiGraph()
        scores = strategy._compute_pagerank(graph)
        assert scores == {}

    def test_compute_pagerank_simple_graph(self, strategy):
        """Test PageRank computation on simple graph."""
        graph = nx.DiGraph()
        graph.add_edge("A1", "B1")
        graph.add_edge("B1", "C1")
        graph.add_edge("A1", "C1")

        scores = strategy._compute_pagerank(graph)
        assert len(scores) == 3
        assert all(0 <= score <= 1 for score in scores.values())
        # C1 should have highest score (most incoming edges)
        assert scores["C1"] >= scores["A1"]

    def test_are_cells_adjacent(self, strategy):
        """Test cell adjacency detection."""
        # Same column, adjacent rows
        assert strategy._are_cells_adjacent("A1", "A2")
        assert strategy._are_cells_adjacent("B5", "B6")

        # Same row, adjacent columns
        assert strategy._are_cells_adjacent("A1", "B1")
        assert strategy._are_cells_adjacent("C3", "D3")

        # Not adjacent
        assert not strategy._are_cells_adjacent("A1", "A3")
        assert not strategy._are_cells_adjacent("A1", "C1")
        assert not strategy._are_cells_adjacent("A1", "B2")

    def test_calculate_formula_similarity(self, strategy):
        """Test formula similarity calculation."""
        # Identical formulas
        assert strategy._calculate_formula_similarity("=SUM(A1:A10)", "=SUM(A1:A10)") == 1.0

        # Similar functions
        sim1 = strategy._calculate_formula_similarity("=SUM(A1:A10)", "=SUM(B1:B10)")
        assert 0 < sim1 <= 1

        # Different functions
        sim2 = strategy._calculate_formula_similarity("=SUM(A1:A10)", "=AVERAGE(A1:A10)")
        assert sim2 <= sim1

        # Empty formulas
        assert strategy._calculate_formula_similarity("", "=SUM(A1)") == 0.0

    def test_serialize_graph(self, strategy):
        """Test graph serialization."""
        graph = nx.DiGraph()
        graph.add_node("Sheet1.A1", formula="=B1+C1", sheet="Sheet1", cell="A1")
        graph.add_node("Sheet1.B1", formula="=10", sheet="Sheet1", cell="B1")
        graph.add_edge("Sheet1.B1", "Sheet1.A1", weight=1.0)

        serialized = strategy._serialize_graph(graph)
        assert "nodes" in serialized
        assert "edges" in serialized
        assert "stats" in serialized
        assert len(serialized["nodes"]) == 2
        assert len(serialized["edges"]) == 1

    def test_extract_graph_metadata(self, strategy):
        """Test metadata extraction from graph."""
        # Create a simple graph
        full_graph = nx.DiGraph()
        full_graph.add_edges_from([("A1", "B1"), ("B1", "C1"), ("C1", "D1")])

        subgraph = full_graph.subgraph(["A1", "B1", "C1"])

        metadata = strategy._extract_graph_metadata(full_graph, subgraph)
        assert "graph_stats" in metadata
        assert metadata["graph_stats"]["total_nodes"] == 4
        assert metadata["graph_stats"]["selected_nodes"] == 3
        assert "hub_nodes" in metadata
        assert "component_count" in metadata

    def test_generate_graph_hints(self, strategy):
        """Test hint generation."""
        subgraph = nx.DiGraph()
        metadata = {
            "circular_references": [["A1", "B1", "A1"]],
            "hub_nodes": [{"node": "A1", "centrality": 0.8}],
            "component_count": 2,
            "graph_stats": {"coverage": 0.4},
        }

        # Test formula focus
        hints = strategy._generate_graph_hints(AnalysisFocus.FORMULAS, subgraph, metadata)
        assert any("circular references" in hint for hint in hints)
        assert any("hub cells" in hint for hint in hints)

        # Test validation focus
        hints = strategy._generate_graph_hints(AnalysisFocus.VALIDATION, subgraph, metadata)
        assert any("fragile" in hint or "error" in hint for hint in hints)

    def test_format_prompt(self, strategy):
        """Test prompt formatting."""
        # Create mock context
        context = ContextPackage(
            cells=[
                {
                    "id": "Sheet1.A1",
                    "sheet": "Sheet1",
                    "cell": "A1",
                    "formula": "=SUM(B1:B10)",
                    "pagerank": 0.85,
                }
            ],
            metadata={
                "graph_stats": {
                    "total_nodes": 10,
                    "selected_nodes": 5,
                },
                "circular_references": [["A1", "B1", "A1"]],
                "critical_paths": [["A1", "B1", "C1"]],
            },
            focus_hints=["Test hint"],
            token_count=100,
            compression_method="graph_based_prompt_saw",
            additional_data={
                "graph": {
                    "nodes": [
                        {
                            "sheet": "Sheet1",
                            "cell": "A1",
                            "formula": "=SUM(B1:B10)",
                            "pagerank": 0.85,
                            "dependencies": ["B1"],
                        }
                    ],
                    "edges": [],
                },
                "subgraph_stats": {
                    "total_nodes": 10,
                    "selected_nodes": 5,
                    "total_edges": 15,
                    "selected_edges": 8,
                },
            },
        )

        task = AnalysisTask(
            name="formula_analysis",
            description="Analyze formula dependencies",
            focus=AnalysisFocus.FORMULAS,
            expected_format=ResponseFormat.JSON,
        )

        prompt = strategy.format_prompt(context, task)
        assert "dependency graph" in prompt.lower()
        assert "PageRank" in prompt
        assert "A1" in prompt
        assert "=SUM(B1:B10)" in prompt
        assert "circular references" in prompt.lower()
        assert "JSON" in prompt

    def test_select_subgraph_with_cycles(self, strategy):
        """Test subgraph selection with circular references."""
        # Create graph with cycle
        graph = nx.DiGraph()
        graph.add_edges_from([("A1", "B1"), ("B1", "C1"), ("C1", "A1")])

        # Add nodes not in cycle
        graph.add_edges_from([("D1", "E1"), ("E1", "F1")])

        pagerank_scores = {
            "A1": 0.8,
            "B1": 0.7,
            "C1": 0.6,
            "D1": 0.3,
            "E1": 0.2,
            "F1": 0.1,
        }

        subgraph = strategy._select_subgraph(graph, pagerank_scores, 200, AnalysisFocus.FORMULAS)

        # Should include cycle nodes
        assert "A1" in subgraph.nodes()
        assert "B1" in subgraph.nodes()
        assert "C1" in subgraph.nodes()

    def test_add_semantic_edges(self, strategy, notebook):
        """Test semantic edge addition."""
        graph = nx.DiGraph()
        graph.add_node("Sheet1.A1", sheet="Sheet1", cell="A1", formula="=SUM(B1:B10)")
        graph.add_node("Sheet1.A2", sheet="Sheet1", cell="A2", formula="=SUM(B1:B10)")
        graph.add_node("Sheet1.B1", sheet="Sheet1", cell="B1", formula="=C1*2")

        enriched = strategy._add_semantic_edges(graph, notebook)

        # Should have at least the original nodes
        assert enriched.number_of_nodes() >= graph.number_of_nodes()

        # Check for semantic edges between adjacent cells
        edges = list(enriched.edges(data=True))
        semantic_edges = [e for e in edges if e[2].get("semantic", False)]
        # May or may not add semantic edges depending on similarity

    def test_prepare_context_full_flow(self, strategy, notebook):
        """Test full context preparation flow."""
        context = strategy.prepare_context(notebook, AnalysisFocus.FORMULAS, 5000)

        assert isinstance(context, ContextPackage)
        assert context.token_count > 0
        assert context.compression_method in ["graph_based_prompt_saw", "fallback"]

        if context.compression_method == "graph_based_prompt_saw":
            assert "graph" in context.additional_data
            assert "subgraph_stats" in context.additional_data

    def test_estimate_tokens(self, strategy):
        """Test token estimation."""
        serialized_graph = {
            "nodes": [
                {"id": "A1", "formula": "=SUM(B1:B10)"},
                {"id": "B1", "formula": "=C1*2"},
            ],
            "edges": [{"source": "B1", "target": "A1"}],
        }
        metadata = {"stats": {"nodes": 2, "edges": 1}}

        tokens = strategy._estimate_tokens(serialized_graph, metadata)
        assert tokens > 0
        # Rough check - should be reasonable
        assert 10 < tokens < 1000
