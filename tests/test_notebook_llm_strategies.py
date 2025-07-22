"""Tests for the notebook LLM strategy layer."""

import pytest

from spreadsheet_analyzer.notebook_llm.nap.protocols import Cell, CellType, NotebookDocument
from spreadsheet_analyzer.notebook_llm.strategies import (
    AnalysisFocus,
    AnalysisTask,
    BaseStrategy,
    ContextPackage,
    HierarchicalStrategy,
    ResponseFormat,
    StrategyRegistry,
    get_registry,
    register_strategy,
)


class TestStrategyRegistry:
    """Test the strategy registry functionality."""

    def test_registry_singleton(self):
        """Test that get_registry returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_manual_registration(self):
        """Test manual strategy registration."""
        registry = StrategyRegistry()

        # Register the built-in strategy
        registry.register("hierarchical", HierarchicalStrategy)

        # Verify it's registered
        assert "hierarchical" in registry.list_strategies()

        # Get the strategy
        strategy = registry.get_strategy("hierarchical")
        assert isinstance(strategy, HierarchicalStrategy)

    def test_strategy_info(self):
        """Test getting strategy information."""
        registry = StrategyRegistry()
        registry.register("hierarchical", HierarchicalStrategy)

        info = registry.get_strategy_info("hierarchical")
        assert info["name"] == "hierarchical"
        assert info["class"] == "HierarchicalStrategy"
        assert "hierarchical exploration" in info["doc"].lower()

    def test_unknown_strategy_error(self):
        """Test error when requesting unknown strategy."""
        registry = StrategyRegistry()

        with pytest.raises(ValueError, match="Unknown strategy"):
            registry.get_strategy("nonexistent")


class TestHierarchicalStrategy:
    """Test the hierarchical exploration strategy."""

    @pytest.fixture
    def sample_notebook(self):
        """Create a sample notebook for testing."""
        cells = [
            Cell(
                id="1",
                cell_type=CellType.MARKDOWN,
                source="# Data Analysis\nThis notebook analyzes Excel data.",
                metadata={},
                outputs=[],
            ),
            Cell(
                id="2", cell_type=CellType.CODE, source="import pandas as pd\nimport openpyxl", metadata={}, outputs=[]
            ),
            Cell(
                id="3",
                cell_type=CellType.CODE,
                source="# Load Excel file\ndf = pd.read_excel('data.xlsx')",
                metadata={},
                outputs=[],
            ),
            Cell(
                id="4",
                cell_type=CellType.MARKDOWN,
                source="## Formula Analysis\nAnalyzing Excel formulas",
                metadata={},
                outputs=[],
            ),
            Cell(
                id="5",
                cell_type=CellType.CODE,
                source="# Extract formulas\nformulas = extract_formulas(df)",
                metadata={},
                outputs=[],
            ),
        ]
        return NotebookDocument(
            id="test_notebook_1",
            cells=cells,
            metadata={"name": "test_notebook"},
            kernel_spec={"name": "python3", "display_name": "Python 3"},
            language_info={"name": "python", "version": "3.12"},
        )

    def test_strategy_initialization(self):
        """Test strategy initialization with config."""
        config = {"summarization_algorithm": "abstractive", "compression_ratio": 0.2, "max_sections": 5}
        strategy = HierarchicalStrategy(config)

        assert strategy.summarization_algorithm == "abstractive"
        assert strategy.compression_ratio == 0.2
        assert strategy.max_sections == 5

    def test_prepare_context(self, sample_notebook):
        """Test context preparation."""
        strategy = HierarchicalStrategy()

        context = strategy.prepare_context(sample_notebook, AnalysisFocus.STRUCTURE, token_budget=1000)

        assert isinstance(context, ContextPackage)
        assert context.compression_method == "hierarchical"
        assert "summary" in context.additional_data
        assert "sections" in context.additional_data
        assert context.token_count > 0

    def test_format_prompt(self, sample_notebook):
        """Test prompt formatting."""
        strategy = HierarchicalStrategy()

        # Prepare context first
        context = strategy.prepare_context(sample_notebook, AnalysisFocus.FORMULAS, token_budget=1000)

        # Create a task
        task = AnalysisTask(
            name="formula_analysis",
            description="Analyze Excel formulas in the notebook",
            focus=AnalysisFocus.FORMULAS,
            expected_format=ResponseFormat.JSON,
            focus_area="Formula extraction and validation",
        )

        # Format prompt
        prompt = strategy.format_prompt(context, task)

        assert isinstance(prompt, str)
        assert "Excel" in prompt
        assert "formula" in prompt.lower()
        assert "JSON" in prompt

    def test_section_detection(self, sample_notebook):
        """Test structural section detection."""
        strategy = HierarchicalStrategy({"section_detection_method": "structural"})

        sections = strategy._detect_structural_sections(sample_notebook)

        assert len(sections) == 2  # Two markdown headers
        assert sections[0]["name"] == "Data Analysis"
        assert sections[1]["name"] == "Formula Analysis"

    def test_key_cell_identification(self, sample_notebook):
        """Test identification of key cells based on focus."""
        strategy = HierarchicalStrategy()

        # Test with formula focus
        key_cells = strategy._identify_key_cells(sample_notebook, AnalysisFocus.FORMULAS)

        # Cell at index 3 (id="4") and index 4 (id="5") should be identified as key for formula analysis
        assert 3 in key_cells  # Has "Formula Analysis" header
        assert 4 in key_cells  # Has "formulas" in code

    def test_budget_allocation(self):
        """Test token budget allocation across hierarchy levels."""
        strategy = HierarchicalStrategy()

        hierarchy = {
            "sections": [{"name": "Section 1"}, {"name": "Section 2"}],
            "key_cells": [1, 2, 3],
            "summary": "Test summary",
        }

        allocation = strategy._allocate_budget(hierarchy, 1000)

        assert allocation["summary"] == 100  # 10%
        assert allocation["sections"] == 300  # 30%
        assert allocation["cells"] == 500  # 50%
        assert allocation["metadata"] == 100  # 10%
        assert sum(allocation.values()) == 1000


class TestCustomStrategy:
    """Test creating a custom strategy."""

    def test_custom_strategy_implementation(self):
        """Test implementing a custom strategy."""

        class SimpleStrategy(BaseStrategy):
            """A simple test strategy."""

            def prepare_context(
                self, notebook: NotebookDocument, focus: AnalysisFocus, token_budget: int
            ) -> ContextPackage:
                # Simple implementation - just return first few cells
                cells = [{"id": i, "source": cell.source[:50]} for i, cell in enumerate(notebook.cells[:3])]

                return ContextPackage(
                    cells=cells,
                    metadata={"cell_count": len(notebook.cells)},
                    focus_hints=["This is a simple strategy"],
                    token_count=100,
                    compression_method="simple",
                )

            def format_prompt(self, context: ContextPackage, task: AnalysisTask) -> str:
                return f"Task: {task.description}\nCells: {len(context.cells)}"

        # Register and use the custom strategy
        registry = StrategyRegistry()
        registry.register("simple", SimpleStrategy)

        strategy = registry.get_strategy("simple")
        assert isinstance(strategy, SimpleStrategy)


def test_global_registration():
    """Test global strategy registration function."""
    # Clear the global registry first
    global_registry = get_registry()
    if "test_strategy" in global_registry.list_strategies():
        global_registry.unregister("test_strategy")

    # Define a test strategy
    class TestStrategy(BaseStrategy):
        def prepare_context(self, notebook, focus, token_budget):
            return ContextPackage([], {}, [], 0)

        def format_prompt(self, context, task):
            return "test"

    # Register globally
    register_strategy("test_strategy", TestStrategy)

    # Verify it's registered
    assert "test_strategy" in get_registry().list_strategies()

    # Clean up
    get_registry().unregister("test_strategy")
