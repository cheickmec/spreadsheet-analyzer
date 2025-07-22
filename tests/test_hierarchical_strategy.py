"""Tests for hierarchical analysis strategy."""

import pytest

from spreadsheet_analyzer.notebook_llm.nap.protocols import Cell, CellType, NotebookDocument
from spreadsheet_analyzer.notebook_llm.strategies.base import (
    AnalysisFocus,
    AnalysisTask,
    ContextPackage,
    ResponseFormat,
)
from spreadsheet_analyzer.notebook_llm.strategies.hierarchical import HierarchicalStrategy


class TestHierarchicalStrategy:
    """Tests for HierarchicalStrategy class."""

    @pytest.fixture
    def strategy(self):
        """Create strategy instance."""
        return HierarchicalStrategy()

    @pytest.fixture
    def notebook(self):
        """Create test notebook with multiple sheets."""
        cells = [
            Cell(
                id="1",
                cell_type=CellType.MARKDOWN,
                source="# Sheet Analysis\nOverview of sheets",
                metadata={"sheet": "Overview"},
            ),
            Cell(
                id="2",
                cell_type=CellType.CODE,
                source="# Sheet1 formulas\nA1: =SUM(B1:B10)\nB1: =C1*2",
                metadata={"sheet": "Sheet1"},
            ),
            Cell(
                id="3",
                cell_type=CellType.CODE,
                source="# Sheet2 formulas\nA1: =Sheet1!A1 + 10\nB1: =VLOOKUP(A1, Sheet1!A:B, 2)",
                metadata={"sheet": "Sheet2"},
            ),
            Cell(
                id="4",
                cell_type=CellType.CODE,
                source="# Sheet3 data\nContains raw data values",
                metadata={"sheet": "Sheet3"},
            ),
        ]
        return NotebookDocument(
            id="test",
            cells=cells,
            metadata={"sheets": ["Sheet1", "Sheet2", "Sheet3"]},
            kernel_spec={"name": "python3"},
            language_info={"name": "python"},
        )

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert isinstance(strategy, HierarchicalStrategy)
        assert hasattr(strategy, "config")

    def test_prepare_context_basic(self, strategy, notebook):
        """Test basic context preparation."""
        context = strategy.prepare_context(notebook, AnalysisFocus.STRUCTURE, 1000)

        assert isinstance(context, ContextPackage)
        assert context.compression_method == "hierarchical"
        assert len(context.cells) > 0
        assert "hierarchy" in context.additional_data

    def test_build_hierarchy(self, strategy, notebook):
        """Test building hierarchical structure."""
        hierarchy = strategy._build_hierarchy(notebook)

        assert isinstance(hierarchy, dict)
        assert "sheets" in hierarchy
        assert "workbook_summary" in hierarchy
        assert "cross_sheet_references" in hierarchy

    def test_extract_sheets(self, strategy, notebook):
        """Test extracting sheet information."""
        sheets = strategy._extract_sheets(notebook)

        assert isinstance(sheets, dict)
        # Should have sheets from notebook
        assert len(sheets) > 0

    def test_identify_cross_sheet_references(self, strategy, notebook):
        """Test identifying cross-sheet references."""
        refs = strategy._identify_cross_sheet_references(notebook)

        assert isinstance(refs, list)
        # Sheet2 has references to Sheet1
        assert any("Sheet1" in str(ref) for ref in refs)

    def test_create_workbook_summary(self, strategy):
        """Test creating workbook summary."""
        sheets = {
            "Sheet1": [
                {"source": "A1: =SUM(B:B)", "metadata": {"formula": True}},
                {"source": "B1: 100", "metadata": {"value": 100}},
            ],
            "Sheet2": [
                {"source": "A1: =Sheet1!A1", "metadata": {"formula": True}},
            ],
        }

        summary = strategy._create_workbook_summary(sheets, [])

        assert isinstance(summary, dict)
        assert "total_sheets" in summary
        assert summary["total_sheets"] == 2
        assert "total_cells" in summary

    def test_hierarchical_summarization(self, strategy):
        """Test hierarchical summarization at different levels."""
        cells = [
            {
                "id": f"cell_{i}",
                "source": f"A{i}: =SUM(B{i}:C{i})",
                "metadata": {"formula": True},
            }
            for i in range(1, 11)
        ]

        # Cell level summary
        cell_summary = strategy._summarize_at_level(cells[:5], "cell", 100)
        assert isinstance(cell_summary, dict)
        assert "summary" in cell_summary

        # Sheet level summary
        sheet_summary = strategy._summarize_at_level(cells, "sheet", 200)
        assert isinstance(sheet_summary, dict)
        assert "summary" in sheet_summary

    def test_compress_with_hierarchy(self, strategy, notebook):
        """Test compression using hierarchy."""
        hierarchy = strategy._build_hierarchy(notebook)
        compressed = strategy._compress_with_hierarchy(hierarchy, 500)

        assert isinstance(compressed, dict)
        assert "selected_cells" in compressed
        assert "summaries" in compressed

    def test_format_prompt(self, strategy):
        """Test prompt formatting."""
        context = ContextPackage(
            cells=[{"sheet": "Sheet1", "content": "A1: =SUM(B:B)"}],
            metadata={"total_sheets": 3, "total_cells": 100},
            focus_hints=["Focus on cross-sheet dependencies"],
            token_count=500,
            compression_method="hierarchical",
            additional_data={
                "hierarchy": {
                    "workbook_summary": {"sheets": 3, "formulas": 25},
                    "sheet_summaries": {
                        "Sheet1": {"formulas": 10, "data_cells": 50},
                    },
                },
            },
        )

        task = AnalysisTask(
            name="analyze_structure",
            description="Analyze workbook structure",
            focus=AnalysisFocus.STRUCTURE,
            expected_format=ResponseFormat.JSON,
        )

        prompt = strategy.format_prompt(context, task)

        assert "hierarchical" in prompt.lower()
        assert "Sheet1" in prompt
        assert "JSON" in prompt

    def test_handle_large_workbook(self, strategy):
        """Test handling large workbook with many cells."""
        # Create a large notebook
        large_cells = []
        for sheet_num in range(5):
            for cell_num in range(20):
                large_cells.append(
                    Cell(
                        id=f"s{sheet_num}_c{cell_num}",
                        cell_type=CellType.CODE,
                        source=f"Cell {cell_num}: =FORMULA({cell_num})",
                        metadata={"sheet": f"Sheet{sheet_num}"},
                    )
                )

        large_notebook = NotebookDocument(
            id="large",
            cells=large_cells,
            metadata={"sheets": [f"Sheet{i}" for i in range(5)]},
            kernel_spec={"name": "python3"},
            language_info={"name": "python"},
        )

        context = strategy.prepare_context(large_notebook, AnalysisFocus.OVERVIEW, 1000)

        assert isinstance(context, ContextPackage)
        assert context.compression_method == "hierarchical"
        # Should have compressed the content
        assert context.token_count <= 1000

    def test_progressive_detail_levels(self, strategy, notebook):
        """Test progressive detail based on token budget."""
        # Small budget should give high-level summary
        small_context = strategy.prepare_context(notebook, AnalysisFocus.OVERVIEW, 200)
        assert small_context.token_count <= 200

        # Large budget should give more detail
        large_context = strategy.prepare_context(notebook, AnalysisFocus.OVERVIEW, 2000)
        assert large_context.token_count > small_context.token_count
        assert len(large_context.cells) >= len(small_context.cells)

    def test_focus_specific_compression(self, strategy, notebook):
        """Test different compression for different focus areas."""
        # Formula focus should prioritize formula cells
        formula_context = strategy.prepare_context(notebook, AnalysisFocus.FORMULAS, 500)
        formula_cells = [c for c in formula_context.cells if "=" in str(c.get("source", ""))]

        # Data focus should include more data cells
        data_context = strategy.prepare_context(notebook, AnalysisFocus.DATA, 500)

        # Different focus should result in different selections
        assert formula_context.cells != data_context.cells
