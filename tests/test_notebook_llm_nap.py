"""Tests for NAP (Notebook Analysis Protocol) components."""

from spreadsheet_analyzer.notebook_llm.nap.protocols import (
    Cell,
    CellExecutionResult,
    CellSelector,
    CellType,
    NotebookDocument,
)


class TestCellType:
    """Tests for CellType enum."""

    def test_cell_types(self):
        """Test cell type values."""
        assert CellType.CODE.value == "code"
        assert CellType.MARKDOWN.value == "markdown"
        assert CellType.RAW.value == "raw"


class TestCell:
    """Tests for Cell class."""

    def test_cell_creation(self):
        """Test creating notebook cells."""
        cell = Cell(
            id="cell1",
            cell_type=CellType.CODE,
            source="print('Hello')",
            metadata={"tags": ["test"]},
        )
        assert cell.cell_type == CellType.CODE
        assert cell.source == "print('Hello')"
        assert "test" in cell.metadata.get("tags", [])

    def test_markdown_cell(self):
        """Test creating markdown cell."""
        cell = Cell(
            id="cell2",
            cell_type=CellType.MARKDOWN,
            source="# Header\n\nThis is markdown",
            metadata={},
        )
        assert cell.cell_type == CellType.MARKDOWN
        assert "# Header" in cell.source

    def test_cell_with_outputs(self):
        """Test cell with execution outputs."""
        cell = Cell(
            id="cell3",
            cell_type=CellType.CODE,
            source="1 + 1",
            metadata={},
            outputs=[{"output_type": "execute_result", "data": {"text/plain": "2"}}],
        )
        assert cell.outputs is not None
        assert len(cell.outputs) == 1
        assert cell.outputs[0]["data"]["text/plain"] == "2"


class TestCellExecutionResult:
    """Tests for CellExecutionResult."""

    def test_successful_execution(self):
        """Test successful cell execution result."""
        result = CellExecutionResult(
            cell_id="cell1",
            success=True,
            outputs=[{"output_type": "stream", "text": "Output"}],
            execution_count=1,
        )
        assert result.success
        assert len(result.outputs) == 1
        assert result.error is None

    def test_failed_execution(self):
        """Test failed cell execution result."""
        result = CellExecutionResult(
            cell_id="cell2",
            success=False,
            outputs=[],
            error={
                "ename": "ValueError",
                "evalue": "Invalid input",
                "traceback": ["Traceback..."],
            },
            execution_count=2,
        )
        assert not result.success
        assert result.error is not None
        assert result.error["ename"] == "ValueError"

    def test_execution_with_metadata(self):
        """Test execution result with metadata."""
        result = CellExecutionResult(
            cell_id="cell3",
            success=True,
            outputs=[],
            execution_count=3,
            execution_time_ms=500,
        )
        assert result.execution_time_ms == 500


class TestCellSelector:
    """Tests for CellSelector."""

    def test_select_by_index(self):
        """Test selecting cells by index."""
        selector = CellSelector(index_range=(0, 4))
        assert selector.index_range == (0, 4)

    def test_select_by_tag(self):
        """Test selecting cells by tag."""
        selector = CellSelector(tag_filter=["analysis", "visualization"])
        assert selector.tag_filter is not None
        assert "analysis" in selector.tag_filter
        assert "visualization" in selector.tag_filter

    def test_select_by_type(self):
        """Test selecting cells by type."""
        selector = CellSelector(cell_types=[CellType.CODE])
        assert CellType.CODE in selector.cell_types
        assert CellType.MARKDOWN not in selector.cell_types

    def test_combined_selection(self):
        """Test combined selection criteria."""
        selector = CellSelector(
            index_range=(0, 1),
            tag_filter=["important"],
            cell_types=[CellType.CODE, CellType.MARKDOWN],
        )
        assert selector.index_range == (0, 1)
        assert selector.tag_filter is not None
        assert len(selector.tag_filter) == 1
        assert selector.cell_types is not None
        assert len(selector.cell_types) == 2


class TestNotebookDocument:
    """Tests for NotebookDocument."""

    def test_notebook_creation(self):
        """Test creating notebook document."""
        cells = [
            Cell(id="1", cell_type=CellType.MARKDOWN, source="# Title", metadata={}),
            Cell(id="2", cell_type=CellType.CODE, source="import pandas as pd", metadata={}),
        ]

        notebook = NotebookDocument(
            id="nb1",
            cells=cells,
            metadata={"kernel": "python3", "language": "python"},
            kernel_spec={"name": "python3"},
            language_info={"name": "python"},
        )
        assert len(notebook.cells) == 2
        assert notebook.metadata["kernel"] == "python3"

    def test_notebook_with_source_file(self):
        """Test notebook with source file reference."""

        notebook = NotebookDocument(
            id="nb2",
            cells=[],
            metadata={"source_file": "data.xlsx"},
            kernel_spec={"name": "python3"},
            language_info={"name": "python"},
        )
        assert notebook.metadata["source_file"] == "data.xlsx"

    def test_add_cell_to_notebook(self):
        """Test adding cells to notebook."""
        notebook = NotebookDocument(
            id="nb3",
            cells=[],
            metadata={},
            kernel_spec={"name": "python3"},
            language_info={"name": "python"},
        )

        cell = Cell(id="4", cell_type=CellType.CODE, source="x = 1", metadata={})
        notebook.cells.append(cell)

        assert len(notebook.cells) == 1
        assert notebook.cells[0].source == "x = 1"


class TestNotebookProtocol:
    """Tests for NotebookProtocol interface."""

    def test_notebook_protocol_implementation(self):
        """Test implementing the notebook protocol."""

        class MockNotebookExecutor:
            def execute_cell(self, cell: Cell) -> CellExecutionResult:
                if "error" in cell.source:
                    return CellExecutionResult(
                        cell_id=cell.id,
                        success=False,
                        outputs=[],
                        error={"ename": "Error", "evalue": "Test error"},
                    )
                return CellExecutionResult(
                    cell_id=cell.id,
                    success=True,
                    outputs=[{"output_type": "stream", "text": "Success"}],
                )

            def execute_notebook(self, notebook: NotebookDocument) -> list[CellExecutionResult]:
                return [self.execute_cell(cell) for cell in notebook.cells]

            def select_cells(self, notebook: NotebookDocument, selector: CellSelector) -> list[Cell]:
                selected = []
                for i, cell in enumerate(notebook.cells):
                    if (selector.index_range and selector.index_range[0] <= i <= selector.index_range[1]) or (
                        selector.cell_types and cell.cell_type in selector.cell_types
                    ):
                        selected.append(cell)
                return selected

        # Test the implementation
        executor = MockNotebookExecutor()

        # Test cell execution
        cell = Cell(id="test1", cell_type=CellType.CODE, source="print('test')", metadata={})
        result = executor.execute_cell(cell)
        assert result.success

        # Test error handling
        error_cell = Cell(id="test2", cell_type=CellType.CODE, source="raise error", metadata={})
        error_result = executor.execute_cell(error_cell)
        assert not error_result.success

        # Test notebook execution
        notebook = NotebookDocument(
            id="test_nb",
            cells=[cell, error_cell],
            metadata={},
            kernel_spec={"name": "python3"},
            language_info={"name": "python"},
        )
        results = executor.execute_notebook(notebook)
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success

        # Test cell selection
        selector = CellSelector(index_range=(0, 0))
        selected = executor.select_cells(notebook, selector)
        assert len(selected) == 1
        assert selected[0].source == "print('test')"
