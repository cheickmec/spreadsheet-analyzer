"""
Tests for the generic NotebookBuilder module.

This test suite validates the core notebook building functionality including:
- NotebookBuilder cell management
- Source formatting compliance with Jupyter standards
- Execution count management
- Metadata handling
- nbformat dictionary conversion
"""

from spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder


class TestNotebookBuilder:
    """Test NotebookBuilder functionality."""

    def test_empty_builder_creation(self) -> None:
        """Test creating an empty NotebookBuilder."""
        builder = NotebookBuilder()
        assert len(builder.to_notebook().cells) == 0
        assert builder.to_notebook().metadata.kernelspec.name == "python3"

    def test_add_markdown_cell(self) -> None:
        """Test adding a markdown cell."""
        builder = NotebookBuilder()
        builder.add_markdown_cell("# Welcome")
        notebook = builder.to_notebook()
        assert len(notebook.cells) == 1
        assert notebook.cells[0].cell_type == "markdown"
        assert "# Welcome" in notebook.cells[0].source

    def test_add_code_cell(self) -> None:
        """Test adding a code cell."""
        builder = NotebookBuilder()
        builder.add_code_cell("print('Hello')")
        notebook = builder.to_notebook()
        assert len(notebook.cells) == 1
        assert notebook.cells[0].cell_type == "code"
        assert "print('Hello')" in notebook.cells[0].source
        assert notebook.cells[0].execution_count == 1

    def test_execution_count_increment(self) -> None:
        """Test that execution count increments properly for code cells."""
        builder = NotebookBuilder()
        builder.add_code_cell("x = 1")
        builder.add_markdown_cell("---")
        builder.add_code_cell("y = 2")
        notebook = builder.to_notebook()
        assert notebook.cells[0].execution_count == 1
        assert notebook.cells[2].execution_count == 2

    def test_to_dict_conversion(self) -> None:
        """Test converting notebook to dictionary."""
        builder = NotebookBuilder()
        builder.add_markdown_cell("# Analysis")
        builder.add_code_cell("import pandas as pd")
        notebook_dict = builder.to_dict()
        assert isinstance(notebook_dict, dict)
        assert len(notebook_dict["cells"]) == 2
        assert notebook_dict["cells"][0]["cell_type"] == "markdown"
        assert notebook_dict["cells"][1]["cell_type"] == "code"

    def test_chaining_operations(self) -> None:
        """Test that builder methods can be chained."""
        builder = NotebookBuilder()
        builder.add_markdown_cell("# Title").add_code_cell("a=1").add_code_cell("b=2")
        assert builder.cell_count() == 3
