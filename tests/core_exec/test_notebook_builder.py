"""
Tests for the generic NotebookBuilder module.

This test suite validates the core notebook building functionality including:
- NotebookCell creation and validation
- CellType enumeration
- NotebookBuilder cell management
- Source formatting compliance with Jupyter standards
- Execution count management
- Metadata handling
- nbformat dictionary conversion

Following TDD principles with functional tests - no mocking used.
All tests verify actual notebook structure compliance.
"""

import json
from typing import Any

from spreadsheet_analyzer.core_exec.notebook_builder import (
    CellType,
    NotebookBuilder,
    NotebookCell,
)


class TestCellType:
    """Test CellType enumeration."""

    def test_cell_type_values(self) -> None:
        """Test that CellType has correct enumeration values."""
        assert CellType.CODE.value == "code"
        assert CellType.MARKDOWN.value == "markdown"
        assert CellType.RAW.value == "raw"

    def test_cell_type_iteration(self) -> None:
        """Test that all cell types can be iterated."""
        cell_types = list(CellType)
        assert len(cell_types) == 3
        assert CellType.CODE in cell_types
        assert CellType.MARKDOWN in cell_types
        assert CellType.RAW in cell_types


class TestNotebookCell:
    """Test NotebookCell data structure."""

    def test_markdown_cell_creation(self) -> None:
        """Test creating a markdown NotebookCell."""
        source = ["# Test Markdown\n", "This is a test."]
        metadata = {"tags": ["header"]}

        cell = NotebookCell(cell_type=CellType.MARKDOWN, source=source, metadata=metadata)

        assert cell.cell_type == CellType.MARKDOWN
        assert cell.source == source
        assert cell.metadata == metadata
        assert cell.outputs is None
        assert cell.execution_count is None

    def test_code_cell_creation(self) -> None:
        """Test creating a code NotebookCell."""
        source = ["print('Hello, World!')\n", "2 + 2"]
        metadata = {"collapsed": False}
        outputs = [{"output_type": "stream", "text": "Hello, World!\n"}]
        execution_count = 1

        cell = NotebookCell(
            cell_type=CellType.CODE, source=source, metadata=metadata, outputs=outputs, execution_count=execution_count
        )

        assert cell.cell_type == CellType.CODE
        assert cell.source == source
        assert cell.metadata == metadata
        assert cell.outputs == outputs
        assert cell.execution_count == execution_count

    def test_raw_cell_creation(self) -> None:
        """Test creating a raw NotebookCell."""
        source = ["Raw text content\n", "Not processed by Jupyter"]
        metadata = {"format": "restructuredtext"}

        cell = NotebookCell(cell_type=CellType.RAW, source=source, metadata=metadata)

        assert cell.cell_type == CellType.RAW
        assert cell.source == source
        assert cell.metadata == metadata
        assert cell.outputs is None
        assert cell.execution_count is None

    def test_cell_to_dict_markdown(self) -> None:
        """Test converting markdown cell to dictionary format."""
        source = ["# Header\n", "Content"]
        metadata = {"tags": ["intro"]}

        cell = NotebookCell(cell_type=CellType.MARKDOWN, source=source, metadata=metadata)

        cell_dict = cell.to_dict()

        # ID should be present and deterministic
        assert "id" in cell_dict
        assert len(cell_dict["id"]) == 12  # We use 12 chars from SHA256

        # Check other fields
        assert cell_dict["cell_type"] == "markdown"
        assert cell_dict["metadata"] == {"tags": ["intro"]}
        assert cell_dict["source"] == ["# Header\n", "Content"]

    def test_cell_to_dict_code(self) -> None:
        """Test converting code cell to dictionary format."""
        source = ["x = 1\n", "print(x)"]
        metadata = {"collapsed": True}
        outputs = [{"output_type": "stream", "text": "1\n"}]
        execution_count = 5

        cell = NotebookCell(
            cell_type=CellType.CODE, source=source, metadata=metadata, outputs=outputs, execution_count=execution_count
        )

        cell_dict = cell.to_dict()

        # ID should be present and deterministic
        assert "id" in cell_dict
        assert len(cell_dict["id"]) == 12  # We use 12 chars from SHA256

        # Check other fields
        assert cell_dict["cell_type"] == "code"
        assert cell_dict["metadata"] == {"collapsed": True}
        assert cell_dict["source"] == ["x = 1\n", "print(x)"]
        assert cell_dict["execution_count"] == 5
        assert cell_dict["outputs"] == [{"output_type": "stream", "text": "1\n"}]

    def test_cell_to_dict_code_without_outputs(self) -> None:
        """Test converting code cell without outputs to dictionary format."""
        source = ["# This code hasn't been executed yet"]
        metadata: dict[str, Any] = {}

        cell = NotebookCell(cell_type=CellType.CODE, source=source, metadata=metadata, execution_count=None)

        cell_dict = cell.to_dict()

        # ID should be present and deterministic
        assert "id" in cell_dict
        assert len(cell_dict["id"]) == 12  # We use 12 chars from SHA256

        # Check other fields
        assert cell_dict["cell_type"] == "code"
        assert cell_dict["metadata"] == {}
        assert cell_dict["source"] == ["# This code hasn't been executed yet"]
        assert cell_dict["execution_count"] is None
        assert cell_dict["outputs"] == []


class TestNotebookBuilder:
    """Test NotebookBuilder functionality."""

    def test_empty_builder_creation(self) -> None:
        """Test creating an empty NotebookBuilder."""
        builder = NotebookBuilder()

        assert len(builder.cells) == 0
        assert builder.kernel_name == "python3"
        assert builder.kernel_display_name == "Python 3"
        assert builder._execution_count == 0

    def test_custom_kernel_builder_creation(self) -> None:
        """Test creating NotebookBuilder with custom kernel settings."""
        builder = NotebookBuilder(kernel_name="julia-1.6", kernel_display_name="Julia 1.6")

        assert builder.kernel_name == "julia-1.6"
        assert builder.kernel_display_name == "Julia 1.6"

    def test_add_markdown_cell(self) -> None:
        """Test adding a markdown cell."""
        builder = NotebookBuilder()
        metadata = {"tags": ["introduction"]}

        result = builder.add_markdown_cell("# Welcome\n\nThis is a test.", metadata)

        # Should return self for chaining
        assert result is builder

        # Should have one cell
        assert len(builder.cells) == 1

        cell = builder.cells[0]
        assert cell.cell_type == CellType.MARKDOWN
        assert cell.metadata == metadata
        # Source should be properly formatted
        expected_source = ["# Welcome\n", "\n", "This is a test."]
        assert cell.source == expected_source

    def test_add_markdown_cell_without_metadata(self) -> None:
        """Test adding a markdown cell without metadata."""
        builder = NotebookBuilder()

        builder.add_markdown_cell("Simple markdown")

        cell = builder.cells[0]
        assert cell.metadata == {}

    def test_add_code_cell_basic(self) -> None:
        """Test adding a basic code cell."""
        builder = NotebookBuilder()

        result = builder.add_code_cell("print('Hello, World!')")

        # Should return self for chaining
        assert result is builder

        # Should have one cell
        assert len(builder.cells) == 1

        cell = builder.cells[0]
        assert cell.cell_type == CellType.CODE
        assert cell.execution_count == 1
        assert cell.outputs == []
        assert cell.metadata == {}
        # Source should be properly formatted
        expected_source = ["print('Hello, World!')"]
        assert cell.source == expected_source

    def test_add_code_cell_with_outputs(self) -> None:
        """Test adding a code cell with outputs."""
        builder = NotebookBuilder()
        outputs = [{"output_type": "stream", "text": "Hello, World!\n"}]
        metadata = {"collapsed": False}

        builder.add_code_cell("print('Hello, World!')", outputs=outputs, metadata=metadata)

        cell = builder.cells[0]
        assert cell.outputs == outputs
        assert cell.metadata == metadata
        assert cell.execution_count == 1

    def test_add_raw_cell(self) -> None:
        """Test adding a raw cell."""
        builder = NotebookBuilder()
        metadata = {"format": "restructuredtext"}

        result = builder.add_raw_cell("Raw content\nNot processed", metadata)

        # Should return self for chaining
        assert result is builder

        cell = builder.cells[0]
        assert cell.cell_type == CellType.RAW
        assert cell.metadata == metadata
        expected_source = ["Raw content\n", "Not processed"]
        assert cell.source == expected_source

    def test_execution_count_increment(self) -> None:
        """Test that execution count increments properly for code cells."""
        builder = NotebookBuilder()

        # Add multiple code cells
        builder.add_code_cell("x = 1")
        builder.add_code_cell("y = 2")
        builder.add_code_cell("print(x + y)")

        assert len(builder.cells) == 3
        assert builder.cells[0].execution_count == 1
        assert builder.cells[1].execution_count == 2
        assert builder.cells[2].execution_count == 3

    def test_mixed_cell_types_execution_count(self) -> None:
        """Test execution count with mixed cell types."""
        builder = NotebookBuilder()

        # Add various cell types
        builder.add_markdown_cell("# Header")
        builder.add_code_cell("x = 1")
        builder.add_raw_cell("Raw content")
        builder.add_code_cell("y = 2")

        assert len(builder.cells) == 4

        # Only code cells should have execution counts
        assert builder.cells[0].execution_count is None  # markdown
        assert builder.cells[1].execution_count == 1  # code
        assert builder.cells[2].execution_count is None  # raw
        assert builder.cells[3].execution_count == 2  # code

    def test_format_source_single_line(self) -> None:
        """Test source formatting for single line content."""
        builder = NotebookBuilder()

        formatted = builder._format_source("print('hello')")

        assert formatted == ["print('hello')"]

    def test_format_source_multiple_lines(self) -> None:
        """Test source formatting for multi-line content."""
        builder = NotebookBuilder()

        content = "import pandas as pd\ndf = pd.DataFrame()\nprint(df)"
        formatted = builder._format_source(content)

        expected = ["import pandas as pd\n", "df = pd.DataFrame()\n", "print(df)"]
        assert formatted == expected

    def test_format_source_empty_content(self) -> None:
        """Test source formatting for empty content."""
        builder = NotebookBuilder()

        formatted = builder._format_source("")

        assert formatted == [""]

    def test_format_source_content_with_trailing_newline(self) -> None:
        """Test source formatting for content ending with newline."""
        builder = NotebookBuilder()

        content = "line1\nline2\n"
        formatted = builder._format_source(content)

        expected = ["line1\n", "line2\n", ""]
        assert formatted == expected

    def test_clear_cells(self) -> None:
        """Test clearing all cells from the builder."""
        builder = NotebookBuilder()

        # Add some cells
        builder.add_markdown_cell("# Header")
        builder.add_code_cell("x = 1")

        assert len(builder.cells) == 2

        # Clear cells
        builder.clear()

        assert len(builder.cells) == 0
        # Execution count should reset
        assert builder._execution_count == 0

    def test_cell_count_methods(self) -> None:
        """Test cell counting methods."""
        builder = NotebookBuilder()

        # Add various cell types
        builder.add_markdown_cell("# Header 1")
        builder.add_markdown_cell("# Header 2")
        builder.add_code_cell("x = 1")
        builder.add_code_cell("y = 2")
        builder.add_code_cell("z = 3")
        builder.add_raw_cell("Raw content")

        assert builder.cell_count() == 6
        assert builder.code_cell_count() == 3
        assert builder.markdown_cell_count() == 2
        assert builder.raw_cell_count() == 1

    def test_to_dict_empty_notebook(self) -> None:
        """Test converting empty notebook to dictionary."""
        builder = NotebookBuilder()

        notebook_dict = builder.to_dict()

        expected_structure = {
            "cells": [],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        assert notebook_dict == expected_structure

    def test_to_dict_with_cells(self) -> None:
        """Test converting notebook with cells to dictionary."""
        builder = NotebookBuilder()

        # Add some cells
        builder.add_markdown_cell("# Analysis")
        builder.add_code_cell("import pandas as pd")
        builder.add_code_cell("df = pd.DataFrame({'x': [1, 2, 3]})")

        notebook_dict = builder.to_dict()

        assert "cells" in notebook_dict
        assert len(notebook_dict["cells"]) == 3

        # Check first cell (markdown)
        first_cell = notebook_dict["cells"][0]
        assert first_cell["cell_type"] == "markdown"
        assert first_cell["source"] == ["# Analysis"]

        # Check second cell (code)
        second_cell = notebook_dict["cells"][1]
        assert second_cell["cell_type"] == "code"
        assert second_cell["execution_count"] == 1
        assert "outputs" in second_cell

        # Check metadata structure
        assert "metadata" in notebook_dict
        assert "kernelspec" in notebook_dict["metadata"]
        assert notebook_dict["metadata"]["kernelspec"]["name"] == "python3"

    def test_to_dict_custom_kernel(self) -> None:
        """Test dictionary conversion with custom kernel."""
        builder = NotebookBuilder(kernel_name="julia-1.6", kernel_display_name="Julia 1.6")

        notebook_dict = builder.to_dict()

        kernelspec = notebook_dict["metadata"]["kernelspec"]
        assert kernelspec["name"] == "julia-1.6"
        assert kernelspec["display_name"] == "Julia 1.6"

    def test_chaining_operations(self) -> None:
        """Test that builder methods can be chained."""
        builder = NotebookBuilder()

        # Chain multiple operations
        result = (
            builder.add_markdown_cell("# Data Analysis")
            .add_code_cell("import pandas as pd")
            .add_code_cell("import numpy as np")
            .add_markdown_cell("## Load Data")
            .add_code_cell("df = pd.read_csv('data.csv')")
        )

        # Should return the same builder instance
        assert result is builder

        # Should have all cells
        assert len(builder.cells) == 5
        assert builder.code_cell_count() == 3
        assert builder.markdown_cell_count() == 2

    def test_complex_source_formatting(self) -> None:
        """Test source formatting with complex code structures."""
        builder = NotebookBuilder()

        complex_code = """def analyze_data(df):
    '''Analyze DataFrame and return summary.'''
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory': df.memory_usage().sum()
    }
    return summary

# Call the function
result = analyze_data(my_df)
print(result)"""

        builder.add_code_cell(complex_code)

        cell = builder.cells[0]

        # Should properly format multi-line code
        assert len(cell.source) > 1
        assert cell.source[0].startswith("def analyze_data(df):")
        assert cell.source[-1] == "print(result)"

        # All lines except last should end with newline
        for line in cell.source[:-1]:
            assert line.endswith("\n")

        # Last line should not end with newline
        assert not cell.source[-1].endswith("\n")

    def test_metadata_preservation(self) -> None:
        """Test that cell metadata is properly preserved."""
        builder = NotebookBuilder()

        # Add cells with complex metadata
        markdown_meta = {
            "tags": ["header", "introduction"],
            "slideshow": {"slide_type": "slide"},
            "custom_field": {"nested": {"value": 42}},
        }

        code_meta = {"tags": ["analysis"], "collapsed": False, "scrolled": True, "execution": {"timeout": 30}}

        builder.add_markdown_cell("# Introduction", markdown_meta)
        builder.add_code_cell("import pandas", metadata=code_meta)

        # Check metadata preservation in dictionary format
        notebook_dict = builder.to_dict()

        markdown_cell = notebook_dict["cells"][0]
        code_cell = notebook_dict["cells"][1]

        assert markdown_cell["metadata"] == markdown_meta
        assert code_cell["metadata"] == code_meta

    def test_large_notebook_creation(self) -> None:
        """Test creating a large notebook with many cells."""
        builder = NotebookBuilder()

        # Add many cells to test performance and correctness
        num_cells = 100

        for i in range(num_cells):
            if i % 3 == 0:
                builder.add_markdown_cell(f"## Section {i}")
            elif i % 3 == 1:
                builder.add_code_cell(f"result_{i} = {i} * 2")
            else:
                builder.add_raw_cell(f"Raw content {i}")

        assert len(builder.cells) == num_cells
        assert builder.cell_count() == num_cells

        # Check execution count for code cells
        code_cells = [cell for cell in builder.cells if cell.cell_type == CellType.CODE]
        expected_code_count = len([i for i in range(num_cells) if i % 3 == 1])
        assert len(code_cells) == expected_code_count

        # Check execution counts are sequential
        for i, cell in enumerate(code_cells):
            assert cell.execution_count == i + 1

    def test_notebook_serialization(self) -> None:
        """Test that generated notebook can be serialized to JSON."""
        builder = NotebookBuilder()

        # Create a notebook with various cell types
        builder.add_markdown_cell("# Test Notebook")
        builder.add_code_cell("import json\ndata = {'test': True}")
        builder.add_raw_cell("Raw text content")

        notebook_dict = builder.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(notebook_dict, indent=2)

        # Should be deserializable
        deserialized = json.loads(json_str)

        assert deserialized == notebook_dict

        # Check key structure elements
        assert "cells" in deserialized
        assert "metadata" in deserialized
        assert "nbformat" in deserialized
        assert deserialized["nbformat"] == 4
