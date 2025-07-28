"""
Tests for the generic NotebookIO module.

This test suite validates the notebook I/O functionality including:
- Reading and writing notebook files
- Format validation and error handling
- NotebookBuilder integration
- File path management and safety
- Format conversion between NotebookBuilder and nbformat
- Error recovery and validation

Following TDD principles with functional tests - no mocking used.
All tests use real file I/O operations and actual notebook files.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from spreadsheet_analyzer.core_exec.notebook_builder import (
    NotebookBuilder,
)
from spreadsheet_analyzer.core_exec.notebook_io import (
    NotebookFormatError,
    NotebookIO,
)


class TestNotebookIO:
    """Test NotebookIO file operations."""

    def test_write_and_read_empty_notebook(self, tmp_path: Path) -> None:
        """Test writing and reading an empty notebook."""
        # Create empty notebook
        builder = NotebookBuilder()
        output_path = tmp_path / "empty_notebook.ipynb"

        # Write notebook
        NotebookIO.write_notebook(builder, output_path)

        # File should exist
        assert output_path.exists()
        assert output_path.suffix == ".ipynb"

        # Read notebook back
        read_builder = NotebookIO.read_notebook(output_path)

        # Should have same structure
        assert read_builder.notebook.metadata.kernelspec.name == "python3"
        assert read_builder.notebook.metadata.kernelspec.display_name == "Python 3"
        assert len(read_builder.notebook.cells) == 0

    def test_write_and_read_notebook_with_cells(self, tmp_path: Path) -> None:
        """Test writing and reading a notebook with various cell types."""
        # Create notebook with different cell types
        builder = NotebookBuilder()
        builder.add_markdown_cell("# Data Analysis Notebook")
        builder.add_code_cell("import pandas as pd\nimport numpy as np")
        builder.add_markdown_cell("## Load Data")
        builder.add_code_cell("df = pd.read_csv('data.csv')")
        builder.add_raw_cell("Raw text content")

        output_path = tmp_path / "analysis_notebook.ipynb"

        # Write notebook
        NotebookIO.write_notebook(builder, output_path)

        # Read notebook back
        read_builder = NotebookIO.read_notebook(output_path)

        # Should have same number of cells
        assert len(read_builder.notebook.cells) == 5

        # Check cell types and content
        assert read_builder.notebook.cells[0].cell_type == "markdown"
        # Source can be a string or list of strings
        source = read_builder.notebook.cells[0].source
        if isinstance(source, list):
            source = "".join(source)
        assert "# Data Analysis Notebook" in source

        assert read_builder.notebook.cells[1].cell_type == "code"
        source = read_builder.notebook.cells[1].source
        if isinstance(source, list):
            source = "".join(source)
        assert "import pandas as pd" in source
        assert read_builder.notebook.cells[1].execution_count == 1

        assert read_builder.notebook.cells[2].cell_type == "markdown"
        source = read_builder.notebook.cells[2].source
        if isinstance(source, list):
            source = "".join(source)
        assert "## Load Data" in source

        assert read_builder.notebook.cells[3].cell_type == "code"
        source = read_builder.notebook.cells[3].source
        if isinstance(source, list):
            source = "".join(source)
        assert "df = pd.read_csv('data.csv')" in source
        assert read_builder.notebook.cells[3].execution_count == 2

        assert read_builder.notebook.cells[4].cell_type == "raw"
        source = read_builder.notebook.cells[4].source
        if isinstance(source, list):
            source = "".join(source)
        assert "Raw text content" in source

    def test_write_notebook_with_outputs(self, tmp_path: Path) -> None:
        """Test writing and reading notebook with cell outputs."""
        builder = NotebookBuilder()

        # Add code cell with outputs
        outputs: list[dict[str, Any]] = [
            {"output_type": "stream", "name": "stdout", "text": ["Hello, World!\n"]},
            {"output_type": "execute_result", "execution_count": 1, "data": {"text/plain": "42"}, "metadata": {}},
        ]
        builder.add_code_cell("print('Hello, World!')\n42", outputs=outputs)

        output_path = tmp_path / "notebook_with_outputs.ipynb"

        # Write and read back
        NotebookIO.write_notebook(builder, output_path)
        read_builder = NotebookIO.read_notebook(output_path)

        # Check outputs are preserved
        cell = read_builder.notebook.cells[0]
        assert len(cell.outputs) == 2
        assert cell.outputs[0]["output_type"] == "stream"
        assert cell.outputs[0]["text"][0] == "Hello, World!\n"
        assert cell.outputs[1]["output_type"] == "execute_result"
        # Text data might be stored as list of strings
        text_plain = cell.outputs[1]["data"]["text/plain"]
        if isinstance(text_plain, list):
            text_plain = "".join(text_plain)
        assert text_plain == "42"

    def test_write_notebook_with_metadata(self, tmp_path: Path) -> None:
        """Test writing and reading notebook with complex metadata."""
        builder = NotebookBuilder()

        # Add cells with complex metadata
        markdown_meta = {
            "tags": ["introduction", "header"],
            "slideshow": {"slide_type": "slide"},
            "custom": {"nested": {"value": 123}},
        }

        code_meta = {"tags": ["analysis"], "collapsed": False, "scrolled": True}

        builder.add_markdown_cell("# Introduction", markdown_meta)
        builder.add_code_cell("import pandas as pd", metadata=code_meta)

        output_path = tmp_path / "notebook_with_metadata.ipynb"

        # Write and read back
        NotebookIO.write_notebook(builder, output_path)
        read_builder = NotebookIO.read_notebook(output_path)

        # Check metadata preservation
        assert read_builder.notebook.cells[0].metadata == markdown_meta
        assert read_builder.notebook.cells[1].metadata == code_meta

    def test_write_notebook_custom_kernel(self, tmp_path: Path) -> None:
        """Test writing notebook with custom kernel specification."""
        builder = NotebookBuilder(kernel_name="julia-1.6", kernel_display_name="Julia 1.6")
        builder.add_code_cell('println("Hello, Julia!")')

        output_path = tmp_path / "julia_notebook.ipynb"

        # Write and read back
        NotebookIO.write_notebook(builder, output_path)
        read_builder = NotebookIO.read_notebook(output_path)

        # Check kernel information
        assert read_builder.notebook.metadata.kernelspec.name == "julia-1.6"
        assert read_builder.notebook.metadata.kernelspec.display_name == "Julia 1.6"

    def test_write_notebook_creates_directory(self, tmp_path: Path) -> None:
        """Test that write_notebook creates parent directories if needed."""
        builder = NotebookBuilder()
        builder.add_markdown_cell("# Test")

        # Use nested path that doesn't exist
        nested_path = tmp_path / "deeply" / "nested" / "path" / "notebook.ipynb"

        # Should create directories
        NotebookIO.write_notebook(builder, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_write_notebook_string_path(self, tmp_path: Path) -> None:
        """Test writing notebook with string path instead of Path object."""
        builder = NotebookBuilder()
        builder.add_markdown_cell("# String Path Test")

        # Use string path
        output_path_str = str(tmp_path / "string_path_notebook.ipynb")

        NotebookIO.write_notebook(builder, output_path_str)

        # Should create file
        assert Path(output_path_str).exists()

        # Should be readable
        read_builder = NotebookIO.read_notebook(output_path_str)
        assert len(read_builder.notebook.cells) == 1

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading a file that doesn't exist."""
        nonexistent_path = tmp_path / "does_not_exist.ipynb"

        with pytest.raises(FileNotFoundError, match="Notebook file not found"):
            NotebookIO.read_notebook(nonexistent_path)

    def test_read_non_ipynb_file(self, tmp_path: Path) -> None:
        """Test reading a file without .ipynb extension."""
        text_file = tmp_path / "not_notebook.txt"
        text_file.write_text("This is not a notebook")

        with pytest.raises(NotebookFormatError, match="File must have .ipynb extension"):
            NotebookIO.read_notebook(text_file)

    def test_read_invalid_json_file(self, tmp_path: Path) -> None:
        """Test reading a file with invalid JSON."""
        invalid_file = tmp_path / "invalid.ipynb"
        invalid_file.write_text("{ invalid json content }")

        with pytest.raises(NotebookFormatError, match="Invalid JSON"):
            NotebookIO.read_notebook(invalid_file)

    def test_read_invalid_notebook_format(self, tmp_path: Path) -> None:
        """Test reading a JSON file that's not a valid notebook."""
        invalid_notebook = tmp_path / "invalid_notebook.ipynb"

        # Valid JSON but not valid notebook format
        invalid_content = {"not_a_notebook": True, "missing_required_fields": "yes"}

        invalid_notebook.write_text(json.dumps(invalid_content))

        with pytest.raises(NotebookFormatError, match="Invalid notebook format"):
            NotebookIO.read_notebook(invalid_notebook)

    def test_round_trip_compatibility(self, tmp_path: Path) -> None:
        """Test that notebooks can be round-tripped through the system."""
        # Create a complex notebook
        original = NotebookBuilder()

        # Add various cell types with complex content
        original.add_markdown_cell("# Complex Notebook\n\nWith **formatting** and `code`.", {"tags": ["header"]})

        original.add_code_cell(
            "import pandas as pd\n\n# Load data\ndf = pd.DataFrame({\n    'x': [1, 2, 3],\n    'y': [4, 5, 6]\n})\n\nprint(df.head())",
            outputs=[{"output_type": "stream", "name": "stdout", "text": ["   x  y\n0  1  4\n1  2  5\n2  3  6\n"]}],
            metadata={"collapsed": False},
        )

        original.add_raw_cell("Raw content\nWith multiple lines")

        # Save and reload multiple times
        paths = []
        builders = [original]

        for i in range(3):
            path = tmp_path / f"round_trip_{i}.ipynb"
            paths.append(path)

            # Write current builder
            NotebookIO.write_notebook(builders[-1], path)

            # Read it back
            reloaded = NotebookIO.read_notebook(path)
            builders.append(reloaded)

        # All versions should be equivalent
        final_builder = builders[-1]

        assert len(final_builder.notebook.cells) == 3
        assert final_builder.notebook.cells[0].cell_type == "markdown"
        assert final_builder.notebook.cells[1].cell_type == "code"
        assert final_builder.notebook.cells[2].cell_type == "raw"

        # Content should be preserved
        # Check content of cells
        for i, expected in enumerate(["# Complex Notebook", "import pandas as pd", "Raw content"]):
            source = final_builder.notebook.cells[i].source
            if isinstance(source, list):
                source = "".join(source)
            assert expected in source

        # Metadata should be preserved
        assert final_builder.notebook.cells[0].metadata["tags"] == ["header"]
        assert final_builder.notebook.cells[1].metadata["collapsed"] is False

        # Outputs should be preserved
        assert len(final_builder.notebook.cells[1].outputs) == 1
        assert final_builder.notebook.cells[1].outputs[0].output_type == "stream"

    def test_concurrent_file_operations(self, tmp_path: Path) -> None:
        """Test concurrent read/write operations don't interfere."""
        import threading
        import time

        # Create multiple notebooks
        builders = []
        for i in range(5):
            builder = NotebookBuilder()
            builder.add_markdown_cell(f"# Notebook {i}")
            builder.add_code_cell(f"result_{i} = {i} * 10")
            builders.append(builder)

        paths = [tmp_path / f"concurrent_{i}.ipynb" for i in range(5)]
        results: list[NotebookBuilder | Exception | None] = [None] * 5

        def write_and_read(index: int) -> None:
            """Write and read a notebook."""
            try:
                # Write
                NotebookIO.write_notebook(builders[index], paths[index])

                # Small delay to test concurrency
                time.sleep(0.01)

                # Read back
                read_builder = NotebookIO.read_notebook(paths[index])
                results[index] = read_builder
            except Exception as e:
                results[index] = e

        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_and_read, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # All should succeed
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Thread {i} failed: {result}"
            assert isinstance(result, NotebookBuilder)
            assert len(result.notebook.cells) == 2
            source = result.notebook.cells[0].source
            if isinstance(source, list):
                source = "".join(source)
            assert f"# Notebook {i}" in source

    def test_large_notebook_handling(self, tmp_path: Path) -> None:
        """Test handling of large notebooks with many cells."""
        # Create large notebook
        builder = NotebookBuilder()

        num_cells = 1000
        for i in range(num_cells):
            if i % 2 == 0:
                builder.add_markdown_cell(f"## Section {i}")
            else:
                builder.add_code_cell(f"result_{i} = {i} ** 2")

        output_path = tmp_path / "large_notebook.ipynb"

        # Write (should handle large files)
        NotebookIO.write_notebook(builder, output_path)

        # File should exist and be substantial size
        assert output_path.exists()
        assert output_path.stat().st_size > 10000  # Should be reasonably large

        # Read back (should handle large files)
        read_builder = NotebookIO.read_notebook(output_path)

        assert len(read_builder.notebook.cells) == num_cells
        assert read_builder.code_cell_count() == num_cells // 2
        assert read_builder.markdown_cell_count() == num_cells // 2

    def test_special_characters_handling(self, tmp_path: Path) -> None:
        """Test handling of notebooks with special characters and unicode."""
        builder = NotebookBuilder()

        # Add cells with various special characters
        builder.add_markdown_cell("# Spécial Çharacters: éñ中文🚀")
        builder.add_code_cell("# Unicode: αβγδε\nprint('Emojis: 🐍🔬📊')")
        builder.add_raw_cell("Raw with newlines\nand\ttabs")

        output_path = tmp_path / "special_chars.ipynb"

        # Write and read
        NotebookIO.write_notebook(builder, output_path)
        read_builder = NotebookIO.read_notebook(output_path)

        # Special characters should be preserved
        # Check special characters in cells
        for i, expectations in enumerate(
            [["Spécial Çharacters: éñ中文🚀"], ["Unicode: αβγδε", "🐍🔬📊"], [None, "\t"]]
        ):
            source = read_builder.notebook.cells[i].source
            if isinstance(source, list):
                for j, expected in enumerate(expectations):
                    if expected and j < len(source):
                        assert expected in source[j]
            else:
                if expectations[0]:
                    assert expectations[0] in source

    def test_file_permissions_handling(self, tmp_path: Path) -> None:
        """Test handling of file permission issues."""
        builder = NotebookBuilder()
        builder.add_markdown_cell("# Permission Test")

        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        readonly_file = readonly_dir / "notebook.ipynb"

        try:
            # Should raise PermissionError
            with pytest.raises(PermissionError):
                NotebookIO.write_notebook(builder, readonly_file)
        finally:
            # Cleanup: restore permissions
            readonly_dir.chmod(0o755)

    def test_overwrite_existing_file(self, tmp_path: Path) -> None:
        """Test overwriting existing notebook files."""
        builder = NotebookBuilder()
        builder.add_markdown_cell("# Original Content")

        notebook_path = tmp_path / "notebook.ipynb"

        # Write initial version
        NotebookIO.write_notebook(builder, notebook_path)
        assert notebook_path.exists()

        # Try to write again without overwrite flag - should fail
        builder.add_code_cell("print('Modified')")
        with pytest.raises(FileExistsError, match="File exists and overwrite=False"):
            NotebookIO.write_notebook(builder, notebook_path, overwrite=False)

        # Write with overwrite=True - should succeed
        NotebookIO.write_notebook(builder, notebook_path, overwrite=True)

        # Current file should have both cells
        current_builder = NotebookIO.read_notebook(notebook_path)
        assert len(current_builder.notebook.cells) == 2
