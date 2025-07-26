#!/usr/bin/env python3
"""Simple test script to verify the NotebookBuilder module works correctly."""

import json
from pathlib import Path

from spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder


def test_notebook_builder():
    """Test basic NotebookBuilder functionality."""
    print("Testing NotebookBuilder...")

    # Create a builder
    builder = NotebookBuilder()

    # Add some cells
    builder.add_markdown_cell("# Test Notebook")
    builder.add_code_cell("print('Hello, World!')")
    builder.add_markdown_cell("This is a test.")
    builder.add_code_cell("x = 2 + 2\nprint(f'Result: {x}')")

    # Test cell counts
    assert builder.cell_count() == 4
    assert builder.code_cell_count() == 2
    assert builder.markdown_cell_count() == 2

    # Convert to notebook format
    notebook = builder.to_notebook()

    # Verify structure
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 4
    assert notebook["cells"][0]["cell_type"] == "markdown"
    assert notebook["cells"][1]["cell_type"] == "code"

    # Test with outputs
    builder.clear()
    assert builder.cell_count() == 0

    # Add a cell with mock outputs
    mock_outputs = [{"output_type": "stream", "name": "stdout", "text": ["Hello from test!\n"]}]

    builder.add_code_cell_with_outputs("print('Hello from test!')", mock_outputs)

    notebook = builder.to_notebook()
    assert len(notebook["cells"]) == 1
    assert len(notebook["cells"][0]["outputs"]) == 1

    print("✅ All NotebookBuilder tests passed!")
    return True


def test_save_notebook():
    """Test saving a notebook to file."""
    print("Testing notebook file saving...")

    builder = NotebookBuilder()
    builder.add_markdown_cell("# File Save Test")
    builder.add_code_cell("print('This notebook was saved to file!')")

    # Save to file
    output_path = Path("examples/output/test_notebook.ipynb")
    output_path.parent.mkdir(exist_ok=True)

    notebook_data = builder.to_notebook()
    with open(output_path, "w") as f:
        json.dump(notebook_data, f, indent=2)

    # Verify file exists and can be read
    assert output_path.exists()

    with open(output_path) as f:
        loaded_data = json.load(f)

    assert loaded_data["nbformat"] == 4
    assert len(loaded_data["cells"]) == 2

    print(f"✅ Notebook successfully saved to: {output_path}")
    return True


if __name__ == "__main__":
    test_notebook_builder()
    test_save_notebook()
    print("\n🎉 All tests completed successfully!")
