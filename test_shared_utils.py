#!/usr/bin/env python3
"""Test script to verify shared utilities are working correctly."""

import asyncio
from pathlib import Path

from spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder
from spreadsheet_analyzer.core_exec.notebook_io import NotebookIO
from spreadsheet_analyzer.plugins.spreadsheet.io.excel_io import check_file_size
from spreadsheet_analyzer.utils import calculate_cost


async def test_utils():
    """Test shared utilities and proper architecture usage."""
    print("Testing shared utilities and architecture layers...")

    # Test cost calculation (from utils - correct usage)
    cost = calculate_cost(model="claude-3-5-sonnet-20241022", input_tokens=1000, output_tokens=500)
    print(f"✓ Cost calculation: ${cost:.4f}")

    # Test file size check (from plugins - correct usage)
    file_size, is_large = await check_file_size(Path(__file__))
    print(f"✓ File size check: {file_size:.2f}MB, is_large={is_large}")

    # Test notebook creation using core_exec (proper architecture)
    builder = NotebookBuilder()
    builder.add_markdown_cell("# Test Notebook")
    builder.add_code_cell("# Testing code\nprint('Hello from test!')")
    notebook = builder.build()
    print(f"✓ Notebook creation: {len(notebook['cells'])} cells")

    # Test notebook I/O using core_exec
    io = NotebookIO()
    output_path = Path("test_output/test_notebook.ipynb")
    output_path.parent.mkdir(exist_ok=True)

    try:
        io.save_notebook(notebook, output_path)
        print(f"✓ Notebook saved: {output_path}")

        # Test loading it back
        loaded = io.load_notebook(output_path)
        print(f"✓ Notebook loaded: {len(loaded['cells'])} cells")

        # Cleanup
        import shutil

        shutil.rmtree("test_output", ignore_errors=True)

    except Exception as e:
        print(f"✗ Notebook I/O failed: {e}")

    print("\nAll architecture layers tested! ✨")
    print("- utils: Generic utilities (cost calculation)")
    print("- plugins: Domain-specific (file size check)")
    print("- core_exec: Notebook primitives (builder, I/O)")


if __name__ == "__main__":
    asyncio.run(test_utils())
