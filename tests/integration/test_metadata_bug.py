#!/usr/bin/env python3
"""Test if the metadata assignment bug is causing issues."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder


def test_metadata_bug():
    """Test the metadata assignment issue."""
    builder = NotebookBuilder()
    builder.add_code_cell("print('test')")

    # This is what the code is trying to do
    try:
        builder.to_dict()["metadata"]["execution_stats"] = {"test": 123}
        print("No error - but this doesn't actually modify the notebook!")
    except Exception as e:
        print(f"Error: {e}")

    # Check if metadata was actually added
    nb_dict = builder.to_dict()
    print(f"Notebook metadata: {nb_dict.get('metadata', {})}")
    print(f"Has execution_stats: {'execution_stats' in nb_dict.get('metadata', {})}")


if __name__ == "__main__":
    test_metadata_bug()
