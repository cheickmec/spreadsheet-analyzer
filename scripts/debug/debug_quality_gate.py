#!/usr/bin/env python3
"""Debug script to test quality gate functionality."""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
    SheetState,
    inspect_notebook_quality,
    quality_gate_node,
)
from spreadsheet_analyzer.notebook_llm.nap.protocols import Cell, CellType, NotebookDocument


async def test_quality_gate():
    """Test the quality gate logic directly."""

    # Create a minimal notebook with insufficient analysis
    notebook = NotebookDocument(
        id="test",
        cells=[
            Cell(
                id="setup",
                cell_type=CellType.CODE,
                source="import pandas as pd",
                metadata={},
                outputs=[],
                execution_count=1,
            ),
            Cell(
                id="load",
                cell_type=CellType.CODE,
                source="df = pd.read_excel('test.xlsx')",
                metadata={},
                outputs=[],
                execution_count=2,
            ),
            Cell(
                id="basic",
                cell_type=CellType.CODE,
                source="print(df.head())",
                metadata={},
                outputs=[{"output_type": "stream", "text": "Some output"}],
                execution_count=3,
            ),
        ],
        metadata={},
        kernel_spec={},
        language_info={},
    )

    # Test inspect_notebook_quality directly
    print("=== Testing inspect_notebook_quality ===")
    needs_more, reason = inspect_notebook_quality(notebook)
    print(f"Needs more analysis: {needs_more}")
    print(f"Reason: {reason}")
    print()

    # Convert notebook to dict format
    notebook_dict = {
        "id": notebook.id,
        "cells": [
            {
                "id": cell.id,
                "cell_type": cell.cell_type.value,
                "source": cell.source,
                "outputs": cell.outputs,
                "execution_count": cell.execution_count,
                "metadata": cell.metadata,
            }
            for cell in notebook.cells
        ],
        "metadata": notebook.metadata,
        "kernel_spec": notebook.kernel_spec,
        "language_info": notebook.language_info,
    }

    # Create state with the minimal notebook
    state = SheetState(
        excel_path=Path("test.xlsx"),
        sheet_name="Sheet1",
        notebook_final=notebook_dict,
        quality_iterations=0,
        max_quality_iterations=3,
        quality_reasons=[],
        total_cost=0.0,
    )

    print("=== Testing quality_gate_node ===")
    print(f"Initial quality_iterations: {state['quality_iterations']}")
    print(f"Max quality_iterations: {state['max_quality_iterations']}")

    # Call quality gate node
    result = await quality_gate_node(state)

    print(f"Needs refinement: {result.get('needs_refinement')}")
    print(f"Quality feedback: {result.get('quality_feedback')}")
    print(f"New quality_iterations: {result.get('quality_iterations')}")
    print(f"Quality reasons: {result.get('quality_reasons')}")

    # Test with a more complete notebook
    print("\n=== Testing with more complete notebook ===")

    complete_notebook = NotebookDocument(
        id="test2",
        cells=[
            Cell(
                id="setup",
                cell_type=CellType.CODE,
                source="import pandas as pd\nimport matplotlib.pyplot as plt",
                metadata={},
                outputs=[],
                execution_count=1,
            ),
            Cell(
                id="load",
                cell_type=CellType.CODE,
                source="df = pd.read_excel('test.xlsx')",
                metadata={},
                outputs=[],
                execution_count=2,
            ),
            Cell(
                id="describe",
                cell_type=CellType.CODE,
                source="df.describe()",
                metadata={},
                outputs=[{"output_type": "execute_result", "data": {"text/plain": "statistics output"}}],
                execution_count=3,
            ),
            Cell(
                id="dtypes",
                cell_type=CellType.CODE,
                source="df.dtypes",
                metadata={},
                outputs=[{"output_type": "execute_result", "data": {"text/plain": "dtypes output"}}],
                execution_count=4,
            ),
            Cell(
                id="null_check",
                cell_type=CellType.CODE,
                source="df.isna().sum()",
                metadata={},
                outputs=[{"output_type": "execute_result", "data": {"text/plain": "null counts"}}],
                execution_count=5,
            ),
            Cell(
                id="viz",
                cell_type=CellType.CODE,
                source="plt.plot(df['x']); plt.show()",
                metadata={},
                outputs=[{"output_type": "display_data", "data": {"image/png": "base64data"}}],
                execution_count=6,
            ),
            Cell(
                id="insights",
                cell_type=CellType.MARKDOWN,
                source="## Key Insights\n\n1. The data shows...\n2. We found...",
                metadata={},
                outputs=[],
                execution_count=None,
            ),
        ],
        metadata={},
        kernel_spec={},
        language_info={},
    )

    needs_more2, reason2 = inspect_notebook_quality(complete_notebook)
    print(f"Needs more analysis: {needs_more2}")
    print(f"Reason: {reason2}")


if __name__ == "__main__":
    asyncio.run(test_quality_gate())
