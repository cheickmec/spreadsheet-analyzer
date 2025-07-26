#!/usr/bin/env python3
"""Simple test to debug notebook cell execution outputs."""

import asyncio
import logging

from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import execute_notebook_cells
from spreadsheet_analyzer.notebook_llm.nap.protocols import Cell, CellType, NotebookDocument

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Focus on kernel manager logging
logging.getLogger("spreadsheet_analyzer.agents.kernel_manager").setLevel(logging.DEBUG)


async def test_execution():
    """Test direct notebook execution."""
    # Create a simple notebook with a cell that prints something
    notebook = NotebookDocument(
        id="test_notebook",
        cells=[
            Cell(
                id="cell1",
                cell_type=CellType.CODE,
                source="print('Hello from cell 1!')\nprint('Second line')",
                metadata={},
                outputs=[],
            ),
            Cell(
                id="cell2",
                cell_type=CellType.CODE,
                source="x = 5 + 3\nprint(f'Result: {x}')\nx",
                metadata={},
                outputs=[],
            ),
            Cell(
                id="cell3",
                cell_type=CellType.CODE,
                source="print('This is cell 3')\nfor i in range(3):\n    print(f'  Line {i}')",
                metadata={},
                outputs=[],
            ),
        ],
        metadata={},
        kernel_spec={"name": "python3", "display_name": "Python 3"},
        language_info={"name": "python", "version": "3.12"},
    )

    print("Original notebook cells:")
    for i, cell in enumerate(notebook.cells):
        print(f"  Cell {i}: outputs = {cell.outputs}")

    # Execute the notebook
    print("\nExecuting notebook...")
    executed_notebook = await execute_notebook_cells(notebook)

    print("\nExecuted notebook cells:")
    for i, cell in enumerate(executed_notebook.cells):
        print(f"  Cell {i}: outputs = {cell.outputs}")
        if cell.outputs:
            for j, output in enumerate(cell.outputs):
                print(f"    Output {j}: {output}")


if __name__ == "__main__":
    asyncio.run(test_execution())
