#!/usr/bin/env python3
"""Test ExecutionBridge with multiple cells to identify timing issues."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec.bridge import ExecutionBridge
from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder
from src.spreadsheet_analyzer.core_exec.notebook_io import NotebookIO


async def test_multiple_cells():
    """Test executing multiple cells in sequence."""
    # Create a kernel service
    kernel = KernelService(profile=KernelProfile())
    bridge = ExecutionBridge(kernel)

    # Create a notebook with multiple cells
    notebook = NotebookBuilder()

    # Cell 1: Import
    notebook.add_code_cell("import pandas as pd\nprint('Cell 1: Imports complete')")

    # Cell 2: Load data
    notebook.add_code_cell("""# Load Excel data
file_path = r"test_assets/collection/business-accounting/Business Accounting.xlsx"
sheet_name = "Yiriden Transactions 2025"
df = pd.read_excel(file_path, sheet_name=sheet_name)
print(f"Cell 2: Loaded {len(df)} rows and {len(df.columns)} columns")""")

    # Cell 3: Analysis
    notebook.add_code_cell("""# Basic analysis
print(f"Cell 3: Data shape: {df.shape}")
print(f"Cell 3: Columns: {list(df.columns)}")""")

    print(f"Original notebook has {len(notebook.cells)} cells")

    # Execute the notebook
    session_id = await kernel.create_session("test")
    try:
        executed_notebook = await bridge.execute_notebook(session_id, notebook)

        print(f"\nExecuted notebook has {len(executed_notebook.cells)} cells")
        for i, cell in enumerate(executed_notebook.cells):
            print(f"\nCell {i}:")
            print(f"  Outputs: {len(cell.outputs) if cell.outputs else 0}")
            print(f"  Execution count: {cell.execution_count}")
            if cell.outputs:
                for output in cell.outputs:
                    if output.get("output_type") == "stream":
                        print(f"  Output text: {output.get('text', [])}")

        # Save to file for inspection
        io = NotebookIO()
        io.write_notebook(executed_notebook, "test_bridge_multiple_executed.ipynb", overwrite=True)
        print("\nSaved executed notebook to test_bridge_multiple_executed.ipynb")

    finally:
        await kernel.close_session(session_id)


if __name__ == "__main__":
    asyncio.run(test_multiple_cells())
