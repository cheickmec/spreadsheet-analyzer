#!/usr/bin/env python3
"""Test ExecutionBridge output attachment to notebook cells."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec.bridge import ExecutionBridge
from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder
from src.spreadsheet_analyzer.core_exec.notebook_io import NotebookIO


async def test_bridge_outputs():
    """Test that ExecutionBridge properly attaches outputs to cells."""
    # Create a kernel service
    kernel = KernelService(profile=KernelProfile())
    bridge = ExecutionBridge(kernel)

    # Create a notebook with the problematic cell
    notebook = NotebookBuilder()
    notebook.add_code_cell("""# Load Excel data
file_path = r"test_assets/collection/business-accounting/Business Accounting.xlsx"
sheet_name = "Yiriden Transactions 2025"

try:
    import pandas as pd
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns from Yiriden Transactions 2025")
    print(f"📊 Data shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    df = pd.DataFrame()  # Empty fallback""")

    print("Original notebook:")
    print(f"Number of cells: {len(notebook.cells)}")
    print(f"Cell 0 outputs: {notebook.cells[0].outputs}")

    # Execute the notebook
    session_id = await kernel.create_session("test")
    try:
        executed_notebook = await bridge.execute_notebook(session_id, notebook)

        print("\nExecuted notebook:")
        print(f"Number of cells: {len(executed_notebook.cells)}")
        print(f"Cell 0 outputs: {executed_notebook.cells[0].outputs}")
        print(f"Cell 0 execution_count: {executed_notebook.cells[0].execution_count}")

        # Save to file for inspection
        io = NotebookIO()
        io.write_notebook(executed_notebook, "test_bridge_executed.ipynb", overwrite=True)
        print("\nSaved executed notebook to test_bridge_executed.ipynb")

    finally:
        await kernel.close_session(session_id)


if __name__ == "__main__":
    asyncio.run(test_bridge_outputs())
