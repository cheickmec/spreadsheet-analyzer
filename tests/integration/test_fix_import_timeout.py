#!/usr/bin/env python3
"""Test fixing the import timeout issue."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.spreadsheet_analyzer.core_exec.bridge import ExecutionBridge
from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder
from src.spreadsheet_analyzer.plugins.spreadsheet.tasks import DataProfilingTask


async def test_fix_import_timeout():
    """Test fixing the import timeout issue."""
    # Generate cells from task
    task = DataProfilingTask()
    context = {
        "file_path": "test_assets/collection/business-accounting/Business Accounting.xlsx",
        "sheet_name": "Yiriden Transactions 2025",
    }

    cells = task.build_initial_cells(context)

    # Build notebook
    builder = NotebookBuilder()
    for cell in cells:
        builder.add_cell(cell)

    # Test 1: Use longer timeout profile
    print("Test 1: Using longer timeout profile")
    print("=" * 60)

    # Create profile with longer timeout for imports
    profile = KernelProfile(max_execution_time=60.0)  # 60 seconds instead of default 30
    kernel_service = KernelService(profile=profile)
    bridge = ExecutionBridge(kernel_service)

    session_id = await kernel_service.create_session("long_timeout_test")
    try:
        executed = await bridge.execute_notebook(session_id, builder)

        print(f"\nExecuted notebook has {len(executed.cells)} cells:")
        for i, cell in enumerate(executed.cells):
            if cell.cell_type.value == "code":
                print(f"\nCell {i}:")
                print(f"  Outputs: {len(cell.outputs) if cell.outputs else 0}")
                if cell.outputs:
                    for output in cell.outputs:
                        if output.get("output_type") == "stream":
                            text = output.get("text", [])
                            if isinstance(text, list):
                                text = "".join(text)
                            print(f"  Output: {text[:100]}...")
    finally:
        await kernel_service.close_session(session_id)

    # Test 2: Split imports into smaller chunks
    print("\n\nTest 2: Split imports approach")
    print("=" * 60)

    # Create custom builder with split imports
    builder2 = NotebookBuilder()

    # Split the import cell into two
    builder2.add_code_cell("""# Basic imports
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
print("Basic imports complete")""")

    builder2.add_code_cell("""# Plotting imports (slower)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('default')
sns.set_palette("husl")
print("Plotting imports complete")""")

    # Add remaining cells
    for i, cell in enumerate(cells[1:], 1):
        builder2.add_cell(cell)

    session_id2 = await kernel_service.create_session("split_imports_test")
    try:
        executed2 = await bridge.execute_notebook(session_id2, builder2)

        print(f"\nExecuted notebook has {len(executed2.cells)} cells:")
        for i, cell in enumerate(executed2.cells):
            if cell.cell_type.value == "code":
                print(f"\nCell {i}:")
                print(f"  Outputs: {len(cell.outputs) if cell.outputs else 0}")
                if cell.outputs:
                    for output in cell.outputs:
                        if output.get("output_type") == "stream":
                            text = output.get("text", [])
                            if isinstance(text, list):
                                text = "".join(text)
                            print(f"  Output: {text[:100]}...")
    finally:
        await kernel_service.close_session(session_id2)
        await kernel_service.shutdown()


if __name__ == "__main__":
    asyncio.run(test_fix_import_timeout())
