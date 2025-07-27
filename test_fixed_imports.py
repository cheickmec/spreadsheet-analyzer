#!/usr/bin/env python3
"""Test the fixed import code."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec.bridge import ExecutionBridge
from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder
from src.spreadsheet_analyzer.plugins.spreadsheet.tasks import DataProfilingTask


async def test_fixed_imports():
    """Test the fixed import code."""
    # Generate cells from task
    task = DataProfilingTask()
    context = {
        "file_path": "test_assets/collection/business-accounting/Business Accounting.xlsx",
        "sheet_name": "Yiriden Transactions 2025",
    }

    cells = task.build_initial_cells(context)
    print(f"Task generated {len(cells)} cells")

    # Build notebook
    builder = NotebookBuilder()
    for cell in cells:
        builder.add_cell(cell)

    # Execute with default profile
    kernel_service = KernelService(profile=KernelProfile())
    bridge = ExecutionBridge(kernel_service)

    session_id = await kernel_service.create_session("fixed_test")
    try:
        print("\nExecuting notebook with fixed imports...")
        executed = await bridge.execute_notebook(session_id, builder)

        print(f"\nExecuted notebook has {len(executed.cells)} cells:")
        for i, cell in enumerate(executed.cells):
            if cell.cell_type.value == "code":
                print(f"\nCell {i}:")
                print(f"  First line: {cell.source[0][:50] if cell.source else 'empty'}...")
                print(f"  Outputs: {len(cell.outputs) if cell.outputs else 0}")
                if cell.outputs:
                    for j, output in enumerate(cell.outputs):
                        if output.get("output_type") == "stream":
                            text = output.get("text", [])
                            if isinstance(text, list):
                                text = "".join(text)
                            print(f"  Output {j}: {text.strip()}")
                        elif output.get("output_type") == "error":
                            print(f"  Error: {output.get('ename')}: {output.get('evalue')}")

    finally:
        await kernel_service.close_session(session_id)
        await kernel_service.shutdown()


if __name__ == "__main__":
    asyncio.run(test_fixed_imports())
