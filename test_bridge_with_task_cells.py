#!/usr/bin/env python3
"""Test ExecutionBridge with actual task-generated cells."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec.bridge import ExecutionBridge
from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder
from src.spreadsheet_analyzer.plugins.spreadsheet.tasks import DataProfilingTask


async def test_bridge_with_task_cells():
    """Test ExecutionBridge with cells from DataProfilingTask."""
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

    print(f"Built notebook with {len(builder.cells)} cells")

    # Test 1: Execute with ExecutionBridge
    print("\n" + "=" * 60)
    print("Test 1: ExecutionBridge execution")
    print("=" * 60)

    kernel_service = KernelService(profile=KernelProfile())
    bridge = ExecutionBridge(kernel_service)

    session_id = await kernel_service.create_session("bridge_test")
    try:
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
                            print(f"  Output {j}: {text[:100]}...")
                else:
                    # Show the actual code that should produce output
                    if i == 1:  # Setup cell
                        print("  Expected output from:")
                        print(f"    {cell.source[7]}")  # The print statement
                    elif i == 2:  # Data loading cell
                        print("  Expected outputs from:")
                        for line in cell.source:
                            if "print(" in line:
                                print(f"    {line.strip()}")
    finally:
        await kernel_service.close_session(session_id)

    # Test 2: Execute cells individually
    print("\n" + "=" * 60)
    print("Test 2: Individual cell execution")
    print("=" * 60)

    session_id2 = await kernel_service.create_session("individual_test")
    try:
        for i, cell in enumerate(builder.cells):
            if cell.cell_type.value == "code":
                code = "".join(cell.source)
                print(f"\nExecuting cell {i} individually...")
                result = await kernel_service.execute(session_id2, code)
                print(f"  Status: {result.status}")
                print(f"  Outputs: {len(result.outputs)}")
                if result.outputs:
                    for output in result.outputs:
                        if output.get("type") == "stream":
                            print(f"  Output: {output.get('text', '')[:100]}...")

    finally:
        await kernel_service.close_session(session_id2)
        await kernel_service.shutdown()


if __name__ == "__main__":
    asyncio.run(test_bridge_with_task_cells())
