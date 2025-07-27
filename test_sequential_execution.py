#!/usr/bin/env python3
"""Test sequential execution of cells to find the issue."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec.bridge import ExecutionBridge
from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder
from src.spreadsheet_analyzer.plugins.spreadsheet.tasks import DataProfilingTask


async def test_sequential_execution():
    """Test sequential execution of cells."""
    # Get cells from task
    task = DataProfilingTask()
    context = {
        "file_path": "test_assets/collection/business-accounting/Business Accounting.xlsx",
        "sheet_name": "Yiriden Transactions 2025",
    }
    cells = task.build_initial_cells(context)

    # Test 1: Execute cells one by one manually
    print("Test 1: Manual sequential execution")
    print("=" * 60)

    kernel = KernelService(profile=KernelProfile())
    session_id = await kernel.create_session("manual_test")

    try:
        for i, cell in enumerate(cells):
            if cell.cell_type.value == "code":
                code = "".join(cell.source)
                print(f"\nExecuting cell {i} manually...")
                print(f"First line: {code.split('\\n')[0][:50]}...")

                result = await kernel.execute(session_id, code)
                print(f"Result: status={result.status}, outputs={len(result.outputs)}")

                if result.outputs:
                    for output in result.outputs:
                        if output.get("type") == "stream":
                            text = output.get("text", "")[:100]
                            print(f"  Output: {text}...")

                # Add delay between cells
                await asyncio.sleep(0.2)

    finally:
        await kernel.close_session(session_id)

    # Test 2: Execute with bridge but add debugging
    print("\n\nTest 2: Bridge execution with debugging")
    print("=" * 60)

    # Build notebook
    builder = NotebookBuilder()
    for cell in cells:
        builder.add_cell(cell)

    # Create bridge with hooks
    bridge = ExecutionBridge(kernel)

    # Add hooks to debug execution
    async def pre_hook(context):
        cell_idx = context["cell_index"]
        print(f"\n[PRE-EXEC] About to execute cell {cell_idx}")

    async def post_hook(context):
        cell_idx = context["cell_index"]
        result = context["execution_result"]
        print(f"[POST-EXEC] Cell {cell_idx} completed: status={result.status}, outputs={len(result.outputs)}")

    bridge.add_pre_execution_hook(pre_hook)
    bridge.add_post_execution_hook(post_hook)

    session_id2 = await kernel.create_session("bridge_test")
    try:
        executed = await bridge.execute_notebook(session_id2, builder)

        print(f"\n\nFinal result: {len(executed.cells)} cells")
        for i, cell in enumerate(executed.cells):
            if cell.cell_type.value == "code":
                print(f"\nCell {i}: Outputs={len(cell.outputs) if cell.outputs else 0}")

    finally:
        await kernel.close_session(session_id2)
        await kernel.shutdown()


if __name__ == "__main__":
    asyncio.run(test_sequential_execution())
