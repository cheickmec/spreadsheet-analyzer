#!/usr/bin/env python3
"""Test a simplified version of the workflow."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.spreadsheet_analyzer.core_exec.bridge import ExecutionBridge
from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder


async def test_simple_workflow():
    """Test a minimal workflow."""
    # Create notebook
    builder = NotebookBuilder()
    builder.add_code_cell('print("Cell 1")')
    builder.add_code_cell('print("Cell 2")')
    builder.add_code_cell('x = 5\nprint(f"x = {x}")')

    print(f"Built notebook with {len(builder.cells)} cells")

    # Initialize kernel
    kernel = KernelService(profile=KernelProfile())
    bridge = ExecutionBridge(kernel)

    # Create session and execute
    session_id = await kernel.create_session("test")
    try:
        print("\nExecuting notebook...")
        executed = await bridge.execute_notebook(session_id, builder)

        print(f"\nExecuted notebook has {len(executed.cells)} cells:")
        for i, cell in enumerate(executed.cells):
            if cell.cell_type.value == "code":
                print(f"\nCell {i}:")
                print(f"  Source: {cell.source}")
                print(f"  Outputs: {len(cell.outputs) if cell.outputs else 0}")
                if cell.outputs:
                    for output in cell.outputs:
                        if output.get("output_type") == "stream":
                            print(f"  Output text: {output.get('text')}")

    finally:
        await kernel.close_session(session_id)
        await kernel.shutdown()


if __name__ == "__main__":
    asyncio.run(test_simple_workflow())
