#!/usr/bin/env python3
"""Replicate the exact workflow execution to debug missing outputs."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.spreadsheet_analyzer.core_exec.bridge import ExecutionBridge
from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.plugins.spreadsheet.tasks import DataProfilingTask
from src.spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow


async def test_exact_workflow():
    """Test the exact workflow execution path."""
    # Create workflow and task
    workflow = NotebookWorkflow()
    task = DataProfilingTask()

    # Set up context
    context = {
        "file_path": "test_assets/collection/business-accounting/Business Accounting.xlsx",
        "sheet_name": "Yiriden Transactions 2025",
    }

    # Build notebook exactly as workflow does
    print("Building notebook from task...")
    notebook_builder = workflow._build_notebook_from_task(task, context)
    print(f"Built notebook with {len(notebook_builder.cells)} cells")

    # Execute with bridge exactly as workflow does
    kernel_service = KernelService(profile=KernelProfile())
    bridge = ExecutionBridge(kernel_service)

    session_id = await kernel_service.create_session("workflow_test")
    try:
        print("\nExecuting notebook with bridge...")
        executed_notebook = await bridge.execute_notebook(session_id, notebook_builder)

        print(f"\nExecuted notebook has {len(executed_notebook.cells)} cells:")
        for i, cell in enumerate(executed_notebook.cells):
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

    finally:
        await kernel_service.close_session(session_id)
        await kernel_service.shutdown()


if __name__ == "__main__":
    asyncio.run(test_exact_workflow())
