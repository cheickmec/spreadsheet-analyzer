#!/usr/bin/env python3
"""Test what ExecutionBridge actually returns."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec.bridge import ExecutionBridge
from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow, WorkflowConfig, WorkflowMode


async def test_bridge_return():
    """Test what ExecutionBridge returns when executing a notebook."""
    # First build a notebook using workflow
    config = WorkflowConfig(
        file_path="test_assets/collection/business-accounting/Business Accounting.xlsx",
        sheet_name="Yiriden Transactions 2025",
        output_path=None,  # Don't save
        mode=WorkflowMode.BUILD_ONLY,
        tasks=["data_profiling"],
    )

    workflow = NotebookWorkflow()
    try:
        result = await workflow.run(config)
        built_notebook = result.notebook

        print(f"Built notebook has {len(built_notebook.cells)} cells")
        print(f"Built notebook ID: {id(built_notebook)}")

        # Now execute it manually
        kernel = KernelService(profile=KernelProfile())
        bridge = ExecutionBridge(kernel)

        session_id = await kernel.create_session("test")
        try:
            print("\nExecuting notebook...")
            executed_notebook = await bridge.execute_notebook(session_id, built_notebook)

            print(f"\nExecuted notebook ID: {id(executed_notebook)}")
            print(f"Executed notebook has {len(executed_notebook.cells)} cells")

            for i, cell in enumerate(executed_notebook.cells):
                if cell.cell_type.value == "code":
                    print(f"\nCell {i} (code):")
                    print(f"  Outputs: {len(cell.outputs) if cell.outputs else 0}")
                    print(f"  Execution count: {cell.execution_count}")
                    if cell.outputs:
                        for output in cell.outputs:
                            if output.get("output_type") == "stream":
                                print(f"  Output text: {output.get('text', [])}")

        finally:
            await kernel.close_session(session_id)

    finally:
        await workflow.cleanup()


if __name__ == "__main__":
    asyncio.run(test_bridge_return())
