#!/usr/bin/env python3
"""Debug workflow execution to understand why outputs are missing."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")

from src.spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow, WorkflowConfig, WorkflowMode


async def test_workflow_debug():
    """Debug notebook workflow execution."""
    config = WorkflowConfig(
        file_path="test_assets/collection/business-accounting/Business Accounting.xlsx",
        sheet_name="Yiriden Transactions 2025",
        output_path="test_workflow_debug_executed.ipynb",
        mode=WorkflowMode.BUILD_AND_EXECUTE,
        tasks=["data_profiling"],
    )

    workflow = NotebookWorkflow()
    try:
        result = await workflow.run(config)
        print("\nWorkflow completed:")
        print(f"  Errors: {result.errors}")
        print(f"  Warnings: {result.warnings}")

        # Check the notebook in the result object directly
        print(f"\nResult notebook has {len(result.notebook.cells)} cells")
        for i, cell in enumerate(result.notebook.cells):
            if cell.cell_type.value == "code":
                print(f"\nCell {i} (code):")
                print(f"  Outputs: {len(cell.outputs) if cell.outputs else 0}")
                print(f"  Execution count: {cell.execution_count}")
                if cell.outputs:
                    print(f"  Output content: {cell.outputs}")

        # Now check the saved file
        from src.spreadsheet_analyzer.core_exec.notebook_io import NotebookIO

        io = NotebookIO()
        saved_notebook = io.read_notebook("test_workflow_debug_executed.ipynb")
        print(f"\nSaved notebook has {len(saved_notebook.cells)} cells")
        for i, cell in enumerate(saved_notebook.cells):
            if cell.cell_type.value == "code":
                print(f"\nSaved Cell {i} (code):")
                print(f"  Outputs: {len(cell.outputs) if cell.outputs else 0}")
                print(f"  Execution count: {cell.execution_count}")
                if cell.outputs:
                    print(f"  Output content: {cell.outputs}")

    finally:
        await workflow.cleanup()


if __name__ == "__main__":
    asyncio.run(test_workflow_debug())
