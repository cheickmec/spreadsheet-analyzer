#!/usr/bin/env python3
"""Test workflow execution with debug logging."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")

from src.spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow, WorkflowConfig, WorkflowMode


async def test_workflow():
    """Test notebook workflow execution."""
    config = WorkflowConfig(
        file_path="test_assets/collection/business-accounting/Business Accounting.xlsx",
        sheet_name="Yiriden Transactions 2025",
        output_path="test_workflow_executed.ipynb",
        mode=WorkflowMode.BUILD_AND_EXECUTE,
        tasks=["data_profiling"],
    )

    workflow = NotebookWorkflow()
    try:
        result = await workflow.run(config)
        print("\nWorkflow completed:")
        print(f"  Errors: {result.errors}")
        print(f"  Warnings: {result.warnings}")
        if result.execution_stats:
            print(f"  Execution stats: {result.execution_stats}")
        print(f"  Output saved to: {result.output_path}")

        # Check the notebook cells
        print(f"\nNotebook has {len(result.notebook.cells)} cells")
        for i, cell in enumerate(result.notebook.cells):
            if cell.cell_type.value == "code":
                print(f"\nCell {i} (code):")
                print(f"  Outputs: {len(cell.outputs) if cell.outputs else 0}")
                print(f"  Execution count: {cell.execution_count}")
                if cell.outputs:
                    print(f"  First output: {cell.outputs[0]}")

    finally:
        await workflow.cleanup()


if __name__ == "__main__":
    asyncio.run(test_workflow())
