#!/usr/bin/env python3
"""Debug the full workflow execution to find where outputs are lost."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow, WorkflowConfig, WorkflowMode


async def test_full_workflow():
    """Test the complete workflow with debugging."""
    config = WorkflowConfig(
        file_path="test_assets/collection/business-accounting/Business Accounting.xlsx",
        sheet_name="Yiriden Transactions 2025",
        output_path="debug_executed.ipynb",
        mode=WorkflowMode.BUILD_AND_EXECUTE,
        tasks=["data_profiling"],
    )

    workflow = NotebookWorkflow()
    try:
        print("Running workflow...")
        result = await workflow.run(config)

        print("\nWorkflow completed:")
        print(f"  Success: {result.success}")
        print(f"  Errors: {result.errors}")
        print(f"  Warnings: {result.warnings}")

        if result.notebook:
            cells = result.notebook.to_dict()["cells"]
            print(f"\nNotebook has {len(cells)} cells:")

            for i, cell in enumerate(cells):
                if cell["cell_type"] == "code":
                    print(f"\nCell {i} (code):")
                    print(f"  Source preview: {cell['source'][0][:50] if cell['source'] else 'empty'}...")
                    print(f"  Outputs: {len(cell.get('outputs', []))}")
                    if cell.get("outputs"):
                        for j, output in enumerate(cell["outputs"]):
                            print(
                                f"  Output {j}: {output.get('output_type')} - {output.get('text', output.get('data', 'no text/data'))[:100]}"
                            )

        # Check the saved file
        if result.output_path:
            print(f"\nNotebook saved to: {result.output_path}")

    finally:
        await workflow.cleanup()


if __name__ == "__main__":
    asyncio.run(test_full_workflow())
