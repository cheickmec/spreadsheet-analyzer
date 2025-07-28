#!/usr/bin/env python3
"""Test the full workflow with all fixes applied."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.spreadsheet_analyzer.plugins.spreadsheet.tasks import DataProfilingTask
from src.spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow, WorkflowConfig, WorkflowMode


async def test_full_workflow():
    """Test the full workflow execution."""
    # Create workflow
    workflow = NotebookWorkflow()

    # Create config
    config = WorkflowConfig(
        file_path="test_assets/collection/business-accounting/Business Accounting.xlsx",
        output_path="test_output.ipynb",
        sheet_name="Yiriden Transactions 2025",
        mode=WorkflowMode.BUILD_AND_EXECUTE,
        tasks=[DataProfilingTask],
    )

    print("Running full workflow...")
    result = await workflow.run(config)

    print("\nWorkflow completed!")
    print(f"Success: {result.success}")
    print(f"Output path: {result.output_path}")

    if result.notebook:
        print(f"\nNotebook has {len(result.notebook.cells)} cells:")
        for i, cell in enumerate(result.notebook.cells):
            if cell["cell_type"] == "code":
                print(f"\nCell {i}:")
                print(f"  First line: {cell['source'][0][:50] if cell['source'] else 'empty'}...")
                outputs = cell.get("outputs", [])
                print(f"  Outputs: {len(outputs)}")

                if outputs:
                    for j, output in enumerate(outputs):
                        if output.get("output_type") == "stream":
                            text = output.get("text", [])
                            if isinstance(text, list):
                                text = "".join(text)
                            print(f"  Output {j}: {text[:100].strip()}...")
                        elif output.get("output_type") == "error":
                            print(f"  Error: {output.get('ename')}: {output.get('evalue')}")

    if result.errors:
        print(f"\nErrors: {result.errors}")


if __name__ == "__main__":
    asyncio.run(test_full_workflow())
