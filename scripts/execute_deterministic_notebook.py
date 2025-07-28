#!/usr/bin/env python
"""Execute the already generated deterministic notebook."""

import asyncio
from pathlib import Path

from spreadsheet_analyzer.core_exec.kernel_service import KernelProfile
from spreadsheet_analyzer.workflows.notebook_workflow import WorkflowConfig, WorkflowMode


async def main():
    """Execute the existing deterministic analysis notebook."""
    # Path to the generated notebook
    notebook_path = "analysis_results/Business Accounting/Sheet1.ipynb"

    if not Path(notebook_path).exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return

    print(f"Executing notebook: {notebook_path}")
    print("-" * 60)

    try:
        # Create workflow config with proper kernel profile
        from spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow

        config = WorkflowConfig(
            file_path=notebook_path,
            output_path=notebook_path,  # Overwrite with executed version
            mode=WorkflowMode.EXECUTE_EXISTING,
            kernel_profile=KernelProfile(
                name="python3",
                max_execution_time=120,
                idle_timeout_seconds=60,
            ),
            execute_timeout=120,
        )

        # Run the workflow
        workflow = NotebookWorkflow()
        try:
            result = await workflow.run(config)
        finally:
            await workflow.cleanup()

        print("\n" + "=" * 60)
        print("EXECUTION COMPLETE!")
        print("=" * 60)

        # Show execution statistics
        if result.execution_stats:
            stats = result.execution_stats
            print(f"\n✅ Executed cells: {stats.executed_cells}")
            print(f"✅ Successful cells: {stats.successful_cells}")

            if stats.failed_cells > 0:
                print(f"⚠️  Failed cells: {stats.failed_cells}")

            print(f"⏱️  Total execution time: {stats.total_execution_time:.2f}s")

            if stats.cell_execution_times:
                print("\n📊 Cell execution times:")
                for i, time in enumerate(stats.cell_execution_times, 1):
                    print(f"  Cell {i}: {time:.2f}s")

        # Show any errors
        if result.errors:
            print("\n❌ Errors encountered:")
            for error in result.errors:
                print(f"  - {error}")

        # Show any warnings
        if result.warnings:
            print("\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

        if result.output_path:
            print(f"\n✨ Executed notebook saved to: {result.output_path}")
            print("Open the notebook to see the actual outputs from the deterministic analysis!")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
