#!/usr/bin/env python
"""Run deterministic analysis with execution (no LLM)."""

import asyncio
from pathlib import Path

from spreadsheet_analyzer.core_exec import KernelProfile
from spreadsheet_analyzer.plugins.base import registry
from spreadsheet_analyzer.plugins.spreadsheet import register_all_plugins as register_spreadsheet_plugins
from spreadsheet_analyzer.workflows import NotebookWorkflow, WorkflowConfig, WorkflowMode


async def main():
    """Run deterministic analysis with execution."""
    # Register plugins
    register_spreadsheet_plugins()

    # Configuration
    excel_path = Path("test_assets/collection/business-accounting/Business Accounting.xlsx")

    # Get the first available sheet
    from spreadsheet_analyzer.plugins.spreadsheet.io.excel_io import list_sheets

    sheets = list_sheets(excel_path)
    sheet_name = sheets[0] if sheets else "Sheet1"  # Use first sheet

    output_dir = Path("analysis_results/Business Accounting")
    output_path = output_dir / f"{sheet_name.replace(' ', '_')}_executed.ipynb"

    if not excel_path.exists():
        print(f"Error: File not found at {excel_path}")
        return

    print(f"🔍 Analyzing {excel_path.name}")
    print(f"   📄 Sheet: {sheet_name}")
    print("   🔧 Mode: Deterministic with execution (no LLM)")
    print(f"   💾 Output: {output_path}")
    print("-" * 60)

    # Get available tasks
    tasks = registry.list_tasks()

    # Filter for deterministic tasks only
    print("\n🔍 Checking tasks for deterministic flag:")
    for task in tasks:
        has_flag = hasattr(task, "is_deterministic")
        flag_value = getattr(task, "is_deterministic", None) if has_flag else None
        print(f"   - {task.name}: has_flag={has_flag}, value={flag_value}")

    deterministic_tasks = [task for task in tasks if hasattr(task, "is_deterministic") and task.is_deterministic]
    if not deterministic_tasks:
        print("   ℹ️  No tasks have is_deterministic flag, using all tasks")
        deterministic_tasks = tasks  # Use all if no explicit deterministic flag

    # Get task names
    task_names = [task.name for task in deterministic_tasks]

    print(f"📋 Using {len(task_names)} deterministic tasks:")
    for name in task_names:
        print(f"   - {name}")

    # Create workflow config with BUILD_AND_EXECUTE mode
    config = WorkflowConfig(
        file_path=str(excel_path),
        output_path=str(output_path),
        sheet_name=sheet_name,
        mode=WorkflowMode.BUILD_AND_EXECUTE,  # Force execution mode
        tasks=task_names,
        kernel_profile=KernelProfile(
            name="python3",
            max_execution_time=120,
            idle_timeout_seconds=60,
            wait_for_shell_reply=True,
            output_drain_timeout_ms=500,
            output_drain_max_timeout_ms=2000,
            output_drain_max_attempts=5,
        ),
        execute_timeout=120,
    )

    # Run workflow
    workflow = NotebookWorkflow()
    try:
        result = await workflow.run(config)
    finally:
        await workflow.cleanup()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
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
        print("🎉 Open the notebook to see the deterministic analysis outputs!")
        print(f'\n🚀 jupyter notebook "{result.output_path}"')


if __name__ == "__main__":
    asyncio.run(main())
