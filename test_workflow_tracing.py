#!/usr/bin/env python3
"""Trace workflow execution to find where outputs are lost."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s")

from src.spreadsheet_analyzer.workflows.notebook_workflow import NotebookWorkflow, WorkflowConfig, WorkflowMode


async def test_workflow_tracing():
    """Trace notebook workflow execution with detailed logging."""
    config = WorkflowConfig(
        file_path="test_assets/collection/business-accounting/Business Accounting.xlsx",
        sheet_name="Yiriden Transactions 2025",
        output_path="test_workflow_traced.ipynb",
        mode=WorkflowMode.BUILD_AND_EXECUTE,
        tasks=["data_profiling"],
    )

    workflow = NotebookWorkflow()

    # Monkey patch to add logging
    original_build = workflow._build_notebook
    original_execute = workflow._execute_notebook

    async def traced_build(config, result):
        print("\n=== BEFORE BUILD ===")
        print(f"Result notebook ID: {id(result.notebook)}")
        print(f"Result notebook cells: {len(result.notebook.cells)}")

        await original_build(config, result)

        print("\n=== AFTER BUILD ===")
        print(f"Result notebook ID: {id(result.notebook)}")
        print(f"Result notebook cells: {len(result.notebook.cells)}")
        for i, cell in enumerate(result.notebook.cells):
            print(f"  Cell {i}: type={cell.cell_type.value}, outputs={len(cell.outputs) if cell.outputs else 0}")

    async def traced_execute(config, result):
        print("\n=== BEFORE EXECUTE ===")
        print(f"Result notebook ID: {id(result.notebook)}")
        print(f"Result notebook cells: {len(result.notebook.cells)}")
        for i, cell in enumerate(result.notebook.cells):
            print(f"  Cell {i}: type={cell.cell_type.value}, outputs={len(cell.outputs) if cell.outputs else 0}")

        await original_execute(config, result)

        print("\n=== AFTER EXECUTE ===")
        print(f"Result notebook ID: {id(result.notebook)}")
        print(f"Result notebook cells: {len(result.notebook.cells)}")
        for i, cell in enumerate(result.notebook.cells):
            print(
                f"  Cell {i}: type={cell.cell_type.value}, outputs={len(cell.outputs) if cell.outputs else 0}, exec_count={cell.execution_count}"
            )

    workflow._build_notebook = traced_build
    workflow._execute_notebook = traced_execute

    try:
        result = await workflow.run(config)
        print("\n=== FINAL RESULT ===")
        print(f"Errors: {result.errors}")
        print(f"Warnings: {result.warnings}")
        print(f"Result notebook ID: {id(result.notebook)}")
        print(f"Result notebook cells: {len(result.notebook.cells)}")
        for i, cell in enumerate(result.notebook.cells):
            if cell.cell_type.value == "code":
                print(
                    f"  Cell {i}: outputs={len(cell.outputs) if cell.outputs else 0}, exec_count={cell.execution_count}"
                )

    finally:
        await workflow.cleanup()


if __name__ == "__main__":
    asyncio.run(test_workflow_tracing())
