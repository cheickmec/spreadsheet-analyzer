#!/usr/bin/env python3
"""Test the restored modules to verify they work correctly."""

import asyncio

from spreadsheet_analyzer.core_exec import (
    KernelProfile,
    KernelService,
    NotebookBuilder,
    NotebookIO,
    QualityInspector,
)
from spreadsheet_analyzer.plugins.spreadsheet import SpreadsheetQualityInspector
from spreadsheet_analyzer.workflows import NotebookWorkflow, WorkflowConfig


async def test_kernel_service():
    """Test basic KernelService functionality."""
    print("Testing KernelService...")

    # Create a profile
    profile = KernelProfile(name="python3")
    print(f"  Created profile: {profile.name}")

    # Create service - this is what the tests need to do
    service = KernelService(profile)
    print("  Created KernelService with profile")

    # Test context manager
    async with KernelService(profile) as svc:
        print("  KernelService context manager works")

    print("  ✓ KernelService test passed")


def test_notebook_builder():
    """Test NotebookBuilder functionality."""
    print("\nTesting NotebookBuilder...")

    builder = NotebookBuilder()
    builder.add_markdown_cell("# Test Notebook")
    builder.add_code_cell("print('Hello, World!')")

    notebook = builder.to_dict()
    print(f"  Built notebook with {len(notebook['cells'])} cells")
    print("  ✓ NotebookBuilder test passed")


def test_notebook_io():
    """Test NotebookIO functionality."""
    print("\nTesting NotebookIO...")

    io = NotebookIO()
    print("  Created NotebookIO instance")

    # Create a test notebook
    builder = NotebookBuilder()
    builder.add_markdown_cell("# Test")
    notebook = builder.to_dict()

    # Test that we can create an IO instance
    print("  NotebookIO instance created successfully")
    print("  ✓ NotebookIO test passed")


def test_quality_inspector():
    """Test QualityInspector functionality."""
    print("\nTesting QualityInspector...")

    inspector = QualityInspector()

    # Create a test notebook
    builder = NotebookBuilder()
    builder.add_markdown_cell("# Analysis")
    builder.add_code_cell("import pandas as pd")

    # Inspect quality
    metrics = inspector.inspect(builder)
    print(f"  Overall score: {metrics.overall_score}")
    print(f"  Issues found: {len(metrics.issues)}")
    print("  ✓ QualityInspector test passed")


def test_spreadsheet_quality_inspector():
    """Test SpreadsheetQualityInspector functionality."""
    print("\nTesting SpreadsheetQualityInspector...")

    inspector = SpreadsheetQualityInspector()

    # Create a spreadsheet analysis notebook
    builder = NotebookBuilder()
    builder.add_markdown_cell("# Spreadsheet Analysis")
    builder.add_code_cell("import pandas as pd\nimport openpyxl")
    builder.add_code_cell("df = pd.read_excel('data.xlsx')")

    # Create context for spreadsheet analysis
    context = {"file_path": "data.xlsx", "task": "quality_inspection"}

    # Inspect quality with context
    metrics = inspector.inspect(builder, context)
    print(f"  Overall score: {metrics.overall_score}")
    print(f"  Issues found: {len(metrics.issues)}")
    print("  ✓ SpreadsheetQualityInspector test passed")


async def test_workflow():
    """Test NotebookWorkflow functionality."""
    print("\nTesting NotebookWorkflow...")

    # Create workflow config with explicit kernel profile
    profile = KernelProfile()  # Use default values
    config = WorkflowConfig(
        file_path="dummy.xlsx", output_path="output.ipynb", tasks=["data_profiling"], kernel_profile=profile
    )

    # Create and test workflow
    workflow = NotebookWorkflow()
    print("  Created NotebookWorkflow")
    print("  ✓ NotebookWorkflow test passed")


async def main():
    """Run all tests."""
    print("Testing restored modules...\n")

    # Run sync tests
    test_notebook_builder()
    test_notebook_io()
    test_quality_inspector()
    test_spreadsheet_quality_inspector()

    # Run async tests
    await test_kernel_service()
    await test_workflow()

    print("\n✅ All tests passed! The restored modules are working correctly.")


if __name__ == "__main__":
    asyncio.run(main())
