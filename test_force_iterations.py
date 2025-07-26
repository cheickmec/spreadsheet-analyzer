#!/usr/bin/env python3
"""Test script to force quality iterations by modifying quality inspection."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
    analyze_sheet_with_langchain,
)

# TODO: notebook_io has been refactored - update this test to use core_exec.notebook_io
# from spreadsheet_analyzer.core_exec.notebook_io import NotebookIO


async def test_force_iterations():
    """Test quality iterations by temporarily making quality checks stricter."""

    # Use a test Excel file
    excel_file = Path("test-files/business-accounting/Business Accounting.xlsx")
    sheet_name = "Yiriden mileages"

    if not excel_file.exists():
        print(f"ERROR: Excel file not found: {excel_file}")
        return

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    print("=== Testing Forced Quality Iterations ===")
    print(f"Excel: {excel_file}")
    print(f"Sheet: {sheet_name}")
    print()

    # Enable debug logging
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Save original function
    original_inspect = notebook_io.inspect_notebook_quality

    # Counter for iterations
    iteration_count = [0]

    # Create a stricter quality inspection function
    def strict_inspect_notebook_quality(notebook, min_insights=3):
        """Modified inspection that forces iterations for testing."""
        iteration_count[0] += 1

        # First 2 calls always say needs more
        if iteration_count[0] <= 2:
            reasons = ["Test: Forcing iteration for testing purposes", f"This is iteration {iteration_count[0]} of 3"]
            return True, f"Quality issues: {'; '.join(reasons)}"

        # After 2 iterations, use normal inspection
        return original_inspect(notebook, min_insights)

    # Monkey patch the function
    notebook_io.inspect_notebook_quality = strict_inspect_notebook_quality

    try:
        # Run the analysis
        final_state = await analyze_sheet_with_langchain(
            excel_path=excel_file,
            sheet_name=sheet_name,
            skip_deterministic=True,  # Skip to speed up testing
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            enable_tracing=False,
        )

        # Print final state info
        print("\n=== Final State ===")
        print(f"Quality iterations: {final_state.get('quality_iterations', 0)}")
        print(f"Quality reasons: {final_state.get('quality_reasons', [])}")
        print(f"Needs refinement: {final_state.get('needs_refinement', False)}")
        print(f"Total cost: ${final_state.get('total_cost', 0.0):.4f}")
        print(f"Inspection calls: {iteration_count[0]}")

        # Check iteration history
        if final_state.get("quality_iterations", 0) > 1:
            print("\n✅ SUCCESS: Multiple quality iterations performed!")
            print(f"   - Total iterations: {final_state.get('quality_iterations', 0)}")
            print(f"   - Reasons: {final_state.get('quality_reasons', [])}")
        else:
            print("\n❌ Only single iteration performed")

    finally:
        # Restore original function
        notebook_io.inspect_notebook_quality = original_inspect


if __name__ == "__main__":
    asyncio.run(test_force_iterations())
