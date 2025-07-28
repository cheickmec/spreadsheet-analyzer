#!/usr/bin/env python3
"""Debug script to test the full LangGraph flow with quality iterations."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
    analyze_sheet_with_langchain,
)


async def test_full_flow():
    """Test the full LangGraph flow with a real Excel file."""

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

    print("=== Testing Full LangGraph Flow ===")
    print(f"Excel: {excel_file}")
    print(f"Sheet: {sheet_name}")
    print()

    # Enable debug logging
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Run the analysis with debug output
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
    print(f"Execution errors: {final_state.get('execution_errors', [])}")

    if final_state.get("output_path"):
        print(f"\nOutput saved to: {final_state['output_path']}")


if __name__ == "__main__":
    asyncio.run(test_full_flow())
