#!/usr/bin/env python3
"""Test script that intentionally generates incomplete analysis to trigger quality iterations."""

import asyncio
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
    analyze_sheet_with_langchain,
)


async def test_incomplete_analysis():
    """Test with a prompt that generates minimal analysis to trigger iterations."""

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

    print("=== Testing Quality Iterations with Incomplete Analysis ===")
    print(f"Excel: {excel_file}")
    print(f"Sheet: {sheet_name}")
    print()

    # Enable debug logging
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Temporarily modify the system prompt to generate minimal analysis
    from spreadsheet_analyzer.notebook_llm.llm_providers import langchain_integration

    # Save original prompt
    original_prompt = langchain_integration.SYSTEM_PROMPT

    # Use a minimal prompt that will trigger quality gate
    langchain_integration.SYSTEM_PROMPT = """You are a data analyst.
Analyze the Excel sheet by adding only 1-2 simple code cells.
Keep your analysis very brief and basic.
Do NOT include any of these: visualizations, outlier detection, data validation, or detailed insights."""

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

        # Check iteration history
        if final_state.get("quality_iterations", 0) > 1:
            print("\n✅ SUCCESS: Multiple quality iterations performed!")
        else:
            print("\n❌ Only single iteration performed")

    finally:
        # Restore original prompt
        langchain_integration.SYSTEM_PROMPT = original_prompt


if __name__ == "__main__":
    asyncio.run(test_incomplete_analysis())
