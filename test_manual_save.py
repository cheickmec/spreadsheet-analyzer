#!/usr/bin/env python3
"""Quick test to manually run analysis and save the notebook."""

import asyncio
from pathlib import Path

from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
    analyze_sheet_with_langchain,
)


async def main():
    """Run analysis and save manually."""
    excel_path = Path("test-files/business-accounting/Business Accounting.xlsx")
    sheet_name = "Truck Revenue Projections"

    print(f"Running analysis on {sheet_name}...")

    try:
        result = await analyze_sheet_with_langchain(
            excel_path=excel_path,
            sheet_name=sheet_name,
            skip_deterministic=True,
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            enable_tracing=False,
        )

        print(f"\nResult keys: {list(result.keys())}")

        # Check if we have a notebook
        if "notebook_final" in result:
            print(f"Final notebook has {len(result['notebook_final'].get('cells', []))} cells")

            # Count cells with outputs
            cells_with_outputs = sum(
                1
                for cell in result["notebook_final"].get("cells", [])
                if cell.get("outputs") and len(cell.get("outputs", [])) > 0
            )
            print(f"Cells with outputs: {cells_with_outputs}")

        elif "output_path" in result:
            print(f"Notebook saved to: {result['output_path']}")
        else:
            print("No notebook found in results")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
