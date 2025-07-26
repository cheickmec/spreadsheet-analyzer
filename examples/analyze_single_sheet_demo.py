"""Demo script to analyze a single sheet from Business Accounting.xlsx with kernel manager.

This script demonstrates the LangChain/LangGraph workflow with real kernel execution.
It analyzes one sheet to ensure completion and show the notebook output location.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
    analyze_sheet_with_langchain,
)


async def analyze_single_sheet() -> None:
    """Analyze a single sheet from the Business Accounting Excel file."""

    # Check for Anthropic API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set your Anthropic API key to use Claude Sonnet 4.")
        return

    # File path
    excel_file = Path(__file__).parent.parent / "test-files" / "business-accounting" / "Business Accounting.xlsx"

    if not excel_file.exists():
        print(f"ERROR: Excel file not found: {excel_file}")
        return

    # Analyze just the Truck Revenue Projections sheet
    sheet_name = "Truck Revenue Projections"

    print("=== Single Sheet Analysis Demo ===")
    print(f"File: {excel_file.name}")
    print(f"Sheet: {sheet_name}")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # Use the proper LangChain integration with kernel manager
        final_state = await analyze_sheet_with_langchain(
            excel_path=excel_file,
            sheet_name=sheet_name,
            skip_deterministic=False,  # Include deterministic analysis
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            temperature=0.1,
            enable_tracing=False,  # Disable tracing for cleaner output
        )

        # Check for errors
        if final_state.get("execution_errors"):
            print("⚠️  Analysis completed with some errors:")
            for error in final_state["execution_errors"]:
                print(f"    - {error}")
        else:
            print("✅ Analysis completed successfully!")

        # Report results
        if "output_path" in final_state:
            notebook_path = final_state["output_path"]
            print(f"\n📓 Notebook saved to: {notebook_path}")

            # Verify file exists and show size
            if Path(notebook_path).exists():
                size = Path(notebook_path).stat().st_size
                print(f"   File size: {size:,} bytes")
            else:
                print("   ⚠️  Warning: File not found at expected location")

        if "metadata_path" in final_state:
            print(f"📄 Metadata saved to: {final_state['metadata_path']}")

        # Track token usage and cost
        if "token_usage" in final_state:
            usage = final_state["token_usage"]
            print(f"\n💰 Token usage: {usage.get('total_tokens', 'N/A')} tokens")

        if "total_cost" in final_state:
            cost = final_state["total_cost"]
            print(f"💲 Cost: ${cost:.4f}")

        print("\n🚀 Open the notebook in Jupyter to view the detailed analysis!")
        print(f"   Command: jupyter notebook '{final_state.get('output_path', '')}'")

    except Exception as e:
        print(f"❌ ERROR analyzing sheet: {e}")
        import traceback

        traceback.print_exc()


def main() -> None:
    """Main entry point that runs the async analysis."""
    asyncio.run(analyze_single_sheet())


if __name__ == "__main__":
    main()
