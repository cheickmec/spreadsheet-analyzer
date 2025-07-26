"""Analyze Business Accounting.xlsx with Claude Sonnet 4 using kernel manager.

This example demonstrates the full LangChain/LangGraph workflow with real kernel execution.
It analyzes all sheets in the Business Accounting Excel file and generates notebooks
with executed code cells showing real analysis results.
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
    analyze_sheet_with_langchain,
)
from spreadsheet_analyzer.plugins.spreadsheet.io.excel_io import list_sheets


async def analyze_business_accounting() -> None:
    """Analyze the Business Accounting Excel file with proper kernel manager integration."""

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

    print("=== Analyzing Business Accounting Excel File ===")
    print(f"File: {excel_file.name}")
    print(f"Size: {excel_file.stat().st_size / 1024:.1f} KB")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Get all sheets in the file
    sheets = list_sheets(excel_file)
    if not sheets:
        print("ERROR: No sheets found in the Excel file")
        return

    print(f"Found {len(sheets)} sheets to analyze:")
    for i, sheet in enumerate(sheets, 1):
        print(f"  {i}. {sheet}")
    print()

    # Analyze each sheet using the LangChain integration
    results = {}
    total_cost = 0.0

    for sheet_name in sheets:
        print(f"--- Analyzing Sheet: {sheet_name} ---")

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

            # Store results
            results[sheet_name] = final_state

            # Check for errors
            if final_state.get("execution_errors"):
                print(f"⚠️  Analysis completed with errors for {sheet_name}:")
                for error in final_state["execution_errors"]:
                    print(f"    - {error}")
            else:
                print(f"✓ Analysis completed successfully for {sheet_name}")

            # Report individual sheet results
            if "output_path" in final_state:
                print(f"  Notebook saved to: {final_state['output_path']}")

            # Track token usage and cost
            if "token_usage" in final_state:
                usage = final_state["token_usage"]
                print(f"  Token usage: {usage.get('total_tokens', 'N/A')} tokens")

            if "total_cost" in final_state:
                cost = final_state["total_cost"]
                total_cost += cost
                print(f"  Cost: ${cost:.4f}")

            print()

        except Exception as e:
            print(f"ERROR analyzing sheet '{sheet_name}': {e}")
            results[sheet_name] = {"error": str(e)}
            print()

    # Print final summary
    print("=== Analysis Complete ===")
    print(f"Analyzed {len(sheets)} sheets from {excel_file.name}")
    print(f"Total cost: ${total_cost:.4f}")

    successful_analyses = [sheet for sheet, result in results.items() if "error" not in result]
    failed_analyses = [sheet for sheet, result in results.items() if "error" in result]

    if successful_analyses:
        print(f"\nSuccessful analyses ({len(successful_analyses)}):")
        for sheet in successful_analyses:
            result = results[sheet]
            if "output_path" in result:
                print(f"  - {sheet}: {result['output_path']}")

    if failed_analyses:
        print(f"\nFailed analyses ({len(failed_analyses)}):")
        for sheet in failed_analyses:
            print(f"  - {sheet}: {results[sheet]['error']}")

    print("\nAll notebooks are saved in the analysis_results/ directory.")
    print("Open the notebook files in Jupyter to view the detailed analysis.")


def main() -> None:
    """Main entry point that runs the async analysis."""
    asyncio.run(analyze_business_accounting())


if __name__ == "__main__":
    main()
