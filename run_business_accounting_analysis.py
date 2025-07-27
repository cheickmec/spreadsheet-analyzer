#!/usr/bin/env python
"""Run LangChain analysis on Business Accounting Excel file."""

import asyncio
import os
from pathlib import Path

from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
    analyze_sheet_with_langchain,
)
from spreadsheet_analyzer.plugins.spreadsheet.io.excel_io import list_sheets


async def main():
    """Run analysis on Business Accounting file."""
    # Path to the Business Accounting file
    excel_path = Path("test_assets/collection/business-accounting/Business Accounting.xlsx")

    if not excel_path.exists():
        print(f"Error: File not found at {excel_path}")
        return

    print(f"Analyzing file: {excel_path}")

    # List available sheets
    sheets = list_sheets(excel_path)
    print(f"\nAvailable sheets: {sheets}")

    # Select the first sheet
    sheet_name = sheets[0]
    print(f"\nAnalyzing sheet: '{sheet_name}'")

    # Run analysis with quality-driven iterations
    print("\nStarting LangChain analysis with quality gate...")
    print("This will perform iterative analysis to ensure comprehensive coverage.")
    print("-" * 60)

    try:
        result = await analyze_sheet_with_langchain(
            excel_path=excel_path,
            sheet_name=sheet_name,
            skip_deterministic=False,  # Include deterministic analysis
            provider="anthropic",  # Use Claude
            model="claude-3-5-sonnet-20241022",
            temperature=0.1,
            enable_tracing=False,
        )

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)

        # Display results
        if "output_path" in result:
            print(f"\n✅ Jupyter notebook saved to: {result['output_path']}")

        if "metadata_path" in result:
            print(f"✅ Metadata saved to: {result['metadata_path']}")

        # Show quality iterations
        quality_iterations = result.get("quality_iterations", 0)
        if quality_iterations > 0:
            print(f"\n📊 Quality iterations performed: {quality_iterations}")
            print("Quality improvement reasons:")
            for i, reason in enumerate(result.get("quality_reasons", []), 1):
                print(f"  {i}. {reason}")

        # Show cost information
        total_cost = result.get("total_cost", 0.0)
        print(f"\n💰 Total cost: ${total_cost:.4f}")

        # Show token usage
        token_usage = result.get("token_usage", {})
        if token_usage:
            print("\n📊 Token usage:")
            print(f"  - Prompt tokens: {token_usage.get('prompt_tokens', 0):,}")
            print(f"  - Completion tokens: {token_usage.get('completion_tokens', 0):,}")
            print(f"  - Total tokens: {token_usage.get('total_tokens', 0):,}")

        # Show any errors
        errors = result.get("execution_errors", [])
        if errors:
            print(f"\n⚠️  Errors encountered: {len(errors)}")
            for error in errors:
                print(f"  - {error}")

        print("\n✨ Analysis complete! Open the Jupyter notebook to see the results.")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it before running this script.")
        exit(1)

    # Run the analysis
    asyncio.run(main())
