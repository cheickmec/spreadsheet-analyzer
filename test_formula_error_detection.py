#!/usr/bin/env python
"""Test script for Excel formula error detection functionality."""

import asyncio
from pathlib import Path

from spreadsheet_analyzer.plugins.spreadsheet.analysis.formula_errors import (
    generate_error_analysis_code,
    scan_workbook_for_errors,
)

# For testing with the sample Excel file
TEST_FILE = Path("examples/data/financial_sample.xlsx")


def test_formula_error_detection():
    """Test the formula error detection functionality."""
    print("=" * 60)
    print("Testing Formula Error Detection")
    print("=" * 60)

    if not TEST_FILE.exists():
        print(f"Test file not found: {TEST_FILE}")
        print("Please provide a valid Excel file path.")
        return

    print(f"\nScanning file: {TEST_FILE}")

    try:
        # Scan for errors
        errors = scan_workbook_for_errors(TEST_FILE)

        if errors:
            print(f"\nFound errors in {len(errors)} sheet(s):")
            for sheet_name, sheet_errors in errors.items():
                print(f"\n  Sheet: {sheet_name}")
                for error_type, error_list in sheet_errors.items():
                    print(f"    {error_type}: {len(error_list)} occurrence(s)")
                    for error in error_list[:3]:  # Show first 3 errors
                        print(f"      - Cell {error['cell']}")
                        if error.get("formula"):
                            print(f"        Formula: {error['formula']}")
        else:
            print("\nNo formula errors found in the workbook!")

        # Generate analysis code
        print("\n" + "=" * 60)
        print("Generated Analysis Code:")
        print("=" * 60)

        code = generate_error_analysis_code(TEST_FILE)
        print(code[:500] + "...\n")  # Show first 500 characters

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback

        traceback.print_exc()


async def test_langchain_integration():
    """Test the formula error detection in LangChain workflow."""
    print("\n" + "=" * 60)
    print("Testing LangChain Integration with Formula Error Detection")
    print("=" * 60)

    try:
        # Get first sheet name
        import openpyxl

        from spreadsheet_analyzer.notebook_llm.llm_providers.langchain_integration import (
            analyze_sheet_with_langchain,
        )

        wb = openpyxl.load_workbook(TEST_FILE, read_only=True)
        sheet_name = wb.sheetnames[0]
        wb.close()

        print(f"\nAnalyzing sheet '{sheet_name}' with LangChain...")

        # Run analysis
        result = await analyze_sheet_with_langchain(
            excel_path=TEST_FILE,
            sheet_name=sheet_name,
            skip_deterministic=True,  # Skip to save time
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.1,
        )

        print("\nAnalysis completed!")
        print(f"Output notebook: {result.get('output_path', 'Not saved')}")
        print(f"Total cost: ${result.get('total_cost', 0):.4f}")

        if result.get("quality_iterations", 0) > 0:
            print(f"\nQuality iterations performed: {result['quality_iterations']}")
            print("Quality reasons:")
            for reason in result.get("quality_reasons", []):
                print(f"  - {reason}")

    except ImportError:
        print("\nLangChain integration not available. Skipping this test.")
    except Exception as e:
        print(f"\nError during LangChain test: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all tests."""
    # Test basic error detection
    test_formula_error_detection()

    # Test LangChain integration
    print("\nWould you like to test the LangChain integration? (y/n)")
    response = input().strip().lower()

    if response == "y":
        asyncio.run(test_langchain_integration())
    else:
        print("\nSkipping LangChain integration test.")

    print("\n" + "=" * 60)
    print("Formula Error Detection Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
