#!/usr/bin/env python3
"""Quick test of iterative refinement with command-line arguments.

This script demonstrates the new --max-rounds and --cost-limit arguments.
"""

import subprocess
from pathlib import Path


def show_help():
    """Show the updated help text with refinement options."""
    print("📚 Checking new command-line arguments...\n")

    result = subprocess.run(
        ["uv", "run", "src/spreadsheet_analyzer/cli/analyze_sheet.py", "--help"], capture_output=True, text=True
    )

    print(result.stdout)

    # Highlight the new options
    if "--max-rounds" in result.stdout and "--cost-limit" in result.stdout:
        print("\n✅ New refinement options are available!")
        print("   --max-rounds: Control maximum refinement iterations")
        print("   --cost-limit: Set spending limit for LLM calls")
    else:
        print("\n⚠️  Refinement options not found in help text")


def test_refinement_options():
    """Test different refinement configurations."""
    # First, create a simple test file
    print("\n🔧 Creating test Excel file...")

    test_file = Path("test_refinement.xlsx")

    # Create using pandas
    create_code = """
import pandas as pd

# Create data with issues
data = {
    'Product': ['A', 'B', 'C', 'D'],
    'Price': ['$10.50', '15.00', 'N/A', '20'],  # Mixed formats
    'Quantity': ['5', '10 units', '7', 'Eight'],  # Text in numeric
    'Date': ['2024-01-01', '01/02/2024', '2024-01-03', 'Jan 4, 2024']  # Mixed dates
}

df = pd.DataFrame(data)
df.to_excel('test_refinement.xlsx', index=False)
print("Created test_refinement.xlsx")
"""

    result = subprocess.run(["uv", "run", "python", "-c", create_code], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Failed to create test file: {result.stderr}")
        return

    print(result.stdout)

    # Test 1: Default behavior (up to 3 rounds)
    print("\n📊 Test 1: Default refinement (max 3 rounds)...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "src/spreadsheet_analyzer/cli/analyze_sheet.py",
            "test_refinement.xlsx",
            "Sheet1",
            "--verbose",
            "--output-dir",
            "test_output/default",
        ],
        capture_output=False,  # Show output directly
    )

    # Test 2: Single round only
    print("\n\n📊 Test 2: Single round (no refinement)...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "src/spreadsheet_analyzer/cli/analyze_sheet.py",
            "test_refinement.xlsx",
            "Sheet1",
            "--max-rounds",
            "1",
            "--verbose",
            "--output-dir",
            "test_output/single",
        ],
        capture_output=False,
    )

    # Test 3: High round limit
    print("\n\n📊 Test 3: Up to 5 refinement rounds...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "src/spreadsheet_analyzer/cli/analyze_sheet.py",
            "test_refinement.xlsx",
            "Sheet1",
            "--max-rounds",
            "5",
            "--verbose",
            "--output-dir",
            "test_output/five_rounds",
        ],
        capture_output=False,
    )

    # Test 4: Very low cost limit
    print("\n\n📊 Test 4: Low cost limit ($0.05)...")
    result = subprocess.run(
        [
            "uv",
            "run",
            "src/spreadsheet_analyzer/cli/analyze_sheet.py",
            "test_refinement.xlsx",
            "Sheet1",
            "--cost-limit",
            "0.05",
            "--verbose",
            "--output-dir",
            "test_output/low_cost",
        ],
        capture_output=False,
    )

    # Clean up
    if test_file.exists():
        test_file.unlink()
        print(f"\n🧹 Cleaned up {test_file}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("🔄 Testing Iterative Refinement Options")
    print("=" * 60)

    # Show updated help
    show_help()

    # Test refinement options
    print("\n" + "-" * 60)
    test_refinement_options()

    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("Check test_output/ directory for results")
    print("=" * 60)


if __name__ == "__main__":
    main()
