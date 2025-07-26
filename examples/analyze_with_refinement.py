#!/usr/bin/env python3
"""Example of analyzing Excel files with iterative refinement.

This script demonstrates the new iterative refinement feature that automatically
fixes data type issues, handles errors, and improves analysis quality through
multiple rounds of LLM interaction.
"""

import json
import subprocess
import sys
from pathlib import Path


def analyze_excel_with_refinement(excel_file: Path, sheet_name: str):
    """Analyze an Excel sheet with iterative refinement enabled."""

    print(f"\n📊 Analyzing: {excel_file.name} - Sheet: {sheet_name}")
    print("=" * 60)

    # Build command with refinement options
    cmd = [
        "uv",
        "run",
        "src/spreadsheet_analyzer/cli/analyze_sheet.py",
        str(excel_file),
        sheet_name,
        "--model",
        "claude-sonnet-4-20250514",
        "--strategy",
        "hierarchical",
        "--max-rounds",
        "3",  # Allow up to 3 refinement rounds
        "--cost-limit",
        "0.50",  # $0.50 cost limit
        "--verbose",  # Show refinement progress
    ]

    print("Running with iterative refinement:")
    print("  --max-rounds 3: Allow up to 3 refinement iterations")
    print("  --cost-limit 0.50: Stop if cost exceeds $0.50")
    print()

    # Run analysis
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print("\n✅ Analysis completed successfully!")

        # Check for refinement details in the log
        base_name = excel_file.stem
        log_path = Path(f"analysis_results/{base_name}/{sheet_name}_llm_log.json")

        if log_path.exists():
            with open(log_path) as f:
                log_data = json.load(f)

            # Count refinement rounds
            requests = [e for e in log_data if e["type"] == "request"]
            if len(requests) > 1:
                print("\n🔄 Refinement Summary:")
                print(f"  - Total rounds: {len(requests)}")
                print(f"  - Total cost: ${sum(e.get('cost', 0) for e in log_data):.4f}")

                # Look for error fixes
                for i, entry in enumerate(log_data):
                    if entry["type"] == "response" and i > 0:
                        content = entry.get("content", "")
                        if "pd.to_numeric" in content or "errors='coerce'" in content:
                            print("  - Fixed: Data type conversion issues")
                        if "try:" in content and "except" in content:
                            print("  - Added: Error handling")
                        if "print(" in content and i == 2:  # Second response
                            print("  - Added: Output statements for visibility")

        # Show output location
        notebook_path = Path(f"analysis_results/{base_name}/{sheet_name}.ipynb")
        if notebook_path.exists():
            print(f"\n📓 Generated notebook: {notebook_path}")
            print('   View with: jupyter notebook "{notebook_path}"')

    else:
        print("\n❌ Analysis failed!")
        return False

    return True


def compare_with_without_refinement(excel_file: Path, sheet_name: str):
    """Compare analysis results with and without refinement."""

    print("\n" + "=" * 60)
    print("🔬 Comparing analysis WITH and WITHOUT refinement")
    print("=" * 60)

    # Run without refinement (single round)
    print("\n1️⃣ Running WITHOUT refinement (--max-rounds 1)...")
    cmd_single = [
        "uv",
        "run",
        "src/spreadsheet_analyzer/cli/analyze_sheet.py",
        str(excel_file),
        sheet_name,
        "--model",
        "claude-sonnet-4-20250514",
        "--max-rounds",
        "1",
        "--output-dir",
        "analysis_results/no_refinement",
        "--verbose",
    ]

    result1 = subprocess.run(cmd_single, capture_output=True, text=True)

    # Run with refinement
    print("\n2️⃣ Running WITH refinement (--max-rounds 3)...")
    cmd_refined = [
        "uv",
        "run",
        "src/spreadsheet_analyzer/cli/analyze_sheet.py",
        str(excel_file),
        sheet_name,
        "--model",
        "claude-sonnet-4-20250514",
        "--max-rounds",
        "3",
        "--output-dir",
        "analysis_results/with_refinement",
        "--verbose",
    ]

    result2 = subprocess.run(cmd_refined, capture_output=True, text=True)

    # Compare results
    print("\n📊 Comparison Results:")

    # Check single round log
    single_log = Path(f"analysis_results/no_refinement/{excel_file.stem}/{sheet_name}_llm_log.json")
    if single_log.exists():
        with open(single_log) as f:
            single_data = json.load(f)
        single_rounds = len([e for e in single_data if e["type"] == "request"])
        single_cost = sum(e.get("cost", 0) for e in single_data)
        print("\nWithout refinement:")
        print(f"  - Rounds: {single_rounds}")
        print(f"  - Cost: ${single_cost:.4f}")
        print(f"  - Errors in output: {'Yes' if 'error' in result1.stdout.lower() else 'No'}")

    # Check refined log
    refined_log = Path(f"analysis_results/with_refinement/{excel_file.stem}/{sheet_name}_llm_log.json")
    if refined_log.exists():
        with open(refined_log) as f:
            refined_data = json.load(f)
        refined_rounds = len([e for e in refined_data if e["type"] == "request"])
        refined_cost = sum(e.get("cost", 0) for e in refined_data)
        print("\nWith refinement:")
        print(f"  - Rounds: {refined_rounds}")
        print(f"  - Cost: ${refined_cost:.4f}")
        print(f"  - Errors fixed: {'Yes' if refined_rounds > 1 else 'N/A'}")


def main():
    """Main example demonstrating iterative refinement."""

    print("🔄 Excel Analysis with Iterative Refinement")
    print("=" * 60)

    # Option 1: Use existing test file
    test_files = [
        Path("examples/sample_data_with_issues.xlsx"),
        Path("test-files/business-accounting/Business Accounting.xlsx"),
    ]

    excel_file = None
    for tf in test_files:
        if tf.exists():
            excel_file = tf
            break

    if not excel_file:
        print("📝 Creating sample Excel file with data issues...")
        # Create sample file
        create_script = Path("examples/create_sample_excel_with_issues.py")
        if create_script.exists():
            subprocess.run(["uv", "run", str(create_script)])
            excel_file = Path("examples/sample_data_with_issues.xlsx")
        else:
            print("❌ No test files found. Please provide an Excel file.")
            sys.exit(1)

    print(f"\n📁 Using file: {excel_file}")

    # List sheets
    print("\n📋 Available sheets:")
    result = subprocess.run(
        ["uv", "run", "src/spreadsheet_analyzer/cli/analyze_sheet.py", str(excel_file), "--list-sheets"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    # Get first sheet name from output
    sheets = []
    for line in result.stdout.split("\n"):
        if line.strip() and not line.startswith("Available") and not line.startswith("Found"):
            sheets.append(line.strip())

    if not sheets:
        print("❌ No sheets found!")
        sys.exit(1)

    # Analyze first sheet
    sheet_name = sheets[0]

    # Example 1: Basic analysis with refinement
    print(f"\n{'=' * 60}")
    print("Example 1: Basic Analysis with Refinement")
    print(f"{'=' * 60}")

    if analyze_excel_with_refinement(excel_file, sheet_name):
        print("\n✨ Refinement helps by:")
        print("  - Automatically fixing data type errors")
        print("  - Adding error handling for edge cases")
        print("  - Ensuring output visibility")
        print("  - Improving analysis completeness")

    # Example 2: Compare with/without refinement
    if len(sheets) > 1:
        print(f"\n{'=' * 60}")
        print("Example 2: Comparison Study")
        print(f"{'=' * 60}")
        compare_with_without_refinement(excel_file, sheets[1])

    # Example 3: Cost control demonstration
    print(f"\n{'=' * 60}")
    print("Example 3: Cost Control")
    print(f"{'=' * 60}")
    print("\nRunning with very low cost limit ($0.02)...")

    cmd_low_cost = [
        "uv",
        "run",
        "src/spreadsheet_analyzer/cli/analyze_sheet.py",
        str(excel_file),
        sheet_name,
        "--model",
        "claude-sonnet-4-20250514",
        "--cost-limit",
        "0.02",
        "--output-dir",
        "analysis_results/low_cost",
        "--verbose",
    ]

    result = subprocess.run(cmd_low_cost, capture_output=True, text=True)
    if "cost limit" in result.stdout.lower():
        print("✅ Cost limit enforced - analysis stopped early to save money")

    # Final summary
    print(f"\n{'=' * 60}")
    print("🎯 Key Takeaways")
    print(f"{'=' * 60}")
    print("\n1. Iterative refinement is now enabled by default")
    print("2. Use --max-rounds to control iterations (default: 3)")
    print("3. Use --cost-limit to control spending (default: $0.50)")
    print("4. The system automatically fixes common Excel data issues:")
    print("   - Text in numeric columns")
    print("   - Currency symbols ($, €, £)")
    print("   - Percentage symbols (%)")
    print("   - Date format inconsistencies")
    print("   - Missing or N/A values")
    print("\n5. Check the LLM logs to see refinement in action!")
    print("\n📂 All results saved in: analysis_results/")


if __name__ == "__main__":
    main()
