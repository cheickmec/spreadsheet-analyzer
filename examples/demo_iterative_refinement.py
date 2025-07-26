#!/usr/bin/env python3
"""Demonstration of iterative refinement with a challenging Excel file.

This script:
1. Creates a sample Excel file with data quality issues
2. Runs the analyzer with verbose output to show refinement rounds
3. Compares results with and without refinement
"""

import json
import subprocess
import sys
from pathlib import Path


def run_analysis(excel_file: Path, sheet_name: str, extra_args: list[str] = None):
    """Run spreadsheet analysis and return the process result."""
    cmd = [
        "uv",
        "run",
        "src/spreadsheet_analyzer/cli/analyze_sheet.py",
        str(excel_file),
        sheet_name,
        "--model",
        "claude-sonnet-4-20250514",
        "--verbose",
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"\n📊 Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def analyze_llm_log(log_path: Path) -> dict:
    """Analyze the LLM log to extract refinement information."""
    if not log_path.exists():
        return {"rounds": 0, "errors_fixed": 0}

    with open(log_path) as f:
        log_data = json.load(f)

    rounds = len([entry for entry in log_data if entry["type"] == "request"])

    # Look for error patterns in responses
    errors_fixed = 0
    for i, entry in enumerate(log_data):
        if entry["type"] == "response" and i > 0:
            if "pd.to_numeric" in entry["content"] or "errors='coerce'" in entry["content"]:
                errors_fixed += 1

    return {
        "rounds": rounds,
        "errors_fixed": errors_fixed,
        "total_cost": sum(entry.get("cost", 0) for entry in log_data),
    }


def main():
    """Run the iterative refinement demonstration."""
    print("=" * 60)
    print("🔄 Iterative Refinement Demonstration")
    print("=" * 60)

    # Step 1: Create sample Excel file
    print("\n📝 Step 1: Creating sample Excel file with data issues...")
    create_script = Path("examples/create_sample_excel_with_issues.py")

    result = subprocess.run(["uv", "run", str(create_script)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to create sample file: {result.stderr}")
        sys.exit(1)

    excel_file = Path("examples/sample_data_with_issues.xlsx")
    if not excel_file.exists():
        print("❌ Sample file was not created!")
        sys.exit(1)

    print(result.stdout)

    # Step 2: List sheets
    print("\n📋 Step 2: Listing available sheets...")
    result = subprocess.run(
        ["uv", "run", "src/spreadsheet_analyzer/cli/analyze_sheet.py", str(excel_file), "--list-sheets"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    # Step 3: Analyze Sales Data sheet (most problematic)
    print("\n🔬 Step 3: Analyzing 'Sales Data' sheet with iterative refinement...")
    print("This sheet has:")
    print("  - Mixed data types (text in numeric columns)")
    print("  - Currency symbols ($) in price fields")
    print("  - Percentage symbols (%) in discount fields")
    print("  - Invalid values and N/A entries")
    print("  - Formula errors due to text values")

    result = run_analysis(excel_file, "Sales Data", ["--strategy", "hierarchical"])

    if result.returncode == 0:
        print("\n✅ Analysis completed successfully!")

        # Check the LLM log for refinement details
        log_path = Path("analysis_results/sample_data_with_issues/Sales Data_llm_log.json")
        if log_path.exists():
            log_analysis = analyze_llm_log(log_path)
            print("\n📊 Refinement Summary:")
            print(f"  - Analysis rounds: {log_analysis['rounds']}")
            print(f"  - Errors fixed: {log_analysis['errors_fixed']}")
            print(f"  - Total cost: ${log_analysis['total_cost']:.4f}")
    else:
        print(f"\n❌ Analysis failed: {result.stderr}")

    # Step 4: Show the generated notebook location
    notebook_path = Path("analysis_results/sample_data_with_issues/Sales Data.ipynb")
    if notebook_path.exists():
        print(f"\n📓 Generated notebook: {notebook_path}")
        print("\nTo view the notebook:")
        print(f'  jupyter notebook "{notebook_path}"')

        # Show a snippet of the analysis
        print("\n📝 Key insights from the analysis:")
        metadata_path = Path("analysis_results/sample_data_with_issues/Sales Data_metadata.json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

            if "ai_analysis" in metadata and "insights" in metadata["ai_analysis"]:
                insights = metadata["ai_analysis"]["insights"]
                print("\nAI-discovered insights:")
                for i, insight in enumerate(insights[:3], 1):
                    print(f"  {i}. {insight}")

    # Step 5: Demonstrate max rounds limit
    print("\n\n🔄 Step 4: Testing max rounds limit...")
    print("Running with --max-rounds 1 to show single-shot behavior:")

    result = run_analysis(
        excel_file, "Inventory", ["--max-rounds", "1", "--output-dir", "analysis_results/single_round"]
    )

    single_log = Path("analysis_results/single_round/sample_data_with_issues/Inventory_llm_log.json")
    if single_log.exists():
        log_analysis = analyze_llm_log(single_log)
        print(f"\n  Single round analysis: {log_analysis['rounds']} round(s)")

    # Step 6: Test cost limit
    print("\n💰 Step 5: Testing cost limit...")
    print("Running with --cost-limit 0.01 (very low) to trigger early stop:")

    result = run_analysis(
        excel_file, "Financial Summary", ["--cost-limit", "0.01", "--output-dir", "analysis_results/cost_limited"]
    )

    if "Cost limit" in result.stdout:
        print("  ✓ Cost limit enforcement working correctly")

    print("\n" + "=" * 60)
    print("🎉 Demonstration Complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("1. The analyzer automatically detected and fixed data type issues")
    print("2. Multiple refinement rounds were used to handle errors")
    print("3. Cost tracking prevented runaway LLM usage")
    print("4. The final analysis included proper data cleaning code")
    print("\nCheck the analysis_results/ directory for full outputs!")


if __name__ == "__main__":
    main()
