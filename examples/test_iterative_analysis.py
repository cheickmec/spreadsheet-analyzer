"""Example script to test iterative refinement with a real Excel file.

This demonstrates how the analyze_sheet.py script now performs iterative
refinement based on execution feedback.
"""

import subprocess
import sys
from pathlib import Path

# Check if test Excel file exists
excel_file = Path("examples/sample_data.xlsx")
if not excel_file.exists():
    print(f"ERROR: {excel_file} not found!")
    print("Please create a sample Excel file first.")
    sys.exit(1)

# Get list of sheets
print("Checking available sheets...")
result = subprocess.run(
    ["uv", "run", "src/spreadsheet_analyzer/cli/analyze_sheet.py", str(excel_file), "--list-sheets"],
    capture_output=True,
    text=True,
)

if result.returncode != 0:
    print(f"Error listing sheets: {result.stderr}")
    sys.exit(1)

print(result.stdout)

# Run analysis on first sheet with verbose output
print("\n" + "=" * 60)
print("Running iterative analysis on first sheet...")
print("Watch for refinement rounds if errors occur!")
print("=" * 60 + "\n")

# Use a model that might make mistakes to demonstrate refinement
cmd = [
    "uv",
    "run",
    "src/spreadsheet_analyzer/cli/analyze_sheet.py",
    str(excel_file),
    "Sheet1",  # Adjust based on actual sheet name
    "--model",
    "claude-sonnet-4-20250514",
    "--verbose",
    "--strategy",
    "hierarchical",
]

print(f"Command: {' '.join(cmd)}\n")

# Run the analysis
result = subprocess.run(cmd)

if result.returncode == 0:
    print("\n✅ Analysis completed successfully!")
    print("Check analysis_results/ directory for the generated notebook.")
else:
    print("\n❌ Analysis failed!")
    sys.exit(1)
