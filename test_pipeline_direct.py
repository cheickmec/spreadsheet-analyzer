"""Test the pipeline directly without async wrapper."""

from pathlib import Path

from spreadsheet_analyzer.pipeline import DeterministicPipeline

# Create pipeline
pipeline = DeterministicPipeline()

# Run analysis on test file
test_file = Path("test_data/simple_test.xlsx")
result = pipeline.run(test_file)

print(f"Success: {result.success}")
print(f"Errors: {result.errors}")
print(f"Integrity: {result.integrity}")
print(f"Security: {result.security}")
print(f"Structure: {result.structure}")
print(f"Formulas: {result.formulas}")
print(f"Content: {result.content}")

if result.structure:
    print("\nStructure details:")
    print(f"  Sheets: {result.structure.sheet_count}")
    print(f"  Total cells: {result.structure.total_cells}")
    print(f"  Total formulas: {result.structure.total_formulas}")
