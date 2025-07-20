"""Test CLI with lenient security settings."""

from pathlib import Path

from spreadsheet_analyzer.pipeline import DeterministicPipeline, create_lenient_pipeline_options

# Create pipeline with lenient options
options = create_lenient_pipeline_options()
pipeline = DeterministicPipeline(options)

# Run analysis
test_file = Path("test_data/clean_test.xlsx")
result = pipeline.run(test_file)

print(f"Success: {result.success}")
print(f"Errors: {result.errors}")

if result.structure:
    print("\nStructure:")
    print(f"  Sheets: {result.structure.sheet_count}")
    print(f"  Total cells: {result.structure.total_cells}")
    print(f"  Total formulas: {result.structure.total_formulas}")
