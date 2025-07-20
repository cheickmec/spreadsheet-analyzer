"""Test the full pipeline to see all stages."""

from pathlib import Path

from spreadsheet_analyzer.pipeline import DeterministicPipeline

# Create pipeline with progress callback
pipeline = DeterministicPipeline()


def progress_callback(update):
    print(f"[{update.stage}] {update.progress * 100:.0f}%: {update.message}")


pipeline.add_progress_observer(progress_callback)

# Run analysis on clean test file
test_file = Path("test_data/clean_test.xlsx")
result = pipeline.run(test_file)

print(f"\nSuccess: {result.success}")
print(f"Execution time: {result.execution_time:.2f}s")

if result.structure:
    print("\nStructure:")
    print(f"  Sheets: {result.structure.sheet_count}")
    print(f"  Total cells: {result.structure.total_cells}")
    print(f"  Total formulas: {result.structure.total_formulas}")

if result.formulas:
    print("\nFormulas:")
    print(f"  Dependency graph size: {len(result.formulas.dependency_graph)}")
    print(f"  Max depth: {result.formulas.max_dependency_depth}")
    print(f"  Circular refs: {len(result.formulas.circular_references)}")

if result.security:
    print("\nSecurity:")
    print(f"  Risk level: {result.security.risk_level}")
    print(f"  Threats: {len(result.security.threats)}")
