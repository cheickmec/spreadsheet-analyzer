#!/usr/bin/env python3
"""
Explore what a specific test file produces when analyzed.

This script demonstrates how to load and work with captured test outputs
as proper dataclasses.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spreadsheet_analyzer.pipeline.types import PipelineResult
from spreadsheet_analyzer.testing.loader import FixtureLoader


def explore_test_output(test_file: str) -> None:
    """
    Load and display the analysis output for a specific test file.

    Args:
        test_file: Relative path to test file (e.g., "business-accounting/Business Accounting.xlsx")
    """
    # Convert to fixture path
    fixture_name = Path(test_file).with_suffix(".json")
    fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "captured_outputs" / fixture_name

    if not fixture_path.exists():
        print(f"No fixture found for: {test_file}")
        print(f"Looking for: {fixture_path}")
        return

    # Load fixture
    with fixture_path.open() as f:
        data = json.load(f)

    # Display metadata
    print(f"\n{'=' * 80}")
    print(f"TEST FILE: {test_file}")
    print(f"{'=' * 80}\n")

    metadata = data.get("metadata", {})
    print("METADATA:")
    print(f"  File Size: {metadata.get('file_size', 0):,} bytes")
    print(f"  Processing Time: {metadata.get('processing_time', 0):.3f}s")
    print(f"  Success: {metadata.get('pipeline_success', False)}")

    # Load using fixture loader
    try:
        loader = FixtureLoader()
        result = loader.load_as_dataclass(test_file)
        explore_pipeline_result(result)
    except Exception as e:
        print(f"\nError loading result: {e}")
        print("\nRaw pipeline result:")
        print(json.dumps(data.get("pipeline_result", {}), indent=2)[:500] + "...")


def explore_pipeline_result(result: PipelineResult) -> None:
    """Display information from the pipeline result using dataclass fields."""
    print("\nPIPELINE RESULT (as dataclass):")
    print(f"  Type: {type(result).__name__}")
    print(f"  Success: {result.success}")
    print(f"  Execution Time: {result.execution_time:.3f}s")

    if result.errors:
        print(f"  Errors: {list(result.errors)}")

    # Stage 0: Integrity
    if result.integrity:
        print("\n  INTEGRITY (Stage 0):")
        print(f"    Trust Tier: {result.integrity.trust_tier}")
        print(f"    Processing Class: {result.integrity.processing_class}")
        print(f"    Is Excel: {result.integrity.is_excel}")
        print(f"    Is OOXML: {result.integrity.is_ooxml}")

    # Stage 1: Security
    if result.security:
        print("\n  SECURITY (Stage 1):")
        print(f"    Risk Level: {result.security.risk_level}")
        print(f"    Risk Score: {result.security.risk_score}/100")
        print(f"    Has Macros: {result.security.has_macros}")
        print(f"    Threat Count: {result.security.threat_count}")
        if result.security.threats:
            print(f"    First Threat: {result.security.threats[0].threat_type}")

    # Stage 2: Structure
    if result.structure:
        print("\n  STRUCTURE (Stage 2):")
        print(f"    Sheet Count: {result.structure.sheet_count}")
        print(f"    Total Cells: {result.structure.total_cells:,}")
        print(f"    Total Formulas: {result.structure.total_formulas:,}")
        print(f"    Complexity Score: {result.structure.complexity_score}/100")
        if result.structure.sheets:
            print(f"    First Sheet: {result.structure.sheets[0].name}")
            print(f"      - Rows: {result.structure.sheets[0].row_count}")
            print(f"      - Cells: {result.structure.sheets[0].cell_count:,}")

    # Stage 3: Formulas
    if result.formulas:
        print("\n  FORMULAS (Stage 3):")
        print(f"    Has Circular Refs: {result.formulas.has_circular_references}")
        print(f"    Max Dependency Depth: {result.formulas.max_dependency_depth}")
        print(f"    Formula Complexity: {result.formulas.formula_complexity_score}")
        if result.formulas.volatile_formulas:
            print(f"    Volatile Formula Count: {len(result.formulas.volatile_formulas)}")

    # Stage 4: Content
    if result.content:
        print("\n  CONTENT (Stage 4):")
        print(f"    Data Quality Score: {result.content.data_quality_score}/100")
        print(f"    Pattern Count: {len(result.content.data_patterns)}")
        print(f"    Insight Count: {len(result.content.insights)}")
        if result.content.summary:
            print(f"    Summary: {result.content.summary[:100]}...")


def list_available_fixtures() -> None:
    """List all available test file fixtures."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures" / "captured_outputs"

    print("\nAvailable test file fixtures:")
    print("-" * 40)

    for json_file in sorted(fixtures_dir.rglob("*.json")):
        if json_file.name in ["capture_summary.json", "fixture_manifest.json"]:
            continue

        relative_path = json_file.relative_to(fixtures_dir)
        test_file = str(relative_path).replace(".json", ".xlsx")

        # Check for .xlsm
        if not (fixtures_dir.parent.parent.parent / "test-files" / test_file).exists():
            test_file = str(relative_path).replace(".json", ".xlsm")

        print(f"  {test_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Explore captured test file outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available fixtures
  %(prog)s --list

  # Explore a specific test file output
  %(prog)s "business-accounting/Business Accounting.xlsx"

  # Explore with shorter path
  %(prog)s "data-analysis/advanced_excel_formulas.xlsx"
        """,
    )

    parser.add_argument("test_file", nargs="?", help='Test file to explore (e.g., "category/file.xlsx")')

    parser.add_argument("--list", action="store_true", help="List available test file fixtures")

    args = parser.parse_args()

    if args.list or not args.test_file:
        list_available_fixtures()
        if not args.test_file:
            print("\nUse one of the above with this script to explore its output.")
    else:
        explore_test_output(args.test_file)


if __name__ == "__main__":
    main()
