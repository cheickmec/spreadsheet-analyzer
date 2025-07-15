#!/usr/bin/env python3
"""
Analyze a single Excel file with the deterministic pipeline.

Usage:
    python analyze_excel.py <file_path> [options]

Examples:
    python analyze_excel.py test.xlsx
    python analyze_excel.py test.xlsx --strict
    python analyze_excel.py test.xlsx --fast
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spreadsheet_analyzer.pipeline import (
    analyze_with_console_progress,
    create_fast_pipeline_options,
    create_lenient_pipeline_options,
    create_strict_pipeline_options,
)


def main():
    """Analyze a single Excel file."""
    parser = argparse.ArgumentParser(description="Analyze an Excel file with the deterministic pipeline")
    parser.add_argument("file", help="Path to Excel file to analyze")
    parser.add_argument(
        "--mode", choices=["lenient", "strict", "fast"], default="lenient", help="Analysis mode (default: lenient)"
    )
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis results")

    args = parser.parse_args()
    test_file = Path(args.file)

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return 1

    print(f"Analyzing: {test_file}")
    print("=" * 60)

    # Select pipeline options based on mode
    if args.mode == "strict":
        options = create_strict_pipeline_options()
    elif args.mode == "fast":
        options = create_fast_pipeline_options()
    else:
        options = create_lenient_pipeline_options()

    # Run analysis
    result = analyze_with_console_progress(test_file, options=options)

    # Print summary or detailed results
    if not args.detailed:
        # Just print summary
        if result.success:
            print("\n✅ Analysis completed successfully!")
            if result.structure:
                print(
                    f"   Sheets: {result.structure.sheet_count}, Cells: {result.structure.total_cells:,}, Formulas: {result.structure.total_formulas:,}"
                )
            if result.security:
                print(f"   Security: {result.security.risk_level} risk")
            if result.content:
                print(f"   Data Quality: {result.content.data_quality_score}%")
        else:
            print(f"\n❌ Analysis failed: {result.errors[0] if result.errors else 'Unknown error'}")
        return 0 if result.success else 1

    # Detailed results
    print("\nDetailed Results:")
    print("-" * 60)

    if result.integrity:
        print("\nIntegrity Check:")
        print(f"  File size: {result.integrity.metadata.size_mb} MB")
        print(f"  Trust tier: {result.integrity.trust_tier}/5")
        print(f"  Processing class: {result.integrity.processing_class}")

    if result.security:
        print("\nSecurity Scan:")
        print(f"  Risk level: {result.security.risk_level}")
        print(f"  Risk score: {result.security.risk_score}/100")
        print(f"  Has macros: {result.security.has_macros}")
        print(f"  Threats found: {result.security.threat_count}")

    if result.structure:
        print("\nStructural Analysis:")
        print(f"  Sheets: {result.structure.sheet_count}")
        print(f"  Total cells: {result.structure.total_cells:,}")
        print(f"  Total formulas: {result.structure.total_formulas:,}")
        print(f"  Complexity score: {result.structure.complexity_score}/100")

    if result.formulas:
        print("\nFormula Analysis:")
        print(f"  Max dependency depth: {result.formulas.max_dependency_depth}")
        print(f"  Circular references: {result.formulas.has_circular_references}")
        print(f"  Volatile formulas: {len(result.formulas.volatile_formulas)}")
        print(f"  Complexity score: {result.formulas.formula_complexity_score}/100")

    if result.content:
        print("\nContent Intelligence:")
        print(f"  Data quality score: {result.content.data_quality_score}/100")
        print(f"  Patterns found: {len(result.content.data_patterns)}")
        print(f"  Insights generated: {len(result.content.insights)}")

        # Show patterns
        if result.content.data_patterns:
            print("\n  Data Patterns:")
            for pattern in result.content.data_patterns[:5]:  # Show first 5
                print(f"    - {pattern.pattern_type}: {pattern.description}")

        # Show insights
        if result.content.insights:
            print("\n  Key Insights:")
            for insight in result.content.insights[:5]:  # Show first 5
                print(f"    - [{insight.severity}] {insight.title}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
