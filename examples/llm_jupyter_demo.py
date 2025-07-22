#!/usr/bin/env python3
"""Demo script showing LLM-Jupyter framework usage.

This script demonstrates:
1. How to use the SpreadsheetLLMAnalyzer
2. Different analysis strategies
3. Result interpretation
4. Exporting findings
"""

import json
from pathlib import Path

from spreadsheet_analyzer.notebook_llm.integration import SpreadsheetLLMAnalyzer, analyze_spreadsheet


def demo_basic_usage():
    """Demonstrate basic analyzer usage."""
    print("\n=== Basic Usage Demo ===\n")

    # Initialize analyzer
    analyzer = SpreadsheetLLMAnalyzer()

    # Load a workbook
    workbook_path = Path("examples/data/financial_model.xlsx")
    if not workbook_path.exists():
        print(f"Sample file not found. Please create: {workbook_path}")
        return

    analyzer.load_workbook(workbook_path)
    print(f"Loaded workbook: {workbook_path}")

    # Run structure analysis
    print("\n1. Running structure analysis...")
    structure_result = analyzer.analyze("structure")
    print(f"   Found {len(structure_result.get('sheets', []))} sheets")

    # Run formula analysis
    print("\n2. Running formula analysis...")
    formula_result = analyzer.analyze("formulas")
    print(f"   Analyzed {formula_result.get('metrics', {}).get('total_formulas', 0)} formulas")

    # Get recommendations
    print("\n3. Getting recommendations...")
    recommendations = analyzer.get_recommendations()
    for rec in recommendations[:3]:  # Show top 3
        print(f"   - [{rec['priority']}] {rec['message']}")

    # Clean up
    analyzer.cleanup()


def demo_advanced_analysis():
    """Demonstrate advanced analysis with custom context."""
    print("\n=== Advanced Analysis Demo ===\n")

    analyzer = SpreadsheetLLMAnalyzer()
    workbook_path = Path("examples/data/sales_data.xlsx")

    if not workbook_path.exists():
        print("Sample file not found. Creating mock analysis...")
        # Mock results for demo
        return demo_mock_analysis()

    analyzer.load_workbook(workbook_path)

    # Custom context for analysis
    context = {
        "business_domain": "sales",
        "expected_patterns": ["monthly trends", "product categories"],
        "validation_rules": {"sales_amount": {"min": 0, "max": 1000000}, "discount": {"min": 0, "max": 0.5}},
    }

    # Run data quality analysis with context
    print("Running data quality analysis with business context...")
    dq_result = analyzer.analyze("data_quality", context=context)

    print("\nData Quality Findings:")
    for issue in dq_result.get("issues", [])[:5]:
        print(f"  - {issue['type']}: {issue['description']}")

    # Run cell-level analysis for specific sheets
    print("\nRunning cell-level analysis on specific sheets...")
    cell_result = analyzer.analyze("cell_level", sheets=["Sales_Q1", "Sales_Q2"], sample_size=100)

    print(f"Analyzed {cell_result.get('cells_analyzed', 0)} cells")

    # Export results
    output_path = Path("examples/output/analysis_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analyzer.export_results(output_path, format="json")
    print(f"\nResults exported to: {output_path}")

    analyzer.cleanup()


def demo_mock_analysis():
    """Demonstrate analysis with mock data when no file available."""
    print("\nUsing mock analysis results for demonstration...\n")

    # Mock analyzer with simulated results
    mock_results = {
        "structure": {
            "sheets": [
                {"name": "Sales_Q1", "rows": 1000, "columns": 20},
                {"name": "Sales_Q2", "rows": 1200, "columns": 20},
                {"name": "Summary", "rows": 50, "columns": 10},
            ],
            "metrics": {"total_sheets": 3, "total_cells": 45000},
            "summary": "3 sheets with quarterly sales data and summary",
        },
        "formulas": {
            "formulas": [
                {
                    "location": "Summary!B10",
                    "formula": "=SUM(Sales_Q1:Sales_Q2!D:D)",
                    "dependencies": ["Sales_Q1!D:D", "Sales_Q2!D:D"],
                    "complexity": "medium",
                }
            ],
            "metrics": {"total_formulas": 125, "complex_formulas": 15},
            "summary": "125 formulas with 15 complex cross-sheet references",
        },
        "data_quality": {
            "issues": [
                {
                    "type": "missing_data",
                    "description": "15% missing values in Sales_Q2 column E",
                    "severity": "medium",
                },
                {"type": "outlier", "description": "Unusual spike in Sales_Q1 row 456", "severity": "low"},
            ],
            "metrics": {"issues_found": 12, "data_completeness": 0.85},
            "summary": "Good data quality with minor gaps in Q2 data",
        },
    }

    # Display mock results
    for strategy, result in mock_results.items():
        print(f"\n{strategy.upper()} Analysis:")
        print(f"  Summary: {result['summary']}")
        print(f"  Metrics: {json.dumps(result['metrics'], indent=4)}")


def demo_convenience_function():
    """Demonstrate the convenience function for quick analysis."""
    print("\n=== Convenience Function Demo ===\n")

    # Quick analysis with all strategies
    workbook_path = Path("examples/data/budget_tracker.xlsx")

    print(f"Analyzing {workbook_path} with all strategies...")

    try:
        summary = analyze_spreadsheet(
            workbook_path,
            strategies=["structure", "formulas"],  # Specific strategies
            output_path=Path("examples/output/quick_analysis.json"),
        )

        print("\nAnalysis Summary:")
        print(json.dumps(summary, indent=2))

    except Exception as e:
        print(f"Analysis failed: {e}")
        print("Using mock data instead...")
        demo_mock_analysis()


def demo_strategy_comparison():
    """Compare results from different analysis strategies."""
    print("\n=== Strategy Comparison Demo ===\n")

    # Create comparison table
    strategies_info = {
        "structure": "Analyzes workbook organization and sheet relationships",
        "formulas": "Examines formula complexity and dependencies",
        "data_quality": "Checks for data issues and validation problems",
        "cell_level": "Performs detailed cell-by-cell analysis",
    }

    print("Available Analysis Strategies:\n")
    for name, description in strategies_info.items():
        print(f"  {name:15} - {description}")

    print("\nStrategy Selection Guide:")
    print("  - Use 'structure' for initial workbook overview")
    print("  - Use 'formulas' when debugging calculation issues")
    print("  - Use 'data_quality' for data validation and cleaning")
    print("  - Use 'cell_level' for detailed forensic analysis")


def demo_result_interpretation():
    """Show how to interpret analysis results."""
    print("\n=== Result Interpretation Demo ===\n")

    # Example result structure
    example_result = {
        "summary": "High-level finding about the analysis",
        "metrics": {"total_items": 150, "issues_found": 12, "coverage": 0.95},
        "details": [
            {"location": "Sheet1!A1", "finding": "Complex formula", "severity": "low"},
            {"location": "Sheet2!B10", "finding": "Circular reference", "severity": "high"},
        ],
        "recommendations": [
            {
                "priority": "high",
                "category": "formulas",
                "message": "Resolve circular reference in Sheet2",
                "action": "Review and refactor formula in Sheet2!B10",
            }
        ],
    }

    print("Understanding Analysis Results:\n")
    print("1. Summary - Quick overview of findings")
    print(f"   Example: '{example_result['summary']}'")

    print("\n2. Metrics - Quantitative measurements")
    for metric, value in example_result["metrics"].items():
        print(f"   {metric}: {value}")

    print("\n3. Details - Specific findings with location")
    for detail in example_result["details"][:2]:
        print(f"   [{detail['severity']}] {detail['location']}: {detail['finding']}")

    print("\n4. Recommendations - Actionable next steps")
    for rec in example_result["recommendations"]:
        print(f"   Priority: {rec['priority']}")
        print(f"   Action: {rec['action']}")


def main():
    """Run all demo functions."""
    print("=" * 60)
    print("LLM-Jupyter Spreadsheet Analysis Framework Demo")
    print("=" * 60)

    # Run demos
    demo_basic_usage()
    demo_advanced_analysis()
    demo_convenience_function()
    demo_strategy_comparison()
    demo_result_interpretation()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
