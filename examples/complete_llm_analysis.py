#!/usr/bin/env python3
"""Complete example showing full LLM-Jupyter framework integration.

This example demonstrates:
1. Loading a workbook
2. Running multiple analysis strategies
3. Interpreting results
4. Generating actionable recommendations
5. Exporting findings
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from spreadsheet_analyzer.notebook_llm.integration import SpreadsheetLLMAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_financial_model():
    """Analyze a financial model spreadsheet."""
    logger.info("Starting financial model analysis...")

    # Initialize analyzer
    analyzer = SpreadsheetLLMAnalyzer()

    # Define the workbook path
    workbook_path = Path("examples/data/financial_model.xlsx")

    # For demo purposes, create a mock analysis if file doesn't exist
    if not workbook_path.exists():
        logger.warning(f"File not found: {workbook_path}")
        return create_mock_analysis()

    try:
        # Load the workbook
        analyzer.load_workbook(workbook_path)
        logger.info(f"Loaded workbook: {workbook_path}")

        # Define analysis context
        context = {
            "domain": "financial_modeling",
            "focus_areas": ["revenue projections", "cost calculations", "profitability"],
            "time_period": "2024-2026",
            "validation_rules": {"growth_rate": {"min": -0.5, "max": 2.0}, "margin": {"min": 0, "max": 1.0}},
        }

        # 1. Structure Analysis
        logger.info("Running structure analysis...")
        structure_results = analyzer.analyze("structure", context=context)
        print_section("Structure Analysis", structure_results)

        # 2. Formula Analysis
        logger.info("Running formula analysis...")
        formula_results = analyzer.analyze("formulas", context=context)
        print_section("Formula Analysis", formula_results)

        # 3. Data Quality Analysis
        logger.info("Running data quality analysis...")
        dq_results = analyzer.analyze("data_quality", context=context)
        print_section("Data Quality Analysis", dq_results)

        # 4. Get comprehensive summary
        summary = analyzer.get_summary()
        print_section("Analysis Summary", summary)

        # 5. Get actionable recommendations
        recommendations = analyzer.get_recommendations()
        print_recommendations(recommendations)

        # 6. Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("examples/output")
        output_dir.mkdir(exist_ok=True, parents=True)

        # Export as JSON
        json_path = output_dir / f"financial_analysis_{timestamp}.json"
        analyzer.export_results(json_path, format="json")
        logger.info(f"Exported JSON results to: {json_path}")

        # Export as Markdown
        md_path = output_dir / f"financial_analysis_{timestamp}.md"
        analyzer.export_results(md_path, format="markdown")
        logger.info(f"Exported Markdown report to: {md_path}")

        return analyzer.get_summary()

    except Exception:
        logger.exception("Analysis failed")
        raise

    finally:
        # Clean up resources
        analyzer.cleanup()
        logger.info("Cleanup completed")


def create_mock_analysis():
    """Create mock analysis results for demonstration."""
    logger.info("Creating mock analysis for demonstration...")

    mock_results = {
        "workbook": "financial_model.xlsx (mock)",
        "analyses_performed": ["structure", "formulas", "data_quality"],
        "key_findings": [
            {"source": "structure", "finding": "Model has 5 sheets: Assumptions, Revenue, Costs, P&L, Dashboard"},
            {
                "source": "formulas",
                "finding": "Complex SUMIFS and VLOOKUP formulas linking assumptions to calculations",
            },
            {
                "source": "data_quality",
                "finding": "Missing values in Q4 2024 projections, potential circular reference in P&L",
            },
        ],
        "metrics": {
            "structure": {"total_sheets": 5, "total_cells": 15000, "linked_sheets": 4},
            "formulas": {"total_formulas": 450, "complex_formulas": 75, "external_references": 0},
            "data_quality": {"completeness": 0.92, "validation_errors": 3, "outliers": 7},
        },
        "recommendations": [
            {
                "priority": "critical",
                "category": "formulas",
                "message": "Resolve circular reference in P&L sheet",
                "action": "Check formulas in cells P&L!E15:E20",
            },
            {
                "priority": "high",
                "category": "data_quality",
                "message": "Complete missing Q4 2024 projections",
                "action": "Fill in revenue projections for Oct-Dec 2024",
            },
            {
                "priority": "medium",
                "category": "structure",
                "message": "Consider consolidating similar calculations",
                "action": "Merge duplicate formulas in Revenue and Costs sheets",
            },
        ],
    }

    print_section("Mock Analysis Results", mock_results)
    return mock_results


def print_section(title: str, data: dict):
    """Pretty print a section of results."""
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    if isinstance(data, dict):
        # Print summary if available
        if "summary" in data:
            print(f"\nSummary: {data['summary']}")

        # Print metrics if available
        if "metrics" in data:
            print("\nMetrics:")
            for key, value in data["metrics"].items():
                print(f"  - {key}: {value}")

        # Print first few details if available
        if "details" in data and isinstance(data["details"], list):
            print(f"\nTop findings ({len(data['details'])} total):")
            for item in data["details"][:3]:
                print(f"  - {item}")
    else:
        print(json.dumps(data, indent=2))


def print_recommendations(recommendations: list):
    """Pretty print recommendations."""
    print(f"\n{'=' * 60}")
    print("Actionable Recommendations")
    print(f"{'=' * 60}\n")

    if not recommendations:
        print("No recommendations available.")
        return

    # Group by priority
    by_priority = {}
    for rec in recommendations:
        priority = rec.get("priority", "info")
        if priority not in by_priority:
            by_priority[priority] = []
        by_priority[priority].append(rec)

    # Print in priority order
    priority_order = ["critical", "high", "medium", "low", "info"]
    for priority in priority_order:
        if priority in by_priority:
            print(f"\n{priority.upper()} Priority:")
            for rec in by_priority[priority]:
                print(f"\n  [{rec.get('category', 'general')}]")
                print(f"  {rec.get('message', '')}")
                if "action" in rec:
                    print(f"  Action: {rec['action']}")
                if "source" in rec:
                    print(f"  Source: {rec['source']}")


def main():
    """Run the complete analysis example."""
    print("\n" + "=" * 60)
    print("Complete LLM-Jupyter Spreadsheet Analysis")
    print("=" * 60 + "\n")

    try:
        # Run the analysis
        results = analyze_financial_model()

        # Print final summary
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
        print(f"\nAnalyses performed: {', '.join(results.get('analyses_performed', []))}")
        print(f"Total findings: {len(results.get('key_findings', []))}")
        print(f"Recommendations: {results.get('recommendations_count', 0)}")

    except Exception as e:
        logger.exception("Analysis failed")
        print(f"\nError: {e}")
        print("\nRunning mock analysis instead...")
        create_mock_analysis()


if __name__ == "__main__":
    main()
