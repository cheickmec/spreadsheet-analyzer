#!/usr/bin/env python3
"""Demo script showing how to use the Jinja2 template system for LLM prompts."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spreadsheet_analyzer.notebook_llm import (
    StrategyTemplateLoader,
    TemplateManager,
    render_prompt,
)


def demo_basic_rendering():
    """Demonstrate basic template rendering."""
    print("=== Basic Template Rendering ===\n")

    # Create template manager
    manager = TemplateManager()

    # Define context for rendering
    context = {
        "task": {"description": "Analyze Q4 financial results and identify trends"},
        "excel_metadata": {
            "filename": "financial_report_2024.xlsx",
            "size_mb": 3.2,
            "sheet_count": 12,
            "sheet_names": ["Summary", "Q1", "Q2", "Q3", "Q4", "YTD"],
            "total_cells": 25000,
            "formula_count": 1500,
            "last_modified": "2024-01-20",
        },
        "analysis_focus": {
            "sheet_name": "Q4",
            "analysis_type": "Trend Analysis",
            "specific_interest": "Revenue growth patterns",
        },
    }

    # Render default strategy
    result = manager.render("strategies/default/analysis.jinja2", context)
    print("Default Strategy Output:")
    print("-" * 50)
    print(result[:500] + "...")  # Show first 500 chars
    print()


def demo_hierarchical_strategy():
    """Demonstrate hierarchical exploration strategy."""
    print("\n=== Hierarchical Strategy Demo ===\n")

    loader = StrategyTemplateLoader(TemplateManager())

    # Context for hierarchical exploration
    context = {
        "task": {"description": "Explore complex workbook structure"},
        "exploration_level": "overview",
        "notebook": {
            "variables": {
                "wb": {"type": "Workbook", "description": "Loaded Excel workbook"},
                "sheets": {"type": "list", "description": "List of sheet names"},
            },
            "executed_cells": [],
        },
    }

    # Render hierarchical exploration
    result = loader.render_strategy_prompt("hierarchical", "exploration", context)

    print("Hierarchical Strategy Output:")
    print("-" * 50)
    print(result[:500] + "...")
    print()


def demo_graph_based_strategy():
    """Demonstrate graph-based analysis strategy."""
    print("\n=== Graph-Based Strategy Demo ===\n")

    # Use convenience function
    context = {
        "task": {"description": "Analyze formula dependencies in financial model"},
        "focus_cells": ["Summary!B10", "Summary!B20", "Summary!B30"],
        "max_depth": 3,
        "dependency_graph": {
            "top_nodes": [
                {"cell": "Summary!B10", "score": 0.89, "in_degree": 5, "out_degree": 2},
                {"cell": "Q4!C15", "score": 0.76, "in_degree": 3, "out_degree": 4},
                {"cell": "YTD!D20", "score": 0.65, "in_degree": 2, "out_degree": 1},
            ],
            "patterns": [
                {"type": "Aggregation", "description": "Multiple quarters sum to YTD"},
                {"type": "Cascade", "description": "Tax calculations cascade through sheets"},
            ],
        },
    }

    result = render_prompt("strategies/graph_based/analysis.jinja2", context)

    print("Graph-Based Strategy Output:")
    print("-" * 50)
    print(result[:500] + "...")
    print()


def demo_chain_of_thought():
    """Demonstrate chain-of-thought reasoning strategy."""
    print("\n=== Chain-of-Thought Strategy Demo ===\n")

    context = {
        "task": {"description": "Debug formula errors in budget calculations"},
        "previous_thoughts": [
            {"summary": "Identified #REF! errors in Q3 sheet"},
            {"summary": "Found broken links to external data source"},
            {"summary": "Discovered circular reference in tax calculations"},
        ],
        "cell_samples": [
            {"coordinate": "Q3!B15", "formula": "=VLOOKUP(A15,#REF!,2,FALSE)", "data_type": "error"},
            {"coordinate": "Q3!C15", "formula": "=B15*TaxRate", "value": "#REF!", "data_type": "error"},
        ],
    }

    result = render_prompt("strategies/chain_of_thought/reasoning.jinja2", context)

    print("Chain-of-Thought Strategy Output:")
    print("-" * 50)
    print(result[:500] + "...")
    print()


def demo_custom_filters():
    """Demonstrate custom Jinja2 filters."""
    print("\n=== Custom Filters Demo ===\n")

    manager = TemplateManager()

    template_str = """
    Custom Filter Examples:
    - Truncate Middle: {{ long_path | truncate_middle(30) }}
    - Format Cell Ref: {{ 'A1' | format_cell_ref('DataSheet') }}
    - Format Bytes: {{ file_size | format_bytes }}
    - Highlight Formula: {{ formula | highlight_formulas }}
    """

    context = {
        "long_path": "/very/long/path/to/excel/file/in/deep/directory/structure.xlsx",
        "file_size": 5242880,  # 5 MB
        "formula": "The cell contains =SUM(A1:A100) and =AVERAGE(B1:B50)",
    }

    result = manager.render_string(template_str, context)
    print(result)


def list_available_templates():
    """List all available templates."""
    print("\n=== Available Templates ===\n")

    manager = TemplateManager()
    templates = manager.list_templates()

    print(f"Found {len(templates)} templates:\n")

    # Group by category
    base_templates = [t for t in templates if t.startswith("base/")]
    strategy_templates = [t for t in templates if t.startswith("strategies/")]

    print("Base Templates:")
    for template in sorted(base_templates):
        print(f"  - {template}")

    print("\nStrategy Templates:")
    for template in sorted(strategy_templates):
        print(f"  - {template}")


if __name__ == "__main__":
    print("Spreadsheet Analyzer - Template System Demo")
    print("=" * 60)

    # Run all demos
    demo_basic_rendering()
    demo_hierarchical_strategy()
    demo_graph_based_strategy()
    demo_chain_of_thought()
    demo_custom_filters()
    list_available_templates()

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
