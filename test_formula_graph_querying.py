#!/usr/bin/env python3
"""
Test Formula Graph Querying Capabilities

This script tests the formula dependency graph querying functionality
on the advanced_excel_formulas.xlsx file to ensure accurate graph traversal,
depth calculation, and dependency tracking.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spreadsheet_analyzer.graph_db.query_interface import create_enhanced_query_interface
from spreadsheet_analyzer.pipeline import DeterministicPipeline


def main():
    """Run formula graph querying tests."""
    print("=== Formula Graph Querying Test ===\n")

    # Load the Excel file
    excel_path = Path("test_assets/collection/data-analysis/advanced_excel_formulas.xlsx")
    if not excel_path.exists():
        print(f"ERROR: Excel file not found at {excel_path}")
        return

    print(f"Loading Excel file: {excel_path}")

    # Run the deterministic pipeline
    print("\nRunning deterministic pipeline...")
    pipeline = DeterministicPipeline()
    result = pipeline.run(excel_path)

    if not result.success:
        print(f"ERROR: Pipeline failed: {result.errors}")
        return

    print(f"✅ Pipeline completed in {result.execution_time:.2f} seconds")

    if not result.formulas:
        print("ERROR: No formula analysis results")
        return

    # Create query interface
    print("\nCreating query interface...")
    query_interface = create_enhanced_query_interface(result.formulas)

    # Get formula statistics
    print("\n=== Formula Statistics ===")
    try:
        stats = query_interface.get_formula_statistics_with_ranges()
        print(f"Total formulas: {stats['total_formulas']:,}")
        print(f"Formulas with dependencies: {stats['formulas_with_dependencies']:,}")
        print(f"Unique cells referenced: {stats['unique_cells_referenced']:,}")
        print(f"Max dependency depth: {stats['max_dependency_depth']}")
        print(f"Circular references: {stats['circular_reference_chains']}")
        print(f"Formula complexity score: {stats['complexity_score']:.2f}")
    except Exception as e:
        print(f"ERROR getting formula statistics: {e}")
        # Use basic statistics instead
        print(f"Total formulas in graph: {len(result.formulas.dependency_graph)}")
        print(f"Max dependency depth: {result.formulas.max_dependency_depth}")
        print(f"Circular references: {list(result.formulas.circular_references)}")
        print(f"Formula complexity score: {result.formulas.formula_complexity_score:.2f}")

    # Find some formulas to test
    print("\n=== Testing Specific Formulas ===")

    # Look for formulas with dependencies
    formulas_with_deps = []
    for key, node in result.formulas.dependency_graph.items():
        if node.dependencies:
            formulas_with_deps.append((key, node))

    if formulas_with_deps:
        print(f"\nFound {len(formulas_with_deps)} formulas with dependencies:")
        # Test the first few
        for cell_key, node in formulas_with_deps[:5]:
            sheet, cell_ref = cell_key.split("!")
            print(f"\n--- Testing {cell_key} ---")
            print(f"Formula: {node.formula}")
            print(f"Dependencies: {list(node.dependencies)}")

            # Use query interface to get full dependency info
            dep_result = query_interface.get_cell_dependencies(sheet, cell_ref)
            print(f"Direct dependencies: {dep_result.direct_dependencies}")
            print(f"Range dependencies: {dep_result.range_dependencies}")
            print(f"Direct dependents: {dep_result.direct_dependents}")
            print(f"Total dependencies: {dep_result.total_dependencies}")
            print(f"Total dependents: {dep_result.total_dependents}")
    else:
        print("No formulas with dependencies found!")

    # Test range queries
    print("\n=== Testing Range Queries ===")

    # Find sheets with formulas
    sheets_with_formulas = set()
    for key in result.formulas.dependency_graph:
        sheet = key.split("!")[0]
        sheets_with_formulas.add(sheet)

    print(f"Sheets with formulas: {sorted(sheets_with_formulas)}")

    # Test finding cells affecting a range
    if sheets_with_formulas:
        test_sheet = sorted(sheets_with_formulas)[0]
        print(f"\nTesting cells affecting range A1:A10 in sheet '{test_sheet}':")

        affecting_cells = query_interface.find_cells_affecting_range(test_sheet, "A1", "A10")
        if affecting_cells:
            for cell, deps in list(affecting_cells.items())[:5]:
                print(f"  {cell} affects the range via: {deps}")
        else:
            print("  No cells found affecting this range")

        # Test finding empty cells in formula ranges
        print(f"\nTesting empty cells in formula ranges for sheet '{test_sheet}':")
        empty_cells = query_interface.find_empty_cells_in_formula_ranges(test_sheet)
        if empty_cells:
            print(f"  Found {len(empty_cells)} empty cells in formula ranges")
            print(f"  First 10: {empty_cells[:10]}")
        else:
            print("  No empty cells found in formula ranges")

    # Compare with captured JSON
    print("\n=== Comparing with Captured JSON ===")
    json_path = Path("tests/fixtures/captured_outputs/data-analysis/advanced_excel_formulas.json")
    if json_path.exists():
        with open(json_path) as f:
            captured_data = json.load(f)

        captured_formulas = captured_data["pipeline_result"]["formulas"]
        print(f"Captured formula count: {len(captured_formulas['dependency_graph'])}")
        print(f"Current formula count: {len(result.formulas.dependency_graph)}")

        # Compare max depth
        captured_max_depth = captured_formulas["max_dependency_depth"]
        current_max_depth = result.formulas.max_dependency_depth
        print(f"Captured max depth: {captured_max_depth}")
        print(f"Current max depth: {current_max_depth}")

        if captured_max_depth != current_max_depth:
            print("⚠️  WARNING: Max depth mismatch!")
    else:
        print(f"Captured JSON not found at {json_path}")

    # Look for more complex formulas
    print("\n=== Looking for Complex Formulas ===")
    complex_formulas = []
    for key, node in result.formulas.dependency_graph.items():
        # Look for VLOOKUP, INDEX, MATCH, etc.
        formula_upper = node.formula.upper()
        if any(func in formula_upper for func in ["VLOOKUP", "INDEX", "MATCH", "OFFSET", "INDIRECT"]):
            complex_formulas.append((key, node))

    if complex_formulas:
        print(f"Found {len(complex_formulas)} complex formulas:")
        for cell_key, node in complex_formulas[:5]:
            print(f"\n{cell_key}: {node.formula}")
            print(f"  Dependencies: {list(node.dependencies)}")
            print(f"  Depth: {node.depth}")
    else:
        print("No complex formulas (VLOOKUP, INDEX, MATCH, etc.) found")

    # Test cross-sheet dependencies
    print("\n=== Testing Cross-Sheet Dependencies ===")
    cross_sheet_deps = []
    for key, node in result.formulas.dependency_graph.items():
        sheet = key.split("!")[0]
        for dep in node.dependencies:
            dep_sheet = dep.split("!")[0].strip("'")
            if dep_sheet != sheet:
                cross_sheet_deps.append((key, node, dep))
                if len(cross_sheet_deps) >= 5:
                    break

    if cross_sheet_deps:
        print(f"Found {len([x for x, _, _ in cross_sheet_deps])} formulas with cross-sheet dependencies:")
        for cell_key, node, dep in cross_sheet_deps[:5]:
            print(f"\n{cell_key}: {node.formula}")
            print(f"  Cross-sheet dependency: {dep}")
    else:
        print("No cross-sheet dependencies found")

    # Investigate depth calculation
    print("\n=== Investigating Depth Calculation ===")

    # Check if depth is being calculated
    depths = {}
    none_depth_count = 0
    for key, node in result.formulas.dependency_graph.items():
        if node.depth is None:
            none_depth_count += 1
        else:
            depths[node.depth] = depths.get(node.depth, 0) + 1

    print(f"Formulas with None depth: {none_depth_count}")
    print(f"Depth distribution: {depths}")

    # Find formulas that should have depth > 0
    print("\n=== Finding Dependency Chains ===")
    # Look for formulas that depend on other formulas
    formula_chains = []
    for key, node in result.formulas.dependency_graph.items():
        for dep in node.dependencies:
            if dep in result.formulas.dependency_graph:
                # This formula depends on another formula
                formula_chains.append((key, dep))
                if len(formula_chains) >= 5:
                    break

    if formula_chains:
        print(f"Found {len(formula_chains)} formula dependency chains:")
        for formula, dependency in formula_chains[:5]:
            print(f"  {formula} depends on {dependency}")
            formula_node = result.formulas.dependency_graph[formula]
            dep_node = result.formulas.dependency_graph[dependency]
            print(f"    {formula}: {formula_node.formula}")
            print(f"    {dependency}: {dep_node.formula}")
    else:
        print("No formula dependency chains found")

    # Test specific sheet with many formulas
    print("\n=== Testing Sheet with Many Formulas ===")
    sheet_formula_counts = {}
    for key in result.formulas.dependency_graph:
        sheet = key.split("!")[0]
        sheet_formula_counts[sheet] = sheet_formula_counts.get(sheet, 0) + 1

    # Find sheet with most formulas
    most_formulas_sheet = max(sheet_formula_counts.items(), key=lambda x: x[1])
    print(f"Sheet with most formulas: {most_formulas_sheet[0]} ({most_formulas_sheet[1]} formulas)")

    # Test range queries on this sheet
    test_sheet = most_formulas_sheet[0]
    print(f"\nTesting range B1:B10 in sheet '{test_sheet}':")
    affecting_cells = query_interface.find_cells_affecting_range(test_sheet, "B1", "B10")
    if affecting_cells:
        print(f"  Found {len(affecting_cells)} cells affecting this range")
        for cell, deps in list(affecting_cells.items())[:3]:
            print(f"    {cell} affects the range")
    else:
        print("  No cells found affecting this range")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
