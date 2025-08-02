"""Example usage of functional context management.

This file demonstrates how to use the context management system
to optimize spreadsheet data for LLM consumption.

CLAUDE-KNOWLEDGE: This example shows common patterns for context
optimization in spreadsheet analysis scenarios.
"""

from .builder import create_default_builder, create_minimal_builder
from .strategies import create_hybrid, create_pattern_compression, create_sliding_window
from .token_management import allocate_budget, get_model_config
from .types import ContextQuery


def example_basic_context_building():
    """Basic example of building context from spreadsheet cells."""
    # Sample spreadsheet data
    cells = [
        {"location": "Sheet1!A1", "content": "Product", "type": "value"},
        {"location": "Sheet1!B1", "content": "Sales", "type": "value"},
        {"location": "Sheet1!A2", "content": "Widget A", "type": "value"},
        {"location": "Sheet1!B2", "content": 1000, "type": "value"},
        {"location": "Sheet1!A3", "content": "Widget B", "type": "value"},
        {"location": "Sheet1!B3", "content": 1500, "type": "value"},
        {"location": "Sheet1!C2", "content": "=B2*1.1", "type": "formula"},
        {"location": "Sheet1!C3", "content": "=B3*1.1", "type": "formula"},
    ]

    # Create a context query
    query = ContextQuery(query_text="sales analysis", include_formulas=True, include_values=True)

    # Get model configuration
    model_config = get_model_config("gpt-4").unwrap()

    # Allocate token budget
    budget = allocate_budget(model_config.token_limit, "standard").unwrap()

    # Create builder and build context
    builder = create_default_builder("gpt-4")
    result = builder.build(cells, query, budget)

    if result.is_ok():
        package = result.unwrap()
        print("Context built successfully!")
        print(f"Token count: {package.token_count}")
        print(f"Cells included: {len(package.cells)}")
        print(f"Compression method: {package.compression_method}")
    else:
        print(f"Error: {result.unwrap_err()}")


def example_pattern_compression():
    """Example of compressing repeated patterns."""
    # Spreadsheet with repeated formulas
    cells = []

    # Headers
    cells.append({"location": "Sheet1!A1", "content": "Item", "type": "value"})
    cells.append({"location": "Sheet1!B1", "content": "Price", "type": "value"})
    cells.append({"location": "Sheet1!C1", "content": "Tax", "type": "value"})
    cells.append({"location": "Sheet1!D1", "content": "Total", "type": "value"})

    # Repeated pattern: many rows with similar formulas
    for i in range(2, 102):  # 100 rows
        cells.extend(
            [
                {"location": f"Sheet1!A{i}", "content": f"Item {i - 1}", "type": "value"},
                {"location": f"Sheet1!B{i}", "content": 10 + i, "type": "value"},
                {"location": f"Sheet1!C{i}", "content": f"=B{i}*0.08", "type": "formula"},
                {"location": f"Sheet1!D{i}", "content": f"=B{i}+C{i}", "type": "formula"},
            ]
        )

    # Query focusing on formulas
    query = ContextQuery(query_text="tax calculations", include_formulas=True, include_values=True)

    # Use minimal builder for tight budget
    budget = allocate_budget(4000, "context_heavy").unwrap()
    builder = create_minimal_builder("gpt-3.5-turbo")

    result = builder.build(cells, query, budget)

    if result.is_ok():
        package = result.unwrap()
        print("\nPattern compression example:")
        print(f"Original cells: {len(cells)}")
        print(f"Compressed to: {len(package.cells)} cells")
        print(f"Token reduction: {(1 - package.token_count / budget.context) * 100:.1f}%")

        # Show some compressed cells
        for cell in package.cells[:5]:
            if cell.cell_type == "pattern_summary":
                print(f"\nPattern found: {cell.content}")
                print(f"Metadata: {cell.metadata}")


def example_custom_strategy():
    """Example of creating custom compression strategy."""
    # Create a custom strategy chain
    custom_strategy = create_hybrid(
        # First: keep only important cells
        create_sliding_window(importance_threshold=0.5, prefer_recent=True),
        # Then: compress patterns aggressively
        create_pattern_compression(min_frequency=2),
    )

    # Large dataset
    cells = []
    for sheet in ["Sales", "Costs", "Profit"]:
        for row in range(1, 1001):
            cells.append(
                {"location": f"{sheet}!A{row}", "content": row * 100 if row > 1 else "Amount", "type": "value"}
            )
            if row > 1:
                cells.append({"location": f"{sheet}!B{row}", "content": f"=A{row}*0.15", "type": "formula"})

    # Build with custom strategy
    from .builder import ContextBuilder

    builder = ContextBuilder(default_strategy=custom_strategy, model="gpt-4", include_metadata=True)

    query = ContextQuery(query_text="profit margins", sheet_names=("Profit",), relevance_threshold=0.4)

    budget = allocate_budget(8000, "analysis_heavy").unwrap()
    result = builder.build(cells, query, budget)

    if result.is_ok():
        package = result.unwrap()
        print("\nCustom strategy example:")
        print(f"Focused on sheets: {query.sheet_names}")
        print(f"Cells after filtering: {len(package.cells)}")
        print(f"Focus hints: {package.focus_hints}")


def example_model_specific_optimization():
    """Example of optimizing for different models."""
    cells = [{"location": f"Data!A{i}", "content": f"Value {i}", "type": "value"} for i in range(1, 201)]

    models = ["gpt-3.5-turbo", "gpt-4", "claude-3-opus"]

    for model in models:
        # Get model-specific configuration
        config = get_model_config(model).unwrap()

        # Allocate budget based on model limits
        budget = allocate_budget(config.token_limit, "standard").unwrap()

        # Build context optimized for model
        builder = create_default_builder(model)
        query = ContextQuery(query_text="data analysis")

        result = builder.build(cells, query, budget)

        if result.is_ok():
            package = result.unwrap()
            print(f"\n{model}:")
            print(f"  Token limit: {config.token_limit}")
            print(f"  Context budget: {budget.context}")
            print(f"  Cells included: {len(package.cells)}")
            print(f"  Utilization: {package.token_count / budget.context * 100:.1f}%")


if __name__ == "__main__":
    print("=== Context Management Examples ===\n")

    example_basic_context_building()
    example_pattern_compression()
    example_custom_strategy()
    example_model_specific_optimization()

    print("\n=== Examples Complete ===")
