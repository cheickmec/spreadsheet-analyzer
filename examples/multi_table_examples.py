#!/usr/bin/env python3
"""Examples demonstrating different ways to use the multi-table detection workflow.

This file shows:
1. Basic workflow usage
2. Custom configuration
3. Direct agent usage
4. Integration with existing analysis
"""

import asyncio
from pathlib import Path

import pandas as pd
from langchain_core.messages import HumanMessage

from spreadsheet_analyzer.agents.table_detector_agent import create_table_detector
from spreadsheet_analyzer.agents.types import AgentId, AgentMessage, AgentState
from spreadsheet_analyzer.cli.notebook_analysis import AnalysisConfig
from spreadsheet_analyzer.workflows.multi_table_workflow import (
    SpreadsheetAnalysisState,
    create_multi_table_workflow,
    run_multi_table_analysis,
)


async def example_1_basic_workflow() -> None:
    """Example 1: Basic usage of the multi-table workflow."""
    print("\n=== Example 1: Basic Workflow ===")

    # Create a simple test file
    test_file = Path("test_data/basic_multi_table.xlsx")
    test_file.parent.mkdir(exist_ok=True)

    # Create sample data
    df1 = pd.DataFrame({"ID": range(1, 11), "Value": range(100, 110)})
    empty = pd.DataFrame([[None, None]] * 2)
    df2 = pd.DataFrame({"Category": ["A", "B", "C"], "Total": [500, 600, 700]})

    combined = pd.concat([df1, empty, df2], ignore_index=True)
    combined.to_excel(test_file, index=False)

    # Run workflow with minimal config
    result = await run_multi_table_analysis(test_file)

    if result.is_ok():
        data = result.unwrap()
        print(f"✓ Found {data['tables_found']} tables")
        print(f"✓ Detection notebook: {data['detection_notebook']}")
        print(f"✓ Analysis notebook: {data['analysis_notebook']}")
    else:
        print(f"✗ Error: {result.unwrap_err()}")


async def example_2_custom_config() -> None:
    """Example 2: Using custom configuration."""
    print("\n=== Example 2: Custom Configuration ===")

    test_file = Path("test_data/custom_config.xlsx")
    test_file.parent.mkdir(exist_ok=True)

    # Create more complex data
    orders = pd.DataFrame(
        {
            "Order ID": [f"ORD-{i:04d}" for i in range(1, 51)],
            "Customer": [f"Customer {i % 10}" for i in range(1, 51)],
            "Amount": [100 * i for i in range(1, 51)],
        }
    )

    orders.to_excel(test_file, index=False)

    # Custom configuration
    config = AnalysisConfig(
        excel_path=test_file,
        sheet_index=0,
        output_dir=Path("outputs/custom_analysis"),
        model="gpt-4o-mini",  # Or your preferred model
        max_rounds=5,  # More thorough analysis
        auto_save_rounds=True,
        verbose=True,
        track_costs=True,  # Enable cost tracking
    )

    result = await run_multi_table_analysis(test_file, config=config)

    if result.is_ok():
        print("✓ Analysis complete with custom config")
    else:
        print(f"✗ Error: {result.unwrap_err()}")


async def example_3_direct_agent_usage() -> None:
    """Example 3: Using the table detector agent directly."""
    print("\n=== Example 3: Direct Agent Usage ===")

    # Create sample dataframe
    df = pd.DataFrame(
        {
            "A": [1, 2, 3, None, None, 10, 20, 30],
            "B": ["x", "y", "z", None, None, "p", "q", "r"],
            "C": [100, 200, 300, None, None, 1000, 2000, 3000],
        }
    )

    # Create detector agent
    detector = create_table_detector()

    # Create message
    message = AgentMessage.create(
        sender=AgentId.generate("test"),
        receiver=detector.id,
        content={"dataframe": df, "sheet_name": "DirectTest", "file_path": "direct_test.xlsx"},
    )

    # Run detection
    state = AgentState(agent_id=detector.id, status="idle")
    result = detector.process(message, state)

    if result.is_ok():
        detection = result.unwrap().content
        print(f"✓ Detected {len(detection.tables)} tables:")
        for table in detection.tables:
            print(f"  - {table.description}: rows {table.start_row}-{table.end_row}")
    else:
        print(f"✗ Error: {result.unwrap_err()}")


async def example_4_workflow_customization() -> None:
    """Example 4: Customizing the workflow with your own state."""
    print("\n=== Example 4: Workflow Customization ===")

    test_file = Path("test_data/workflow_custom.xlsx")
    test_file.parent.mkdir(exist_ok=True)

    # Create test data
    pd.DataFrame(
        {
            "Metric": ["Revenue", "Cost", "Profit"],
            "Q1": [1000, 600, 400],
            "Q2": [1200, 700, 500],
        }
    ).to_excel(test_file, index=False)

    # Create workflow
    workflow = create_multi_table_workflow()

    # Custom initial state
    initial_state = SpreadsheetAnalysisState(
        excel_file_path=str(test_file),
        sheet_index=0,
        sheet_name="Financial",
        config=AnalysisConfig(
            excel_path=test_file,
            sheet_index=0,
            output_dir=Path("outputs/custom_workflow"),
            model="gpt-4o-mini",
        ),
        table_boundaries=None,
        detection_notebook_path=None,
        detection_error=None,
        analysis_notebook_path=None,
        analysis_error=None,
        messages=[HumanMessage(content="Analyze financial metrics with focus on trends")],
        current_agent="",
        workflow_complete=False,
    )

    # Run workflow
    final_state = await workflow.ainvoke(initial_state)

    print(f"✓ Workflow complete: {final_state.get('workflow_complete', False)}")
    if final_state.get("table_boundaries"):
        print(f"✓ Detected {len(final_state['table_boundaries'].tables)} tables")


async def example_5_integration_with_existing() -> None:
    """Example 5: Integrating with existing analysis pipeline."""
    print("\n=== Example 5: Integration with Existing Pipeline ===")

    test_file = Path("test_data/integration.xlsx")
    test_file.parent.mkdir(exist_ok=True)

    # Create complex multi-sheet file
    with pd.ExcelWriter(test_file) as writer:
        # Sheet 1: Sales data
        sales = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=20),
                "Sales": [100 + i * 10 for i in range(20)],
            }
        )
        sales.to_excel(writer, sheet_name="Sales", index=False)

        # Sheet 2: Multi-table inventory
        inventory1 = pd.DataFrame({"Item": ["A", "B", "C"], "Stock": [50, 30, 70]})
        empty = pd.DataFrame([[None, None]] * 2)
        inventory2 = pd.DataFrame({"Item": ["X", "Y", "Z"], "Stock": [20, 40, 60]})
        combined = pd.concat([inventory1, empty, inventory2], ignore_index=True)
        combined.to_excel(writer, sheet_name="Inventory", index=False)

    # Analyze each sheet
    for sheet_idx, sheet_name in enumerate(["Sales", "Inventory"]):
        print(f"\nAnalyzing sheet: {sheet_name}")

        # For this example, let's use multi-table workflow on the Inventory sheet
        # which we know has multiple tables
        if sheet_name == "Inventory":
            print("  → Using multi-table workflow")
            result = await run_multi_table_analysis(
                test_file,
                sheet_index=sheet_idx,
                config=AnalysisConfig(
                    excel_path=test_file,
                    sheet_index=sheet_idx,
                    output_dir=Path(f"outputs/integration/{sheet_name}"),
                    model="gpt-4o-mini",
                ),
            )
        else:
            print("  → Using standard analysis (single table)")
            # Here you would use the standard analysis pipeline
            # For this example, we'll just note it
            result = None

        if result and result.is_ok():
            data = result.unwrap()
            print(f"  ✓ Analysis complete: {data['tables_found']} tables found")


async def main() -> None:
    """Run all examples."""
    print("Multi-Table Detection Workflow Examples")
    print("=" * 50)

    # Run examples
    await example_1_basic_workflow()
    await example_2_custom_config()
    await example_3_direct_agent_usage()
    await example_4_workflow_customization()
    await example_5_integration_with_existing()

    print("\n" + "=" * 50)
    print("All examples complete!")
    print("\nCheck the outputs directories for generated notebooks.")


if __name__ == "__main__":
    # Set up logging
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Run examples
    asyncio.run(main())
