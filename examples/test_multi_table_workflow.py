#!/usr/bin/env python3
"""Test script for multi-table detection workflow.

This demonstrates how the table detection and analysis agents work together.
"""

import asyncio
from pathlib import Path

from spreadsheet_analyzer.cli.notebook_analysis import AnalysisConfig
from spreadsheet_analyzer.workflows.multi_table_workflow import run_multi_table_analysis


async def test_workflow() -> None:
    """Test the multi-table workflow with a sample file."""
    # Use a test Excel file
    test_file = Path("test_assets/multi_table_sample.xlsx")

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        print("Creating a sample multi-table Excel file...")

        # Create sample data
        import numpy as np
        import pandas as pd

        # Table 1: Product inventory (30 rows)
        products = pd.DataFrame(
            {
                "Product ID": [f"PROD-{i:03d}" for i in range(1, 31)],
                "Product Name": [f"Product {i}" for i in range(1, 31)],
                "Stock": np.random.randint(10, 100, 30),
                "Price": np.round(np.random.uniform(10, 200, 30), 2),
            }
        )

        # Add some empty rows (3 rows)
        empty_rows1 = pd.DataFrame([[None] * 4] * 3)

        # Table 2: Sales by region (5 rows)
        sales_by_region = pd.DataFrame(
            {
                "Region": ["North", "South", "East", "West", "Central"],
                "Q1 Sales": [125000, 98000, 145000, 112000, 87000],
                "Q2 Sales": [132000, 101000, 151000, 118000, 92000],
                "Total": [257000, 199000, 296000, 230000, 179000],
            }
        )

        # More empty rows (2 rows)
        empty_rows2 = pd.DataFrame([[None] * 4] * 2)

        # Table 3: Top customers (10 rows)
        customers = pd.DataFrame(
            {
                "Customer ID": [f"CUST-{i:04d}" for i in range(1, 11)],
                "Customer Name": [f"Customer {i}" for i in range(1, 11)],
                "Total Orders": np.random.randint(5, 50, 10),
                "Revenue": np.round(np.random.uniform(5000, 50000, 10), 2),
            }
        )

        # Combine all parts
        combined = pd.concat([products, empty_rows1, sales_by_region, empty_rows2, customers], ignore_index=True)

        # Save to Excel
        test_file.parent.mkdir(exist_ok=True)
        combined.to_excel(test_file, index=False)
        print(f"Created test file: {test_file}")
        print(f"Total rows: {len(combined)}")
        print(f"Empty rows at indices: {combined.isnull().all(axis=1).nonzero()[0].tolist()}")

    print("\nRunning multi-table analysis workflow...")
    print("=" * 60)

    # Create analysis config
    output_dir = Path("outputs/multi_table_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AnalysisConfig(
        excel_path=test_file,
        sheet_index=0,
        output_dir=output_dir,
        model="gpt-4o-mini",  # Or your preferred model
        max_rounds=3,
        auto_save_rounds=True,
        verbose=True,
    )

    # Run the workflow
    result = await run_multi_table_analysis(test_file, sheet_index=0, config=config)

    if result.is_ok():
        analysis = result.unwrap()
        print("\n✅ Analysis completed successfully!")
        print("\nResults:")
        print(f"  - Tables found: {analysis['tables_found']}")
        print(f"  - Detection notebook: {analysis['detection_notebook']}")
        print(f"  - Analysis notebook: {analysis['analysis_notebook']}")

        if analysis.get("detection_error"):
            print(f"  ⚠️  Detection warning: {analysis['detection_error']}")
        if analysis.get("analysis_error"):
            print(f"  ⚠️  Analysis warning: {analysis['analysis_error']}")

        # Show paths for inspection
        if analysis["detection_notebook"]:
            print("\nTo view detection details:")
            print(f"  jupyter notebook {analysis['detection_notebook']}")

        if analysis["analysis_notebook"]:
            print("\nTo view analysis results:")
            print(f"  jupyter notebook {analysis['analysis_notebook']}")
    else:
        print(f"\n❌ Analysis failed: {result.unwrap_err()}")

    print("\n" + "=" * 60)
    print("Workflow test complete!")
    print("\nCheck the outputs directory for generated notebooks.")


if __name__ == "__main__":
    # Set up logging for better debugging
    import logging

    logging.basicConfig(level=logging.INFO)

    asyncio.run(test_workflow())
