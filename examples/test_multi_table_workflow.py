#!/usr/bin/env python3
"""Test script for multi-table detection workflow.

This demonstrates how the table detection and analysis agents work together.
"""

import asyncio
from pathlib import Path

from spreadsheet_analyzer.workflows import run_multi_table_analysis


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

        # Table 1: Product inventory
        products = pd.DataFrame(
            {
                "Product ID": [f"PROD-{i:03d}" for i in range(1, 31)],
                "Product Name": [f"Product {i}" for i in range(1, 31)],
                "Stock": np.random.randint(10, 100, 30),
                "Price": np.round(np.random.uniform(10, 200, 30), 2),
            }
        )

        # Add some empty rows
        empty_rows = pd.DataFrame([[None] * 4] * 3)

        # Table 2: Sales summary
        summary = pd.DataFrame(
            {
                "Region": ["North", "South", "East", "West"],
                "Total Sales": [125000, 98000, 145000, 112000],
                "Units Sold": [1250, 980, 1450, 1120],
            }
        )

        # Combine with empty rows between
        combined = pd.concat([products, empty_rows, summary], ignore_index=True)

        # Save to Excel
        test_file.parent.mkdir(exist_ok=True)
        combined.to_excel(test_file, index=False)
        print(f"Created test file: {test_file}")

    print("\nRunning multi-table analysis workflow...")
    print("=" * 60)

    result = await run_multi_table_analysis(test_file, sheet_index=0)

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
    else:
        print(f"\n❌ Analysis failed: {result.unwrap_err()}")

    print("\n" + "=" * 60)
    print("Workflow test complete!")


if __name__ == "__main__":
    asyncio.run(test_workflow())
