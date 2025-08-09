#!/usr/bin/env python3
"""Simple test of the table detector agent.

This shows how to use the table detector agent directly without the full workflow.
"""

from pathlib import Path

import pandas as pd

from spreadsheet_analyzer.agents.table_detector_agent import create_table_detector
from spreadsheet_analyzer.agents.types import AgentId, AgentMessage, AgentState


def test_table_detection():
    """Test table detection on a sample Excel file."""
    # Create sample data with multiple tables
    print("Creating sample data with multiple tables...")

    # Table 1: Orders (rows 0-48)
    orders = pd.DataFrame(
        {
            "Order ID": [f"ORD-{i:04d}" for i in range(1, 50)],
            "Customer": [f"Customer {i}" for i in range(1, 50)],
            "Amount": [100 + i * 10 for i in range(1, 50)],
            "Date": pd.date_range("2024-01-01", periods=49, freq="D"),
        }
    )

    # Empty rows separator
    empty_rows = pd.DataFrame([[None, None, None, None]] * 3)

    # Table 2: Regional Summary (rows 52-56)
    summary = pd.DataFrame(
        {
            "Region": ["North", "South", "East", "West"],
            "Total Sales": [50000, 45000, 60000, 55000],
            "Order Count": [100, 90, 120, 110],
            "Avg Order Value": [500, 500, 500, 500],
        }
    )

    # Combine all parts
    full_data = pd.concat([orders, empty_rows, summary], ignore_index=True)

    print(f"\nDataFrame shape: {full_data.shape}")
    print(f"Empty rows at indices: {full_data.isnull().all(axis=1).nonzero()[0].tolist()}")

    # Create the detector
    print("\nCreating table detector agent...")
    detector = create_table_detector()

    # Create message
    message = AgentMessage.create(
        sender=AgentId.generate("test"),
        receiver=detector.id,
        content={"dataframe": full_data, "sheet_name": "MultiTableSheet", "file_path": "test_multi_table.xlsx"},
    )

    # Run detection
    print("Running table detection...")
    state = AgentState(agent_id=detector.id, status="idle")
    result = detector.process(message, state)

    if result.is_ok():
        detection_result = result.unwrap().content
        print("\n✅ Detection successful!")
        print(f"Detection method: {detection_result.detection_method}")
        print(f"Number of tables found: {len(detection_result.tables)}")

        for i, table in enumerate(detection_result.tables):
            print(f"\nTable {i + 1}: {table.description}")
            print(f"  - Location: rows {table.start_row}-{table.end_row}, cols {table.start_col}-{table.end_col}")
            print(f"  - Type: {table.table_type.value}")
            print(f"  - Entity: {table.entity_type}")
            print(f"  - Size: {table.row_count} rows x {table.col_count} columns")
            print(f"  - Confidence: {table.confidence:.2%}")

            # Show sample data
            sample = full_data.iloc[
                table.start_row : min(table.start_row + 3, table.end_row + 1), table.start_col : table.end_col + 1
            ]
            print("  - Preview:")
            print(sample.to_string(index=True, max_cols=4).replace("\n", "\n    "))

        # Test detection metrics if available
        if detection_result.metrics:
            print("\nDetection Metrics:")
            print(f"  - Empty row blocks: {detection_result.metrics.empty_row_blocks}")
            print(f"  - Header rows: {detection_result.metrics.header_rows_detected}")
    else:
        print(f"\n❌ Detection failed: {result.unwrap_err()}")

    # Save the test data
    save_path = Path("test_multi_table.xlsx")
    full_data.to_excel(save_path, index=False)
    print(f"\nTest data saved to: {save_path}")


if __name__ == "__main__":
    test_table_detection()
