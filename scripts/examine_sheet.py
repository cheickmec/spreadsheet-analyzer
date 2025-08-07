#!/usr/bin/env python3
"""Examine sheet structure to identify table boundaries."""

import sys
from pathlib import Path

import pandas as pd


def examine_sheet(excel_path: str, sheet_index: int):
    """Print detailed sheet structure for analysis."""

    # Load the sheet
    df = pd.read_excel(excel_path, sheet_name=sheet_index)

    print(f"=== SHEET {sheet_index} ANALYSIS ===")
    print(f"Shape: {df.shape} (rows: {df.shape[0]}, cols: {df.shape[1]})")
    print(f"Columns: {list(df.columns)}")

    print("\n=== FULL SHEET CONTENT ===")
    # Print with row and column indices for precise boundary identification
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)

    print(df.to_string())

    print("\n=== EMPTY CELL ANALYSIS ===")
    # Check for empty rows
    empty_rows = df.isnull().all(axis=1)
    if empty_rows.any():
        empty_indices = empty_rows[empty_rows].index.tolist()
        print(f"Empty rows at indices: {empty_indices}")
    else:
        print("No completely empty rows")

    # Check for empty columns
    empty_cols = df.isnull().all(axis=0)
    if empty_cols.any():
        empty_col_names = empty_cols[empty_cols].index.tolist()
        print(f"Empty columns: {empty_col_names}")
    else:
        print("No completely empty columns")

    print("\n=== NON-NULL VALUE COUNTS BY COLUMN ===")
    for i, col in enumerate(df.columns):
        non_null = df[col].notna().sum()
        print(f"Col {i} ('{col}'): {non_null}/{len(df)} non-null values")

    print("\n=== DATA TYPE ANALYSIS ===")
    print(df.dtypes)

    print("\n=== POTENTIAL TABLE BOUNDARIES ===")
    print("Looking for patterns that indicate table separations...")

    # Check for mostly empty rows that might separate tables
    row_density = df.notna().sum(axis=1) / len(df.columns)
    sparse_rows = row_density[row_density < 0.2].index.tolist()
    if sparse_rows:
        print(f"Sparse rows (< 20% filled): {sparse_rows}")

    # Check for changes in data patterns
    print("\n=== FIRST 5 ROWS ===")
    print(df.head().to_string())

    print("\n=== LAST 5 ROWS ===")
    print(df.tail().to_string())


if __name__ == "__main__":
    # Default to Business Accounting sheet 0
    excel_path = "test_assets/collection/business-accounting/Business Accounting.xlsx"
    sheet_index = 0

    if len(sys.argv) > 1:
        excel_path = sys.argv[1]
    if len(sys.argv) > 2:
        sheet_index = int(sys.argv[2])

    # Check if file exists
    if not Path(excel_path).exists():
        print(f"Error: File not found: {excel_path}")
        # Try alternative paths
        alt_paths = [
            "test_assets/multi_table_sample.xlsx",
            "test_assets/generated/financial_model.xlsx",
        ]
        for alt in alt_paths:
            if Path(alt).exists():
                print(f"Using alternative file: {alt}")
                excel_path = alt
                break

    examine_sheet(excel_path, sheet_index)
