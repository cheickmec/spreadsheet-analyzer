#!/usr/bin/env python
"""Check the actual sheet names in Business Accounting.xlsx."""

from pathlib import Path

from spreadsheet_analyzer.plugins.spreadsheet.io.excel_io import list_sheets

excel_path = Path("test_assets/collection/business-accounting/Business Accounting.xlsx")

if not excel_path.exists():
    print(f"Error: File not found at {excel_path}")
else:
    sheets = list_sheets(excel_path)
    print(f"Sheets in {excel_path.name}:")
    for i, sheet in enumerate(sheets, 1):
        print(f"  {i}. '{sheet}'")
