#!/usr/bin/env python3
"""Test the spreadsheet analyzer with all fixes applied."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.main import main

# Run with test file
sys.argv = [
    "spreadsheet_analyzer",
    "test_assets/collection/business-accounting/Business Accounting.xlsx",
    "--sheet",
    "Yiriden Transactions 2025",
    "--output",
    "test_final_output.ipynb",
    "--execute",
]

main()
