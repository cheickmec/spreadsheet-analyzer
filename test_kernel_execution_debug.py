#!/usr/bin/env python3
"""Debug kernel execution to understand output capture."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService


async def test_execution():
    """Test direct kernel execution with the problematic code."""
    kernel = KernelService(profile=KernelProfile())
    session_id = await kernel.create_session("test")

    try:
        # Test 1: Simple print
        print("\n=== Test 1: Simple print ===")
        result1 = await kernel.execute(session_id, "print('Hello World')")
        print(f"Status: {result1.status}")
        print(f"Outputs: {result1.outputs}")

        # Test 2: Import pandas and print
        print("\n=== Test 2: Import and print ===")
        result2 = await kernel.execute(
            session_id,
            """
import pandas as pd
print('Pandas imported successfully')
""",
        )
        print(f"Status: {result2.status}")
        print(f"Outputs: {result2.outputs}")

        # Test 3: The exact problematic code
        print("\n=== Test 3: Exact problematic code ===")
        code = """# Load Excel data
file_path = r"test_assets/collection/business-accounting/Business Accounting.xlsx"
sheet_name = "Yiriden Transactions 2025"

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns from Yiriden Transactions 2025")
    print(f"📊 Data shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    df = pd.DataFrame()  # Empty fallback
"""
        result3 = await kernel.execute(session_id, code)
        print(f"Status: {result3.status}")
        print(f"Outputs: {result3.outputs}")
        print(f"Error: {result3.error}")

    finally:
        await kernel.close_session(session_id)
        await kernel.cleanup()


if __name__ == "__main__":
    asyncio.run(test_execution())
