#!/usr/bin/env python3
"""Test direct execution of the problematic code."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec import KernelProfile, KernelService


async def test_direct_execution():
    """Test direct execution of the problematic code."""
    profile = KernelProfile(
        output_drain_timeout_ms=500,
        output_drain_max_timeout_ms=2000,
        output_drain_max_attempts=5,
        wait_for_shell_reply=True,
    )

    # The exact code from the notebook that has no output
    code = """# Load Excel data
file_path = r"test_assets/collection/business-accounting/Business Accounting.xlsx"
sheet_name = "Yiriden Transactions 2025"

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns from Yiriden Transactions 2025")
    print(f"📊 Data shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    df = pd.DataFrame()  # Empty fallback"""

    async with KernelService(profile) as service:
        session_id = await service.create_session("test")

        # First execute imports
        import_code = """import pandas as pd
import numpy as np"""

        print("Executing imports...")
        result1 = await service.execute(session_id, import_code)
        print(f"Import result: status={result1.status}, outputs={len(result1.outputs)}")

        # Then execute the problematic code
        print("\nExecuting data loading code...")
        result2 = await service.execute(session_id, code)
        print(f"Data loading result: status={result2.status}, outputs={len(result2.outputs)}")
        print(f"Outputs: {result2.outputs}")

        # Also test a simple print
        print("\nExecuting simple print...")
        result3 = await service.execute(session_id, "print('Test print')")
        print(f"Simple print result: status={result3.status}, outputs={len(result3.outputs)}")
        print(f"Outputs: {result3.outputs}")


if __name__ == "__main__":
    asyncio.run(test_direct_execution())
