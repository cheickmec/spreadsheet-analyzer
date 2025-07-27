#!/usr/bin/env python3
"""Test executing individual cells to debug the issue."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService


async def test_individual_cells():
    """Test executing cells individually."""
    kernel = KernelService(profile=KernelProfile())

    session_id = await kernel.create_session("test")
    try:
        # Cell 1: Imports
        print("Executing Cell 1: Imports")
        cell1_code = """# Data profiling imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")
print("Imports completed!")"""

        result1 = await kernel.execute(session_id, cell1_code)
        print(f"Cell 1 result: status={result1.status}, outputs={len(result1.outputs)}")
        for output in result1.outputs:
            print(f"  Output: {output}")

        # Cell 2: Load data
        print("\nExecuting Cell 2: Load data")
        cell2_code = """# Load Excel data
file_path = r"test_assets/collection/business-accounting/Business Accounting.xlsx"
sheet_name = "Yiriden Transactions 2025"

try:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns from {sheet_name}")
    print(f"📊 Data shape: {df.shape}")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    df = pd.DataFrame()  # Empty fallback"""

        result2 = await kernel.execute(session_id, cell2_code)
        print(f"Cell 2 result: status={result2.status}, outputs={len(result2.outputs)}")
        for output in result2.outputs:
            print(f"  Output: {output}")

        # Cell 3: Simple test
        print("\nExecuting Cell 3: Simple test")
        cell3_code = """print("Simple test")
print(f"DataFrame exists: {'df' in locals()}")
if 'df' in locals():
    print(f"DataFrame shape: {df.shape}")"""

        result3 = await kernel.execute(session_id, cell3_code)
        print(f"Cell 3 result: status={result3.status}, outputs={len(result3.outputs)}")
        for output in result3.outputs:
            print(f"  Output: {output}")

    finally:
        await kernel.close_session(session_id)


if __name__ == "__main__":
    asyncio.run(test_individual_cells())
