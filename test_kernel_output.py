#!/usr/bin/env python3
"""Test script to debug kernel output collection."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec import KernelProfile, KernelService

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_kernel_output():
    """Test kernel output collection."""
    profile = KernelProfile(
        output_drain_timeout_ms=150,
        output_drain_max_timeout_ms=1000,
        output_drain_max_attempts=3,
        wait_for_shell_reply=True,
    )

    async with KernelService(profile) as service:
        session_id = await service.create_session("test")

        # Test 1: Simple print statement
        print("\n=== Test 1: Simple print ===")
        result = await service.execute(session_id, 'print("Hello World")')
        print(f"Status: {result.status}")
        print(f"Outputs: {result.outputs}")
        print(f"Duration: {result.duration_seconds:.2f}s")

        # Test 2: Multiple prints
        print("\n=== Test 2: Multiple prints ===")
        code = """
print("Line 1")
print("Line 2")
print("Line 3")
"""
        result = await service.execute(session_id, code)
        print(f"Status: {result.status}")
        print(f"Outputs: {result.outputs}")

        # Test 3: Mixed output
        print("\n=== Test 3: Mixed output ===")
        code = """
print("Starting...")
x = 42
print(f"x = {x}")
x * 2
"""
        result = await service.execute(session_id, code)
        print(f"Status: {result.status}")
        print(f"Outputs: {result.outputs}")

        # Test 4: Import and print
        print("\n=== Test 4: Import and print ===")
        code = """
import pandas as pd
print("Pandas imported successfully")
print(f"Pandas version: {pd.__version__}")
"""
        result = await service.execute(session_id, code)
        print(f"Status: {result.status}")
        print(f"Outputs: {result.outputs}")

        # Test 5: The exact pattern from our notebook
        print("\n=== Test 5: Excel loading pattern ===")
        code = """
import pandas as pd
import numpy as np

# Simulate Excel load
data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
df = pd.DataFrame(data)
print(f"✅ Successfully loaded {len(df)} rows and {len(df.columns)} columns from Test Sheet")
print(f"📊 Data shape: {df.shape}")
"""
        result = await service.execute(session_id, code)
        print(f"Status: {result.status}")
        print(f"Outputs: {result.outputs}")

        # Test 6: Timing test - immediate print
        print("\n=== Test 6: Immediate print ===")
        code = """print("Immediate output")"""
        result = await service.execute(session_id, code)
        print(f"Status: {result.status}")
        print(f"Outputs: {result.outputs}")

        # Test 7: Delayed print
        print("\n=== Test 7: Delayed print ===")
        code = """
import time
time.sleep(0.1)
print("Delayed output")
"""
        result = await service.execute(session_id, code)
        print(f"Status: {result.status}")
        print(f"Outputs: {result.outputs}")


if __name__ == "__main__":
    asyncio.run(test_kernel_output())
