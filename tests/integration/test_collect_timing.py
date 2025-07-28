#!/usr/bin/env python3
"""Test timing of message collection."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService


async def test_collect_timing():
    """Test if messages are arriving before collection starts."""
    kernel = KernelService(profile=KernelProfile())

    session_id = await kernel.create_session("test")
    try:
        # Test 1: Check for messages immediately after execute
        print("Test 1: Immediate message check")
        client = kernel._clients[session_id]

        msg_id = client.execute('print("Test 1")')

        # Try to get messages immediately
        try:
            msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=0.01)
            print(f"Got immediate message: {msg.get('msg_type')}")
        except TimeoutError:
            print("No immediate message")

        # Now execute with kernel service
        print("\nTest 2: Using kernel.execute()")
        result = await kernel.execute(session_id, 'print("Test 2")')
        print(f"Result: status={result.status}, outputs={len(result.outputs)}")
        for output in result.outputs:
            print(f"  Output: {output}")

        # Test 3: Multiple rapid executions
        print("\nTest 3: Multiple executions")
        for i in range(3):
            result = await kernel.execute(session_id, f'print("Test 3.{i}")')
            print(f"Execution {i}: status={result.status}, outputs={len(result.outputs)}")

    finally:
        await kernel.close_session(session_id)


if __name__ == "__main__":
    asyncio.run(test_collect_timing())
