#!/usr/bin/env python3
"""Test to see what messages the kernel is actually sending."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService


async def test_kernel_messages():
    """Test what messages we're getting from the kernel."""
    kernel = KernelService(profile=KernelProfile())

    session_id = await kernel.create_session("test")
    try:
        # Simple code that should produce output
        code = """print("Hello from kernel")"""

        # Get the client directly
        client = kernel._clients[session_id]

        # Execute code
        print(f"Executing: {code}")
        msg_id = client.execute(code)
        print(f"Message ID: {msg_id}")

        # Collect ALL messages for debugging
        all_messages = []
        deadline = asyncio.get_event_loop().time() + 5.0

        while asyncio.get_event_loop().time() < deadline:
            try:
                msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=0.5)
                all_messages.append(msg)
                print("\nMessage received:")
                print(f"  Type: {msg.get('msg_type')}")
                print(f"  Parent msg_id: {msg.get('parent_header', {}).get('msg_id')}")
                print(f"  Content: {msg.get('content')}")

                # Check if it's for our execution
                if msg.get("parent_header", {}).get("msg_id") == msg_id:
                    print("  ✓ This is for our execution!")
                    if msg["msg_type"] == "stream":
                        print(f"  Stream output: {msg['content'].get('text')}")
                    elif msg["msg_type"] == "status" and msg["content"].get("execution_state") == "idle":
                        print("  Execution completed (idle)")

            except TimeoutError:
                print(".", end="", flush=True)

        print(f"\n\nTotal messages received: {len(all_messages)}")

    finally:
        await kernel.close_session(session_id)


if __name__ == "__main__":
    asyncio.run(test_kernel_messages())
