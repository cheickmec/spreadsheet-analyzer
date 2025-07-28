#!/usr/bin/env python3
"""Debug kernel message flow."""

import asyncio
import logging

from jupyter_client import AsyncKernelManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def debug_kernel_messages():
    """Debug kernel message flow to understand the issue."""
    # Create kernel manager
    km = AsyncKernelManager(kernel_name="python3")
    await km.start_kernel()

    # Create client
    client = km.client()
    client.start_channels()

    # Wait for channels
    await asyncio.sleep(0.5)

    # Execute simple code
    code = "print('Hello')"
    logger.info(f"Executing: {code}")
    msg_id = client.execute(code)
    logger.info(f"msg_id: {msg_id}")

    # Try different approaches to get messages
    logger.info("Approach 1: Direct get_iopub_msg")
    try:
        for i in range(10):
            msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=0.5)
            logger.info(
                f"Message {i}: type={msg.get('msg_type')}, parent_id={msg.get('parent_header', {}).get('msg_id')}"
            )
            if msg.get("msg_type") == "status" and msg.get("content", {}).get("execution_state") == "idle":
                logger.info("Got idle status")
                break
    except TimeoutError:
        logger.warning("Timeout waiting for messages")

    # Try shell channel too
    logger.info("Approach 2: Check shell channel")
    try:
        shell_msg = await asyncio.wait_for(client.get_shell_msg(), timeout=0.5)
        logger.info(f"Shell message: {shell_msg}")
    except TimeoutError:
        logger.warning("No shell message")

    # Execute with different approach - wait first
    logger.info("\nApproach 3: Execute and immediate wait")
    await asyncio.sleep(0.5)  # Let previous execution settle

    code2 = "2 + 2"
    logger.info(f"Executing: {code2}")
    msg_id2 = client.execute(code2)

    # Try to consume all messages
    messages = []
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < 2.0:
        try:
            msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=0.1)
            messages.append(msg)
            logger.info(f"Got message: {msg.get('msg_type')}")
        except TimeoutError:
            if messages:
                break  # Got some messages, timeout means no more
            else:
                await asyncio.sleep(0.05)  # Wait a bit more if no messages yet

    logger.info(f"Total messages collected: {len(messages)}")
    for i, msg in enumerate(messages):
        logger.info(f"Message {i}: {msg.get('msg_type')} - parent: {msg.get('parent_header', {}).get('msg_id')}")

    # Cleanup
    client.stop_channels()
    await km.shutdown_kernel()


if __name__ == "__main__":
    asyncio.run(debug_kernel_messages())
