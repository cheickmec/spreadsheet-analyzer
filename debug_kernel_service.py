#!/usr/bin/env python3
"""Debug kernel service to understand the issue."""

import asyncio
import logging

from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def debug_kernel_service():
    """Debug kernel service execution."""
    profile = KernelProfile()
    service = KernelService(profile)

    async with service:
        session_id = await service.create_session("debug")

        # Get the client directly for debugging
        client = service._clients[session_id]

        # Test 1: Simple print
        logger.info("=== Test 1: Simple print ===")
        code = "print('Hello, World!')"

        # Clear any pending messages first
        logger.info("Clearing pending messages...")
        while True:
            try:
                msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=0.01)
                logger.info(f"Cleared message: {msg.get('msg_type')}")
            except TimeoutError:
                break

        # Execute
        logger.info(f"Executing: {code}")
        msg_id = client.execute(code)
        logger.info(f"Got msg_id: {msg_id}")

        # Collect ALL messages for 2 seconds
        logger.info("Collecting messages for 2 seconds...")
        messages = []
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < 2.0:
            try:
                msg = await asyncio.wait_for(client.get_iopub_msg(), timeout=0.1)
                msg_info = {
                    "msg_type": msg.get("msg_type"),
                    "parent_msg_id": msg.get("parent_header", {}).get("msg_id"),
                    "content": msg.get("content", {}),
                }
                messages.append(msg_info)
                logger.info(f"Got message: type={msg_info['msg_type']}, parent={msg_info['parent_msg_id']}")

                # Log details for stream messages
                if msg_info["msg_type"] == "stream":
                    logger.info(f"  Stream content: {msg_info['content']}")

            except TimeoutError:
                # Small wait between timeouts
                await asyncio.sleep(0.05)

        logger.info(f"\nTotal messages collected: {len(messages)}")
        logger.info(f"Messages with our msg_id: {sum(1 for m in messages if m['parent_msg_id'] == msg_id)}")

        # Now try using the service's execute method
        logger.info("\n=== Test 2: Using service.execute ===")
        result = await service.execute(session_id, "print('Via service')")
        logger.info(f"Result status: {result.status}")
        logger.info(f"Result outputs: {result.outputs}")
        logger.info(f"Result msg_id: {result.msg_id}")


if __name__ == "__main__":
    asyncio.run(debug_kernel_service())
