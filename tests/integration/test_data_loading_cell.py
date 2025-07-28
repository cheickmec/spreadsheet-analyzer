#!/usr/bin/env python3
"""Debug the data loading cell specifically."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.plugins.spreadsheet.tasks import DataProfilingTask


async def test_data_loading_cell():
    """Test the data loading cell in isolation."""
    # Get the data loading code
    task = DataProfilingTask()
    context = {
        "file_path": "test_assets/collection/business-accounting/Business Accounting.xlsx",
        "sheet_name": "Yiriden Transactions 2025",
    }

    load_code = task._generate_load_code(context["file_path"], context["sheet_name"])
    print("Data loading code:")
    print("=" * 60)
    print(load_code)
    print("=" * 60)

    # Test with different kernel configurations
    configs = [
        {"name": "Default", "profile": KernelProfile()},
        {"name": "Longer timeout", "profile": KernelProfile(max_execution_time=60.0)},
        {
            "name": "Longer drain timeout",
            "profile": KernelProfile(output_drain_timeout_ms=500, output_drain_max_timeout_ms=2000),
        },
    ]

    for config in configs:
        print(f"\n\nTesting with {config['name']} configuration:")
        print("-" * 60)

        kernel = KernelService(profile=config["profile"])
        session_id = await kernel.create_session(f"test_{config['name']}")

        try:
            # First execute imports
            import_code = """import pandas as pd
import numpy as np
print("Imports ready")"""

            print("Executing imports...")
            result1 = await kernel.execute(session_id, import_code)
            print(f"Import result: status={result1.status}, outputs={len(result1.outputs)}")

            # Wait a bit
            await asyncio.sleep(0.5)

            # Then execute data loading
            print("\nExecuting data loading code...")
            result2 = await kernel.execute(session_id, load_code)
            print(
                f"Load result: status={result2.status}, outputs={len(result2.outputs)}, duration={result2.duration_seconds:.3f}s"
            )

            if result2.outputs:
                for i, output in enumerate(result2.outputs):
                    if output.get("type") == "stream":
                        print(f"  Output {i}: {output.get('text', '')}")
                    elif output.get("type") == "error":
                        print(f"  Error {i}: {output.get('ename')}: {output.get('evalue')}")
            else:
                print("  No outputs captured!")

                # Try a simple print after
                print("\nTrying simple print after data load...")
                result3 = await kernel.execute(session_id, 'print("Test after load")')
                print(f"Test result: status={result3.status}, outputs={len(result3.outputs)}")
                if result3.outputs:
                    for output in result3.outputs:
                        if output.get("type") == "stream":
                            print(f"  Output: {output.get('text', '')}")

        finally:
            await kernel.close_session(session_id)

    await kernel.shutdown()


if __name__ == "__main__":
    asyncio.run(test_data_loading_cell())
