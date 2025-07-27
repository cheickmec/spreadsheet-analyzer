#!/usr/bin/env python3
"""Test if the issue is specific to the first cell or import statements."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService


async def test_first_cell():
    """Test different scenarios for first cell execution."""
    kernel = KernelService(profile=KernelProfile())

    # Test different session scenarios
    scenarios = [
        {
            "name": "Imports only",
            "cells": [
                "import pandas as pd\nimport numpy as np",
                'print("After imports")',
            ],
        },
        {
            "name": "Print then imports",
            "cells": [
                'print("Before imports")',
                "import pandas as pd\nimport numpy as np",
                'print("After imports")',
            ],
        },
        {
            "name": "Imports with print",
            "cells": [
                'print("Starting imports...")\nimport pandas as pd\nimport numpy as np\nprint("Imports complete")',
                'print("Next cell")',
            ],
        },
        {
            "name": "Simple prints only",
            "cells": [
                'print("Cell 1")',
                'print("Cell 2")',
                'print("Cell 3")',
            ],
        },
    ]

    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"Testing scenario: {scenario['name']}")
        print(f"{'=' * 60}")

        session_id = await kernel.create_session(f"test_{scenario['name']}")
        try:
            for i, code in enumerate(scenario["cells"]):
                print(f"\nExecuting cell {i}: {code[:50]}...")
                result = await kernel.execute(session_id, code)
                print(
                    f"Result: status={result.status}, outputs={len(result.outputs)}, duration={result.duration_seconds:.3f}s"
                )

                if result.outputs:
                    for j, output in enumerate(result.outputs):
                        if output.get("type") == "stream":
                            print(f"  Output {j}: {output.get('text', '')}")
                        else:
                            print(f"  Output {j}: {output}")
                else:
                    print("  No outputs captured")

                # Add delay between cells
                await asyncio.sleep(0.1)

        finally:
            await kernel.close_session(session_id)

    await kernel.shutdown()


if __name__ == "__main__":
    asyncio.run(test_first_cell())
