#!/usr/bin/env python3
"""Simple test of kernel execution and direct notebook saving."""

import asyncio
from pathlib import Path

from spreadsheet_analyzer.agents.kernel_manager import AgentKernelManager


async def test_direct_kernel_execution():
    """Test kernel execution and direct notebook saving."""

    print("Testing direct kernel execution...")

    async with AgentKernelManager() as manager:
        async with manager.acquire_kernel("simple-test-agent") as (km, session):
            print(f"Acquired kernel session: {session.session_id}")
            print(f"Agent ID: {session.agent_id}")
            print(f"Kernel ID: {session.kernel_id}")

            # Test code execution
            test_codes = [
                "print('Hello from Jupyter kernel!')",
                "x = 2 + 2\nprint(f'2 + 2 = {x}')",
                "import pandas as pd\nprint(f'Pandas version: {pd.__version__}')",
                "import numpy as np\ndata = np.array([1, 2, 3, 4, 5])\nprint(f'Data: {data}')\nprint(f'Mean: {data.mean()}')",
            ]

            for i, code in enumerate(test_codes, 1):
                print(f"\n--- Executing code block {i} ---")
                print(f"Code: {code}")

                result = await manager.execute_code(session, code)

                print(f"Result type: {type(result)}")
                print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")

                if isinstance(result, dict):
                    if result.get("stdout"):
                        print(f"STDOUT: {result['stdout']}")
                    if result.get("stderr"):
                        print(f"STDERR: {result['stderr']}")
                    if result.get("data"):
                        print(f"DATA: {result['data']}")
                    if result.get("outputs"):
                        print(f"OUTPUTS: {result['outputs']}")
                        for j, output in enumerate(result["outputs"]):
                            print(f"  Output {j}: {output}")
                            print(f"    Type: {type(output)}")
                            print(f"    Keys: {list(output.keys()) if isinstance(output, dict) else 'Not dict'}")

            # Check execution history
            print("\n--- Execution History ---")
            print(f"Total executions: {len(session.execution_history)}")
            for i, execution in enumerate(session.execution_history):
                print(f"Execution {i + 1}:")
                print(f"  Code: {execution['code'][:50]}...")
                print(
                    f"  Result keys: {list(execution['result'].keys()) if isinstance(execution['result'], dict) else 'Not a dict'}"
                )

            # Test direct notebook saving
            output_path = Path("simple_test_output.ipynb")
            try:
                manager.save_session_as_notebook(session, output_path, "Simple Test Results")
                print("\n--- Notebook Saved ---")
                print(f"Saved to: {output_path}")

                if output_path.exists():
                    size = output_path.stat().st_size
                    print(f"File size: {size} bytes")

                    # Read and show notebook structure
                    import json

                    with open(output_path) as f:
                        notebook_data = json.load(f)

                    print(f"Notebook cells count: {len(notebook_data.get('cells', []))}")

                    for i, cell in enumerate(notebook_data.get("cells", [])):
                        cell_type = cell.get("cell_type", "unknown")
                        source = cell.get("source", "")
                        outputs = cell.get("outputs", [])

                        if isinstance(source, list):
                            source = "".join(source)

                        print(f"Cell {i + 1}: {cell_type}")
                        print(f"  Source: {source[:50]}...")
                        print(f"  Outputs: {len(outputs)} outputs")

                        for j, output in enumerate(outputs):
                            output_type = output.get("output_type", "unknown")
                            print(f"    Output {j + 1}: {output_type}")
                            if "text" in output:
                                text = output["text"]
                                if isinstance(text, list):
                                    text = "".join(text)
                                print(f"      Text: {text[:100]}...")
                else:
                    print("ERROR: Notebook file was not created!")

            except Exception as e:
                print(f"ERROR saving notebook: {e}")
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_direct_kernel_execution())
