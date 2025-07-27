#!/usr/bin/env python3
"""Debug notebook execution to find why outputs are missing."""

import json
from pathlib import Path

# Load the executed notebook and check outputs
notebook_path = Path("analysis_results/Business Accounting/Yiriden_Transactions_2025_executed.ipynb")

with open(notebook_path) as f:
    notebook = json.load(f)

print("Notebook cell analysis:")
print("=" * 60)

for i, cell in enumerate(notebook["cells"]):
    cell_type = cell.get("cell_type")
    if cell_type == "code":
        source = "".join(cell.get("source", []))
        outputs = cell.get("outputs", [])
        exec_count = cell.get("execution_count")

        print(f"\nCell {i} (execution_count={exec_count}):")
        print(f"Source preview: {source[:100]}...")
        print(f"Number of outputs: {len(outputs)}")

        if outputs:
            for j, output in enumerate(outputs):
                output_type = output.get("output_type", output.get("type", "unknown"))
                print(f"  Output {j}: type={output_type}")

                if output_type == "stream":
                    text = output.get("text", [])
                    if isinstance(text, list):
                        text_str = "".join(text)
                    else:
                        text_str = str(text)
                    print(f"    Text: {text_str[:100]}...")
                elif output_type == "error":
                    print(f"    Error: {output.get('ename', 'Unknown')}: {output.get('evalue', 'Unknown error')}")
        else:
            print("  NO OUTPUTS!")
            # Check if this cell should have outputs
            if "print" in source:
                print("  ⚠️  This cell contains print statements but has no outputs!")
