#!/usr/bin/env python3
"""Check cell outputs in the executed notebook."""

import json

with open("analysis_results/Business Accounting/Yiriden_Transactions_2025_executed.ipynb") as f:
    notebook = json.load(f)

print(f"Notebook has {len(notebook['cells'])} cells\n")

for i, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        cell_id = cell.get("id", "no-id")
        outputs = cell.get("outputs", [])
        source = cell.get("source", [])
        first_line = source[0][:50] if source else "empty"

        print(f"Cell {i} (id: {cell_id}):")
        print(f"  First line: {first_line}...")
        print(f"  Outputs: {len(outputs)}")

        if outputs:
            for j, output in enumerate(outputs[:2]):  # Show first 2 outputs
                output_type = output.get("output_type", "unknown")
                if output_type == "stream":
                    text = output.get("text", [])
                    if isinstance(text, list):
                        text = "".join(text)
                    print(f"    Output {j}: {output_type} - {text[:80].strip()}...")
                elif output_type == "execute_result" or output_type == "display_data":
                    print(f"    Output {j}: {output_type} - has data")
        print()
