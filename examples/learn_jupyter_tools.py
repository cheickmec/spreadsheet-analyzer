"""
Learn Jupyter Tools - Direct Usage

Using Jupyter ecosystem tools directly without wrappers to understand
the core functionality step by step.
"""

from pathlib import Path

import nbclient
import nbformat as nbf

cell_0 = """
import os
from pathlib import Path
import pandas as pd

print(f"Current working directory: {Path.cwd()}")
df = pd.read_excel("test_assets/generated/employee_records.xlsx", sheet_name=0)
print(f"✅ Successfully loaded the first sheet")
# print(df.head())"""

cell_1 = """
df.describe()"""

cell_2 = """
df.info()"""

# let's add the cells to a notebook
print("1. Creating notebook with nbformat...")
nb = nbf.v4.new_notebook()

# Add all cells to notebook
code_cell_0 = nbf.v4.new_code_cell(cell_0)
code_cell_1 = nbf.v4.new_code_cell(cell_1)
code_cell_2 = nbf.v4.new_code_cell(cell_2)

nb.cells.extend([code_cell_0, code_cell_1, code_cell_2])
print(f"   Created notebook with {len(nb.cells)} cells")

# Create client and use context manager for proper kernel handling
print("2. Creating nbclient and executing cells individually...")
client = nbclient.NotebookClient(nb=nb, kernel_name="python3")

# Use the client's setup method to handle kernel lifecycle properly
with client.setup_kernel():
    print("   Kernel started successfully!")

    print("   Executing cell 0 (data loading)...")
    client.execute_cell(nb.cells[0], 0)
    print(f"      Cell 0 outputs: {len(nb.cells[0].outputs)}")

    print("   Executing cell 1 (describe)...")
    client.execute_cell(nb.cells[1], 1)
    print(f"      Cell 1 outputs: {len(nb.cells[1].outputs)}")

    print("   Executing cell 2 (info)...")
    client.execute_cell(nb.cells[2], 2)
    print(f"      Cell 2 outputs: {len(nb.cells[2].outputs)}")

# Check outputs from each cell
print("3. Checking individual cell outputs...")
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        print(f"   Cell {i}: {len(cell.outputs)} outputs")
        for j, output in enumerate(cell.outputs):
            print(f"      Output {j}: {output.output_type}")

# save the notebook to a file
print("4. Saving notebook with nbformat...")
with Path("examples/excel_analysis_direct.ipynb").open("w") as f:
    nbf.write(nb, f)

print("   Saved to: examples/excel_analysis_direct.ipynb")
print("✅ Done!")
