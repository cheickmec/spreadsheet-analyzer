"""
Learn Jupyter Tools - Using core_exec modules

Using the spreadsheet_analyzer core_exec modules to demonstrate
controlled cell-by-cell execution.
"""

import asyncio

from spreadsheet_analyzer.core_exec import KernelProfile, KernelService, NotebookBuilder, NotebookIO, QualityInspector

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


async def main() -> None:
    """Main execution using core_exec modules."""

    # 1. Create notebook with NotebookBuilder
    print("1. Creating notebook with NotebookBuilder...")
    builder = NotebookBuilder(kernel_name="python3")
    builder.add_code_cell(cell_0)
    builder.add_code_cell(cell_1)
    builder.add_code_cell(cell_2)
    print(f"   Created notebook with {builder.cell_count()} cells")

    # 2. Set up kernel service for individual execution
    print("2. Setting up KernelService...")
    profile = KernelProfile(name="python3", max_execution_time=30.0, max_memory_mb=1024)

    async with KernelService(profile) as kernel_service:
        session_id = "excel_analysis_session"
        await kernel_service.create_session(session_id)
        print(f"   Created kernel session: {session_id}")

        # 3. Execute cells individually
        print("3. Executing cells individually...")

        print("   Executing cell 0 (data loading)...")
        result_0 = await kernel_service.execute(session_id, cell_0)
        print(f"      Status: {result_0.status}, Outputs: {len(result_0.outputs)}")

        print("   Executing cell 1 (describe)...")
        result_1 = await kernel_service.execute(session_id, cell_1)
        print(f"      Status: {result_1.status}, Outputs: {len(result_1.outputs)}")

        print("   Executing cell 2 (info)...")
        result_2 = await kernel_service.execute(session_id, cell_2)
        print(f"      Status: {result_2.status}, Outputs: {len(result_2.outputs)}")

        # 4. Update notebook with execution results
        print("4. Updating notebook with results...")
        notebook = builder.to_notebook()

        # Update cells with outputs
        if result_0.status == "ok":
            notebook.cells[0].outputs = NotebookIO.convert_outputs_to_nbformat(result_0.outputs)
            notebook.cells[0].execution_count = result_0.execution_count

        if result_1.status == "ok":
            notebook.cells[1].outputs = NotebookIO.convert_outputs_to_nbformat(result_1.outputs)
            notebook.cells[1].execution_count = result_1.execution_count

        if result_2.status == "ok":
            notebook.cells[2].outputs = NotebookIO.convert_outputs_to_nbformat(result_2.outputs)
            notebook.cells[2].execution_count = result_2.execution_count

    # 5. Inspect quality
    print("5. Inspecting notebook quality...")
    inspector = QualityInspector()
    metrics = inspector.inspect(builder)
    print(f"   Quality score: {metrics.overall_score:.1f}/100 ({metrics.overall_level.value})")
    print(f"   Issues found: {len(metrics.issues)}")

    # 6. Save notebook with NotebookIO
    print("6. Saving notebook with NotebookIO...")
    output_path = NotebookIO.write_notebook(builder, "examples/excel_analysis_core_exec.ipynb", overwrite=True)
    print(f"   Saved to: {output_path}")
    print("✅ Done!")


if __name__ == "__main__":
    asyncio.run(main())
