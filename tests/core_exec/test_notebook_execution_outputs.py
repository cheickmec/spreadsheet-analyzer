#!/usr/bin/env python3
"""Tests for notebook execution with actual output generation."""

from pathlib import Path

import pytest

from src.spreadsheet_analyzer.core_exec.bridge import ExecutionBridge
from src.spreadsheet_analyzer.core_exec.kernel_service import KernelProfile, KernelService
from src.spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder
from src.spreadsheet_analyzer.core_exec.notebook_io import NotebookIO


class TestNotebookExecutionOutputs:
    """Test notebook execution with actual kernel outputs."""

    @pytest.fixture
    def output_dir(self, tmp_path: Path) -> Path:
        """Create output directory for test notebooks."""
        output_dir = tmp_path / "notebook_outputs"
        output_dir.mkdir(exist_ok=True)
        return output_dir

    @pytest.fixture
    def test_outputs_dir(self) -> Path:
        """Directory for persistent test outputs."""
        output_dir = Path("tests/test_outputs/notebooks")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @pytest.mark.asyncio
    async def test_basic_notebook_execution(self, output_dir: Path, test_outputs_dir: Path) -> None:
        """Test basic notebook execution with simple outputs."""
        profile = KernelProfile()
        kernel_service = KernelService(profile)
        bridge = ExecutionBridge(kernel_service)

        async with kernel_service:
            session_id = await kernel_service.create_session("test-basic")

            # Create notebook
            notebook = NotebookBuilder()
            notebook.add_markdown_cell("# Basic Test Notebook")
            notebook.add_code_cell("print('Hello, World!')")
            notebook.add_code_cell("2 + 2")
            notebook.add_code_cell("import pandas as pd\ndf = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})\ndf")

            # Execute notebook
            executed = await bridge.execute_notebook(session_id, notebook)

            # Save to both locations
            temp_path = output_dir / "basic_execution.ipynb"
            NotebookIO.write_notebook(executed, temp_path)

            persistent_path = test_outputs_dir / "basic_execution.ipynb"
            NotebookIO.write_notebook(executed, persistent_path, overwrite=True)

            # Verify outputs
            assert temp_path.exists()
            assert persistent_path.exists()

            # Check outputs are present
            loaded = NotebookIO.read_notebook(temp_path)
            code_cells = [cell for cell in loaded.cells if cell.cell_type.value == "code"]

            # First code cell should have print output
            assert len(code_cells[0].outputs) > 0
            assert any("Hello, World!" in str(output) for output in code_cells[0].outputs)

            # Second code cell should have execute result
            assert len(code_cells[1].outputs) > 0
            assert any("4" in str(output) for output in code_cells[1].outputs)

            # Third code cell should have dataframe output
            assert len(code_cells[2].outputs) > 0

    @pytest.mark.asyncio
    async def test_visualization_notebook_execution(self, output_dir: Path, test_outputs_dir: Path) -> None:
        """Test notebook execution with matplotlib visualizations."""
        profile = KernelProfile()
        kernel_service = KernelService(profile)
        bridge = ExecutionBridge(kernel_service)

        async with kernel_service:
            session_id = await kernel_service.create_session("test-viz")

            # Create notebook with visualizations
            notebook = NotebookBuilder()
            notebook.add_markdown_cell("# Visualization Test Notebook")
            notebook.add_code_cell("import matplotlib.pyplot as plt\nimport numpy as np")
            notebook.add_code_cell("""
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(8, 6))
plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
""")

            # Execute notebook
            executed = await bridge.execute_notebook(session_id, notebook)

            # Save outputs
            temp_path = output_dir / "visualization_execution.ipynb"
            NotebookIO.write_notebook(executed, temp_path)

            persistent_path = test_outputs_dir / "visualization_execution.ipynb"
            NotebookIO.write_notebook(executed, persistent_path, overwrite=True)

            # Verify plot output
            loaded = NotebookIO.read_notebook(temp_path)
            code_cells = [cell for cell in loaded.cells if cell.cell_type.value == "code"]

            # Second cell should have display_data output with image
            plot_cell = code_cells[1]
            assert len(plot_cell.outputs) > 0
            assert any(
                output.get("output_type") == "display_data" and "image/png" in output.get("data", {})
                for output in plot_cell.outputs
            )

    @pytest.mark.asyncio
    async def test_error_handling_notebook_execution(self, output_dir: Path, test_outputs_dir: Path) -> None:
        """Test notebook execution with errors and mixed outputs."""
        profile = KernelProfile()
        kernel_service = KernelService(profile)
        bridge = ExecutionBridge(kernel_service)

        async with kernel_service:
            session_id = await kernel_service.create_session("test-errors")

            # Create notebook with errors
            notebook = NotebookBuilder()
            notebook.add_markdown_cell("# Error Handling Test Notebook")
            notebook.add_code_cell("print('Before error')")
            notebook.add_code_cell("1/0  # This will cause an error")
            notebook.add_code_cell("print('After error - this still runs')")
            notebook.add_code_cell("""
# Multiple output types
from IPython.display import HTML, Markdown
display(HTML('<h3 style="color: red;">Error Test</h3>'))
display(Markdown('**This** is _markdown_'))
""")

            # Execute notebook
            executed = await bridge.execute_notebook(session_id, notebook)

            # Save outputs
            temp_path = output_dir / "error_handling_execution.ipynb"
            NotebookIO.write_notebook(executed, temp_path)

            persistent_path = test_outputs_dir / "error_handling_execution.ipynb"
            NotebookIO.write_notebook(executed, persistent_path, overwrite=True)

            # Verify error handling
            loaded = NotebookIO.read_notebook(temp_path)
            code_cells = [cell for cell in loaded.cells if cell.cell_type.value == "code"]

            # First cell should have print output
            assert len(code_cells[0].outputs) > 0
            assert any("Before error" in str(output) for output in code_cells[0].outputs)

            # Second cell should have error output
            error_cell = code_cells[1]
            assert len(error_cell.outputs) > 0
            assert any(
                output.get("output_type") == "error" and output.get("ename") == "ZeroDivisionError"
                for output in error_cell.outputs
            )

            # Third cell should still execute
            assert any("After error" in str(output) for output in code_cells[2].outputs)

            # Fourth cell should have multiple display outputs
            display_cell = code_cells[3]
            assert len(display_cell.outputs) >= 2

    @pytest.mark.asyncio
    async def test_data_analysis_notebook_execution(self, output_dir: Path, test_outputs_dir: Path) -> None:
        """Test realistic data analysis notebook execution."""
        profile = KernelProfile()
        kernel_service = KernelService(profile)
        bridge = ExecutionBridge(kernel_service)

        async with kernel_service:
            session_id = await kernel_service.create_session("test-analysis")

            # Create data analysis notebook
            notebook = NotebookBuilder()
            notebook.add_markdown_cell("# Data Analysis Workflow")
            notebook.add_code_cell("""
import pandas as pd
import numpy as np
np.random.seed(42)

# Create sample data
data = {
    'date': pd.date_range('2024-01-01', periods=30),
    'sales': np.random.randint(100, 1000, 30),
    'profit': np.random.randint(10, 200, 30)
}
df = pd.DataFrame(data)
df.head()
""")
            notebook.add_code_cell("df.describe()")
            notebook.add_code_cell("""
# Correlation analysis
correlation = df[['sales', 'profit']].corr()
print("Correlation Matrix:")
correlation
""")
            notebook.add_code_cell("""
# Visualize trends
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(df['date'], df['sales'], marker='o', color='blue', label='Sales')
ax1.set_title('Daily Sales Trend')
ax1.set_ylabel('Sales ($)')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.scatter(df['sales'], df['profit'], alpha=0.6)
ax2.set_title('Sales vs Profit Correlation')
ax2.set_xlabel('Sales ($)')
ax2.set_ylabel('Profit ($)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""")

            # Execute notebook
            executed = await bridge.execute_notebook(session_id, notebook)

            # Save outputs
            temp_path = output_dir / "data_analysis_execution.ipynb"
            NotebookIO.write_notebook(executed, temp_path)

            persistent_path = test_outputs_dir / "data_analysis_execution.ipynb"
            NotebookIO.write_notebook(executed, persistent_path, overwrite=True)

            # Verify all cells executed with outputs
            loaded = NotebookIO.read_notebook(temp_path)
            code_cells = [cell for cell in loaded.cells if cell.cell_type.value == "code"]

            # Check that cells with expected outputs have them
            # First cell creates dataframe and calls head() - should have output
            assert len(code_cells[0].outputs) > 0, "First cell (df.head()) has no outputs"

            # Second cell calls describe() - should have output
            assert len(code_cells[1].outputs) > 0, "Second cell (df.describe()) has no outputs"

            # Third cell prints and returns correlation - should have outputs
            assert len(code_cells[2].outputs) > 0, "Third cell (correlation) has no outputs"

            # Fourth cell creates plots - should have display output
            assert len(code_cells[3].outputs) > 0, "Fourth cell (plots) has no outputs"
            assert any(output.get("output_type") == "display_data" for output in code_cells[3].outputs)


@pytest.mark.asyncio
async def test_generate_example_notebooks_for_docs():
    """Generate example notebooks for documentation purposes."""
    output_dir = Path("tests/test_outputs/example_notebooks")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = KernelProfile()
    kernel_service = KernelService(profile)
    bridge = ExecutionBridge(kernel_service)

    async with kernel_service:
        session_id = await kernel_service.create_session("doc-examples")

        # Example 1: Basic Python notebook
        notebook1 = NotebookBuilder()
        notebook1.add_markdown_cell("# Python Basics Example")
        notebook1.add_code_cell(
            "# Variables and data types\nname = 'Alice'\nage = 30\nprint(f'{name} is {age} years old')"
        )
        notebook1.add_code_cell(
            "# Lists and loops\nnumbers = [1, 2, 3, 4, 5]\nsquares = [n**2 for n in numbers]\nprint(f'Squares: {squares}')"
        )

        executed1 = await bridge.execute_notebook(session_id, notebook1)
        NotebookIO.write_notebook(executed1, output_dir / "python_basics.ipynb", overwrite=True)

        # Example 2: Data Science notebook
        notebook2 = NotebookBuilder()
        notebook2.add_markdown_cell("# Data Science Example")
        notebook2.add_code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt")
        notebook2.add_code_cell("""
# Generate sample data
np.random.seed(123)
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})
df.head()
""")
        notebook2.add_code_cell("""
# Visualize data
plt.figure(figsize=(10, 6))
for cat in df['category'].unique():
    mask = df['category'] == cat
    plt.scatter(df.loc[mask, 'x'], df.loc[mask, 'y'], label=cat, alpha=0.6)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot by Category')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
""")

        executed2 = await bridge.execute_notebook(session_id, notebook2)
        NotebookIO.write_notebook(executed2, output_dir / "data_science_example.ipynb", overwrite=True)

        print(f"Example notebooks saved to: {output_dir}")
