"""Demonstrate the AgentKernelManager features with notebook generation.

This script showcases the kernel manager's capabilities by:
- Creating and managing real Jupyter kernels
- Executing code cells with persistent state
- Creating markdown documentation cells
- Demonstrating session persistence and checkpointing
- Generating a viewable Jupyter notebook as output

The output notebook is deleted and recreated on each run.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from spreadsheet_analyzer.agents.kernel_manager import (
    AgentKernelManager,
    KernelResourceLimits,
)
from spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder


async def demonstrate_kernel_manager() -> None:
    """Main demonstration of kernel manager features."""

    # Setup output path
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)

    notebook_path = output_dir / "kernel_manager_demo.ipynb"

    # Delete existing notebook if it exists
    if notebook_path.exists():
        notebook_path.unlink()
        print(f"Deleted existing notebook: {notebook_path}")

    # Initialize notebook builder
    notebook = NotebookBuilder()

    # Add title and introduction
    notebook.add_markdown_cell(f"""# Kernel Manager Demo

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This notebook demonstrates the AgentKernelManager features including:
- Real Jupyter kernel execution
- Session persistence across multiple acquisitions
- Code execution with state preservation
- Error handling and timeout management
- Session checkpointing and restoration
""")

    # Configure kernel manager with custom limits
    resource_limits = KernelResourceLimits(
        max_cpu_percent=70.0, max_memory_mb=512, max_execution_time=10.0, max_output_size_mb=5
    )

    manager = AgentKernelManager(max_kernels=2, resource_limits=resource_limits)

    notebook.add_markdown_cell("""## Kernel Manager Configuration

The kernel manager is configured with:
- Maximum 2 concurrent kernels
- CPU limit: 70%
- Memory limit: 512 MB
- Execution timeout: 10 seconds
- Output size limit: 5 MB
""")

    async with manager:
        print("=== Starting Kernel Manager Demo ===")

        # Demo 1: Basic code execution
        notebook.add_markdown_cell("## Demo 1: Basic Code Execution")

        async with manager.acquire_kernel("demo-agent-1") as (km, session):
            print(f"Acquired kernel for agent: {session.agent_id}")
            print(f"Kernel ID: {session.kernel_id}")

            # Execute basic calculations
            code1 = "# Basic arithmetic\nresult = 2 + 2\nprint(f'2 + 2 = {result}')\nresult"
            outputs1 = await manager.execute_code_for_notebook(session, code1)
            notebook.add_code_cell_with_outputs(code1, outputs1)

            # Create variables for persistence demo
            code2 = "# Create some variables for persistence testing\nimport math\nimport random\n\npi_value = math.pi\nrandom_number = random.randint(1, 100)\nmy_list = [1, 2, 3, 4, 5]\n\nprint(f'π = {pi_value:.4f}')\nprint(f'Random number: {random_number}')\nprint(f'List: {my_list}')"
            outputs2 = await manager.execute_code_for_notebook(session, code2)
            notebook.add_code_cell_with_outputs(code2, outputs2)

        # Demo 2: Session persistence
        notebook.add_markdown_cell("## Demo 2: Session Persistence")
        notebook.add_markdown_cell("The same agent acquires the kernel again. Variables should persist:")

        async with manager.acquire_kernel("demo-agent-1") as (km, session):
            # Test persistence
            code3 = "# Test that variables persist across kernel acquisitions\nprint(f'π is still: {pi_value:.4f}')\nprint(f'Random number is still: {random_number}')\nprint(f'List is still: {my_list}')\n\n# Modify the list\nmy_list.append(6)\nprint(f'Modified list: {my_list}')"
            outputs3 = await manager.execute_code_for_notebook(session, code3)
            notebook.add_code_cell_with_outputs(code3, outputs3)

        # Demo 3: Multiple agents
        notebook.add_markdown_cell("## Demo 3: Multiple Agents with Separate Sessions")
        notebook.add_markdown_cell("Different agents get isolated kernel sessions:")

        async with manager.acquire_kernel("demo-agent-2") as (km, session):
            code4 = "# This is a different agent - variables should not exist\ntry:\n    print(f'Trying to access pi_value: {pi_value}')\nexcept NameError as e:\n    print(f'Expected error: {e}')\n\n# Create different variables\nagent2_data = {'name': 'Agent 2', 'value': 42}\nprint(f'Agent 2 data: {agent2_data}')"
            outputs4 = await manager.execute_code_for_notebook(session, code4)
            notebook.add_code_cell_with_outputs(code4, outputs4)

        # Demo 4: Data analysis example
        notebook.add_markdown_cell("## Demo 4: Data Analysis Example")

        async with manager.acquire_kernel("data-analyst") as (km, session):
            code5 = """# Data analysis example
import pandas as pd
import numpy as np

# Create sample data
data = {
    'product': ['A', 'B', 'C', 'D', 'E'],
    'sales': [100, 150, 200, 75, 125],
    'profit': [20, 30, 50, 15, 25]
}

df = pd.DataFrame(data)
print("Sample Sales Data:")
print(df)
print()

# Calculate metrics
total_sales = df['sales'].sum()
avg_profit = df['profit'].mean()
best_product = df.loc[df['sales'].idxmax(), 'product']

print(f"Total Sales: ${total_sales}")
print(f"Average Profit: ${avg_profit:.2f}")
print(f"Best Product: {best_product}")"""

            outputs5 = await manager.execute_code_for_notebook(session, code5)
            notebook.add_code_cell_with_outputs(code5, outputs5)

            # Create a simple visualization
            code6 = """# Simple visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)
plt.bar(df['product'], df['sales'])
plt.title('Sales by Product')
plt.ylabel('Sales ($)')

plt.subplot(1, 2, 2)
plt.bar(df['product'], df['profit'])
plt.title('Profit by Product')
plt.ylabel('Profit ($)')

plt.tight_layout()
plt.savefig('examples/output/demo_charts.png', dpi=100, bbox_inches='tight')
plt.show()

print("Charts saved to examples/output/demo_charts.png")"""

            outputs6 = await manager.execute_code_for_notebook(session, code6)
            notebook.add_code_cell_with_outputs(code6, outputs6)

        # Demo 5: Error handling
        notebook.add_markdown_cell("## Demo 5: Error Handling")

        async with manager.acquire_kernel("error-demo") as (km, session):
            code7 = """# Demonstrate error handling
try:
    # This will cause a division by zero error
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught error in Python: {e}")

# This will cause a NameError
try:
    print(undefined_variable)
except NameError as e:
    print(f"Caught another error: {e}")

print("Error handling demo complete")"""

            outputs7 = await manager.execute_code_for_notebook(session, code7)
            notebook.add_code_cell_with_outputs(code7, outputs7)

        # Demo 6: Session checkpointing
        notebook.add_markdown_cell("## Demo 6: Session Checkpointing")

        # Create a session with some state
        async with manager.acquire_kernel("checkpoint-demo") as (km, session):
            code8 = (
                """# Create some state to checkpoint
checkpoint_data = {
    'timestamp': '"""
                + datetime.now().isoformat()
                + """',
    'calculations': [],
    'counter': 0
}

for i in range(5):
    checkpoint_data['counter'] += 1
    result = i ** 2
    checkpoint_data['calculations'].append(f"{i}² = {result}")

print(f"Counter: {checkpoint_data['counter']}")
print("Calculations:")
for calc in checkpoint_data['calculations']:
    print(f"  {calc}")"""
            )

            outputs8 = await manager.execute_code_for_notebook(session, code8)
            notebook.add_code_cell_with_outputs(code8, outputs8)

            # Save checkpoint
            checkpoint_data = manager.save_checkpoint(session)

        notebook.add_markdown_cell(f"""### Checkpoint Data Created

Session checkpoint saved with:
- Session ID: {checkpoint_data.get("session_id", "N/A")}
- Agent ID: {checkpoint_data.get("agent_id", "N/A")}
- Execution history entries: {len(checkpoint_data.get("execution_history", []))}
- Checkpoint timestamp: {checkpoint_data.get("checkpoint_timestamp", "N/A")}
""")

        # Demo 7: Session statistics
        notebook.add_markdown_cell("## Demo 7: Session Statistics")

        stats_content = f"""### Kernel Manager Session Summary

Total active sessions: {len(manager.sessions)}

Session details:
"""

        for agent_id, session in manager.sessions.items():
            stats_content += f"""
**Agent: {agent_id}**
- Kernel ID: {session.kernel_id}
- Created: {datetime.fromtimestamp(session.created_at).strftime("%H:%M:%S")}
- Executions: {len(session.execution_history)}
- Last checkpoint: {datetime.fromtimestamp(session.last_checkpoint).strftime("%H:%M:%S") if session.last_checkpoint else "None"}
"""

        notebook.add_markdown_cell(stats_content)

        print("=== Demo completed successfully ===")

    # Save the notebook
    notebook_data = notebook.to_notebook()

    with open(notebook_path, "w") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"\n✅ Demo notebook saved to: {notebook_path}")
    print(f"📊 Total cells created: {len(notebook_data['cells'])}")
    print(f"📝 Code cells: {len([c for c in notebook_data['cells'] if c['cell_type'] == 'code'])}")
    print(f"📄 Markdown cells: {len([c for c in notebook_data['cells'] if c['cell_type'] == 'markdown'])}")
    print("\nTo view the notebook, run:")
    print(f"  jupyter notebook {notebook_path}")


if __name__ == "__main__":
    print("Starting Kernel Manager Demo...")
    print("This will create a Jupyter notebook demonstrating kernel manager features.\n")

    try:
        asyncio.run(demonstrate_kernel_manager())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()
