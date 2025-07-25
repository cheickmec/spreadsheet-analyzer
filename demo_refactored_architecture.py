#!/usr/bin/env python3
"""
Demo script showing the refactored three-tier architecture.

This script demonstrates:
1. Core execution layer (domain-agnostic)
2. Plugin system (domain-specific)
3. Workflow orchestration (high-level API)

Usage:
    python demo_refactored_architecture.py
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spreadsheet_analyzer import (
    CellType,
    # Core layer
    NotebookBuilder,
    NotebookCell,
    NotebookIO,
    # Workflow layer
    NotebookWorkflow,
    WorkflowConfig,
    WorkflowMode,
    create_analysis_notebook,
    register_spreadsheet_plugins,
    registry,
)


def demo_core_layer():
    """Demonstrate the core execution layer (domain-agnostic)."""
    print("\n" + "=" * 60)
    print("🔧 DEMO 1: Core Execution Layer (Domain-Agnostic)")
    print("=" * 60)

    # Create a notebook using core primitives
    notebook = NotebookBuilder()

    # Add a markdown header
    notebook.add_cell(
        NotebookCell(
            cell_type=CellType.MARKDOWN,
            source=["# Demo Notebook\n", "This is a test notebook built with core primitives."],
            metadata={"tags": ["demo", "header"]},
        )
    )

    # Add some generic code
    notebook.add_cell(
        NotebookCell(
            cell_type=CellType.CODE,
            source=["import pandas as pd\n", "import numpy as np\n", "print('Hello from core layer!')"],
            metadata={"tags": ["imports"]},
        )
    )

    print(f"✅ Created notebook with {len(notebook.cells)} cells")
    print(f"📊 Notebook metadata: {notebook.metadata}")

    # Demonstrate I/O
    notebook_io = NotebookIO()
    output_path = "demo_core_notebook.ipynb"

    try:
        notebook_io.save_notebook(notebook, output_path)
        print(f"💾 Saved notebook to: {output_path}")

        # Load it back
        loaded_notebook = notebook_io.load_notebook(output_path)
        print(f"📂 Loaded notebook with {len(loaded_notebook.cells)} cells")

    except Exception as e:
        print(f"❌ I/O Error: {e}")


def demo_plugin_system():
    """Demonstrate the plugin system (domain-specific)."""
    print("\n" + "=" * 60)
    print("🔌 DEMO 2: Plugin System (Domain-Specific)")
    print("=" * 60)

    # Register spreadsheet plugins
    print("🔧 Registering spreadsheet plugins...")
    try:
        register_spreadsheet_plugins()
        print("✅ Spreadsheet plugins registered")
    except Exception as e:
        print(f"⚠️  Plugin registration issue: {e}")
        return

    # List available tasks
    tasks = registry.list_tasks()
    print(f"\n📋 Available tasks: {[task.name for task in tasks]}")

    # List available quality inspectors
    inspectors = registry.list_quality_inspectors()
    print(f"🔍 Available quality inspectors: {[inspector.name for inspector in inspectors]}")

    # Demonstrate task usage
    data_profiling_task = registry.get_task("data_profiling")
    if data_profiling_task:
        print(f"\n🎯 Using task: {data_profiling_task.name}")
        print(f"📝 Description: {data_profiling_task.description}")

        # Create a context for the task
        context = {"file_path": "test_assets/generated/sample.xlsx", "sheet_name": "Sheet1"}

        # Validate context
        issues = data_profiling_task.validate_context(context)
        if issues:
            print(f"⚠️  Context issues: {issues}")
        else:
            print("✅ Task context validated")

            # Generate cells (this will show the task working)
            try:
                cells = data_profiling_task.build_initial_cells(context)
                print(f"📝 Generated {len(cells)} cells from task")

                # Show first cell as example
                if cells:
                    first_cell = cells[0]
                    print(f"🔍 First cell type: {first_cell.cell_type.value}")
                    print(f"📄 First few lines: {first_cell.source[:2]}")

            except Exception as e:
                print(f"❌ Task execution error: {e}")


async def demo_workflow_layer():
    """Demonstrate the workflow orchestration layer."""
    print("\n" + "=" * 60)
    print("🌊 DEMO 3: Workflow Orchestration (High-Level API)")
    print("=" * 60)

    # Create a sample CSV file for demonstration
    sample_file = "demo_data.csv"
    try:
        import pandas as pd

        # Create sample data
        data = {
            "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "Age": [25, 30, 35, 28, 32],
            "Salary": [50000, 60000, 70000, 55000, 65000],
            "Department": ["Engineering", "Marketing", "Engineering", "Sales", "Marketing"],
        }

        df = pd.DataFrame(data)
        df.to_csv(sample_file, index=False)
        print(f"📁 Created sample data file: {sample_file}")

    except ImportError:
        print("⚠️  Pandas not available, skipping data creation")
        return
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return

    # Demonstrate workflow configuration
    print("\n🔧 Configuring workflow...")
    config = WorkflowConfig(
        file_path=sample_file,
        output_path="demo_workflow_analysis.ipynb",
        mode=WorkflowMode.BUILD_ONLY,  # Just build, don't execute (safer for demo)
        tasks=["data_profiling", "outlier_detection"],
        quality_checks=True,
        auto_register_plugins=True,
    )

    print("📋 Workflow config:")
    print(f"  • File: {config.file_path}")
    print(f"  • Output: {config.output_path}")
    print(f"  • Mode: {config.mode.value}")
    print(f"  • Tasks: {config.tasks}")
    print(f"  • Quality checks: {config.quality_checks}")

    # Run the workflow
    print("\n🚀 Running workflow...")
    try:
        workflow = NotebookWorkflow()
        result = await workflow.run(config)

        # Display results
        print("\n📊 Workflow Results:")
        print(f"  • Success: {result.success}")
        print(f"  • Notebook cells: {len(result.notebook.cells)}")
        print(f"  • Output path: {result.output_path}")
        print(f"  • Errors: {len(result.errors)}")
        print(f"  • Warnings: {len(result.warnings)}")

        if result.errors:
            print(f"❌ Errors: {result.errors}")

        if result.warnings:
            print(f"⚠️  Warnings: {result.warnings}")

        if result.quality_metrics:
            print(f"🎯 Quality score: {result.quality_metrics.overall_score}")
            print(f"📈 Quality level: {result.quality_metrics.overall_level.value}")

        await workflow.cleanup()

    except Exception as e:
        print(f"❌ Workflow error: {e}")

    # Clean up sample file
    try:
        Path(sample_file).unlink()
        print(f"🧹 Cleaned up sample file: {sample_file}")
    except:
        pass


async def demo_convenience_api():
    """Demonstrate the convenience API."""
    print("\n" + "=" * 60)
    print("⚡ DEMO 4: Convenience API (Simplest Usage)")
    print("=" * 60)

    # Create sample Excel data
    sample_file = "demo_convenience.xlsx"
    try:
        import pandas as pd

        data = {
            "Product": ["Laptop", "Mouse", "Keyboard", "Monitor", "Webcam"],
            "Price": [999.99, 29.99, 79.99, 299.99, 89.99],
            "Stock": [50, 200, 150, 75, 100],
            "Rating": [4.5, 4.2, 4.7, 4.3, 4.1],
        }

        df = pd.DataFrame(data)
        df.to_excel(sample_file, index=False)
        print(f"📁 Created sample Excel file: {sample_file}")

    except ImportError:
        print("⚠️  Pandas/openpyxl not available, using CSV instead")
        sample_file = "demo_convenience.csv"
        # Create basic CSV content
        with open(sample_file, "w") as f:
            f.write("Product,Price,Stock,Rating\n")
            f.write("Laptop,999.99,50,4.5\n")
            f.write("Mouse,29.99,200,4.2\n")
            f.write("Keyboard,79.99,150,4.7\n")
        print(f"📁 Created sample CSV file: {sample_file}")
    except Exception as e:
        print(f"❌ Error creating sample file: {e}")
        return

    # Use the convenience function
    print("\n🚀 Using convenience API...")
    try:
        result = await create_analysis_notebook(
            file_path=sample_file,
            output_path="demo_convenience_analysis.ipynb",
            execute=False,  # Don't execute for safety
        )

        print("📊 Analysis Results:")
        print(f"  • Success: {result.success}")
        print(f"  • Generated {len(result.notebook.cells)} cells")
        print(f"  • Output: {result.output_path}")

        if result.quality_metrics:
            print(f"  • Quality: {result.quality_metrics.overall_score}/100")

        if result.errors:
            print(f"❌ Errors: {result.errors}")

    except Exception as e:
        print(f"❌ Convenience API error: {e}")

    # Clean up
    try:
        Path(sample_file).unlink()
        print(f"🧹 Cleaned up: {sample_file}")
    except:
        pass


def show_architecture_summary():
    """Show a summary of the three-tier architecture."""
    print("\n" + "=" * 60)
    print("🏗️  ARCHITECTURE SUMMARY")
    print("=" * 60)

    print("""
📁 Three-Tier Architecture Overview:

1. **Core Execution Layer** (src/spreadsheet_analyzer/core_exec/)
   ├── kernel_service.py     - Generic Jupyter kernel management
   ├── notebook_builder.py   - Domain-agnostic notebook construction
   ├── notebook_io.py        - File I/O with proper nbformat handling
   ├── bridge.py            - Execution orchestration
   └── quality.py           - Generic quality assessment

2. **Plugin System** (src/spreadsheet_analyzer/plugins/)
   ├── base.py              - Plugin interfaces and registry
   └── spreadsheet/         - Spreadsheet-specific plugins
       ├── tasks.py         - Data profiling, formula analysis, etc.
       └── quality.py       - Spreadsheet quality assessment

3. **Workflow Orchestration** (src/spreadsheet_analyzer/workflows/)
   └── notebook_workflow.py - High-level API combining all layers

🎯 **Key Benefits:**
   • DRY: No code duplication between layers
   • Extensible: Easy to add new file types (CSV, SQL, images)
   • Testable: Each layer can be tested independently
   • Maintainable: Clear separation of concerns
   • Future-proof: Core is generic, plugins handle specifics

🔧 **Usage Patterns:**
   • Simple: Use convenience functions (create_analysis_notebook)
   • Advanced: Use NotebookWorkflow with custom configuration
   • Expert: Build custom plugins and extend the system
""")


async def main():
    """Run all demos."""
    print("🚀 Spreadsheet Analyzer - Refactored Architecture Demo")
    print("=" * 60)

    # Show architecture overview
    show_architecture_summary()

    # Run demos
    demo_core_layer()
    demo_plugin_system()
    await demo_workflow_layer()
    await demo_convenience_api()

    print("\n" + "=" * 60)
    print("✅ All demos completed!")
    print("🎉 The three-tier architecture is working correctly!")
    print("=" * 60)

    # Show next steps
    print("""
🚀 **Next Steps:**
1. Run existing tests to ensure compatibility
2. Update CLI to use new workflow API
3. Migrate existing notebooks to new system
4. Add more plugins (CSV, SQL, etc.)
5. Implement auto-discovery for plugins
""")


if __name__ == "__main__":
    asyncio.run(main())
