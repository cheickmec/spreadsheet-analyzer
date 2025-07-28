"""
Demo Notebook Tools - Entry Point

Demonstrates the complete notebook tools interface with LangChain integration.
This is the main entry point to see everything working together.
"""

import asyncio

from result import Ok

from spreadsheet_analyzer.notebook_llm_interface import create_notebook_tool_descriptions, get_notebook_tools
from spreadsheet_analyzer.notebook_session import notebook_session


async def demo_basic_session() -> None:
    """Demonstrate basic notebook session usage."""
    print("=== Basic Session Demo ===")

    async with notebook_session(session_id="demo_session") as session:
        print(f"Session started: {session.get_session_id()}")

        # Execute some code
        code1 = """
import pandas as pd
import numpy as np

print("Setting up data...")
data = pd.DataFrame({
    'A': np.random.randn(10),
    'B': np.random.randn(10),
    'C': np.random.randn(10)
})
print(f"Created DataFrame with shape: {data.shape}")
data.head()
"""

        result = await session.toolkit.execute_code(code1)
        if isinstance(result, Ok):
            print(f"✅ Code executed: {result.value.cell_id}")
            for output in result.value.outputs:
                print(f"  {output.output_type}: {output.content}")
        else:
            print(f"❌ Error: {result.value}")


async def demo_cell_types() -> None:
    """Demonstrate different cell types."""
    print("\n=== Cell Types Demo ===")

    async with notebook_session(session_id="cell_types_demo") as session:
        print(f"Session started: {session.get_session_id()}")

        # 1. Code cell (executes)
        print("\n1. Adding code cell...")
        code_result = await session.toolkit.execute_code("print('Hello from code cell!')")
        if isinstance(code_result, Ok):
            print(f"✅ Code cell executed: {code_result.value.cell_id}")

        # 2. Markdown cell (renders)
        print("\n2. Adding markdown cell...")
        markdown_content = """
# Analysis Report

This is a **markdown cell** with:
- Bullet points
- **Bold text**
- *Italic text*

## Section 2
Some more content here.
"""
        markdown_result = await session.toolkit.render_markdown(markdown_content)
        if isinstance(markdown_result, Ok):
            print(f"✅ Markdown cell rendered: {markdown_result.value.cell_id}")
            print(f"  Rendered content: {markdown_result.value.outputs[0].content[:100]}...")

        # 3. Raw cell (just stored)
        print("\n3. Adding raw cell...")
        raw_content = "This is raw text that won't be processed by the kernel."
        raw_result = await session.toolkit.add_raw_cell(raw_content)
        if isinstance(raw_result, Ok):
            print(f"✅ Raw cell added: {raw_result.value.cell_id}")

        # Show state
        state = session.toolkit.get_state()
        print(f"\n📊 Final state: {len(state.cells)} cells")


async def demo_llm_tools() -> None:
    """Demonstrate the LLM tool interface."""
    print("\n=== LLM Tools Demo ===")

    # Get the tools
    tools = get_notebook_tools()
    print(f"Available tools: {len(tools)}")

    # Show tool descriptions
    descriptions = create_notebook_tool_descriptions()
    print("\nTool descriptions:")
    print(descriptions)

    # Demonstrate tool usage (simulated)
    print("\nSimulated LLM tool usage:")

    # Example: LLM would call execute_code
    print("1. LLM calls execute_code with: print('Hello from LLM!')")

    # Example: LLM would call add_markdown_cell
    print("2. LLM calls add_markdown_cell with: # Analysis Results")

    # Example: LLM would call edit_and_execute
    print("3. LLM calls edit_and_execute to fix a cell")


async def main() -> None:
    """Run all demos."""
    print("🚀 Notebook Tools Demo")
    print("=" * 50)

    try:
        # Run basic session demo
        await demo_basic_session()

        # Run cell types demo
        await demo_cell_types()

        # Run LLM tools demo
        await demo_llm_tools()

        print("\n✅ All demos completed successfully!")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
