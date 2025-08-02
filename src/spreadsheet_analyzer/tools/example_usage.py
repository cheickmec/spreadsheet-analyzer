"""Example usage of the functional tools system.

This module demonstrates how to use the functional tools system
for spreadsheet analysis tasks.

CLAUDE-KNOWLEDGE: This example shows common patterns for tool usage,
composition, and integration with agents.
"""

from pathlib import Path

from ..core.types import Result
from .composition import (
    chain_tools,
    conditional_tool,
    fallback_tool,
    parallel_tools,
    retry_tool,
)
from .impl import (
    create_cell_reader_tool,
    create_formula_analyzer_tool,
    create_markdown_generator_tool,
    create_notebook_builder_tool,
    create_range_reader_tool,
    create_sheet_reader_tool,
    create_workbook_reader_tool,
)
from .registry import create_default_registry, get_global_registry


def example_basic_tool_usage():
    """Example of basic tool usage."""
    print("=== Basic Tool Usage ===")
    
    # Create a cell reader tool
    cell_reader = create_cell_reader_tool()
    
    # Execute the tool
    result = cell_reader.execute({
        "file_path": "example.xlsx",
        "sheet_name": "Sheet1",
        "cell_reference": "A1"
    })
    
    if result.is_ok():
        print(f"Cell value: {result.unwrap()}")
    else:
        print(f"Error: {result.unwrap_err()}")


def example_tool_chaining():
    """Example of chaining tools together."""
    print("\n=== Tool Chaining ===")
    
    # Create tools
    workbook_reader = create_workbook_reader_tool()
    sheet_reader = create_sheet_reader_tool()
    formula_analyzer = create_formula_analyzer_tool()
    
    # Chain them together
    analysis_chain = chain_tools(
        workbook_reader,
        sheet_reader,
        formula_analyzer
    )
    
    # Execute chain
    initial_args = {"file_path": "example.xlsx"}
    result = analysis_chain.execute(initial_args)
    
    if result.is_ok():
        for tool_result in result.unwrap():
            print(f"Tool: {tool_result.tool_name}")
            print(f"Success: {tool_result.success}")
            if tool_result.success:
                print(f"Output: {tool_result.output}")


def example_parallel_tools():
    """Example of parallel tool execution."""
    print("\n=== Parallel Tool Execution ===")
    
    # Create multiple analysis tools
    sheet_reader = create_sheet_reader_tool()
    formula_analyzer = create_formula_analyzer_tool()
    
    # Define combiner function
    def combine_results(results):
        return {
            "sheet_data": results[0],
            "formula_analysis": results[1],
            "combined": True
        }
    
    # Create parallel execution
    parallel_analysis = parallel_tools(
        [sheet_reader, formula_analyzer],
        combine_results
    )
    
    # Execute
    args = {
        "file_path": "example.xlsx",
        "sheet_name": "Sheet1"
    }
    result = parallel_analysis.execute(args)
    
    if result.is_ok():
        print(f"Combined results: {result.unwrap()}")


def example_conditional_tools():
    """Example of conditional tool execution."""
    print("\n=== Conditional Tool Execution ===")
    
    # Create tools
    simple_reader = create_cell_reader_tool()
    complex_analyzer = create_formula_analyzer_tool()
    
    # Define condition
    def needs_complex_analysis(args):
        # In real use, this might check file size or content
        return args.get("complex", False)
    
    # Create conditional tool
    smart_reader = conditional_tool(
        condition=needs_complex_analysis,
        if_true=complex_analyzer,
        if_false=simple_reader
    )
    
    # Execute for simple case
    simple_args = {"file_path": "example.xlsx", "complex": False}
    result = smart_reader.execute(simple_args)
    print(f"Simple result: {result}")
    
    # Execute for complex case
    complex_args = {"file_path": "example.xlsx", "complex": True}
    result = smart_reader.execute(complex_args)
    print(f"Complex result: {result}")


def example_fallback_tools():
    """Example of fallback tool pattern."""
    print("\n=== Fallback Tool Pattern ===")
    
    # Create primary and fallback tools
    primary_reader = create_range_reader_tool()
    fallback_reader = create_sheet_reader_tool()
    
    # Create fallback tool
    safe_reader = fallback_tool(primary_reader, fallback_reader)
    
    # Execute
    args = {
        "file_path": "example.xlsx",
        "sheet_name": "Sheet1",
        "range_reference": "A1:Z1000"  # Might fail on large range
    }
    result = safe_reader.execute(args)
    
    if result.is_ok():
        print("Reading succeeded")
    else:
        print(f"Both tools failed: {result.unwrap_err()}")


def example_retry_tool():
    """Example of retry tool pattern."""
    print("\n=== Retry Tool Pattern ===")
    
    # Create a tool that might fail
    cell_reader = create_cell_reader_tool()
    
    # Add retry logic
    reliable_reader = retry_tool(
        cell_reader,
        max_retries=3,
        retry_predicate=lambda error: "timeout" in str(error)
    )
    
    # Execute
    args = {
        "file_path": "example.xlsx",
        "sheet_name": "Sheet1",
        "cell_reference": "A1"
    }
    result = reliable_reader.execute(args)
    
    if result.is_ok():
        print(f"Success after retries: {result.unwrap()}")


def example_registry_usage():
    """Example of using the tool registry."""
    print("\n=== Tool Registry Usage ===")
    
    # Get default registry
    registry = get_global_registry()
    
    # List all tools
    print("Available tools:")
    for metadata in registry.list_tools():
        print(f"  - {metadata.name}: {metadata.description}")
    
    # Search for Excel tools
    print("\nExcel tools:")
    excel_tools = registry.search("excel")
    for metadata in excel_tools:
        print(f"  - {metadata.name}")
    
    # Get tools by category
    print("\nNotebook tools:")
    notebook_tools = registry.list_tools("notebook")
    for metadata in notebook_tools:
        print(f"  - {metadata.name}")
    
    # Get specific tool
    cell_reader_opt = registry.get("read_cell")
    if cell_reader_opt.is_some():
        tool = cell_reader_opt.unwrap()
        print(f"\nFound tool: {tool.metadata.name}")


def example_markdown_generation():
    """Example of generating markdown reports."""
    print("\n=== Markdown Generation ===")
    
    # Create markdown generator
    markdown_gen = create_markdown_generator_tool()
    
    # Prepare analysis results
    analysis_results = {
        "statistics": {
            "total_cells": 1000,
            "formula_cells": 150,
            "empty_cells": 200
        },
        "formulas": [
            {"cell": "A1", "formula": "=SUM(B1:B10)", "references": ["B1", "B2", "B3"]},
            {"cell": "C1", "formula": "=AVERAGE(D1:D10)", "references": ["D1", "D2", "D3"]}
        ],
        "insights": [
            "High formula density in column A",
            "Circular references detected in sheet 2"
        ],
        "recommendations": [
            "Consider breaking complex formulas into smaller parts",
            "Add data validation to prevent errors"
        ]
    }
    
    # Generate markdown
    result = markdown_gen.execute({
        "analysis_type": "Excel Formula",
        "results": analysis_results,
        "metadata": {
            "file": "example.xlsx",
            "analyzed_by": "spreadsheet-analyzer",
            "date": "2024-01-01"
        },
        "include_summary": True
    })
    
    if result.is_ok():
        markdown = result.unwrap()
        print("Generated markdown:")
        print(markdown[:200] + "...")  # First 200 chars


def example_langchain_integration():
    """Example of LangChain integration."""
    print("\n=== LangChain Integration ===")
    
    # Create a tool
    cell_reader = create_cell_reader_tool()
    
    # Convert to LangChain format
    langchain_tool = cell_reader.to_langchain()
    
    print(f"LangChain tool name: {langchain_tool.name}")
    print(f"LangChain tool description: {langchain_tool.description}")
    
    # Use with LangChain (pseudo-code)
    # from langchain.agents import AgentExecutor
    # agent = AgentExecutor(tools=[langchain_tool], ...)
    # result = agent.run("Read cell A1 from Sheet1")


def run_all_examples():
    """Run all examples."""
    examples = [
        example_basic_tool_usage,
        example_tool_chaining,
        example_parallel_tools,
        example_conditional_tools,
        example_fallback_tools,
        example_retry_tool,
        example_registry_usage,
        example_markdown_generation,
        example_langchain_integration
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    run_all_examples()