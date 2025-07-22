"""Analyze Business Accounting.xlsx with Claude Sonnet 4."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from spreadsheet_analyzer.excel_to_notebook import ExcelToNotebookConverter
from spreadsheet_analyzer.notebook_llm.llm_providers import get_provider
from spreadsheet_analyzer.notebook_llm.llm_providers.base import Message, Role
from spreadsheet_analyzer.notebook_llm.strategies import get_strategy
from spreadsheet_analyzer.notebook_llm.strategies.base import AnalysisFocus, AnalysisTask, ResponseFormat
from spreadsheet_analyzer.pipeline.pipeline import DeterministicPipeline

if TYPE_CHECKING:
    from spreadsheet_analyzer.excel_to_notebook.base import NotebookDocument
    from spreadsheet_analyzer.notebook_llm.nap.protocols import NotebookDocument as NAPNotebook


def convert_to_nap_cells(notebook: "NotebookDocument") -> "NAPNotebook":
    """Convert excel_to_notebook cells to NAP protocol cells."""
    from spreadsheet_analyzer.notebook_llm.nap.protocols import Cell
    from spreadsheet_analyzer.notebook_llm.nap.protocols import CellType as NAPCellType

    nap_cells = []
    for i, cell in enumerate(notebook.cells):
        # Map cell types
        cell_type_str = cell.cell_type.value if hasattr(cell.cell_type, "value") else str(cell.cell_type)
        if cell_type_str.lower() == "markdown":
            nap_type = NAPCellType.MARKDOWN
        elif cell_type_str.lower() == "code":
            nap_type = NAPCellType.CODE
        else:
            nap_type = NAPCellType.RAW

        nap_cell = Cell(
            id=f"cell_{i}",
            cell_type=nap_type,
            source=getattr(cell, "content", getattr(cell, "source", "")),
            metadata=cell.metadata if hasattr(cell, "metadata") else {},
        )
        nap_cells.append(nap_cell)

    # Create a NAP notebook document
    from spreadsheet_analyzer.notebook_llm.nap.protocols import NotebookDocument as NAPNotebook

    return NAPNotebook(
        id="excel_analysis",
        cells=nap_cells,
        metadata=notebook.metadata if hasattr(notebook, "metadata") else {},
        kernel_spec={"name": "python3"},
        language_info={"name": "python"},
    )


def analyze_business_accounting() -> None:
    """Analyze the Business Accounting Excel file with Claude Sonnet 4."""

    # Check for Anthropic API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set!")
        print("Please set your Anthropic API key to use Claude Sonnet 4.")
        return

    # File path
    excel_file = Path(
        "/Users/cheickberthe/PycharmProjects/spreadsheet-analyzer/test-files/business-accounting/Business Accounting.xlsx"
    )

    if not excel_file.exists():
        print(f"ERROR: Excel file not found: {excel_file}")
        return

    print("=== Analyzing Business Accounting Excel File ===")
    print(f"File: {excel_file.name}")
    print(f"Size: {excel_file.stat().st_size / 1024:.1f} KB")
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Step 1: Run deterministic analysis first
    print("Step 1: Running deterministic analysis...")
    pipeline = DeterministicPipeline()
    pipeline_result = pipeline.run(excel_file)

    if pipeline_result.errors:
        print(f"ERROR in deterministic analysis: {pipeline_result.errors}")
        return

    print("✓ Deterministic analysis complete")
    print(f"  - Execution time: {pipeline_result.execution_time:.2f}s")
    if pipeline_result.structure:
        print(f"  - Sheets analyzed: {len(pipeline_result.structure.sheets)}")
    if pipeline_result.formulas:
        total_formulas = len(pipeline_result.formulas.dependency_graph)
        print(f"  - Total formulas found: {total_formulas}")
    if pipeline_result.security:
        print(f"  - Security risk level: {pipeline_result.security.risk_level}")
    print()

    # Step 2: Convert to notebook format
    print("Step 2: Converting to notebook format...")
    converter = ExcelToNotebookConverter()
    excel_notebook = converter.convert(excel_file)

    print(f"✓ Excel notebook created with {len(excel_notebook.cells)} cells")

    # Convert to NAP protocol format for strategies
    notebook = convert_to_nap_cells(excel_notebook)

    # Add deterministic analysis results to notebook if available
    if pipeline_result.success:
        results_summary = ["# Deterministic Analysis Results"]
        if pipeline_result.structure:
            results_summary.append(f"# Sheets: {len(pipeline_result.structure.sheets)}")
        if pipeline_result.formulas:
            total_formulas = len(pipeline_result.formulas.dependency_graph)
            results_summary.append(f"# Formulas: {total_formulas}")
        if pipeline_result.security:
            results_summary.append(f"# Security risk: {pipeline_result.security.risk_level}")

        # Create a cell using the same structure as existing cells
        from spreadsheet_analyzer.notebook_llm.nap.protocols import Cell, CellType

        analysis_cell = Cell(
            id=f"analysis_{len(notebook.cells)}",
            cell_type=CellType.CODE,
            source="\n".join(results_summary),
            metadata={"analysis_type": "deterministic"},
        )
        notebook.cells.append(analysis_cell)

    # Step 3: Initialize Claude Sonnet 4
    print("\nStep 3: Initializing Claude Sonnet 4...")
    try:
        # Use claude-sonnet-4 model
        provider = get_provider("anthropic", model="claude-sonnet-4-20250514")
        print(f"✓ Initialized {provider.model_name}")
        print(f"  - Max context: {provider.max_context_tokens:,} tokens\n")
    except Exception as e:
        print(f"ERROR initializing Claude: {e}")
        return

    # Step 4: Analyze with different strategies
    print("Step 4: Running LLM analysis with multiple strategies...\n")

    strategies = ["graph_based", "hierarchical"]
    results = {}

    for strategy_name in strategies:
        print(f"--- Using {strategy_name} strategy ---")

        try:
            strategy = get_strategy(strategy_name)

            # Prepare context with formula focus
            context = strategy.prepare_context(
                notebook,
                AnalysisFocus.FORMULAS,
                token_budget=4000,  # Conservative budget for Sonnet
            )

            print("Context prepared:")
            print(f"  - Compression method: {context.compression_method}")
            print(f"  - Token count: {context.token_count}")
            print(f"  - Cells selected: {len(context.cells)}")

            # Create analysis task
            task = AnalysisTask(
                name="business_accounting_analysis",
                description="Analyze the business accounting spreadsheet structure, formulas, and data flow",
                focus=AnalysisFocus.FORMULAS,
                expected_format=ResponseFormat.STRUCTURED,
            )

            # Format prompt
            prompt = strategy.format_prompt(context, task)

            # Get LLM response
            messages = [
                Message(
                    role=Role.SYSTEM,
                    content="You are an expert Excel analyst specializing in business accounting spreadsheets. Analyze the structure, formulas, and data relationships.",
                ),
                Message(role=Role.USER, content=prompt),
            ]

            print("Sending request to Claude...")
            response = provider.complete(messages, temperature=0.1)

            print("✓ Response received")
            print(f"  - Tokens used: {response.usage}")

            results[strategy_name] = {
                "context": context.to_dict(),
                "response": response.content,
                "usage": response.usage,
            }

            # Add result to notebook
            result_cell = Cell(
                id=f"llm_{strategy_name}_{len(notebook.cells)}",
                cell_type=CellType.MARKDOWN,
                source=f"## {strategy_name.replace('_', ' ').title()} Strategy Analysis\n\n{response.content}",
                metadata={"analysis_type": "llm", "strategy": strategy_name},
            )
            notebook.cells.append(result_cell)

        except Exception as e:
            print(f"ERROR with {strategy_name} strategy: {e}")
            results[strategy_name] = {"error": str(e)}

        print()

    # Step 5: Save results
    print("Step 5: Saving results...")

    # Save notebook
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    notebook_file = output_dir / f"business_accounting_analysis_{timestamp}.ipynb"

    # Convert notebook to Jupyter format
    def format_cell_source(cell: Any) -> list[str]:
        """Format cell source content for Jupyter notebook."""
        content = getattr(cell, "source", getattr(cell, "content", ""))

        if isinstance(content, str):
            # Split content into lines, preserving newlines
            lines = content.split("\n")

            # Add newline to each line except the last (nbformat spec)
            formatted_lines = []
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    formatted_lines.append(line + "\n")
                else:
                    # Last line doesn't need trailing newline
                    formatted_lines.append(line)

            return formatted_lines
        elif isinstance(content, list):
            return content
        else:
            return [str(content)]

    jupyter_notebook = {
        "cells": [
            {
                "cell_type": cell.cell_type.value,
                "metadata": cell.metadata,
                "source": format_cell_source(cell),
                "outputs": cell.outputs if hasattr(cell, "outputs") else [],
            }
            for cell in notebook.cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with notebook_file.open("w") as f:
        json.dump(jupyter_notebook, f, indent=2)

    print(f"✓ Notebook saved to: {notebook_file}")

    # Save raw results
    results_file = output_dir / f"business_accounting_results_{timestamp}.json"
    with results_file.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Raw results saved to: {results_file}")

    # Print summary
    print("\n=== Analysis Complete ===")
    print("Total tokens used:")
    for strategy, result in results.items():
        if "usage" in result and isinstance(result["usage"], dict) and "total_tokens" in result["usage"]:
            print(f"  - {strategy}: {result['usage']['total_tokens']} tokens")
        elif "usage" in result:
            print(f"  - {strategy}: {result['usage']}")

    print(f"\nResults saved in: {output_dir}/")
    print("Open the notebook file in Jupyter to view the full analysis.")


if __name__ == "__main__":
    analyze_business_accounting()
