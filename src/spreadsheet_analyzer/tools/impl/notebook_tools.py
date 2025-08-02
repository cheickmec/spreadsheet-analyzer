"""Notebook-specific tool implementations.

This module provides functional tools for working with Jupyter notebooks
including building notebooks, executing cells, and generating reports.

CLAUDE-KNOWLEDGE: These tools provide a functional interface to notebook
operations while maintaining immutability and safety.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ...core.errors import ToolError
from ...core.types import Result, err, ok
from ..types import FunctionalTool, create_tool


# Input schemas for notebook tools
class CellExecutorInput(BaseModel):
    """Input for executing a notebook cell."""

    code: str = Field(description="Code to execute")
    cell_type: str = Field(default="code", description="Type of cell (code or markdown)")
    kernel_name: str = Field(default="python3", description="Kernel to use")


class NotebookBuilderInput(BaseModel):
    """Input for building a notebook."""

    title: str = Field(description="Notebook title")
    cells: list[dict[str, Any]] = Field(description="List of cell definitions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Notebook metadata")


class NotebookSaverInput(BaseModel):
    """Input for saving a notebook."""

    notebook_data: dict[str, Any] = Field(description="Notebook data structure")
    output_path: str = Field(description="Path to save the notebook")
    overwrite: bool = Field(default=False, description="Overwrite existing file")


class MarkdownGeneratorInput(BaseModel):
    """Input for generating markdown from analysis results."""

    analysis_type: str = Field(description="Type of analysis performed")
    results: dict[str, Any] = Field(description="Analysis results")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    include_summary: bool = Field(default=True, description="Include executive summary")


# Tool implementations
def create_cell_executor_tool() -> FunctionalTool:
    """Create a tool for executing notebook cells."""

    def execute(args: CellExecutorInput) -> Result[dict[str, Any], ToolError]:
        """Execute a notebook cell."""
        try:
            # Import here to avoid circular dependencies

            if args.cell_type == "markdown":
                # Markdown cells don't execute
                return ok({"cell_type": "markdown", "source": args.code, "outputs": [], "execution_count": None})

            # For code cells, we need to execute
            # In a real implementation, this would use the kernel service
            # For now, return a mock result
            result = {
                "cell_type": "code",
                "source": args.code,
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": "# Code execution would happen here\n"}
                ],
                "execution_count": 1,
                "metadata": {"kernel": args.kernel_name},
            }

            return ok(result)

        except Exception as e:
            return err(ToolError(f"Failed to execute cell: {e}", tool_name="cell_executor", cause=e))

    return create_tool(
        name="execute_cell",
        description="Execute a notebook cell and return results",
        args_schema=CellExecutorInput,
        execute_fn=execute,
        category="notebook",
        return_type=dict,
        tags=["notebook", "execute", "cell"],
    )


def create_notebook_builder_tool() -> FunctionalTool:
    """Create a tool for building notebooks."""

    def execute(args: NotebookBuilderInput) -> Result[dict[str, Any], ToolError]:
        """Build a notebook structure."""
        try:
            # Create notebook structure
            notebook = {
                "cells": [],
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.9.0"},
                    **args.metadata,
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            }

            # Add title cell
            notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [f"# {args.title}\n"]})

            # Add provided cells
            for cell_def in args.cells:
                cell = {
                    "cell_type": cell_def.get("type", "code"),
                    "metadata": cell_def.get("metadata", {}),
                    "source": cell_def.get("source", "").split("\n")
                    if isinstance(cell_def.get("source", ""), str)
                    else cell_def.get("source", []),
                }

                if cell["cell_type"] == "code":
                    cell["execution_count"] = None
                    cell["outputs"] = []

                notebook["cells"].append(cell)

            return ok(notebook)

        except Exception as e:
            return err(ToolError(f"Failed to build notebook: {e}", tool_name="notebook_builder", cause=e))

    return create_tool(
        name="build_notebook",
        description="Build a Jupyter notebook structure",
        args_schema=NotebookBuilderInput,
        execute_fn=execute,
        category="notebook",
        return_type=dict,
        tags=["notebook", "build", "create"],
    )


def create_notebook_saver_tool() -> FunctionalTool:
    """Create a tool for saving notebooks."""

    def execute(args: NotebookSaverInput) -> Result[dict[str, Any], ToolError]:
        """Save a notebook to disk."""
        try:
            import json

            output_path = Path(args.output_path)

            # Check if file exists
            if output_path.exists() and not args.overwrite:
                return err(
                    ToolError(
                        f"File already exists: {output_path}",
                        tool_name="notebook_saver",
                        details={"use_overwrite": True},
                    )
                )

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save notebook
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(args.notebook_data, f, indent=2, ensure_ascii=False)

            result = {
                "saved_path": str(output_path),
                "file_size": output_path.stat().st_size,
                "cell_count": len(args.notebook_data.get("cells", [])),
                "success": True,
            }

            return ok(result)

        except Exception as e:
            return err(ToolError(f"Failed to save notebook: {e}", tool_name="notebook_saver", cause=e))

    return create_tool(
        name="save_notebook",
        description="Save a Jupyter notebook to disk",
        args_schema=NotebookSaverInput,
        execute_fn=execute,
        category="notebook",
        return_type=dict,
        tags=["notebook", "save", "write"],
    )


def create_markdown_generator_tool() -> FunctionalTool:
    """Create a tool for generating markdown reports."""

    def execute(args: MarkdownGeneratorInput) -> Result[str, ToolError]:
        """Generate markdown from analysis results."""
        try:
            # Use the functional markdown generator
            sections = []

            # Title
            sections.append(f"# {args.analysis_type} Analysis Report")

            # Metadata section
            if args.metadata:
                sections.append("## Metadata")
                for key, value in args.metadata.items():
                    sections.append(f"- **{key}**: {value}")

            # Executive summary
            if args.include_summary:
                sections.append("## Executive Summary")

                # Extract key metrics
                if "statistics" in args.results:
                    stats = args.results["statistics"]
                    sections.append("### Key Metrics")
                    for key, value in stats.items():
                        sections.append(f"- **{key.replace('_', ' ').title()}**: {value}")

            # Results sections
            sections.append("## Analysis Results")

            # Format results based on type
            if args.analysis_type == "formula":
                if "formulas" in args.results:
                    sections.append(f"### Formula Analysis ({len(args.results['formulas'])} formulas found)")
                    sections.append("| Cell | Formula | References |")
                    sections.append("|------|---------|------------|")
                    for formula in args.results["formulas"][:10]:  # First 10
                        refs = ", ".join(formula.get("references", []))
                        sections.append(f"| {formula['cell']} | `{formula['formula']}` | {refs} |")

                    if len(args.results["formulas"]) > 10:
                        sections.append(f"\n*... and {len(args.results['formulas']) - 10} more formulas*")

            elif args.analysis_type == "pattern":
                if "patterns" in args.results:
                    sections.append("### Detected Patterns")
                    for pattern_type, patterns in args.results["patterns"].items():
                        if patterns:
                            sections.append(f"#### {pattern_type.title()} Patterns")
                            for pattern in patterns:
                                sections.append(f"- {pattern}")

            else:
                # Generic results formatting
                for key, value in args.results.items():
                    if isinstance(value, (list, dict)) and value:
                        sections.append(f"### {key.replace('_', ' ').title()}")
                        sections.append(f"```json\n{value}\n```")
                    elif isinstance(value, (str, int, float, bool)):
                        sections.append(f"**{key.replace('_', ' ').title()}**: {value}")

            # Insights section
            if "insights" in args.results:
                sections.append("## Insights")
                for insight in args.results["insights"]:
                    sections.append(f"- {insight}")

            # Recommendations
            if "recommendations" in args.results:
                sections.append("## Recommendations")
                for i, rec in enumerate(args.results["recommendations"], 1):
                    sections.append(f"{i}. {rec}")

            markdown = "\n\n".join(sections)
            return ok(markdown)

        except Exception as e:
            return err(ToolError(f"Failed to generate markdown: {e}", tool_name="markdown_generator", cause=e))

    return create_tool(
        name="generate_markdown",
        description="Generate markdown report from analysis results",
        args_schema=MarkdownGeneratorInput,
        execute_fn=execute,
        category="notebook",
        return_type=str,
        tags=["notebook", "markdown", "report", "generate"],
    )
