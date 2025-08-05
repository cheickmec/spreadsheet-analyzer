"""Workflow implementations for multi-agent coordination.

This module contains LangGraph-based workflows for coordinating
multiple agents in spreadsheet analysis tasks.
"""

from .multi_table_workflow import (
    SpreadsheetAnalysisState,
    create_multi_table_workflow,
    run_multi_table_analysis,
)

__all__ = [
    "SpreadsheetAnalysisState",
    "create_multi_table_workflow",
    "run_multi_table_analysis",
]
