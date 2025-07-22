"""Notebook-LLM interface framework for spreadsheet analysis.

This module provides the interface between Jupyter notebooks and Large Language Models
for intelligent spreadsheet analysis. It implements a three-layer architecture:

1. Orchestration Layer - Workflow management and multi-agent coordination
2. Strategy Layer - Prompt engineering and context compression strategies
3. NAP Protocol Layer - Low-level notebook operations

Key components:
- Template management system using Jinja2
- Plugin architecture for extensible strategies
- Context compression for efficient token usage
- Multi-tier model routing for cost optimization
"""

from spreadsheet_analyzer.notebook_llm.protocol.base import (
    AnalysisPhase,
    NotebookCell,
    NotebookCellType,
)
from spreadsheet_analyzer.notebook_llm.templates import (
    StrategyTemplateLoader,
    TemplateManager,
    get_template_manager,
    render_prompt,
)

__all__ = [
    "AnalysisPhase",
    "NotebookCell",
    "NotebookCellType",
    "StrategyTemplateLoader",
    "TemplateManager",
    "get_template_manager",
    "render_prompt",
]
