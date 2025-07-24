"""
Workflow orchestration layer for notebook generation and execution.

This module provides high-level orchestration that combines:
- Core execution primitives (kernel, notebook builder, I/O)  
- Plugin-based task selection and execution
- Quality assessment and validation
- CLI/API endpoint integration

Workflows are the main entry points for users and provide the glue
between domain-agnostic core functionality and domain-specific plugins.
"""

from .notebook_workflow import (
    NotebookWorkflow, 
    WorkflowConfig, 
    WorkflowResult, 
    WorkflowMode,
    create_analysis_notebook,
    execute_notebook
)

__all__ = [
    "NotebookWorkflow",
    "WorkflowConfig", 
    "WorkflowResult",
    "WorkflowMode",
    "create_analysis_notebook",
    "execute_notebook"
] 