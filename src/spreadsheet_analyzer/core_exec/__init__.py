"""
Core execution layer for notebook management.

This module provides generic, domain-agnostic functionality for:
- Jupyter kernel management and execution
- Notebook building and manipulation
- I/O operations with proper nbformat handling
- Execution bridging between notebooks and kernels
"""

from .bridge import ExecutionBridge, ExecutionStats
from .kernel_service import ExecutionResult, KernelProfile, KernelResourceLimitError, KernelService, KernelTimeoutError
from .notebook_builder import NotebookBuilder
from .notebook_io import NotebookIO
from .quality import QualityInspector, QualityIssue, QualityLevel, QualityMetrics

__all__ = [
    "ExecutionBridge",
    "ExecutionResult",
    "ExecutionStats",
    "KernelProfile",
    "KernelResourceLimitError",
    "KernelService",
    "KernelTimeoutError",
    "NotebookBuilder",
    "NotebookIO",
    "QualityInspector",
    "QualityIssue",
    "QualityLevel",
    "QualityMetrics",
]
