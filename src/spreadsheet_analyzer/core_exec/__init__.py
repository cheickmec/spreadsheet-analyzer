"""
Core execution layer for notebook management.

This module provides generic, domain-agnostic functionality for:
- Jupyter kernel management and execution
- Notebook building and manipulation
- I/O operations with proper nbformat handling
- Execution bridging between notebooks and kernels
"""

from .kernel_service import (
    KernelService, 
    KernelProfile, 
    ExecutionResult,
    KernelTimeoutError,
    KernelResourceLimitError
)
from .notebook_builder import NotebookBuilder, NotebookCell, CellType
from .notebook_io import NotebookIO
from .bridge import ExecutionBridge, ExecutionStats
from .quality import QualityMetrics, QualityInspector, QualityLevel, QualityIssue

__all__ = [
    "KernelService",
    "KernelProfile", 
    "ExecutionResult",
    "KernelTimeoutError",
    "KernelResourceLimitError",
    "NotebookBuilder",
    "NotebookCell",
    "CellType",
    "NotebookIO",
    "ExecutionBridge",
    "ExecutionStats",
    "QualityMetrics",
    "QualityInspector",
    "QualityLevel",
    "QualityIssue",
] 