"""
Deterministic Analysis Pipeline for Excel Files.

This module implements a 5-stage pipeline for comprehensive Excel file analysis
using a mix of functional and object-oriented programming paradigms.
"""

from .pipeline import (
    DeterministicPipeline,
    analyze_with_console_progress,
    create_fast_pipeline_options,
    create_lenient_pipeline_options,
    create_strict_pipeline_options,
    run_analysis,
)
from .types import (
    ContentAnalysis,
    Err,
    FormulaAnalysis,
    # Stage results
    IntegrityResult,
    # Result types
    Ok,
    # Pipeline types
    PipelineContext,
    PipelineResult,
    ProcessingClass,
    ProgressUpdate,
    Result,
    # Enums
    RiskLevel,
    SecurityReport,
    WorkbookStructure,
)

__all__ = [
    # Pipeline types
    "ContentAnalysis",
    # Main interface
    "DeterministicPipeline",
    "Err",
    "FormulaAnalysis",
    "IntegrityResult",
    "Ok",
    "PipelineContext",
    "PipelineResult",
    "ProcessingClass",
    "ProgressUpdate",
    "Result",
    "RiskLevel",
    "SecurityReport",
    "WorkbookStructure",
    "analyze_with_console_progress",
    "create_fast_pipeline_options",
    "create_lenient_pipeline_options",
    "create_strict_pipeline_options",
    "run_analysis",
]

__version__ = "0.1.0"
