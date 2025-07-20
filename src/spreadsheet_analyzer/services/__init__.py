"""Service layer for business logic.

This module provides service classes that encapsulate business logic and can be
used from both CLI commands and future API endpoints.
"""

from spreadsheet_analyzer.pipeline.types import AnalysisOptions, AnalysisResult
from spreadsheet_analyzer.services.analysis_service import AnalysisService

__all__ = [
    "AnalysisOptions",
    "AnalysisResult",
    "AnalysisService",
]
