"""
Spreadsheet analysis plugin for notebook generation.

This plugin provides spreadsheet-specific tasks and quality inspection:
- Excel/CSV data profiling tasks
- Formula validation and analysis
- Outlier detection and data quality checks
- Business logic validation
- Spreadsheet-specific quality metrics

Usage:
    from spreadsheet_analyzer.plugins.spreadsheet import register_all_plugins
    register_all_plugins()
"""

from .tasks import DataProfilingTask, FormulaAnalysisTask, OutlierDetectionTask
from .quality import SpreadsheetQualityInspector

# Auto-register plugins
def register_all_plugins() -> None:
    """Register all spreadsheet plugins with the global registry."""
    from ..base import registry
    
    # Register tasks
    registry.register_task(DataProfilingTask())
    registry.register_task(FormulaAnalysisTask())
    registry.register_task(OutlierDetectionTask())
    
    # Register quality inspector
    registry.register_quality_inspector(SpreadsheetQualityInspector())

__all__ = [
    "DataProfilingTask",
    "FormulaAnalysisTask", 
    "OutlierDetectionTask",
    "SpreadsheetQualityInspector",
    "register_all_plugins"
] 