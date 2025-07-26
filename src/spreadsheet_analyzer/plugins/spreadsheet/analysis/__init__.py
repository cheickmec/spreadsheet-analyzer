"""
Spreadsheet analysis functionality.

This module contains analysis operations for spreadsheet data including:
- Formula error detection
- Data profiling and quality assessment
"""

from .data_profiling import DataProfiler
from .formula_errors import FormulaErrorDetector

__all__ = ["DataProfiler", "FormulaErrorDetector"]
