"""
Spreadsheet IO functionality.

This module contains I/O operations for spreadsheet files.
All domain-specific IO logic belongs here, NOT in utils.
"""

from .excel_io import (
    LARGE_FILE_ROW_LIMIT,
    LARGE_FILE_THRESHOLD_MB,
    check_file_size,
    detect_sheet_features,
    get_sheet_dimensions,
    list_sheets,
    read_sheet_async,
)

__all__ = [
    "LARGE_FILE_ROW_LIMIT",
    "LARGE_FILE_THRESHOLD_MB",
    "check_file_size",
    "detect_sheet_features",
    "get_sheet_dimensions",
    "list_sheets",
    "read_sheet_async",
]
