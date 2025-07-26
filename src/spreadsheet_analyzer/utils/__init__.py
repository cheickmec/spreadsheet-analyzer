"""
Shared utilities for spreadsheet analyzer.

This module contains ONLY generic shared utilities that don't belong to
core_exec, plugins, or workflows layers.

IMPORTANT: Domain-specific functionality MUST go in plugins/, not here!
"""

from .cost import calculate_cost, get_token_pricing

__all__ = [
    "calculate_cost",
    "get_token_pricing",
]
