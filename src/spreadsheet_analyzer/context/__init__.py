"""Context management system for optimizing LLM token usage.

This package provides composable strategies for managing context
in spreadsheet analysis, including compression, summarization,
and intelligent selection of relevant information.
"""

from .types import (
    CompressionMetrics,
    ContextCell,
    ContextManager,
    ContextPackage,
    ContextQuery,
    ContextStrategy,
    PatternInfo,
    RangeInfo,
    StrategyChain,
    StrategyConfig,
)

__all__ = [
    # Metrics and info
    "CompressionMetrics",
    # Core types
    "ContextCell",
    "ContextManager",
    "ContextPackage",
    "ContextQuery",
    "ContextStrategy",
    "PatternInfo",
    "RangeInfo",
    # Composition
    "StrategyChain",
    "StrategyConfig",
]
