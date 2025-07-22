"""Context compression and optimization utilities for LLM interactions.

This package provides sophisticated context management strategies to optimize
token usage while preserving analytical insights. It implements the context
engineering approaches described in the system design document.

Key components:
- BaseCompressor: Abstract base class for compression strategies
- SpreadsheetLLMCompressor: Advanced compression using SpreadsheetLLM techniques
- TokenOptimizer: Dynamic token budget management and pipeline selection
- TokenCounter: Utility for accurate token counting across different models
"""

from spreadsheet_analyzer.notebook_llm.context.compressors import (
    BaseCompressor,
    CellObservation,
    CompressionMetrics,
    SpreadsheetLLMCompressor,
    TokenCounter,
)
from spreadsheet_analyzer.notebook_llm.context.token_optimization import (
    CompressionPipeline,
    OptimizationResult,
    TokenOptimizer,
)

__all__ = [
    "BaseCompressor",
    "CellObservation",
    "CompressionMetrics",
    "CompressionPipeline",
    "OptimizationResult",
    "SpreadsheetLLMCompressor",
    "TokenCounter",
    "TokenOptimizer",
]
