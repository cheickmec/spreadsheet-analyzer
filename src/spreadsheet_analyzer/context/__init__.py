"""Context management system for optimizing LLM token usage.

This package provides composable strategies for managing context
in spreadsheet analysis, including compression, summarization,
and intelligent selection of relevant information.
"""

from .builder import (
    ContextBuilder,
    build_context_from_cells,
    create_analysis_builder,
    create_default_builder,
    create_minimal_builder,
    estimate_context_size,
)
from .strategies import (
    HybridStrategy,
    PatternCompressionStrategy,
    RangeAggregationStrategy,
    SlidingWindowStrategy,
    create_hybrid,
    create_pattern_compression,
    create_range_aggregation,
    create_sliding_window,
)
from .token_management import (
    ALLOCATION_STRATEGIES,
    MODEL_CONFIGS,
    AllocationStrategy,
    ModelConfig,
    TokenBudget,
    allocate_budget,
    calculate_context_tokens,
    estimate_tokens,
    get_compression_target,
    get_model_config,
    optimize_for_budget,
)
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
    # Token management
    "ALLOCATION_STRATEGIES",
    "AllocationStrategy",
    # Builder
    "ContextBuilder",
    # Core types
    "ContextCell",
    "ContextManager",
    "ContextPackage",
    "ContextQuery",
    "ContextStrategy",
    # Metrics and info
    "CompressionMetrics",
    # Strategies
    "HybridStrategy",
    "MODEL_CONFIGS",
    "ModelConfig",
    "PatternCompressionStrategy",
    "PatternInfo",
    "RangeAggregationStrategy",
    "RangeInfo",
    "SlidingWindowStrategy",
    # Composition
    "StrategyChain",
    "StrategyConfig",
    "TokenBudget",
    "allocate_budget",
    "build_context_from_cells",
    "calculate_context_tokens",
    "create_analysis_builder",
    "create_default_builder",
    "create_hybrid",
    "create_minimal_builder",
    "create_pattern_compression",
    "create_range_aggregation",
    "create_sliding_window",
    "estimate_context_size",
    "estimate_tokens",
    "get_compression_target",
    "get_model_config",
    "optimize_for_budget",
]
