"""Strategy layer for LLM-Jupyter notebook integration.

This package provides the strategy pattern implementation for different
approaches to prompt engineering and context management.
"""

from typing import Any

from .base import (
    AnalysisFocus,
    AnalysisResult,
    AnalysisStrategy,
    AnalysisTask,
    BaseStrategy,
    ContextPackage,
    LLMInterface,
    ResponseFormat,
)
from .graph_based import GraphBasedStrategy
from .hierarchical import HierarchicalStrategy
from .registry import StrategyRegistry, get_registry, register_strategy


def get_strategy(name: str, **kwargs: Any) -> BaseStrategy:
    """Get a strategy by name.

    Args:
        name: Strategy name ('graph_based' or 'hierarchical')
        **kwargs: Additional strategy-specific arguments

    Returns:
        BaseStrategy: Strategy instance

    Raises:
        ValueError: If strategy name is not recognized
    """
    # Manual mapping for now since we don't have entry points set up
    strategies = {
        "graph_based": GraphBasedStrategy,
        "hierarchical": HierarchicalStrategy,
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](**kwargs)


__all__ = [
    "AnalysisFocus",
    "AnalysisResult",
    "AnalysisStrategy",
    "AnalysisTask",
    "BaseStrategy",
    "ContextPackage",
    "GraphBasedStrategy",
    "HierarchicalStrategy",
    "LLMInterface",
    "ResponseFormat",
    "StrategyRegistry",
    "get_registry",
    "get_strategy",
    "register_strategy",
]
