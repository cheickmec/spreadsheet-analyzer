"""Model registry package for managing LLM providers and models.

This package organizes model providers into separate modules and provides
a unified registry interface for model discovery and validation.
"""

from .base import AgentType, ModelCapabilities, ModelInfo, ModelProvider, TaskComplexity
from .registry import (
    format_model_list,
    get_available_models,
    get_model_info,
    get_models_by_complexity,
    get_models_by_provider,
    get_recommendation_for_agent,
    get_recommended_models_for_agent,
    validate_model,
)

__all__ = [
    # Base types
    "ModelProvider",
    "AgentType",
    "TaskComplexity",
    "ModelCapabilities",
    "ModelInfo",
    # Registry functions
    "get_available_models",
    "get_models_by_provider",
    "get_recommended_models_for_agent",
    "get_models_by_complexity",
    "get_model_info",
    "validate_model",
    "format_model_list",
    "get_recommendation_for_agent",
]
