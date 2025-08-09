"""Base types and enums for model registry.

This module provides the core data structures and types used across
all model provider modules.
"""

from dataclasses import dataclass
from enum import Enum


class ModelProvider(Enum):
    """Available model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    OLLAMA = "ollama"


class AgentType(Enum):
    """Types of agents in the system."""

    TABLE_DETECTOR = "table_detector"
    DATA_ANALYST = "data_analyst"
    FORMULA_ANALYZER = "formula_analyzer"
    STRUCTURE_ANALYZER = "structure_analyzer"
    PATTERN_FINDER = "pattern_finder"
    COORDINATOR = "coordinator"
    GENERAL_PURPOSE = "general_purpose"


class TaskComplexity(Enum):
    """Task complexity levels."""

    SIMPLE = "simple"  # Basic pattern recognition, simple analysis
    MODERATE = "moderate"  # Multi-step analysis, some reasoning
    COMPLEX = "complex"  # Deep reasoning, complex multi-step tasks
    RESEARCH = "research"  # Multi-step research and synthesis


@dataclass(frozen=True)
class ModelCapabilities:
    """Model capabilities and characteristics."""

    reasoning_strength: int  # 1-10 scale
    speed: int  # 1-10 scale (higher = faster)
    cost_efficiency: int  # 1-10 scale (higher = more cost efficient)
    context_length: int  # Maximum context tokens
    supports_tools: bool  # Supports function calling
    supports_vision: bool  # Supports image inputs
    supports_audio: bool  # Supports audio inputs/outputs
    multimodal: bool  # General multimodal capabilities
    supports_thinking: bool = False  # Supports extended thinking
    supports_interleaved_thinking: bool = False  # Supports interleaved thinking (Claude 4+)


@dataclass(frozen=True)
class ModelInfo:
    """Complete model information."""

    model_id: str
    display_name: str
    provider: ModelProvider
    capabilities: ModelCapabilities
    recommended_for: list[AgentType]
    best_complexity: list[TaskComplexity]
    description: str
    notes: str | None = None
    deprecated: bool = False
