"""Anthropic Claude models for the model registry.

This module defines all available Claude models with their capabilities,
recommended use cases, and agent-specific recommendations.

CLAUDE-KNOWLEDGE: Based on Anthropic guidelines, Claude models excel at
reasoning, analysis, and tool use. Model selection should prioritize
accuracy for complex tasks, with consideration for cost and speed.
"""

from .base import AgentType, ModelCapabilities, ModelInfo, ModelProvider, TaskComplexity

# Claude Models (Anthropic)
CLAUDE_MODELS = {
    # Claude 3.7 Sonnet - Newest flagship
    "claude-3-7-sonnet-20250219": ModelInfo(
        model_id="claude-3-7-sonnet-20250219",
        display_name="Claude 3.7 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        capabilities=ModelCapabilities(
            reasoning_strength=9,
            speed=7,
            cost_efficiency=6,
            context_length=200000,
            supports_tools=True,
            supports_vision=True,
            supports_audio=False,
            multimodal=True,
        ),
        recommended_for=[
            AgentType.DATA_ANALYST,
            AgentType.FORMULA_ANALYZER,
            AgentType.COORDINATOR,
            AgentType.GENERAL_PURPOSE,
        ],
        best_complexity=[TaskComplexity.COMPLEX, TaskComplexity.MODERATE],
        description="Latest Claude flagship model with enhanced reasoning and analysis capabilities",
    ),
    # Claude Sonnet 4 - Most powerful
    "claude-sonnet-4-20250514": ModelInfo(
        model_id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        provider=ModelProvider.ANTHROPIC,
        capabilities=ModelCapabilities(
            reasoning_strength=10,
            speed=6,
            cost_efficiency=5,
            context_length=200000,
            supports_tools=True,
            supports_vision=True,
            supports_audio=False,
            multimodal=True,
        ),
        recommended_for=[
            AgentType.FORMULA_ANALYZER,
            AgentType.DATA_ANALYST,
            AgentType.PATTERN_FINDER,
            AgentType.COORDINATOR,
        ],
        best_complexity=[TaskComplexity.COMPLEX, TaskComplexity.RESEARCH],
        description="Most powerful Claude model for complex analytical tasks and deep reasoning",
    ),
    # Claude Opus 4 - Maximum capability
    "claude-opus-4-20250514": ModelInfo(
        model_id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        provider=ModelProvider.ANTHROPIC,
        capabilities=ModelCapabilities(
            reasoning_strength=10,
            speed=4,
            cost_efficiency=3,
            context_length=200000,
            supports_tools=True,
            supports_vision=True,
            supports_audio=False,
            multimodal=True,
        ),
        recommended_for=[AgentType.FORMULA_ANALYZER, AgentType.DATA_ANALYST, AgentType.PATTERN_FINDER],
        best_complexity=[TaskComplexity.COMPLEX, TaskComplexity.RESEARCH],
        description="Premium Claude model for the most demanding analytical tasks",
        notes="Highest cost but maximum capability - use for critical analyses",
    ),
    # Claude Opus 4.1 - Latest premium
    "claude-opus-4-1-20250805": ModelInfo(
        model_id="claude-opus-4-1-20250805",
        display_name="Claude Opus 4.1",
        provider=ModelProvider.ANTHROPIC,
        capabilities=ModelCapabilities(
            reasoning_strength=10,
            speed=4,
            cost_efficiency=3,
            context_length=200000,
            supports_tools=True,
            supports_vision=True,
            supports_audio=False,
            multimodal=True,
        ),
        recommended_for=[AgentType.FORMULA_ANALYZER, AgentType.DATA_ANALYST, AgentType.PATTERN_FINDER],
        best_complexity=[TaskComplexity.COMPLEX, TaskComplexity.RESEARCH],
        description="Latest premium Claude model with enhanced capabilities",
    ),
    # Claude 3.5 Sonnet - Balanced flagship
    "claude-3-5-sonnet-20241022": ModelInfo(
        model_id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet",
        provider=ModelProvider.ANTHROPIC,
        capabilities=ModelCapabilities(
            reasoning_strength=8,
            speed=8,
            cost_efficiency=7,
            context_length=200000,
            supports_tools=True,
            supports_vision=True,
            supports_audio=False,
            multimodal=True,
        ),
        recommended_for=[
            AgentType.TABLE_DETECTOR,
            AgentType.DATA_ANALYST,
            AgentType.STRUCTURE_ANALYZER,
            AgentType.GENERAL_PURPOSE,
        ],
        best_complexity=[TaskComplexity.MODERATE, TaskComplexity.COMPLEX],
        description="Excellent balanced model for most spreadsheet analysis tasks",
    ),
    # Claude 3.5 Haiku - Fast and efficient
    "claude-3-5-haiku-20241022": ModelInfo(
        model_id="claude-3-5-haiku-20241022",
        display_name="Claude 3.5 Haiku",
        provider=ModelProvider.ANTHROPIC,
        capabilities=ModelCapabilities(
            reasoning_strength=6,
            speed=9,
            cost_efficiency=9,
            context_length=200000,
            supports_tools=True,
            supports_vision=True,
            supports_audio=False,
            multimodal=True,
        ),
        recommended_for=[AgentType.TABLE_DETECTOR, AgentType.STRUCTURE_ANALYZER, AgentType.PATTERN_FINDER],
        best_complexity=[TaskComplexity.SIMPLE, TaskComplexity.MODERATE],
        description="Fast, cost-efficient model for basic analysis and pattern detection",
    ),
    # Legacy models
    "claude-3-5-sonnet-20240620": ModelInfo(
        model_id="claude-3-5-sonnet-20240620",
        display_name="Claude 3.5 Sonnet (Legacy)",
        provider=ModelProvider.ANTHROPIC,
        capabilities=ModelCapabilities(
            reasoning_strength=8,
            speed=8,
            cost_efficiency=7,
            context_length=200000,
            supports_tools=True,
            supports_vision=True,
            supports_audio=False,
            multimodal=True,
        ),
        recommended_for=[AgentType.GENERAL_PURPOSE],
        best_complexity=[TaskComplexity.MODERATE],
        description="Legacy version of Claude 3.5 Sonnet",
        notes="Recommend upgrading to claude-3-5-sonnet-20241022",
    ),
    "claude-3-haiku-20240307": ModelInfo(
        model_id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku (Legacy)",
        provider=ModelProvider.ANTHROPIC,
        capabilities=ModelCapabilities(
            reasoning_strength=5,
            speed=10,
            cost_efficiency=10,
            context_length=200000,
            supports_tools=True,
            supports_vision=False,
            supports_audio=False,
            multimodal=False,
        ),
        recommended_for=[AgentType.TABLE_DETECTOR],
        best_complexity=[TaskComplexity.SIMPLE],
        description="Legacy fast, cost-efficient model",
        notes="Recommend upgrading to claude-3-5-haiku-20241022",
    ),
}
