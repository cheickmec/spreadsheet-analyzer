"""Google Gemini models for the model registry.

This module defines all available Google Gemini models with their capabilities,
recommended use cases, and agent-specific recommendations.

CLAUDE-KNOWLEDGE: Gemini models offer excellent multimodal capabilities and
large context windows (up to 2M tokens). The 2.5 series provides state-of-the-art
reasoning while Flash variants optimize for speed and cost efficiency.
"""

from .base import AgentType, ModelCapabilities, ModelInfo, ModelProvider, TaskComplexity

# Google Gemini Models
GOOGLE_MODELS = {
    # Gemini 2.5 - Latest flagship models
    "gemini-2.5-pro": ModelInfo(
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        provider=ModelProvider.GOOGLE,
        capabilities=ModelCapabilities(
            reasoning_strength=10,
            speed=6,
            cost_efficiency=5,
            context_length=1048576,  # 1M tokens
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,
            multimodal=True,
        ),
        recommended_for=[
            AgentType.FORMULA_ANALYZER,
            AgentType.DATA_ANALYST,
            AgentType.COORDINATOR,
            AgentType.PATTERN_FINDER,
        ],
        best_complexity=[TaskComplexity.COMPLEX, TaskComplexity.RESEARCH],
        description="State-of-the-art thinking model with maximum response accuracy and reasoning capabilities",
    ),
    "gemini-2.5-flash": ModelInfo(
        model_id="gemini-2.5-flash",
        display_name="Gemini 2.5 Flash",
        provider=ModelProvider.GOOGLE,
        capabilities=ModelCapabilities(
            reasoning_strength=8,
            speed=8,
            cost_efficiency=7,
            context_length=1048576,
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,
            multimodal=True,
        ),
        recommended_for=[
            AgentType.DATA_ANALYST,
            AgentType.TABLE_DETECTOR,
            AgentType.STRUCTURE_ANALYZER,
            AgentType.GENERAL_PURPOSE,
        ],
        best_complexity=[TaskComplexity.MODERATE, TaskComplexity.COMPLEX],
        description="Best price-performance model with well-rounded capabilities and adaptive thinking",
    ),
    "gemini-2.5-flash-lite": ModelInfo(
        model_id="gemini-2.5-flash-lite",
        display_name="Gemini 2.5 Flash-Lite",
        provider=ModelProvider.GOOGLE,
        capabilities=ModelCapabilities(
            reasoning_strength=7,
            speed=9,
            cost_efficiency=9,
            context_length=1048576,
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,
            multimodal=True,
        ),
        recommended_for=[AgentType.TABLE_DETECTOR, AgentType.STRUCTURE_ANALYZER, AgentType.PATTERN_FINDER],
        best_complexity=[TaskComplexity.SIMPLE, TaskComplexity.MODERATE],
        description="Most cost-efficient model optimized for high throughput and low latency",
    ),
    # Gemini 2.0 Flash models
    "gemini-2.0-flash": ModelInfo(
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        provider=ModelProvider.GOOGLE,
        capabilities=ModelCapabilities(
            reasoning_strength=8,
            speed=8,
            cost_efficiency=7,
            context_length=1048576,
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,
            multimodal=True,
        ),
        recommended_for=[AgentType.DATA_ANALYST, AgentType.GENERAL_PURPOSE, AgentType.COORDINATOR],
        best_complexity=[TaskComplexity.MODERATE, TaskComplexity.COMPLEX],
        description="Next-generation model with superior speed, native tool use, and streaming capabilities",
    ),
    "gemini-2.0-flash-lite": ModelInfo(
        model_id="gemini-2.0-flash-lite",
        display_name="Gemini 2.0 Flash-Lite",
        provider=ModelProvider.GOOGLE,
        capabilities=ModelCapabilities(
            reasoning_strength=7,
            speed=9,
            cost_efficiency=9,
            context_length=1048576,
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,
            multimodal=True,
        ),
        recommended_for=[AgentType.TABLE_DETECTOR, AgentType.STRUCTURE_ANALYZER],
        best_complexity=[TaskComplexity.SIMPLE, TaskComplexity.MODERATE],
        description="Cost-efficient and low-latency variant of Gemini 2.0 Flash",
    ),
    # Gemini 1.5 models (legacy but still useful)
    "gemini-1.5-pro": ModelInfo(
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        provider=ModelProvider.GOOGLE,
        capabilities=ModelCapabilities(
            reasoning_strength=8,
            speed=6,
            cost_efficiency=6,
            context_length=2097152,  # 2M tokens
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,
            multimodal=True,
        ),
        recommended_for=[AgentType.DATA_ANALYST, AgentType.FORMULA_ANALYZER],
        best_complexity=[TaskComplexity.COMPLEX],
        description="Mid-size multimodal model optimized for wide-range reasoning tasks with very long context",
        notes="Consider upgrading to Gemini 2.5 Pro for better performance. Deprecated Sep 2025.",
    ),
    "gemini-1.5-flash": ModelInfo(
        model_id="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        provider=ModelProvider.GOOGLE,
        capabilities=ModelCapabilities(
            reasoning_strength=7,
            speed=8,
            cost_efficiency=8,
            context_length=1048576,
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,
            multimodal=True,
        ),
        recommended_for=[AgentType.TABLE_DETECTOR, AgentType.STRUCTURE_ANALYZER, AgentType.GENERAL_PURPOSE],
        best_complexity=[TaskComplexity.MODERATE],
        description="Fast and versatile multimodal model for scaling across diverse tasks",
        notes="Consider upgrading to Gemini 2.5 Flash for better performance. Deprecated Sep 2025.",
    ),
    "gemini-1.5-flash-8b": ModelInfo(
        model_id="gemini-1.5-flash-8b",
        display_name="Gemini 1.5 Flash-8B",
        provider=ModelProvider.GOOGLE,
        capabilities=ModelCapabilities(
            reasoning_strength=6,
            speed=9,
            cost_efficiency=9,
            context_length=1048576,
            supports_tools=True,
            supports_vision=True,
            supports_audio=True,
            multimodal=True,
        ),
        recommended_for=[AgentType.TABLE_DETECTOR, AgentType.STRUCTURE_ANALYZER],
        best_complexity=[TaskComplexity.SIMPLE],
        description="Small model designed for lower intelligence, high-volume tasks",
        notes="Consider upgrading to Gemini 2.5 Flash-Lite. Deprecated Sep 2025.",
    ),
}
