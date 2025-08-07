"""Model registry with recommendations for different agent types.

This module provides a comprehensive registry of available LLM models
with specific recommendations for different agent types and use cases.

CLAUDE-KNOWLEDGE: Based on OpenAI and Anthropic guidelines, model selection
should optimize for accuracy first, then cost and latency. Different agent
types have different requirements.
"""

# Import from individual provider modules
from .anthropic import CLAUDE_MODELS
from .base import AgentType, ModelInfo, ModelProvider, TaskComplexity
from .google import GOOGLE_MODELS
from .openai import OPENAI_MODELS

# Combined model registry
ALL_MODELS = {**CLAUDE_MODELS, **OPENAI_MODELS, **GOOGLE_MODELS}


def get_available_models() -> dict[str, ModelInfo]:
    """Get all available models."""
    return ALL_MODELS


def get_models_by_provider(provider: ModelProvider) -> dict[str, ModelInfo]:
    """Get models from a specific provider."""
    return {model_id: model_info for model_id, model_info in ALL_MODELS.items() if model_info.provider == provider}


def get_recommended_models_for_agent(agent_type: AgentType) -> list[ModelInfo]:
    """Get recommended models for a specific agent type."""
    return [
        model_info
        for model_info in ALL_MODELS.values()
        if agent_type in model_info.recommended_for and not model_info.deprecated
    ]


def get_models_by_complexity(complexity: TaskComplexity) -> list[ModelInfo]:
    """Get models suitable for a specific complexity level."""
    return [
        model_info
        for model_info in ALL_MODELS.values()
        if complexity in model_info.best_complexity and not model_info.deprecated
    ]


def get_model_info(model_id: str) -> ModelInfo | None:
    """Get information for a specific model."""
    return ALL_MODELS.get(model_id)


def validate_model(model_id: str) -> bool:
    """Check if a model ID is valid and not deprecated."""
    model_info = get_model_info(model_id)
    return model_info is not None and not model_info.deprecated


def format_model_list() -> str:
    """Format a human-readable list of available models organized by provider and purpose."""
    output = []

    # Group by provider
    for provider in ModelProvider:
        provider_models = get_models_by_provider(provider)
        if not provider_models:
            continue

        output.append(f"\n🏢 {provider.value.upper()} MODELS")
        output.append("=" * 50)

        # Group by purpose/agent type
        reasoning_models = []
        flagship_models = []
        cost_optimized = []
        legacy_models = []

        for model_info in provider_models.values():
            if model_info.deprecated:
                legacy_models.append(model_info)
            elif model_info.capabilities.reasoning_strength >= 9:
                reasoning_models.append(model_info)
            elif model_info.capabilities.cost_efficiency >= 8:
                cost_optimized.append(model_info)
            else:
                flagship_models.append(model_info)

        # Output sections
        if reasoning_models:
            output.append("\n🧠 REASONING MODELS (Best for complex analysis):")
            for model in reasoning_models:
                agent_types = ", ".join([at.value for at in model.recommended_for])
                output.append(f"  • {model.model_id}")
                output.append(f"    {model.description}")
                output.append(f"    Recommended for: {agent_types}")
                if model.notes:
                    output.append(f"    Note: {model.notes}")

        if flagship_models:
            output.append("\n🚀 FLAGSHIP MODELS (Balanced performance):")
            for model in flagship_models:
                agent_types = ", ".join([at.value for at in model.recommended_for])
                output.append(f"  • {model.model_id}")
                output.append(f"    {model.description}")
                output.append(f"    Recommended for: {agent_types}")

        if cost_optimized:
            output.append("\n💰 COST-OPTIMIZED MODELS (Fast and efficient):")
            for model in cost_optimized:
                agent_types = ", ".join([at.value for at in model.recommended_for])
                output.append(f"  • {model.model_id}")
                output.append(f"    {model.description}")
                output.append(f"    Recommended for: {agent_types}")

        if legacy_models:
            output.append("\n⚠️  LEGACY MODELS (Consider upgrading):")
            for model in legacy_models:
                output.append(f"  • {model.model_id} - {model.notes or 'Deprecated'}")

    output.append("\n" + "=" * 70)
    output.append("🎯 AGENT TYPE RECOMMENDATIONS:")
    output.append("=" * 70)

    for agent_type in AgentType:
        recommended = get_recommended_models_for_agent(agent_type)
        if recommended:
            output.append(f"\n{agent_type.value}:")
            # Sort by reasoning strength and cost efficiency
            recommended.sort(
                key=lambda x: (x.capabilities.reasoning_strength, x.capabilities.cost_efficiency), reverse=True
            )
            for model in recommended[:3]:  # Top 3 recommendations
                output.append(f"  ✓ {model.model_id} ({model.provider.value})")

    return "\n".join(output)


# Agent-specific recommendations for common use cases
AGENT_RECOMMENDATIONS = {
    AgentType.TABLE_DETECTOR: {
        "recommended": "claude-3-5-haiku-20241022",
        "alternatives": ["gpt-4.1-mini", "gpt-4o-mini"],
        "reason": "Fast pattern detection and structure analysis, cost-efficient for repetitive tasks",
    },
    AgentType.DATA_ANALYST: {
        "recommended": "claude-3-5-sonnet-20241022",
        "alternatives": ["o4-mini", "gpt-4.1"],
        "reason": "Balanced reasoning and speed for comprehensive data analysis",
    },
    AgentType.FORMULA_ANALYZER: {
        "recommended": "o3",
        "alternatives": ["claude-sonnet-4-20250514", "o3-pro"],
        "reason": "Strong reasoning capabilities for complex formula validation and optimization",
    },
    AgentType.COORDINATOR: {
        "recommended": "claude-3-7-sonnet-20250219",
        "alternatives": ["gpt-4.1", "claude-3-5-sonnet-20241022"],
        "reason": "Excellent at orchestrating multiple agents and synthesizing results",
    },
}


def get_recommendation_for_agent(agent_type: AgentType) -> dict[str, any] | None:
    """Get the top recommendation for a specific agent type."""
    return AGENT_RECOMMENDATIONS.get(agent_type)
