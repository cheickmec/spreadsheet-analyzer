"""Configuration for extended thinking capabilities.

This module provides configuration classes and utilities for managing
extended thinking parameters based on model capabilities and agent types.

CLAUDE-KNOWLEDGE: Extended thinking is available for Claude Opus 4.1/4,
Sonnet 4, and Sonnet 3.7. Budget allocation should be optimized per agent type.
"""

from dataclasses import dataclass
from enum import Enum

from .model_registry import AgentType, get_model_info


class ThinkingMode(Enum):
    """Extended thinking modes."""

    AUTO = "auto"  # Auto-enable for compatible models
    ENABLED = "enabled"  # Force enable (with validation)
    DISABLED = "disabled"  # Force disable


@dataclass(frozen=True)
class ThinkingConfig:
    """Configuration for extended thinking."""

    enabled: bool
    budget_tokens: int
    interleaved: bool = False  # For tool use with Claude 4 models

    @classmethod
    def create_for_agent(
        cls,
        model_id: str,
        agent_type: AgentType,
        mode: ThinkingMode = ThinkingMode.AUTO,
        budget_override: int | None = None,
    ) -> "ThinkingConfig":
        """Create thinking config optimized for specific agent type.

        Args:
            model_id: The model being used
            agent_type: Type of agent (for budget optimization)
            mode: Thinking mode (auto, enabled, disabled)
            budget_override: Override default budget allocation

        Returns:
            ThinkingConfig with appropriate settings
        """
        # Check if model supports thinking
        model_info = get_model_info(model_id)
        supports_thinking = model_info and _model_supports_thinking(model_id)

        # Determine if thinking should be enabled
        if mode == ThinkingMode.DISABLED:
            enabled = False
        elif mode == ThinkingMode.ENABLED:
            enabled = True
            if not supports_thinking:
                # Log warning but don't fail - graceful degradation
                print(f"⚠️  Warning: Model {model_id} does not support extended thinking")
                enabled = False
        else:  # AUTO mode
            enabled = supports_thinking

        # Determine budget based on agent type
        if budget_override:
            budget = budget_override
        else:
            budget = _get_default_budget(agent_type, model_id)

        # Enable interleaved thinking for Claude 4 models with tool use
        interleaved = enabled and _supports_interleaved_thinking(model_id) and _agent_uses_tools(agent_type)

        return cls(enabled=enabled, budget_tokens=budget, interleaved=interleaved)

    def to_api_params(self) -> dict:
        """Convert to API parameters for Anthropic calls."""
        if not self.enabled:
            return {}

        return {"thinking": {"type": "enabled", "budget_tokens": self.budget_tokens}}

    def to_beta_headers(self) -> dict:
        """Get beta headers needed for advanced features."""
        headers = {}

        if self.interleaved:
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        return headers


# Model capability detection
def _model_supports_thinking(model_id: str) -> bool:
    """Check if model supports extended thinking."""
    thinking_models = {
        "claude-opus-4-1-20250805",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
    }
    return model_id in thinking_models


def _supports_interleaved_thinking(model_id: str) -> bool:
    """Check if model supports interleaved thinking (Claude 4 only)."""
    claude_4_models = {"claude-opus-4-1-20250805", "claude-opus-4-20250514", "claude-sonnet-4-20250514"}
    return model_id in claude_4_models


def _agent_uses_tools(agent_type: AgentType) -> bool:
    """Check if agent type typically uses tools."""
    tool_using_agents = {AgentType.TABLE_DETECTOR, AgentType.DATA_ANALYST, AgentType.FORMULA_ANALYZER}
    return agent_type in tool_using_agents


# Agent-specific budget defaults
_AGENT_BUDGETS = {
    AgentType.TABLE_DETECTOR: 8000,  # Complex spatial reasoning
    AgentType.DATA_ANALYST: 16000,  # Deep analysis and patterns
    AgentType.FORMULA_ANALYZER: 16000,  # Multi-step formula logic
    AgentType.COORDINATOR: 12000,  # Workflow orchestration
    AgentType.STRUCTURE_ANALYZER: 10000,  # Structure analysis
    AgentType.PATTERN_FINDER: 10000,  # Pattern recognition
    AgentType.GENERAL_PURPOSE: 10000,  # Balanced default
}


def _get_default_budget(agent_type: AgentType, model_id: str) -> int:
    """Get default thinking budget for agent type and model."""
    base_budget = _AGENT_BUDGETS.get(agent_type, 10000)

    # Claude 4 models can handle higher budgets efficiently due to summarization
    if _supports_interleaved_thinking(model_id):
        # Increase budget by 50% for Claude 4 models
        return int(base_budget * 1.5)

    return base_budget


def get_thinking_summary(config: ThinkingConfig, model_id: str) -> str:
    """Get human-readable summary of thinking configuration."""
    if not config.enabled:
        return "❌ Extended thinking: Disabled"

    model_info = get_model_info(model_id)
    model_name = model_info.display_name if model_info else model_id

    parts = [f"🧠 Extended thinking: Enabled ({config.budget_tokens:,} tokens)"]

    if config.interleaved:
        parts.append("⚡ Interleaved thinking: Enabled")

    parts.append(f"🤖 Model: {model_name}")

    return " | ".join(parts)
