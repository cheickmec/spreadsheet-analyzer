"""Functional token budget management for context optimization.

This module provides pure functions for managing token budgets
across different LLM models and use cases.

CLAUDE-KNOWLEDGE: Token counting varies by model and tokenizer.
Approximations are used here, but production systems should use
model-specific tokenizers for accuracy.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..core.types import Result, err, ok


@dataclass(frozen=True)
class TokenBudget:
    """Immutable token budget allocation."""

    total: int
    system_prompt: int = 0
    task_prompt: int = 0
    context: int = 0
    response_reserve: int = 0
    safety_margin: int = 0

    @property
    def available_for_context(self) -> int:
        """Calculate tokens available for context."""
        used = self.system_prompt + self.task_prompt + self.response_reserve + self.safety_margin
        return max(0, self.total - used)

    @property
    def total_allocated(self) -> int:
        """Total tokens allocated."""
        return self.system_prompt + self.task_prompt + self.context + self.response_reserve + self.safety_margin

    @property
    def is_valid(self) -> bool:
        """Check if budget allocation is valid."""
        return self.total_allocated <= self.total

    def with_context(self, context_tokens: int) -> "TokenBudget":
        """Create new budget with updated context allocation."""
        from dataclasses import replace

        return replace(self, context=context_tokens)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    token_limit: int
    pricing_per_1k_input: float = 0.0
    pricing_per_1k_output: float = 0.0
    supports_json_mode: bool = False
    supports_tools: bool = False

    @property
    def cost_estimate(self) -> Callable[[int, int], float]:
        """Get cost estimation function."""

        def estimate(input_tokens: int, output_tokens: int) -> float:
            input_cost = (input_tokens / 1000) * self.pricing_per_1k_input
            output_cost = (output_tokens / 1000) * self.pricing_per_1k_output
            return input_cost + output_cost

        return estimate


# Model configurations
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4": ModelConfig(
        name="gpt-4",
        token_limit=8192,
        pricing_per_1k_input=0.03,
        pricing_per_1k_output=0.06,
        supports_json_mode=True,
        supports_tools=True,
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        token_limit=128000,
        pricing_per_1k_input=0.01,
        pricing_per_1k_output=0.03,
        supports_json_mode=True,
        supports_tools=True,
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        token_limit=4096,
        pricing_per_1k_input=0.0005,
        pricing_per_1k_output=0.0015,
        supports_json_mode=True,
        supports_tools=True,
    ),
    # Claude models
    "claude-3-opus": ModelConfig(
        name="claude-3-opus",
        token_limit=200000,
        pricing_per_1k_input=0.015,
        pricing_per_1k_output=0.075,
        supports_json_mode=False,
        supports_tools=True,
    ),
    "claude-3-sonnet": ModelConfig(
        name="claude-3-sonnet",
        token_limit=200000,
        pricing_per_1k_input=0.003,
        pricing_per_1k_output=0.015,
        supports_json_mode=False,
        supports_tools=True,
    ),
    "claude-3-haiku": ModelConfig(
        name="claude-3-haiku",
        token_limit=200000,
        pricing_per_1k_input=0.00025,
        pricing_per_1k_output=0.00125,
        supports_json_mode=False,
        supports_tools=True,
    ),
}


@dataclass(frozen=True)
class AllocationStrategy:
    """Strategy for allocating token budget."""

    name: str
    system_ratio: float = 0.10
    task_ratio: float = 0.05
    response_ratio: float = 0.30
    safety_ratio: float = 0.05

    @property
    def context_ratio(self) -> float:
        """Calculate context ratio from other allocations."""
        return 1.0 - (self.system_ratio + self.task_ratio + self.response_ratio + self.safety_ratio)

    def is_valid(self) -> bool:
        """Check if ratios sum to <= 1.0."""
        total = self.system_ratio + self.task_ratio + self.response_ratio + self.safety_ratio
        return 0 < total <= 1.0


# Predefined allocation strategies
ALLOCATION_STRATEGIES = {
    "standard": AllocationStrategy(
        name="standard", system_ratio=0.10, task_ratio=0.05, response_ratio=0.30, safety_ratio=0.05
    ),
    "analysis_heavy": AllocationStrategy(
        name="analysis_heavy", system_ratio=0.08, task_ratio=0.07, response_ratio=0.40, safety_ratio=0.05
    ),
    "context_heavy": AllocationStrategy(
        name="context_heavy", system_ratio=0.05, task_ratio=0.05, response_ratio=0.20, safety_ratio=0.05
    ),
    "minimal": AllocationStrategy(
        name="minimal", system_ratio=0.05, task_ratio=0.03, response_ratio=0.15, safety_ratio=0.02
    ),
}


def get_model_config(model_name: str) -> Result[ModelConfig, str]:
    """Get configuration for a model.

    Args:
        model_name: Name of the model

    Returns:
        Model configuration or error
    """
    # Direct lookup
    if model_name in MODEL_CONFIGS:
        return ok(MODEL_CONFIGS[model_name])

    # Fuzzy matching for model variants
    model_lower = model_name.lower()
    for key, config in MODEL_CONFIGS.items():
        if key in model_lower or model_lower in key:
            return ok(config)

    # Default for unknown models
    return ok(
        ModelConfig(
            name=model_name,
            token_limit=4096,  # Conservative default
            supports_json_mode=False,
            supports_tools=False,
        )
    )


def allocate_budget(total_tokens: int, strategy: AllocationStrategy | str = "standard") -> Result[TokenBudget, str]:
    """Allocate token budget using a strategy.

    Args:
        total_tokens: Total available tokens
        strategy: Allocation strategy or strategy name

    Returns:
        Allocated token budget or error
    """
    if isinstance(strategy, str):
        if strategy not in ALLOCATION_STRATEGIES:
            return err(f"Unknown allocation strategy: {strategy}")
        strategy = ALLOCATION_STRATEGIES[strategy]

    if not strategy.is_valid():
        return err("Invalid allocation strategy ratios")

    budget = TokenBudget(
        total=total_tokens,
        system_prompt=int(total_tokens * strategy.system_ratio),
        task_prompt=int(total_tokens * strategy.task_ratio),
        response_reserve=int(total_tokens * strategy.response_ratio),
        safety_margin=int(total_tokens * strategy.safety_ratio),
        context=int(total_tokens * strategy.context_ratio),
    )

    return ok(budget)


def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for text.

    This is a simple approximation. Production systems should
    use model-specific tokenizers.

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer selection

    Returns:
        Estimated token count
    """
    # Simple approximation: ~4 characters per token
    # This varies significantly by model and content

    if "claude" in model.lower():
        # Claude tends to have slightly different tokenization
        return len(text) // 3
    else:
        # GPT models
        return len(text) // 4


def calculate_context_tokens(cells: list[dict[str, Any]], metadata: dict[str, Any], model: str = "gpt-4") -> int:
    """Calculate tokens for context package.

    Args:
        cells: List of cell dictionaries
        metadata: Metadata dictionary
        model: Model for token estimation

    Returns:
        Total estimated tokens
    """
    total = 0

    # Count cell tokens
    for cell in cells:
        cell_text = f"{cell.get('location', '')}: {cell.get('content', '')}"
        if cell.get("metadata"):
            cell_text += f" {cell['metadata']}"
        total += estimate_tokens(cell_text, model)

    # Count metadata tokens
    if metadata:
        total += estimate_tokens(str(metadata), model)

    return total


def optimize_for_budget(
    required_tokens: int, available_budget: TokenBudget, priority_order: list[str] = None
) -> Result[TokenBudget, str]:
    """Optimize budget allocation for required tokens.

    Adjusts allocations to fit required context tokens while
    maintaining minimum requirements for other components.

    Args:
        required_tokens: Tokens required for context
        available_budget: Current budget allocation
        priority_order: Order to reduce allocations

    Returns:
        Optimized budget or error
    """
    if priority_order is None:
        priority_order = ["safety_margin", "task_prompt", "system_prompt", "response_reserve"]

    if required_tokens > available_budget.total * 0.9:
        return err("Required tokens exceed 90% of total budget")

    # Start with current allocations
    new_budget = available_budget

    # If we already have enough space, just update context
    if required_tokens <= new_budget.available_for_context:
        return ok(new_budget.with_context(required_tokens))

    # Need to reduce other allocations
    needed = required_tokens - new_budget.available_for_context

    # Minimum allocations (5% each for critical components)
    min_allocations = {
        "system_prompt": int(available_budget.total * 0.05),
        "task_prompt": int(available_budget.total * 0.03),
        "response_reserve": int(available_budget.total * 0.10),
        "safety_margin": int(available_budget.total * 0.02),
    }

    # Try to reduce allocations in priority order
    reductions = {}
    for component in priority_order:
        if needed <= 0:
            break

        current = getattr(new_budget, component)
        min_required = min_allocations.get(component, 0)
        available_reduction = max(0, current - min_required)

        if available_reduction > 0:
            reduction = min(needed, available_reduction)
            reductions[component] = current - reduction
            needed -= reduction

    if needed > 0:
        return err(f"Cannot fit required tokens. Need {needed} more tokens.")

    # Apply reductions
    from dataclasses import replace

    new_budget = replace(new_budget, **reductions, context=required_tokens)

    return ok(new_budget)


def get_compression_target(current_tokens: int, budget: TokenBudget) -> float:
    """Calculate target compression ratio.

    Args:
        current_tokens: Current context token count
        budget: Token budget

    Returns:
        Target compression ratio (0.0 = no compression, 1.0 = complete compression)
    """
    if current_tokens <= budget.context:
        return 0.0  # No compression needed

    target_reduction = current_tokens - budget.context
    return min(target_reduction / current_tokens, 0.95)  # Cap at 95% compression
