"""Model interface abstractions, routing logic, and cost tracking.

This module provides the infrastructure for intelligent model selection,
cost optimization, and usage tracking across different LLM providers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Tiers of models based on capability and cost."""

    SMALL = "small"  # Fast, cheap models for simple tasks
    MEDIUM = "medium"  # Balanced models for general analysis
    LARGE = "large"  # Powerful models for complex reasoning
    SPECIALIZED = "specialized"  # Task-specific fine-tuned models


class ModelProvider(Enum):
    """Supported model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


class AnalysisComplexity(Enum):
    """Complexity levels for analysis tasks."""

    LOW = "low"  # Simple extraction, counting
    MEDIUM = "medium"  # Pattern recognition, basic analysis
    HIGH = "high"  # Complex reasoning, synthesis
    VERY_HIGH = "very_high"  # Multi-step reasoning, validation


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    provider: ModelProvider
    tier: ModelTier
    max_tokens: int
    cost_per_1k_tokens: float  # Input tokens
    cost_per_1k_completion_tokens: float  # Output tokens
    supports_functions: bool = False
    supports_vision: bool = False
    context_window: int = 4096
    metadata: dict[str, Any] = field(default_factory=dict)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.cost_per_1k_tokens
        output_cost = (output_tokens / 1000) * self.cost_per_1k_completion_tokens
        return input_cost + output_cost


@dataclass
class ModelUsage:
    """Track model usage and costs."""

    model_name: str
    task_type: str
    input_tokens: int
    output_tokens: int
    cost: float
    duration_seconds: float
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelInterface(Protocol):
    """Protocol for model implementations."""

    @property
    def config(self) -> ModelConfig:
        """Get model configuration."""
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.1,
        **kwargs,
    ) -> str:
        """Generate response from prompt."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        ...

    def supports_task(self, task_type: str) -> bool:
        """Check if model supports a specific task type."""
        ...


class BaseModel(ABC):
    """Abstract base class for model implementations."""

    def __init__(self, config: ModelConfig):
        """Initialize model with configuration."""
        self.config = config
        self._usage_history: list[ModelUsage] = []

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.1,
        **kwargs,
    ) -> str:
        """Generate response from prompt."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass

    def supports_task(self, task_type: str) -> bool:
        """Check if model supports a specific task type.

        Default implementation based on tier.
        """
        simple_tasks = {"extraction", "counting", "classification"}
        medium_tasks = {"pattern_recognition", "summarization", "analysis"}
        complex_tasks = {"reasoning", "synthesis", "validation"}

        if self.config.tier == ModelTier.SMALL:
            return task_type in simple_tasks
        elif self.config.tier == ModelTier.MEDIUM:
            return task_type in simple_tasks | medium_tasks
        elif self.config.tier == ModelTier.LARGE:
            return True  # Large models support all tasks
        else:  # SPECIALIZED
            # Check metadata for supported tasks
            return task_type in self.config.metadata.get("supported_tasks", [])

    def track_usage(
        self,
        task_type: str,
        input_tokens: int,
        output_tokens: int,
        duration_seconds: float,
        success: bool,
        **metadata,
    ) -> None:
        """Track model usage for cost analysis."""
        cost = self.config.estimate_cost(input_tokens, output_tokens)
        usage = ModelUsage(
            model_name=self.config.name,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            duration_seconds=duration_seconds,
            success=success,
            metadata=metadata,
        )
        self._usage_history.append(usage)

    def get_usage_summary(self) -> dict[str, Any]:
        """Get summary of model usage."""
        if not self._usage_history:
            return {
                "total_requests": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "success_rate": 0.0,
            }

        total_requests = len(self._usage_history)
        total_cost = sum(u.cost for u in self._usage_history)
        total_tokens = sum(u.input_tokens + u.output_tokens for u in self._usage_history)
        successful = sum(1 for u in self._usage_history if u.success)

        return {
            "total_requests": total_requests,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "success_rate": successful / total_requests if total_requests > 0 else 0.0,
            "cost_by_task": self._get_cost_by_task(),
        }

    def _get_cost_by_task(self) -> dict[str, float]:
        """Get cost breakdown by task type."""
        costs: dict[str, float] = {}
        for usage in self._usage_history:
            task = usage.task_type
            costs[task] = costs.get(task, 0.0) + usage.cost
        return costs


class ModelRouter:
    """Routes tasks to appropriate models based on complexity and cost."""

    def __init__(self):
        """Initialize the model router."""
        self.models: dict[str, BaseModel] = {}
        self.routing_rules: dict[str, list[str]] = {}
        self._setup_default_models()
        self._setup_routing_rules()

    def _setup_default_models(self) -> None:
        """Set up default model configurations."""
        # These are example configurations - actual values would come from config
        self.model_configs = {
            # Small models
            "gpt-3.5-turbo": ModelConfig(
                name="gpt-3.5-turbo",
                provider=ModelProvider.OPENAI,
                tier=ModelTier.SMALL,
                max_tokens=4096,
                cost_per_1k_tokens=0.0005,
                cost_per_1k_completion_tokens=0.0015,
                supports_functions=True,
                context_window=16384,
            ),
            # Medium models
            "claude-3-haiku": ModelConfig(
                name="claude-3-haiku",
                provider=ModelProvider.ANTHROPIC,
                tier=ModelTier.MEDIUM,
                max_tokens=4096,
                cost_per_1k_tokens=0.00025,
                cost_per_1k_completion_tokens=0.00125,
                supports_functions=True,
                context_window=200000,
            ),
            # Large models
            "gpt-4-turbo": ModelConfig(
                name="gpt-4-turbo",
                provider=ModelProvider.OPENAI,
                tier=ModelTier.LARGE,
                max_tokens=4096,
                cost_per_1k_tokens=0.01,
                cost_per_1k_completion_tokens=0.03,
                supports_functions=True,
                supports_vision=True,
                context_window=128000,
            ),
            "claude-3-opus": ModelConfig(
                name="claude-3-opus",
                provider=ModelProvider.ANTHROPIC,
                tier=ModelTier.LARGE,
                max_tokens=4096,
                cost_per_1k_tokens=0.015,
                cost_per_1k_completion_tokens=0.075,
                supports_functions=True,
                supports_vision=True,
                context_window=200000,
            ),
        }

    def _setup_routing_rules(self) -> None:
        """Set up default routing rules."""
        self.routing_rules = {
            # Simple tasks go to small models
            "extraction": ["gpt-3.5-turbo", "claude-3-haiku"],
            "counting": ["gpt-3.5-turbo"],
            "classification": ["gpt-3.5-turbo", "claude-3-haiku"],
            # Medium complexity tasks
            "pattern_recognition": ["claude-3-haiku", "gpt-4-turbo"],
            "summarization": ["claude-3-haiku", "gpt-3.5-turbo"],
            "sheet_analysis": ["claude-3-haiku", "gpt-4-turbo"],
            # Complex tasks need large models
            "reasoning": ["gpt-4-turbo", "claude-3-opus"],
            "synthesis": ["claude-3-opus", "gpt-4-turbo"],
            "validation": ["gpt-4-turbo", "claude-3-opus"],
            "relationship_analysis": ["claude-3-opus", "gpt-4-turbo"],
        }

    def register_model(self, model: BaseModel) -> None:
        """Register a model with the router."""
        self.models[model.config.name] = model
        logger.info(f"Registered model: {model.config.name}")

    def select_model(
        self,
        task_type: str,
        complexity: AnalysisComplexity,
        tier_preference: ModelTier | None = None,
        max_cost_per_1k: float | None = None,
    ) -> BaseModel:
        """Select appropriate model for the task.

        Args:
            task_type: Type of task to perform
            complexity: Complexity level of the task
            tier_preference: Preferred model tier
            max_cost_per_1k: Maximum cost per 1000 tokens

        Returns:
            Selected model instance
        """
        # Get candidate models from routing rules
        candidates = self.routing_rules.get(task_type, list(self.models.keys()))

        # Filter by tier preference if specified
        if tier_preference:
            candidates = [
                name
                for name in candidates
                if name in self.model_configs and self.model_configs[name].tier == tier_preference
            ]

        # Filter by cost if specified
        if max_cost_per_1k:
            candidates = [
                name
                for name in candidates
                if name in self.model_configs and self.model_configs[name].cost_per_1k_tokens <= max_cost_per_1k
            ]

        # Adjust for complexity
        if complexity in [AnalysisComplexity.HIGH, AnalysisComplexity.VERY_HIGH]:
            # Prefer large models for complex tasks
            large_models = [
                name
                for name in candidates
                if name in self.model_configs and self.model_configs[name].tier == ModelTier.LARGE
            ]
            if large_models:
                candidates = large_models

        # Select first available candidate
        for model_name in candidates:
            if model_name in self.models:
                logger.info(f"Selected model {model_name} for task {task_type}")
                return self.models[model_name]

        # Fallback to any available model
        if self.models:
            model_name = next(iter(self.models))
            logger.warning(f"No ideal model found for {task_type}, falling back to {model_name}")
            return self.models[model_name]

        raise ValueError(f"No models available for task {task_type}")

    def estimate_task_cost(
        self,
        task_type: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        complexity: AnalysisComplexity = AnalysisComplexity.MEDIUM,
    ) -> dict[str, float]:
        """Estimate cost for a task across different model options.

        Returns:
            Dictionary mapping model names to estimated costs
        """
        candidates = self.routing_rules.get(task_type, list(self.model_configs.keys()))
        estimates = {}

        for model_name in candidates:
            if model_name in self.model_configs:
                config = self.model_configs[model_name]
                cost = config.estimate_cost(estimated_input_tokens, estimated_output_tokens)
                estimates[model_name] = cost

        return estimates

    def get_usage_report(self) -> dict[str, Any]:
        """Get usage report across all models."""
        report = {
            "total_cost": 0.0,
            "total_requests": 0,
            "by_model": {},
            "by_task_type": {},
        }

        for model_name, model in self.models.items():
            summary = model.get_usage_summary()
            report["by_model"][model_name] = summary
            report["total_cost"] += summary["total_cost"]
            report["total_requests"] += summary["total_requests"]

            # Aggregate by task type
            for task, cost in summary.get("cost_by_task", {}).items():
                if task not in report["by_task_type"]:
                    report["by_task_type"][task] = {"cost": 0.0, "count": 0}
                report["by_task_type"][task]["cost"] += cost
                report["by_task_type"][task]["count"] += 1

        return report


class CostController:
    """Controls and monitors costs across the analysis."""

    def __init__(self, budget_limit: float):
        """Initialize cost controller with budget limit.

        Args:
            budget_limit: Maximum allowed cost in dollars
        """
        self.budget_limit = budget_limit
        self.spent = 0.0
        self.reservations: dict[str, float] = {}

    def check_budget(self, estimated_cost: float) -> bool:
        """Check if estimated cost fits within budget."""
        return (self.spent + estimated_cost) <= self.budget_limit

    def reserve_budget(self, task_id: str, amount: float) -> bool:
        """Reserve budget for a task.

        Returns:
            True if reservation successful, False otherwise
        """
        if not self.check_budget(amount):
            return False

        self.reservations[task_id] = amount
        return True

    def commit_cost(self, task_id: str, actual_cost: float) -> None:
        """Commit actual cost for a task."""
        # Remove reservation if exists
        reserved = self.reservations.pop(task_id, 0)

        # Add actual cost
        self.spent += actual_cost

        # Log if significantly different from reservation
        if reserved > 0 and abs(actual_cost - reserved) / reserved > 0.2:
            logger.warning(
                f"Task {task_id} cost ${actual_cost:.4f} "
                f"(reserved ${reserved:.4f}, {((actual_cost / reserved - 1) * 100):.1f}% difference)"
            )

    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        reserved_total = sum(self.reservations.values())
        return self.budget_limit - self.spent - reserved_total

    def get_budget_summary(self) -> dict[str, float]:
        """Get budget summary."""
        reserved_total = sum(self.reservations.values())
        return {
            "limit": self.budget_limit,
            "spent": self.spent,
            "reserved": reserved_total,
            "available": self.get_remaining_budget(),
            "utilization": (self.spent / self.budget_limit) if self.budget_limit > 0 else 0.0,
        }
