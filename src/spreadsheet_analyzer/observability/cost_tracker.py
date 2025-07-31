"""
Cost tracking integration with LiteLLM for accurate pricing.

This module provides cost tracking functionality using LiteLLM's
up-to-date pricing data, replacing the manual cost.py implementation.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from litellm import cost_per_token
from structlog import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsage:
    """Track token usage for a single LLM call."""

    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)

    # Cost information (calculated)
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_cost(self) -> None:
        """Calculate costs using LiteLLM pricing data."""
        try:
            # LiteLLM's cost_per_token returns cost per token (not per million)
            model_costs = cost_per_token(
                model=self.model, prompt_tokens=self.input_tokens, completion_tokens=self.output_tokens
            )

            self.input_cost = model_costs.get("prompt_cost", 0.0)
            self.output_cost = model_costs.get("completion_cost", 0.0)
            self.total_cost = self.input_cost + self.output_cost

        except Exception as e:
            logger.warning("Failed to calculate cost with LiteLLM", model=self.model, error=str(e))
            # Fallback to zero cost if calculation fails
            self.input_cost = 0.0
            self.output_cost = 0.0
            self.total_cost = 0.0


@dataclass
class CostTracker:
    """Track cumulative costs across multiple LLM calls."""

    # Tracking data
    usage_history: list[TokenUsage] = field(default_factory=list)

    # Cumulative totals
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    # Cost by model
    cost_by_model: dict[str, float] = field(default_factory=dict)
    tokens_by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    # Configuration
    cost_limit: float | None = None
    save_path: Path | None = None

    def track_usage(
        self, model: str, input_tokens: int, output_tokens: int, metadata: dict[str, Any] | None = None
    ) -> TokenUsage:
        """
        Track token usage for a single LLM call.

        Args:
            model: Model name (e.g., "gpt-4", "claude-3-opus")
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            metadata: Optional additional metadata

        Returns:
            TokenUsage object with calculated costs
        """
        usage = TokenUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            metadata=metadata or {},
        )

        # Calculate cost using LiteLLM
        usage.calculate_cost()

        # Update history
        self.usage_history.append(usage)

        # Update cumulative totals
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += usage.total_tokens
        self.total_cost += usage.total_cost

        # Update per-model tracking
        if model not in self.cost_by_model:
            self.cost_by_model[model] = 0.0
            self.tokens_by_model[model] = {"input": 0, "output": 0, "total": 0}

        self.cost_by_model[model] += usage.total_cost
        self.tokens_by_model[model]["input"] += input_tokens
        self.tokens_by_model[model]["output"] += output_tokens
        self.tokens_by_model[model]["total"] += usage.total_tokens

        # Log the usage
        logger.info(
            "Token usage tracked",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=f"${usage.total_cost:.4f}",
            total_cost_usd=f"${self.total_cost:.4f}",
        )

        # Check cost limit
        if self.cost_limit and self.total_cost > self.cost_limit:
            logger.warning("Cost limit exceeded", limit=f"${self.cost_limit:.2f}", current=f"${self.total_cost:.4f}")

        # Auto-save if configured
        if self.save_path:
            self.save_to_file(self.save_path)

        return usage

    def check_budget(self, estimated_tokens: int, model: str, is_output: bool = True) -> tuple[bool, str]:
        """
        Check if estimated usage would exceed budget.

        Args:
            estimated_tokens: Estimated number of tokens
            model: Model to use
            is_output: Whether tokens are output (True) or input (False)

        Returns:
            Tuple of (within_budget, message)
        """
        if not self.cost_limit:
            return True, "No cost limit set"

        try:
            # Estimate cost for the tokens
            if is_output:
                estimated_cost = cost_per_token(model=model, prompt_tokens=0, completion_tokens=estimated_tokens).get(
                    "completion_cost", 0.0
                )
            else:
                estimated_cost = cost_per_token(model=model, prompt_tokens=estimated_tokens, completion_tokens=0).get(
                    "prompt_cost", 0.0
                )

            projected_total = self.total_cost + estimated_cost

            if projected_total > self.cost_limit:
                return False, (
                    f"Would exceed cost limit of ${self.cost_limit:.2f} "
                    f"(current: ${self.total_cost:.4f}, "
                    f"estimated: ${estimated_cost:.4f})"
                )

            remaining = self.cost_limit - projected_total
            return True, (
                f"Within budget (${self.total_cost:.4f} of ${self.cost_limit:.2f}, ${remaining:.4f} remaining)"
            )

        except Exception as e:
            logger.error("Failed to check budget", error=str(e))
            return True, "Budget check failed, proceeding"

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all tracked usage."""
        return {
            "total_cost_usd": round(self.total_cost, 4),
            "total_tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_tokens,
            },
            "cost_by_model": {model: round(cost, 4) for model, cost in self.cost_by_model.items()},
            "tokens_by_model": self.tokens_by_model,
            "call_count": len(self.usage_history),
            "cost_limit": self.cost_limit,
            "within_budget": (self.total_cost <= self.cost_limit if self.cost_limit else True),
        }

    def save_to_file(self, path: Path) -> None:
        """Save tracking data to JSON file."""
        try:
            data = {
                "summary": self.get_summary(),
                "history": [
                    {
                        "timestamp": usage.timestamp.isoformat(),
                        "model": usage.model,
                        "tokens": {
                            "input": usage.input_tokens,
                            "output": usage.output_tokens,
                            "total": usage.total_tokens,
                        },
                        "cost": {"input": usage.input_cost, "output": usage.output_cost, "total": usage.total_cost},
                        "metadata": usage.metadata,
                    }
                    for usage in self.usage_history
                ],
            }

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error("Failed to save cost tracking data", error=str(e))

    @classmethod
    def load_from_file(cls, path: Path) -> "CostTracker":
        """Load tracking data from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)

            tracker = cls()

            # Reconstruct history
            for item in data.get("history", []):
                usage = TokenUsage(
                    model=item["model"],
                    input_tokens=item["tokens"]["input"],
                    output_tokens=item["tokens"]["output"],
                    total_tokens=item["tokens"]["total"],
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    input_cost=item["cost"]["input"],
                    output_cost=item["cost"]["output"],
                    total_cost=item["cost"]["total"],
                    metadata=item.get("metadata", {}),
                )
                tracker.usage_history.append(usage)

            # Restore summary data
            summary = data.get("summary", {})
            tracker.total_cost = summary.get("total_cost_usd", 0.0)
            tracker.cost_limit = summary.get("cost_limit")

            tokens = summary.get("total_tokens", {})
            tracker.total_input_tokens = tokens.get("input", 0)
            tracker.total_output_tokens = tokens.get("output", 0)
            tracker.total_tokens = tokens.get("total", 0)

            tracker.cost_by_model = summary.get("cost_by_model", {})
            tracker.tokens_by_model = summary.get("tokens_by_model", {})

            return tracker

        except Exception as e:
            logger.error("Failed to load cost tracking data", error=str(e))
            return cls()


# Global cost tracker instance
_cost_tracker: CostTracker | None = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def initialize_cost_tracker(cost_limit: float | None = None, save_path: Path | None = None) -> CostTracker:
    """
    Initialize the global cost tracker.

    Args:
        cost_limit: Optional spending limit in USD
        save_path: Optional path to save tracking data

    Returns:
        Initialized cost tracker
    """
    global _cost_tracker

    # Try to load existing data if save_path provided
    if save_path and save_path.exists():
        _cost_tracker = CostTracker.load_from_file(save_path)
        logger.info("Loaded existing cost tracking data", path=str(save_path))
    else:
        _cost_tracker = CostTracker()

    # Update configuration
    if cost_limit is not None:
        _cost_tracker.cost_limit = cost_limit
    if save_path is not None:
        _cost_tracker.save_path = save_path

    logger.info(
        "Cost tracker initialized",
        cost_limit=f"${cost_limit:.2f}" if cost_limit else "None",
        save_path=str(save_path) if save_path else "None",
    )

    return _cost_tracker
