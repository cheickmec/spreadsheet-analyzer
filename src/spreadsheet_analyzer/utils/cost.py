"""Cost tracking utilities for LLM usage."""

from structlog import get_logger

logger = get_logger(__name__)

# Token pricing (per 1M tokens) as of January 2025
ANTHROPIC_PRICING = {
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.8, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
}

OPENAI_PRICING = {
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
}


def get_token_pricing(model: str) -> tuple[float, float]:
    """Get input/output token pricing for a model.

    Returns:
        Tuple of (input_price_per_million, output_price_per_million)
    """
    # Check Anthropic models
    for model_key, prices in ANTHROPIC_PRICING.items():
        if model_key in model:
            return prices["input"], prices["output"]

    # Check OpenAI models
    for model_key, prices in OPENAI_PRICING.items():
        if model_key in model:
            return prices["input"], prices["output"]

    # Default/unknown model
    logger.warning("Unknown model for pricing", model=model)
    return 3.0, 15.0  # Default to Sonnet pricing


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate total cost for LLM usage.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Total cost in USD
    """
    input_price, output_price = get_token_pricing(model)

    # Calculate cost (prices are per million tokens)
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    total_cost = input_cost + output_cost

    logger.info(
        "LLM cost calculated",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost_usd=f"${total_cost:.4f}",
    )

    return total_cost


def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token average).

    CLAUDE-KNOWLEDGE: This is a rough estimate. Actual tokenization varies
    by model and can be 2-6 characters per token.
    """
    return len(text) // 4


def check_cost_limit(
    current_cost: float, cost_limit: float, estimated_additional_tokens: int, model: str
) -> tuple[bool, str]:
    """Check if we're within cost limits.

    Returns:
        Tuple of (within_limit, message)
    """
    _, output_price = get_token_pricing(model)
    estimated_additional_cost = (estimated_additional_tokens / 1_000_000) * output_price

    if current_cost + estimated_additional_cost > cost_limit:
        return False, f"Would exceed cost limit of ${cost_limit:.2f} (current: ${current_cost:.4f})"

    return True, f"Within cost limit (${current_cost:.4f} of ${cost_limit:.2f})"
