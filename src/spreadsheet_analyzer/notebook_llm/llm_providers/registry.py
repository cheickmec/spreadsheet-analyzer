"""Registry for LLM providers with dynamic discovery and configuration."""

import logging
from typing import Any

from spreadsheet_analyzer.notebook_llm.llm_providers.anthropic_provider import (
    AnthropicProvider,
)
from spreadsheet_analyzer.notebook_llm.llm_providers.base import LLMInterface, LLMProvider
from spreadsheet_analyzer.notebook_llm.llm_providers.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)

# Built-in provider mappings
PROVIDERS: dict[str, type[LLMProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def register_provider(name: str, provider_class: type[LLMProvider]) -> None:
    """Register a custom LLM provider.

    Args:
        name: Name to register the provider under
        provider_class: Provider class implementing LLMProvider
    """
    if not issubclass(provider_class, LLMProvider):
        raise TypeError(f"Provider class must inherit from LLMProvider, got {provider_class}")

    PROVIDERS[name] = provider_class
    logger.info("Registered LLM provider: %s", name)


def list_providers() -> list[str]:
    """List all available provider names.

    Returns:
        List of registered provider names
    """
    return list(PROVIDERS.keys())


def get_provider(
    provider_name: str,
    api_key: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> LLMInterface:
    """Get an LLM provider instance by name.

    Args:
        provider_name: Name of the provider (e.g., "openai", "anthropic")
        api_key: Optional API key (will use environment variable if not provided)
        model: Optional model name (will use provider default if not provided)
        **kwargs: Additional provider-specific configuration

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider name is not recognized
    """
    if provider_name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{provider_name}'. Available providers: {available}")

    provider_class = PROVIDERS[provider_name]

    try:
        return provider_class(api_key=api_key, model=model, **kwargs)
    except Exception as e:
        logger.exception("Failed to initialize %s provider", provider_name)
        raise RuntimeError(f"Failed to initialize {provider_name} provider: {e}") from e
