"""Anthropic provider implementation for Claude models.

This module provides integration with Anthropic's Claude models including
Claude 3 Opus, Sonnet, and Haiku variants.
"""

import os
from typing import Any, Final

from spreadsheet_analyzer.notebook_llm.llm_providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
    Role,
)

# Model token limits
ANTHROPIC_MODEL_LIMITS: Final[dict[str, int]] = {
    # Claude 4 models (latest)
    "claude-opus-4-20250514": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-opus-4-0": 200000,
    "claude-sonnet-4-0": 200000,
    # Claude 3.7
    "claude-3-7-sonnet-20250219": 200000,
    "claude-3-7-sonnet-latest": 200000,
    # Claude 3.5 models
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-5-haiku-20241022": 200000,
    "claude-3-5-haiku-latest": 200000,
    # Claude 3 models
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    # Legacy models
    "claude-2.1": 100000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000,
}


class AnthropicProvider(LLMProvider):
    """Anthropic API provider for Claude models.

    Supports Claude 3 (Opus, Sonnet, Haiku) and Claude 2 models with
    automatic token counting and context window management.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-3-sonnet)
        """
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY environment variable.")

        super().__init__(api_key, model)
        self._client = None
        self._async_client = None

    @property
    def default_model(self) -> str:
        """Get default model for Anthropic."""
        return "claude-3-sonnet-20240229"

    @property
    def max_context_tokens(self) -> int:
        """Get maximum context window size for current model."""
        # Check exact model match first
        if self.model_name in ANTHROPIC_MODEL_LIMITS:
            return ANTHROPIC_MODEL_LIMITS[self.model_name]

        # Check for partial matches
        for model_key, limit in ANTHROPIC_MODEL_LIMITS.items():
            if model_key.split("-")[0] in self.model_name:  # Match "claude-3", "claude-2", etc.
                return limit

        # Default to Claude 2 limit
        return 100000

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise ImportError("Anthropic library not installed. Install with: uv add anthropic") from e

            self._client = Anthropic(api_key=self.api_key)
        return self._client

    async def _get_async_client(self):
        """Get or create async Anthropic client."""
        if self._async_client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError("Anthropic library not installed. Install with: uv add anthropic") from e

            self._async_client = AsyncAnthropic(api_key=self.api_key)
        return self._async_client

    def _prepare_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, str]]]:
        """Prepare messages for Anthropic API format.

        Anthropic expects system message separately and only user/assistant messages.

        Args:
            messages: List of Message objects

        Returns:
            Tuple of (system_message, conversation_messages)
        """
        system_message = None
        conversation = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Concatenate multiple system messages if present
                if system_message:
                    system_message += "\n\n" + msg.content
                else:
                    system_message = msg.content
            else:
                conversation.append(
                    {
                        "role": msg.role.value,
                        "content": msg.content,
                    }
                )

        return system_message, conversation

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using Anthropic API.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            LLM response with generated text and usage metadata
        """
        client = self._get_client()

        try:
            # Prepare messages
            system_message, conversation = self._prepare_messages(messages)

            # Prepare request parameters
            params = {
                "model": self.model_name,
                "messages": conversation,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,  # Anthropic requires max_tokens
            }

            if system_message:
                params["system"] = system_message

            if stop_sequences:
                params["stop_sequences"] = stop_sequences

            # Add any additional Anthropic-specific parameters
            params.update(kwargs)

            # Make API call
            response = client.messages.create(**params)

            # Extract response data
            content = response.content[0].text if response.content else ""

            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                metadata={
                    "id": response.id,
                    "stop_reason": response.stop_reason,
                },
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                error=f"Anthropic API error: {e}",
            )

    async def complete_async(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async version of complete using Anthropic API.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional Anthropic-specific parameters

        Returns:
            LLM response with generated text and usage metadata
        """
        client = await self._get_async_client()

        try:
            # Prepare messages
            system_message, conversation = self._prepare_messages(messages)

            # Prepare request parameters
            params = {
                "model": self.model_name,
                "messages": conversation,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,  # Anthropic requires max_tokens
            }

            if system_message:
                params["system"] = system_message

            if stop_sequences:
                params["stop_sequences"] = stop_sequences

            # Add any additional Anthropic-specific parameters
            params.update(kwargs)

            # Make API call
            response = await client.messages.create(**params)

            # Extract response data
            content = response.content[0].text if response.content else ""

            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                metadata={
                    "id": response.id,
                    "stop_reason": response.stop_reason,
                },
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                error=f"Anthropic API error: {e}",
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens for Anthropic models.

        Anthropic doesn't provide a public tokenizer, so we approximate
        based on character count. This is less accurate than tiktoken
        but sufficient for budget estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated number of tokens
        """
        # Anthropic's rough estimate: ~4 characters per token
        # This is conservative to avoid exceeding limits
        return max(1, len(text) // 3)
