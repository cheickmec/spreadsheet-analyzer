"""OpenAI provider implementation for LLM integration.

This module provides integration with OpenAI's GPT models including
GPT-4, GPT-4 Turbo, and GPT-3.5 Turbo.
"""

import os
from typing import Any, Final

from spreadsheet_analyzer.notebook_llm.llm_providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
)

# Model token limits
OPENAI_MODEL_LIMITS: Final[dict[str, int]] = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-1106-preview": 128000,  # GPT-4 Turbo
    "gpt-4-0125-preview": 128000,  # GPT-4 Turbo
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-1106": 16384,
}


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for GPT models.

    Supports GPT-4, GPT-4 Turbo, and GPT-3.5 Turbo models with
    automatic token counting and context window management.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        organization: str | None = None,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4)
            organization: Optional organization ID
        """
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        super().__init__(api_key, model)
        self.organization = organization
        self._client = None
        self._async_client = None
        self._encoder = None

    @property
    def default_model(self) -> str:
        """Get default model for OpenAI."""
        return "gpt-4"

    @property
    def max_context_tokens(self) -> int:
        """Get maximum context window size for current model."""
        # Check exact model match first
        if self.model_name in OPENAI_MODEL_LIMITS:
            return OPENAI_MODEL_LIMITS[self.model_name]

        # Check for partial matches (e.g., custom fine-tuned models)
        for model_key, limit in OPENAI_MODEL_LIMITS.items():
            if model_key in self.model_name:
                return limit

        # Default to GPT-4 base limit
        return 8192

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("OpenAI library not installed. Install with: uv add openai") from e

            self._client = OpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
        return self._client

    async def _get_async_client(self):
        """Get or create async OpenAI client."""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError("OpenAI library not installed. Install with: uv add openai") from e

            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
            )
        return self._async_client

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion using OpenAI API.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            LLM response with generated text and usage metadata
        """
        client = self._get_client()

        try:
            # Prepare request parameters
            params = {
                "model": self.model_name,
                "messages": self._messages_to_dict(messages),
                "temperature": temperature,
            }

            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            if stop_sequences:
                params["stop"] = stop_sequences

            # Add any additional OpenAI-specific parameters
            params.update(kwargs)

            # Make API call
            response = client.chat.completions.create(**params)

            # Extract response data
            choice = response.choices[0]
            usage = response.usage

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
                metadata={
                    "finish_reason": choice.finish_reason,
                    "id": response.id,
                },
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                error=f"OpenAI API error: {e}",
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
        """Async version of complete using OpenAI API.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional OpenAI-specific parameters

        Returns:
            LLM response with generated text and usage metadata
        """
        client = await self._get_async_client()

        try:
            # Prepare request parameters
            params = {
                "model": self.model_name,
                "messages": self._messages_to_dict(messages),
                "temperature": temperature,
            }

            if max_tokens is not None:
                params["max_tokens"] = max_tokens

            if stop_sequences:
                params["stop"] = stop_sequences

            # Add any additional OpenAI-specific parameters
            params.update(kwargs)

            # Make API call
            response = await client.chat.completions.create(**params)

            # Extract response data
            choice = response.choices[0]
            usage = response.usage

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
                metadata={
                    "finish_reason": choice.finish_reason,
                    "id": response.id,
                },
            )

        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                error=f"OpenAI API error: {e}",
            )

    def count_tokens(self, text: str) -> int:
        """Count tokens using OpenAI's tiktoken library.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if self._encoder is None:
            try:
                import tiktoken
            except ImportError as e:
                raise ImportError("tiktoken library not installed. Install with: uv add tiktoken") from e

            # Get encoding for the model
            try:
                self._encoder = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fall back to cl100k_base for newer models
                self._encoder = tiktoken.get_encoding("cl100k_base")

        return len(self._encoder.encode(text))
