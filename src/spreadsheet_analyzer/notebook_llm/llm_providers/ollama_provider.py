"""Ollama provider implementation for local LLM models.

This module provides integration with Ollama for running open-source models
locally including Llama, Mistral, Mixtral, and other supported models.
"""

import asyncio
import os
from typing import Any, Final

import httpx
import tiktoken

from spreadsheet_analyzer.notebook_llm.llm_providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
)

# Model context window limits (approximate, as Ollama models can vary)
OLLAMA_MODEL_LIMITS: Final[dict[str, int]] = {
    # Mistral models
    "mistral": 32768,
    "mistral:7b": 32768,
    "mistral:7b-instruct": 32768,
    "mistral:7b-instruct-q6_K": 32768,
    "mistral:7b-instruct-q5_K_S": 32768,
    "mistral:7b-instruct-q4_K_M": 32768,
    "mistral:v0.3": 32768,  # v0.3 supports tool calling
    "mistral-nemo": 128000,
    "mistral-small": 32768,
    # Llama models
    "llama3.2": 131072,
    "llama3.2:1b": 131072,
    "llama3.2:3b": 131072,
    "llama3.1": 131072,
    "llama3.1:8b": 131072,
    "llama3.1:70b": 131072,
    "llama3.3": 131072,
    "llama3": 8192,
    "llama3:8b": 8192,
    "llama3:70b": 8192,
    "llama2": 4096,
    "llama2:7b": 4096,
    "llama2:13b": 4096,
    "llama2:70b": 4096,
    # Mixtral models
    "mixtral": 47000,
    "mixtral:8x7b": 47000,
    "mixtral:8x22b": 65536,
    # CodeLlama models
    "codellama": 16384,
    "codellama:7b": 16384,
    "codellama:13b": 16384,
    "codellama:34b": 16384,
    "codellama:70b": 16384,
    # DeepSeek models
    "deepseek-r1": 131072,
    "deepseek-r1:8b": 131072,
    "deepseek-r1:14b": 131072,
    "deepseek-r1:32b": 131072,
    "deepseek-r1:70b": 131072,
    "deepseek-coder-v2": 128000,
    "MFDoom/deepseek-r1-tool-calling": 131072,
    "MFDoom/deepseek-r1-tool-calling:32b": 131072,
    # Qwen models
    "qwen2.5": 131072,
    "qwen2.5:7b": 131072,
    "qwen2.5-coder": 131072,
    "qwen2": 131072,
    "qwen3": 32768,
    # Other models
    "phi3": 128000,
    "phi3:mini": 128000,
    "phi3:medium": 128000,
    "phi4": 128000,
    "gemma2": 8192,
    "gemma2:2b": 8192,
    "gemma2:9b": 8192,
    "gemma2:27b": 8192,
    "command-r": 128000,
    "command-r-plus": 128000,
}

# Models that support tool/function calling
# Based on Ollama's tool support documentation
TOOL_CALLING_MODELS: Final[set[str]] = {
    # Llama models
    "llama3.1",
    "llama3.1:8b",
    "llama3.1:70b",
    "llama3.1:405b",
    "llama3.2",
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.3",
    # Mistral models
    "mistral:v0.3",
    "mistral-nemo",
    "mistral-small",
    "mistral-small3.1",
    "mistral-small3.2",
    # Qwen models
    "qwen2",
    "qwen2.5",
    "qwen2.5:7b",
    "qwen2.5-coder",
    "qwen3",
    # DeepSeek models (custom versions with tool support)
    "MFDoom/deepseek-r1-tool-calling",
    "MFDoom/deepseek-r1-tool-calling:7b",
    "MFDoom/deepseek-r1-tool-calling:32b",
    # Other models
    "command-r",
    "command-r-plus",
    "command-r7b",
    "firefunction-v2",
    "granite3-dense",
    "granite3.1-dense",
    "granite3.2",
    "nemotron",
    "nemotron-mini",
    "hermes3",
    "phi4",
    "phi4-mini",
}


class OllamaProvider(LLMProvider):
    """Ollama API provider for local LLM models.

    Supports various open-source models through Ollama including
    Llama, Mistral, Mixtral, CodeLlama, and others.
    """

    def __init__(
        self,
        api_key: str | None = None,  # Not used for Ollama but kept for interface compatibility
        model: str = "llama3.1:8b",
        base_url: str | None = None,
    ):
        """Initialize Ollama provider.

        Args:
            api_key: Not used for Ollama (kept for interface compatibility)
            model: Model to use (default: llama3.1:8b)
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        super().__init__(api_key, model)
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._client = None
        self._async_client = None
        # Initialize tokenizer for approximate token counting
        self._tokenizer = None

    @property
    def default_model(self) -> str:
        """Get default model for Ollama."""
        return "llama3.1:8b"

    @property
    def max_context_tokens(self) -> int:
        """Get maximum context window size for current model."""
        # Remove version tags for lookup
        base_model = self.model_name.split(":")[0]

        # Check exact match first
        if self.model_name in OLLAMA_MODEL_LIMITS:
            return OLLAMA_MODEL_LIMITS[self.model_name]

        # Check base model
        if base_model in OLLAMA_MODEL_LIMITS:
            return OLLAMA_MODEL_LIMITS[base_model]

        # Check for partial matches
        for model_key, limit in OLLAMA_MODEL_LIMITS.items():
            if model_key in self.model_name or base_model in model_key:
                return limit

        # Default to conservative 4k context
        return 4096

    def _get_tokenizer(self):
        """Get or create tokenizer for token counting."""
        if self._tokenizer is None:
            # Use cl100k_base as a reasonable approximation for most models
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Uses tiktoken for approximate counting as Ollama doesn't provide
        a native token counting API.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate number of tokens
        """
        tokenizer = self._get_tokenizer()
        return len(tokenizer.encode(text))

    def supports_tool_calling(self) -> bool:
        """Check if the current model supports tool/function calling.

        Returns:
            True if the model supports tool calling, False otherwise
        """
        # Check exact model name first
        if self.model_name in TOOL_CALLING_MODELS:
            return True

        # Check base model name (without quantization suffix)
        base_model = self.model_name.split(":")[0]
        if base_model in TOOL_CALLING_MODELS:
            # Special case: base mistral needs v0.3 tag for tool support
            if base_model == "mistral" and ":v0.3" not in self.model_name:
                return False
            return True

        return False

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Prepare messages for Ollama API format.

        Args:
            messages: List of Message objects

        Returns:
            List of message dictionaries in Ollama format
        """
        ollama_messages = []

        for msg in messages:
            ollama_messages.append(
                {
                    "role": msg.role.value,
                    "content": msg.content,
                }
            )

        return ollama_messages

    def _make_request(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        stop_sequences: list[str] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make synchronous request to Ollama API.

        Args:
            messages: Formatted messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional parameters

        Returns:
            API response dictionary
        """
        if self._client is None:
            self._client = httpx.Client(timeout=300.0)  # 5 minute timeout for local models

        url = f"{self.base_url}/api/chat"

        # Build request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        # Add any additional options
        if "options" in kwargs:
            payload["options"].update(kwargs["options"])

        # Make request
        response = self._client.post(url, json=payload)
        response.raise_for_status()

        return response.json()

    async def _make_async_request(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        stop_sequences: list[str] | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make asynchronous request to Ollama API.

        Args:
            messages: Formatted messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional parameters

        Returns:
            API response dictionary
        """
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout

        url = f"{self.base_url}/api/chat"

        # Build request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        if stop_sequences:
            payload["options"]["stop"] = stop_sequences

        # Add any additional options
        if "options" in kwargs:
            payload["options"].update(kwargs["options"])

        # Make request
        response = await self._async_client.post(url, json=payload)
        response.raise_for_status()

        return response.json()

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from the given messages.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Provider-specific parameters

        Returns:
            LLM response with generated text and metadata
        """
        try:
            # Prepare messages
            ollama_messages = self._prepare_messages(messages)

            # Make request
            response_data = self._make_request(ollama_messages, temperature, max_tokens, stop_sequences, **kwargs)

            # Extract response content
            content = response_data.get("message", {}).get("content", "")

            # Extract token usage if available
            usage = {}
            if "prompt_eval_count" in response_data:
                usage["prompt_tokens"] = response_data["prompt_eval_count"]
            if "eval_count" in response_data:
                usage["completion_tokens"] = response_data["eval_count"]
            if usage:
                usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

            # Build metadata
            metadata = {
                "model": response_data.get("model", self.model_name),
                "created_at": response_data.get("created_at"),
                "done_reason": response_data.get("done_reason", "stop"),
                "total_duration": response_data.get("total_duration"),
                "load_duration": response_data.get("load_duration"),
                "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                "eval_duration": response_data.get("eval_duration"),
            }

            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                metadata=metadata,
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"Ollama API error: {e.response.status_code} - {e.response.text}"
            return LLMResponse(
                content="",
                model=self.model_name,
                error=error_msg,
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                error=f"Ollama request failed: {e!s}",
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
        """Async version of complete.

        Args:
            messages: Conversation history
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Provider-specific parameters

        Returns:
            LLM response with generated text and metadata
        """
        try:
            # Prepare messages
            ollama_messages = self._prepare_messages(messages)

            # Make request
            response_data = await self._make_async_request(
                ollama_messages, temperature, max_tokens, stop_sequences, **kwargs
            )

            # Extract response content
            content = response_data.get("message", {}).get("content", "")

            # Extract token usage if available
            usage = {}
            if "prompt_eval_count" in response_data:
                usage["prompt_tokens"] = response_data["prompt_eval_count"]
            if "eval_count" in response_data:
                usage["completion_tokens"] = response_data["eval_count"]
            if usage:
                usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

            # Build metadata
            metadata = {
                "model": response_data.get("model", self.model_name),
                "created_at": response_data.get("created_at"),
                "done_reason": response_data.get("done_reason", "stop"),
                "total_duration": response_data.get("total_duration"),
                "load_duration": response_data.get("load_duration"),
                "prompt_eval_duration": response_data.get("prompt_eval_duration"),
                "eval_duration": response_data.get("eval_duration"),
            }

            return LLMResponse(
                content=content,
                model=self.model_name,
                usage=usage,
                metadata=metadata,
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"Ollama API error: {e.response.status_code} - {e.response.text}"
            return LLMResponse(
                content="",
                model=self.model_name,
                error=error_msg,
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model=self.model_name,
                error=f"Ollama request failed: {e!s}",
            )

    def __del__(self):
        """Clean up HTTP clients."""
        if self._client:
            self._client.close()
        if self._async_client:
            # Need to handle async client cleanup properly
            try:
                asyncio.create_task(self._async_client.aclose())
            except RuntimeError:
                # If no event loop, just let it be garbage collected
                pass
