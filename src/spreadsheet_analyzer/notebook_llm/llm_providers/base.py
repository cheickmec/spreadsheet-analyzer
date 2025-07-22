"""Base interfaces and data structures for LLM providers.

This module defines the protocol and data structures that all LLM providers
must implement to integrate with the spreadsheet analyzer system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class Role(Enum):
    """Message role in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: Role
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)  # tokens used
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class LLMInterface(Protocol):
    """Protocol for LLM interaction.

    All LLM providers must implement this interface to be compatible
    with the spreadsheet analyzer system.
    """

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
        ...

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
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        ...

    @property
    def model_name(self) -> str:
        """Get the model name."""
        ...

    @property
    def max_context_tokens(self) -> int:
        """Get maximum context window size."""
        ...


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Concrete implementations should handle API authentication,
    request formatting, and response parsing for specific providers.
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        """Initialize LLM provider.

        Args:
            api_key: API key for authentication
            model: Model identifier
        """
        self.api_key = api_key
        self._model = model

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model or self.default_model

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Get default model for this provider."""
        pass

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion from the given messages."""
        pass

    @abstractmethod
    async def complete_async(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async version of complete."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        pass

    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Get maximum context window size."""
        pass

    def _messages_to_dict(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to dictionary format.

        Args:
            messages: List of Message objects

        Returns:
            List of message dictionaries
        """
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]
