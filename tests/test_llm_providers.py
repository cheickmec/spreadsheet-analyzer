"""Tests for LLM provider implementations."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from spreadsheet_analyzer.notebook_llm.llm_providers import (
    get_provider,
    list_providers,
    register_provider,
)
from spreadsheet_analyzer.notebook_llm.llm_providers.anthropic_provider import (
    AnthropicProvider,
)
from spreadsheet_analyzer.notebook_llm.llm_providers.base import (
    LLMProvider,
    LLMResponse,
    Message,
    Role,
)
from spreadsheet_analyzer.notebook_llm.llm_providers.openai_provider import OpenAIProvider


class TestLLMProviders:
    """Tests for LLM provider implementations."""

    def test_list_providers(self):
        """Test listing available providers."""
        providers = list_providers()
        assert "openai" in providers
        assert "anthropic" in providers

    def test_get_provider_openai(self):
        """Test getting OpenAI provider."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = get_provider("openai")
            assert isinstance(provider, OpenAIProvider)
            assert provider.model_name == "gpt-4"
            assert provider.max_context_tokens == 8192

    def test_get_provider_anthropic(self):
        """Test getting Anthropic provider."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = get_provider("anthropic")
            assert isinstance(provider, AnthropicProvider)
            assert provider.model_name == "claude-3-sonnet-20240229"
            assert provider.max_context_tokens == 200000

    def test_get_provider_unknown(self):
        """Test getting unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown")

    def test_get_provider_missing_api_key(self):
        """Test provider initialization without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                get_provider("openai")

    def test_register_custom_provider(self):
        """Test registering a custom provider."""

        class CustomProvider(LLMProvider):
            @property
            def default_model(self) -> str:
                return "custom-model"

            def complete(self, messages, **kwargs) -> LLMResponse:
                return LLMResponse(
                    content="Custom response",
                    model="custom-model",
                    usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                )

            async def complete_async(self, messages, **kwargs) -> LLMResponse:
                return self.complete(messages, **kwargs)

            def count_tokens(self, text: str) -> int:
                return len(text.split())

            @property
            def max_context_tokens(self) -> int:
                return 4096

        register_provider("custom", CustomProvider)
        providers = list_providers()
        assert "custom" in providers

        provider = get_provider("custom", api_key="dummy")
        assert isinstance(provider, CustomProvider)


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    @patch("openai.OpenAI")
    def test_openai_provider_initialization(self, mock_openai_class):
        """Test OpenAI provider initialization."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-3.5-turbo")
        assert provider.model_name == "gpt-3.5-turbo"
        assert provider.api_key == "test-key"
        mock_openai_class.assert_called_once_with(api_key="test-key")

    @patch("openai.OpenAI")
    def test_openai_complete(self, mock_openai_class):
        """Test OpenAI completion."""
        # Mock the client and response
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant"),
            Message(role=Role.USER, content="Hello"),
        ]

        response = provider.complete(messages)
        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.usage["total_tokens"] == 15

    @patch("openai.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_openai_complete_async(self, mock_async_openai_class):
        """Test OpenAI async completion."""
        # Mock the async client and response
        mock_client = AsyncMock()
        mock_async_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Async response"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello async")]

        response = await provider.complete_async(messages)
        assert response.content == "Async response"

    @patch("tiktoken.encoding_for_model")
    def test_openai_count_tokens(self, mock_encoding):
        """Test OpenAI token counting."""
        # Mock tiktoken encoder
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_encoding.return_value = mock_encoder

        provider = OpenAIProvider(api_key="test-key")
        count = provider.count_tokens("Hello world test text")
        assert count == 5

    def test_openai_model_properties(self):
        """Test OpenAI model properties."""
        with patch("openai.OpenAI"):
            provider = OpenAIProvider(api_key="test-key", model="gpt-4-turbo")
            assert provider.max_context_tokens == 128000
            assert provider.default_model == "gpt-4"


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    @patch("anthropic.Anthropic")
    def test_anthropic_provider_initialization(self, mock_anthropic_class):
        """Test Anthropic provider initialization."""
        provider = AnthropicProvider(api_key="test-key", model="claude-3-opus-20240229")
        assert provider.model_name == "claude-3-opus-20240229"
        assert provider.api_key == "test-key"
        mock_anthropic_class.assert_called_once_with(api_key="test-key")

    @patch("anthropic.Anthropic")
    def test_anthropic_complete(self, mock_anthropic_class):
        """Test Anthropic completion."""
        # Mock the client and response
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Claude response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-3-sonnet-20240229"
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="test-key")
        messages = [
            Message(role=Role.SYSTEM, content="You are Claude"),
            Message(role=Role.USER, content="Hello"),
        ]

        response = provider.complete(messages)
        assert response.content == "Claude response"
        assert response.model == "claude-3-sonnet-20240229"
        assert response.usage["total_tokens"] == 15

    @patch("anthropic.AsyncAnthropic")
    @pytest.mark.asyncio
    async def test_anthropic_complete_async(self, mock_async_anthropic_class):
        """Test Anthropic async completion."""
        # Mock the async client and response
        mock_client = AsyncMock()
        mock_async_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Async Claude response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-3-sonnet-20240229"
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello async Claude")]

        response = await provider.complete_async(messages)
        assert response.content == "Async Claude response"

    def test_anthropic_count_tokens(self):
        """Test Anthropic token counting (approximation)."""
        provider = AnthropicProvider(api_key="test-key")
        # Anthropic uses approximation: ~4 chars per token
        text = "This is a test message with multiple words"  # 43 chars
        count = provider.count_tokens(text)
        assert 10 <= count <= 12  # Allow some variance in approximation

    def test_anthropic_model_properties(self):
        """Test Anthropic model properties."""
        with patch("anthropic.Anthropic"):
            provider = AnthropicProvider(api_key="test-key")
            assert provider.max_context_tokens == 200000
            assert provider.default_model == "claude-3-sonnet-20240229"

            # Test other models
            provider = AnthropicProvider(api_key="test-key", model="claude-3-haiku-20240307")
            assert provider.max_context_tokens == 200000


class TestLLMProviderBase:
    """Tests for base LLM provider functionality."""

    def test_message_creation(self):
        """Test creating messages."""
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert msg.metadata == {}

    def test_message_with_metadata(self):
        """Test creating messages with metadata."""
        msg = Message(
            role=Role.ASSISTANT,
            content="Response",
            metadata={"timestamp": "2024-01-01", "model": "test"},
        )
        assert msg.metadata["timestamp"] == "2024-01-01"
        assert msg.metadata["model"] == "test"

    def test_llm_response_creation(self):
        """Test creating LLM responses."""
        response = LLMResponse(
            content="Test response",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.usage["total_tokens"] == 15
        assert response.error is None

    def test_llm_response_with_error(self):
        """Test creating LLM responses with errors."""
        response = LLMResponse(
            content="",
            model="test-model",
            error="API rate limit exceeded",
        )
        assert response.content == ""
        assert response.error == "API rate limit exceeded"

    def test_base_provider_messages_to_dict(self):
        """Test converting messages to dictionary format."""
        with patch("openai.OpenAI"):
            provider = OpenAIProvider(api_key="test-key")
            messages = [
                Message(role=Role.SYSTEM, content="System prompt"),
                Message(role=Role.USER, content="User message"),
                Message(role=Role.ASSISTANT, content="Assistant response"),
            ]

            dict_messages = provider._messages_to_dict(messages)
            assert len(dict_messages) == 3
            assert dict_messages[0]["role"] == "system"
            assert dict_messages[0]["content"] == "System prompt"
            assert dict_messages[1]["role"] == "user"
            assert dict_messages[2]["role"] == "assistant"
