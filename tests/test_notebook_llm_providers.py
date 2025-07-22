"""Tests for notebook_llm LLM providers."""

from unittest.mock import Mock, patch

import pytest

from spreadsheet_analyzer.notebook_llm.llm_providers import (
    get_provider,
    list_providers,
)
from spreadsheet_analyzer.notebook_llm.llm_providers.anthropic_provider import (
    AnthropicProvider,
)
from spreadsheet_analyzer.notebook_llm.llm_providers.base import (
    Message,
    Role,
)
from spreadsheet_analyzer.notebook_llm.llm_providers.openai_provider import OpenAIProvider


class TestLLMProvidersRegistry:
    """Tests for LLM provider registry."""

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

    def test_get_provider_anthropic(self):
        """Test getting Anthropic provider."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = get_provider("anthropic")
            assert isinstance(provider, AnthropicProvider)

    def test_get_provider_unknown(self):
        """Test getting unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown")


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    @patch("openai.OpenAI")
    def test_initialization(self, mock_openai_class):
        """Test OpenAI provider initialization."""
        provider = OpenAIProvider(api_key="test-key", model="gpt-3.5-turbo")
        assert provider.model_name == "gpt-3.5-turbo"
        mock_openai_class.assert_called_once()

    @patch("openai.OpenAI")
    def test_complete_sync(self, mock_openai_class):
        """Test synchronous completion."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        mock_response.model = "gpt-4"
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]

        response = provider.complete(messages)
        assert response.content == "Test response"
        assert response.usage["total_tokens"] == 15

    @patch("tiktoken.encoding_for_model")
    def test_count_tokens(self, mock_encoding):
        """Test token counting."""
        mock_encoder = Mock()
        mock_encoder.encode.return_value = [1, 2, 3, 4, 5]
        mock_encoding.return_value = mock_encoder

        provider = OpenAIProvider(api_key="test-key")
        count = provider.count_tokens("Hello world")
        assert count == 5

    def test_model_properties(self):
        """Test model properties."""
        with patch("openai.OpenAI"):
            # GPT-4
            provider = OpenAIProvider(api_key="test-key", model="gpt-4")
            assert provider.max_context_tokens == 8192

            # GPT-4 Turbo
            provider = OpenAIProvider(api_key="test-key", model="gpt-4-turbo")
            assert provider.max_context_tokens == 128000


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    @patch("anthropic.Anthropic")
    def test_initialization(self, mock_anthropic_class):
        """Test Anthropic provider initialization."""
        provider = AnthropicProvider(api_key="test-key")
        mock_anthropic_class.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_complete_sync(self, mock_anthropic_class):
        """Test synchronous completion."""
        mock_client = Mock()
        mock_anthropic_class.return_value = mock_client

        mock_response = Mock()
        mock_response.content = [Mock(text="Claude response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.model = "claude-3-sonnet-20240229"
        mock_client.messages.create.return_value = mock_response

        provider = AnthropicProvider(api_key="test-key")
        messages = [Message(role=Role.USER, content="Hello")]

        response = provider.complete(messages)
        assert response.content == "Claude response"

    def test_count_tokens_approximation(self):
        """Test token counting approximation."""
        provider = AnthropicProvider(api_key="test-key")
        # ~4 chars per token
        text = "This is a test"  # 14 chars
        count = provider.count_tokens(text)
        assert 3 <= count <= 5  # Allow variance

    def test_model_properties(self):
        """Test model properties."""
        with patch("anthropic.Anthropic"):
            provider = AnthropicProvider(api_key="test-key")
            assert provider.max_context_tokens == 200000
            assert provider.default_model == "claude-3-sonnet-20240229"
