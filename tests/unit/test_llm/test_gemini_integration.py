"""Unit tests for Gemini LLM integration."""

import os
from unittest import mock

from spreadsheet_analyzer.cli.llm_interaction import create_llm_instance


def test_gemini_model_name_mapping():
    """Test Gemini model name mapping."""
    test_cases = [
        # Full name mappings
        ("gemini-2.5-pro", "models/gemini-2.5-pro"),
        ("gemini-1.5-pro", "models/gemini-1.5-pro"),
        # Alias mappings
        ("gemini-pro", "models/gemini-2.5-pro"),
        # Mixed case handling
        ("Gemini-2.5-Pro", "models/gemini-2.5-pro"),
        ("Gemini-Pro", "models/gemini-2.5-pro"),
        # Unmapped models should get models/ prefix
        ("gemini-2.0-flash-experimental", "models/gemini-2.0-flash-experimental"),
        ("Gemini-Custom-Model", "models/gemini-custom-model"),
    ]

    for input_model, expected_model in test_cases:
        with (
            mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            mock.patch("spreadsheet_analyzer.cli.llm_interaction.ChatGoogleGenerativeAI") as mock_llm,
        ):
            result = create_llm_instance(input_model)
            assert result.is_ok(), f"Failed to create LLM instance for {input_model}"

            # Verify model name passed to ChatGoogleGenerativeAI
            mock_llm.assert_called_once_with(
                model=expected_model,
                api_key="test-key",
                temperature=0,
                max_tokens=None,
                max_retries=2,
                disable_streaming="tool_calling",
            )


def test_gemini_api_key_validation():
    """Test Gemini API key validation."""
    # Test missing API key
    with mock.patch.dict(os.environ, {}, clear=True):
        result = create_llm_instance("gemini-pro")
        assert result.is_err(), "Should fail when no API key is set"
        assert "No API key provided" in result.unwrap_err()

    # Test with API key from function parameter
    with mock.patch("spreadsheet_analyzer.cli.llm_interaction.ChatGoogleGenerativeAI") as mock_llm:
        result = create_llm_instance("gemini-2.5-pro", api_key="custom-test-key")
        assert result.is_ok(), "Should succeed with custom API key"

        mock_llm.assert_called_once_with(
            model="models/gemini-2.5-pro",
            api_key="custom-test-key",
            temperature=0,
            max_tokens=None,
            max_retries=2,
            disable_streaming="tool_calling",
        )


def test_gemini_instantiation_with_env_var():
    """Test Gemini model instantiation using environment variable."""
    with (
        mock.patch.dict(os.environ, {"GEMINI_API_KEY": "env-test-key"}),
        mock.patch("spreadsheet_analyzer.cli.llm_interaction.ChatGoogleGenerativeAI") as mock_llm,
    ):
        result = create_llm_instance("gemini-2.5-pro")
        assert result.is_ok(), "Should succeed with environment API key"

        mock_llm.assert_called_once_with(
            model="models/gemini-2.5-pro",
            api_key="env-test-key",
            temperature=0,
            max_tokens=None,
            max_retries=2,
            disable_streaming="tool_calling",
        )
