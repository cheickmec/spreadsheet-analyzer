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


def test_gemini_error_handling_dataframe_methods():
    """Test Gemini-specific error handling for DataFrame method mistakes."""
    from spreadsheet_analyzer.cli.llm_interaction import generate_gemini_error_message

    dataframe_methods = ["to_markdown", "tolist", "head", "tail", "describe", "info"]
    mock_tools = [type("MockTool", (), {"name": "execute_code"}) for _ in range(1)]

    for method in dataframe_methods:
        error_msg = generate_gemini_error_message(method, mock_tools)
        assert f"ERROR: '{method}' is a pandas DataFrame method, NOT a tool!" in error_msg
        assert "execute_code(code=" in error_msg
        assert "Please retry using the execute_code tool" in error_msg


def test_gemini_error_handling_unknown_tools():
    """Test Gemini-specific error handling for unknown tools."""
    from spreadsheet_analyzer.cli.llm_interaction import generate_gemini_error_message

    mock_tools = [
        type("MockTool", (), {"name": "tool1"}),
        type("MockTool", (), {"name": "tool2"}),
        type("MockTool", (), {"name": "tool3"}),
        type("MockTool", (), {"name": "tool4"}),
        type("MockTool", (), {"name": "tool5"}),
    ]

    error_msg = generate_gemini_error_message("unknown_tool", mock_tools)
    assert "Unknown tool 'unknown_tool'" in error_msg
    assert "Available tools are: tool1, tool2, tool3, tool4, tool5..." in error_msg
    assert "Please use one of the available tools" in error_msg
