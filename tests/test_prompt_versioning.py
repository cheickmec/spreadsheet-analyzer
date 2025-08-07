"""Tests for prompt versioning and hash validation system.

This module tests the prompt registry, loader, and hash validation
to ensure prompt changes are tracked and enforced.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from spreadsheet_analyzer.prompts import (
    PROMPT_REGISTRY,
    PromptDefinition,
    compute_file_hash,
    get_prompt_definition,
    list_prompts,
    load_prompt,
    validate_all_prompts,
)


class TestPromptRegistry:
    """Test the prompt registry functionality."""

    def test_registry_has_all_required_prompts(self):
        """Test that all required prompts are in the registry."""
        required_prompts = {
            "data_analyst_system",
            "data_analyst_initial",
            "table_aware_analyst_system",
            "table_detector_system",
        }

        registered_prompts = set(PROMPT_REGISTRY.keys())
        assert required_prompts == registered_prompts

    def test_prompt_definitions_are_immutable(self):
        """Test that PromptDefinition is truly immutable."""
        definition = PROMPT_REGISTRY["data_analyst_system"]

        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            definition.content_hash = "new_hash"

        with pytest.raises(AttributeError):
            definition.name = "new_name"

    def test_get_prompt_definition(self):
        """Test getting prompt definitions by name."""
        # Valid prompt
        definition = get_prompt_definition("data_analyst_system")
        assert definition is not None
        assert definition.name == "data_analyst_system"
        assert definition.file_name == "data_analyst_system.prompt.yml"

        # Invalid prompt
        definition = get_prompt_definition("nonexistent_prompt")
        assert definition is None

    def test_list_prompts(self):
        """Test listing all prompt names."""
        prompts = list_prompts()
        assert len(prompts) == 4
        assert "data_analyst_system" in prompts
        assert "table_detector_system" in prompts


class TestPromptLoader:
    """Test the prompt loading and hash validation."""

    def test_load_valid_prompt(self):
        """Test loading a prompt with valid hash."""
        result = load_prompt("data_analyst_system")
        assert result.is_ok()

        prompt_data = result.unwrap()
        assert "template" in prompt_data
        assert "input_variables" in prompt_data
        assert isinstance(prompt_data["input_variables"], list)

    def test_load_unknown_prompt(self):
        """Test loading a non-existent prompt."""
        result = load_prompt("nonexistent_prompt")
        assert result.is_err()
        assert "Unknown prompt" in result.unwrap_err()
        assert "Available prompts" in result.unwrap_err()

    def test_hash_mismatch_detection(self):
        """Test that modified prompts are detected."""
        # Create a temporary file with different content
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""_type: prompt
input_variables:
- test
template: |
  Modified content that will have a different hash
""")
            temp_path = Path(f.name)

        try:
            # Patch the file path to use our temp file
            definition = PROMPT_REGISTRY["data_analyst_system"]
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "__truediv__", return_value=temp_path):
                    result = load_prompt("data_analyst_system")

                    assert result.is_err()
                    assert "has been modified" in result.unwrap_err()
                    assert "Expected hash:" in result.unwrap_err()
                    assert "Current hash:" in result.unwrap_err()
                    assert "update the hash in registry.py" in result.unwrap_err()
        finally:
            temp_path.unlink()

    def test_missing_file_handling(self):
        """Test handling of missing prompt files."""
        # Create a fake prompt definition
        fake_definition = PromptDefinition(
            name="fake_prompt", file_name="nonexistent.yaml", content_hash="sha256:fake", description="Test prompt"
        )

        with patch.dict(PROMPT_REGISTRY, {"fake_prompt": fake_definition}):
            result = load_prompt("fake_prompt")
            assert result.is_err()
            assert "not found" in result.unwrap_err()

    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: {")
            temp_path = Path(f.name)

        try:
            # Compute the hash of the invalid file
            actual_hash = compute_file_hash(temp_path)

            # Create a definition with the correct hash
            fake_definition = PromptDefinition(
                name="invalid_yaml", file_name="invalid.yaml", content_hash=actual_hash, description="Test prompt"
            )

            with patch.dict(PROMPT_REGISTRY, {"invalid_yaml": fake_definition}):
                with patch.object(Path, "__truediv__", return_value=temp_path):
                    with patch.object(Path, "exists", return_value=True):
                        result = load_prompt("invalid_yaml")
                        assert result.is_err()
                        assert "Failed to parse YAML" in result.unwrap_err()
        finally:
            temp_path.unlink()

    def test_validate_all_prompts(self):
        """Test validating all prompts at once."""
        result = validate_all_prompts()
        assert result.is_ok()

        validation_results = result.unwrap()
        assert len(validation_results) == 4
        assert all(validation_results.values())  # All should be valid


class TestHashComputation:
    """Test hash computation functionality."""

    def test_compute_file_hash(self):
        """Test SHA-256 hash computation."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            hash_value = compute_file_hash(temp_path)
            assert hash_value.startswith("sha256:")
            assert len(hash_value) == 71  # "sha256:" + 64 hex chars

            # Same content should produce same hash
            hash_value2 = compute_file_hash(temp_path)
            assert hash_value == hash_value2
        finally:
            temp_path.unlink()

    def test_hash_changes_with_content(self):
        """Test that different content produces different hashes."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("content1")
            temp_path1 = Path(f.name)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("content2")
            temp_path2 = Path(f.name)

        try:
            hash1 = compute_file_hash(temp_path1)
            hash2 = compute_file_hash(temp_path2)
            assert hash1 != hash2
        finally:
            temp_path1.unlink()
            temp_path2.unlink()


class TestIntegrationWithCode:
    """Test integration with actual code that uses prompts."""

    def test_llm_interaction_can_load_prompts(self):
        """Test that llm_interaction module can load prompts."""
        from spreadsheet_analyzer.cli.llm_interaction import create_system_prompt

        # Should not raise any exceptions
        prompt = create_system_prompt(
            excel_file_name="test.xlsx",
            sheet_index=0,
            sheet_name="Sheet1",
            notebook_state="# Test notebook",
            table_boundaries=None,
        )
        assert isinstance(prompt, str)
        assert "data analyst" in prompt.lower()

    def test_workflow_can_load_prompts(self):
        """Test that workflow module can load prompts."""
        # Just verify the import works and prompt can be loaded
        from spreadsheet_analyzer.prompts import load_prompt

        result = load_prompt("table_detector_system")
        assert result.is_ok()


class TestDeveloperExperience:
    """Test developer-facing error messages and tooling."""

    def test_error_message_is_actionable(self):
        """Test that hash mismatch errors provide clear instructions."""
        # Simulate a hash mismatch
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("template: modified")
            temp_path = Path(f.name)

        try:
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "__truediv__", return_value=temp_path):
                    result = load_prompt("data_analyst_system")

                    error_msg = result.unwrap_err()
                    # Check for actionable parts
                    assert "❌" in error_msg  # Visual indicator
                    assert "has been modified" in error_msg
                    assert "Expected hash:" in error_msg
                    assert "Current hash:" in error_msg
                    assert "To fix this" in error_msg
                    assert "PROMPT_REGISTRY" in error_msg
                    assert ".content_hash" in error_msg
        finally:
            temp_path.unlink()
