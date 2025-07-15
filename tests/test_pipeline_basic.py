"""
Basic tests for the pipeline functionality.

These are minimal tests to ensure test discovery works and basic
functionality is verified.
"""

from pathlib import Path

import pytest

from spreadsheet_analyzer.pipeline.pipeline import DeterministicPipeline
from spreadsheet_analyzer.pipeline.stages.stage_0_integrity import stage_0_integrity_probe
from spreadsheet_analyzer.pipeline.types import Err, Ok


def test_ok_result_creation():
    """Test that Ok result wrapper works correctly."""
    result = Ok("test_value")
    assert result.value == "test_value"


def test_err_result_creation():
    """Test that Err result wrapper works correctly."""
    error = Err("test_error")
    assert error.error == "test_error"
    assert error.details is None

    error_with_details = Err("test_error", {"key": "value"})
    assert error_with_details.details == {"key": "value"}


def test_basic_imports():
    """Test that main pipeline modules can be imported."""
    # Basic instantiation test
    pipeline = DeterministicPipeline()
    assert pipeline is not None


def test_path_validation():
    """Test basic path validation functionality."""
    # Test with non-existent file
    fake_path = Path("/nonexistent/file.xlsx")
    result = stage_0_integrity_probe(fake_path)

    assert isinstance(result, Err)
    assert "not found" in result.error.lower()


@pytest.mark.skipif(not Path("test-files").exists(), reason="test-files directory not available")
def test_with_real_file():
    """Test with real Excel file if available."""

    test_files_dir = Path("test-files")
    excel_files = list(test_files_dir.glob("*.xlsx"))

    if excel_files:
        result = stage_0_integrity_probe(excel_files[0])
        # Should either pass or fail gracefully, not crash
        # CLAUDE-GOTCHA: Cannot use union syntax (Ok | Err) here as isinstance()
        # doesn't support types.UnionType at runtime
        assert isinstance(result, (Ok, Err))  # noqa: UP038
