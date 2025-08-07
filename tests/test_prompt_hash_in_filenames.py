"""Tests for prompt hash inclusion in generated filenames."""

from pathlib import Path

from spreadsheet_analyzer.cli.utils.naming import (
    FileNameConfig,
    generate_log_name,
    generate_notebook_name,
    get_cost_tracking_path,
    get_short_hash,
)


class TestPromptHashInFilenames:
    """Test that prompt hashes are correctly included in file names."""

    def test_get_short_hash(self):
        """Test hash shortening function."""
        # Test with full SHA-256 hash
        full_hash = "sha256:531bcb715f64c7367f8361a1b129c080771684a06b0bdaef32871cd7e0e26280"
        assert get_short_hash(full_hash) == "531bcb71"

        # Test without prefix
        hex_only = "988988e6c8b4421bc6f7ae15bf1c17ca9296acb2e6ff4e3a8dd1a5b6376f2675"
        assert get_short_hash(hex_only) == "988988e6"

        # Test with short hash
        short = "abc123"
        assert get_short_hash(short) == "abc123"

    def test_notebook_name_with_hash(self):
        """Test notebook file name includes prompt hash."""
        config = FileNameConfig(
            excel_file=Path("test_data.xlsx"),
            model="gpt-4",
            sheet_index=0,
            sheet_name="Sheet1",
            max_rounds=5,
            prompt_hash="531bcb71",
        )

        # Without timestamp
        name = generate_notebook_name(config, include_timestamp=False)
        assert "531bcb71" in name
        assert name == "test_data_sheet0_Sheet1_gpt_4_r5_531bcb71.ipynb"

        # With timestamp (just check hash is before timestamp)
        name_with_time = generate_notebook_name(config, include_timestamp=True)
        parts = name_with_time.split("_")
        # Find the hash position
        hash_idx = parts.index("531bcb71")
        # Should be followed by date/time parts
        assert hash_idx < len(parts) - 2  # At least 2 parts after (date and time)

    def test_notebook_name_without_hash(self):
        """Test notebook file name without prompt hash (backwards compatibility)."""
        config = FileNameConfig(
            excel_file=Path("test_data.xlsx"),
            model="gpt-4",
            sheet_index=0,
            sheet_name="Sheet1",
            max_rounds=5,
            prompt_hash=None,  # No hash provided
        )

        name = generate_notebook_name(config, include_timestamp=False)
        assert "531bcb71" not in name
        assert name == "test_data_sheet0_Sheet1_gpt_4_r5.ipynb"

    def test_log_name_with_hash(self):
        """Test log file name includes prompt hash."""
        config = FileNameConfig(
            excel_file=Path("test_data.xlsx"),
            model="claude-3-5-sonnet",
            sheet_index=1,
            max_rounds=3,
            prompt_hash="988988e6",
        )

        name = generate_log_name(config, include_timestamp=False)
        assert "988988e6" in name
        # Log names have "analysis" suffix before timestamp
        assert "_988988e6_analysis.log" in name

    def test_cost_tracking_with_hash(self):
        """Test cost tracking file name includes prompt hash."""
        config = FileNameConfig(
            excel_file=Path("sales.xlsx"),
            model="gpt-4-turbo",
            sheet_index=0,
            max_rounds=5,
            prompt_hash="db5236da",
        )

        path = get_cost_tracking_path(config)
        assert "db5236da" in str(path)
        # Cost tracking has specific suffix
        assert "_db5236da_cost_tracking_" in str(path)
        assert path.suffix == ".json"

    def test_different_hashes_for_different_agents(self):
        """Test that different agents get different hashes in their filenames."""
        # Detector config
        detector_config = FileNameConfig(
            excel_file=Path("data.xlsx"),
            model="claude-3-5-sonnet",
            sheet_index=0,
            max_rounds=3,
            prompt_hash="988988e6",  # Table detector hash
        )

        # Analyst config
        analyst_config = FileNameConfig(
            excel_file=Path("data.xlsx"),
            model="claude-3-5-sonnet",
            sheet_index=0,
            max_rounds=5,
            prompt_hash="531bcb71",  # Data analyst hash
        )

        detector_name = generate_notebook_name(detector_config, include_timestamp=False)
        analyst_name = generate_notebook_name(analyst_config, include_timestamp=False)

        # Different hashes should appear
        assert "988988e6" in detector_name
        assert "531bcb71" in analyst_name
        assert detector_name != analyst_name
