"""
Test analyzer behavior against captured fixtures from test-files/.

This module uses the outputs captured from processing test Excel files
to ensure consistent analyzer behavior across changes.
"""

import json

# Add src to path for imports
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spreadsheet_analyzer.pipeline.pipeline import DeterministicPipeline, create_lenient_pipeline_options
from spreadsheet_analyzer.testing.loader import FixtureLoader


class TestAgainstCapturedFixtures:
    """Test analyzer against captured outputs from test files."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        """Create a pipeline instance for testing."""
        return DeterministicPipeline(create_lenient_pipeline_options())

    @pytest.fixture(scope="class")
    def captured_fixtures_dir(self):
        """Path to captured fixtures directory."""
        return Path(__file__).parent / "fixtures" / "captured_outputs"

    @pytest.fixture(scope="class")
    def test_files_dir(self):
        """Path to test files directory."""
        return Path(__file__).parent.parent / "test-files"

    def load_fixture(self, fixture_path: Path) -> dict[str, Any]:
        """Load a fixture file."""
        with fixture_path.open() as f:
            data: dict[str, Any] = json.load(f)
            return data

    def get_test_file_path(self, fixture_path: Path, test_files_dir: Path) -> Path:
        """Get the test file path from fixture path."""
        # Extract relative path from fixture name
        relative_path = fixture_path.relative_to(fixture_path.parent.parent)
        test_file_name = relative_path.with_suffix(".xlsx")

        # Handle .xlsm files
        test_file_path = test_files_dir / test_file_name
        if not test_file_path.exists():
            test_file_path = test_file_path.with_suffix(".xlsm")

        return test_file_path

    @pytest.mark.parametrize(
        "fixture_file",
        [
            "data-analysis/advanced_excel_formulas.json",
            "edge-cases/sheetjs_test_file.json",
            "financial-models/tesla_valuation_model.json",
            "hr-timesheet/timesheet_generator.json",
        ],
    )
    def test_pipeline_success_status(self, pipeline, captured_fixtures_dir, test_files_dir, fixture_file):
        """Test that pipeline success status matches captured fixture."""
        fixture_path = captured_fixtures_dir / fixture_file
        fixture_data = self.load_fixture(fixture_path)

        # Get test file path
        test_file = self.get_test_file_path(fixture_path, test_files_dir)
        assert test_file.exists(), f"Test file not found: {test_file}"

        # Run pipeline
        result = pipeline.run(test_file)

        # Compare success status
        expected_success = fixture_data["pipeline_result"]["success"]
        assert result.success == expected_success, (
            f"Pipeline success mismatch for {fixture_file}: expected {expected_success}, got {result.success}"
        )

        # Compare error messages if failed
        if not result.success:
            expected_errors = fixture_data["pipeline_result"]["errors"]
            assert list(result.errors) == expected_errors, (
                f"Error mismatch for {fixture_file}: expected {expected_errors}, got {list(result.errors)}"
            )

    def test_successful_file_stages(self, pipeline, test_files_dir):
        """Test that successful files complete all expected stages."""

        # Get test file
        test_file = test_files_dir / "data-analysis" / "advanced_excel_formulas.xlsx"
        assert test_file.exists()

        # Run pipeline
        result = pipeline.run(test_file)

        # Load fixture using the language-agnostic loader
        loader = FixtureLoader()
        expected_result = loader.load_as_dataclass("data-analysis/advanced_excel_formulas.xlsx")

        # Check all stages completed
        assert result.success
        assert result.integrity is not None
        assert result.security is not None
        assert result.structure is not None
        assert result.formulas is not None
        assert result.content is not None

        # Compare key metrics using dataclass fields
        assert result.structure.sheet_count == expected_result.structure.sheet_count
        assert result.structure.total_cells == expected_result.structure.total_cells
        assert result.structure.total_formulas == expected_result.structure.total_formulas

        # The JSON is now clean and language-agnostic
        # Any language can read the raw JSON without Python-specific markers

    def test_edge_case_file_blocked(self, pipeline, test_files_dir):
        """Test that edge case file is blocked as expected."""

        # Get test file
        test_file = test_files_dir / "edge-cases" / "sheetjs_test_file.xlsx"
        assert test_file.exists()

        # Run pipeline
        result = pipeline.run(test_file)

        # Check it was blocked
        assert not result.success
        assert "File blocked due to integrity check" in result.errors

        # Verify no stages completed
        assert result.integrity is None
        assert result.security is None
        assert result.structure is None


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])
