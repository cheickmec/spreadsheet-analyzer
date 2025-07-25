"""Test fixtures and utilities for CLI testing.

This module provides shared fixtures and utilities for testing the
CLI's deterministic generation and notebook validation capabilities.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from spreadsheet_analyzer.plugins.spreadsheet.io.excel_io import list_sheets

# Test data files mapping
TEST_FILES = {
    "simple_sales.xlsx": {
        "description": "Multi-sheet sales data",
        "sheets": ["Monthly Sales", "Regional Breakdown", "Product Performance"],
        "expected_tasks": ["profile", "outliers", "correlations", "summary"],
    },
    "financial_model.xlsx": {
        "description": "Complex formulas and financial calculations",
        "sheets": ["Financial Model"],
        "expected_tasks": ["profile", "formulas", "dependencies", "validation"],
    },
    "inventory_tracking.csv": {
        "description": "CSV inventory data",
        "sheets": [None],  # CSV has no sheets
        "expected_tasks": ["profile", "quality", "outliers", "patterns"],
    },
    "employee_records.xlsx": {
        "description": "Data with intentional quality issues",
        "sheets": ["Sheet1"],  # Default sheet name
        "expected_tasks": ["profile", "quality", "duplicates", "validation"],
    },
}


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory path for generated files."""
    return Path(__file__).parent.parent.parent / "test_assets" / "generated"


@pytest.fixture(scope="session")
def collection_files_dir() -> Path:
    """Get the directory path for the curated collection of real-world files."""
    return Path(__file__).parent.parent.parent / "test_assets" / "collection"


@pytest.fixture(scope="session")
def collection_manifest():
    """Load and return the parsed manifest.json from the collection directory."""
    collection_dir = Path(__file__).parent.parent.parent / "test_assets" / "collection"
    manifest_path = collection_dir / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("Collection manifest.json not found.")
    with open(manifest_path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def reference_notebooks_dir() -> Path:
    """Get the reference notebooks directory path."""
    return Path(__file__).parent.parent.parent / "reference_notebooks"


@pytest.fixture(scope="session")
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="cli_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def cli_runner():
    """Fixture that provides a function to run CLI commands."""

    def run_cli(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
        """
        Run the CLI with the given arguments.

        Args:
            args: Command line arguments to pass to the CLI
            cwd: Working directory to run the command from

        Returns:
            CompletedProcess with results
        """
        cmd = [sys.executable, "-m", "spreadsheet_analyzer.cli.analyze"] + args

        # Set environment to use uv properly
        env = os.environ.copy()

        return subprocess.run(
            ["uv", "run"] + cmd,
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            env=env,
            timeout=300,  # 5 minute timeout
        )

    return run_cli


@pytest.fixture
def validate_notebook():
    """Fixture that provides notebook validation utilities."""
    import nbformat
    from nbformat.validator import validate as nb_validate

    def _validate_notebook(notebook_path: Path) -> dict[str, Any]:
        """
        Validate a Jupyter notebook file.

        Args:
            notebook_path: Path to the notebook file

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": False,
            "exists": False,
            "readable": False,
            "has_cells": False,
            "has_outputs": False,
            "error_cells": 0,
            "total_cells": 0,
            "markdown_cells": 0,
            "code_cells": 0,
            "issues": [],
        }

        try:
            # Check if file exists
            if not notebook_path.exists():
                results["issues"].append(f"Notebook file does not exist: {notebook_path}")
                return results

            results["exists"] = True

            # Try to read the notebook
            try:
                with open(notebook_path, encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)
                results["readable"] = True
            except Exception as e:
                results["issues"].append(f"Failed to read notebook: {e}")
                return results

            # Validate nbformat structure
            try:
                nb_validate(nb)
                results["valid"] = True
            except Exception as e:
                results["issues"].append(f"Invalid notebook format: {e}")

            # Analyze cells
            results["total_cells"] = len(nb.cells)
            results["has_cells"] = results["total_cells"] > 0

            for cell in nb.cells:
                if cell.cell_type == "markdown":
                    results["markdown_cells"] += 1
                elif cell.cell_type == "code":
                    results["code_cells"] += 1

                    # Check for outputs in code cells
                    if hasattr(cell, "outputs") and cell.outputs:
                        results["has_outputs"] = True

                        # Check for error outputs
                        for output in cell.outputs:
                            if output.get("output_type") == "error":
                                results["error_cells"] += 1
                                results["issues"].append(
                                    f"Error in cell: {output.get('ename', 'Unknown')}: "
                                    f"{output.get('evalue', 'No message')}"
                                )

            # Quality checks
            if results["code_cells"] == 0:
                results["issues"].append("No code cells found in notebook")

            if results["markdown_cells"] == 0:
                results["issues"].append("No markdown cells found in notebook")

            if not results["has_outputs"] and results["code_cells"] > 0:
                results["issues"].append("Code cells have no outputs (not executed)")

        except Exception as e:
            results["issues"].append(f"Unexpected error during validation: {e}")

        return results

    return _validate_notebook


@pytest.fixture
def get_test_file_info():
    """Fixture that provides information about test files."""

    def _get_info(filename: str) -> dict[str, Any]:
        """Get information about a test file."""
        if filename not in TEST_FILES:
            raise ValueError(f"Unknown test file: {filename}")

        info = TEST_FILES[filename].copy()
        file_path = Path(__file__).parent.parent.parent / "test_assets" / "generated" / filename

        # Auto-detect sheets for Excel files if needed
        if filename.endswith(".xlsx") and info["sheets"][0] is not None:
            try:
                actual_sheets = list_sheets(file_path)
                info["actual_sheets"] = actual_sheets
            except Exception as e:
                info["sheet_detection_error"] = str(e)
                info["actual_sheets"] = info["sheets"]  # Fallback
        else:
            info["actual_sheets"] = info["sheets"]

        info["file_path"] = file_path
        return info

    return _get_info


@pytest.fixture(scope="session", autouse=True)
def ensure_test_data_exists(test_data_dir):
    """Ensure test data files exist before running tests."""
    required_files = list(TEST_FILES.keys())
    missing_files = []

    for filename in required_files:
        file_path = test_data_dir / filename
        if not file_path.exists():
            missing_files.append(filename)

    if missing_files:
        pytest.fail(
            f"Missing test data files: {missing_files}. Run 'python test_assets/generated/create_test_files.py' first."
        )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "deterministic: marks tests that generate deterministic outputs")
    config.addinivalue_line("markers", "collection: marks tests that use the curated file collection")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark deterministic generation tests
        if "deterministic" in item.nodeid:
            item.add_marker(pytest.mark.deterministic)

        # Mark integration tests
        if any(keyword in item.nodeid for keyword in ["integration", "cli"]):
            item.add_marker(pytest.mark.integration)

        # Mark slow tests (CLI tests are generally slower)
        if "cli" in item.nodeid:
            item.add_marker(pytest.mark.slow)
