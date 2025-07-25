"""Tests the analyzer against the curated collection of real-world files.

This module uses the manifest.json to dynamically generate test cases for each
file in the collection. This ensures that as new files are added to the
collection, they are automatically included in the test suite.
"""

from pathlib import Path
from typing import Any

import pytest


def pytest_generate_tests(metafunc: Any) -> None:
    """
    Dynamically parameterizes tests that request the 'collection_file' fixture.

    This function is a pytest hook that reads the collection_manifest and
    creates a unique test run for each valid file found in the collection.
    """
    if "collection_file" in metafunc.fixturenames:
        # Get fixtures from the configuration
        manifest = metafunc.config.getfixturevalue("collection_manifest")
        collection_dir = metafunc.config.getfixturevalue("collection_files_dir")

        all_files_info: list[dict[str, Any]] = []
        test_ids: list[str] = []

        # Iterate through the manifest to find all available files
        for category, files in manifest.items():
            if isinstance(files, list):
                for file_info in files:
                    file_path = collection_dir / category / file_info["file"]
                    # Only add the file to the test run if it actually exists
                    if file_path.exists():
                        # Bundle all relevant info into a dictionary for the test
                        test_param = {
                            "path": file_path,
                            "category": category,
                            **file_info,
                        }
                        all_files_info.append(test_param)
                        test_ids.append(f"{category}-{file_info['file']}")

        # Parameterize the test with all available files
        metafunc.parametrize("collection_file", all_files_info, ids=test_ids)


@pytest.mark.slow
@pytest.mark.collection
def test_analyzer_on_collection_file(
    cli_runner: Any,
    collection_file: dict[str, Any],
    temp_output_dir: Path,
    validate_notebook: Any,
) -> None:
    """
    Runs a deterministic analysis on a single file from the collection.

    This test is automatically parameterized by `pytest_generate_tests` to run
    for every file in the collection manifest.

    Args:
        cli_runner: Fixture to run CLI commands.
        collection_file: A dictionary containing metadata and the path for one file.
        temp_output_dir: A temporary directory for output artifacts.
        validate_notebook: Fixture for notebook validation.
    """
    file_path = collection_file["path"]
    category = collection_file["category"]
    print(f"\nTesting collection file: {category}/{file_path.name}")

    # Run the CLI analyzer in deterministic mode
    args = [
        str(file_path),
        "--no-llm",  # Use deterministic mode for consistency
        "--output-dir",
        str(temp_output_dir),
    ]

    result = cli_runner(args)

    # Basic CLI execution check
    assert result.returncode == 0, (
        f"CLI failed for {category}/{file_path.name}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    # Verify that an output directory was created
    output_dir = temp_output_dir / file_path.stem
    assert output_dir.exists(), f"Output directory was not created: {output_dir}"

    # Find and validate any generated notebooks
    notebooks = list(output_dir.glob("*.ipynb"))
    assert len(notebooks) > 0, f"No notebooks were generated in {output_dir}"

    for notebook_path in notebooks:
        validation = validate_notebook(notebook_path)
        assert validation["valid"], f"Generated notebook is invalid: {notebook_path}\nIssues: {validation['issues']}"
        assert validation["error_cells"] == 0, (
            f"Notebook has {validation['error_cells']} error cells: {notebook_path}\nIssues: {validation['issues']}"
        )

    print(f"✅ Successfully analyzed {category}/{file_path.name}")


@pytest.mark.collection
@pytest.mark.integration
def test_collection_manifest_integrity(collection_files_dir: Path, collection_manifest: dict[str, Any]) -> None:
    """
    Test that the collection manifest is valid and all referenced files exist.

    This test ensures that:
    1. The manifest.json is properly structured
    2. All referenced files actually exist in the collection
    3. The directory structure matches the manifest
    """
    # Check that manifest is not empty
    assert collection_manifest, "Collection manifest is empty"

    # Track files for validation
    total_files = 0
    missing_files = []

    for category, files in collection_manifest.items():
        if isinstance(files, list):
            category_dir = collection_files_dir / category

            # Check that category directory exists
            if not category_dir.exists():
                missing_files.append(f"Category directory missing: {category}")
                continue

            for file_info in files:
                total_files += 1
                file_path = category_dir / file_info["file"]

                if not file_path.exists():
                    missing_files.append(f"{category}/{file_info['file']}")

    # Report results
    if missing_files:
        pytest.fail(
            f"Found {len(missing_files)} missing files out of {total_files} total:\n"
            + "\n".join(f"  - {f}" for f in missing_files)
        )

    print(f"✅ Collection manifest integrity verified: {total_files} files")


@pytest.mark.collection
def test_collection_categories_exist(collection_files_dir: Path, collection_manifest: dict[str, Any]) -> None:
    """
    Test that all expected collection categories have corresponding directories.
    """
    expected_categories = ["business-accounting", "data-analysis", "financial-models", "hr-timesheet", "edge-cases"]

    existing_categories = set()
    for category in collection_manifest:
        if isinstance(collection_manifest[category], list):
            existing_categories.add(category)

    missing_categories = set(expected_categories) - existing_categories
    if missing_categories:
        print(f"⚠️  Expected categories not found in manifest: {missing_categories}")

    # Check directory existence
    missing_dirs = []
    for category in existing_categories:
        category_dir = collection_files_dir / category
        if not category_dir.exists():
            missing_dirs.append(category)

    assert not missing_dirs, f"Category directories missing: {missing_dirs}"

    print(f"✅ Collection categories verified: {len(existing_categories)} categories")
