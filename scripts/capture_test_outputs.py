#!/usr/bin/env python3
"""
Capture outputs from processing test Excel files and store them as test fixtures.

This script processes all Excel files in the test-files/ directory through the
analyzer pipeline and captures their outputs as fixtures for use in unit tests.
"""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spreadsheet_analyzer.pipeline.pipeline import DeterministicPipeline, create_lenient_pipeline_options
from spreadsheet_analyzer.pipeline.types import PipelineResult
from spreadsheet_analyzer.testing.fixtures import FixtureEncoder, FixtureManager


class FixtureError(Exception):
    """Base exception for fixture-related errors."""


class InvalidSummaryStructureError(FixtureError):
    """Raised when summary dictionary has invalid structure."""

    def __init__(self, field_name: str) -> None:
        """Initialize with the field that has invalid type."""
        super().__init__(f"{field_name} must be a list")


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TestOutputCapturer:
    """Captures outputs from test Excel files and stores them as fixtures."""

    def __init__(self, test_files_dir: Path, *, force_update: bool = False):
        """
        Initialize the capturer.

        Args:
            test_files_dir: Directory containing test Excel files
            force_update: Force update even if fixtures exist
        """
        self.test_files_dir = test_files_dir
        self.force_update = force_update
        self.fixture_manager = FixtureManager()
        self.pipeline = DeterministicPipeline(create_lenient_pipeline_options())

        # Create organized fixture directories
        self.fixtures_base = self.fixture_manager.base_path
        self.captured_outputs_dir = self.fixtures_base / "captured_outputs"
        self.captured_outputs_dir.mkdir(parents=True, exist_ok=True)

        # Track processing results
        self.results: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []

    def find_test_files(self) -> list[Path]:
        """Find all Excel test files in the test directory."""
        excel_extensions = {".xlsx", ".xlsm", ".xls"}
        test_files: list[Path] = []

        for ext in excel_extensions:
            test_files.extend(self.test_files_dir.rglob(f"*{ext}"))

        # Sort for consistent processing order
        return sorted(test_files)

    def get_fixture_path(self, test_file: Path) -> Path:
        """Get the fixture path for a test file."""
        # Create a hierarchical structure matching the test file structure
        relative_path = test_file.relative_to(self.test_files_dir)
        fixture_name = relative_path.with_suffix(".json")
        fixture_path: Path = self.captured_outputs_dir / fixture_name
        return fixture_path

    def should_process_file(self, test_file: Path) -> bool:
        """Check if a file should be processed."""
        fixture_path = self.get_fixture_path(test_file)

        if self.force_update:
            return True

        if not fixture_path.exists():
            return True

        # Check if test file is newer than fixture
        test_mtime = test_file.stat().st_mtime
        fixture_mtime = fixture_path.stat().st_mtime

        return test_mtime > fixture_mtime

    def capture_file_output(self, test_file: Path) -> dict[str, Any]:
        """
        Process a single test file and capture its output.

        Returns:
            Dictionary containing the captured output and metadata
        """
        logger.info("Processing: %s", test_file.relative_to(self.test_files_dir))

        start_time = datetime.now(tz=UTC)

        try:
            # Run the analysis pipeline
            result = self.pipeline.run(test_file)

            # Capture the complete output
            output_data = self._serialize_pipeline_result(result)

            # Add metadata
            metadata = {
                "test_file": str(test_file.relative_to(self.test_files_dir)),
                "file_size": test_file.stat().st_size,
                "processing_time": (datetime.now(tz=UTC) - start_time).total_seconds(),
                "capture_timestamp": datetime.now(tz=UTC).isoformat(),
                "pipeline_success": result.success,
                "error_count": len(result.errors),
                "file_hash": self._compute_file_hash(test_file),
            }

            return {
                "metadata": metadata,
                "pipeline_result": output_data,
                "raw_report": result.to_report() if result else None,
            }

        except Exception as e:
            logger.exception("Failed to process %s", test_file)

            # Capture error information
            return {
                "metadata": {
                    "test_file": str(test_file.relative_to(self.test_files_dir)),
                    "file_size": test_file.stat().st_size,
                    "processing_time": (datetime.now(tz=UTC) - start_time).total_seconds(),
                    "capture_timestamp": datetime.now(tz=UTC).isoformat(),
                    "pipeline_success": False,
                    "error_count": 1,
                    "file_hash": self._compute_file_hash(test_file),
                },
                "error": {"type": type(e).__name__, "message": str(e), "stage": "pipeline_execution"},
            }

    def _serialize_pipeline_result(self, result: PipelineResult) -> dict[str, Any]:
        """Serialize a PipelineResult to a dictionary."""

        # Convert to plain dict without Python-specific type markers
        # This ensures language-agnostic JSON that any language can consume
        result_dict = asdict(result)

        # Clean up Python-specific types for cross-language compatibility
        cleaned: dict[str, Any] = self._clean_for_json(result_dict)
        return cleaned

    def _clean_for_json(self, obj: Any) -> Any:
        """Clean Python-specific types for language-agnostic JSON."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list | tuple):
            return [self._clean_for_json(item) for item in obj]
        if isinstance(obj, set | frozenset):
            return sorted([self._clean_for_json(item) for item in obj])
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        if hasattr(obj, "__str__") and not isinstance(obj, str | int | float | bool):
            return str(obj)
        return obj

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with file_path.open("rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def save_fixture(self, test_file: Path, output_data: dict[str, Any]) -> None:
        """Save captured output as a fixture."""
        fixture_path = self.get_fixture_path(test_file)

        # Ensure parent directory exists
        fixture_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with custom encoder
        with fixture_path.open("w") as f:
            json.dump(output_data, f, indent=2, cls=FixtureEncoder, sort_keys=True)

        logger.info("Saved fixture: %s", fixture_path.relative_to(self.fixtures_base))

    def generate_summary_report(self) -> dict[str, Any]:
        """Generate a summary report of all captured outputs."""
        summary = {
            "capture_timestamp": datetime.now(tz=UTC).isoformat(),
            "total_files": len(self.results) + len(self.errors),
            "successful": len(self.results),
            "failed": len(self.errors),
            "fixtures_directory": str(self.captured_outputs_dir),
            "results": [],
            "errors": [],
        }

        # Add successful results
        results_list = summary["results"]
        # Type narrowing for mypy
        if not isinstance(results_list, list):
            raise InvalidSummaryStructureError("results_list")
        for result in self.results:
            results_list.append(
                {
                    "file": result["metadata"]["test_file"],
                    "size": result["metadata"]["file_size"],
                    "processing_time": result["metadata"]["processing_time"],
                    "stages_completed": self._count_completed_stages(result.get("pipeline_result", {})),
                }
            )

        # Add errors
        errors_list = summary["errors"]
        # Type narrowing for mypy
        if not isinstance(errors_list, list):
            raise InvalidSummaryStructureError("errors_list")
        for error in self.errors:
            errors_list.append(
                {
                    "file": error["metadata"]["test_file"],
                    "error_type": error.get("error", {}).get("type", "Unknown"),
                    "error_message": error.get("error", {}).get("message", "Unknown error"),
                }
            )

        return summary

    def _count_completed_stages(self, pipeline_result: dict[str, Any]) -> int:
        """Count how many stages were completed successfully."""
        stages = ["integrity", "security", "structure", "formulas", "content"]
        return sum(1 for stage in stages if pipeline_result.get(stage) is not None)

    def capture_all(self) -> None:
        """Capture outputs for all test files."""
        test_files = self.find_test_files()

        if not test_files:
            logger.warning("No test files found")
            return

        logger.info("Found %d test files", len(test_files))

        # Process each file
        for i, test_file in enumerate(test_files, 1):
            logger.info("\n[%d/%d] Processing %s", i, len(test_files), test_file.name)

            if not self.should_process_file(test_file):
                logger.info("Skipping (fixture up to date)")
                continue

            # Capture output
            output_data = self.capture_file_output(test_file)

            # Save fixture
            self.save_fixture(test_file, output_data)

            # Track results
            if output_data.get("error"):
                self.errors.append(output_data)
            else:
                self.results.append(output_data)

        # Generate and save summary report
        summary = self.generate_summary_report()
        summary_path = self.captured_outputs_dir / "capture_summary.json"

        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        logger.info("\nCapture complete!")
        logger.info("Successful: %d", len(self.results))
        logger.info("Failed: %d", len(self.errors))
        logger.info("Summary saved to: %s", summary_path)


def create_fixture_manifest(captured_outputs_dir: Path) -> None:
    """Create a manifest file listing all fixtures with metadata."""
    manifest = {"created": datetime.now(tz=UTC).isoformat(), "fixtures": []}

    for fixture_file in captured_outputs_dir.rglob("*.json"):
        if fixture_file.name in {"capture_summary.json", "fixture_manifest.json"}:
            continue

        with fixture_file.open() as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        fixtures_list = manifest["fixtures"]
        # Type narrowing for mypy
        if not isinstance(fixtures_list, list):
            raise InvalidSummaryStructureError("fixtures_list")
        fixtures_list.append(
            {
                "fixture_path": str(fixture_file.relative_to(captured_outputs_dir)),
                "test_file": metadata.get("test_file"),
                "file_size": metadata.get("file_size"),
                "pipeline_success": metadata.get("pipeline_success"),
                "error_count": metadata.get("error_count", 0),
                "capture_timestamp": metadata.get("capture_timestamp"),
            }
        )

    manifest_path = captured_outputs_dir / "fixture_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Created fixture manifest with %d entries", len(manifest["fixtures"]))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Capture outputs from test Excel files as fixtures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture outputs for all test files (skip existing)
  %(prog)s

  # Force update all fixtures
  %(prog)s --force

  # Capture with custom test files directory
  %(prog)s --test-dir /path/to/test/files

  # Generate only the manifest from existing fixtures
  %(prog)s --manifest-only
        """,
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(__file__).parent.parent / "test-files",
        help="Directory containing test Excel files (default: test-files/)",
    )

    parser.add_argument("--force", action="store_true", help="Force update fixtures even if they exist")

    parser.add_argument(
        "--manifest-only", action="store_true", help="Only generate the fixture manifest from existing fixtures"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check test directory exists
    if not args.test_dir.exists():
        logger.error("Test directory not found: %s", args.test_dir)
        sys.exit(1)

    if args.manifest_only:
        # Just create the manifest
        fm = FixtureManager()
        captured_outputs_dir = fm.base_path / "captured_outputs"
        create_fixture_manifest(captured_outputs_dir)
    else:
        # Run the capture process
        capturer = TestOutputCapturer(args.test_dir, force_update=args.force)
        capturer.capture_all()

        # Create manifest
        create_fixture_manifest(capturer.captured_outputs_dir)


if __name__ == "__main__":
    main()
