"""
Test Harness for Running Pipeline on Test Files.

This script runs the deterministic analysis pipeline on all test files
and generates a comprehensive report of the results.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .pipeline.pipeline import DeterministicPipeline, create_console_progress_observer, create_lenient_pipeline_options
from .pipeline.types import PipelineResult

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==================== Test Result Collection ====================


class TestResults:
    """Collects and summarizes test results."""

    def __init__(self):
        """Initialize result collector."""
        self.results: list[dict[str, Any]] = []
        self.summary = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "blocked": 0,
            "total_time": 0.0,
            "by_category": {},
            "errors": [],
            "insights": [],
        }

    def add_result(self, file_path: Path, result: PipelineResult, category: str):
        """Add a single test result."""
        self.summary["total_files"] += 1

        # Track by category
        if category not in self.summary["by_category"]:
            self.summary["by_category"][category] = {"files": 0, "successful": 0, "failed": 0, "avg_time": 0.0}

        cat_summary = self.summary["by_category"][category]
        cat_summary["files"] += 1

        # Create result record
        record = {
            "file": str(file_path),
            "category": category,
            "success": result.success,
            "execution_time": result.execution_time,
            "timestamp": datetime.now().isoformat(),
        }

        if result.success:
            self.summary["successful"] += 1
            cat_summary["successful"] += 1

            # Add analysis details
            if result.integrity:
                record["integrity"] = {
                    "file_size_mb": result.integrity.metadata.size_mb,
                    "mime_type": result.integrity.metadata.mime_type,
                    "trust_tier": result.integrity.trust_tier,
                    "processing_class": result.integrity.processing_class,
                }

                if result.integrity.processing_class == "BLOCKED":
                    self.summary["blocked"] += 1

            if result.security:
                record["security"] = {
                    "risk_level": result.security.risk_level,
                    "risk_score": result.security.risk_score,
                    "has_macros": result.security.has_macros,
                    "threat_count": result.security.threat_count,
                }

            if result.structure:
                record["structure"] = {
                    "sheet_count": result.structure.sheet_count,
                    "total_cells": result.structure.total_cells,
                    "total_formulas": result.structure.total_formulas,
                    "complexity_score": result.structure.complexity_score,
                }

            if result.formulas:
                record["formulas"] = {
                    "has_circular_references": result.formulas.has_circular_references,
                    "max_dependency_depth": result.formulas.max_dependency_depth,
                    "volatile_formula_count": len(result.formulas.volatile_formulas),
                    "complexity_score": result.formulas.formula_complexity_score,
                }

            if result.content:
                record["content"] = {
                    "data_quality_score": result.content.data_quality_score,
                    "pattern_count": len(result.content.data_patterns),
                    "insight_count": len(result.content.insights),
                }

                # Collect insights
                for insight in result.content.insights:
                    self.summary["insights"].append(
                        {
                            "file": str(file_path),
                            "type": insight.insight_type,
                            "title": insight.title,
                            "severity": insight.severity,
                        }
                    )
        else:
            self.summary["failed"] += 1
            cat_summary["failed"] += 1
            record["errors"] = list(result.errors)

            # Track errors
            for error in result.errors:
                self.summary["errors"].append({"file": str(file_path), "error": error})

        self.summary["total_time"] += result.execution_time
        self.results.append(record)

        # Update average time
        cat_summary["avg_time"] = (
            cat_summary["avg_time"] * (cat_summary["files"] - 1) + result.execution_time
        ) / cat_summary["files"]

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive report."""
        return {"summary": self.summary, "results": self.results, "generated_at": datetime.now().isoformat()}


# ==================== Test File Discovery ====================


def discover_test_files(test_dir: Path) -> dict[str, list[Path]]:
    """
    Discover all Excel test files organized by category.

    Returns dict mapping category names to file lists.
    """
    categories = {}

    # Excel file extensions
    excel_extensions = {".xlsx", ".xls", ".xlsm", ".xlsb"}

    # Walk through test directory
    for subdir in test_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            excel_files = []

            # Find Excel files in subdirectory
            for file_path in subdir.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in excel_extensions:
                    excel_files.append(file_path)

            if excel_files:
                categories[subdir.name] = excel_files

    # Also check for files directly in test_dir
    root_files = [f for f in test_dir.iterdir() if f.is_file() and f.suffix.lower() in excel_extensions]

    if root_files:
        categories["uncategorized"] = root_files

    return categories


# ==================== Main Test Runner ====================


def run_pipeline_tests(
    test_dir: Path, output_file: Optional[Path] = None, options: Optional[dict[str, Any]] = None
) -> TestResults:
    """
    Run pipeline on all test files.

    Args:
        test_dir: Directory containing test files
        output_file: Optional path to save results JSON
        options: Pipeline options

    Returns:
        TestResults object with all results
    """
    print(f"\n{'=' * 60}")
    print("Excel Analyzer Pipeline Test Harness")
    print(f"{'=' * 60}\n")

    # Discover test files
    print("Discovering test files...")
    categories = discover_test_files(test_dir)

    total_files = sum(len(files) for files in categories.values())
    print(f"Found {total_files} Excel files in {len(categories)} categories")

    for category, files in categories.items():
        print(f"  {category}: {len(files)} files")
    print()

    # Initialize pipeline and results
    pipeline = DeterministicPipeline(options or create_lenient_pipeline_options())
    pipeline.add_progress_observer(create_console_progress_observer())
    results = TestResults()

    # Process each category
    for category, files in categories.items():
        print(f"\n{'=' * 60}")
        print(f"Processing category: {category}")
        print(f"{'=' * 60}\n")

        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] Analyzing: {file_path.name}")
            print(f"File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")

            start_time = time.time()

            try:
                result = pipeline.run(file_path)
                results.add_result(file_path, result, category)

                # Print summary
                if result.success:
                    print(f"✓ Analysis completed in {result.execution_time:.2f}s")

                    if result.integrity:
                        print(f"  - Processing class: {result.integrity.processing_class}")
                    if result.security:
                        print(f"  - Security risk: {result.security.risk_level}")
                    if result.structure:
                        print(f"  - Sheets: {result.structure.sheet_count}, Cells: {result.structure.total_cells:,}")
                    if result.content:
                        print(f"  - Data quality: {result.content.data_quality_score}%")
                else:
                    print(f"✗ Analysis failed: {result.errors[0] if result.errors else 'Unknown error'}")

            except Exception as e:
                logger.exception(f"Failed to analyze {file_path}")
                print(f"✗ Exception: {e!s}")

                # Create failed result
                failed_result = PipelineResult(
                    context=None,
                    integrity=None,
                    security=None,
                    structure=None,
                    formulas=None,
                    content=None,
                    execution_time=time.time() - start_time,
                    success=False,
                    errors=(f"Exception: {e!s}",),
                )
                results.add_result(file_path, failed_result, category)

    # Print final summary
    print(f"\n{'=' * 60}")
    print("Test Summary")
    print(f"{'=' * 60}\n")

    print(f"Total files processed: {results.summary['total_files']}")
    print(f"Successful: {results.summary['successful']}")
    print(f"Failed: {results.summary['failed']}")
    print(f"Blocked: {results.summary['blocked']}")
    print(f"Total time: {results.summary['total_time']:.2f}s")
    print(f"Average time: {results.summary['total_time'] / max(results.summary['total_files'], 1):.2f}s")

    print("\nBy Category:")
    for category, cat_summary in results.summary["by_category"].items():
        print(f"  {category}:")
        print(f"    - Files: {cat_summary['files']}")
        print(f"    - Success rate: {cat_summary['successful'] / max(cat_summary['files'], 1) * 100:.1f}%")
        print(f"    - Avg time: {cat_summary['avg_time']:.2f}s")

    # Print insights summary
    if results.summary["insights"]:
        print(f"\nKey Insights Found ({len(results.summary['insights'])} total):")

        # Group by severity
        by_severity = {}
        for insight in results.summary["insights"]:
            severity = insight["severity"]
            by_severity.setdefault(severity, []).append(insight)

        for severity in ["HIGH", "MEDIUM", "LOW"]:
            if severity in by_severity:
                print(f"  {severity}: {len(by_severity[severity])} insights")

    # Save results if requested
    if output_file:
        print(f"\nSaving results to {output_file}...")
        report = results.generate_report()

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        print("Results saved successfully")

    return results


# ==================== Entry Point ====================


def main():
    """Main entry point for test harness."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    test_dir = project_root / "test-files"
    output_file = project_root / f"pipeline_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Check test directory exists
    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        sys.exit(1)

    # Run tests
    try:
        results = run_pipeline_tests(test_dir, output_file)

        # Exit with error code if any failures
        if results.summary["failed"] > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("Test harness failed")
        print(f"\nFatal error: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
