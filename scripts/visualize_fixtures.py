#!/usr/bin/env python3
"""
Visualize and explore captured test file outputs.

This script provides visibility into what each test file produces when
processed through the analyzer pipeline.
"""

import argparse
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spreadsheet_analyzer.testing.fixtures import FixtureManager


class FixtureVisualizer:
    """Visualize captured fixtures from test files."""

    def __init__(self):
        """Initialize the visualizer."""
        self.fixture_manager = FixtureManager()
        self.captured_outputs_dir = self.fixture_manager.base_path / "captured_outputs"
        self.manifest_path = self.captured_outputs_dir / "fixture_manifest.json"

        # Load manifest
        if self.manifest_path.exists():
            with self.manifest_path.open() as f:
                self.manifest = json.load(f)
        else:
            self.manifest = {"fixtures": []}

    def list_all_fixtures(self, show_details: bool = False) -> None:
        """List all available fixtures with optional details."""
        fixtures = self.manifest.get("fixtures", [])

        if not fixtures:
            print("No fixtures found.")
            return

        # Group by directory
        by_category = defaultdict(list)
        for fixture in fixtures:
            test_file = fixture.get("test_file", "unknown")
            category = test_file.split("/")[0] if "/" in test_file else "root"
            by_category[category].append(fixture)

        # Display by category
        print(f"\n{'=' * 80}")
        print(f"CAPTURED TEST FILE OUTPUTS ({len(fixtures)} total)")
        print(f"{'=' * 80}\n")

        for category, category_fixtures in sorted(by_category.items()):
            print(f"\n{category.upper()} ({len(category_fixtures)} files)")
            print("-" * len(f"{category.upper()} ({len(category_fixtures)} files)"))

            for fixture in sorted(category_fixtures, key=lambda x: x.get("test_file", "")):
                self._display_fixture_summary(fixture, show_details)

    def _display_fixture_summary(self, fixture_info: dict[str, Any], show_details: bool) -> None:
        """Display summary for a single fixture."""
        test_file = fixture_info.get("test_file", "unknown")
        file_name = test_file.split("/")[-1] if "/" in test_file else test_file

        # Status indicator
        status = "✅" if fixture_info.get("pipeline_success") else "❌"

        # Basic info
        print(f"\n  {status} {file_name}")
        print(f"     Size: {self._format_size(fixture_info.get('file_size', 0))}")

        if show_details:
            # Load full fixture data
            fixture_path = self.captured_outputs_dir / fixture_info["fixture_path"]
            with fixture_path.open() as f:
                fixture_data = json.load(f)

            if fixture_info.get("pipeline_success"):
                self._display_success_details(fixture_data)
            else:
                self._display_error_details(fixture_data)

    def _display_success_details(self, fixture_data: dict[str, Any]) -> None:
        """Display details for a successful processing."""
        pipeline_result = fixture_data.get("pipeline_result", {})

        # Execution time
        exec_time = pipeline_result.get("execution_time", 0)
        print(f"     Time: {exec_time:.2f}s")

        # Stage summaries
        stages_completed = []

        if pipeline_result.get("integrity"):
            integrity = pipeline_result["integrity"]
            stages_completed.append(f"Integrity (tier {integrity.get('trust_tier', '?')})")

        if pipeline_result.get("security"):
            security = pipeline_result["security"]
            risk = security.get("risk_level", "UNKNOWN")
            stages_completed.append(f"Security ({risk} risk)")

        if pipeline_result.get("structure"):
            structure = pipeline_result["structure"]
            sheets = len(structure.get("sheets", []))
            cells = structure.get("total_cells", 0)
            stages_completed.append(f"Structure ({sheets} sheets, {cells:,} cells)")

        if pipeline_result.get("formulas"):
            formulas = pipeline_result["formulas"]
            has_circular = formulas.get("has_circular_references", False)
            circular_text = " ⚠️ circular refs" if has_circular else ""
            stages_completed.append(f"Formulas{circular_text}")

        if pipeline_result.get("content"):
            content = pipeline_result["content"]
            quality = content.get("data_quality_score", 0)
            stages_completed.append(f"Content (quality: {quality}%)")

        if stages_completed:
            print(f"     Stages: {', '.join(stages_completed)}")

    def _display_error_details(self, fixture_data: dict[str, Any]) -> None:
        """Display details for a failed processing."""
        error_info = fixture_data.get("error", {})
        error_type = error_info.get("type", "Unknown")
        error_msg = error_info.get("message", "No message")

        # Truncate long error messages
        if len(error_msg) > 60:
            error_msg = error_msg[:57] + "..."

        print(f"     Error: {error_type}")
        print(f"     Message: {error_msg}")

    def show_fixture_details(self, test_file: str) -> None:
        """Show detailed information for a specific test file."""
        # Find the fixture
        fixture_info = None
        for f in self.manifest.get("fixtures", []):
            if f.get("test_file") == test_file:
                fixture_info = f
                break

        if not fixture_info:
            print(f"No fixture found for: {test_file}")
            return

        # Load full fixture data
        fixture_path = self.captured_outputs_dir / fixture_info["fixture_path"]
        with fixture_path.open() as f:
            fixture_data = json.load(f)

        # Display header
        print(f"\n{'=' * 80}")
        print(f"FIXTURE DETAILS: {test_file}")
        print(f"{'=' * 80}\n")

        # Metadata
        metadata = fixture_data.get("metadata", {})
        print("METADATA:")
        print(f"  File Size: {self._format_size(metadata.get('file_size', 0))}")
        print(f"  Processing Time: {metadata.get('processing_time', 0):.3f}s")
        print(f"  Captured: {metadata.get('capture_timestamp', 'unknown')}")
        print(f"  File Hash: {metadata.get('file_hash', 'unknown')[:16]}...")
        print(f"  Success: {metadata.get('pipeline_success', False)}")

        if metadata.get("pipeline_success"):
            self._show_successful_result_details(fixture_data)
        else:
            self._show_error_result_details(fixture_data)

    def _show_successful_result_details(self, fixture_data: dict[str, Any]) -> None:
        """Show details for a successful pipeline run."""
        pipeline_result = fixture_data.get("pipeline_result", {})

        # Stage 0: Integrity
        if pipeline_result.get("integrity"):
            print("\n\nSTAGE 0 - INTEGRITY:")
            integrity = pipeline_result["integrity"]
            print(f"  File Hash: {integrity.get('file_hash', 'unknown')[:32]}...")
            print(f"  Is Excel: {integrity.get('is_excel', False)}")
            print(f"  Is OOXML: {integrity.get('is_ooxml', False)}")
            print(f"  Trust Tier: {integrity.get('trust_tier', 'unknown')}")
            print(f"  Processing Class: {integrity.get('processing_class', 'unknown')}")
            print(f"  Validation Passed: {integrity.get('validation_passed', False)}")

        # Stage 1: Security
        if pipeline_result.get("security"):
            print("\n\nSTAGE 1 - SECURITY:")
            security = pipeline_result["security"]
            print(f"  Risk Level: {security.get('risk_level', 'unknown')}")
            print(f"  Risk Score: {security.get('risk_score', 0)}/100")
            print(f"  Has Macros: {security.get('has_macros', False)}")
            print(f"  Has External Links: {security.get('has_external_links', False)}")
            print(f"  Has Embedded Objects: {security.get('has_embedded_objects', False)}")
            print(f"  Threat Count: {security.get('threat_count', 0)}")

            threats = security.get("threats", [])
            if threats:
                print(f"\n  Threats Found ({len(threats)}):")
                for threat in threats[:3]:  # Show first 3
                    print(f"    - {threat.get('threat_type', 'unknown')}: {threat.get('description', '')}")

        # Stage 2: Structure
        if pipeline_result.get("structure"):
            print("\n\nSTAGE 2 - STRUCTURE:")
            structure = pipeline_result["structure"]
            print(f"  Sheet Count: {structure.get('sheet_count', 0)}")
            print(f"  Total Cells: {structure.get('total_cells', 0):,}")
            print(f"  Total Formulas: {structure.get('total_formulas', 0):,}")
            print(f"  Complexity Score: {structure.get('complexity_score', 0)}/100")
            print(f"  Has VBA Project: {structure.get('has_vba_project', False)}")
            print(f"  Has External Links: {structure.get('has_external_links', False)}")

            sheets = structure.get("sheets", [])
            if sheets:
                print(f"\n  Sheets ({len(sheets)}):")
                for sheet in sheets[:5]:  # Show first 5
                    print(
                        f"    - {sheet.get('name', 'unnamed')}: "
                        f"{sheet.get('row_count', 0)} rows x {sheet.get('column_count', 0)} cols, "
                        f"{sheet.get('cell_count', 0):,} cells"
                    )

        # Stage 3: Formulas
        if pipeline_result.get("formulas"):
            print("\n\nSTAGE 3 - FORMULAS:")
            formulas = pipeline_result["formulas"]
            print(f"  Has Circular References: {formulas.get('has_circular_references', False)}")
            print(f"  Max Dependency Depth: {formulas.get('max_dependency_depth', 0)}")
            print(f"  Formula Complexity Score: {formulas.get('formula_complexity_score', 0)}")

            circular_refs = formulas.get("circular_references", [])
            if circular_refs:
                print(f"  Circular References ({len(circular_refs)}):")
                for ref in circular_refs[:3]:
                    print(f"    - {' -> '.join(ref)}")

            volatile_formulas = formulas.get("volatile_formulas", [])
            if volatile_formulas:
                print(f"  Volatile Formulas ({len(volatile_formulas)}):")
                for vf in volatile_formulas[:3]:
                    print(f"    - {vf}")

        # Stage 4: Content
        if pipeline_result.get("content"):
            print("\n\nSTAGE 4 - CONTENT:")
            content = pipeline_result["content"]
            print(f"  Data Quality Score: {content.get('data_quality_score', 0)}/100")
            print(f"  Pattern Count: {len(content.get('data_patterns', []))}")
            print(f"  Insight Count: {len(content.get('insights', []))}")

            summary = content.get("summary", "")
            if summary:
                print("\n  Summary:")
                wrapped = textwrap.wrap(summary, width=70)
                for line in wrapped[:3]:  # Show first 3 lines
                    print(f"    {line}")

            insights = content.get("insights", [])
            if insights:
                print(f"\n  Key Insights ({len(insights)}):")
                for insight in insights[:3]:
                    print(f"    - {insight.get('title', 'Untitled')}")
                    print(
                        f"      Type: {insight.get('insight_type', 'unknown')}, "
                        f"Severity: {insight.get('severity', 'unknown')}"
                    )

    def _show_error_result_details(self, fixture_data: dict[str, Any]) -> None:
        """Show details for a failed pipeline run."""
        error_info = fixture_data.get("error", {})

        print("\n\nERROR DETAILS:")
        print(f"  Type: {error_info.get('type', 'Unknown')}")
        print(f"  Stage: {error_info.get('stage', 'Unknown')}")
        print("\n  Message:")

        message = error_info.get("message", "No message")
        wrapped = textwrap.wrap(message, width=70)
        for line in wrapped:
            print(f"    {line}")

        # Check if there's a partial pipeline result
        if "pipeline_result" in fixture_data:
            pipeline_result = fixture_data["pipeline_result"]
            if any(pipeline_result.get(stage) for stage in ["integrity", "security", "structure"]):
                print("\n\nPARTIAL RESULTS (before failure):")
                self._show_successful_result_details(fixture_data)

    def generate_summary_report(self) -> None:
        """Generate a comprehensive summary report of all fixtures."""
        fixtures = self.manifest.get("fixtures", [])

        if not fixtures:
            print("No fixtures to summarize.")
            return

        # Gather statistics
        total_files = len(fixtures)
        successful = sum(1 for f in fixtures if f.get("pipeline_success", False))
        failed = total_files - successful

        # Group by category
        by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "failed": 0, "total_size": 0})

        for fixture in fixtures:
            test_file = fixture.get("test_file", "unknown")
            category = test_file.split("/")[0] if "/" in test_file else "root"

            by_category[category]["total_size"] += fixture.get("file_size", 0)
            if fixture.get("pipeline_success", False):
                by_category[category]["success"] += 1
            else:
                by_category[category]["failed"] += 1

        # Display report
        print(f"\n{'=' * 80}")
        print("FIXTURE SUMMARY REPORT")
        print(f"{'=' * 80}\n")

        print(f"Total Files: {total_files}")
        print(f"Successful: {successful} ({successful / total_files * 100:.1f}%)")
        print(f"Failed: {failed} ({failed / total_files * 100:.1f}%)")

        print("\n\nBY CATEGORY:")
        print(f"{'Category':<25} {'Success':<10} {'Failed':<10} {'Total Size':<15}")
        print("-" * 60)

        for category, stats in sorted(by_category.items()):
            total = stats["success"] + stats["failed"]
            success_rate = stats["success"] / total * 100 if total > 0 else 0
            print(
                f"{category:<25} {stats['success']:<10} {stats['failed']:<10} "
                f"{self._format_size(stats['total_size']):<15} ({success_rate:.0f}%)"
            )

        # Find common issues
        print("\n\nCOMMON ISSUES:")
        error_types: dict[str, int] = defaultdict(int)

        for fixture in fixtures:
            if not fixture.get("pipeline_success", False):
                # Load fixture to get error details
                fixture_path = self.captured_outputs_dir / fixture["fixture_path"]
                try:
                    with fixture_path.open() as f:
                        data = json.load(f)
                    error_type = data.get("error", {}).get("type", "Unknown")
                    error_types[error_type] += 1
                except Exception:
                    pass

        if error_types:
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {error_type}: {count} occurrences")
        else:
            print("  No errors found!")

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        size_float = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB"]:
            if size_float < 1024.0:
                return f"{size_float:.1f} {unit}"
            size_float /= 1024.0
        return f"{size_float:.1f} TB"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize captured test file outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all fixtures with basic info
  %(prog)s list

  # List all fixtures with detailed info
  %(prog)s list --details

  # Show detailed output for a specific test file
  %(prog)s show "business-accounting/Business Accounting.xlsx"

  # Generate summary report
  %(prog)s summary
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List all fixtures")
    list_parser.add_argument("--details", action="store_true", help="Show detailed information")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show details for specific test file")
    show_parser.add_argument("test_file", help='Test file path (e.g., "category/file.xlsx")')

    # Summary command
    subparsers.add_parser("summary", help="Generate summary report")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    visualizer = FixtureVisualizer()

    if args.command == "list":
        visualizer.list_all_fixtures(show_details=args.details)
    elif args.command == "show":
        visualizer.show_fixture_details(args.test_file)
    elif args.command == "summary":
        visualizer.generate_summary_report()


if __name__ == "__main__":
    main()
