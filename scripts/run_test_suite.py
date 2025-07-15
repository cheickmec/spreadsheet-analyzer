#!/usr/bin/env python3
"""
Run comprehensive test suite on all test files and generate detailed report.

This script runs the full deterministic pipeline on all test files,
generates a JSON report with detailed results, and provides a summary
of successes, failures, and insights found.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spreadsheet_analyzer.test_pipeline import run_pipeline_tests


def main():
    """Run all tests and generate report."""
    project_root = Path(__file__).parent
    test_dir = project_root / "test-files"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / f"test_results_{timestamp}.json"

    print("\n" + "=" * 80)
    print("SPREADSHEET ANALYZER - DETERMINISTIC PIPELINE TEST SUITE")
    print("=" * 80)
    print(f"\nTest directory: {test_dir}")
    print(f"Output file: {output_file}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run tests
    results = run_pipeline_tests(test_dir, output_file)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Success rate
    success_rate = results.summary["successful"] / max(results.summary["total_files"], 1) * 100

    print(f"\nOverall Success Rate: {success_rate:.1f}%")
    print(f"Total Execution Time: {results.summary['total_time']:.2f} seconds")

    # Key findings
    if results.summary["errors"]:
        print(f"\n⚠️  Found {len(results.summary['errors'])} errors:")
        # Group errors by type
        error_types = {}
        for error_info in results.summary["errors"]:
            error_msg = error_info["error"]
            error_type = error_msg.split(":")[0] if ":" in error_msg else "Unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {error_type}: {count} occurrences")

    if results.summary["insights"]:
        print(f"\n💡 Generated {len(results.summary['insights'])} insights")

        # Count by severity
        severity_counts = {}
        for insight in results.summary["insights"]:
            severity = insight["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        for severity in ["HIGH", "MEDIUM", "LOW"]:
            if severity in severity_counts:
                print(f"   - {severity}: {severity_counts[severity]} insights")

    print(f"\n📊 Report saved to: {output_file}")
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Return exit code based on success
    return 0 if success_rate > 80 else 1


if __name__ == "__main__":
    sys.exit(main())
