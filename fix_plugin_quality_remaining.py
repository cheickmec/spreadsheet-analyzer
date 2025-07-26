#!/usr/bin/env python
"""Fix remaining issues in plugin quality tests."""

from pathlib import Path


def fix_plugin_quality_remaining():
    """Fix remaining issues in plugin quality tests."""

    test_file = Path("tests/plugins/spreadsheet/test_quality.py")

    # Read the content
    content = test_file.read_text()

    # Fix test_quality_metrics_structure score range assertion
    content = content.replace(
        "assert 0.0 <= metrics.overall_score <= 1.0", "assert 0.0 <= metrics.overall_score <= 100.0"
    )

    # Fix the hasattr check for issue.level -> issue.severity
    content = content.replace("assert hasattr(issue, 'level')", "assert hasattr(issue, 'severity')")

    # Fix quality comparison test score assertions
    content = content.replace("assert results['high_quality'] > 0.6", "assert results['high_quality'] > 60.0")
    content = content.replace("assert results['low_quality'] < 0.5", "assert results['low_quality'] < 50.0")

    # Write back the content
    test_file.write_text(content)
    print(f"Fixed remaining issues in {test_file}")


if __name__ == "__main__":
    fix_plugin_quality_remaining()
