#!/usr/bin/env python
"""Fix plugin quality test files to match restored API."""

import re
from pathlib import Path


def fix_plugin_quality_tests():
    """Fix plugin quality tests to match restored API."""

    test_file = Path("tests/plugins/spreadsheet/test_quality.py")

    # Read the content
    content = test_file.read_text()

    # Replace .build() calls with direct builder usage
    replacements = [
        # Simple replacements where variable = builder.build()
        (r"notebook = self\.builder\.build\(\)", "notebook = self.builder"),
        (r"empty_notebook = self\.builder\.build\(\)", "empty_notebook = self.builder"),
        # Replacements in scenarios
        (
            r"scenarios\.append\(\('high_quality', builder1\.build\(\)\)\)",
            "scenarios.append(('high_quality', builder1))",
        ),
        (
            r"scenarios\.append\(\('medium_quality', builder2\.build\(\)\)\)",
            "scenarios.append(('medium_quality', builder2))",
        ),
        (r"scenarios\.append\(\('low_quality', builder3\.build\(\)\)\)", "scenarios.append(('low_quality', builder3))"),
        # Replace other builder.build() patterns
        (r"notebook = builder\.build\(\)", "notebook = builder"),
    ]

    # Apply all replacements
    for old, new in replacements:
        content = re.sub(old, new, content)

    # Also need to update QualityLevel imports - the restored API uses string severity not enum
    # First check if QualityLevel is being used as an enum
    if "QualityLevel." in content:
        # Replace enum access with strings
        content = content.replace("QualityLevel.CRITICAL", '"critical"')
        content = content.replace("QualityLevel.WARNING", '"warning"')
        content = content.replace("QualityLevel.INFO", '"info"')
        content = content.replace("issue.level", "issue.severity")

    # Write back the content
    test_file.write_text(content)
    print(f"Fixed {test_file}")


if __name__ == "__main__":
    fix_plugin_quality_tests()
