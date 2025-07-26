#!/usr/bin/env python
"""Fix plugin quality test files to add context parameter."""

import re
from pathlib import Path


def fix_plugin_quality_context():
    """Fix plugin quality tests to add context parameter."""

    test_file = Path("tests/plugins/spreadsheet/test_quality.py")

    # Read the content
    content = test_file.read_text()

    # Replace inspector.inspect calls to add context parameter
    # Find all occurrences of metrics = self.inspector.inspect(notebook)
    content = re.sub(
        r"metrics = self\.inspector\.inspect\((notebook|empty_notebook|self\.builder|builder[0-9]?)\)",
        r"metrics = self.inspector.inspect(\1, {})",
        content,
    )

    # Write back the content
    test_file.write_text(content)
    print(f"Fixed {test_file}")


if __name__ == "__main__":
    fix_plugin_quality_context()
