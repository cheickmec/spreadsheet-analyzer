#!/usr/bin/env python3
"""
Fix corrupted characters in markdown files that resulted from encoding conversion.
This script fixes specific character corruptions that occurred during our encoding process.
"""

import re
import sys
from pathlib import Path


def get_character_mappings() -> dict[str, str]:
    """Get mappings from corrupted characters to correct ones."""
    return {
        # Common corrupted characters from encoding conversion
        "‚úÖ": "✅",
        "‚ùå": "❌",
        "‚Ä¶": "• ",
        "‚ÄÇ": "- ",
        "‚ÄØ": " ",
        "‚Äë": "-",
        "‚Äî": "—",
        "‚Äì": '"',
        "‚Äù": '"',
        "‚Äú": '"',
        "‚É£": "≤",
        "‚Üê": "→",
        "‚Üì": "←",
        "‚Üí": "↑",
        "‚â§": "§",
        "‚âà": "∞",
        "‚ö†Ô∏è": "…",
        # Tree structure characters
        "‚îú‚îÄ‚îÄ": "├── ",
        "‚îÇ   ": "│   ",
        "‚îî‚îÄ‚îÄ": "└── ",
        "‚îÇ": "│",
        "‚îú": "├",
        "‚îî": "└",
        "‚îÄ": "─",
        "‚îê": "┐",
        "‚îò": "┘",
        "‚î¨": "┬",
        "‚îå": "┌",
        "‚î¥": "┤",
        "‚ñº": "┴",
        "‚ñ∂": "┼",
        "‚ñ∫": "┻",
        "‚îÇ<": "│<",
        "‚óÄ": "┬",
        "‚î§": "┐",
        # Common text corruptions
        '‚â"€â"€â"€â"€â–¶â"‚': "────▶─│",
        "‚Üí‚ÄØ": "↑ ",
        "‚â§‚ÄØ": "§ ",
        "‚âà‚ÄØ": "∞ ",
    }


def fix_corrupted_characters(text: str) -> str:
    """Fix corrupted characters in text using character mappings."""
    mappings = get_character_mappings()

    # Apply character mappings
    for corrupted, correct in mappings.items():
        text = text.replace(corrupted, correct)

    # Handle remaining complex patterns
    # Remove standalone corrupted characters that don't have specific mappings
    text = re.sub(r"‚[A-Za-z0-9†Ô∏è§àüêìí]*", "", text)

    return text


def fix_file(file_path: Path) -> bool:
    """Fix corrupted characters in a single file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        original_content = content
        fixed_content = fix_corrupted_characters(content)

        if fixed_content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)
            print(f"Fixed: {file_path}")
            return True
        print(f"No changes needed: {file_path}")
        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix corrupted characters in files."""
    if len(sys.argv) < 2:
        print("Usage: python fix_corrupted_chars.py <file_or_directory>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.rglob("*.md"))
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

    fixed_count = 0
    total_count = len(files)

    for file_path in files:
        if fix_file(file_path):
            fixed_count += 1

    print("\nSummary:")
    print(f"  Total files processed: {total_count}")
    print(f"  Files fixed: {fixed_count}")
    print(f"  Files unchanged: {total_count - fixed_count}")


if __name__ == "__main__":
    main()
