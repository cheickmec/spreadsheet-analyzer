#!/usr/bin/env python3
"""
Fix encoding issues in markdown files.

This script detects the actual encoding of markdown files and converts them to UTF-8
to ensure compatibility with mdformat and other tools.
"""

import argparse
import sys
from pathlib import Path

import chardet


def detect_encoding(file_path: Path) -> str | None:
    """Detect the encoding of a file."""
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result.get("encoding")
            confidence = result.get("confidence", 0.0)

            print(f"File: {file_path}")
            print(f"  Detected encoding: {encoding} (confidence: {confidence:.2f})")

            return str(encoding) if encoding else None
    except Exception as e:
        print(f"Error detecting encoding for {file_path}: {e}")
        return None


def convert_to_utf8(file_path: Path, source_encoding: str) -> bool:
    """Convert file from source encoding to UTF-8."""
    try:
        # Read with detected encoding
        with open(file_path, encoding=source_encoding, errors="replace") as f:
            content = f.read()

        # Write back as UTF-8
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"  ✓ Converted {file_path} from {source_encoding} to UTF-8")
        return True

    except Exception as e:
        print(f"  ✗ Error converting {file_path}: {e}")
        return False


def fix_markdown_files(files: list[Path], dry_run: bool = False) -> None:
    """Fix encoding issues in markdown files."""
    total_files = len(files)
    converted_files = 0
    already_utf8 = 0
    errors = 0

    for file_path in files:
        if not file_path.exists():
            print(f"File not found: {file_path}")
            errors += 1
            continue

        encoding = detect_encoding(file_path)
        if not encoding:
            errors += 1
            continue

        # Check if already UTF-8
        if encoding.lower() in ["utf-8", "ascii"]:
            print("  ✓ Already UTF-8/ASCII")
            already_utf8 += 1
            continue

        # Convert to UTF-8
        if not dry_run:
            if convert_to_utf8(file_path, encoding):
                converted_files += 1
            else:
                errors += 1
        else:
            print(f"  → Would convert from {encoding} to UTF-8")
            converted_files += 1

    print("\nSummary:")
    print(f"  Total files: {total_files}")
    print(f"  Already UTF-8/ASCII: {already_utf8}")
    print(f"  Converted: {converted_files}")
    print(f"  Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description="Fix encoding issues in markdown files")
    parser.add_argument("files", nargs="+", help="Markdown files to fix")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")

    args = parser.parse_args()

    # Convert string paths to Path objects
    files = [Path(f) for f in args.files]

    # Filter for markdown files
    markdown_files = [f for f in files if f.suffix.lower() == ".md"]

    if not markdown_files:
        print("No markdown files found.")
        sys.exit(1)

    fix_markdown_files(markdown_files, args.dry_run)


if __name__ == "__main__":
    main()
