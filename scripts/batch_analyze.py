#!/usr/bin/env python3
"""
Batch analyze Excel files in a directory.

Usage:
    python batch_analyze.py [directory] [options]

Examples:
    python batch_analyze.py                    # Analyze test-files/
    python batch_analyze.py /path/to/files     # Analyze specific directory
    python batch_analyze.py --recursive        # Include subdirectories
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spreadsheet_analyzer.pipeline import DeterministicPipeline, create_lenient_pipeline_options

# Constants
MAX_DISPLAY_FILES = 3


def find_excel_files(test_dir: Path, *, recursive: bool) -> list[Path]:
    """Find all Excel files in directory."""
    excel_extensions = {".xlsx", ".xls", ".xlsm", ".xlsb"}
    pattern = test_dir.rglob("*") if recursive else test_dir.glob("*")
    return [file_path for file_path in pattern if file_path.is_file() and file_path.suffix.lower() in excel_extensions]


def print_file_summary(excel_files: list[Path]) -> None:
    """Print summary of files by directory."""
    print("\nFiles by directory:")
    by_dir: dict[str, list[str]] = {}
    for file_path in excel_files:
        dir_name = file_path.parent.name
        by_dir.setdefault(dir_name, []).append(file_path.name)

    for dir_name, files in sorted(by_dir.items()):
        print(f"  {dir_name}: {len(files)} files")
        for file_name in sorted(files)[:MAX_DISPLAY_FILES]:
            print(f"    - {file_name}")
        if len(files) > MAX_DISPLAY_FILES:
            print(f"    ... and {len(files) - MAX_DISPLAY_FILES} more")


def process_files(
    excel_files: list[Path], pipeline: DeterministicPipeline, *, summary_only: bool
) -> dict[str, int | float]:
    """Process all Excel files and return results."""
    results = {"success": 0, "failed": 0, "total_time": 0.0}

    for i, file_path in enumerate(excel_files, 1):
        if not summary_only:
            try:
                display_path = file_path.relative_to(file_path.parent.parent)
            except ValueError:
                display_path = Path(file_path.name)
            print(f"\n[{i}/{len(excel_files)}] Testing: {display_path}")

        start = time.time()

        try:
            result = pipeline.run(file_path)
            elapsed = time.time() - start
            results["total_time"] += elapsed

            if result.success:
                results["success"] += 1
                if not summary_only:
                    print(f"✅ Success in {elapsed:.2f}s")
                    if result.structure:
                        cells = f"{result.structure.total_cells:,}"
                        print(f"   Sheets: {result.structure.sheet_count}, Cells: {cells}")
            else:
                results["failed"] += 1
                if not summary_only:
                    error = result.errors[0] if result.errors else "Unknown"
                    print(f"❌ Failed: {error}")
        except (OSError, ValueError) as e:
            results["failed"] += 1
            if not summary_only:
                print(f"❌ Exception: {e!s}")

        # Progress indicator for summary-only mode
        if summary_only and i % 10 == 0:
            print(".", end="", flush=True)

    if summary_only:
        print()  # New line after progress dots

    return results


def main():
    """Batch analyze Excel files."""
    parser = argparse.ArgumentParser(description="Batch analyze Excel files in a directory")
    parser.add_argument("directory", nargs="?", default="test-files", help="Directory to analyze (default: test-files)")
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories")
    parser.add_argument("--summary-only", action="store_true", help="Show only summary, not individual file results")

    args = parser.parse_args()
    test_dir = Path(args.directory)

    # Check directory exists
    if not test_dir.exists():
        print(f"Error: Directory not found: {test_dir}")
        return 1

    # Find all Excel files
    excel_files = find_excel_files(test_dir, recursive=args.recursive)

    print(f"Found {len(excel_files)} Excel files to test")
    print("=" * 80)

    # Test each file
    pipeline = DeterministicPipeline(create_lenient_pipeline_options())
    results = process_files(excel_files, pipeline, summary_only=args.summary_only)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(excel_files)}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")

    if excel_files:
        success_rate = results["success"] / len(excel_files) * 100
        avg_time = results["total_time"] / len(excel_files)
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Total time: {results['total_time']:.2f}s")
        print(f"Average time: {avg_time:.2f}s per file")

    # List files by directory
    print_file_summary(excel_files)

    return 0 if results["success"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
