#!/usr/bin/env python3
"""
Migration script to reorganize existing experiment outputs to hierarchical structure.

This script will:
1. Scan the outputs directory for existing flat files
2. Parse metadata from filenames and result files
3. Reorganize into the new hierarchical structure:
   outputs/module_name/full-model-name/input-file_sheetindex/timestamp_hash/

Usage:
    uv run experiments/migrate_outputs.py [--dry-run]
"""

import json
import re
import shutil
from pathlib import Path


def parse_filename_components(filename: str) -> dict[str, str]:
    """Extract module, timestamp, and hash from legacy filename pattern."""
    # Pattern: module_timestamp_hash_type.ext
    # Example: sheet_cartographer_openai_20250808_090031_947c0a6e_results.json
    pattern = r"^([^_]+(?:_[^_]+)*)_(\d{8}_\d{6})_([a-f0-9]{8})_(.+)\.(.+)$"
    match = re.match(pattern, filename)

    if match:
        return {
            "module": match.group(1),
            "timestamp": match.group(2),
            "hash": match.group(3),
            "type": match.group(4),
            "extension": match.group(5),
        }
    return {}


def read_results_metadata(results_file: Path) -> tuple[str | None, str | None, int | None]:
    """Extract model, sheet, and Excel file info from results.json."""
    try:
        with open(results_file) as f:
            data = json.load(f)

        # Try to get results data
        results = data.get("results", {})

        # Sheet info from results
        sheet_name = results.get("sheet", "unknown")

        # Try to infer model from any available metadata
        # This is challenging without the actual model in old files
        # We'll need to make educated guesses
        model = "gpt-4o"  # Default assumption for recent runs

        # Check if there's model info in the data
        if "model" in data:
            model = data["model"]
        elif "model_requested" in data:
            model = data["model_requested"]
        elif "model_actual" in data:
            model = data["model_actual"]

        # For sheet index, we'll default to 0 since it wasn't tracked
        sheet_index = 0

        return model, sheet_name, sheet_index

    except Exception as e:
        print(f"  ⚠️  Could not read metadata from {results_file}: {e}")
        return None, None, None


def sanitize_name(name: str) -> str:
    """Sanitize file/directory names."""
    if not name:
        return "unknown"
    # Remove spaces, special chars, lowercase
    sanitized = name.lower().replace(" ", "-").replace(".", "")
    # Keep only alphanumeric and hyphens
    sanitized = "".join(c if c.isalnum() or c == "-" else "" for c in sanitized)
    return sanitized or "unknown"


def infer_excel_filename(sheet_name: str) -> str:
    """Try to infer Excel filename from sheet name."""
    # Map known sheet names to their Excel files
    known_mappings = {
        "yiriden transactions 2025": "business-accounting",
        "yiriden-transactions-2025": "business-accounting",
        # Add more mappings as we discover them
    }

    sheet_lower = sheet_name.lower()
    for pattern, excel_name in known_mappings.items():
        if pattern in sheet_lower:
            return excel_name

    # If no match, use sanitized sheet name
    return sanitize_name(sheet_name)


def migrate_file_group(base_dir: Path, file_group: list[Path], dry_run: bool = False):
    """Migrate a group of related files to the new structure."""
    # Find the results file to extract metadata
    results_file = None
    for file in file_group:
        if file.name.endswith("_results.json"):
            results_file = file
            break

    if not results_file:
        print("  ⚠️  No results file found in group, using defaults")
        model = "gpt-4o"
        excel_name = "unknown"
        sheet_index = 0
    else:
        model, sheet_name, sheet_index = read_results_metadata(results_file)
        model = model or "gpt-4o"
        excel_name = infer_excel_filename(sheet_name) if sheet_name else "unknown"

    # Parse filename for module and timestamp info
    components = parse_filename_components(file_group[0].name)
    if not components:
        print(f"  ⚠️  Could not parse filename: {file_group[0].name}")
        return

    module = components["module"]
    timestamp = components["timestamp"]
    hash_val = components["hash"]

    # Create new directory structure
    new_dir = base_dir / module / model / f"{excel_name}_sh{sheet_index}" / f"{timestamp}_{hash_val}"

    print(f"  📁 {module}/{model}/{excel_name}_sh{sheet_index}/{timestamp}_{hash_val}/")

    if not dry_run:
        new_dir.mkdir(parents=True, exist_ok=True)

    # Move/copy files with new names
    for file in file_group:
        components = parse_filename_components(file.name)
        if components:
            # New filename is just type.extension
            new_name = f"{components['type']}.{components['extension']}"
            new_path = new_dir / new_name

            print(f"    {file.name} → {new_name}")

            if not dry_run:
                shutil.copy2(file, new_path)


def main(dry_run: bool = False):
    """Main migration function."""
    outputs_dir = Path(__file__).parent / "outputs"

    if not outputs_dir.exists():
        print("❌ No outputs directory found")
        return

    print(f"{'🔍 DRY RUN MODE' if dry_run else '🚀 MIGRATION MODE'}")
    print(f"📂 Scanning {outputs_dir}")
    print()

    # Find all files in flat structure
    all_files = [f for f in outputs_dir.iterdir() if f.is_file()]

    # Group files by module_timestamp_hash
    file_groups: dict[str, list[Path]] = {}

    for file in all_files:
        components = parse_filename_components(file.name)
        if components:
            key = f"{components['module']}_{components['timestamp']}_{components['hash']}"
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(file)

    print(f"Found {len(file_groups)} experiment runs with {len(all_files)} total files")
    print()

    # Migrate each group
    for i, (key, files) in enumerate(file_groups.items(), 1):
        print(f"[{i}/{len(file_groups)}] Migrating {key}:")
        migrate_file_group(outputs_dir, files, dry_run)
        print()

    if dry_run:
        print("✅ Dry run complete. Run without --dry-run to perform actual migration.")
    else:
        print("✅ Migration complete!")
        print()
        print("🧹 Old files have been copied to new structure.")
        print("   You can manually delete the old flat files after verifying the migration.")


if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv
    main(dry_run)
