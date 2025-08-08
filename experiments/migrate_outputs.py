#!/usr/bin/env python3
"""
Migrate existing experiment outputs to hierarchical structure.

This script reorganizes the flat output structure to the new hierarchical format:
Old: outputs/sheet_cartographer_openai/{timestamp}_{hash}/
New: outputs/sheet_cartographer_openai/{full_model}/{input_file}_sh{index}/{timestamp}_{hash}/
"""

import re
import shutil
from pathlib import Path


def extract_model_from_logs(log_dir: Path) -> tuple[str, str, int]:
    """Extract model name and input file from log files."""
    main_log = log_dir / "main.log"
    llm_trace = log_dir / "llm_trace.log"

    model_actual = None
    excel_file = None
    sheet_index = 0

    # Try to get actual model from LLM trace
    if llm_trace.exists():
        content = llm_trace.read_text(encoding="utf-8", errors="ignore")
        # Look for ACTUAL_MODEL in logs
        actual_match = re.search(r"ACTUAL_MODEL:\s*(.+)", content)
        if actual_match:
            model_actual = actual_match.group(1).strip()
        else:
            # Look for MODEL: field
            model_match = re.search(r"^MODEL:\s*(.+)$", content, re.MULTILINE)
            if model_match:
                model_actual = model_match.group(1).strip()

    # Try to get configuration from main log
    if main_log.exists():
        content = main_log.read_text(encoding="utf-8", errors="ignore")

        # Extract Excel file
        excel_match = re.search(r"Excel file:\s*(.+\.xlsx)", content)
        if excel_match:
            excel_path = Path(excel_match.group(1).strip())
            excel_file = excel_path.stem

        # Extract sheet index
        sheet_match = re.search(r"Sheet index:\s*(\d+)", content)
        if sheet_match:
            sheet_index = int(sheet_match.group(1))

        # If no model from LLM trace, try main log
        if not model_actual:
            model_match = re.search(r"Model:\s*(.+)", content)
            if model_match:
                model_actual = model_match.group(1).strip()

    # Map short names to full names if needed
    model_map = {
        "gpt-4o": "gpt-4o-2024-08-06",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
        "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    }

    if model_actual in model_map:
        model_actual = model_map[model_actual]

    return model_actual or "unknown_model", excel_file or "unknown_input", sheet_index


def sanitize_filename(name: str) -> str:
    """Sanitize filename for directory name."""
    # Remove extension if present
    if name.endswith(".xlsx"):
        name = name[:-5]

    # Lowercase and replace spaces/dots
    sanitized = name.lower().replace(" ", "-").replace(".", "")

    # Keep only alphanumeric and hyphens
    sanitized = "".join(c if c.isalnum() or c == "-" else "" for c in sanitized)

    return sanitized if sanitized else "unknown"


def migrate_outputs(base_dir: Path, dry_run: bool = True):
    """Migrate outputs to hierarchical structure."""
    modules_to_migrate = ["sheet_cartographer_openai"]

    for module in modules_to_migrate:
        module_dir = base_dir / module
        if not module_dir.exists():
            continue

        print(f"\n📦 Processing module: {module}")

        # Find all timestamp directories (old format)
        timestamp_dirs = []
        for item in module_dir.iterdir():
            if item.is_dir() and re.match(r"^\d{8}_\d{6}_[a-f0-9]{8}$", item.name):
                timestamp_dirs.append(item)

        if not timestamp_dirs:
            print("  No old-format directories found")
            continue

        print(f"  Found {len(timestamp_dirs)} directories to migrate")

        for old_dir in sorted(timestamp_dirs):
            print(f"\n  📁 Processing: {old_dir.name}")

            # Extract metadata from logs
            model, excel_file, sheet_index = extract_model_from_logs(old_dir)

            # Sanitize excel filename
            if excel_file != "unknown_input":
                excel_sanitized = sanitize_filename(excel_file)
                input_dir = f"{excel_sanitized}_sh{sheet_index}"
            else:
                input_dir = f"unknown_sh{sheet_index}"

            # Create new path
            new_parent = module_dir / model / input_dir
            new_dir = new_parent / old_dir.name

            print(f"    Model: {model}")
            print(f"    Input: {input_dir}")
            print(f"    Old path: {old_dir.relative_to(base_dir)}")
            print(f"    New path: {new_dir.relative_to(base_dir)}")

            if not dry_run:
                # Create parent directories
                new_parent.mkdir(parents=True, exist_ok=True)

                # Move directory
                if new_dir.exists():
                    print("    ⚠️  Destination already exists, skipping")
                else:
                    shutil.move(str(old_dir), str(new_dir))
                    print("    ✅ Migrated successfully")
            else:
                print(f"    🔍 [DRY RUN] Would move to: {new_dir.relative_to(base_dir)}")

    # Clean up unknown_sh0 directories from failed runs
    if not dry_run:
        print("\n🧹 Cleaning up unknown_sh0 directories...")
        for module in modules_to_migrate:
            module_dir = base_dir / module
            if not module_dir.exists():
                continue

            # Find all unknown_sh0 directories
            for model_dir in module_dir.iterdir():
                if model_dir.is_dir():
                    unknown_dir = model_dir / "unknown_sh0"
                    if unknown_dir.exists():
                        print(f"  Removing: {unknown_dir.relative_to(base_dir)}")
                        shutil.rmtree(unknown_dir)


def main():
    """Main migration function."""
    outputs_dir = Path(__file__).parent / "outputs"

    print("=" * 80)
    print("📊 EXPERIMENT OUTPUT MIGRATION TOOL")
    print("=" * 80)
    print(f"Base directory: {outputs_dir}")

    # First do a dry run
    print("\n🔍 DRY RUN - Analyzing migration plan...")
    print("-" * 80)
    migrate_outputs(outputs_dir, dry_run=True)

    # Ask for confirmation
    print("\n" + "=" * 80)
    response = input("\n❓ Proceed with migration? (yes/no): ").strip().lower()

    if response == "yes":
        print("\n🚀 EXECUTING MIGRATION...")
        print("-" * 80)
        migrate_outputs(outputs_dir, dry_run=False)
        print("\n✅ Migration complete!")
    else:
        print("\n❌ Migration cancelled")


if __name__ == "__main__":
    main()
