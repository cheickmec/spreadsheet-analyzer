#!/usr/bin/env python3
"""Tool to update prompt hashes when intentionally modified.

Usage:
    uv run tools/update_prompt_hash.py <prompt_name>
    uv run tools/update_prompt_hash.py --all

Examples:
    # Update hash for a specific prompt
    uv run tools/update_prompt_hash.py data_analyst_system

    # Check all prompts and show which need updates
    uv run tools/update_prompt_hash.py --all
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from spreadsheet_analyzer.prompts import (
    PROMPT_REGISTRY,
    compute_file_hash,
    load_prompt,
)


def check_prompt(prompt_name: str, fix: bool = False) -> bool:
    """Check if a prompt's hash matches and optionally show fix.

    Args:
        prompt_name: Name of prompt to check
        fix: If True, show the fix command

    Returns:
        True if hash matches, False if update needed
    """
    if prompt_name not in PROMPT_REGISTRY:
        print(f"❌ Unknown prompt: '{prompt_name}'")
        print(f"   Available: {', '.join(sorted(PROMPT_REGISTRY.keys()))}")
        return False

    definition = PROMPT_REGISTRY[prompt_name]
    prompt_path = Path(__file__).parent.parent / "src" / "spreadsheet_analyzer" / "prompts" / definition.file_name

    if not prompt_path.exists():
        print(f"❌ File not found: {definition.file_name}")
        return False

    current_hash = compute_file_hash(prompt_path)
    expected_hash = definition.content_hash

    if current_hash == expected_hash:
        print(f"✅ {prompt_name}: Hash matches")
        return True
    else:
        print(f"❌ {prompt_name}: Hash mismatch!")
        print(f"   Expected: {expected_hash}")
        print(f"   Current:  {current_hash}")

        if fix:
            print(f"\n   To fix, update in src/spreadsheet_analyzer/prompts/registry.py:")
            print(f'   content_hash="{current_hash}",')

        return False


def main():
    """Main entry point for the hash update tool."""
    parser = argparse.ArgumentParser(
        description="Update prompt hashes when intentionally modified",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check a specific prompt
  %(prog)s data_analyst_system

  # Check all prompts
  %(prog)s --all

  # Show fix commands
  %(prog)s --all --fix
""",
    )

    parser.add_argument("prompt_name", nargs="?", help="Name of prompt to check/update")
    parser.add_argument("--all", action="store_true", help="Check all prompts")
    parser.add_argument("--fix", action="store_true", help="Show fix commands for mismatched hashes")

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.prompt_name:
        parser.error("Either provide a prompt name or use --all")

    if args.all and args.prompt_name:
        parser.error("Cannot specify both prompt name and --all")

    # Process prompts
    if args.all:
        print("Checking all prompts...\n")
        all_valid = True
        for prompt_name in sorted(PROMPT_REGISTRY.keys()):
            valid = check_prompt(prompt_name, fix=args.fix)
            if not valid:
                all_valid = False
                print()  # Add spacing between errors

        if all_valid:
            print("\n✅ All prompts have valid hashes!")
            sys.exit(0)
        else:
            print("\n❌ Some prompts need hash updates")
            if not args.fix:
                print("   Run with --fix to see update commands")
            sys.exit(1)
    else:
        valid = check_prompt(args.prompt_name, fix=True)
        sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
