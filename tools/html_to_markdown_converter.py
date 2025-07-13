#!/usr/bin/env python3
"""
HTML to Markdown Converter using pypandoc.
A robust converter that handles various HTML formats and edge cases.
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pypandoc


def setup_logging(*, verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def validate_pandoc_installation() -> bool:
    """Check if pandoc is installed and accessible."""
    logger = logging.getLogger(__name__)
    try:
        version = pypandoc.get_pandoc_version()
    except OSError:
        logger.error("Pandoc is not installed. Please install it first:")  # noqa: TRY400
        logger.error("  - macOS: brew install pandoc")  # noqa: TRY400
        logger.error("  - Ubuntu/Debian: sudo apt-get install pandoc")  # noqa: TRY400
        logger.error("  - Windows: Download from https://pandoc.org/installing.html")  # noqa: TRY400
        return False
    else:
        logger.info("Found pandoc version: %s", version)
        return True


def clean_html_content(html_content: str) -> str:
    """Pre-process HTML content to handle common issues."""
    # Remove zero-width spaces and other invisible characters
    html_content = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", html_content)

    # Fix common HTML entity issues
    html_content = html_content.replace("&nbsp;", " ")

    # Remove empty paragraphs and divs
    html_content = re.sub(r"<p>\s*</p>", "", html_content)
    return re.sub(r"<div>\s*</div>", "", html_content)


def post_process_markdown(content: str) -> str:
    """Clean up the markdown content after conversion."""
    # Remove excessive blank lines (more than 2 consecutive)
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Fix spacing around headers
    content = re.sub(r"(^|\n)(#{1,6}\s+.*?)(\n|$)", r"\1\n\2\n", content, flags=re.MULTILINE)

    # Clean up escaped characters that shouldn't be escaped
    content = re.sub(r"\\([#*_`'])", r"\1", content)

    # Fix overly escaped angle brackets and quotes in regular text
    content = re.sub(r"\\([<>\"'])", r"\1", content)

    # Fix code block formatting
    content = re.sub(r"``` (.*?)\n", r"```\1\n", content)

    # Ensure proper list formatting
    content = re.sub(r"^(\s*)([-*+])\s+", r"\1\2 ", content, flags=re.MULTILINE)
    content = re.sub(r"^(\s*)(\d+)\.\s+", r"\1\2. ", content, flags=re.MULTILINE)

    # Remove trailing whitespace
    content = re.sub(r" +$", "", content, flags=re.MULTILINE)

    # Remove YAML frontmatter if it only contains title
    # (since we usually want clean markdown without metadata)
    content = re.sub(r"^---\ntitle: .*\n---\n\n", "", content)

    # Ensure document ends with single newline
    return content.strip() + "\n"


def convert_html_to_markdown(
    html_file_path: Path,
    output_file_path: Path | None = None,
    format_options: list[str] | None = None,
) -> bool:
    """
    Convert HTML file to markdown format using pypandoc.

    Args:
        html_file_path: Path to the HTML file
        output_file_path: Path for the output markdown file (optional)
        format_options: Additional pandoc options

    Returns:
        bool: True if conversion successful, False otherwise
    """
    logger = logging.getLogger(__name__)

    # Validate input file
    if not html_file_path.exists():
        logger.error("HTML file not found: %s", html_file_path)
        return False

    if html_file_path.suffix.lower() not in [".html", ".htm"]:
        logger.warning("File may not be HTML: %s", html_file_path)

    # Read HTML content
    try:
        html_content = html_file_path.read_text(encoding="utf-8")
        logger.info("Read HTML file: %s (%d bytes)", html_file_path, len(html_content))
    except OSError:
        logger.exception("Error reading HTML file")
        return False

    # Pre-process HTML
    html_content = clean_html_content(html_content)

    # Determine output path
    if output_file_path is None:
        output_file_path = html_file_path.with_suffix(".md")

    # Set up pandoc options
    default_options = [
        "--wrap=none",  # No line wrapping
        "--strip-comments",  # Remove HTML comments
        "--markdown-headings=atx",  # Use ATX-style headers (## instead of underlines)
        "--standalone",  # Produce complete document
    ]

    extra_args = default_options.copy()
    if format_options:
        extra_args.extend(format_options)

    # Convert using pypandoc
    try:
        logger.info("Converting HTML to Markdown...")
        markdown_content = pypandoc.convert_text(
            html_content,
            "markdown",
            format="html",
            extra_args=extra_args,
        )

        # Post-process the markdown
        markdown_content = post_process_markdown(markdown_content)

        # Write output
        output_file_path.write_text(markdown_content, encoding="utf-8")
        logger.info("Successfully converted to: %s", output_file_path)
    except (RuntimeError, ValueError):
        logger.exception("Error during conversion")
        return False
    else:
        return True


def convert_batch(
    html_files: list[Path],
    output_dir: Path | None = None,
    format_options: list[str] | None = None,
) -> dict[Path, bool]:
    """
    Convert multiple HTML files to markdown.

    Args:
        html_files: List of HTML file paths
        output_dir: Directory for output files (optional)
        format_options: Additional pandoc options

    Returns:
        dict: Mapping of input files to conversion success status
    """
    results = {}

    for html_file in html_files:
        output_file = output_dir / f"{html_file.stem}.md" if output_dir else None

        success = convert_html_to_markdown(html_file, output_file, format_options)
        results[html_file] = success

    return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Convert HTML files to well-formatted Markdown using pypandoc",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.html                    # Convert to input.md
  %(prog)s input.html output.md          # Convert to specific output file
  %(prog)s *.html -o output_dir/         # Batch convert to directory
  %(prog)s input.html --reference-links  # Use reference-style links
  %(prog)s input.html -v                 # Verbose output
        """,
    )

    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="HTML file(s) to convert",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file or directory (for batch conversion)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--inline-links",
        action="store_true",
        help="Use inline links instead of reference links",
    )

    parser.add_argument(
        "--no-wrap",
        action="store_true",
        help="Disable text wrapping (default: enabled)",
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(verbose=args.verbose)

    # Check pandoc installation
    if not validate_pandoc_installation():
        sys.exit(1)

    # Prepare format options
    format_options = []
    # Only add reference-links if NOT using inline links
    # (by default pandoc uses inline links)
    if not args.inline_links:
        format_options.append("--reference-links")

    # Handle single file vs batch conversion
    input_files = [f for f in args.input_files if f.exists()]
    if not input_files:
        logger.error("No valid input files found")
        sys.exit(1)

    if len(input_files) == 1:
        # Single file conversion
        output_path = args.output if args.output and not args.output.is_dir() else None

        success = convert_html_to_markdown(
            input_files[0],
            output_path,
            format_options,
        )
        sys.exit(0 if success else 1)
    else:
        # Batch conversion
        output_dir = args.output if args.output and args.output.is_dir() else None
        results = convert_batch(input_files, output_dir, format_options)

        # Report results
        successful = sum(1 for success in results.values() if success)
        failed = len(results) - successful

        logger.info("Conversion complete: %d successful, %d failed", successful, failed)
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
