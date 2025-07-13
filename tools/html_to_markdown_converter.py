#!/usr/bin/env python3
"""
HTML to Markdown Converter
Extracts content from HTML and converts to well-formatted markdown.
"""

import re
import sys
from pathlib import Path

import html2text
from bs4 import BeautifulSoup

MIN_ARGS = 2
MAX_ARGS = 3


def convert_html_to_markdown(html_file_path, output_file_path=None):
    """
    Convert HTML file to markdown format while preserving all content.

    Args:
        html_file_path (str): Path to the HTML file
        output_file_path (str): Path for the output markdown file
    """

    # Read the HTML file
    try:
        with Path(html_file_path).open(encoding="utf-8") as file:
            html_content = file.read()
    except OSError as e:
        print(f"Error reading HTML file: {e}")
        return False

    # Parse HTML with BeautifulSoup for better structure
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Configure html2text converter
    h = html2text.HTML2Text()
    h.ignore_links = False  # Preserve links
    h.ignore_images = False  # Preserve images
    h.ignore_tables = False  # Preserve tables
    h.body_width = 0  # No line wrapping
    h.unicode_snob = True  # Use unicode characters
    h.mark_code = True  # Mark code blocks
    h.wrap_links = False  # Don't wrap links
    h.inline_links = True  # Use inline links format
    h.use_automatic_links = True  # Convert URLs to links
    h.protect_links = True  # Protect existing links
    h.single_line_break = False  # Use double line breaks for paragraphs

    # Convert HTML to markdown
    try:
        markdown_content = h.handle(str(soup))
    except html2text.Html2TextException as e:
        print(f"Error converting HTML to markdown: {e}")
        return False

    # Clean up the markdown
    markdown_content = clean_markdown(markdown_content)

    # Determine output file path
    if output_file_path is None:
        html_path = Path(html_file_path)
        output_file_path = html_path.parent / f"{html_path.stem}.md"

    # Write the markdown file
    try:
        with Path(output_file_path).open("w", encoding="utf-8") as file:
            file.write(markdown_content)
        print(f"Successfully converted HTML to markdown: {output_file_path}")
    except OSError as e:
        print(f"Error writing markdown file: {e}")
        return False
    else:
        return True


def clean_markdown(content):
    """
    Clean up the markdown content to improve formatting.

    Args:
        content (str): Raw markdown content

    Returns:
        str: Cleaned markdown content
    """

    # Remove excessive blank lines (more than 2 consecutive)
    content = re.sub(r"\n{3,}", "\n\n", content)

    # Fix spacing around headers
    content = re.sub(r"\n(#{1,6}\s+.*?)\n", r"\n\n\1\n\n", content)

    # Fix spacing around list items
    content = re.sub(r"\n(\s*[-*+]\s+.*?)\n", r"\n\1\n", content)
    content = re.sub(r"\n(\s*\d+\.\s+.*?)\n", r"\n\1\n", content)

    # Fix spacing around code blocks
    content = re.sub(r"\n(```.*?```)\n", r"\n\n\1\n\n", content, flags=re.DOTALL)

    # Fix spacing around blockquotes
    content = re.sub(r"\n(>\s+.*?)\n", r"\n\n\1\n\n", content)

    # Clean up whitespace at the beginning and end
    content = content.strip()

    # Ensure document ends with single newline
    return content.rstrip() + "\n"


def main():
    if len(sys.argv) < MIN_ARGS:
        print("Usage: python html_to_markdown_converter.py <html_file> [output_file]")
        sys.exit(1)

    html_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > MIN_ARGS else None

    if not Path(html_file).exists():
        print(f"Error: HTML file '{html_file}' not found.")
        sys.exit(1)

    success = convert_html_to_markdown(html_file, output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
