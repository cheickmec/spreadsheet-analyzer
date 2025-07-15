"""
Stage 0: File Integrity Probe (Functional Programming).

This module implements pure functional validation of Excel files,
checking format, structure, and basic integrity without side effects.
"""

import hashlib
import zipfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ..types import Err, FileMetadata, IntegrityResult, Ok, ProcessingClass, Result

# ==================== Pure Helper Functions ====================


def calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of file - pure function.

    CLAUDE-KNOWLEDGE: We use SHA-256 for file deduplication and integrity checks.
    Reading in 64KB chunks balances memory usage and performance.
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in 64KB chunks for memory efficiency
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def detect_mime_type(file_path: Path) -> str:
    """
    Detect MIME type by examining file structure.

    CLAUDE-GOTCHA: python-magic is not always available, so we implement
    our own detection based on file signatures and structure.
    """
    # Read first few bytes for magic number detection
    with open(file_path, "rb") as f:
        header = f.read(8)

    # Check for ZIP signature (modern Excel)
    if header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06"):
        # Could be XLSX, need to check structure
        if is_valid_ooxml_structure(file_path):
            return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        return "application/zip"

    # Check for OLE2 signature (legacy Excel)
    if header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return "application/vnd.ms-excel"

    # Check for Excel 5.0/95 signature
    if header.startswith(b"\x09\x08\x10\x00\x00\x06\x05\x00"):
        return "application/vnd.ms-excel"

    return "application/octet-stream"


def is_valid_ooxml_structure(file_path: Path) -> bool:
    """
    Validate OOXML structure by checking required files.

    CLAUDE-KNOWLEDGE: Valid XLSX files must contain specific XML files
    within the ZIP structure.
    CLAUDE-SECURITY: Structure validation prevents processing of malformed
    or potentially malicious files disguised as Excel documents.
    """
    try:
        if not zipfile.is_zipfile(file_path):
            return False

        with zipfile.ZipFile(file_path, "r") as zip_file:
            namelist = set(zip_file.namelist())

            # CLAUDE-KNOWLEDGE: These are the minimum required files for valid XLSX
            # Missing any of these indicates corruption or non-Excel format
            required_files = {"[Content_Types].xml", "xl/workbook.xml", "xl/_rels/workbook.xml.rels"}

            # Check if all required files exist
            return required_files.issubset(namelist)

    except (OSError, zipfile.BadZipFile):
        return False


def validate_excel_format(mime_type: str) -> bool:
    """Validate if MIME type indicates Excel format."""
    valid_types = {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
        "application/vnd.ms-excel",  # XLS
        "application/vnd.ms-excel.sheet.binary.macroEnabled.12",  # XLSB
        "application/vnd.ms-excel.sheet.macroEnabled.12",  # XLSM
    }
    return mime_type in valid_types


def determine_processing_class(metadata: FileMetadata, is_excel: bool, is_ooxml: bool) -> ProcessingClass:
    """
    Classify file for processing based on size and format.

    CLAUDE-PERFORMANCE: We use different thresholds for different formats
    based on their typical memory requirements during parsing.
    """
    # Block non-Excel files
    if not is_excel:
        return "BLOCKED"

    # Block suspiciously small files
    if metadata.size_bytes < 1024:  # Less than 1KB
        return "BLOCKED"

    # Size thresholds by format
    if is_ooxml:
        # Modern format - more efficient
        if metadata.size_bytes > 100 * 1024 * 1024:  # 100MB
            return "HEAVY"
    # Legacy format - less efficient
    elif metadata.size_bytes > 30 * 1024 * 1024:  # 30MB
        return "HEAVY"

    return "STANDARD"


def assess_trust_level(metadata: FileMetadata, is_excel: bool, is_ooxml: bool) -> int:
    """
    Assess trust level (1-5) based on various heuristics.

    CLAUDE-SECURITY: Trust scoring helps prioritize security scanning
    for potentially risky files.
    """
    score = 3  # Start neutral

    # Format trust
    if not is_excel:
        return 1  # Minimum trust for non-Excel

    if is_ooxml:
        score += 1  # Modern format is more trustworthy

    # Size heuristics
    size_mb = metadata.size_mb
    if 0.01 <= size_mb <= 50:
        score += 1  # Normal size range
    elif size_mb > 100:
        score -= 1  # Very large files are suspicious
    elif size_mb < 0.001:
        score -= 2  # Tiny files are very suspicious

    # Filename heuristics
    filename = metadata.path.name.lower()

    # Suspicious patterns
    suspicious_patterns = ["temp", "tmp", "test", "copy", "untitled", "~$", "__macosx", ".ds_store"]
    if any(pattern in filename for pattern in suspicious_patterns):
        score -= 1

    # Known good patterns
    good_patterns = ["report", "data", "analysis", "financial"]
    if any(pattern in filename for pattern in good_patterns):
        score += 1

    return max(1, min(5, score))


def check_file_duplication(file_hash: str, known_hashes: set[str]) -> bool:
    """
    Check if file hash exists in known hashes.

    Note: In production, this would query a database or cache.
    """
    return file_hash in known_hashes


# ==================== Main Stage Function ====================


def stage_0_integrity_probe(file_path: Path, known_hashes: set[str] | None = None) -> Result:
    """
    Perform complete integrity analysis using pure functions.

    This is the main entry point that orchestrates all validation checks
    and returns an immutable result.

    Args:
        file_path: Path to Excel file to analyze
        known_hashes: Optional set of known file hashes for deduplication

    Returns:
        Ok(IntegrityResult) if analysis succeeds
        Err(error_message) if analysis fails
    """
    # Validate file exists
    if not file_path.exists():
        return Err(f"File not found: {file_path}")

    if not file_path.is_file():
        return Err(f"Path is not a file: {file_path}")

    try:
        # Get file stats
        stat = file_path.stat()

        # Create immutable metadata
        metadata = FileMetadata(
            path=file_path,
            size_bytes=stat.st_size,
            mime_type=detect_mime_type(file_path),
            created_time=datetime.fromtimestamp(stat.st_ctime),
            modified_time=datetime.fromtimestamp(stat.st_mtime),
        )

        # Perform all checks using pure functions
        file_hash = calculate_file_hash(file_path)
        is_excel = validate_excel_format(metadata.mime_type)
        is_ooxml = is_valid_ooxml_structure(file_path)

        # Check duplication if known hashes provided
        is_duplicate = check_file_duplication(file_hash, known_hashes) if known_hashes else False

        # Determine processing classification
        processing_class = determine_processing_class(metadata, is_excel, is_ooxml)

        # Assess trust level
        trust_tier = assess_trust_level(metadata, is_excel, is_ooxml)

        # Create immutable result
        result = IntegrityResult(
            file_hash=file_hash,
            metadata=metadata,
            is_excel=is_excel,
            is_ooxml=is_ooxml,
            is_duplicate=is_duplicate,
            trust_tier=trust_tier,
            processing_class=processing_class,
            validation_passed=is_excel and processing_class != "BLOCKED",
        )

        return Ok(result)

    except Exception as e:
        return Err(f"Integrity probe failed: {e!s}", {"exception": str(e)})


# ==================== Validator Functions ====================


def create_integrity_validator(size_limit_mb: float = 100.0, required_trust: int = 2) -> Callable[[Path], list[str]]:
    """
    Create a customized validator function with specific requirements.

    This demonstrates function composition and partial application.
    """

    def validator(file_path: Path) -> list[str]:
        """Validate file and return list of issues."""
        issues = []

        # Run integrity probe
        result = stage_0_integrity_probe(file_path)

        if isinstance(result, Err):
            issues.append(f"Probe failed: {result.error}")
            return issues

        integrity = result.value

        # Check requirements
        if not integrity.is_excel:
            issues.append(f"Not a valid Excel file: {integrity.metadata.mime_type}")

        if integrity.metadata.size_mb > size_limit_mb:
            issues.append(f"File too large: {integrity.metadata.size_mb}MB (limit: {size_limit_mb}MB)")

        if integrity.trust_tier < required_trust:
            issues.append(f"Trust level too low: {integrity.trust_tier} (required: {required_trust})")

        if integrity.processing_class == "BLOCKED":
            issues.append("File blocked due to format or size issues")

        if integrity.is_duplicate:
            issues.append("Duplicate file detected")

        return issues

    return validator


# ==================== Utility Functions ====================


def create_batch_validator(validator: Callable[[Path], list[str]]) -> Callable[[list[Path]], dict]:
    """
    Create a batch validator from a single-file validator.

    Demonstrates higher-order function composition.
    """

    def batch_validator(file_paths: list[Path]) -> dict:
        """Validate multiple files and return results."""
        results = {}

        for file_path in file_paths:
            issues = validator(file_path)
            results[str(file_path)] = {"valid": len(issues) == 0, "issues": issues}

        return results

    return batch_validator
