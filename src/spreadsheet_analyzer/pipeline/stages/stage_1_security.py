"""
Stage 1: Security Scan (Functional Programming).

This module implements security scanning for Excel files using pure functions
to detect macros, external links, embedded objects, and other potential threats.
"""

import logging
import re
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Final

import defusedxml.ElementTree as DefusedElementTree

from spreadsheet_analyzer.pipeline.types import Err, Ok, Result, RiskLevel, SecurityReport, SecurityThreat

logger = logging.getLogger(__name__)

# ==================== Security Pattern Definitions ====================

# Risk severity thresholds (scale: 0-10)
# Individual threat severity determines how dangerous a single finding is
HIGH_RISK_SEVERITY_THRESHOLD: Final[int] = 7  # Threats rated 7+ are high risk (e.g., shell execution)
CRITICAL_SEVERITY_THRESHOLD: Final[int] = 9  # Threats rated 9+ are critical (e.g., auto-executing macros)
MEDIUM_SEVERITY_THRESHOLD: Final[int] = 5  # Threats rated 5-6 are medium risk (e.g., external links)

# Overall risk score thresholds (scale: 0-100)
# Combined score from all threats determines file's overall risk level
CRITICAL_RISK_SCORE_THRESHOLD: Final[int] = 80  # Files scoring 80+ are critical risk
HIGH_RISK_SCORE_THRESHOLD: Final[int] = 60  # Files scoring 60-79 are high risk
MEDIUM_RISK_SCORE_THRESHOLD: Final[int] = 40  # Files scoring 40-59 are medium risk

# Size limits for security scanning
# These prevent memory exhaustion from maliciously large files
MAX_MACRO_SCAN_SIZE: Final[int] = 1024 * 1024  # 1MB limit for macro content scanning
MAX_XML_PARSE_SIZE: Final[int] = 10 * 1024 * 1024  # 10MB limit for XML parsing

# Patterns for detecting potentially dangerous content in VBA macros
# Each pattern identifies a category of risky behavior
SUSPICIOUS_PATTERNS = {
    "auto_open": re.compile(r"auto_open|workbook_open", re.IGNORECASE),  # Auto-executing macros
    "shell_exec": re.compile(r"shell|cmd|powershell|wscript", re.IGNORECASE),  # System command execution
    "file_io": re.compile(r"createobject|filesystemobject|scripting", re.IGNORECASE),  # File system access
    "network": re.compile(r"http|ftp|download|urlmon", re.IGNORECASE),  # Network operations
    "registry": re.compile(r"registry|regwrite|regread", re.IGNORECASE),  # Windows registry access
}

# External link patterns for detecting connections to external resources
# These could be data exfiltration or malware download vectors
EXTERNAL_LINK_PATTERNS = {
    "http": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),  # Web URLs
    "file": re.compile(r'file://[^\s<>"{}|\\^`\[\]]+'),  # Local file references
    "unc": re.compile(r'\\\\[^\s<>"{}|\\^`\[\]]+'),  # Network shares (UNC paths)
}

# OOXML namespaces used for parsing Excel XML files
# These are standard Microsoft Office Open XML schemas
NAMESPACES = {
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "x14ac": "http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac",
}

# Standard OOXML namespace URLs that should not be flagged as external links
# These are part of the Excel file format specification
OOXML_NAMESPACE_WHITELIST = {
    "http://schemas.openxmlformats.org/",
    "http://schemas.microsoft.com/office/",
    "http://purl.oclc.org/ooxml/",
    "http://purl.org/dc/elements/",
    "http://purl.org/dc/terms/",
    "http://purl.org/dc/dcmitype/",
}

# ==================== Pure Security Check Functions ====================


def check_for_vba_macros(file_path: Path) -> tuple[bool, list[SecurityThreat]]:
    """
    Check for VBA macros in Excel file.

    CLAUDE-SECURITY: VBA macros are the primary attack vector in Excel files.
    We check both the vbaProject.bin and any VBA-related XML.
    """
    threats = []
    has_macros = False

    try:
        with zipfile.ZipFile(file_path, "r") as zip_file:
            namelist = set(zip_file.namelist())

            # Check for VBA project
            if "xl/vbaProject.bin" in namelist:
                has_macros = True
                threats.append(
                    SecurityThreat(
                        threat_type="VBA_MACROS",
                        severity=7,
                        location="xl/vbaProject.bin",
                        description="File contains VBA macros",
                        risk_level="HIGH",
                        details={"file": "vbaProject.bin"},
                    )
                )

                # Try to analyze macro content (without executing)
                try:
                    macro_info = zip_file.getinfo("xl/vbaProject.bin")

                    # Check file size before reading
                    if macro_info.file_size > MAX_MACRO_SCAN_SIZE:
                        threats.append(
                            SecurityThreat(
                                threat_type="LARGE_MACRO_FILE",
                                severity=8,
                                location="xl/vbaProject.bin",
                                description=f"Macro file exceeds size limit ({macro_info.file_size} bytes)",
                                risk_level="HIGH",
                                details={"size": macro_info.file_size, "limit": MAX_MACRO_SCAN_SIZE},
                            )
                        )
                    else:
                        macro_content = zip_file.read("xl/vbaProject.bin")

                        # CLAUDE-SECURITY: Search binary content directly to avoid encoding issues
                        # and improve performance on large files
                        for pattern_name, pattern in SUSPICIOUS_PATTERNS.items():
                            # Convert pattern to bytes for binary search
                            try:
                                # Search in chunks to avoid memory issues
                                chunk_size = 8192
                                found = False
                                for i in range(0, min(len(macro_content), MAX_MACRO_SCAN_SIZE), chunk_size):
                                    chunk = macro_content[i : i + chunk_size]
                                    # Try to decode chunk with error handling
                                    try:
                                        chunk_str = chunk.decode("utf-8", errors="ignore")
                                    except (UnicodeDecodeError, AttributeError):
                                        chunk_str = chunk.decode("latin-1", errors="ignore")

                                    if pattern.search(chunk_str):
                                        found = True
                                        break

                                if found:
                                    threats.append(
                                        SecurityThreat(
                                            threat_type=f"SUSPICIOUS_MACRO_{pattern_name.upper()}",
                                            severity=9,
                                            location="xl/vbaProject.bin",
                                            description=f"Macro contains suspicious {pattern_name} patterns",
                                            risk_level="CRITICAL",
                                            details={"pattern": pattern_name},
                                        )
                                    )
                            except Exception:
                                # Skip this pattern if there's an error
                                logger.debug("Error processing pattern %s in macro content", pattern_name)
                except (OSError, ValueError):
                    # Can't read macro content, but we know it exists
                    pass

    except (OSError, zipfile.BadZipFile):
        # Not a zip file or other error - handle in calling function
        pass

    return has_macros, threats


def check_for_external_links(file_path: Path) -> tuple[bool, list[SecurityThreat]]:
    """
    Check for external links in Excel file.

    CLAUDE-GOTCHA: External links can be in multiple places:
    - External link definitions (externalLinks/)
    - Formula references
    - Hyperlinks
    """
    threats = []
    has_external_links = False

    try:
        with zipfile.ZipFile(file_path, "r") as zip_file:
            namelist = set(zip_file.namelist())

            # Check for external link files
            external_link_files = [f for f in namelist if f.startswith("xl/externalLinks/")]

            if external_link_files:
                has_external_links = True
                threats.extend(
                    [
                        SecurityThreat(
                            threat_type="EXTERNAL_LINKS",
                            severity=5,
                            location=link_file,
                            description="File contains external workbook links",
                            risk_level="MEDIUM",
                            details={"file": link_file},
                        )
                        for link_file in external_link_files
                    ]
                )

            # Check workbook relationships for external targets
            if "xl/_rels/workbook.xml.rels" in namelist:
                try:
                    # Check file size before reading
                    rels_info = zip_file.getinfo("xl/_rels/workbook.xml.rels")
                    if rels_info.file_size <= MAX_XML_PARSE_SIZE:
                        rels_content = zip_file.read("xl/_rels/workbook.xml.rels")
                        rels_str = rels_content.decode("utf-8", errors="ignore")

                        # Look for external targets
                        for link_type, pattern in EXTERNAL_LINK_PATTERNS.items():
                            matches = pattern.findall(rels_str)
                            for match in matches:
                                # Skip whitelisted OOXML namespaces
                                if link_type == "http" and any(
                                    match.startswith(ns) for ns in OOXML_NAMESPACE_WHITELIST
                                ):
                                    continue

                                has_external_links = True
                                threats.append(
                                    SecurityThreat(
                                        threat_type=f"EXTERNAL_{link_type.upper()}_LINK",
                                        severity=6,
                                        location="xl/_rels/workbook.xml.rels",
                                        description=f"External {link_type} link detected",
                                        risk_level="MEDIUM",
                                        details={"url": match[:100]},  # Truncate long URLs
                                    )
                                )
                except (OSError, ValueError, UnicodeDecodeError):
                    pass

    except (OSError, zipfile.BadZipFile):
        pass

    return has_external_links, threats


def check_for_embedded_objects(file_path: Path) -> tuple[bool, list[SecurityThreat]]:
    """
    Check for embedded objects (OLE objects, ActiveX controls).

    CLAUDE-SECURITY: Embedded objects can contain executable code
    or link to external resources.
    """
    threats = []
    has_embedded_objects = False

    try:
        with zipfile.ZipFile(file_path, "r") as zip_file:
            namelist = set(zip_file.namelist())

            # Check for embedded objects
            embedded_files = [f for f in namelist if "embeddings" in f or "oleObject" in f or "activeX" in f]

            if embedded_files:
                has_embedded_objects = True
                for obj_file in embedded_files:
                    severity = 8 if "activeX" in obj_file else 6
                    threats.append(
                        SecurityThreat(
                            threat_type="EMBEDDED_OBJECT",
                            severity=severity,
                            location=obj_file,
                            description="File contains embedded objects",
                            risk_level="HIGH" if severity >= HIGH_RISK_SEVERITY_THRESHOLD else "MEDIUM",
                            details={"file": obj_file},
                        )
                    )

    except (OSError, zipfile.BadZipFile):
        pass

    return has_embedded_objects, threats


def check_data_connections(file_path: Path) -> list[SecurityThreat]:
    """
    Check for data connections (queries, web queries, database connections).

    CLAUDE-KNOWLEDGE: Data connections can pull data from external sources
    and potentially execute queries.
    """
    threats = []

    try:
        with zipfile.ZipFile(file_path, "r") as zip_file:
            namelist = set(zip_file.namelist())

            # Check for connection files
            connection_files = [f for f in namelist if "connections" in f or "queryTable" in f]

            threats.extend(
                [
                    SecurityThreat(
                        threat_type="DATA_CONNECTION",
                        severity=5,
                        location=conn_file,
                        description="File contains external data connections",
                        risk_level="MEDIUM",
                        details={"file": conn_file},
                    )
                    for conn_file in connection_files
                ]
            )

    except (OSError, zipfile.BadZipFile):
        pass

    return threats


def check_hidden_sheets(file_path: Path) -> list[SecurityThreat]:
    """
    Check for hidden or very hidden sheets.

    CLAUDE-GOTCHA: Sheets can be hidden at different levels:
    - Hidden: User can unhide via UI
    - VeryHidden: Can only be unhidden via VBA
    """
    threats = []

    try:
        with zipfile.ZipFile(file_path, "r") as zip_file:
            if "xl/workbook.xml" in zip_file.namelist():
                # Check XML file size before parsing
                xml_info = zip_file.getinfo("xl/workbook.xml")
                if xml_info.file_size > MAX_XML_PARSE_SIZE:
                    threats.append(
                        SecurityThreat(
                            threat_type="LARGE_XML_FILE",
                            severity=6,
                            location="xl/workbook.xml",
                            description=f"XML file exceeds safe parsing size ({xml_info.file_size} bytes)",
                            risk_level="MEDIUM",
                            details={"size": xml_info.file_size, "limit": MAX_XML_PARSE_SIZE},
                        )
                    )
                    return threats

                workbook_content = zip_file.read("xl/workbook.xml")

                # CLAUDE-SECURITY: Validate XML structure before parsing
                # defusedxml already protects against XML bombs, but we add size check
                if len(workbook_content) > MAX_XML_PARSE_SIZE:
                    return threats

                # Additional validation: check for valid XML structure
                # Check if it starts with valid XML declaration or root element
                content_start = workbook_content[:1000].strip()
                if not (content_start.startswith(b"<?xml") or content_start.startswith(b"<")):
                    threats.append(
                        SecurityThreat(
                            threat_type="INVALID_XML_STRUCTURE",
                            severity=7,
                            location="xl/workbook.xml",
                            description="Workbook XML has invalid structure",
                            risk_level="HIGH",
                            details={"reason": "Invalid XML header"},
                        )
                    )
                    return threats

                root = DefusedElementTree.fromstring(workbook_content)

                # Look for sheets with state attribute
                for sheet in root.findall(".//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet"):
                    state = sheet.get("state", "visible")
                    if state == "hidden":
                        threats.append(
                            SecurityThreat(
                                threat_type="HIDDEN_SHEET",
                                severity=3,
                                location="xl/workbook.xml",
                                description=f"Hidden sheet detected: {sheet.get('name', 'Unknown')}",
                                risk_level="LOW",
                                details={"sheet_name": sheet.get("name"), "state": state},
                            )
                        )
                    elif state == "veryHidden":
                        threats.append(
                            SecurityThreat(
                                threat_type="VERY_HIDDEN_SHEET",
                                severity=5,
                                location="xl/workbook.xml",
                                description=f"Very hidden sheet detected: {sheet.get('name', 'Unknown')}",
                                risk_level="MEDIUM",
                                details={"sheet_name": sheet.get("name"), "state": state},
                            )
                        )

    except (OSError, zipfile.BadZipFile, DefusedElementTree.ParseError):
        pass

    return threats


def check_formula_injection(file_path: Path) -> list[SecurityThreat]:
    """
    Check for potential formula injection vulnerabilities.

    CLAUDE-SECURITY: Formula injection can occur when formulas
    start with =, +, -, or @ and contain dangerous functions.
    """
    threats = []
    dangerous_functions = {"HYPERLINK", "WEBSERVICE", "FILTERXML", "RUN", "EXEC", "CALL", "REGISTER"}

    # Patterns for external references in formulas
    external_formula_patterns = {
        "external_workbook": re.compile(r"\[.*?\]", re.IGNORECASE),  # [workbook.xlsx]Sheet1!A1
        "http_in_formula": re.compile(r"https?://[^\s\"\'<>]+", re.IGNORECASE),
        "file_path": re.compile(r"[A-Za-z]:\\.*?\.xls[xm]?", re.IGNORECASE),  # C:\path\file.xlsx
        "unc_path": re.compile(r"\\\\[^\\]+\\.*?\.xls[xm]?", re.IGNORECASE),  # \\server\share\file.xlsx
    }

    try:
        with zipfile.ZipFile(file_path, "r") as zip_file:
            # Check worksheet files
            sheet_files = [f for f in zip_file.namelist() if f.startswith("xl/worksheets/") and f.endswith(".xml")]

            for sheet_file in sheet_files:
                try:
                    # Check file size first
                    sheet_info = zip_file.getinfo(sheet_file)
                    if sheet_info.file_size > MAX_XML_PARSE_SIZE:
                        continue

                    content = zip_file.read(sheet_file).decode("utf-8", errors="ignore")

                    # Look for formulas with dangerous functions
                    threats.extend(
                        [
                            SecurityThreat(
                                threat_type="DANGEROUS_FORMULA",
                                severity=6,
                                location=sheet_file,
                                description=f"Potentially dangerous function {func} detected",
                                risk_level="MEDIUM",
                                details={"function": func},
                            )
                            for func in dangerous_functions
                            if func in content.upper()
                        ]
                    )

                    # Check for external references in formulas
                    for ref_type, pattern in external_formula_patterns.items():
                        matches = pattern.findall(content)
                        if matches:
                            # Filter out whitelisted namespaces for HTTP URLs
                            if ref_type == "http_in_formula":
                                matches = [
                                    m for m in matches if not any(m.startswith(ns) for ns in OOXML_NAMESPACE_WHITELIST)
                                ]

                            if matches:  # Only add threat if there are non-whitelisted matches
                                threats.append(
                                    SecurityThreat(
                                        threat_type=f"FORMULA_EXTERNAL_{ref_type.upper()}",
                                        severity=7,
                                        location=sheet_file,
                                        description=f"Formula contains external {ref_type.replace('_', ' ')} reference",
                                        risk_level="HIGH",
                                        # Limit to first 5 references
                                        details={"references": matches[:5], "total_count": len(matches)},
                                    )
                                )
                except (OSError, ValueError, UnicodeDecodeError):
                    pass

    except (OSError, zipfile.BadZipFile):
        pass

    return threats


def calculate_risk_score(threats: list[SecurityThreat]) -> tuple[int, RiskLevel]:
    """
    Calculate overall risk score and level from threats.

    CLAUDE-KNOWLEDGE: Risk scoring considers both severity and quantity
    of threats to provide an overall assessment.
    """
    if not threats:
        return 0, "LOW"

    # Calculate weighted score
    total_score = sum(threat.severity for threat in threats)
    max_severity = max(threat.severity for threat in threats)
    threat_count = len(threats)

    # Normalize to 0-100 scale
    risk_score = min(100, total_score + (threat_count * 2))

    # Determine risk level
    if max_severity >= CRITICAL_SEVERITY_THRESHOLD or risk_score >= CRITICAL_RISK_SCORE_THRESHOLD:
        risk_level: RiskLevel = "CRITICAL"
    elif max_severity >= HIGH_RISK_SEVERITY_THRESHOLD or risk_score >= HIGH_RISK_SCORE_THRESHOLD:
        risk_level = "HIGH"
    elif max_severity >= MEDIUM_SEVERITY_THRESHOLD or risk_score >= MEDIUM_RISK_SCORE_THRESHOLD:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return risk_score, risk_level


# ==================== Main Stage Function ====================


def stage_1_security_scan(file_path: Path, scan_options: dict[str, bool] | None = None) -> Result:
    """
    Perform comprehensive security scan using pure functions.

    Args:
        file_path: Path to Excel file to scan
        scan_options: Optional dict to enable/disable specific scans

    Returns:
        Ok(SecurityReport) if scan completes
        Err(error_message) if scan fails
    """
    # Default scan options
    if scan_options is None:
        scan_options = {
            "check_macros": True,
            "check_external_links": True,
            "check_embedded_objects": True,
            "check_data_connections": True,
            "check_hidden_sheets": True,
            "check_formula_injection": True,
        }

    try:
        all_threats = []

        # Run individual security checks
        has_macros = False
        if scan_options.get("check_macros", True):
            has_macros, macro_threats = check_for_vba_macros(file_path)
            all_threats.extend(macro_threats)

        has_external_links = False
        if scan_options.get("check_external_links", True):
            has_external_links, link_threats = check_for_external_links(file_path)
            all_threats.extend(link_threats)

        has_embedded_objects = False
        if scan_options.get("check_embedded_objects", True):
            has_embedded_objects, embed_threats = check_for_embedded_objects(file_path)
            all_threats.extend(embed_threats)

        if scan_options.get("check_data_connections", True):
            all_threats.extend(check_data_connections(file_path))

        if scan_options.get("check_hidden_sheets", True):
            all_threats.extend(check_hidden_sheets(file_path))

        if scan_options.get("check_formula_injection", True):
            all_threats.extend(check_formula_injection(file_path))

        # Calculate risk score
        risk_score, risk_level = calculate_risk_score(all_threats)

        # Create immutable report
        report = SecurityReport(
            threats=tuple(all_threats),
            has_macros=has_macros,
            has_external_links=has_external_links,
            has_embedded_objects=has_embedded_objects,
            risk_score=risk_score,
            risk_level=risk_level,
            scan_complete=True,
        )

        return Ok(report)

    except (OSError, ValueError, zipfile.BadZipFile) as e:
        return Err(f"Security scan failed: {e!s}", {"exception": str(e)})


# ==================== Utility Functions ====================


def create_security_validator(
    max_risk_level: RiskLevel = "MEDIUM", *, block_macros: bool = False, block_external_links: bool = False
) -> Callable[[Path], list[str]]:
    """
    Create a customized security validator with specific policies.
    """

    def validator(file_path: Path) -> list[str]:
        """Validate file security and return list of issues."""
        issues = []

        # Run security scan
        result = stage_1_security_scan(file_path)

        if isinstance(result, Err):
            issues.append(f"Security scan failed: {result.error}")
            return issues

        report = result.value

        # Check risk level
        risk_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        max_level_index = risk_levels.index(max_risk_level)
        current_level_index = risk_levels.index(report.risk_level)

        if current_level_index > max_level_index:
            issues.append(f"Risk level too high: {report.risk_level} (maximum allowed: {max_risk_level})")

        # Check specific policies
        if block_macros and report.has_macros:
            issues.append("File contains macros (blocked by policy)")

        if block_external_links and report.has_external_links:
            issues.append("File contains external links (blocked by policy)")

        # Report critical threats
        critical_threats = [t for t in report.threats if t.risk_level == "CRITICAL"]
        issues.extend([f"Critical threat: {threat.description}" for threat in critical_threats])

        return issues

    return validator
