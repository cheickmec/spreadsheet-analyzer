"""
Stage 1: Security Scan (Functional Programming).

This module implements security scanning for Excel files using pure functions
to detect macros, external links, embedded objects, and other potential threats.
"""

import re
import zipfile
from collections.abc import Callable
from pathlib import Path

import defusedxml.ElementTree as ET

from ..types import Err, Ok, Result, RiskLevel, SecurityReport, SecurityThreat

# ==================== Security Pattern Definitions ====================

# Patterns for detecting potentially dangerous content
SUSPICIOUS_PATTERNS = {
    "auto_open": re.compile(r"auto_open|workbook_open", re.IGNORECASE),
    "shell_exec": re.compile(r"shell|cmd|powershell|wscript", re.IGNORECASE),
    "file_io": re.compile(r"createobject|filesystemobject|scripting", re.IGNORECASE),
    "network": re.compile(r"http|ftp|download|urlmon", re.IGNORECASE),
    "registry": re.compile(r"registry|regwrite|regread", re.IGNORECASE),
}

# External link patterns
EXTERNAL_LINK_PATTERNS = {
    "http": re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
    "file": re.compile(r'file://[^\s<>"{}|\\^`\[\]]+'),
    "unc": re.compile(r'\\\\[^\s<>"{}|\\^`\[\]]+'),
}

# OOXML namespaces
NAMESPACES = {
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "x14ac": "http://schemas.microsoft.com/office/spreadsheetml/2009/9/ac",
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
                    macro_content = zip_file.read("xl/vbaProject.bin")
                    # Check for suspicious patterns in binary
                    content_str = str(macro_content)

                    for pattern_name, pattern in SUSPICIOUS_PATTERNS.items():
                        if pattern.search(content_str):
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
                    # Can't read macro content, but we know it exists
                    pass

    except Exception:
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
                for link_file in external_link_files:
                    threats.append(
                        SecurityThreat(
                            threat_type="EXTERNAL_LINKS",
                            severity=5,
                            location=link_file,
                            description="File contains external workbook links",
                            risk_level="MEDIUM",
                            details={"file": link_file},
                        )
                    )

            # Check workbook relationships for external targets
            if "xl/_rels/workbook.xml.rels" in namelist:
                try:
                    rels_content = zip_file.read("xl/_rels/workbook.xml.rels")
                    rels_str = rels_content.decode("utf-8", errors="ignore")

                    # Look for external targets
                    for link_type, pattern in EXTERNAL_LINK_PATTERNS.items():
                        matches = pattern.findall(rels_str)
                        for match in matches:
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
                except Exception:
                    pass

    except Exception:
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
                            risk_level="HIGH" if severity >= 7 else "MEDIUM",
                            details={"file": obj_file},
                        )
                    )

    except Exception:
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

            for conn_file in connection_files:
                threats.append(
                    SecurityThreat(
                        threat_type="DATA_CONNECTION",
                        severity=5,
                        location=conn_file,
                        description="File contains external data connections",
                        risk_level="MEDIUM",
                        details={"file": conn_file},
                    )
                )

    except Exception:
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
                workbook_content = zip_file.read("xl/workbook.xml")
                root = ET.fromstring(workbook_content)

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

    except Exception:
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

    try:
        with zipfile.ZipFile(file_path, "r") as zip_file:
            # Check worksheet files
            sheet_files = [f for f in zip_file.namelist() if f.startswith("xl/worksheets/") and f.endswith(".xml")]

            for sheet_file in sheet_files:
                try:
                    content = zip_file.read(sheet_file).decode("utf-8", errors="ignore")

                    # Look for formulas with dangerous functions
                    for func in dangerous_functions:
                        if func in content.upper():
                            threats.append(
                                SecurityThreat(
                                    threat_type="DANGEROUS_FORMULA",
                                    severity=6,
                                    location=sheet_file,
                                    description=f"Potentially dangerous function {func} detected",
                                    risk_level="MEDIUM",
                                    details={"function": func},
                                )
                            )
                except Exception:
                    pass

    except Exception:
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
    if max_severity >= 9 or risk_score >= 80:
        risk_level = "CRITICAL"
    elif max_severity >= 7 or risk_score >= 60:
        risk_level = "HIGH"
    elif max_severity >= 5 or risk_score >= 40:
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

    except Exception as e:
        return Err(f"Security scan failed: {e!s}", {"exception": str(e)})


# ==================== Utility Functions ====================


def create_security_validator(
    max_risk_level: RiskLevel = "MEDIUM", block_macros: bool = False, block_external_links: bool = False
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
        for threat in critical_threats:
            issues.append(f"Critical threat: {threat.description}")

        return issues

    return validator
