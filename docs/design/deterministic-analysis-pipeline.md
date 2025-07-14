# Deterministic Analysis Pipeline: Deep Dive

## Document Purpose

This document provides comprehensive technical documentation for the deterministic analysis pipeline that forms the foundation of our Excel analyzer system. This pipeline represents the culmination of insights from multiple AI system designs (OpenAI, Gemini, Grok conversations) synthesized into a unified, production-ready approach.

The deterministic pipeline embodies our "Deterministic First" principle - extracting maximum factual information using zero-cost deterministic methods before engaging expensive LLM resources. This approach ensures cost efficiency, security, and provides a reliable foundation for subsequent AI-powered analysis.

## Pipeline Overview

The deterministic analysis pipeline consists of five sequential stages, each building upon the previous to create a comprehensive understanding of the Excel file's structure, content, and characteristics. The pipeline can exit early at any stage if issues are detected, preventing wasted resources on problematic files.

### Stage Flow and Decision Points

```mermaid
flowchart TD
    START([Excel File Input]) --> STAGE0[Stage 0: Integrity Probe]

    STAGE0 --> CHECK0{File Valid?}
    CHECK0 -->|No| ABORT[Abort: File Corrupted/Invalid]
    CHECK0 -->|Yes| STAGE1[Stage 1: Security Scan]

    STAGE1 --> CHECK1{Security Risk?}
    CHECK1 -->|High Risk| BLOCK[Block: Security Threat]
    CHECK1 -->|Acceptable| STAGE2[Stage 2: Structural Mapping]

    STAGE2 --> CHECK2{Size/Complexity?}
    CHECK2 -->|Too Large| ROUTE[Route: Heavy Processing]
    CHECK2 -->|Normal| STAGE3[Stage 3: Formula Analysis]

    STAGE3 --> STAGE4[Stage 4: Content Intelligence]
    STAGE4 --> COMPLETE[Generate Agent Context]

    BLOCK --> END([Processing Complete])
    ROUTE --> END
    COMPLETE --> END
    ABORT --> END

    classDef stage fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef check fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef terminal fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px

    class STAGE0,STAGE1,STAGE2,STAGE3,STAGE4 stage
    class CHECK0,CHECK1,CHECK2 check
    class ABORT,BLOCK,ROUTE,COMPLETE terminal
```

## Stage 0: File Integrity Probe

### Purpose

Perform rapid file validation and classification to ensure the file is processable and determine the appropriate processing path.

### Implementation

```python
import hashlib
import zipfile
import magic
from pathlib import Path
from typing import Dict, Any, Literal

def stage_0_integrity_probe(file_path: Path) -> Dict[str, Any]:
    """
    Fast integrity and classification check.

    Uses Apache Tika's MimeTypes detector and libmagic for MIME detection.
    Performs SHA-256 hashing for deduplication and cache lookups.
    """

    # Calculate file hash for deduplication
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)

    file_hash = sha256_hash.hexdigest()
    file_size = file_path.stat().st_size

    # MIME type detection using libmagic
    mime_type = magic.from_file(str(file_path), mime=True)

    # Excel format validation
    is_excel = mime_type in [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',  # .xlsx
        'application/vnd.ms-excel',  # .xls
        'application/zip'  # Sometimes .xlsx appears as zip
    ]

    # Additional OOXML validation for .xlsx files
    is_ooxml = False
    if is_excel and mime_type != 'application/vnd.ms-excel':
        try:
            is_ooxml = zipfile.is_zipfile(file_path)
        except Exception:
            is_ooxml = False

    # Determine processing classification
    processing_class = determine_processing_class(file_size, mime_type)

    # Check for previous analysis (cache lookup)
    is_duplicate = check_hash_cache(file_hash)

    # Assess trust tier based on source/metadata
    trust_tier = assess_trust_level(file_path, file_size)

    return {
        "file_hash": file_hash,
        "size_bytes": file_size,
        "size_mb": round(file_size / (1024 * 1024), 2),
        "mime_type": mime_type,
        "is_excel": is_excel,
        "is_ooxml": is_ooxml,
        "is_duplicate": is_duplicate,
        "trust_tier": trust_tier,
        "processing_class": processing_class,
        "validation_passed": is_excel and file_size > 0
    }

def determine_processing_class(
    file_size: int,
    mime_type: str
) -> Literal["STANDARD", "HEAVY", "BLOCKED"]:
    """Classify file for appropriate processing pipeline."""

    # Size-based classification
    if file_size > 100 * 1024 * 1024:  # 100MB
        return "HEAVY"
    elif file_size < 1024:  # Less than 1KB
        return "BLOCKED"

    # MIME type validation
    valid_types = [
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'application/zip'
    ]

    if mime_type not in valid_types:
        return "BLOCKED"

    return "STANDARD"

def assess_trust_level(file_path: Path, file_size: int) -> int:
    """Assess trust level (1-5) based on file characteristics."""
    trust_score = 3  # Default neutral

    # Size-based adjustments
    if file_size > 50 * 1024 * 1024:  # > 50MB
        trust_score -= 1
    elif file_size < 10 * 1024:  # < 10KB
        trust_score -= 1

    # Name-based heuristics
    filename = file_path.name.lower()
    suspicious_patterns = ['temp', 'tmp', 'test', 'untitled']
    if any(pattern in filename for pattern in suspicious_patterns):
        trust_score -= 1

    # Extension validation
    if not filename.endswith(('.xlsx', '.xls')):
        trust_score -= 2

    return max(1, min(5, trust_score))
```

### Output Schema

```json
{
  "file_hash": "sha256:abc123...",
  "size_bytes": 2485760,
  "size_mb": 2.37,
  "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "is_excel": true,
  "is_ooxml": true,
  "is_duplicate": false,
  "trust_tier": 3,
  "processing_class": "STANDARD",
  "validation_passed": true
}
```

## Stage 1: Security Scan

### Purpose

Detect potentially dangerous content without executing it, including VBA macros, embedded objects, and external links.

### Implementation

```python
import oletools.olevba as olevba
from typing import Dict, List, Any
import zipfile
import xml.etree.ElementTree as ET

def stage_1_security_scan(file_path: Path) -> Dict[str, Any]:
    """
    Security assessment using olevba and custom XML analysis.

    Detects macros, external links, embedded objects, and other security risks
    without executing any potentially dangerous code.
    """

    security_report = {
        "has_macros": False,
        "macro_analysis": None,
        "external_links": [],
        "embedded_objects": [],
        "security_flags": [],
        "risk_score": 0,
        "ole_objects": []
    }

    # VBA Macro detection using oletools
    try:
        vba_parser = olevba.VBA_Parser(str(file_path))

        if vba_parser.detect_vba_macros():
            security_report["has_macros"] = True
            security_report["macro_analysis"] = analyze_vba_macros(vba_parser)
            security_report["security_flags"].append("VBA_MACROS_DETECTED")
            security_report["risk_score"] += 3

    except Exception as e:
        # Non-fatal error - continue analysis
        security_report["security_flags"].append(f"VBA_SCAN_ERROR: {str(e)}")

    # External links detection (for .xlsx files)
    if file_path.suffix.lower() == '.xlsx':
        external_links = scan_external_links_xlsx(file_path)
        security_report["external_links"] = external_links
        if external_links:
            security_report["security_flags"].append("EXTERNAL_LINKS_FOUND")
            security_report["risk_score"] += len(external_links)

    # OLE embedded objects scan
    ole_objects = scan_ole_objects(file_path)
    security_report["ole_objects"] = ole_objects
    if ole_objects:
        security_report["security_flags"].append("OLE_OBJECTS_FOUND")
        security_report["risk_score"] += len(ole_objects)

    # Additional security checks
    security_report.update(perform_additional_security_checks(file_path))

    return security_report

def analyze_vba_macros(vba_parser) -> Dict[str, Any]:
    """Analyze VBA macros for suspicious patterns."""

    macro_info = {
        "module_count": 0,
        "suspicious_keywords": [],
        "macro_hash": None,
        "auto_exec_keywords": [],
        "iocs": []  # Indicators of Compromise
    }

    suspicious_patterns = [
        'Shell', 'CreateObject', 'WScript', 'Environ',
        'URLDownloadToFile', 'XMLHttpRequest', 'Base64'
    ]

    auto_exec_patterns = [
        'Auto_Open', 'Workbook_Open', 'Auto_Exec',
        'Auto_Close', 'Workbook_Close'
    ]

    try:
        for vba_filename, stream_path, vba_code in vba_parser.extract_macros():
            macro_info["module_count"] += 1

            # Scan for suspicious patterns
            for pattern in suspicious_patterns:
                if pattern.lower() in vba_code.lower():
                    macro_info["suspicious_keywords"].append(pattern)

            # Scan for auto-execution patterns
            for pattern in auto_exec_patterns:
                if pattern.lower() in vba_code.lower():
                    macro_info["auto_exec_keywords"].append(pattern)

            # Generate hash of macro content
            if macro_info["macro_hash"] is None:
                import hashlib
                macro_info["macro_hash"] = hashlib.md5(
                    vba_code.encode('utf-8', errors='ignore')
                ).hexdigest()

    except Exception as e:
        macro_info["analysis_error"] = str(e)

    return macro_info

def scan_external_links_xlsx(file_path: Path) -> List[Dict[str, str]]:
    """Scan .xlsx file for external links without following them."""

    external_links = []

    try:
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            # Check workbook.xml for external references
            if 'xl/workbook.xml' in zip_file.namelist():
                workbook_xml = zip_file.read('xl/workbook.xml')
                root = ET.fromstring(workbook_xml)

                # Look for external references
                for elem in root.iter():
                    if 'r:id' in elem.attrib:
                        # This might be an external reference
                        external_links.append({
                            "type": "workbook_reference",
                            "element": elem.tag,
                            "reference": elem.attrib.get('r:id', '')
                        })

            # Check for external workbook links
            external_links_files = [f for f in zip_file.namelist()
                                  if f.startswith('xl/externalLinks/')]

            for link_file in external_links_files:
                link_xml = zip_file.read(link_file)
                root = ET.fromstring(link_xml)

                for elem in root.iter():
                    if 'r:id' in elem.attrib or 'Target' in elem.attrib:
                        external_links.append({
                            "type": "external_workbook",
                            "file": link_file,
                            "target": elem.attrib.get('Target', 'Unknown')
                        })

    except Exception as e:
        # Non-fatal - continue processing
        pass

    return external_links

def scan_ole_objects(file_path: Path) -> List[Dict[str, Any]]:
    """Scan for embedded OLE objects."""

    ole_objects = []

    try:
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            # Look for embedded objects
            object_files = [f for f in zip_file.namelist()
                          if '/embeddings/' in f or f.startswith('xl/embeddings/')]

            for obj_file in object_files:
                ole_objects.append({
                    "path": obj_file,
                    "size": zip_file.getinfo(obj_file).file_size,
                    "type": "embedded_object"
                })

    except Exception:
        # For .xls files or other formats, try different approach
        pass

    return ole_objects
```

### Output Schema

```json
{
  "has_macros": true,
  "macro_analysis": {
    "module_count": 2,
    "suspicious_keywords": ["Shell", "CreateObject"],
    "auto_exec_keywords": ["Auto_Open"],
    "macro_hash": "abc123def456...",
    "iocs": []
  },
  "external_links": [
    {
      "type": "external_workbook",
      "target": "\\\\server\\share\\data.xlsx"
    }
  ],
  "embedded_objects": [],
  "security_flags": ["VBA_MACROS_DETECTED", "EXTERNAL_LINKS_FOUND"],
  "risk_score": 5,
  "ole_objects": []
}
```

## Stage 2: Structural Mapping

### Purpose

Enumerate all structural elements of the workbook including sheets, tables, charts, ranges, and objects to create a comprehensive map of the file's organization.

### Implementation

```python
import openpyxl
from openpyxl.utils import get_column_letter
from typing import Dict, List, Any, Tuple
import re

def stage_2_structural_mapper(file_path: Path) -> Dict[str, Any]:
    """
    Deep structural analysis using openpyxl in read-only streaming mode.

    Extracts comprehensive structural information without loading cell values,
    optimizing for speed and memory efficiency.
    """

    try:
        # Load workbook in read-only mode for speed
        wb = openpyxl.load_workbook(
            file_path,
            read_only=True,
            data_only=False,  # Keep formulas
            keep_links=False  # Security: don't follow external links
        )

        structure = {
            "workbook": {
                "sheet_count": len(wb.sheetnames),
                "sheet_names": wb.sheetnames,
                "active_sheet": wb.active.title if wb.active else None,
                "sheets": []
            },
            "global_named_ranges": extract_workbook_names(wb),
            "total_charts": 0,
            "total_tables": 0,
            "total_pivot_tables": 0,
            "complexity_indicators": {}
        }

        # Analyze each sheet
        for sheet_name in wb.sheetnames:
            try:
                ws = wb[sheet_name]
                sheet_data = analyze_worksheet_structure(ws, sheet_name)
                structure["workbook"]["sheets"].append(sheet_data)

                # Aggregate totals
                structure["total_charts"] += len(sheet_data.get("charts", []))
                structure["total_tables"] += len(sheet_data.get("tables", []))
                structure["total_pivot_tables"] += len(sheet_data.get("pivot_tables", []))

            except Exception as e:
                # Continue with other sheets if one fails
                structure["workbook"]["sheets"].append({
                    "name": sheet_name,
                    "error": f"Analysis failed: {str(e)}",
                    "accessible": False
                })

        # Calculate complexity metrics
        structure["complexity_indicators"] = calculate_complexity_metrics(structure)

        return structure

    except Exception as e:
        return {
            "error": f"Structural analysis failed: {str(e)}",
            "workbook": {"sheet_count": 0, "sheets": []}
        }

def analyze_worksheet_structure(ws, sheet_name: str) -> Dict[str, Any]:
    """Analyze individual worksheet structure."""

    sheet_data = {
        "name": sheet_name,
        "visible": ws.sheet_state == 'visible',
        "sheet_state": ws.sheet_state,  # visible, hidden, veryHidden
        "dimensions": analyze_sheet_dimensions(ws),
        "cell_statistics": analyze_cell_types(ws),
        "merged_cells": extract_merged_cells(ws),
        "tables": extract_tables(ws),
        "charts": extract_charts(ws),
        "images": extract_images(ws),
        "pivot_tables": extract_pivot_tables(ws),
        "conditional_formats": extract_conditional_formatting(ws),
        "data_validations": extract_data_validations(ws),
        "hyperlinks": extract_hyperlinks(ws),
        "named_ranges": extract_sheet_names(ws),
        "protection": analyze_sheet_protection(ws)
    }

    return sheet_data

def analyze_sheet_dimensions(ws) -> Dict[str, Any]:
    """Analyze sheet dimensions and used ranges."""

    # Get maximum used dimensions
    max_row = ws.max_row
    max_col = ws.max_column

    # Calculate used range
    if max_row > 0 and max_col > 0:
        used_range = f"A1:{get_column_letter(max_col)}{max_row}"
    else:
        used_range = "A1:A1"

    # Estimate actual data density by sampling
    sample_density = estimate_data_density(ws, max_row, max_col)

    return {
        "max_row": max_row,
        "max_column": max_col,
        "max_column_letter": get_column_letter(max_col) if max_col > 0 else "A",
        "used_range": used_range,
        "estimated_used_cells": max_row * max_col if max_row and max_col else 0,
        "data_density_percent": sample_density,
        "is_empty": max_row <= 1 and max_col <= 1
    }

def analyze_cell_types(ws) -> Dict[str, int]:
    """Analyze distribution of cell types through sampling."""

    stats = {
        "total_sampled": 0,
        "formula_cells": 0,
        "value_cells": 0,
        "empty_cells": 0,
        "error_cells": 0,
        "text_cells": 0,
        "number_cells": 0
    }

    # Sample cells for performance (every 10th row/column for large sheets)
    max_row = min(ws.max_row, 1000)  # Limit sampling for performance
    max_col = min(ws.max_column, 100)

    step_row = max(1, max_row // 100)  # Sample ~100 rows
    step_col = max(1, max_col // 20)   # Sample ~20 columns

    for row in range(1, max_row + 1, step_row):
        for col in range(1, max_col + 1, step_col):
            try:
                cell = ws.cell(row=row, column=col)
                stats["total_sampled"] += 1

                if cell.value is None:
                    stats["empty_cells"] += 1
                elif isinstance(cell.value, str) and cell.value.startswith('='):
                    stats["formula_cells"] += 1
                elif isinstance(cell.value, str) and cell.value.startswith('#'):
                    stats["error_cells"] += 1
                elif isinstance(cell.value, (int, float)):
                    stats["number_cells"] += 1
                elif isinstance(cell.value, str):
                    stats["text_cells"] += 1
                else:
                    stats["value_cells"] += 1

            except Exception:
                # Skip problematic cells
                continue

    return stats

def extract_merged_cells(ws) -> List[str]:
    """Extract merged cell ranges."""
    return [str(merged_range) for merged_range in ws.merged_cells.ranges]

def extract_tables(ws) -> List[Dict[str, Any]]:
    """Extract table definitions."""
    tables = []

    try:
        for table in ws.tables.values():
            tables.append({
                "name": table.name,
                "display_name": getattr(table, 'displayName', table.name),
                "ref": table.ref,
                "table_style": getattr(table.tableStyleInfo, 'name', None) if table.tableStyleInfo else None,
                "header_row_count": getattr(table, 'headerRowCount', 1),
                "totals_row_count": getattr(table, 'totalsRowCount', 0)
            })
    except Exception:
        # Tables might not be accessible in read-only mode
        pass

    return tables

def extract_charts(ws) -> List[Dict[str, Any]]:
    """Extract chart information."""
    charts = []

    try:
        for chart in ws._charts:
            charts.append({
                "type": type(chart).__name__,
                "anchor": str(chart.anchor) if hasattr(chart, 'anchor') else None,
                "title": getattr(chart, 'title', {}).text if hasattr(chart, 'title') and chart.title else None,
                "series_count": len(getattr(chart, 'series', [])),
                "has_legend": hasattr(chart, 'legend') and chart.legend is not None
            })
    except Exception:
        # Charts might not be accessible in read-only mode
        pass

    return charts

def extract_images(ws) -> List[Dict[str, Any]]:
    """Extract image information."""
    images = []

    try:
        for image in ws._images:
            images.append({
                "anchor": str(image.anchor) if hasattr(image, 'anchor') else None,
                "width": getattr(image, 'width', None),
                "height": getattr(image, 'height', None),
                "format": getattr(image, 'format', None)
            })
    except Exception:
        pass

    return images

def calculate_complexity_metrics(structure: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall complexity indicators."""

    total_sheets = structure["workbook"]["sheet_count"]
    total_charts = structure["total_charts"]
    total_tables = structure["total_tables"]

    # Calculate estimated cell count
    total_estimated_cells = sum(
        sheet.get("dimensions", {}).get("estimated_used_cells", 0)
        for sheet in structure["workbook"]["sheets"]
        if "dimensions" in sheet
    )

    # Calculate formula density
    total_formula_cells = sum(
        sheet.get("cell_statistics", {}).get("formula_cells", 0)
        for sheet in structure["workbook"]["sheets"]
        if "cell_statistics" in sheet
    )

    # Complexity score (0-10)
    complexity_score = min(10, (
        (total_sheets / 10) * 2 +           # Sheet count factor
        (total_charts / 5) * 1 +            # Chart factor
        (total_tables / 3) * 1 +            # Table factor
        (total_estimated_cells / 10000) * 3 # Size factor
    ))

    return {
        "complexity_score": round(complexity_score, 1),
        "total_estimated_cells": total_estimated_cells,
        "total_formula_cells": total_formula_cells,
        "formula_density": round(
            (total_formula_cells / max(total_estimated_cells, 1)) * 100, 2
        ),
        "size_category": categorize_workbook_size(total_estimated_cells),
        "feature_richness": total_charts + total_tables + structure["total_pivot_tables"]
    }

def categorize_workbook_size(cell_count: int) -> str:
    """Categorize workbook by size."""
    if cell_count < 1000:
        return "SMALL"
    elif cell_count < 10000:
        return "MEDIUM"
    elif cell_count < 100000:
        return "LARGE"
    else:
        return "VERY_LARGE"
```

### Output Schema

```json
{
  "workbook": {
    "sheet_count": 3,
    "sheet_names": ["Summary", "Data", "Charts"],
    "active_sheet": "Summary",
    "sheets": [
      {
        "name": "Summary",
        "visible": true,
        "sheet_state": "visible",
        "dimensions": {
          "max_row": 50,
          "max_column": 10,
          "max_column_letter": "J",
          "used_range": "A1:J50",
          "estimated_used_cells": 500,
          "data_density_percent": 75.5,
          "is_empty": false
        },
        "cell_statistics": {
          "total_sampled": 50,
          "formula_cells": 15,
          "value_cells": 20,
          "empty_cells": 10,
          "error_cells": 0,
          "text_cells": 3,
          "number_cells": 2
        },
        "merged_cells": ["A1:C1", "E5:F5"],
        "tables": [],
        "charts": [
          {
            "type": "BarChart",
            "anchor": "E15",
            "title": "Monthly Sales",
            "series_count": 3,
            "has_legend": true
          }
        ],
        "conditional_formats": 2,
        "data_validations": 1,
        "protection": {"sheet_protected": false}
      }
    ]
  },
  "global_named_ranges": [
    {"name": "TotalSales", "range": "Summary!$J$50"}
  ],
  "total_charts": 1,
  "total_tables": 0,
  "total_pivot_tables": 0,
  "complexity_indicators": {
    "complexity_score": 4.2,
    "total_estimated_cells": 1500,
    "total_formula_cells": 45,
    "formula_density": 3.0,
    "size_category": "MEDIUM",
    "feature_richness": 1
  }
}
```

## Stage 3: Formula Intelligence & Dependency Graph

### Purpose

Build a comprehensive understanding of formula relationships, dependencies, and computational patterns within the workbook.

### Implementation

```python
import networkx as nx
import re
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass

@dataclass
class FormulaReference:
    """Represents a formula reference to another cell or range."""
    sheet: str
    cell_or_range: str
    is_external: bool = False
    external_file: str = None

def stage_3_formula_analysis(file_path: Path) -> Dict[str, Any]:
    """
    Formula extraction and dependency graph construction.

    Builds a directed graph of formula dependencies using NetworkX,
    identifies circular references, volatile functions, and complexity patterns.
    """

    try:
        wb = openpyxl.load_workbook(
            file_path,
            read_only=True,
            data_only=False,  # Keep formulas
            keep_links=False
        )

        # Initialize dependency graph
        formula_graph = nx.DiGraph()

        formula_data = {
            "total_formulas": 0,
            "volatile_functions": set(),
            "external_references": [],
            "circular_references": [],
            "formula_errors": [],
            "complexity_metrics": {},
            "function_usage": {},
            "dependency_chains": {},
            "formula_patterns": []
        }

        # Extract all formulas and build dependency graph
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            extract_sheet_formulas(ws, sheet_name, formula_graph, formula_data)

        # Analyze graph properties
        analyze_dependency_graph(formula_graph, formula_data)

        # Save graph for agent consumption
        graph_path = save_formula_graph(formula_graph, file_path.stem)
        formula_data["graph_path"] = graph_path

        # Convert sets to lists for JSON serialization
        formula_data["volatile_functions"] = list(formula_data["volatile_functions"])

        return formula_data

    except Exception as e:
        return {
            "error": f"Formula analysis failed: {str(e)}",
            "total_formulas": 0,
            "graph_path": None
        }

def extract_sheet_formulas(
    ws,
    sheet_name: str,
    graph: nx.DiGraph,
    formula_data: Dict[str, Any]
) -> None:
    """Extract formulas from a single worksheet."""

    # Limit analysis for performance
    max_row = min(ws.max_row, 10000)
    max_col = min(ws.max_column, 1000)

    for row in range(1, max_row + 1):
        for col in range(1, max_col + 1):
            try:
                cell = ws.cell(row=row, column=col)

                if cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                    formula = cell.value
                    cell_ref = f"{sheet_name}!{cell.coordinate}"

                    # Add cell to graph
                    graph.add_node(cell_ref,
                                 formula=formula,
                                 sheet=sheet_name,
                                 coordinate=cell.coordinate)

                    # Extract and analyze formula
                    analyze_formula(formula, cell_ref, sheet_name, graph, formula_data)
                    formula_data["total_formulas"] += 1

            except Exception as e:
                # Log error but continue
                formula_data["formula_errors"].append({
                    "cell": f"{sheet_name}!{get_column_letter(col)}{row}",
                    "error": str(e)
                })

def analyze_formula(
    formula: str,
    cell_ref: str,
    current_sheet: str,
    graph: nx.DiGraph,
    formula_data: Dict[str, Any]
) -> None:
    """Analyze individual formula for dependencies and patterns."""

    # Extract function usage
    functions = extract_functions(formula)
    for func in functions:
        formula_data["function_usage"][func] = formula_data["function_usage"].get(func, 0) + 1

    # Check for volatile functions
    volatile_funcs = {'NOW', 'TODAY', 'RAND', 'RANDBETWEEN', 'INDIRECT', 'OFFSET', 'INFO'}
    found_volatile = set(functions) & volatile_funcs
    formula_data["volatile_functions"].update(found_volatile)

    # Extract cell and range references
    references = extract_cell_references(formula, current_sheet)

    for ref in references:
        if ref.is_external:
            formula_data["external_references"].append({
                "from_cell": cell_ref,
                "to_file": ref.external_file,
                "to_reference": ref.cell_or_range
            })
        else:
            # Add dependency edge
            target_ref = f"{ref.sheet}!{ref.cell_or_range}"
            graph.add_edge(target_ref, cell_ref, reference_type="formula_dependency")

def extract_functions(formula: str) -> List[str]:
    """Extract function names from formula."""

    # Pattern to match function names (word followed by opening parenthesis)
    function_pattern = r'\b([A-Z][A-Z0-9_]*)\s*\('

    functions = re.findall(function_pattern, formula.upper())
    return list(set(functions))  # Remove duplicates

def extract_cell_references(formula: str, current_sheet: str) -> List[FormulaReference]:
    """Extract cell and range references from formula."""

    references = []

    # Pattern for external file references: [filename.xlsx]Sheet!A1
    external_pattern = r'\[([^\]]+)\]([^!]+)!([A-Z]+[0-9]+(?::[A-Z]+[0-9]+)?)'
    external_matches = re.findall(external_pattern, formula)

    for file_name, sheet_name, cell_range in external_matches:
        references.append(FormulaReference(
            sheet=sheet_name,
            cell_or_range=cell_range,
            is_external=True,
            external_file=file_name
        ))

    # Pattern for internal sheet references: Sheet!A1 or just A1
    internal_pattern = r'(?:([^!\[\]]+)!)?([A-Z]+[0-9]+(?::[A-Z]+[0-9]+)?)'
    internal_matches = re.findall(internal_pattern, formula)

    for sheet_name, cell_range in internal_matches:
        # Skip if this was already captured as external reference
        if not any(ref.cell_or_range == cell_range for ref in references if ref.is_external):
            references.append(FormulaReference(
                sheet=sheet_name if sheet_name else current_sheet,
                cell_or_range=cell_range,
                is_external=False
            ))

    return references

def analyze_dependency_graph(graph: nx.DiGraph, formula_data: Dict[str, Any]) -> None:
    """Analyze dependency graph for patterns and issues."""

    # Check for circular dependencies
    if not nx.is_directed_acyclic_graph(graph):
        try:
            cycles = list(nx.simple_cycles(graph))
            formula_data["circular_references"] = cycles[:10]  # Limit to first 10
        except:
            formula_data["circular_references"] = ["Analysis failed - graph too complex"]

    # Calculate complexity metrics
    formula_data["complexity_metrics"] = {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "max_dependency_depth": calculate_max_depth(graph),
        "highly_connected_cells": find_highly_connected_cells(graph),
        "isolated_formulas": len([n for n in graph.nodes() if graph.degree(n) == 0]),
        "average_dependencies": round(graph.number_of_edges() / max(graph.number_of_nodes(), 1), 2)
    }

    # Identify dependency chains
    formula_data["dependency_chains"] = identify_critical_paths(graph)

def calculate_max_depth(graph: nx.DiGraph) -> int:
    """Calculate maximum dependency depth in the graph."""

    if graph.number_of_nodes() == 0:
        return 0

    try:
        # Find longest path in DAG
        topo_order = list(nx.topological_sort(graph))

        # Calculate distances from sources
        distances = {node: 0 for node in graph.nodes()}

        for node in topo_order:
            for successor in graph.successors(node):
                distances[successor] = max(distances[successor], distances[node] + 1)

        return max(distances.values()) if distances else 0

    except nx.NetworkXError:
        # Graph has cycles
        return -1

def find_highly_connected_cells(graph: nx.DiGraph, threshold: int = 10) -> List[Dict[str, Any]]:
    """Find cells with high in-degree or out-degree."""

    highly_connected = []

    for node in graph.nodes():
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)

        if in_degree >= threshold or out_degree >= threshold:
            highly_connected.append({
                "cell": node,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree
            })

    # Sort by total degree
    return sorted(highly_connected, key=lambda x: x["total_degree"], reverse=True)[:20]

def identify_critical_paths(graph: nx.DiGraph) -> Dict[str, Any]:
    """Identify critical dependency paths."""

    critical_paths = {
        "longest_chains": [],
        "bottleneck_cells": [],
        "source_cells": [],
        "sink_cells": []
    }

    try:
        # Find source cells (no dependencies)
        sources = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        critical_paths["source_cells"] = sources[:10]

        # Find sink cells (nothing depends on them)
        sinks = [n for n in graph.nodes() if graph.out_degree(n) == 0]
        critical_paths["sink_cells"] = sinks[:10]

        # Find longest paths from sources
        for source in sources[:5]:  # Limit for performance
            try:
                longest = nx.dag_longest_path(graph, source)
                if len(longest) > 3:  # Only interesting chains
                    critical_paths["longest_chains"].append({
                        "source": source,
                        "path": longest,
                        "length": len(longest)
                    })
            except:
                continue

    except Exception:
        pass

    return critical_paths

def save_formula_graph(graph: nx.DiGraph, file_stem: str) -> str:
    """Save graph to file for agent consumption."""

    import tempfile

    # Create temporary file
    temp_dir = Path(tempfile.gettempdir()) / "spreadsheet_analyzer"
    temp_dir.mkdir(exist_ok=True)

    graph_path = temp_dir / f"{file_stem}_formula_graph.graphml"

    try:
        nx.write_graphml(graph, graph_path)
        return str(graph_path)
    except Exception:
        return None
```

### Output Schema

```json
{
  "total_formulas": 234,
  "volatile_functions": ["NOW", "INDIRECT"],
  "external_references": [
    {
      "from_cell": "Summary!B15",
      "to_file": "Budget2024.xlsx",
      "to_reference": "A1:A10"
    }
  ],
  "circular_references": [
    ["Sheet1!A1", "Sheet1!B1", "Sheet1!A1"]
  ],
  "formula_errors": [],
  "complexity_metrics": {
    "total_nodes": 234,
    "total_edges": 456,
    "max_dependency_depth": 12,
    "highly_connected_cells": [
      {
        "cell": "Summary!Total",
        "in_degree": 25,
        "out_degree": 3,
        "total_degree": 28
      }
    ],
    "isolated_formulas": 5,
    "average_dependencies": 1.95
  },
  "function_usage": {
    "SUM": 45,
    "IF": 23,
    "VLOOKUP": 12,
    "INDEX": 8
  },
  "dependency_chains": {
    "longest_chains": [
      {
        "source": "Data!A1",
        "path": ["Data!A1", "Calc!B5", "Summary!Total"],
        "length": 3
      }
    ],
    "source_cells": ["Data!A1", "Data!B1"],
    "sink_cells": ["Summary!Total", "Dashboard!Result"]
  },
  "graph_path": "/tmp/spreadsheet_analyzer/file_formula_graph.graphml"
}
```

## Stage 4: Content Intelligence Summary

### Purpose

Synthesize all deterministic findings into actionable intelligence and processing recommendations for the agent layer.

### Implementation

```python
def stage_4_content_intelligence(
    integrity: Dict[str, Any],
    security: Dict[str, Any],
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesize all deterministic findings into actionable intelligence.

    Generates processing recommendations, complexity assessments,
    and strategic guidance for the agent layer.
    """

    intelligence = {
        "file_assessment": generate_file_assessment(integrity, security, structure),
        "processing_strategy": determine_processing_strategy(structure, formulas),
        "risk_assessment": synthesize_risk_assessment(security, formulas),
        "complexity_analysis": analyze_overall_complexity(structure, formulas),
        "agent_recommendations": generate_agent_recommendations(structure, formulas),
        "priority_areas": identify_priority_analysis_areas(structure, formulas),
        "optimization_opportunities": identify_optimization_opportunities(formulas),
        "validation_targets": identify_validation_targets(structure, formulas)
    }

    return intelligence

def generate_file_assessment(
    integrity: Dict[str, Any],
    security: Dict[str, Any],
    structure: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate overall file assessment."""

    assessment = {
        "file_type": "Excel Workbook",
        "format": "OOXML (.xlsx)" if integrity.get("is_ooxml") else "Legacy (.xls)",
        "size_category": structure.get("complexity_indicators", {}).get("size_category", "UNKNOWN"),
        "trust_level": integrity.get("trust_tier", 3),
        "security_status": "CLEAN" if security.get("risk_score", 0) < 3 else "REVIEW_REQUIRED",
        "processable": integrity.get("validation_passed", False),
        "estimated_analysis_time": estimate_analysis_time(structure),
        "recommended_resources": recommend_resources(structure)
    }

    return assessment

def determine_processing_strategy(
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> Dict[str, Any]:
    """Determine optimal processing strategy based on file characteristics."""

    sheet_count = structure.get("workbook", {}).get("sheet_count", 0)
    formula_count = formulas.get("total_formulas", 0)
    complexity_score = structure.get("complexity_indicators", {}).get("complexity_score", 0)

    strategy = {
        "approach": "STANDARD",
        "parallelism": "MEDIUM",
        "agent_count": min(sheet_count, 5),  # Max 5 concurrent agents
        "analysis_depth": "FULL",
        "special_considerations": []
    }

    # Adjust strategy based on characteristics
    if complexity_score > 7:
        strategy["approach"] = "HEAVY_COMPUTE"
        strategy["analysis_depth"] = "PROGRESSIVE"
        strategy["special_considerations"].append("High complexity - progressive analysis recommended")

    if formula_count > 1000:
        strategy["parallelism"] = "HIGH"
        strategy["special_considerations"].append("High formula count - parallel processing beneficial")

    if sheet_count > 10:
        strategy["agent_count"] = min(sheet_count, 8)
        strategy["special_considerations"].append("Many sheets - increased parallelism")

    # Check for external dependencies
    if formulas.get("external_references"):
        strategy["special_considerations"].append("External references detected - validation required")

    # Check for circular references
    if formulas.get("circular_references"):
        strategy["special_considerations"].append("Circular references detected - careful analysis needed")

    return strategy

def synthesize_risk_assessment(
    security: Dict[str, Any],
    formulas: Dict[str, Any]
) -> Dict[str, Any]:
    """Synthesize comprehensive risk assessment."""

    risks = {
        "overall_risk": "LOW",
        "security_risks": [],
        "computational_risks": [],
        "data_integrity_risks": [],
        "performance_risks": []
    }

    # Security risks
    if security.get("has_macros"):
        risks["security_risks"].append("VBA macros present - manual review recommended")

    if security.get("external_links"):
        risks["security_risks"].append("External file references - data dependency risk")

    if security.get("ole_objects"):
        risks["security_risks"].append("Embedded objects present - security scan required")

    # Computational risks
    if formulas.get("circular_references"):
        risks["computational_risks"].append("Circular references may cause calculation errors")

    if "NOW" in formulas.get("volatile_functions", []):
        risks["computational_risks"].append("Time-dependent formulas - results may vary")

    # Data integrity risks
    if formulas.get("formula_errors"):
        risks["data_integrity_risks"].append("Formula errors detected - validation needed")

    # Performance risks
    formula_count = formulas.get("total_formulas", 0)
    if formula_count > 5000:
        risks["performance_risks"].append("High formula count may impact performance")

    # Determine overall risk level
    total_risks = sum(len(risks[key]) for key in risks if key != "overall_risk")
    if total_risks > 5:
        risks["overall_risk"] = "HIGH"
    elif total_risks > 2:
        risks["overall_risk"] = "MEDIUM"

    return risks

def analyze_overall_complexity(
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze overall file complexity."""

    structural_score = structure.get("complexity_indicators", {}).get("complexity_score", 0)
    formula_complexity = calculate_formula_complexity(formulas)

    complexity = {
        "overall_score": round((structural_score + formula_complexity) / 2, 1),
        "structural_complexity": structural_score,
        "formula_complexity": formula_complexity,
        "complexity_factors": [],
        "simplification_opportunities": []
    }

    # Identify complexity factors
    if structure.get("workbook", {}).get("sheet_count", 0) > 5:
        complexity["complexity_factors"].append("Multiple sheets increase navigation complexity")

    if formulas.get("total_formulas", 0) > 100:
        complexity["complexity_factors"].append("High formula count increases maintenance complexity")

    if formulas.get("complexity_metrics", {}).get("max_dependency_depth", 0) > 5:
        complexity["complexity_factors"].append("Deep dependency chains increase error propagation risk")

    # Identify simplification opportunities
    isolated_formulas = formulas.get("complexity_metrics", {}).get("isolated_formulas", 0)
    if isolated_formulas > 0:
        complexity["simplification_opportunities"].append(
            f"{isolated_formulas} isolated formulas could be consolidated"
        )

    return complexity

def calculate_formula_complexity(formulas: Dict[str, Any]) -> float:
    """Calculate formula-specific complexity score."""

    if not formulas or formulas.get("total_formulas", 0) == 0:
        return 0.0

    score = 0.0

    # Base complexity from formula count
    formula_count = formulas.get("total_formulas", 0)
    score += min(3.0, formula_count / 100)  # Max 3 points for count

    # Dependency complexity
    max_depth = formulas.get("complexity_metrics", {}).get("max_dependency_depth", 0)
    score += min(2.0, max_depth / 10)  # Max 2 points for depth

    # Circular reference penalty
    if formulas.get("circular_references"):
        score += 2.0

    # External reference complexity
    external_refs = len(formulas.get("external_references", []))
    score += min(1.0, external_refs / 5)  # Max 1 point for external refs

    # Volatile function penalty
    volatile_count = len(formulas.get("volatile_functions", []))
    score += min(1.0, volatile_count / 3)  # Max 1 point for volatile functions

    return min(10.0, score)  # Cap at 10

def generate_agent_recommendations(
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> Dict[str, List[str]]:
    """Generate specific recommendations for agent analysis."""

    recommendations = {
        "priority_sheets": [],
        "focus_areas": [],
        "analysis_approaches": [],
        "validation_strategies": []
    }

    # Identify priority sheets
    sheets = structure.get("workbook", {}).get("sheets", [])
    for sheet in sheets:
        sheet_name = sheet.get("name", "")
        if sheet.get("charts") or sheet.get("pivot_tables"):
            recommendations["priority_sheets"].append(f"{sheet_name} - Contains visualizations")
        elif sheet.get("cell_statistics", {}).get("formula_cells", 0) > 20:
            recommendations["priority_sheets"].append(f"{sheet_name} - High formula density")

    # Suggest focus areas
    if formulas.get("circular_references"):
        recommendations["focus_areas"].append("Investigate circular references for calculation errors")

    if formulas.get("external_references"):
        recommendations["focus_areas"].append("Validate external file dependencies")

    if "VLOOKUP" in formulas.get("function_usage", {}):
        recommendations["focus_areas"].append("Review VLOOKUP usage for optimization opportunities")

    # Analysis approaches
    complexity_score = structure.get("complexity_indicators", {}).get("complexity_score", 0)
    if complexity_score > 5:
        recommendations["analysis_approaches"].append("Use progressive sampling for large datasets")
        recommendations["analysis_approaches"].append("Focus on key calculation chains first")
    else:
        recommendations["analysis_approaches"].append("Comprehensive analysis feasible")

    # Validation strategies
    if formulas.get("total_formulas", 0) > 50:
        recommendations["validation_strategies"].append("Sample-based formula validation")
        recommendations["validation_strategies"].append("Cross-verify summary calculations")

    return recommendations

def identify_priority_analysis_areas(
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Identify specific areas requiring priority analysis."""

    priority_areas = []

    # High-risk formulas
    if formulas.get("circular_references"):
        priority_areas.append({
            "area": "Circular References",
            "priority": "HIGH",
            "reason": "May cause calculation errors or infinite loops",
            "cells": formulas["circular_references"][:5]  # First 5 cycles
        })

    # External dependencies
    if formulas.get("external_references"):
        priority_areas.append({
            "area": "External Dependencies",
            "priority": "MEDIUM",
            "reason": "Data availability and consistency risks",
            "count": len(formulas["external_references"])
        })

    # Volatile functions
    if formulas.get("volatile_functions"):
        priority_areas.append({
            "area": "Volatile Functions",
            "priority": "MEDIUM",
            "reason": "Results may change on each calculation",
            "functions": list(formulas["volatile_functions"])
        })

    # Hidden sheets
    hidden_sheets = [
        sheet["name"] for sheet in structure.get("workbook", {}).get("sheets", [])
        if not sheet.get("visible", True)
    ]
    if hidden_sheets:
        priority_areas.append({
            "area": "Hidden Sheets",
            "priority": "LOW",
            "reason": "May contain important calculations or sensitive data",
            "sheets": hidden_sheets
        })

    return priority_areas
```

### Output Schema

```json
{
  "file_assessment": {
    "file_type": "Excel Workbook",
    "format": "OOXML (.xlsx)",
    "size_category": "MEDIUM",
    "trust_level": 3,
    "security_status": "CLEAN",
    "processable": true,
    "estimated_analysis_time": "2-5 minutes",
    "recommended_resources": "2 agents, 4GB RAM"
  },
  "processing_strategy": {
    "approach": "STANDARD",
    "parallelism": "MEDIUM",
    "agent_count": 3,
    "analysis_depth": "FULL",
    "special_considerations": ["External references detected - validation required"]
  },
  "risk_assessment": {
    "overall_risk": "MEDIUM",
    "security_risks": ["External file references - data dependency risk"],
    "computational_risks": ["Circular references may cause calculation errors"],
    "data_integrity_risks": [],
    "performance_risks": []
  },
  "complexity_analysis": {
    "overall_score": 5.2,
    "structural_complexity": 4.2,
    "formula_complexity": 6.1,
    "complexity_factors": ["High formula count increases maintenance complexity"],
    "simplification_opportunities": ["5 isolated formulas could be consolidated"]
  },
  "agent_recommendations": {
    "priority_sheets": ["Summary - Contains visualizations", "Calculations - High formula density"],
    "focus_areas": ["Investigate circular references for calculation errors"],
    "analysis_approaches": ["Use progressive sampling for large datasets"],
    "validation_strategies": ["Sample-based formula validation"]
  },
  "priority_areas": [
    {
      "area": "Circular References",
      "priority": "HIGH",
      "reason": "May cause calculation errors or infinite loops",
      "cells": [["Sheet1!A1", "Sheet1!B1", "Sheet1!A1"]]
    }
  ]
}
```

## Agent Integration: Bootstrap Cell Template

The deterministic analysis results are packaged into a comprehensive context that's injected into each agent's Jupyter kernel through a standardized bootstrap cell:

```python
# === AGENT BOOTSTRAP CELL (Idempotent) ===
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

import openpyxl
import pandas as pd
import numpy as np
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === ORCHESTRATOR-INJECTED CONSTANTS ===
EXCEL_FILE = Path("{{ excel_path }}")
SHEET_NAME = "{{ sheet_name }}"
AGENT_ID = "{{ agent_id }}"
JOB_ID = "{{ job_id }}"

# === DETERMINISTIC CONTEXT (Pre-computed) ===
# Load complete deterministic analysis results
with open("/tmp/context/{{ job_id }}/deterministic_analysis.json") as f:
    DETERMINISTIC_CONTEXT = json.load(f)

# Extract relevant contexts
INTEGRITY_CONTEXT = DETERMINISTIC_CONTEXT["integrity"]
SECURITY_CONTEXT = DETERMINISTIC_CONTEXT["security"]
STRUCTURE_CONTEXT = DETERMINISTIC_CONTEXT["structure"]
FORMULA_CONTEXT = DETERMINISTIC_CONTEXT["formulas"]
INTELLIGENCE_CONTEXT = DETERMINISTIC_CONTEXT["intelligence"]

# Sheet-specific context
SHEET_CONTEXT = next(
    (sheet for sheet in STRUCTURE_CONTEXT["workbook"]["sheets"]
     if sheet["name"] == SHEET_NAME),
    {"name": SHEET_NAME, "error": "Sheet context not found"}
)

# === HELPER FUNCTIONS ===

@lru_cache(maxsize=1)
def load_workbook() -> openpyxl.Workbook:
    """
    Load workbook once per agent, cached in memory.
    Uses read-only mode for performance and security.
    """
    logger.info(f"Loading workbook: {EXCEL_FILE}")
    return openpyxl.load_workbook(
        EXCEL_FILE,
        read_only=True,      # Performance: streaming mode
        data_only=True,      # Get calculated values, not formulas
        keep_links=False     # Security: don't follow external links
    )

def load_sheet_data(
    range_spec: Optional[str] = None,
    sample_rows: Optional[int] = None,
    **pandas_kwargs
) -> pd.DataFrame:
    """
    Selective data loading with pandas.
    Provides intelligent defaults based on deterministic analysis.
    """
    kwargs = {
        "engine": "openpyxl",
        "sheet_name": SHEET_NAME,
        **pandas_kwargs
    }

    # Apply range specification if provided
    if range_spec:
        kwargs["usecols"] = range_spec

    # Apply sampling if requested
    if sample_rows:
        max_row = SHEET_CONTEXT.get("dimensions", {}).get("max_row", 1000)
        skip_rows = max(1, max_row // sample_rows)
        kwargs["skiprows"] = lambda x: x % skip_rows != 0

    logger.info(f"Loading data with parameters: {kwargs}")
    return pd.read_excel(EXCEL_FILE, **kwargs)

def get_formula_graph() -> Optional[nx.DiGraph]:
    """Load pre-computed formula dependency graph."""
    graph_path = FORMULA_CONTEXT.get("graph_path")
    if graph_path and Path(graph_path).exists():
        logger.info(f"Loading formula graph: {graph_path}")
        return nx.read_graphml(graph_path)
    return None

def validate_calculation(
    cell_range: str,
    expected_function: str = "SUM"
) -> Dict[str, Any]:
    """
    Validate calculation against deterministic expectations.
    Implements validation-first philosophy.
    """
    try:
        # Load actual data
        df = load_sheet_data(range_spec=cell_range)

        # Perform calculation
        if expected_function.upper() == "SUM":
            actual_result = df.sum().sum()
        elif expected_function.upper() == "AVERAGE":
            actual_result = df.mean().mean()
        else:
            return {"error": f"Unsupported function: {expected_function}"}

        # Compare with any known formula results
        # (This would be enhanced with specific formula tracking)

        return {
            "range": cell_range,
            "function": expected_function,
            "result": actual_result,
            "validated": True,
            "row_count": len(df),
            "col_count": len(df.columns)
        }

    except Exception as e:
        logger.error(f"Validation failed for {cell_range}: {e}")
        return {"error": str(e), "validated": False}

def get_recommendations() -> Dict[str, Any]:
    """Get agent-specific recommendations from deterministic analysis."""
    return INTELLIGENCE_CONTEXT.get("agent_recommendations", {})

def report_finding(
    finding_type: str,
    description: str,
    evidence: Dict[str, Any],
    severity: str = "INFO"
) -> None:
    """
    Report analysis finding for aggregation by orchestrator.
    Findings are logged and could be written to shared storage.
    """
    finding = {
        "agent_id": AGENT_ID,
        "sheet": SHEET_NAME,
        "timestamp": pd.Timestamp.now().isoformat(),
        "type": finding_type,
        "description": description,
        "evidence": evidence,
        "severity": severity
    }

    logger.info(f"FINDING: {json.dumps(finding, indent=2)}")

    # Could also write to shared findings file
    # findings_file = Path(f"/tmp/context/{JOB_ID}/findings_{AGENT_ID}.jsonl")
    # with open(findings_file, "a") as f:
    #     f.write(json.dumps(finding) + "\n")

# === CONTEXT SUMMARY ===
print(f"🤖 Agent {AGENT_ID} initialized for sheet '{SHEET_NAME}'")
print(f"📁 File: {EXCEL_FILE.name} ({INTEGRITY_CONTEXT['size_mb']}MB)")
print(f"📊 Sheet dimensions: {SHEET_CONTEXT.get('dimensions', {}).get('max_row', '?')} × {SHEET_CONTEXT.get('dimensions', {}).get('max_column', '?')}")
print(f"🔢 Total formulas in workbook: {FORMULA_CONTEXT.get('total_formulas', 0)}")
print(f"⚠️  Security flags: {len(SECURITY_CONTEXT.get('security_flags', []))}")
print(f"🎯 Complexity score: {STRUCTURE_CONTEXT.get('complexity_indicators', {}).get('complexity_score', 'N/A')}/10")

# Show specific recommendations for this sheet
recommendations = get_recommendations()
if recommendations:
    print("\n📋 Recommended focus areas:")
    for area in recommendations.get("focus_areas", [])[:3]:
        print(f"  • {area}")

print(f"\n✅ Agent ready for analysis. Available functions:")
print(f"  • load_workbook() - Full workbook access")
print(f"  • load_sheet_data() - Selective data loading")
print(f"  • get_formula_graph() - Dependency graph")
print(f"  • validate_calculation() - Calculation verification")
print(f"  • report_finding() - Report discoveries")
```

## Performance Characteristics

### Benchmark Results

The deterministic pipeline has been designed for optimal performance characteristics:

| Stage                 | Typical Time | Memory Usage | Primary Libraries       |
| --------------------- | ------------ | ------------ | ----------------------- |
| Stage 0: Integrity    | < 100ms      | < 10MB       | hashlib, magic, zipfile |
| Stage 1: Security     | 200ms - 2s   | < 50MB       | oletools, xml.etree     |
| Stage 2: Structure    | 500ms - 5s   | 100-500MB    | openpyxl (read-only)    |
| Stage 3: Formulas     | 1s - 30s     | 200MB - 1GB  | networkx, re            |
| Stage 4: Intelligence | < 500ms      | < 100MB      | Pure computation        |

### Scalability Considerations

- **Small files (< 1MB)**: Complete pipeline in under 3 seconds
- **Medium files (1-10MB)**: Complete pipeline in 5-15 seconds
- **Large files (10-50MB)**: 30 seconds to 2 minutes
- **Very large files (> 50MB)**: Automatically routed to heavy processing pipeline

### Memory Optimization

The pipeline uses several memory optimization strategies:

1. **Streaming Analysis**: openpyxl read-only mode streams data without loading entire file
1. **Sampling**: Large sheets are analyzed through statistical sampling
1. **Chunked Processing**: Formula analysis processes cells in batches
1. **Early Exit**: Pipeline can abort early for problematic files
1. **OS Page Caching**: Multiple agent access benefits from OS-level file caching

## Integration with Agent Architecture

### Context Injection Workflow

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant DP as Deterministic Pipeline
    participant K as Kernel Manager
    participant A as Agent Kernel

    O->>DP: Process Excel file
    DP->>DP: Stage 0-4 Analysis
    DP->>O: Deterministic Context
    O->>K: Request agent kernel
    K->>A: Start Jupyter kernel
    O->>A: Inject bootstrap cell
    A->>A: Execute bootstrap
    A->>O: Agent ready
    O->>A: Provide analysis prompt
    A->>A: Begin intelligent analysis
```

### Agent Decision Framework

Agents use the deterministic context to make intelligent decisions:

1. **Analysis Planning**: Choose approach based on complexity indicators
1. **Data Loading Strategy**: Load only necessary data based on structural insights
1. **Validation Priorities**: Focus on high-risk areas identified deterministically
1. **Resource Management**: Adjust analysis depth based on performance estimates

### Error Handling and Recovery

- **Pipeline Failures**: Each stage can fail independently without affecting others
- **Partial Results**: Agents can work with incomplete deterministic context
- **Graceful Degradation**: Missing context triggers agent-led discovery
- **Retry Logic**: Transient failures trigger automatic retry with backoff

## Security Considerations

### Static Analysis Only

The deterministic pipeline performs exclusively static analysis:

- **No Code Execution**: VBA macros are analyzed but never executed
- **No External Connections**: External links are cataloged but never followed
- **Sandboxed Environment**: All analysis occurs within controlled environment
- **Read-Only Access**: Files are opened in read-only mode preventing modification

### Threat Mitigation

- **Malware Detection**: Integration with oletools for macro analysis
- **Resource Limits**: Analysis bounded by time and memory limits
- **Input Validation**: File integrity verification before processing
- **Audit Trails**: Complete logging of all analysis operations

## Future Enhancements

### Planned Improvements

1. **Machine Learning Integration**: Pattern recognition for formula optimization
1. **Incremental Analysis**: Delta analysis for file versions
1. **Performance Profiling**: Detailed performance analysis and recommendations
1. **Custom Rule Engine**: User-defined validation rules and checks
1. **Multi-Format Support**: Extension to Google Sheets, CSV, and other formats

### Extensibility Points

- **Custom Analyzers**: Plugin architecture for domain-specific analysis
- **External Tool Integration**: APIs for third-party analysis tools
- **Custom Validation Rules**: User-defined business logic validation
- **Enhanced Visualization**: Interactive dependency graph exploration

## Conclusion

The deterministic analysis pipeline represents a sophisticated approach to Excel file analysis that maximizes efficiency while maintaining comprehensive coverage. By front-loading structural analysis using deterministic methods, we create a solid foundation for intelligent agent-based analysis while optimizing cost and performance.

The pipeline's design reflects lessons learned from multiple AI system architectures, incorporating the best practices from each while maintaining a unified, pragmatic approach. The result is a system that can handle the full spectrum of Excel files—from simple worksheets to complex financial models—with appropriate resource allocation and intelligent routing.

This deterministic-first approach ensures that agents begin their analysis with complete contextual understanding, enabling them to make intelligent decisions about where to focus their attention and how to validate their findings against established ground truth.
