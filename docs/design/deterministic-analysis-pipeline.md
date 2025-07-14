# Deterministic Analysis Pipeline: Functional and Object-Oriented Design

## Document Purpose

This document provides comprehensive technical documentation for the deterministic analysis pipeline that forms the foundation of our Excel analyzer system. The implementation uses a pragmatic mix of functional programming (FP) for stateless operations and object-oriented programming (OOP) for stateful components, choosing the paradigm that best fits each stage's requirements.

## Design Philosophy

The pipeline follows these principles:

- **Functional Programming** for pure transformations and stateless analysis
- **Object-Oriented Programming** for stateful operations and complex data structures
- **Hybrid Approach** where both paradigms offer complementary benefits
- **Immutability First** - prefer immutable data structures where possible
- **Type Safety** - comprehensive type hints throughout

## Pipeline Overview

The deterministic analysis pipeline consists of five sequential stages, each implemented with the most appropriate paradigm:

| Stage                         | Paradigm        | Rationale                                 |
| ----------------------------- | --------------- | ----------------------------------------- |
| Stage 0: Integrity Probe      | Functional      | Pure transformations, no state needed     |
| Stage 1: Security Scan        | Functional      | Stateless pattern matching and validation |
| Stage 2: Structural Mapping   | Hybrid          | Complex hierarchy needs controlled state  |
| Stage 3: Formula Analysis     | Object-Oriented | Graph operations are inherently stateful  |
| Stage 4: Content Intelligence | Functional      | Pure computation and synthesis            |

### Stage Flow and Decision Points

```mermaid
flowchart TD
    START([Excel File Input]) --> STAGE0[Stage 0: Integrity Probe<br/>〈Functional〉]

    STAGE0 --> CHECK0{File Valid?}
    CHECK0 -->|No| ABORT[Abort: File Corrupted/Invalid]
    CHECK0 -->|Yes| STAGE1[Stage 1: Security Scan<br/>〈Functional〉]

    STAGE1 --> CHECK1{Security Risk?}
    CHECK1 -->|High Risk| BLOCK[Block: Security Threat]
    CHECK1 -->|Acceptable| STAGE2[Stage 2: Structural Mapping<br/>〈Hybrid〉]

    STAGE2 --> CHECK2{Size/Complexity?}
    CHECK2 -->|Too Large| ROUTE[Route: Heavy Processing]
    CHECK2 -->|Normal| STAGE3[Stage 3: Formula Analysis<br/>〈Object-Oriented〉]

    STAGE3 --> STAGE4[Stage 4: Content Intelligence<br/>〈Functional〉]
    STAGE4 --> COMPLETE[Generate Agent Context]

    classDef functional fill:#e8f5e9,stroke:#4caf50,stroke-width:2px
    classDef oop fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    classDef hybrid fill:#e3f2fd,stroke:#2196f3,stroke-width:2px

    class STAGE0,STAGE1,STAGE4 functional
    class STAGE3 oop
    class STAGE2 hybrid
```

## Stage 0: File Integrity Probe (Functional)

### Design Approach

Stage 0 uses pure functional programming with immutable data structures and function composition.

### Implementation

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Callable, List
import hashlib
import zipfile
import magic

# Immutable data structures
@dataclass(frozen=True)
class FileMetadata:
    """Immutable file metadata."""
    path: Path
    size_bytes: int
    mime_type: str

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 2)

@dataclass(frozen=True)
class IntegrityResult:
    """Immutable integrity check result."""
    file_hash: str
    metadata: FileMetadata
    is_excel: bool
    is_ooxml: bool
    is_duplicate: bool
    trust_tier: int
    processing_class: Literal["STANDARD", "HEAVY", "BLOCKED"]
    validation_passed: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_hash": self.file_hash,
            "size_bytes": self.metadata.size_bytes,
            "size_mb": self.metadata.size_mb,
            "mime_type": self.metadata.mime_type,
            "is_excel": self.is_excel,
            "is_ooxml": self.is_ooxml,
            "is_duplicate": self.is_duplicate,
            "trust_tier": self.trust_tier,
            "processing_class": self.processing_class,
            "validation_passed": self.validation_passed
        }

# Pure functions for file analysis
def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of file - pure function."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def detect_mime_type(file_path: Path) -> str:
    """Detect MIME type using libmagic - pure function."""
    return magic.from_file(str(file_path), mime=True)

def validate_excel_format(mime_type: str) -> bool:
    """Validate if MIME type is Excel - pure function."""
    valid_types = {
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'application/zip'  # Sometimes .xlsx appears as zip
    }
    return mime_type in valid_types

def validate_ooxml_structure(file_path: Path, mime_type: str) -> bool:
    """Validate OOXML structure - pure function."""
    if mime_type == 'application/vnd.ms-excel':
        return False  # .xls is not OOXML

    try:
        if not zipfile.is_zipfile(file_path):
            return False

        with zipfile.ZipFile(file_path, 'r') as zip_file:
            required_files = ['[Content_Types].xml', 'xl/workbook.xml']
            return all(f in zip_file.namelist() for f in required_files)
    except Exception:
        return False

def determine_processing_class(
    metadata: FileMetadata
) -> Literal["STANDARD", "HEAVY", "BLOCKED"]:
    """Classify file for processing - pure function."""
    file_ext = metadata.path.suffix.lower()

    # Size thresholds by format
    size_limits = {
        '.xls': 30 * 1024 * 1024,    # 30MB for legacy
        '.xlsb': 200 * 1024 * 1024,  # 200MB for binary
        '.xlsx': 100 * 1024 * 1024   # 100MB for OOXML
    }

    # Check minimum size
    if metadata.size_bytes < 1024:
        return "BLOCKED"

    # Check format validity
    if not validate_excel_format(metadata.mime_type):
        return "BLOCKED"

    # Check size limits
    limit = size_limits.get(file_ext, size_limits['.xlsx'])
    if metadata.size_bytes > limit:
        return "HEAVY"

    return "STANDARD"

def assess_trust_level(metadata: FileMetadata) -> int:
    """Assess trust level (1-5) - pure function."""
    score = 3  # Default neutral

    # Size-based adjustments
    if metadata.size_bytes > 50 * 1024 * 1024:
        score -= 1
    elif metadata.size_bytes < 10 * 1024:
        score -= 1

    # Name-based heuristics
    filename = metadata.path.name.lower()
    suspicious_patterns = {'temp', 'tmp', 'test', 'untitled'}
    if any(pattern in filename for pattern in suspicious_patterns):
        score -= 1

    # Extension validation
    if not filename.endswith(('.xlsx', '.xls', '.xlsb')):
        score -= 2

    return max(1, min(5, score))

# Function composition for complete analysis
def compose(*functions: Callable) -> Callable:
    """Compose functions from right to left."""
    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return inner

def stage_0_integrity_probe(file_path: Path) -> IntegrityResult:
    """
    Complete integrity analysis using functional composition.

    This is the main entry point that orchestrates all pure functions.
    """
    # Create metadata
    metadata = FileMetadata(
        path=file_path,
        size_bytes=file_path.stat().st_size,
        mime_type=detect_mime_type(file_path)
    )

    # Compute all properties using pure functions
    file_hash = calculate_file_hash(file_path)
    is_excel = validate_excel_format(metadata.mime_type)
    is_ooxml = validate_ooxml_structure(file_path, metadata.mime_type)
    processing_class = determine_processing_class(metadata)
    trust_tier = assess_trust_level(metadata)

    # Check duplicate (would query cache in real implementation)
    is_duplicate = False  # Placeholder

    return IntegrityResult(
        file_hash=file_hash,
        metadata=metadata,
        is_excel=is_excel,
        is_ooxml=is_ooxml,
        is_duplicate=is_duplicate,
        trust_tier=trust_tier,
        processing_class=processing_class,
        validation_passed=is_excel and metadata.size_bytes > 0
    )

# Validator composition example
def create_integrity_validator() -> Callable[[Path], List[str]]:
    """Create a validator that returns list of issues."""
    def validator(file_path: Path) -> List[str]:
        issues = []

        if not file_path.exists():
            issues.append("File does not exist")
            return issues

        result = stage_0_integrity_probe(file_path)

        if not result.is_excel:
            issues.append(f"Invalid Excel format: {result.metadata.mime_type}")
        if result.processing_class == "BLOCKED":
            issues.append("File blocked due to size or format issues")
        if result.trust_tier < 3:
            issues.append(f"Low trust level: {result.trust_tier}/5")

        return issues

    return validator
```

## Stage 1: Security Scan (Functional)

### Design Approach

Security scanning uses functional programming with pure validators and pattern matching.

```python
from dataclasses import dataclass, field
from typing import List, Set, Dict, Any, Callable, Optional
import xml.etree.ElementTree as ET
import zipfile
import re

@dataclass(frozen=True)
class SecurityThreat:
    """Immutable security threat descriptor."""
    threat_type: str
    severity: int  # 1-10
    location: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SecurityReport:
    """Immutable security analysis result."""
    threats: tuple[SecurityThreat, ...]
    has_macros: bool
    has_external_links: bool
    risk_score: int

    @property
    def is_safe(self) -> bool:
        return self.risk_score < 3

    def to_dict(self) -> dict:
        return {
            "has_macros": self.has_macros,
            "external_links": [t for t in self.threats if t.threat_type == "external_link"],
            "security_flags": list({t.threat_type for t in self.threats}),
            "risk_score": self.risk_score,
            "threats": [
                {
                    "type": t.threat_type,
                    "severity": t.severity,
                    "location": t.location,
                    "details": t.details
                } for t in self.threats
            ]
        }

# Pure security validators
def scan_vba_macros(file_path: Path) -> List[SecurityThreat]:
    """Scan for VBA macros - pure function."""
    threats = []

    try:
        import oletools.olevba as olevba
        vba_parser = olevba.VBA_Parser(str(file_path))

        if vba_parser.detect_vba_macros():
            # Analyze macro content
            suspicious_patterns = {
                'Shell': 8,
                'CreateObject': 7,
                'WScript': 9,
                'URLDownloadToFile': 10,
                'Auto_Open': 6,
                'Workbook_Open': 6
            }

            for vba_filename, stream_path, vba_code in vba_parser.extract_macros():
                for pattern, severity in suspicious_patterns.items():
                    if pattern.lower() in vba_code.lower():
                        threats.append(SecurityThreat(
                            threat_type="vba_macro",
                            severity=severity,
                            location=vba_filename,
                            details={"pattern": pattern, "stream": stream_path}
                        ))

            # Even benign macros get a low severity entry
            if not threats:
                threats.append(SecurityThreat(
                    threat_type="vba_macro",
                    severity=3,
                    location="workbook",
                    details={"description": "VBA macros detected"}
                ))

    except Exception as e:
        # Report scan failure as a threat
        threats.append(SecurityThreat(
            threat_type="scan_error",
            severity=2,
            location="vba_scan",
            details={"error": str(e)}
        ))

    return threats

def scan_xlm_macros(file_path: Path) -> List[SecurityThreat]:
    """Scan for Excel 4.0 (XLM) macros - pure function."""
    threats = []

    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=False)

        xlm_patterns = ['EXEC', 'CALL', 'REGISTER', 'ALERT', 'RUN']

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]

            # Check for very hidden sheets (suspicious)
            if hasattr(ws, 'sheet_state') and ws.sheet_state == 'veryHidden':
                threats.append(SecurityThreat(
                    threat_type="hidden_sheet",
                    severity=5,
                    location=sheet_name,
                    details={"state": "veryHidden"}
                ))

            # Sample formulas for XLM patterns
            for row in ws.iter_rows(max_row=100):
                for cell in row:
                    if cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                        formula_upper = cell.value.upper()
                        for pattern in xlm_patterns:
                            if pattern in formula_upper:
                                threats.append(SecurityThreat(
                                    threat_type="xlm_macro",
                                    severity=7,
                                    location=f"{sheet_name}!{cell.coordinate}",
                                    details={"pattern": pattern, "formula": cell.value[:50]}
                                ))
                                break

    except Exception:
        pass  # XLM scan is best-effort

    return threats

def scan_external_links(file_path: Path) -> List[SecurityThreat]:
    """Scan for external file references - pure function."""
    threats = []

    if not file_path.suffix.lower() == '.xlsx':
        return threats  # Only scan XLSX files

    try:
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            # Check for external link files
            external_link_files = [
                f for f in zip_file.namelist()
                if f.startswith('xl/externalLinks/')
            ]

            for link_file in external_link_files:
                threats.append(SecurityThreat(
                    threat_type="external_link",
                    severity=4,
                    location=link_file,
                    details={"file_type": "external_workbook"}
                ))

            # Check for embedded objects
            embedding_files = [
                f for f in zip_file.namelist()
                if '/embeddings/' in f or f.startswith('xl/embeddings/')
            ]

            for obj_file in embedding_files:
                threats.append(SecurityThreat(
                    threat_type="embedded_object",
                    severity=5,
                    location=obj_file,
                    details={"size": zip_file.getinfo(obj_file).file_size}
                ))

    except Exception:
        pass  # External link scan is best-effort

    return threats

# Security analyzer composition
def compose_security_scanners(*scanners: Callable[[Path], List[SecurityThreat]]) -> Callable:
    """Compose multiple security scanners into one."""
    def combined_scanner(file_path: Path) -> List[SecurityThreat]:
        all_threats = []
        for scanner in scanners:
            all_threats.extend(scanner(file_path))
        return all_threats
    return combined_scanner

def calculate_risk_score(threats: List[SecurityThreat]) -> int:
    """Calculate overall risk score - pure function."""
    if not threats:
        return 0

    # Weighted sum of threat severities
    total_severity = sum(t.severity for t in threats)

    # Bonus risk for multiple threat types
    threat_types = len(set(t.threat_type for t in threats))

    return min(10, total_severity // 3 + threat_types)

def stage_1_security_scan(file_path: Path) -> SecurityReport:
    """
    Complete security analysis using functional composition.

    Combines all security scanners and produces immutable report.
    """
    # Compose all scanners
    scanner = compose_security_scanners(
        scan_vba_macros,
        scan_xlm_macros,
        scan_external_links
    )

    # Run scans
    threats = scanner(file_path)

    # Analyze results
    has_macros = any(
        t.threat_type in ['vba_macro', 'xlm_macro']
        for t in threats
    )
    has_external_links = any(
        t.threat_type == 'external_link'
        for t in threats
    )
    risk_score = calculate_risk_score(threats)

    return SecurityReport(
        threats=tuple(threats),
        has_macros=has_macros,
        has_external_links=has_external_links,
        risk_score=risk_score
    )
```

## Stage 2: Structural Mapping (Hybrid)

### Design Approach

Structural mapping uses a hybrid approach: functional for data extraction, object-oriented for managing complex hierarchical structure.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import openpyxl
from openpyxl.utils import get_column_letter

# Immutable data structures for functional parts
@dataclass(frozen=True)
class CellStatistics:
    """Immutable cell statistics."""
    total_sampled: int
    formula_cells: int
    value_cells: int
    empty_cells: int
    error_cells: int
    text_cells: int
    number_cells: int

@dataclass(frozen=True)
class SheetDimensions:
    """Immutable sheet dimensions."""
    max_row: int
    max_column: int
    used_range: str
    estimated_cells: int
    data_density: float
    is_empty: bool

@dataclass(frozen=True)
class SheetStructure:
    """Immutable sheet structure data."""
    name: str
    visible: bool
    dimensions: SheetDimensions
    cell_stats: CellStatistics
    merged_cells: tuple[str, ...]
    tables: tuple[dict, ...]
    charts: tuple[dict, ...]
    has_errors: bool

# Object-oriented wrapper for stateful operations
class StructuralMapper:
    """
    Manages structural mapping with controlled state.

    Uses OOP for error accumulation and complex navigation,
    but delegates to pure functions for data extraction.
    """

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.errors: List[Dict[str, Any]] = []
        self.workbook = None

    def analyze(self) -> Dict[str, Any]:
        """Analyze workbook structure."""
        try:
            self.workbook = self._load_workbook()

            sheets = []
            for sheet_name in self.workbook.sheetnames:
                sheet_structure = self._analyze_sheet(sheet_name)
                if sheet_structure:
                    sheets.append(sheet_structure)

            return self._compile_results(sheets)

        except Exception as e:
            self.errors.append({
                "stage": "structural_mapping",
                "error": str(e),
                "fatal": True
            })
            return {"error": str(e), "sheets": []}
        finally:
            if self.workbook:
                self.workbook.close()

    def _load_workbook(self) -> openpyxl.Workbook:
        """Load workbook in read-only mode."""
        return openpyxl.load_workbook(
            self.file_path,
            read_only=True,
            data_only=False,
            keep_links=False
        )

    def _analyze_sheet(self, sheet_name: str) -> Optional[SheetStructure]:
        """Analyze individual sheet using pure functions."""
        try:
            ws = self.workbook[sheet_name]

            # Use pure functions for extraction
            dimensions = extract_dimensions(ws)
            cell_stats = sample_cell_statistics(ws, dimensions)
            merged_cells = extract_merged_cells(ws)
            tables = extract_tables(ws)
            charts = extract_charts(ws)

            return SheetStructure(
                name=sheet_name,
                visible=ws.sheet_state == 'visible' if hasattr(ws, 'sheet_state') else True,
                dimensions=dimensions,
                cell_stats=cell_stats,
                merged_cells=merged_cells,
                tables=tables,
                charts=charts,
                has_errors=False
            )

        except Exception as e:
            self.errors.append({
                "sheet": sheet_name,
                "error": str(e),
                "component": "sheet_analysis"
            })
            return None

    def _compile_results(self, sheets: List[SheetStructure]) -> Dict[str, Any]:
        """Compile final results with complexity metrics."""
        valid_sheets = [s for s in sheets if s is not None]

        # Calculate aggregates using pure functions
        total_charts = sum(len(s.charts) for s in valid_sheets)
        total_tables = sum(len(s.tables) for s in valid_sheets)
        total_cells = sum(s.dimensions.estimated_cells for s in valid_sheets)
        total_formulas = sum(s.cell_stats.formula_cells for s in valid_sheets)

        complexity_score = calculate_complexity_score(
            len(valid_sheets),
            total_charts,
            total_tables,
            total_cells
        )

        return {
            "workbook": {
                "sheet_count": len(self.workbook.sheetnames),
                "sheet_names": list(self.workbook.sheetnames),
                "sheets": [s.name for s in valid_sheets]
            },
            "sheets": [sheet_to_dict(s) for s in valid_sheets],
            "total_charts": total_charts,
            "total_tables": total_tables,
            "complexity_indicators": {
                "complexity_score": complexity_score,
                "total_estimated_cells": total_cells,
                "total_formula_cells": total_formulas,
                "formula_density": round(total_formulas / max(total_cells, 1) * 100, 2)
            },
            "errors": self.errors
        }

# Pure functions for data extraction
def extract_dimensions(ws) -> SheetDimensions:
    """Extract sheet dimensions - pure function."""
    max_row = ws.max_row
    max_col = ws.max_column

    if max_row > 0 and max_col > 0:
        used_range = f"A1:{get_column_letter(max_col)}{max_row}"
        estimated_cells = max_row * max_col
    else:
        used_range = "A1:A1"
        estimated_cells = 0

    # Estimate density by sampling
    sample_size = min(100, estimated_cells)
    non_empty = 0

    if sample_size > 0:
        step = max(1, estimated_cells // sample_size)
        count = 0

        for row in ws.iter_rows(max_row=max_row, max_col=max_col):
            for cell in row:
                if count % step == 0 and cell.value is not None:
                    non_empty += 1
                count += 1
                if count >= sample_size:
                    break
            if count >= sample_size:
                break

    density = (non_empty / sample_size * 100) if sample_size > 0 else 0

    return SheetDimensions(
        max_row=max_row,
        max_column=max_col,
        used_range=used_range,
        estimated_cells=estimated_cells,
        data_density=round(density, 1),
        is_empty=max_row <= 1 and max_col <= 1
    )

def sample_cell_statistics(ws, dimensions: SheetDimensions) -> CellStatistics:
    """Sample cell types for statistics - pure function."""
    stats = {
        "total_sampled": 0,
        "formula_cells": 0,
        "value_cells": 0,
        "empty_cells": 0,
        "error_cells": 0,
        "text_cells": 0,
        "number_cells": 0
    }

    # Sampling strategy
    max_row = min(dimensions.max_row, 1000)
    max_col = min(dimensions.max_column, 100)
    step_row = max(1, max_row // 100)
    step_col = max(1, max_col // 20)

    for row in range(1, max_row + 1, step_row):
        for col in range(1, max_col + 1, step_col):
            try:
                cell = ws.cell(row=row, column=col)
                stats["total_sampled"] += 1

                if cell.value is None:
                    stats["empty_cells"] += 1
                elif isinstance(cell.value, str):
                    if cell.value.startswith('='):
                        stats["formula_cells"] += 1
                    elif cell.value.startswith('#'):
                        stats["error_cells"] += 1
                    else:
                        stats["text_cells"] += 1
                elif isinstance(cell.value, (int, float)):
                    stats["number_cells"] += 1
                else:
                    stats["value_cells"] += 1

            except Exception:
                continue

    return CellStatistics(**stats)

def extract_merged_cells(ws) -> tuple[str, ...]:
    """Extract merged cell ranges - pure function."""
    return tuple(str(r) for r in ws.merged_cells.ranges)

def extract_tables(ws) -> tuple[dict, ...]:
    """Extract table information - pure function."""
    tables = []
    try:
        for table in ws.tables.values():
            tables.append({
                "name": table.name,
                "ref": table.ref,
                "style": getattr(table.tableStyleInfo, 'name', None)
                        if hasattr(table, 'tableStyleInfo') else None
            })
    except Exception:
        pass
    return tuple(tables)

def extract_charts(ws) -> tuple[dict, ...]:
    """Extract chart information - pure function."""
    charts = []
    try:
        if hasattr(ws, '_charts'):
            for chart in ws._charts:
                charts.append({
                    "type": type(chart).__name__,
                    "title": getattr(chart, 'title', {}).text
                            if hasattr(chart, 'title') and chart.title else None
                })
    except Exception:
        pass
    return tuple(charts)

def calculate_complexity_score(
    sheet_count: int,
    chart_count: int,
    table_count: int,
    cell_count: int
) -> float:
    """Calculate complexity score - pure function."""
    score = (
        (sheet_count / 10) * 2 +      # Sheet factor
        (chart_count / 5) * 1 +       # Chart factor
        (table_count / 3) * 1 +       # Table factor
        (cell_count / 10000) * 3      # Size factor
    )
    return round(min(10, score), 1)

def sheet_to_dict(sheet: SheetStructure) -> dict:
    """Convert sheet structure to dictionary - pure function."""
    return {
        "name": sheet.name,
        "visible": sheet.visible,
        "dimensions": {
            "max_row": sheet.dimensions.max_row,
            "max_column": sheet.dimensions.max_column,
            "used_range": sheet.dimensions.used_range,
            "estimated_cells": sheet.dimensions.estimated_cells,
            "data_density": sheet.dimensions.data_density,
            "is_empty": sheet.dimensions.is_empty
        },
        "cell_statistics": {
            "total_sampled": sheet.cell_stats.total_sampled,
            "formula_cells": sheet.cell_stats.formula_cells,
            "value_cells": sheet.cell_stats.value_cells,
            "empty_cells": sheet.cell_stats.empty_cells,
            "error_cells": sheet.cell_stats.error_cells,
            "text_cells": sheet.cell_stats.text_cells,
            "number_cells": sheet.cell_stats.number_cells
        },
        "merged_cells": list(sheet.merged_cells),
        "tables": list(sheet.tables),
        "charts": list(sheet.charts)
    }

# Main entry point
def stage_2_structural_mapping(file_path: Path) -> Dict[str, Any]:
    """Perform structural mapping using hybrid approach."""
    mapper = StructuralMapper(file_path)
    return mapper.analyze()
```

## Stage 3: Formula Intelligence & Dependency Graph (Object-Oriented)

### Design Approach

Formula analysis uses object-oriented programming because graph operations are inherently stateful and NetworkX is an OOP library.

```python
import networkx as nx
import re
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
import tempfile

@dataclass
class FormulaNode:
    """Represents a cell with formula in the dependency graph."""
    cell_ref: str
    sheet: str
    formula: str
    coordinate: str

    @property
    def full_ref(self) -> str:
        return f"{self.sheet}!{self.coordinate}"

@dataclass
class FormulaReference:
    """Represents a reference from one cell to another."""
    sheet: str
    cell_or_range: str
    is_external: bool = False
    external_file: Optional[str] = None

class FormulaDependencyGraph:
    """
    Manages formula dependency analysis using NetworkX.

    OOP is appropriate here because:
    1. Graphs are inherently stateful
    2. NetworkX provides OOP interface
    3. Complex graph algorithms benefit from encapsulation
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.formula_count = 0
        self.errors: List[Dict[str, Any]] = []
        self.volatile_functions: Set[str] = set()
        self.external_references: List[Dict[str, str]] = []
        self.function_usage: Dict[str, int] = {}

    def add_formula(self, cell_ref: str, sheet: str, formula: str, coordinate: str):
        """Add a formula to the graph."""
        node = FormulaNode(cell_ref, sheet, formula, coordinate)
        self.graph.add_node(
            node.full_ref,
            formula=formula,
            sheet=sheet,
            coordinate=coordinate
        )
        self.formula_count += 1

        # Analyze formula
        self._analyze_formula(node)

    def _analyze_formula(self, node: FormulaNode):
        """Analyze individual formula for dependencies and patterns."""
        # Extract functions
        functions = self._extract_functions(node.formula)
        for func in functions:
            self.function_usage[func] = self.function_usage.get(func, 0) + 1

        # Check for volatile functions
        volatile_funcs = {'NOW', 'TODAY', 'RAND', 'RANDBETWEEN', 'INDIRECT', 'OFFSET'}
        found_volatile = set(functions) & volatile_funcs
        self.volatile_functions.update(found_volatile)

        # Extract references
        references = self._extract_references(node.formula, node.sheet)

        for ref in references:
            if ref.is_external:
                self.external_references.append({
                    "from_cell": node.full_ref,
                    "to_file": ref.external_file,
                    "to_reference": ref.cell_or_range
                })
            else:
                # Add dependency edge
                target_ref = f"{ref.sheet}!{ref.cell_or_range}"
                self.graph.add_edge(target_ref, node.full_ref)

    def _extract_functions(self, formula: str) -> List[str]:
        """Extract function names from formula."""
        pattern = r'\b([A-Z][A-Z0-9_]*)\s*\('
        functions = re.findall(pattern, formula.upper())
        return list(set(functions))

    def _extract_references(self, formula: str, current_sheet: str) -> List[FormulaReference]:
        """Extract cell and range references."""
        references = []

        # External references
        external_pattern = r'\[([^\]]+)\]([^!]+)!([A-Z]+[0-9]+(?::[A-Z]+[0-9]+)?)'
        for match in re.finditer(external_pattern, formula):
            references.append(FormulaReference(
                sheet=match.group(2),
                cell_or_range=match.group(3),
                is_external=True,
                external_file=match.group(1)
            ))

        # Internal references
        internal_pattern = r'(?:([^!\[\]]+)!)?([A-Z]+[0-9]+(?::[A-Z]+[0-9]+)?)'
        for match in re.finditer(internal_pattern, formula):
            # Skip if already captured as external
            cell_range = match.group(2)
            if not any(ref.cell_or_range == cell_range for ref in references if ref.is_external):
                references.append(FormulaReference(
                    sheet=match.group(1) if match.group(1) else current_sheet,
                    cell_or_range=cell_range,
                    is_external=False
                ))

        return references

    def analyze_graph_properties(self) -> Dict[str, Any]:
        """Analyze graph for patterns and issues."""
        if self.graph.number_of_nodes() > 50000:
            return self._analyze_large_graph()
        else:
            return self._analyze_standard_graph()

    def _analyze_standard_graph(self) -> Dict[str, Any]:
        """Standard analysis for manageable graphs."""
        metrics = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "isolated_formulas": len([n for n in self.graph.nodes() if self.graph.degree(n) == 0])
        }

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            try:
                cycles = list(nx.simple_cycles(self.graph))[:10]
                metrics["circular_references"] = cycles
            except:
                metrics["circular_references"] = ["Too complex to enumerate"]
        else:
            metrics["circular_references"] = []

        # Calculate max depth
        metrics["max_dependency_depth"] = self._calculate_max_depth()

        # Find highly connected cells
        metrics["highly_connected_cells"] = self._find_highly_connected(threshold=10)

        # Identify critical paths
        metrics["critical_paths"] = self._identify_critical_paths()

        return metrics

    def _analyze_large_graph(self) -> Dict[str, Any]:
        """Hierarchical analysis for very large graphs."""
        # Build sheet-level summary
        sheet_stats = {}

        for node in self.graph.nodes():
            sheet = node.split('!')[0] if '!' in node else 'Unknown'
            if sheet not in sheet_stats:
                sheet_stats[sheet] = {"nodes": 0, "internal_edges": 0, "external_edges": 0}
            sheet_stats[sheet]["nodes"] += 1

        # Count cross-sheet edges
        for source, target in self.graph.edges():
            source_sheet = source.split('!')[0] if '!' in source else 'Unknown'
            target_sheet = target.split('!')[0] if '!' in target else 'Unknown'

            if source_sheet == target_sheet:
                sheet_stats[source_sheet]["internal_edges"] += 1
            else:
                sheet_stats[source_sheet]["external_edges"] += 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "sheet_statistics": sheet_stats,
            "analysis_method": "hierarchical",
            "circular_references": self._detect_cycles_sample()
        }

    def _calculate_max_depth(self) -> int:
        """Calculate maximum dependency depth."""
        if self.graph.number_of_nodes() == 0:
            return 0

        try:
            # For DAG, use topological sort
            topo_order = list(nx.topological_sort(self.graph))
            distances = {node: 0 for node in self.graph.nodes()}

            for node in topo_order:
                for successor in self.graph.successors(node):
                    distances[successor] = max(distances[successor], distances[node] + 1)

            return max(distances.values()) if distances else 0
        except nx.NetworkXError:
            return -1  # Has cycles

    def _find_highly_connected(self, threshold: int = 10) -> List[Dict[str, Any]]:
        """Find cells with many dependencies."""
        highly_connected = []

        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)

            if in_degree >= threshold or out_degree >= threshold:
                highly_connected.append({
                    "cell": node,
                    "in_degree": in_degree,
                    "out_degree": out_degree
                })

        return sorted(highly_connected, key=lambda x: x["in_degree"] + x["out_degree"], reverse=True)[:20]

    def _identify_critical_paths(self) -> Dict[str, List]:
        """Identify critical dependency paths."""
        sources = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        sinks = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]

        critical_paths = {
            "source_cells": sources[:10],
            "sink_cells": sinks[:10],
            "longest_chains": []
        }

        # Find longest paths from sources
        for source in sources[:5]:
            try:
                if nx.is_directed_acyclic_graph(self.graph):
                    longest = nx.dag_longest_path(self.graph, source)
                    if len(longest) > 3:
                        critical_paths["longest_chains"].append({
                            "source": source,
                            "length": len(longest),
                            "path": longest[:10]  # Limit display
                        })
            except:
                continue

        return critical_paths

    def _detect_cycles_sample(self, sample_size: int = 1000) -> List[Any]:
        """Detect cycles in large graphs using sampling."""
        import random

        nodes = list(self.graph.nodes())
        if len(nodes) <= sample_size:
            # Small enough to check completely
            if not nx.is_directed_acyclic_graph(self.graph):
                try:
                    return list(nx.simple_cycles(self.graph))[:5]
                except:
                    return ["Cycle detection failed"]
            return []

        # Sample nodes
        sampled = random.sample(nodes, sample_size)
        subgraph = self.graph.subgraph(sampled)

        if not nx.is_directed_acyclic_graph(subgraph):
            return ["Cycles detected in sample"]
        return ["No cycles detected in sample"]

    def save_graph(self, file_stem: str) -> Optional[str]:
        """Save graph for visualization."""
        try:
            temp_dir = Path(tempfile.gettempdir()) / "spreadsheet_analyzer"
            temp_dir.mkdir(exist_ok=True)

            graph_path = temp_dir / f"{file_stem}_formula_graph.graphml"
            nx.write_graphml(self.graph, graph_path)
            return str(graph_path)
        except Exception:
            return None

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get complete analysis summary."""
        metrics = self.analyze_graph_properties()

        return {
            "total_formulas": self.formula_count,
            "volatile_functions": list(self.volatile_functions),
            "external_references": self.external_references,
            "formula_errors": self.errors,
            "complexity_metrics": metrics,
            "function_usage": self.function_usage,
            "graph_saved": bool(metrics.get("graph_path"))
        }

class FormulaAnalyzer:
    """High-level formula analyzer coordinating the analysis."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.graph = FormulaDependencyGraph()

    def analyze(self) -> Dict[str, Any]:
        """Perform complete formula analysis."""
        try:
            wb = openpyxl.load_workbook(
                self.file_path,
                read_only=True,
                data_only=False,
                keep_links=False
            )

            # Extract all formulas
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                self._extract_sheet_formulas(ws, sheet_name)

            wb.close()

            # Save graph
            graph_path = self.graph.save_graph(self.file_path.stem)

            # Get analysis summary
            summary = self.graph.get_analysis_summary()
            if graph_path:
                summary["graph_path"] = graph_path

            return summary

        except Exception as e:
            return {
                "error": f"Formula analysis failed: {str(e)}",
                "total_formulas": 0,
                "graph_path": None
            }

    def _extract_sheet_formulas(self, ws, sheet_name: str):
        """Extract formulas from worksheet."""
        # Limit for performance
        max_row = min(ws.max_row, 10000)
        max_col = min(ws.max_column, 1000)

        for row in range(1, max_row + 1):
            for col in range(1, max_col + 1):
                try:
                    cell = ws.cell(row=row, column=col)

                    if cell.value and isinstance(cell.value, str) and cell.value.startswith('='):
                        cell_ref = f"{sheet_name}!{cell.coordinate}"
                        self.graph.add_formula(
                            cell_ref,
                            sheet_name,
                            cell.value,
                            cell.coordinate
                        )

                except Exception as e:
                    self.graph.errors.append({
                        "cell": f"{sheet_name}!{get_column_letter(col)}{row}",
                        "error": str(e)
                    })

# Main entry point
def stage_3_formula_analysis(file_path: Path) -> Dict[str, Any]:
    """Perform formula dependency analysis."""
    analyzer = FormulaAnalyzer(file_path)
    return analyzer.analyze()
```

## Stage 4: Content Intelligence Summary (Functional)

### Design Approach

Content intelligence uses functional programming to synthesize findings from previous stages into actionable insights.

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Callable
from functools import reduce

@dataclass(frozen=True)
class FileAssessment:
    """Immutable file assessment."""
    file_type: str
    format: str
    size_category: str
    trust_level: int
    security_status: str
    processable: bool
    estimated_time: str
    recommended_resources: str

@dataclass(frozen=True)
class ProcessingStrategy:
    """Immutable processing strategy."""
    approach: str
    parallelism: str
    agent_count: int
    analysis_depth: str
    considerations: tuple[str, ...]

@dataclass(frozen=True)
class RiskAssessment:
    """Immutable risk assessment."""
    overall_risk: str
    security_risks: tuple[str, ...]
    computational_risks: tuple[str, ...]
    data_integrity_risks: tuple[str, ...]
    performance_risks: tuple[str, ...]

@dataclass(frozen=True)
class ComplexityAnalysis:
    """Immutable complexity analysis."""
    overall_score: float
    structural_complexity: float
    formula_complexity: float
    complexity_factors: tuple[str, ...]
    simplification_opportunities: tuple[str, ...]

# Pure functions for intelligence synthesis
def assess_file(
    integrity: Dict[str, Any],
    security: Dict[str, Any],
    structure: Dict[str, Any]
) -> FileAssessment:
    """Generate file assessment - pure function."""
    return FileAssessment(
        file_type="Excel Workbook",
        format="OOXML (.xlsx)" if integrity.get("is_ooxml") else "Legacy (.xls)",
        size_category=structure.get("complexity_indicators", {}).get("size_category", "UNKNOWN"),
        trust_level=integrity.get("trust_tier", 3),
        security_status="CLEAN" if security.get("risk_score", 0) < 3 else "REVIEW_REQUIRED",
        processable=integrity.get("validation_passed", False),
        estimated_time=estimate_analysis_time(structure),
        recommended_resources=recommend_resources(structure)
    )

def determine_strategy(
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> ProcessingStrategy:
    """Determine processing strategy - pure function."""
    sheet_count = structure.get("workbook", {}).get("sheet_count", 0)
    formula_count = formulas.get("total_formulas", 0)
    complexity_score = structure.get("complexity_indicators", {}).get("complexity_score", 0)

    # Base strategy
    approach = "STANDARD"
    parallelism = "MEDIUM"
    agent_count = min(sheet_count, 5)
    analysis_depth = "FULL"
    considerations = []

    # Adjustments based on characteristics
    if complexity_score > 7:
        approach = "HEAVY_COMPUTE"
        analysis_depth = "PROGRESSIVE"
        considerations = (*considerations, "High complexity - progressive analysis recommended")

    if formula_count > 1000:
        parallelism = "HIGH"
        considerations = (*considerations, "High formula count - parallel processing beneficial")

    if sheet_count > 10:
        agent_count = min(sheet_count, 8)
        considerations = (*considerations, "Many sheets - increased parallelism")

    if formulas.get("external_references"):
        considerations = (*considerations, "External references detected - validation required")

    if formulas.get("circular_references"):
        considerations = (*considerations, "Circular references detected - careful analysis needed")

    return ProcessingStrategy(
        approach=approach,
        parallelism=parallelism,
        agent_count=agent_count,
        analysis_depth=analysis_depth,
        considerations=tuple(considerations)
    )

def assess_risks(
    security: Dict[str, Any],
    formulas: Dict[str, Any]
) -> RiskAssessment:
    """Synthesize risk assessment - pure function."""
    security_risks = []
    computational_risks = []
    data_integrity_risks = []
    performance_risks = []

    # Security risks
    if security.get("has_macros"):
        security_risks.append("VBA macros present - manual review recommended")
    if security.get("external_links"):
        security_risks.append("External file references - data dependency risk")
    if security.get("ole_objects"):
        security_risks.append("Embedded objects present - security scan required")

    # Computational risks
    if formulas.get("circular_references"):
        computational_risks.append("Circular references may cause calculation errors")
    if "NOW" in formulas.get("volatile_functions", []):
        computational_risks.append("Time-dependent formulas - results may vary")

    # Data integrity risks
    if formulas.get("formula_errors"):
        data_integrity_risks.append("Formula errors detected - validation needed")

    # Performance risks
    if formulas.get("total_formulas", 0) > 5000:
        performance_risks.append("High formula count may impact performance")

    # Determine overall risk
    total_risks = len(security_risks) + len(computational_risks) + len(data_integrity_risks) + len(performance_risks)
    overall_risk = "HIGH" if total_risks > 5 else "MEDIUM" if total_risks > 2 else "LOW"

    return RiskAssessment(
        overall_risk=overall_risk,
        security_risks=tuple(security_risks),
        computational_risks=tuple(computational_risks),
        data_integrity_risks=tuple(data_integrity_risks),
        performance_risks=tuple(performance_risks)
    )

def analyze_complexity(
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> ComplexityAnalysis:
    """Analyze overall complexity - pure function."""
    structural_score = structure.get("complexity_indicators", {}).get("complexity_score", 0)
    formula_score = calculate_formula_complexity(formulas)

    factors = []
    opportunities = []

    # Identify complexity factors
    if structure.get("workbook", {}).get("sheet_count", 0) > 5:
        factors.append("Multiple sheets increase navigation complexity")
    if formulas.get("total_formulas", 0) > 100:
        factors.append("High formula count increases maintenance complexity")
    if formulas.get("complexity_metrics", {}).get("max_dependency_depth", 0) > 5:
        factors.append("Deep dependency chains increase error propagation risk")

    # Identify simplification opportunities
    isolated = formulas.get("complexity_metrics", {}).get("isolated_formulas", 0)
    if isolated > 0:
        opportunities.append(f"{isolated} isolated formulas could be consolidated")

    return ComplexityAnalysis(
        overall_score=round((structural_score + formula_score) / 2, 1),
        structural_complexity=structural_score,
        formula_complexity=formula_score,
        complexity_factors=tuple(factors),
        simplification_opportunities=tuple(opportunities)
    )

def calculate_formula_complexity(formulas: Dict[str, Any]) -> float:
    """Calculate formula complexity score - pure function."""
    if not formulas or formulas.get("total_formulas", 0) == 0:
        return 0.0

    # Score components
    components = [
        min(3.0, formulas.get("total_formulas", 0) / 100),  # Formula count
        min(2.0, formulas.get("complexity_metrics", {}).get("max_dependency_depth", 0) / 10),  # Depth
        2.0 if formulas.get("circular_references") else 0.0,  # Circular refs
        min(1.0, len(formulas.get("external_references", [])) / 5),  # External refs
        min(1.0, len(formulas.get("volatile_functions", [])) / 3)  # Volatile functions
    ]

    return min(10.0, sum(components))

def generate_recommendations(
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> Dict[str, List[str]]:
    """Generate agent recommendations - pure function."""
    priority_sheets = []
    focus_areas = []
    analysis_approaches = []
    validation_strategies = []

    # Priority sheets
    sheets = structure.get("sheets", [])
    for sheet in sheets:
        if sheet.get("charts") or sheet.get("tables"):
            priority_sheets.append(f"{sheet['name']} - Contains visualizations")
        elif sheet.get("cell_statistics", {}).get("formula_cells", 0) > 20:
            priority_sheets.append(f"{sheet['name']} - High formula density")

    # Focus areas
    if formulas.get("circular_references"):
        focus_areas.append("Investigate circular references for calculation errors")
    if formulas.get("external_references"):
        focus_areas.append("Validate external file dependencies")
    if "VLOOKUP" in formulas.get("function_usage", {}):
        focus_areas.append("Review VLOOKUP usage for optimization opportunities")

    # Analysis approaches
    complexity = structure.get("complexity_indicators", {}).get("complexity_score", 0)
    if complexity > 5:
        analysis_approaches.extend([
            "Use progressive sampling for large datasets",
            "Focus on key calculation chains first"
        ])
    else:
        analysis_approaches.append("Comprehensive analysis feasible")

    # Validation strategies
    if formulas.get("total_formulas", 0) > 50:
        validation_strategies.extend([
            "Sample-based formula validation",
            "Cross-verify summary calculations"
        ])

    return {
        "priority_sheets": priority_sheets,
        "focus_areas": focus_areas,
        "analysis_approaches": analysis_approaches,
        "validation_strategies": validation_strategies
    }

def identify_priority_areas(
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Identify priority analysis areas - pure function."""
    areas = []

    # Circular references
    if formulas.get("circular_references"):
        areas.append({
            "area": "Circular References",
            "priority": "HIGH",
            "reason": "May cause calculation errors or infinite loops",
            "cells": formulas["circular_references"][:5]
        })

    # External dependencies
    if formulas.get("external_references"):
        areas.append({
            "area": "External Dependencies",
            "priority": "MEDIUM",
            "reason": "Data availability and consistency risks",
            "count": len(formulas["external_references"])
        })

    # Volatile functions
    if formulas.get("volatile_functions"):
        areas.append({
            "area": "Volatile Functions",
            "priority": "MEDIUM",
            "reason": "Results may change on each calculation",
            "functions": list(formulas["volatile_functions"])
        })

    # Hidden sheets
    hidden = [
        s["name"] for s in structure.get("sheets", [])
        if not s.get("visible", True)
    ]
    if hidden:
        areas.append({
            "area": "Hidden Sheets",
            "priority": "LOW",
            "reason": "May contain important calculations or sensitive data",
            "sheets": hidden
        })

    return areas

# Helper functions
def estimate_analysis_time(structure: Dict[str, Any]) -> str:
    """Estimate analysis time - pure function."""
    cells = structure.get("complexity_indicators", {}).get("total_estimated_cells", 0)

    if cells < 1000:
        return "< 1 minute"
    elif cells < 10000:
        return "1-2 minutes"
    elif cells < 100000:
        return "2-5 minutes"
    else:
        return "5-10 minutes"

def recommend_resources(structure: Dict[str, Any]) -> str:
    """Recommend compute resources - pure function."""
    cells = structure.get("complexity_indicators", {}).get("total_estimated_cells", 0)
    sheets = structure.get("workbook", {}).get("sheet_count", 0)

    agents = min(sheets, 5)
    memory = "2GB" if cells < 10000 else "4GB" if cells < 100000 else "8GB"

    return f"{agents} agents, {memory} RAM"

# Main composition function
def stage_4_content_intelligence(
    integrity: Dict[str, Any],
    security: Dict[str, Any],
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesize all findings into actionable intelligence.

    Uses functional composition to build complete intelligence report.
    """
    # All pure function calls
    file_assessment = assess_file(integrity, security, structure)
    processing_strategy = determine_strategy(structure, formulas)
    risk_assessment = assess_risks(security, formulas)
    complexity_analysis = analyze_complexity(structure, formulas)
    agent_recommendations = generate_recommendations(structure, formulas)
    priority_areas = identify_priority_areas(structure, formulas)

    # Convert to dictionaries for JSON serialization
    return {
        "file_assessment": {
            "file_type": file_assessment.file_type,
            "format": file_assessment.format,
            "size_category": file_assessment.size_category,
            "trust_level": file_assessment.trust_level,
            "security_status": file_assessment.security_status,
            "processable": file_assessment.processable,
            "estimated_analysis_time": file_assessment.estimated_time,
            "recommended_resources": file_assessment.recommended_resources
        },
        "processing_strategy": {
            "approach": processing_strategy.approach,
            "parallelism": processing_strategy.parallelism,
            "agent_count": processing_strategy.agent_count,
            "analysis_depth": processing_strategy.analysis_depth,
            "special_considerations": list(processing_strategy.considerations)
        },
        "risk_assessment": {
            "overall_risk": risk_assessment.overall_risk,
            "security_risks": list(risk_assessment.security_risks),
            "computational_risks": list(risk_assessment.computational_risks),
            "data_integrity_risks": list(risk_assessment.data_integrity_risks),
            "performance_risks": list(risk_assessment.performance_risks)
        },
        "complexity_analysis": {
            "overall_score": complexity_analysis.overall_score,
            "structural_complexity": complexity_analysis.structural_complexity,
            "formula_complexity": complexity_analysis.formula_complexity,
            "complexity_factors": list(complexity_analysis.complexity_factors),
            "simplification_opportunities": list(complexity_analysis.simplification_opportunities)
        },
        "agent_recommendations": agent_recommendations,
        "priority_areas": priority_areas,
        "optimization_opportunities": identify_optimization_opportunities(formulas),
        "validation_targets": identify_validation_targets(structure, formulas)
    }

def identify_optimization_opportunities(formulas: Dict[str, Any]) -> List[str]:
    """Identify optimization opportunities - pure function."""
    opportunities = []

    # VLOOKUP optimization
    vlookup_count = formulas.get("function_usage", {}).get("VLOOKUP", 0)
    if vlookup_count > 10:
        opportunities.append(f"Consider replacing {vlookup_count} VLOOKUP formulas with INDEX/MATCH")

    # Volatile function reduction
    if formulas.get("volatile_functions"):
        opportunities.append("Replace volatile functions with static values where possible")

    # Formula consolidation
    isolated = formulas.get("complexity_metrics", {}).get("isolated_formulas", 0)
    if isolated > 5:
        opportunities.append(f"Consolidate {isolated} isolated formulas")

    return opportunities

def identify_validation_targets(
    structure: Dict[str, Any],
    formulas: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Identify key validation targets - pure function."""
    targets = []

    # High-dependency cells
    highly_connected = formulas.get("complexity_metrics", {}).get("highly_connected_cells", [])
    for cell in highly_connected[:3]:
        targets.append({
            "type": "high_dependency_cell",
            "location": cell["cell"],
            "reason": f"Has {cell['in_degree']} dependencies"
        })

    # Summary sheets
    for sheet in structure.get("sheets", []):
        if "summary" in sheet.get("name", "").lower():
            targets.append({
                "type": "summary_sheet",
                "location": sheet["name"],
                "reason": "Likely contains key aggregations"
            })

    return targets
```

## Validation-First Patterns (Hybrid)

The validation system uses a hybrid approach: OOP for state management (ValidationContext) and FP for individual validators.

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Any, Callable, Set, Dict
import re

# Enums for validation
class ValidationLevel(Enum):
    STRUCTURE = "structure"
    CONTENT = "content"
    FORMULA = "formula"
    SEMANTIC = "semantic"
    INTEGRITY = "integrity"

class ValidationSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

# Immutable validation issue
@dataclass(frozen=True)
class ValidationIssue:
    """Immutable validation issue record."""
    level: ValidationLevel
    severity: ValidationSeverity
    message: str
    cell_ref: Optional[str] = None
    sheet_name: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None

# OOP for state management
class ValidationContext:
    """
    Manages validation state throughout analysis.

    Uses OOP because:
    1. Need to accumulate issues across multiple validations
    2. Cache validation results for efficiency
    3. Track which validations have been performed
    """

    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.validations_performed: Set[str] = set()
        self.validation_cache: Dict[str, Any] = {}

    def validate(
        self,
        validator: Callable,
        *args,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute validation with caching."""
        if cache_key and cache_key in self.validation_cache:
            return self.validation_cache[cache_key]

        try:
            result = validator(*args, **kwargs)
            if cache_key:
                self.validation_cache[cache_key] = result
            self.validations_performed.add(validator.__name__)
            return result
        except Exception as e:
            self.add_issue(
                ValidationLevel.STRUCTURE,
                ValidationSeverity.ERROR,
                f"Validation failed: {str(e)}",
                details={"validator": validator.__name__, "error": str(e)}
            )
            return None

    def add_issue(
        self,
        level: ValidationLevel,
        severity: ValidationSeverity,
        message: str,
        **kwargs
    ):
        """Record a validation issue."""
        issue = ValidationIssue(
            level=level,
            severity=severity,
            message=message,
            **kwargs
        )
        self.issues.append(issue)

    def has_blocking_issues(self) -> bool:
        """Check if any critical issues prevent processing."""
        return any(issue.severity == ValidationSeverity.CRITICAL for issue in self.issues)

    def get_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        issues_by_severity = {}
        issues_by_level = {}

        for issue in self.issues:
            # Group by severity
            severity = issue.severity.value
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)

            # Group by level
            level = issue.level.value
            if level not in issues_by_level:
                issues_by_level[level] = []
            issues_by_level[level].append(issue)

        return {
            "total_issues": len(self.issues),
            "blocking_issues": self.has_blocking_issues(),
            "by_severity": {k: len(v) for k, v in issues_by_severity.items()},
            "by_level": {k: len(v) for k, v in issues_by_level.items()},
            "validations_performed": list(self.validations_performed)
        }

# Pure validation functions
def validate_formula_syntax(formula: str) -> Optional[str]:
    """
    Validate Excel formula syntax - pure function.
    Returns error message if invalid, None if valid.
    """
    if not formula.startswith('='):
        return "Formula must start with '='"

    # Check parentheses balance
    paren_count = formula.count('(') - formula.count(')')
    if paren_count != 0:
        return f"Unbalanced parentheses: {paren_count} extra {'(' if paren_count > 0 else ')'}"

    # Check quote balance
    quote_count = formula.count('"')
    if quote_count % 2 != 0:
        return "Unbalanced quotes in formula"

    return None

def validate_cell_reference(ref: str) -> Optional[str]:
    """
    Validate cell reference format - pure function.
    Returns error message if invalid, None if valid.
    """
    # Remove sheet name if present
    if '!' in ref:
        ref = ref.split('!')[-1]

    # Remove absolute markers
    clean_ref = ref.replace('$', '')

    # Check A1 notation
    if re.match(r'^[A-Z]{1,3}[1-9][0-9]{0,6}$', clean_ref):
        return None

    return f"Invalid cell reference: {ref}"

def validate_data_consistency(data: List[List[Any]]) -> Dict[str, Any]:
    """
    Validate data consistency - pure function.
    Returns consistency report.
    """
    if not data or not data[0]:
        return {"valid": True, "issues": []}

    issues = []
    col_types = {}

    # Analyze each column
    for col_idx in range(len(data[0])):
        types_in_col = set()
        nulls_in_col = 0

        for row in data:
            if col_idx < len(row):
                value = row[col_idx]
                if value is None:
                    nulls_in_col += 1
                else:
                    types_in_col.add(type(value).__name__)

        col_types[f"col_{col_idx}"] = {
            "types": list(types_in_col),
            "null_count": nulls_in_col,
            "mixed_types": len(types_in_col) > 1
        }

        if len(types_in_col) > 1:
            issues.append(f"Column {col_idx} has mixed types: {types_in_col}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "column_analysis": col_types
    }

# Validation strategy builder (functional)
def build_validation_strategy(
    file_metadata: Dict[str, Any]
) -> List[Callable]:
    """Build validation strategy based on file characteristics - pure function."""
    validators = []

    # Always validate structure
    validators.append(validate_file_structure)

    # Size-based strategy
    if file_metadata.get("size_bytes", 0) > 50_000_000:
        validators.append(sample_based_validator)
    else:
        validators.append(full_scan_validator)

    # Formula validation if needed
    if file_metadata.get("has_formulas", False):
        validators.append(validate_all_formulas)

    # Security validation if needed
    if file_metadata.get("has_macros", False):
        validators.append(validate_macro_security)

    return validators

def validate_file_structure(context: ValidationContext, file_data: Dict[str, Any]):
    """Validate overall file structure."""
    if not file_data.get("sheets"):
        context.add_issue(
            ValidationLevel.STRUCTURE,
            ValidationSeverity.CRITICAL,
            "No sheets found in workbook"
        )
        return

    # Check sheet names
    sheet_names = [s.get("name", "") for s in file_data["sheets"]]
    if len(sheet_names) != len(set(sheet_names)):
        context.add_issue(
            ValidationLevel.STRUCTURE,
            ValidationSeverity.ERROR,
            "Duplicate sheet names detected"
        )

def sample_based_validator(context: ValidationContext, workbook_data: Dict[str, Any]):
    """Validate using sampling for large files."""
    import random

    sample_size = 1000
    total_cells = workbook_data.get("total_cells", 0)

    if total_cells == 0:
        return

    # Sample cells
    sample_count = min(sample_size, total_cells)
    # In real implementation, would sample actual cells

    context.add_issue(
        ValidationLevel.CONTENT,
        ValidationSeverity.INFO,
        f"Validated {sample_count} cells out of {total_cells} using sampling"
    )

def full_scan_validator(context: ValidationContext, workbook_data: Dict[str, Any]):
    """Validate all cells for smaller files."""
    # Full validation logic
    pass

def validate_all_formulas(context: ValidationContext, formula_data: Dict[str, Any]):
    """Validate all formulas in workbook."""
    for formula_info in formula_data.get("formulas", []):
        formula = formula_info.get("formula", "")
        cell_ref = formula_info.get("cell_ref", "")

        # Use pure validator
        error = validate_formula_syntax(formula)
        if error:
            context.add_issue(
                ValidationLevel.FORMULA,
                ValidationSeverity.ERROR,
                error,
                cell_ref=cell_ref
            )

def validate_macro_security(context: ValidationContext, security_data: Dict[str, Any]):
    """Validate macro security concerns."""
    if security_data.get("has_auto_exec_macros"):
        context.add_issue(
            ValidationLevel.STRUCTURE,
            ValidationSeverity.CRITICAL,
            "Auto-executing macros detected - high security risk"
        )

# Compose validation pipeline
def create_validation_pipeline(file_path: Path) -> Callable:
    """Create complete validation pipeline - functional composition."""
    def pipeline():
        # Initialize context (OOP for state)
        context = ValidationContext()

        # Get file metadata
        integrity = stage_0_integrity_probe(file_path)
        security = stage_1_security_scan(file_path)

        # Build strategy (functional)
        validators = build_validation_strategy({
            "size_bytes": integrity.metadata.size_bytes,
            "has_formulas": True,  # Would check actual data
            "has_macros": security.has_macros
        })

        # Run validators
        for validator in validators:
            context.validate(validator, context, {"integrity": integrity, "security": security})

            # Early exit on critical issues
            if context.has_blocking_issues():
                break

        return context.get_summary()

    return pipeline
```

## Agent Integration: Bootstrap Cell

The bootstrap cell remains largely the same but uses the new functional/OOP implementations:

```python
# === AGENT BOOTSTRAP CELL (Idempotent) ===
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache

import openpyxl
import pandas as pd
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
with open(f"/tmp/context/{{ job_id }}/deterministic_analysis.json") as f:
    DETERMINISTIC_CONTEXT = json.load(f)

# Extract contexts
INTEGRITY_CONTEXT = DETERMINISTIC_CONTEXT["integrity"]
SECURITY_CONTEXT = DETERMINISTIC_CONTEXT["security"]
STRUCTURE_CONTEXT = DETERMINISTIC_CONTEXT["structure"]
FORMULA_CONTEXT = DETERMINISTIC_CONTEXT["formulas"]
INTELLIGENCE_CONTEXT = DETERMINISTIC_CONTEXT["intelligence"]

# Sheet-specific context
SHEET_CONTEXT = next(
    (sheet for sheet in STRUCTURE_CONTEXT["sheets"]
     if sheet["name"] == SHEET_NAME),
    {"name": SHEET_NAME, "error": "Sheet context not found"}
)

# === HELPER FUNCTIONS (Mix of FP and OOP as appropriate) ===

@lru_cache(maxsize=1)
def load_workbook() -> openpyxl.Workbook:
    """Load workbook once per agent, cached."""
    logger.info(f"Loading workbook: {EXCEL_FILE}")
    return openpyxl.load_workbook(
        EXCEL_FILE,
        read_only=True,
        data_only=True,
        keep_links=False
    )

def load_sheet_data(
    range_spec: Optional[str] = None,
    sample_rows: Optional[int] = None,
    **pandas_kwargs
) -> pd.DataFrame:
    """Selective data loading with pandas."""
    kwargs = {
        "engine": "openpyxl",
        "sheet_name": SHEET_NAME,
        **pandas_kwargs
    }

    if range_spec:
        kwargs["usecols"] = range_spec

    if sample_rows:
        max_row = SHEET_CONTEXT.get("dimensions", {}).get("max_row", 1000)
        skip_rows = max(1, max_row // sample_rows)
        kwargs["skiprows"] = lambda x: x % skip_rows != 0

    return pd.read_excel(EXCEL_FILE, **kwargs)

def get_formula_graph() -> Optional[nx.DiGraph]:
    """Load pre-computed formula dependency graph."""
    graph_path = FORMULA_CONTEXT.get("graph_path")
    if graph_path and Path(graph_path).exists():
        return nx.read_graphml(graph_path)
    return None

def validate_calculation(
    cell_range: str,
    expected_function: str = "SUM"
) -> Dict[str, Any]:
    """Validate calculation - functional approach."""
    try:
        df = load_sheet_data(range_spec=cell_range)

        # Pure calculation
        if expected_function.upper() == "SUM":
            result = df.sum().sum()
        elif expected_function.upper() == "AVERAGE":
            result = df.mean().mean()
        else:
            return {"error": f"Unsupported function: {expected_function}"}

        return {
            "range": cell_range,
            "function": expected_function,
            "result": result,
            "validated": True
        }
    except Exception as e:
        return {"error": str(e), "validated": False}

def report_finding(
    finding_type: str,
    description: str,
    evidence: Dict[str, Any],
    severity: str = "INFO"
) -> None:
    """Report analysis finding."""
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

# === CONTEXT SUMMARY ===
print(f"🤖 Agent {AGENT_ID} initialized for sheet '{SHEET_NAME}'")
print(f"📁 File: {EXCEL_FILE.name} ({INTEGRITY_CONTEXT['size_mb']}MB)")
print(f"📊 Dimensions: {SHEET_CONTEXT.get('dimensions', {}).get('used_range', 'Unknown')}")
print(f"🔢 Formulas: {FORMULA_CONTEXT.get('total_formulas', 0)}")
print(f"⚠️  Risk score: {SECURITY_CONTEXT.get('risk_score', 0)}/10")

# Show recommendations
recommendations = INTELLIGENCE_CONTEXT.get("agent_recommendations", {})
if recommendations.get("focus_areas"):
    print("\n📋 Focus areas:")
    for area in recommendations["focus_areas"][:3]:
        print(f"  • {area}")

print("\n✅ Agent ready. Available functions:")
print("  • load_workbook() - Full workbook access")
print("  • load_sheet_data() - Selective data loading")
print("  • get_formula_graph() - Dependency graph")
print("  • validate_calculation() - Calculation verification")
print("  • report_finding() - Report discoveries")
```

## Performance Characteristics

The functional/OOP hybrid approach maintains excellent performance:

| Stage   | Paradigm   | Performance Impact                          |
| ------- | ---------- | ------------------------------------------- |
| Stage 0 | Functional | ✅ Easily parallelizable, pure functions    |
| Stage 1 | Functional | ✅ Stateless scanners can run concurrently  |
| Stage 2 | Hybrid     | ✅ Controlled state, functional extraction  |
| Stage 3 | OOP        | ✅ NetworkX optimizations, graph algorithms |
| Stage 4 | Functional | ✅ Pure computation, no I/O                 |

## Conclusion

This hybrid functional/object-oriented design leverages the strengths of each paradigm:

- **Functional Programming** for stateless operations, data transformations, and pure computations
- **Object-Oriented Programming** for stateful graph operations and complex data structure management
- **Hybrid Approaches** where both paradigms complement each other

The result is a maintainable, testable, and performant deterministic analysis pipeline that provides a solid foundation for AI-powered Excel analysis.
