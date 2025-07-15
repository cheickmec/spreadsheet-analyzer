"""
Shared immutable data structures for the deterministic analysis pipeline.

This module defines all the frozen dataclasses and type definitions used
across the various pipeline stages, following functional programming principles.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Type aliases for clarity
CellRef = str  # Excel cell reference (e.g., 'A1', 'B2')
SheetName = str
Formula = str
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ProcessingClass = Literal["STANDARD", "HEAVY", "BLOCKED"]

# ==================== Result Types ====================


@dataclass(frozen=True)
class Ok[T]:
    """Successful result wrapper."""

    value: T


@dataclass(frozen=True)
class Err:
    """Error result wrapper."""

    error: str
    details: dict[str, Any] | None = None


# For backwards compatibility
Result = Ok | Err

# ==================== Stage 0: Integrity Types ====================


@dataclass(frozen=True)
class FileMetadata:
    """Immutable file metadata."""

    path: Path
    size_bytes: int
    mime_type: str
    created_time: datetime
    modified_time: datetime

    @property
    def size_mb(self) -> float:
        """File size in megabytes."""
        return round(self.size_bytes / (1024 * 1024), 2)

    @property
    def extension(self) -> str:
        """File extension lowercased."""
        return self.path.suffix.lower()


@dataclass(frozen=True)
class IntegrityResult:
    """Stage 0 integrity check result."""

    file_hash: str
    metadata: FileMetadata
    is_excel: bool
    is_ooxml: bool
    is_duplicate: bool
    trust_tier: int  # 1-5, where 5 is most trusted
    processing_class: ProcessingClass
    validation_passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_hash": self.file_hash,
            "file_path": str(self.metadata.path),
            "size_bytes": self.metadata.size_bytes,
            "size_mb": self.metadata.size_mb,
            "mime_type": self.metadata.mime_type,
            "is_excel": self.is_excel,
            "is_ooxml": self.is_ooxml,
            "is_duplicate": self.is_duplicate,
            "trust_tier": self.trust_tier,
            "processing_class": self.processing_class,
            "validation_passed": self.validation_passed,
        }


# ==================== Stage 1: Security Types ====================


@dataclass(frozen=True)
class SecurityThreat:
    """Immutable security threat descriptor."""

    threat_type: str
    severity: int  # 1-10
    location: str
    description: str
    risk_level: RiskLevel
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SecurityReport:
    """Stage 1 security analysis result."""

    threats: tuple[SecurityThreat, ...]
    has_macros: bool
    has_external_links: bool
    has_embedded_objects: bool
    risk_score: int  # 0-100
    risk_level: RiskLevel
    scan_complete: bool

    @property
    def is_safe(self) -> bool:
        """Check if file is safe for processing."""
        return self.risk_level in ("LOW", "MEDIUM")

    @property
    def threat_count(self) -> int:
        """Total number of threats found."""
        return len(self.threats)


# ==================== Stage 2: Structure Types ====================


@dataclass(frozen=True)
class CellInfo:
    """Information about a single cell."""

    sheet: SheetName
    address: CellRef
    value: Any
    formula: Formula | None
    data_type: str
    has_formatting: bool


@dataclass(frozen=True)
class SheetStructure:
    """Structure of a single worksheet."""

    name: SheetName
    row_count: int
    column_count: int
    used_range: str
    has_data: bool
    has_formulas: bool
    has_charts: bool
    has_pivot_tables: bool
    cell_count: int
    formula_count: int


@dataclass(frozen=True)
class WorkbookStructure:
    """Stage 2 structural analysis result."""

    sheets: tuple[SheetStructure, ...]
    total_cells: int
    total_formulas: int
    named_ranges: tuple[str, ...]
    has_vba_project: bool
    has_external_links: bool
    complexity_score: int  # 0-100

    @property
    def sheet_count(self) -> int:
        """Number of sheets in workbook."""
        return len(self.sheets)


# ==================== Stage 3: Formula Types ====================


@dataclass(frozen=True)
class CellReference:
    """Reference to a cell in a formula."""

    sheet: SheetName
    cell: CellRef
    is_absolute: bool
    is_range: bool


@dataclass(frozen=True)
class FormulaNode:
    """Node in formula dependency graph."""

    sheet: SheetName
    cell: CellRef
    formula: Formula
    dependencies: frozenset[CellReference]
    dependents: frozenset[CellReference]
    depth: int  # Distance from leaf nodes


@dataclass(frozen=True)
class FormulaAnalysis:
    """Stage 3 formula analysis result."""

    dependency_graph: dict[str, FormulaNode]
    circular_references: tuple[tuple[str, ...], ...]
    volatile_formulas: tuple[str, ...]
    external_references: tuple[str, ...]
    max_dependency_depth: int
    formula_complexity_score: int

    @property
    def has_circular_references(self) -> bool:
        """Check if workbook has circular references."""
        return len(self.circular_references) > 0


# ==================== Stage 4: Content Types ====================


@dataclass(frozen=True)
class DataPattern:
    """Detected pattern in data."""

    pattern_type: str
    confidence: float  # 0.0-1.0
    locations: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class ContentInsight:
    """Single insight from content analysis."""

    insight_type: str
    title: str
    description: str
    severity: str
    affected_areas: tuple[str, ...]
    recommendation: str | None


@dataclass(frozen=True)
class ContentAnalysis:
    """Stage 4 content intelligence result."""

    data_patterns: tuple[DataPattern, ...]
    insights: tuple[ContentInsight, ...]
    data_quality_score: int  # 0-100
    summary: str
    key_metrics: dict[str, Any]


# ==================== Pipeline Orchestration Types ====================


@dataclass(frozen=True)
class PipelineContext:
    """Immutable context passed through pipeline stages."""

    file_path: Path
    start_time: datetime
    options: dict[str, Any]
    stage_results: dict[str, Any] = field(default_factory=dict)

    def with_stage_result(self, stage: str, result: Any) -> "PipelineContext":
        """Create new context with additional stage result."""
        new_results = {**self.stage_results, stage: result}
        return PipelineContext(
            file_path=self.file_path, start_time=self.start_time, options=self.options, stage_results=new_results
        )


@dataclass(frozen=True)
class PipelineResult:
    """Complete pipeline execution result."""

    context: PipelineContext
    integrity: IntegrityResult | None
    security: SecurityReport | None
    structure: WorkbookStructure | None
    formulas: FormulaAnalysis | None
    content: ContentAnalysis | None
    execution_time: float
    success: bool
    errors: tuple[str, ...]

    def to_report(self) -> dict[str, Any]:
        """Generate comprehensive analysis report."""
        return {
            "file": str(self.context.file_path),
            "execution_time": self.execution_time,
            "success": self.success,
            "errors": list(self.errors),
            "integrity": self.integrity.to_dict() if self.integrity else None,
            "security": {
                "risk_level": self.security.risk_level,
                "threat_count": self.security.threat_count,
                "is_safe": self.security.is_safe,
            }
            if self.security
            else None,
            "structure": {
                "sheet_count": self.structure.sheet_count,
                "total_cells": self.structure.total_cells,
                "total_formulas": self.structure.total_formulas,
                "complexity_score": self.structure.complexity_score,
            }
            if self.structure
            else None,
            "formulas": {
                "has_circular_references": self.formulas.has_circular_references,
                "max_dependency_depth": self.formulas.max_dependency_depth,
                "volatile_formula_count": len(self.formulas.volatile_formulas),
            }
            if self.formulas
            else None,
            "content": {
                "data_quality_score": self.content.data_quality_score,
                "pattern_count": len(self.content.data_patterns),
                "insight_count": len(self.content.insights),
                "summary": self.content.summary,
            }
            if self.content
            else None,
        }


# ==================== Progress Tracking Types ====================


@dataclass(frozen=True)
class ProgressUpdate:
    """Immutable progress update."""

    stage: str
    progress: float  # 0.0-1.0
    message: str
    timestamp: datetime
    details: dict[str, Any] | None = None
