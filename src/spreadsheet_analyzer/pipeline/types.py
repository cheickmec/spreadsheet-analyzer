"""
Shared immutable data structures for the deterministic analysis pipeline.

This module defines all the frozen dataclasses and type definitions used
across the various pipeline stages, following functional programming principles.
"""

from collections.abc import Callable
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
            "created_time": self.metadata.created_time.isoformat(),
            "modified_time": self.metadata.modified_time.isoformat(),
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "threat_type": self.threat_type,
            "severity": self.severity,
            "location": self.location,
            "description": self.description,
            "risk_level": self.risk_level,
            "details": self.details,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "threats": [t.to_dict() for t in self.threats],
            "has_macros": self.has_macros,
            "has_external_links": self.has_external_links,
            "has_embedded_objects": self.has_embedded_objects,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "scan_complete": self.scan_complete,
            "is_safe": self.is_safe,
            "threat_count": self.threat_count,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "used_range": self.used_range,
            "has_data": self.has_data,
            "has_formulas": self.has_formulas,
            "has_charts": self.has_charts,
            "has_pivot_tables": self.has_pivot_tables,
            "cell_count": self.cell_count,
            "formula_count": self.formula_count,
        }


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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sheet_count": self.sheet_count,
            "sheets": [s.to_dict() for s in self.sheets],
            "total_cells": self.total_cells,
            "total_formulas": self.total_formulas,
            "named_ranges": list(self.named_ranges),
            "has_vba_project": self.has_vba_project,
            "has_external_links": self.has_external_links,
            "complexity_score": self.complexity_score,
        }


# ==================== Stage 3: Formula Types ====================


@dataclass(frozen=True)
class EdgeMetadata:
    """
    Metadata for dependency graph edges with semantic information.

    This structure captures the relationship type and context between
    cells in the dependency graph.
    """

    edge_type: str  # e.g., "SUMS_OVER", "LOOKS_UP_IN"
    function_name: str | None  # The function creating this dependency
    argument_position: int | None  # Which argument position
    weight: float = 1.0  # Edge weight based on range size or importance
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RangeMembershipIndex:
    """
    Index for efficient range membership queries.

    CLAUDE-PERFORMANCE: This index dramatically improves performance when
    handling large ranges by avoiding repeated range expansion.
    """

    sheet_ranges: dict[str, list[tuple[int, int, int, int, str]]]

    def is_cell_in_any_range(self, sheet: str, row: int, col: int) -> bool:
        """Check if a cell is part of any indexed range."""
        if sheet not in self.sheet_ranges:
            return False

        for min_row, max_row, min_col, max_col, _ in self.sheet_ranges[sheet]:
            if min_row <= row <= max_row and min_col <= col <= max_col:
                return True

        return False


@dataclass(frozen=True)
class CellReference:
    """Reference to a cell in a formula."""

    sheet: SheetName
    cell: CellRef
    is_absolute: bool
    is_range: bool
    ref_type: Literal["single_cell", "range", "named_range", "external"] = "single_cell"
    edge_label: str = "DEPENDS_ON"  # Semantic edge type


@dataclass(frozen=True)
class FormulaNode:
    """
    Represents a cell with a formula in the dependency graph.

    This enhanced node structure includes optional semantic metadata
    for richer analysis capabilities. Used by both formula analysis
    and graph database components.
    """

    sheet: str
    cell: str
    formula: str
    dependencies: frozenset[str]

    # Analysis metadata
    volatile: bool = False
    external: bool = False
    complexity_score: float = 1.0

    # Optional semantic metadata (populated during enhanced analysis)
    edge_labels: dict[str, EdgeMetadata] | None = None
    cell_metadata: dict[str, Any] | None = None

    # Graph-specific fields (populated during graph operations)
    dependents: frozenset[str] | None = None
    depth: int | None = None  # Distance from leaf nodes
    pagerank: float | None = None  # Pre-computed importance score

    def with_semantic_data(
        self, edge_labels: dict[str, EdgeMetadata], cell_metadata: dict[str, Any] | None = None
    ) -> "FormulaNode":
        """Create a new node with semantic data added."""
        from dataclasses import replace

        return replace(self, edge_labels=edge_labels, cell_metadata=cell_metadata)

    def with_graph_data(self, dependents: frozenset[str], depth: int, pagerank: float | None = None) -> "FormulaNode":
        """Create a new node with graph analysis data added."""
        from dataclasses import replace

        return replace(self, dependents=dependents, depth=depth, pagerank=pagerank)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "sheet": self.sheet,
            "cell": self.cell,
            "formula": self.formula,
            "dependencies": list(self.dependencies),
            "volatile": self.volatile,
            "external": self.external,
            "complexity_score": self.complexity_score,
        }

        # Add optional fields if present
        if self.edge_labels:
            result["edge_labels"] = {k: v.__dict__ for k, v in self.edge_labels.items()}
        if self.cell_metadata:
            result["cell_metadata"] = self.cell_metadata
        if self.dependents is not None:
            result["dependents"] = list(self.dependents)
        if self.depth is not None:
            result["depth"] = self.depth
        if self.pagerank is not None:
            result["pagerank"] = self.pagerank

        return result


@dataclass(frozen=True)
class FormulaAnalysis:
    """
    Complete formula analysis results with optional semantic enhancements.

    This structure contains all analysis outputs including the dependency
    graph, detected issues, and complexity metrics.
    """

    dependency_graph: dict[str, FormulaNode]
    circular_references: frozenset[frozenset[str]]
    volatile_formulas: frozenset[str]
    external_references: frozenset[str]
    max_dependency_depth: int
    formula_complexity_score: float
    statistics: dict[str, int]
    range_index: RangeMembershipIndex

    @property
    def has_circular_references(self) -> bool:
        """Check if workbook has circular references."""
        return len(self.circular_references) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dependency_graph": {k: v.to_dict() for k, v in self.dependency_graph.items()},
            "circular_references": [list(cycle) for cycle in self.circular_references],
            "volatile_formulas": list(self.volatile_formulas),
            "external_references": list(self.external_references),
            "max_dependency_depth": self.max_dependency_depth,
            "formula_complexity_score": self.formula_complexity_score,
            "has_circular_references": self.has_circular_references,
            "statistics": self.statistics,
        }


# ==================== Stage 4: Content Types ====================


@dataclass(frozen=True)
class DataPattern:
    """Detected pattern in data."""

    pattern_type: str
    confidence: float  # 0.0-1.0
    locations: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern_type": self.pattern_type,
            "confidence": self.confidence,
            "locations": list(self.locations),
            "description": self.description,
        }


@dataclass(frozen=True)
class ContentInsight:
    """Single insight from content analysis."""

    insight_type: str
    title: str
    description: str
    severity: str
    affected_areas: tuple[str, ...]
    recommendation: str | None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "insight_type": self.insight_type,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "affected_areas": list(self.affected_areas),
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class ContentAnalysis:
    """Stage 4 content intelligence result."""

    data_patterns: tuple[DataPattern, ...]
    insights: tuple[ContentInsight, ...]
    data_quality_score: int  # 0-100
    summary: str
    key_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "data_patterns": [p.to_dict() for p in self.data_patterns],
            "insights": [i.to_dict() for i in self.insights],
            "data_quality_score": self.data_quality_score,
            "summary": self.summary,
            "key_metrics": self.key_metrics,
        }


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


# ==================== Service Layer Types ====================


@dataclass
class AnalysisOptions:
    """Options for analysis that can come from CLI args or API requests."""

    mode: str = "standard"  # 'fast', 'standard', 'deep'
    include_formulas: bool = True
    include_security: bool = True
    include_content: bool = True
    max_depth: int | None = None
    progress_callback: Callable[[str, float, str], None] | None = None

    # Performance options
    max_cells_to_analyze: int = 10_000
    parallel_sheets: bool = True

    # Output options
    include_raw_data: bool = False
    include_statistics: bool = True


@dataclass
class AnalysisResult:
    """Result of spreadsheet analysis."""

    file_path: Path
    file_size: int
    analysis_mode: str

    # Pipeline results
    integrity: IntegrityResult | None = None
    security: SecurityReport | None = None
    structure: WorkbookStructure | None = None
    formulas: FormulaAnalysis | None = None
    content: ContentAnalysis | None = None

    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_seconds: float = 0.0

    # Summary
    issues: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        """Check if analysis found no critical issues."""
        return len(self.issues) == 0

    @property
    def has_warnings(self) -> bool:
        """Check if analysis found any warnings."""
        return len(self.warnings) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "analysis_mode": self.analysis_mode,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "is_healthy": self.is_healthy,
            "has_warnings": self.has_warnings,
            "issues": self.issues,
            "warnings": self.warnings,
            "statistics": self.statistics,
            "results": {
                "integrity": self.integrity.to_dict() if self.integrity else None,
                "security": self.security.to_dict() if self.security else None,
                "structure": self.structure.to_dict() if self.structure else None,
                "formulas": self.formulas.to_dict() if self.formulas else None,
                "content": self.content.to_dict() if self.content else None,
            },
        }
