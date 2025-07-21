"""
Language-agnostic fixture loading with optional type reconstruction.

This module provides clean JSON fixtures without Python-specific type markers,
while still allowing type-safe reconstruction when needed.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import jsonschema

    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

from spreadsheet_analyzer.pipeline.types import (
    ContentAnalysis,
    ContentInsight,
    DataPattern,
    FormulaAnalysis,
    IntegrityResult,
    PipelineContext,
    PipelineResult,
    SecurityReport,
    SecurityThreat,
    SheetStructure,
    WorkbookStructure,
)


class FixtureLoader:
    """Load fixtures from language-agnostic JSON files."""

    def __init__(self, fixtures_dir: Path | None = None):
        if fixtures_dir is None:
            self.fixtures_dir = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "captured_outputs"
        else:
            self.fixtures_dir = fixtures_dir

    def load_raw(self, test_file: str) -> dict[str, Any]:
        """
        Load raw JSON fixture without any type reconstruction.

        This returns plain Python dictionaries that can be used by any language.
        """
        fixture_path = self._get_fixture_path(test_file)
        with fixture_path.open() as f:
            data: dict[str, Any] = json.load(f)
            return data

    def load_as_dataclass(self, test_file: str) -> PipelineResult:
        """
        Load fixture and reconstruct as Python dataclasses.

        This provides type-safe access for Python code while the underlying
        JSON remains language-agnostic.
        """
        data = self.load_raw(test_file)
        return self._reconstruct_pipeline_result(data["pipeline_result"])

    def _get_fixture_path(self, test_file: str) -> Path:
        """Get the fixture path for a test file."""
        # Handle both with and without extension
        if not test_file.endswith((".xlsx", ".xlsm", ".json")):
            test_file += ".xlsx"

        # Convert to JSON fixture name
        fixture_name = Path(test_file).with_suffix(".json")
        return self.fixtures_dir / fixture_name

    def _reconstruct_pipeline_result(self, data: dict[str, Any]) -> PipelineResult:
        """Reconstruct PipelineResult from plain dict."""
        # Handle each stage
        if data.get("integrity"):
            data["integrity"] = self._reconstruct_integrity(data["integrity"])

        if data.get("security"):
            data["security"] = self._reconstruct_security(data["security"])

        if data.get("structure"):
            data["structure"] = self._reconstruct_structure(data["structure"])

        if data.get("formulas"):
            data["formulas"] = self._reconstruct_formulas(data["formulas"])

        if data.get("content"):
            data["content"] = self._reconstruct_content(data["content"])

        # Handle context
        if data.get("context"):
            data["context"] = self._reconstruct_context(data["context"])

        # Handle errors (convert list to frozenset)
        if "errors" in data:
            data["errors"] = frozenset(data["errors"])

        return PipelineResult(**data)

    def _reconstruct_integrity(self, data: dict[str, Any]) -> IntegrityResult:
        """Reconstruct IntegrityResult from dict."""
        # Handle path
        if "file_path" in data and isinstance(data["file_path"], str):
            data["file_path"] = Path(data["file_path"])

        # Handle datetime
        if "scan_timestamp" in data and isinstance(data["scan_timestamp"], str):
            data["scan_timestamp"] = datetime.fromisoformat(data["scan_timestamp"])

        return IntegrityResult(**data)

    def _reconstruct_security(self, data: dict[str, Any]) -> SecurityReport:
        """Reconstruct SecurityReport from dict."""
        # Handle threats
        if "threats" in data:
            data["threats"] = tuple(SecurityThreat(**threat) for threat in data["threats"])

        # Handle lists as tuples
        for field in ["vba_modules", "external_links"]:
            if field in data:
                data[field] = tuple(data[field])

        # Handle datetime
        if "scan_timestamp" in data and isinstance(data["scan_timestamp"], str):
            data["scan_timestamp"] = datetime.fromisoformat(data["scan_timestamp"])

        return SecurityReport(**data)

    def _reconstruct_structure(self, data: dict[str, Any]) -> WorkbookStructure:
        """Reconstruct WorkbookStructure from dict."""
        # Handle sheets
        if "sheets" in data:
            data["sheets"] = tuple(SheetStructure(**sheet) for sheet in data["sheets"])

        # Handle named ranges
        if "named_ranges" in data:
            data["named_ranges"] = tuple(data["named_ranges"])

        return WorkbookStructure(**data)

    def _reconstruct_formulas(self, data: dict[str, Any]) -> FormulaAnalysis:
        """Reconstruct FormulaAnalysis from dict."""
        # Note: formula_nodes is complex and may need special handling
        # For now, we'll skip it as it's not in the fixtures

        # Handle lists as tuples/frozensets
        for field in ["circular_references", "volatile_formulas", "external_references"]:
            if field in data:
                if field == "circular_references":
                    # Nested lists -> frozenset of frozensets
                    data[field] = frozenset(frozenset(ref) for ref in data[field])
                else:
                    # Lists -> frozensets
                    data[field] = frozenset(data[field])

        # Ensure required fields are present
        if "dependency_graph" not in data:
            data["dependency_graph"] = {}

        if "statistics" not in data:
            data["statistics"] = {}

        # Handle range_index - create a dummy one if not present
        if "range_index" not in data:
            from spreadsheet_analyzer.pipeline.types import RangeMembershipIndex

            data["range_index"] = RangeMembershipIndex(sheet_ranges={})

        return FormulaAnalysis(**data)

    def _reconstruct_content(self, data: dict[str, Any]) -> ContentAnalysis:
        """Reconstruct ContentAnalysis from dict."""
        # Handle patterns
        if "data_patterns" in data:
            data["data_patterns"] = tuple(DataPattern(**pattern) for pattern in data["data_patterns"])

        # Handle insights
        if "insights" in data:
            data["insights"] = tuple(ContentInsight(**insight) for insight in data["insights"])

        return ContentAnalysis(**data)

    def _reconstruct_context(self, data: dict[str, Any]) -> PipelineContext:
        """Reconstruct PipelineContext from dict."""
        # Handle path
        if "file_path" in data and isinstance(data["file_path"], str):
            data["file_path"] = Path(data["file_path"])

        # Handle datetime
        if "start_time" in data and isinstance(data["start_time"], str):
            data["start_time"] = datetime.fromisoformat(data["start_time"])

        # Handle stage_results (should be empty dict)
        if "stage_results" not in data:
            data["stage_results"] = {}

        return PipelineContext(**data)


def validate_fixture_against_schema(fixture_path: Path, schema_path: Path | None = None) -> bool:
    """
    Validate a fixture file against the JSON schema.

    This ensures the fixture conforms to the language-agnostic schema.
    """
    if not HAS_JSONSCHEMA:
        print("jsonschema not installed, skipping validation")
        return True

    if schema_path is None:
        schema_path = (
            Path(__file__).parent.parent.parent.parent
            / "tests"
            / "fixtures"
            / "schemas"
            / "pipeline_result.schema.json"
        )

    with fixture_path.open() as f:
        fixture_data = json.load(f)

    with schema_path.open() as f:
        schema = json.load(f)

    try:
        # Validate just the pipeline_result part
        jsonschema.validate(fixture_data.get("pipeline_result", {}), schema)
    except jsonschema.ValidationError as e:
        print(f"Schema validation failed: {e}")
        return False
    else:
        return True
