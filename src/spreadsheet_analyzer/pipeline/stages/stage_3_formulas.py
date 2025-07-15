"""
Stage 3: Formula Analysis (Object-Oriented Programming).

This module implements formula dependency analysis using OOP principles
for managing the complex stateful graph operations required.
"""

import logging
import re
from collections import defaultdict, deque
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Final

import openpyxl
from openpyxl.formula import Tokenizer

from spreadsheet_analyzer.pipeline.types import CellReference, Err, FormulaAnalysis, FormulaNode, Ok, Result

logger = logging.getLogger(__name__)

# ==================== Constants ====================

# Formula complexity thresholds
HIGH_FORMULA_COUNT: Final[int] = 1000
MEDIUM_FORMULA_COUNT: Final[int] = 100
LOW_FORMULA_COUNT: Final[int] = 10

HIGH_DEPTH_THRESHOLD: Final[int] = 10
MEDIUM_DEPTH_THRESHOLD: Final[int] = 5
LOW_DEPTH_THRESHOLD: Final[int] = 2

# ==================== Formula Parsing Utilities ====================


class FormulaParser:
    """
    Parser for Excel formulas to extract cell references.

    CLAUDE-COMPLEX: Excel formula parsing is complex due to:
    - Multiple reference styles (A1, R1C1)
    - Sheet references (Sheet1!A1)
    - Range references (A1:B10)
    - Named ranges
    - External references ([Book1.xlsx]Sheet1!A1)
    """

    # Regex patterns for different reference types
    CELL_PATTERN = re.compile(r"(?:(?P<sheet>[\w\s]+)!)?(?P<col>\$?[A-Z]+)(?P<row>\$?\d+)")
    RANGE_PATTERN = re.compile(
        r"(?:(?P<sheet>[\w\s]+)!)?(?P<start_col>\$?[A-Z]+)(?P<start_row>\$?\d+)"
        r":(?P<end_col>\$?[A-Z]+)(?P<end_row>\$?\d+)"
    )

    def __init__(self, current_sheet: str):
        """Initialize parser with current sheet context."""
        self.current_sheet = current_sheet

    def parse_formula(self, formula: str) -> set[CellReference]:
        """
        Parse formula and extract all cell references.

        CLAUDE-GOTCHA: Excel's formula parser is very complex.
        We use a simplified approach that covers most common cases.
        """
        references: set[CellReference] = set()

        if not formula or not formula.startswith("="):
            return references

        # Remove the leading '='
        formula = formula[1:]

        # Try to use openpyxl's tokenizer if available
        try:
            tokenizer = Tokenizer(formula)
            for token in tokenizer.items:
                if token.type == "OPERAND" and token.subtype == "RANGE":
                    refs = self._parse_reference(token.value)
                    references.update(refs)
        except (ValueError, AttributeError, TypeError):
            # Fallback to regex parsing
            references.update(self._parse_with_regex(formula))

        return references

    def _parse_reference(self, ref_str: str) -> set[CellReference]:
        """Parse a single reference string."""
        references: set[CellReference] = set()

        # Check for range reference
        if ":" in ref_str:
            match = self.RANGE_PATTERN.match(ref_str)
            if match:
                sheet = match.group("sheet") or self.current_sheet
                # For ranges, we just note it as a range reference
                references.add(CellReference(sheet=sheet, cell=ref_str, is_absolute="$" in ref_str, is_range=True))
        else:
            # Single cell reference
            match = self.CELL_PATTERN.match(ref_str)
            if match:
                sheet = match.group("sheet") or self.current_sheet
                col = match.group("col")
                row = match.group("row")
                references.add(
                    CellReference(sheet=sheet, cell=f"{col}{row}", is_absolute="$" in ref_str, is_range=False)
                )

        return references

    def _parse_with_regex(self, formula: str) -> set[CellReference]:
        """Fallback regex-based parsing."""
        references: set[CellReference] = set()

        # Find all cell references
        for match in self.CELL_PATTERN.finditer(formula):
            sheet = match.group("sheet") or self.current_sheet
            col = match.group("col")
            row = match.group("row")
            references.add(
                CellReference(sheet=sheet, cell=f"{col}{row}", is_absolute="$" in col or "$" in row, is_range=False)
            )

        # Find all range references
        for match in self.RANGE_PATTERN.finditer(formula):
            sheet = match.group("sheet") or self.current_sheet
            start = f"{match.group('start_col')}{match.group('start_row')}"
            end = f"{match.group('end_col')}{match.group('end_row')}"
            references.add(
                CellReference(sheet=sheet, cell=f"{start}:{end}", is_absolute="$" in match.group(0), is_range=True)
            )

        return references


# ==================== Dependency Graph Classes ====================


class DependencyGraph:
    """
    Manages formula dependency relationships.

    CLAUDE-KNOWLEDGE: We use an adjacency list representation
    for efficient traversal and cycle detection.
    """

    def __init__(self):
        """Initialize empty graph."""
        self.nodes: dict[str, FormulaNode] = {}
        self.edges: dict[str, set[str]] = defaultdict(set)
        self.reverse_edges: dict[str, set[str]] = defaultdict(set)
        self._circular_refs: list[tuple[str, ...]] = []

    def add_node(self, cell_key: str, sheet: str, cell: str, formula: str):
        """Add a formula node to the graph."""
        if cell_key not in self.nodes:
            self.nodes[cell_key] = FormulaNode(
                sheet=sheet, cell=cell, formula=formula, dependencies=frozenset(), dependents=frozenset(), depth=0
            )

    def add_edge(self, from_cell: str, to_cell: str):
        """Add dependency edge from from_cell to to_cell."""
        self.edges[from_cell].add(to_cell)
        self.reverse_edges[to_cell].add(from_cell)

    def finalize_dependencies(self):
        """
        Finalize dependency relationships in nodes.

        CLAUDE-IMPORTANT: This must be called after all edges are added
        to update the immutable FormulaNode structures.
        """
        for cell_key, node in self.nodes.items():
            # Create frozen sets of dependencies
            deps = frozenset(
                CellReference(
                    sheet=self.nodes[dep].sheet,
                    cell=self.nodes[dep].cell,
                    is_absolute=False,  # Simplified
                    is_range=False,
                )
                for dep in self.edges.get(cell_key, set())
                if dep in self.nodes
            )

            dependents = frozenset(
                CellReference(sheet=self.nodes[dep].sheet, cell=self.nodes[dep].cell, is_absolute=False, is_range=False)
                for dep in self.reverse_edges.get(cell_key, set())
                if dep in self.nodes
            )

            # Update node with dependencies
            self.nodes[cell_key] = FormulaNode(
                sheet=node.sheet,
                cell=node.cell,
                formula=node.formula,
                dependencies=deps,
                dependents=dependents,
                depth=node.depth,
            )

    def detect_circular_references(self) -> list[tuple[str, ...]]:
        """
        Detect circular references using DFS.

        CLAUDE-KNOWLEDGE: Circular references in Excel can span
        multiple cells and sheets, making detection complex.
        """
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = tuple(path[cycle_start:])
                    self._circular_refs.append(cycle)
                    return True

            path.pop()
            rec_stack.remove(node)
            return False

        # Check all nodes
        for node in self.nodes:
            if node not in visited:
                dfs(node)

        return self._circular_refs

    def calculate_depths(self):
        """
        Calculate depth of each node (distance from leaf nodes).

        CLAUDE-PERFORMANCE: This helps identify the most complex
        formulas with deep dependency chains.
        """
        # Find leaf nodes (no dependencies)
        leaf_nodes = {node for node in self.nodes if not self.edges.get(node)}

        # BFS from leaf nodes
        queue = deque(leaf_nodes)
        depths = dict.fromkeys(leaf_nodes, 0)

        while queue:
            current = queue.popleft()
            current_depth = depths[current]

            # Update dependents
            for dependent in self.reverse_edges.get(current, set()):
                if dependent not in depths:
                    depths[dependent] = current_depth + 1
                    queue.append(dependent)
                else:
                    depths[dependent] = max(depths[dependent], current_depth + 1)

        # Update nodes with depths
        for cell_key, node in self.nodes.items():
            depth = depths.get(cell_key, 0)
            self.nodes[cell_key] = FormulaNode(
                sheet=node.sheet,
                cell=node.cell,
                formula=node.formula,
                dependencies=node.dependencies,
                dependents=node.dependents,
                depth=depth,
            )

    def get_max_depth(self) -> int:
        """Get maximum dependency depth."""
        if not self.nodes:
            return 0
        return max(node.depth for node in self.nodes.values())


# ==================== Formula Analyzer Class ====================


class FormulaAnalyzer:
    """
    Main analyzer for formula dependencies and patterns.

    CLAUDE-COMPLEX: This class orchestrates the analysis of all
    formulas in a workbook, building a complete dependency graph.
    """

    VOLATILE_FUNCTIONS: ClassVar[set[str]] = {
        "NOW",
        "TODAY",
        "RAND",
        "RANDBETWEEN",
        "INDIRECT",
        "OFFSET",
        "CELL",
        "INFO",
    }

    def __init__(self):
        """Initialize analyzer."""
        self.graph = DependencyGraph()
        self.volatile_formulas: list[str] = []
        self.external_references: list[str] = []
        self.parser_cache: dict[str, FormulaParser] = {}

    def analyze_workbook(self, workbook) -> FormulaAnalysis:
        """
        Analyze all formulas in workbook.

        CLAUDE-PERFORMANCE: We analyze sheets sequentially to avoid
        memory issues with large workbooks.

        Returns complete formula analysis.
        """
        # CLAUDE-KNOWLEDGE: First pass collects all formulas before analyzing
        # dependencies to ensure we have complete reference information
        for sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
            self._analyze_sheet(sheet, sheet_name)

        # Finalize graph structure
        self.graph.finalize_dependencies()

        # CLAUDE-GOTCHA: Excel allows some circular references with iterative calculation
        # but they can cause performance issues and calculation errors
        circular_refs = self.graph.detect_circular_references()

        # Calculate depths
        self.graph.calculate_depths()

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score()

        # Create analysis result
        return FormulaAnalysis(
            dependency_graph=dict(self.graph.nodes),
            circular_references=tuple(circular_refs),
            volatile_formulas=tuple(set(self.volatile_formulas)),
            external_references=tuple(set(self.external_references)),
            max_dependency_depth=self.graph.get_max_depth(),
            formula_complexity_score=complexity_score,
        )

    def _analyze_sheet(self, sheet, sheet_name: str):
        """Analyze all formulas in a sheet."""
        parser = self._get_parser(sheet_name)

        # Iterate through cells
        for row in sheet.iter_rows():
            for cell in row:
                # Check if cell contains a formula
                if hasattr(cell, "data_type") and cell.data_type == "f" and cell.value:
                    self._analyze_formula(cell, sheet_name, parser)

    def _analyze_formula(self, cell, sheet_name: str, parser: FormulaParser):
        """Analyze a single formula."""
        # When data_only=False, formula is stored in cell.value
        formula = str(cell.value) if cell.value else ""
        cell_ref = cell.coordinate
        cell_key = f"{sheet_name}!{cell_ref}"

        # Add node to graph
        self.graph.add_node(cell_key, sheet_name, cell_ref, formula)

        # Check for volatile functions
        if any(func in formula.upper() for func in self.VOLATILE_FUNCTIONS):
            self.volatile_formulas.append(cell_key)

        # Check for external references
        if "[" in formula and "]" in formula:
            self.external_references.append(cell_key)

        # Parse dependencies
        dependencies = parser.parse_formula(formula)

        # Add edges to graph
        for dep in dependencies:
            dep_key = f"{dep.sheet}!{dep.cell}"
            # Only add edge if it's a single cell reference
            if not dep.is_range:
                self.graph.add_edge(cell_key, dep_key)

    def _get_parser(self, sheet_name: str) -> FormulaParser:
        """Get or create parser for sheet."""
        if sheet_name not in self.parser_cache:
            self.parser_cache[sheet_name] = FormulaParser(sheet_name)
        return self.parser_cache[sheet_name]

    def _calculate_complexity_score(self) -> int:
        """
        Calculate formula complexity score (0-100).

        Considers:
        - Number of formulas
        - Dependency depth
        - Circular references
        - Volatile functions
        - External references
        """
        score = 0

        # Base score from formula count
        formula_count = len(self.graph.nodes)
        if formula_count > HIGH_FORMULA_COUNT:
            score += 20
        elif formula_count > MEDIUM_FORMULA_COUNT:
            score += 10
        elif formula_count > LOW_FORMULA_COUNT:
            score += 5

        # Dependency depth score
        max_depth = self.graph.get_max_depth()
        if max_depth > HIGH_DEPTH_THRESHOLD:
            score += 30
        elif max_depth > MEDIUM_DEPTH_THRESHOLD:
            score += 20
        elif max_depth > LOW_DEPTH_THRESHOLD:
            score += 10

        # Circular reference penalty
        if self.graph._circular_refs:  # noqa: SLF001
            score += 20

        # Volatile function penalty
        volatile_ratio = len(self.volatile_formulas) / max(formula_count, 1)
        score += int(volatile_ratio * 20)

        # External reference penalty
        if self.external_references:
            score += 10

        return min(100, score)


# ==================== Main Stage Function ====================


def stage_3_formula_analysis(file_path: Path, *, read_only: bool = True) -> Result:
    """
    Perform formula dependency analysis.

    Args:
        file_path: Path to Excel file
        read_only: Whether to open in read-only mode

    Returns:
        Ok(FormulaAnalysis) if analysis succeeds
        Err(error_message) if analysis fails
    """
    try:
        # Open workbook
        # CLAUDE-IMPORTANT: We need data_only=False to get formulas
        workbook = openpyxl.load_workbook(
            filename=str(file_path),
            read_only=read_only,
            keep_vba=False,
            data_only=False,  # Need formulas, not values
            keep_links=False,
        )

        try:
            # Create analyzer and run analysis
            analyzer = FormulaAnalyzer()
            analysis = analyzer.analyze_workbook(workbook)

            return Ok(analysis)

        finally:
            workbook.close()

    except (OSError, ValueError, TypeError, MemoryError) as e:
        return Err(f"Formula analysis failed: {e!s}", {"exception": str(e)})


# ==================== Utility Functions ====================


def create_formula_validator(
    max_depth: int = 10, *, allow_circular: bool = False, allow_volatile: bool = True, allow_external: bool = False
) -> Callable[[Path], list[str]]:
    """
    Create a formula validator with specific policies.
    """

    def validator(file_path: Path) -> list[str]:
        """Validate formulas and return issues."""
        issues = []

        # Run formula analysis
        result = stage_3_formula_analysis(file_path)

        if isinstance(result, Err):
            issues.append(f"Formula analysis failed: {result.error}")
            return issues

        analysis = result.value

        # Check depth
        if analysis.max_dependency_depth > max_depth:
            issues.append(f"Formula dependency too deep: {analysis.max_dependency_depth} (limit: {max_depth})")

        # Check circular references
        if not allow_circular and analysis.has_circular_references:
            issues.append(f"Circular references detected: {len(analysis.circular_references)} cycles")

        # Check volatile functions
        if not allow_volatile and analysis.volatile_formulas:
            issues.append(f"Volatile functions detected: {len(analysis.volatile_formulas)} formulas")

        # Check external references
        if not allow_external and analysis.external_references:
            issues.append(f"External references detected: {len(analysis.external_references)} references")

        return issues

    return validator
