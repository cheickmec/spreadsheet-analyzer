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

# Type imports for optional features
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal

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

# Performance settings
FORMULA_ANALYSIS_CHUNK_SIZE: Final[int] = 1000  # Process 1000 rows at a time

# Range handling settings
SMALL_RANGE_THRESHOLD: Final[int] = 10  # Expand ranges smaller than this
MEDIUM_RANGE_THRESHOLD: Final[int] = 100  # Create range nodes for larger ranges
RANGE_HANDLING_MODE: Final[str] = "smart"  # "skip" | "expand" | "summarize" | "smart"
ENABLE_RANGE_MEMBERSHIP_INDEX: Final[bool] = True  # Track empty cell memberships

# Complexity score weights
SCORE_HIGH_FORMULA_COUNT: Final[int] = 20
SCORE_MEDIUM_FORMULA_COUNT: Final[int] = 10
SCORE_LOW_FORMULA_COUNT: Final[int] = 5

SCORE_HIGH_DEPTH: Final[int] = 30
SCORE_MEDIUM_DEPTH: Final[int] = 20
SCORE_LOW_DEPTH: Final[int] = 10

SCORE_CIRCULAR_REFERENCE_PENALTY: Final[int] = 20
SCORE_VOLATILE_FUNCTION_WEIGHT: Final[int] = 20
SCORE_EXTERNAL_REFERENCE_PENALTY: Final[int] = 10

MAX_COMPLEXITY_SCORE: Final[int] = 100

# ==================== Helper Functions ====================


def format_cell_key(sheet_name: str, cell_ref: str) -> str:
    """
    Format a standardized cell key from sheet and cell reference.

    Args:
        sheet_name: Name of the sheet
        cell_ref: Cell reference (e.g., "A1", "B2:C3")

    Returns:
        Formatted cell key (e.g., "Sheet1!A1")
    """
    return f"{sheet_name}!{cell_ref}"


def parse_range_size(range_ref: str) -> tuple[int, dict[str, Any]]:
    """
    Calculate the size of a range reference and extract metadata.

    Args:
        range_ref: Range reference (e.g., "B1:B10", "A:A", "1:1")

    Returns:
        Tuple of (size, metadata_dict)
    """
    import re

    from openpyxl.utils import column_index_from_string

    # Handle full column ranges (e.g., "A:A", "B:D")
    if re.match(r"^[A-Z]+:[A-Z]+$", range_ref):
        return (1048576, {"type": "column", "full_column": True})
    # Handle full row ranges (e.g., "1:1", "5:10")
    if re.match(r"^\d+:\d+$", range_ref):
        return (16384, {"type": "row", "full_row": True})

    # Parse normal range (e.g., "B1:D10")
    match = re.match(r"^([A-Z]+)(\d+):([A-Z]+)(\d+)$", range_ref)
    if match:
        start_col, start_row, end_col, end_row = match.groups()

        col_start = column_index_from_string(start_col)
        col_end = column_index_from_string(end_col)
        row_start = int(start_row)
        row_end = int(end_row)

        num_cols = col_end - col_start + 1
        num_rows = row_end - row_start + 1
        size = num_cols * num_rows

        metadata = {
            "type": "normal",
            "start_cell": f"{start_col}{start_row}",
            "end_cell": f"{end_col}{end_row}",
            "rows": num_rows,
            "cols": num_cols,
            "is_column_range": num_cols == 1 and num_rows > 1,
            "is_row_range": num_rows == 1 and num_cols > 1,
        }

        return (size, metadata)

    # Default for unparseable ranges
    return (1, {"type": "unknown"})


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

        # Try to use openpyxl's tokenizer - it requires the = sign
        try:
            tokenizer = Tokenizer(formula)  # Keep the = sign for proper parsing
            for token in tokenizer.items:
                if token.type == "OPERAND" and token.subtype == "RANGE":
                    refs = self._parse_reference(token.value)
                    references.update(refs)
        except (ValueError, AttributeError, TypeError):
            # Fallback to regex parsing if tokenizer fails
            # Remove the leading '=' for regex parsing
            formula_without_eq = formula[1:]
            references.update(self._parse_with_regex(formula_without_eq))

        return references

    def _parse_reference(self, ref_str: str) -> set[CellReference]:
        """Parse a single reference string from tokenizer output."""
        references: set[CellReference] = set()

        # Extract sheet name if present
        sheet = self.current_sheet
        cell_part = ref_str

        # Handle quoted sheet names like 'Sheet 2'!B1
        if ref_str.startswith("'") and "'!" in ref_str:
            sheet_end = ref_str.index("'!")
            sheet = ref_str[1:sheet_end]  # Remove quotes
            cell_part = ref_str[sheet_end + 2 :]  # Skip '!
        # Handle unquoted sheet names like TestSheet!A1:A10
        elif "!" in ref_str and not ref_str.startswith("'"):
            sheet, cell_part = ref_str.split("!", 1)

        # Check for range reference
        if ":" in cell_part:
            # This is a range like A1:A10
            references.add(CellReference(sheet=sheet, cell=cell_part, is_absolute="$" in ref_str, is_range=True))
        else:
            # Single cell reference
            references.add(CellReference(sheet=sheet, cell=cell_part, is_absolute="$" in ref_str, is_range=False))

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
    CLAUDE-IMPORTANT: Now supports range nodes as first-class citizens
    to handle range dependencies without edge explosion.
    """

    def __init__(self):
        """Initialize empty graph."""
        self.nodes: set[str] = set()  # All node keys
        self.edges: dict[str, set[str]] = defaultdict(set)
        self.reverse_edges: dict[str, set[str]] = defaultdict(set)
        self.formulas: dict[str, str] = {}  # node_key -> formula
        self.node_types: dict[str, str] = {}  # node_key -> "cell" | "range"
        self.range_metadata: dict[str, dict] = {}  # node_key -> range info
        self.node_depths: dict[str, int] = {}  # node_key -> depth
        self._circular_refs: list[tuple[str, ...]] = []

    def add_node(
        self,
        node_key: str,
        _sheet: str,
        _cell_ref: str,
        formula: str | None,
        *,
        node_type: str = "cell",
        range_metadata: dict | None = None,
    ):
        """Add a node (cell or range) to the graph."""
        if node_key not in self.nodes:
            self.nodes.add(node_key)
            self.node_types[node_key] = node_type
            if formula:
                self.formulas[node_key] = formula
            if range_metadata:
                self.range_metadata[node_key] = range_metadata

    def add_edge(self, from_node: str, to_node: str):
        """Add dependency edge from from_node to to_node."""
        # Ensure both nodes exist
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        # Add edges
        self.edges[from_node].add(to_node)
        self.reverse_edges[to_node].add(from_node)

    def finalize(self) -> dict[str, FormulaNode]:
        """
        Convert internal representation to immutable FormulaNodes.

        CLAUDE-IMPORTANT: This creates the final output structure
        with proper handling of range nodes.
        """
        formula_nodes = {}

        for node_key in self.nodes:
            # Parse node key to get sheet and cell
            if "!" in node_key:
                sheet, cell_ref = node_key.split("!", 1)
            else:
                sheet = ""
                cell_ref = node_key

            # Get node info
            node_type_str = self.node_types.get(node_key, "cell")
            # Ensure node_type is a proper literal type
            node_type: Literal["cell", "range"] = "range" if node_type_str == "range" else "cell"
            formula = self.formulas.get(node_key, "")
            depth = self.node_depths.get(node_key, 0)

            # Convert dependencies to CellReference objects
            dependencies = frozenset(
                CellReference(
                    sheet=dep.split("!")[0] if "!" in dep else sheet,
                    cell=dep.split("!")[1] if "!" in dep else dep,
                    is_absolute=False,
                    is_range=self.node_types.get(dep, "cell") == "range",
                )
                for dep in self.edges.get(node_key, set())
            )

            # Convert dependents to CellReference objects
            dependents = frozenset(
                CellReference(
                    sheet=dep.split("!")[0] if "!" in dep else sheet,
                    cell=dep.split("!")[1] if "!" in dep else dep,
                    is_absolute=False,
                    is_range=self.node_types.get(dep, "cell") == "range",
                )
                for dep in self.reverse_edges.get(node_key, set())
            )

            # Create FormulaNode with range info if applicable
            formula_nodes[node_key] = FormulaNode(
                sheet=sheet,
                cell=cell_ref,
                formula=formula,
                dependencies=dependencies,
                dependents=dependents,
                depth=depth,
                node_type=node_type,
                range_info=self.range_metadata.get(node_key),
            )

        return formula_nodes

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

        # Store depths for use in finalize
        self.node_depths = depths

    def get_max_depth(self) -> int:
        """Get maximum dependency depth."""
        if not self.node_depths:
            return 0
        return max(self.node_depths.values(), default=0)


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
        # Range membership tracking for empty cell queries
        if TYPE_CHECKING:
            from spreadsheet_analyzer.graph_db.range_membership import RangeMembershipIndex

            self.range_index: RangeMembershipIndex | None
        self.range_index = None

        if ENABLE_RANGE_MEMBERSHIP_INDEX:
            from spreadsheet_analyzer.graph_db.range_membership import RangeMembershipIndex

            self.range_index = RangeMembershipIndex()

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

        # Calculate depths first
        self.graph.calculate_depths()

        # Detect circular references
        # CLAUDE-GOTCHA: Excel allows some circular references with iterative calculation
        # but they can cause performance issues and calculation errors
        circular_refs = self.graph.detect_circular_references()

        # Finalize graph structure to create immutable nodes
        formula_nodes = self.graph.finalize()

        # Calculate complexity score
        complexity_score = self._calculate_complexity_score()

        # Create analysis result
        return FormulaAnalysis(
            dependency_graph=formula_nodes,
            circular_references=tuple(circular_refs),
            volatile_formulas=tuple(set(self.volatile_formulas)),
            external_references=tuple(set(self.external_references)),
            max_dependency_depth=self.graph.get_max_depth(),
            formula_complexity_score=complexity_score,
            range_membership_index=self.range_index,
        )

    def _analyze_sheet(self, sheet, sheet_name: str):
        """
        Analyze all formulas in a sheet.

        CLAUDE-PERFORMANCE: Process formulas in chunks to avoid memory issues
        with large spreadsheets.
        """
        parser = self._get_parser(sheet_name)

        # Process in chunks to avoid memory issues
        min_row = sheet.min_row
        max_row = sheet.max_row

        for start_row in range(min_row, max_row + 1, FORMULA_ANALYSIS_CHUNK_SIZE):
            end_row = min(start_row + FORMULA_ANALYSIS_CHUNK_SIZE - 1, max_row)

            # Iterate through cells in this chunk
            for row in sheet.iter_rows(min_row=start_row, max_row=end_row):
                for cell in row:
                    # Check if cell contains a formula
                    if hasattr(cell, "data_type") and cell.data_type == "f" and cell.value:
                        self._analyze_formula(cell, sheet_name, parser)

    def _analyze_formula(self, cell, sheet_name: str, parser: FormulaParser):
        """Analyze a single formula."""
        # When data_only=False, formula is stored in cell.value
        formula = str(cell.value) if cell.value else ""
        cell_ref = cell.coordinate
        cell_key = format_cell_key(sheet_name, cell_ref)

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

        # Add edges to graph with range handling
        for dep in dependencies:
            dep_key = format_cell_key(dep.sheet, dep.cell)

            if not dep.is_range:
                # Single cell reference - add direct edge
                self.graph.add_edge(cell_key, dep_key)
            else:
                # Range reference - handle based on size and configuration
                self._handle_range_dependency(cell_key, dep, dep_key)

    def _handle_range_dependency(self, formula_key: str, dep: CellReference, range_key: str):
        """Handle range dependencies based on size and configuration."""
        # Calculate range size and metadata
        range_size, range_metadata = parse_range_size(dep.cell)

        # Add to range membership index for empty cell tracking
        if self.range_index:
            self.range_index.add_range(dep.cell, formula_key, dep.sheet)

        # Determine handling strategy
        if RANGE_HANDLING_MODE == "skip":
            # Don't add any edges for ranges
            return

        elif RANGE_HANDLING_MODE == "expand" and range_size <= SMALL_RANGE_THRESHOLD:
            # Expand small ranges to individual cell edges
            self._expand_range_to_cells(formula_key, dep.sheet, dep.cell)

        elif RANGE_HANDLING_MODE == "smart":
            # Smart handling based on range size
            if range_size <= SMALL_RANGE_THRESHOLD:
                # Small range - expand to individual cells
                self._expand_range_to_cells(formula_key, dep.sheet, dep.cell)
            else:
                # Medium/large range - create range node
                self.graph.add_node(
                    range_key,
                    dep.sheet,
                    dep.cell,
                    None,
                    node_type="range",
                    range_metadata=range_metadata,
                )
                self.graph.add_edge(formula_key, range_key)

        else:  # "summarize" or default
            # Always create range nodes
            self.graph.add_node(
                range_key,
                dep.sheet,
                dep.cell,
                None,
                node_type="range",
                range_metadata=range_metadata,
            )
            self.graph.add_edge(formula_key, range_key)

    def _expand_range_to_cells(self, formula_key: str, sheet: str, range_ref: str):
        """Expand a range reference to individual cell edges."""
        import re

        from openpyxl.utils import column_index_from_string, get_column_letter

        # Parse range (e.g., "B1:D3")
        match = re.match(r"^([A-Z]+)(\d+):([A-Z]+)(\d+)$", range_ref)
        if match:
            start_col, start_row, end_col, end_row = match.groups()
            col_start = column_index_from_string(start_col)
            col_end = column_index_from_string(end_col)
            row_start = int(start_row)
            row_end = int(end_row)

            # Create edges to each cell in range
            for row in range(row_start, row_end + 1):
                for col_idx in range(col_start, col_end + 1):
                    col_letter = get_column_letter(col_idx)
                    cell_ref = f"{col_letter}{row}"
                    cell_key = format_cell_key(sheet, cell_ref)
                    self.graph.add_edge(formula_key, cell_key)

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
            score += SCORE_HIGH_FORMULA_COUNT
        elif formula_count > MEDIUM_FORMULA_COUNT:
            score += SCORE_MEDIUM_FORMULA_COUNT
        elif formula_count > LOW_FORMULA_COUNT:
            score += SCORE_LOW_FORMULA_COUNT

        # Dependency depth score
        max_depth = self.graph.get_max_depth()
        if max_depth > HIGH_DEPTH_THRESHOLD:
            score += SCORE_HIGH_DEPTH
        elif max_depth > MEDIUM_DEPTH_THRESHOLD:
            score += SCORE_MEDIUM_DEPTH
        elif max_depth > LOW_DEPTH_THRESHOLD:
            score += SCORE_LOW_DEPTH

        # Circular reference penalty
        if self.graph._circular_refs:  # noqa: SLF001
            score += SCORE_CIRCULAR_REFERENCE_PENALTY

        # Volatile function penalty
        volatile_ratio = len(self.volatile_formulas) / max(formula_count, 1)
        score += int(volatile_ratio * SCORE_VOLATILE_FUNCTION_WEIGHT)

        # External reference penalty
        if self.external_references:
            score += SCORE_EXTERNAL_REFERENCE_PENALTY

        return min(MAX_COMPLEXITY_SCORE, score)


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
