"""Hierarchical exploration strategy for multi-level summarization.

This strategy implements a progressive disclosure approach, starting with
high-level patterns and diving into details based on discovered insights.
"""

import logging
from collections import defaultdict
from typing import Any

from spreadsheet_analyzer.notebook_llm.nap.protocols import CellType, NotebookDocument

from .base import (
    AnalysisFocus,
    AnalysisTask,
    BaseStrategy,
    ContextPackage,
    ResponseFormat,
)

logger = logging.getLogger(__name__)


class HierarchicalStrategy(BaseStrategy):
    """Strategy for hierarchical exploration and summarization.

    This strategy analyzes notebooks at multiple levels:
    1. Full notebook summary
    2. Section/group summaries
    3. Key cell details

    It's particularly effective for:
    - Initial exploration of unknown spreadsheets
    - Understanding overall structure and purpose
    - Identifying areas that need deeper analysis
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the hierarchical strategy.

        Args:
            config: Configuration options including:
                - summarization_algorithm: 'extractive' or 'abstractive'
                - compression_ratio: Target compression ratio (0.0-1.0)
                - section_detection_method: 'semantic' or 'structural'
                - similarity_threshold: Threshold for grouping similar cells
                - max_sections: Maximum number of sections to create
        """
        super().__init__(config)

        # Configuration defaults
        cfg = self.config  # Use self.config which is set by parent class
        self.summarization_algorithm = cfg.get("summarization_algorithm", "extractive")
        self.compression_ratio = cfg.get("compression_ratio", 0.1)
        self.section_detection_method = cfg.get("section_detection_method", "structural")
        self.similarity_threshold = cfg.get("similarity_threshold", 0.7)
        self.max_sections = cfg.get("max_sections", 10)

    def prepare_context(self, notebook: NotebookDocument, focus: AnalysisFocus, token_budget: int) -> ContextPackage:
        """Prepare hierarchical context for the notebook.

        Args:
            notebook: The notebook to analyze
            focus: The analysis focus area
            token_budget: Maximum tokens to use

        Returns:
            Context package with hierarchical structure
        """
        logger.info(
            f"Preparing hierarchical context for notebook with {len(notebook.cells)} cells, "
            f"focus: {focus.value}, budget: {token_budget} tokens"
        )

        # Create hierarchical structure
        hierarchy = self._build_hierarchy(notebook, focus)

        # Allocate token budget across levels
        budget_allocation = self._allocate_budget(hierarchy, token_budget)

        # Create compressed representation
        compressed_hierarchy = self._compress_hierarchy(notebook, hierarchy, budget_allocation)

        # Extract metadata
        metadata = self._extract_metadata(notebook, hierarchy)

        # Generate focus hints
        focus_hints = self._generate_focus_hints(focus, hierarchy)

        # Count tokens (simplified - would use actual tokenizer in production)
        token_count = self._estimate_tokens(compressed_hierarchy)

        return ContextPackage(
            cells=compressed_hierarchy["cells"],
            metadata=metadata,
            focus_hints=focus_hints,
            token_count=token_count,
            compression_method="hierarchical",
            additional_data={
                "hierarchy": compressed_hierarchy,
                "sections": compressed_hierarchy.get("sections", []),
                "summary": compressed_hierarchy.get("summary", ""),
            },
        )

    def format_prompt(self, context: ContextPackage, task: AnalysisTask) -> str:
        """Format hierarchical prompt for LLM.

        Args:
            context: Prepared context package
            task: Analysis task to perform

        Returns:
            Formatted prompt string
        """
        # Build prompt with hierarchical structure
        prompt_parts = []

        # System context
        prompt_parts.append(
            "You are analyzing a Jupyter notebook containing Excel spreadsheet analysis. "
            "You excel at hierarchical data exploration, starting with high-level patterns "
            "and progressively diving into details based on discovered insights."
        )

        # Task description
        prompt_parts.append(f"\nTask: {task.description}")
        if task.focus_area:
            prompt_parts.append(f"Focus Area: {task.focus_area}")

        # Hierarchical context presentation
        prompt_parts.append("\n## Hierarchical Overview")

        # Level 1: Notebook summary
        summary = context.additional_data.get("summary", "")
        if summary:
            prompt_parts.append(f"\n### Level 1: Notebook Summary\n{summary}")

        # Level 2: Section summaries
        sections = context.additional_data.get("sections", [])
        if sections:
            prompt_parts.append("\n### Level 2: Section Summaries")
            for i, section in enumerate(sections, 1):
                prompt_parts.append(
                    f"\n#### Section {i}: {section.get('name', 'Unnamed')}\n"
                    f"{section.get('summary', '')}\n"
                    f"Cells: {section.get('cell_count', 0)}"
                )

        # Level 3: Key cells
        if context.cells:
            prompt_parts.append("\n### Level 3: Key Cells")
            for cell in context.cells[:10]:  # Limit to top 10 cells
                cell_type = cell.get("cell_type", "unknown")
                prompt_parts.append(
                    f"\n**Cell {cell.get('id', '?')} ({cell_type})**\n```\n{cell.get('source', '')[:200]}...\n```"
                )

        # Focus hints
        if context.focus_hints:
            prompt_parts.append("\n## Analysis Hints")
            for hint in context.focus_hints:
                prompt_parts.append(f"- {hint}")

        # Expected output format
        prompt_parts.append(f"\n## Expected Output Format: {task.expected_format.value}")

        if task.expected_format == ResponseFormat.JSON:
            prompt_parts.append(
                "\nProvide your analysis as a JSON object with the following structure:\n"
                "```json\n"
                "{\n"
                '  "summary": "Overall analysis summary",\n'
                '  "key_findings": ["finding1", "finding2", ...],\n'
                '  "areas_of_interest": [{"name": "area", "description": "...", "priority": "high/medium/low"}],\n'
                '  "recommendations": ["recommendation1", "recommendation2", ...]\n'
                "}\n"
                "```"
            )

        return "\n".join(prompt_parts)

    def _build_hierarchy(self, notebook: NotebookDocument, focus: AnalysisFocus) -> dict[str, Any]:
        """Build hierarchical representation of the notebook.

        Args:
            notebook: The notebook to analyze
            focus: The analysis focus

        Returns:
            Hierarchical structure dictionary
        """
        hierarchy = {
            "notebook_id": notebook.metadata.get("id", "unknown"),
            "total_cells": len(notebook.cells),
            "sections": [],
            "key_cells": [],
            "summary": "",
        }

        # Identify sections based on method
        if self.section_detection_method == "structural":
            sections = self._detect_structural_sections(notebook)
        else:
            sections = self._detect_semantic_sections(notebook)

        hierarchy["sections"] = sections

        # Identify key cells based on focus
        hierarchy["key_cells"] = self._identify_key_cells(notebook, focus)

        # Generate notebook summary
        hierarchy["summary"] = self._generate_notebook_summary(notebook, sections)

        return hierarchy

    def _detect_structural_sections(self, notebook: NotebookDocument) -> list[dict[str, Any]]:
        """Detect sections based on structural patterns.

        Looks for:
        - Markdown headers
        - Cell type transitions
        - Empty cells as separators

        Args:
            notebook: The notebook to analyze

        Returns:
            List of detected sections
        """
        sections: list[dict[str, Any]] = []
        current_section = None
        current_cells: list[int] = []

        for i, cell in enumerate(notebook.cells):
            # Check if this cell starts a new section
            is_section_start = False
            section_name = None

            if cell.cell_type == CellType.MARKDOWN:
                # Look for headers
                source = cell.source.strip()
                if source.startswith("#"):
                    is_section_start = True
                    # Extract header text
                    section_name = source.split("\n")[0].lstrip("#").strip()

            if is_section_start:
                # Save previous section if exists
                if current_section is not None:
                    sections.append(
                        {
                            "name": current_section,
                            "start_index": current_cells[0] if current_cells else 0,
                            "end_index": i - 1,
                            "cells": current_cells,
                            "cell_count": len(current_cells),
                        }
                    )

                # Start new section
                current_section = section_name or f"Section {len(sections) + 1}"
                current_cells = []

            # Add cell to current section
            if current_section is not None:
                current_cells.append(i)
            elif not sections:
                # If no sections yet, create initial section
                current_section = "Introduction"
                current_cells = [i]

        # Save final section
        if current_section is not None and current_cells:
            sections.append(
                {
                    "name": current_section,
                    "start_index": current_cells[0],
                    "end_index": len(notebook.cells) - 1,
                    "cells": current_cells,
                    "cell_count": len(current_cells),
                }
            )

        # Limit to max sections
        if len(sections) > self.max_sections:
            # Merge smaller sections
            sections = self._merge_sections(sections, self.max_sections)

        return sections

    def _detect_semantic_sections(self, notebook: NotebookDocument) -> list[dict[str, Any]]:
        """Detect sections based on semantic similarity.

        Groups cells with similar content or purpose.

        Args:
            notebook: The notebook to analyze

        Returns:
            List of detected sections
        """
        # Simplified implementation - would use embeddings in production
        sections: list[dict[str, Any]] = []

        # Group by cell type as a simple semantic grouping
        type_groups = defaultdict(list)
        for i, cell in enumerate(notebook.cells):
            type_groups[cell.cell_type].append(i)

        # Create sections from groups
        for cell_type, indices in type_groups.items():
            if indices:
                sections.append(
                    {
                        "name": f"{cell_type.value.title()} Cells",
                        "start_index": min(indices),
                        "end_index": max(indices),
                        "cells": indices,
                        "cell_count": len(indices),
                    }
                )

        return sorted(sections, key=lambda s: s["start_index"])

    def _identify_key_cells(self, notebook: NotebookDocument, focus: AnalysisFocus) -> list[int]:
        """Identify the most important cells based on focus.

        Args:
            notebook: The notebook to analyze
            focus: The analysis focus

        Returns:
            List of cell indices for key cells
        """
        key_cells = []

        for i, cell in enumerate(notebook.cells):
            importance_score = 0

            # Score based on content indicators
            source_lower = cell.source.lower()

            # Focus-specific scoring
            if focus == AnalysisFocus.FORMULAS:
                if "formula" in source_lower or "=" in cell.source:
                    importance_score += 3
                if "calculate" in source_lower or "compute" in source_lower:
                    importance_score += 2

            elif focus == AnalysisFocus.STRUCTURE:
                if cell.cell_type == CellType.MARKDOWN and "#" in cell.source:
                    importance_score += 3
                if "structure" in source_lower or "layout" in source_lower:
                    importance_score += 2

            elif focus == AnalysisFocus.DATA_VALIDATION:
                if "valid" in source_lower or "check" in source_lower:
                    importance_score += 3
                if "error" in source_lower or "warning" in source_lower:
                    importance_score += 2

            # General importance indicators
            if "important" in source_lower or "key" in source_lower:
                importance_score += 2
            if "summary" in source_lower or "result" in source_lower:
                importance_score += 1
            if cell.cell_type == CellType.CODE and len(cell.source) > 100:
                importance_score += 1

            if importance_score > 0:
                key_cells.append((i, importance_score))

        # Sort by importance and return top cells
        key_cells.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in key_cells[:20]]  # Top 20 cells

    def _generate_notebook_summary(self, notebook: NotebookDocument, sections: list[dict[str, Any]]) -> str:
        """Generate a high-level summary of the notebook.

        Args:
            notebook: The notebook to analyze
            sections: Detected sections

        Returns:
            Summary text
        """
        # Basic statistics
        total_cells = len(notebook.cells)
        code_cells = sum(1 for c in notebook.cells if c.cell_type == CellType.CODE)
        markdown_cells = sum(1 for c in notebook.cells if c.cell_type == CellType.MARKDOWN)

        summary_parts = [
            f"Notebook contains {total_cells} cells ({code_cells} code, {markdown_cells} markdown).",
            f"Organized into {len(sections)} sections.",
        ]

        # Add section overview
        if sections:
            section_names = [s["name"] for s in sections[:5]]  # First 5 sections
            if len(sections) > 5:
                section_names.append(f"and {len(sections) - 5} more...")
            summary_parts.append(f"Sections: {', '.join(section_names)}")

        # Look for key patterns
        has_data_loading = any(
            "read" in c.source.lower() or "load" in c.source.lower()
            for c in notebook.cells
            if c.cell_type == CellType.CODE
        )
        has_visualization = any(
            "plot" in c.source.lower() or "chart" in c.source.lower()
            for c in notebook.cells
            if c.cell_type == CellType.CODE
        )
        has_formulas = any("formula" in c.source.lower() or "=" in c.source for c in notebook.cells)

        if has_data_loading:
            summary_parts.append("Contains data loading operations.")
        if has_visualization:
            summary_parts.append("Includes data visualization.")
        if has_formulas:
            summary_parts.append("Analyzes Excel formulas.")

        return " ".join(summary_parts)

    def _allocate_budget(self, hierarchy: dict[str, Any], token_budget: int) -> dict[str, int]:
        """Allocate token budget across hierarchy levels.

        Args:
            hierarchy: The hierarchical structure
            token_budget: Total token budget

        Returns:
            Budget allocation dictionary
        """
        # Default allocation strategy
        allocation = {
            "summary": int(token_budget * 0.1),  # 10% for summary
            "sections": int(token_budget * 0.3),  # 30% for sections
            "cells": int(token_budget * 0.5),  # 50% for key cells
            "metadata": int(token_budget * 0.1),  # 10% for metadata
        }

        # Adjust based on structure
        if len(hierarchy["sections"]) == 0:
            # No sections, allocate more to cells
            allocation["cells"] += allocation["sections"]
            allocation["sections"] = 0
        elif len(hierarchy["sections"]) > 10:
            # Many sections, allocate more to section summaries
            allocation["sections"] = int(token_budget * 0.4)
            allocation["cells"] = int(token_budget * 0.4)

        return allocation

    def _compress_hierarchy(
        self, notebook: NotebookDocument, hierarchy: dict[str, Any], budget_allocation: dict[str, int]
    ) -> dict[str, Any]:
        """Compress hierarchy to fit within token budget.

        Args:
            notebook: The notebook document
            hierarchy: The full hierarchy
            budget_allocation: Token budget for each level

        Returns:
            Compressed hierarchy
        """
        compressed = {"summary": hierarchy["summary"][: budget_allocation["summary"]], "sections": [], "cells": []}

        # Compress sections
        section_budget_per = budget_allocation["sections"] // max(len(hierarchy["sections"]), 1)
        for section in hierarchy["sections"]:
            compressed_section = {
                "name": section["name"],
                "cell_count": section["cell_count"],
                "summary": self._summarize_section(section, section_budget_per),
            }
            compressed["sections"].append(compressed_section)

        # Select and compress key cells
        cell_budget_per = budget_allocation["cells"] // max(len(hierarchy["key_cells"]), 1)
        for cell_idx in hierarchy["key_cells"]:
            if cell_idx < len(notebook.cells):
                cell = notebook.cells[cell_idx]
                compressed_cell = {
                    "id": cell_idx,
                    "cell_type": cell.cell_type.value,
                    "source": cell.source[:cell_budget_per],
                }
                compressed["cells"].append(compressed_cell)

        return compressed

    def _summarize_section(self, section: dict[str, Any], token_budget: int) -> str:
        """Generate summary for a section.

        Args:
            section: Section information
            token_budget: Token budget for this section

        Returns:
            Section summary
        """
        # Simplified implementation
        summary = f"Section '{section['name']}' contains {section['cell_count']} cells"

        # Would implement actual summarization logic here
        # For now, just truncate to budget
        return summary[:token_budget]

    def _merge_sections(self, sections: list[dict[str, Any]], target_count: int) -> list[dict[str, Any]]:
        """Merge sections to reach target count.

        Args:
            sections: Original sections
            target_count: Target number of sections

        Returns:
            Merged sections
        """
        if len(sections) <= target_count:
            return sections

        # Simple merging strategy - merge adjacent small sections
        merged: list[dict[str, Any]] = []
        i = 0

        while i < len(sections) and len(merged) < target_count:
            if (
                i + 1 < len(sections)
                and len(merged) < target_count - 1
                and sections[i]["cell_count"] < 5
                and sections[i + 1]["cell_count"] < 5
            ):
                # Merge small adjacent sections
                merged_section = {
                    "name": f"{sections[i]['name']} & {sections[i + 1]['name']}",
                    "start_index": sections[i]["start_index"],
                    "end_index": sections[i + 1]["end_index"],
                    "cells": sections[i]["cells"] + sections[i + 1]["cells"],
                    "cell_count": sections[i]["cell_count"] + sections[i + 1]["cell_count"],
                }
                merged.append(merged_section)
                i += 2
                continue

            merged.append(sections[i])
            i += 1

        return merged

    def _extract_metadata(self, notebook: NotebookDocument, hierarchy: dict[str, Any]) -> dict[str, Any]:
        """Extract relevant metadata from the notebook.

        Args:
            notebook: The notebook document
            hierarchy: The hierarchical structure

        Returns:
            Metadata dictionary
        """
        metadata = {
            "total_cells": len(notebook.cells),
            "sections": len(hierarchy["sections"]),
            "key_cells": len(hierarchy["key_cells"]),
            "notebook_metadata": notebook.metadata,
        }

        # Cell type distribution
        cell_types: defaultdict[Any, int] = defaultdict(int)
        for cell in notebook.cells:
            cell_types[cell.cell_type.value] += 1
        metadata["cell_type_distribution"] = dict(cell_types)

        # Average cell size
        if notebook.cells:
            avg_size = sum(len(c.source) for c in notebook.cells) / len(notebook.cells)
            metadata["average_cell_size"] = int(avg_size)

        return metadata

    def _generate_focus_hints(self, focus: AnalysisFocus, hierarchy: dict[str, Any]) -> list[str]:
        """Generate analysis hints based on focus and structure.

        Args:
            focus: The analysis focus
            hierarchy: The hierarchical structure

        Returns:
            List of analysis hints
        """
        hints = []

        # Focus-specific hints
        if focus == AnalysisFocus.FORMULAS:
            hints.append("Pay special attention to cells containing Excel formulas and calculations")
            hints.append("Look for formula dependencies and circular references")
            hints.append("Identify complex formulas that might need simplification")

        elif focus == AnalysisFocus.STRUCTURE:
            hints.append("Analyze the overall organization and flow of the notebook")
            hints.append("Identify the main sections and their purposes")
            hints.append("Look for structural patterns or inconsistencies")

        elif focus == AnalysisFocus.DATA_VALIDATION:
            hints.append("Focus on data validation rules and constraints")
            hints.append("Identify potential data quality issues")
            hints.append("Look for validation formulas and error checking")

        elif focus == AnalysisFocus.RELATIONSHIPS:
            hints.append("Map relationships between different data elements")
            hints.append("Identify key dependencies and data flows")
            hints.append("Look for linked data across sheets or cells")

        # Structure-based hints
        if len(hierarchy["sections"]) > 10:
            hints.append("This notebook has many sections - consider focusing on the most relevant ones")

        if len(hierarchy["key_cells"]) > 0:
            hints.append(f"Identified {len(hierarchy['key_cells'])} key cells that warrant deeper analysis")

        return hints

    def _estimate_tokens(self, compressed_hierarchy: dict[str, Any]) -> int:
        """Estimate token count for the compressed hierarchy.

        Simplified estimation - in production would use actual tokenizer.

        Args:
            compressed_hierarchy: The compressed hierarchy

        Returns:
            Estimated token count
        """
        total_chars = 0

        # Count summary
        total_chars += len(compressed_hierarchy.get("summary", ""))

        # Count sections
        for section in compressed_hierarchy.get("sections", []):
            total_chars += len(section.get("name", ""))
            total_chars += len(section.get("summary", ""))
            total_chars += 20  # Overhead

        # Count cells
        for cell in compressed_hierarchy.get("cells", []):
            total_chars += len(str(cell.get("source", "")))
            total_chars += 20  # Overhead

        # Rough estimate: 4 chars per token
        return total_chars // 4
