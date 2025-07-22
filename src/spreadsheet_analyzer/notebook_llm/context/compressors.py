"""Context compression implementations for optimizing LLM token usage.

This module implements various compression strategies based on the design document's
context engineering approaches, including hierarchical summarization, pattern
compression, and the SpreadsheetLLM approach.

CLAUDE-KNOWLEDGE: SpreadsheetLLM compression techniques are based on the paper
"SpreadsheetLLM: Encoding Spreadsheets for Large Language Models" which introduces
specialized encoding methods for tabular data.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Protocol

import tiktoken

from spreadsheet_analyzer.notebook_llm.strategies.base import ContextPackage


class TokenCounterProtocol(Protocol):
    """Protocol for token counting implementations."""

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        ...


@dataclass
class CompressionMetrics:
    """Metrics about compression performance."""

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    cells_processed: int
    patterns_detected: int
    time_elapsed: float
    method_used: str


@dataclass
class CellObservation:
    """Represents an observation about a cell or range."""

    location: str  # Cell address or range
    observation_type: str  # "value", "formula", "format", etc.
    content: Any
    importance: float = 1.0  # Priority score for retention
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class TokenCounter:
    """Utility class for counting tokens across different LLM models.

    Supports OpenAI and Anthropic token counting with fallback to approximation.
    """

    def __init__(self, model: str = "gpt-4"):
        """Initialize token counter for specific model.

        Args:
            model: Model name for token counting (e.g., "gpt-4", "claude-3")
        """
        self.model = model
        self._encoder = None
        self._init_encoder()

    def _init_encoder(self) -> None:
        """Initialize the appropriate encoder for the model."""
        try:
            if "gpt" in self.model.lower() or "turbo" in self.model.lower():
                # Use tiktoken for OpenAI models
                encoding_name = "cl100k_base"  # Default for GPT-4 and newer
                if "gpt-3.5" in self.model.lower():
                    encoding_name = "cl100k_base"
                elif "davinci" in self.model.lower():
                    encoding_name = "p50k_base"
                self._encoder = tiktoken.get_encoding(encoding_name)
            # For other models, fall back to approximation
        except Exception:
            # If tiktoken fails, use approximation
            self._encoder = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0

        if self._encoder:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                # Fall back to approximation if encoding fails
                self._encoder = None

        # Approximation: ~0.75 words per token, ~5 chars per word
        return max(1, len(text) // 4)

    def estimate_json_tokens(self, data: dict[str, Any]) -> int:
        """Estimate tokens for JSON data.

        Args:
            data: Dictionary to be serialized to JSON

        Returns:
            Estimated token count
        """
        import json

        json_str = json.dumps(data, separators=(",", ":"))
        return self.count_tokens(json_str)


class BaseCompressor(ABC):
    """Abstract base class for context compression strategies."""

    def __init__(self, token_counter: TokenCounterProtocol | None = None):
        """Initialize compressor with optional token counter.

        Args:
            token_counter: Token counter to use, defaults to TokenCounter
        """
        self.token_counter = token_counter or TokenCounter()
        self.metrics = CompressionMetrics(
            original_tokens=0,
            compressed_tokens=0,
            compression_ratio=0.0,
            cells_processed=0,
            patterns_detected=0,
            time_elapsed=0.0,
            method_used=self.__class__.__name__,
        )

    @abstractmethod
    def compress(
        self, observations: list[CellObservation], token_budget: int, preserve_structure: bool = True
    ) -> ContextPackage:
        """Compress observations to fit within token budget.

        Args:
            observations: List of cell observations to compress
            token_budget: Maximum tokens allowed
            preserve_structure: Whether to preserve structural relationships

        Returns:
            Compressed context package
        """
        pass

    def _calculate_metrics(
        self, original_observations: list[CellObservation], compressed_package: ContextPackage, time_elapsed: float
    ) -> None:
        """Update compression metrics."""
        original_tokens = sum(obs.tokens for obs in original_observations)
        self.metrics = CompressionMetrics(
            original_tokens=original_tokens,
            compressed_tokens=compressed_package.token_count,
            compression_ratio=1 - (compressed_package.token_count / max(1, original_tokens)),
            cells_processed=len(original_observations),
            patterns_detected=len(compressed_package.metadata.get("patterns", [])),
            time_elapsed=time_elapsed,
            method_used=compressed_package.compression_method or self.__class__.__name__,
        )


class SpreadsheetLLMCompressor(BaseCompressor):
    """Advanced compressor implementing SpreadsheetLLM encoding techniques.

    This compressor uses specialized strategies for spreadsheet data:
    1. Structural anchors for preserving sheet relationships
    2. Pattern-based compression for repetitive formulas
    3. Hierarchical summarization for large ranges
    4. Semantic clustering for related cells
    """

    def __init__(
        self,
        token_counter: TokenCounterProtocol | None = None,
        enable_pattern_detection: bool = True,
        enable_range_aggregation: bool = True,
        enable_semantic_clustering: bool = True,
    ):
        """Initialize SpreadsheetLLM compressor.

        Args:
            token_counter: Token counter to use
            enable_pattern_detection: Enable formula pattern detection
            enable_range_aggregation: Enable range-based aggregation
            enable_semantic_clustering: Enable semantic cell clustering
        """
        super().__init__(token_counter)
        self.enable_pattern_detection = enable_pattern_detection
        self.enable_range_aggregation = enable_range_aggregation
        self.enable_semantic_clustering = enable_semantic_clustering

    def compress(
        self, observations: list[CellObservation], token_budget: int, preserve_structure: bool = True
    ) -> ContextPackage:
        """Compress observations using SpreadsheetLLM techniques.

        Applies multiple compression strategies in sequence:
        1. Pattern detection and consolidation
        2. Range aggregation for contiguous cells
        3. Hierarchical summarization
        4. Priority-based retention

        Args:
            observations: List of cell observations to compress
            token_budget: Maximum tokens allowed
            preserve_structure: Whether to preserve structural relationships

        Returns:
            Compressed context package
        """
        import time

        start_time = time.time()

        # Calculate initial tokens
        for obs in observations:
            if obs.tokens == 0:
                obs.tokens = self._estimate_observation_tokens(obs)

        # Apply compression pipeline
        compressed_obs = observations.copy()

        if self.enable_pattern_detection:
            compressed_obs = self._detect_and_compress_patterns(compressed_obs)

        if self.enable_range_aggregation:
            compressed_obs = self._aggregate_ranges(compressed_obs)

        if self.enable_semantic_clustering:
            compressed_obs = self._cluster_semantically(compressed_obs)

        # Build context package within budget
        package = self._build_context_package(compressed_obs, token_budget, preserve_structure)

        # Update metrics
        self._calculate_metrics(observations, package, time.time() - start_time)

        return package

    def _estimate_observation_tokens(self, obs: CellObservation) -> int:
        """Estimate tokens for a single observation."""
        # Build string representation
        parts = [f"Cell {obs.location}:", f"Type: {obs.observation_type}", f"Content: {obs.content}"]
        if obs.metadata:
            parts.append(f"Metadata: {obs.metadata}")

        text = " ".join(str(p) for p in parts)
        return self.token_counter.count_tokens(text)

    def _detect_and_compress_patterns(self, observations: list[CellObservation]) -> list[CellObservation]:
        """Detect and compress formula patterns.

        Groups similar formulas and represents them as patterns rather than
        individual instances.
        """
        formula_obs = [o for o in observations if o.observation_type == "formula"]
        non_formula_obs = [o for o in observations if o.observation_type != "formula"]

        if not formula_obs:
            return observations

        # Group formulas by pattern
        pattern_groups: dict[str, list[CellObservation]] = defaultdict(list)

        for obs in formula_obs:
            pattern = self._extract_formula_pattern(str(obs.content))
            pattern_groups[pattern].append(obs)

        # Create compressed representations
        compressed = []
        patterns_metadata = []

        for pattern, group in pattern_groups.items():
            if len(group) > 2:  # Only compress if pattern appears 3+ times
                # Create pattern observation
                locations = [o.location for o in group]
                pattern_obs = CellObservation(
                    location=f"Pattern({locations[0]}...{locations[-1]})",
                    observation_type="formula_pattern",
                    content=pattern,
                    importance=max(o.importance for o in group),
                    metadata={
                        "instances": len(group),
                        "locations": [*locations[:5], "..."] if len(locations) > 5 else locations,
                        "pattern": pattern,
                    },
                )
                pattern_obs.tokens = self._estimate_observation_tokens(pattern_obs)
                compressed.append(pattern_obs)
                patterns_metadata.append({"pattern": pattern, "count": len(group)})
            else:
                # Keep individual formulas if pattern is rare
                compressed.extend(group)

        # Store pattern information in metadata
        if patterns_metadata:
            for obs in compressed:
                if "patterns" not in obs.metadata:
                    obs.metadata["patterns"] = patterns_metadata

        return non_formula_obs + compressed

    def _extract_formula_pattern(self, formula: str) -> str:
        """Extract pattern from formula by replacing cell references.

        Example: "=SUM(A1:A10)" -> "=SUM(<range>)"
        """
        import re

        # Replace cell references with placeholders
        pattern = formula

        # Replace ranges (A1:B10)
        pattern = re.sub(r"[A-Z]+\d+:[A-Z]+\d+", "<range>", pattern)

        # Replace individual cells (A1)
        pattern = re.sub(r"[A-Z]+\d+", "<cell>", pattern)

        # Replace numbers
        pattern = re.sub(r"\b\d+\.?\d*\b", "<num>", pattern)

        return pattern

    def _aggregate_ranges(self, observations: list[CellObservation]) -> list[CellObservation]:
        """Aggregate observations for contiguous ranges.

        Combines multiple cell observations into range summaries when they
        represent contiguous data with similar characteristics.
        """
        # Group by sheet and type
        sheet_groups: dict[tuple[str, str], list[CellObservation]] = defaultdict(list)

        for obs in observations:
            # Extract sheet from location (assumes "Sheet!A1" format)
            sheet = "Sheet1"  # Default
            if "!" in obs.location:
                sheet, _ = obs.location.split("!", 1)

            key = (sheet, obs.observation_type)
            sheet_groups[key].append(obs)

        compressed = []

        for (sheet, obs_type), group in sheet_groups.items():
            if len(group) < 5:  # Don't aggregate small groups
                compressed.extend(group)
                continue

            # Try to identify contiguous ranges
            ranges = self._identify_contiguous_ranges(group)

            for range_info in ranges:
                if range_info["count"] > 5:
                    # Create range observation
                    range_obs = CellObservation(
                        location=f"{sheet}!{range_info['start']}:{range_info['end']}",
                        observation_type=f"{obs_type}_range",
                        content=range_info["summary"],
                        importance=range_info["avg_importance"],
                        metadata={
                            "cell_count": range_info["count"],
                            "range_type": range_info["type"],
                            "samples": range_info["samples"],
                        },
                    )
                    range_obs.tokens = self._estimate_observation_tokens(range_obs)
                    compressed.append(range_obs)
                else:
                    # Keep individual observations for small ranges
                    compressed.extend(range_info["observations"])

        return compressed

    def _identify_contiguous_ranges(self, observations: list[CellObservation]) -> list[dict[str, Any]]:
        """Identify contiguous ranges in observations."""
        # Simple implementation - in practice would use more sophisticated logic
        # For now, just group all observations as one range
        if not observations:
            return []

        return [
            {
                "start": observations[0].location,
                "end": observations[-1].location,
                "count": len(observations),
                "type": "data",
                "summary": f"Range of {len(observations)} cells",
                "avg_importance": sum(o.importance for o in observations) / len(observations),
                "samples": [o.content for o in observations[:3]],
                "observations": observations,
            }
        ]

    def _cluster_semantically(self, observations: list[CellObservation]) -> list[CellObservation]:
        """Cluster semantically related observations.

        Groups observations that refer to related business concepts or
        share semantic relationships.
        """
        # Simple keyword-based clustering
        clusters: dict[str, list[CellObservation]] = defaultdict(list)
        unclustered = []

        for obs in observations:
            # Extract keywords from content and metadata
            keywords = self._extract_keywords(obs)

            if keywords:
                # Use first keyword as cluster key (simplified)
                cluster_key = next(iter(keywords))
                clusters[cluster_key].append(obs)
            else:
                unclustered.append(obs)

        # Build compressed representations for large clusters
        compressed = []

        for keyword, cluster in clusters.items():
            if len(cluster) > 3:
                # Create cluster summary
                cluster_obs = CellObservation(
                    location=f"Cluster({keyword})",
                    observation_type="semantic_cluster",
                    content=f"Cluster of {len(cluster)} related cells about '{keyword}'",
                    importance=max(o.importance for o in cluster),
                    metadata={
                        "keyword": keyword,
                        "count": len(cluster),
                        "types": list({o.observation_type for o in cluster}),
                        "sample_locations": [o.location for o in cluster[:3]],
                    },
                )
                cluster_obs.tokens = self._estimate_observation_tokens(cluster_obs)
                compressed.append(cluster_obs)

                # Keep most important individual observations
                important = sorted(cluster, key=lambda o: o.importance, reverse=True)[:2]
                compressed.extend(important)
            else:
                compressed.extend(cluster)

        compressed.extend(unclustered)
        return compressed

    def _extract_keywords(self, obs: CellObservation) -> set[str]:
        """Extract semantic keywords from observation."""
        keywords = set()

        # Extract from content
        content_str = str(obs.content).lower()

        # Common business terms
        business_terms = {
            "revenue",
            "cost",
            "profit",
            "sales",
            "total",
            "sum",
            "average",
            "count",
            "budget",
            "forecast",
            "actual",
            "variance",
            "ytd",
            "mtd",
            "quarter",
            "year",
        }

        for term in business_terms:
            if term in content_str:
                keywords.add(term)

        # Extract from location (sheet names often contain keywords)
        if "!" in obs.location:
            sheet_name = obs.location.split("!")[0].lower()
            for term in business_terms:
                if term in sheet_name:
                    keywords.add(term)

        return keywords

    def _build_context_package(
        self, observations: list[CellObservation], token_budget: int, preserve_structure: bool
    ) -> ContextPackage:
        """Build final context package within token budget.

        Prioritizes observations by importance and structural significance.
        """
        # Sort by importance
        sorted_obs = sorted(observations, key=lambda o: o.importance, reverse=True)

        # Build context incrementally
        cells = []
        metadata = {
            "compression_method": "SpreadsheetLLM",
            "preserved_structure": preserve_structure,
            "total_observations": len(observations),
        }
        focus_hints = []
        current_tokens = 50  # Reserve tokens for package structure

        # Always include structural anchors if preserving structure
        if preserve_structure:
            structural_obs = [
                o for o in sorted_obs if o.observation_type in ["formula_pattern", "semantic_cluster", "range_summary"]
            ]
            for obs in structural_obs[:5]:  # Top 5 structural elements
                if current_tokens + obs.tokens <= token_budget:
                    cells.append(self._observation_to_cell(obs))
                    current_tokens += obs.tokens
                    focus_hints.append(f"Structural: {obs.observation_type}")

        # Add remaining observations by priority
        for obs in sorted_obs:
            if current_tokens + obs.tokens <= token_budget:
                cell_data = self._observation_to_cell(obs)
                if cell_data not in cells:  # Avoid duplicates
                    cells.append(cell_data)
                    current_tokens += obs.tokens

        # Add pattern information to metadata
        patterns = []
        for obs in observations:
            if "patterns" in obs.metadata:
                patterns.extend(obs.metadata["patterns"])
        if patterns:
            metadata["patterns"] = patterns

        return ContextPackage(
            cells=cells,
            metadata=metadata,
            focus_hints=focus_hints,
            token_count=current_tokens,
            compression_method="SpreadsheetLLM",
        )

    def _observation_to_cell(self, obs: CellObservation) -> dict[str, Any]:
        """Convert observation to cell dictionary format."""
        cell = {"location": obs.location, "type": obs.observation_type, "content": obs.content}

        # Add relevant metadata
        if obs.observation_type == "formula_pattern":
            cell["pattern_info"] = obs.metadata
        elif obs.observation_type == "semantic_cluster":
            cell["cluster_info"] = obs.metadata
        elif obs.metadata:
            cell["metadata"] = obs.metadata

        return cell
