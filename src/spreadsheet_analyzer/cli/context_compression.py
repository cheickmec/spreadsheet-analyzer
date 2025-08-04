"""Hierarchical context compression for managing LLM context windows.

This module implements progressive compression strategies to handle
context window limitations by removing less important information first.

CLAUDE-KNOWLEDGE: Context compression is applied only when needed,
preserving maximum useful context while staying within model limits.
"""

import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from structlog import get_logger

logger = get_logger(__name__)

# Get the LLM message logger if it exists
# This follows Python logging best practices - we get the logger by name
# rather than passing it around as a parameter
llm_logger = logging.getLogger("llm_messages")


@dataclass(frozen=True)
class CompressionLevel:
    """Definition of a compression level in the hierarchy."""

    name: str
    description: str
    compress_func: Callable[[list[Any], int], list[Any]]
    estimated_reduction: float  # Estimated token reduction ratio (0.1 = 10% reduction)


class HierarchicalContextCompressor:
    """Manages progressive context compression for LLM interactions."""

    def __init__(self):
        self.compression_hierarchy = self._initialize_hierarchy()
        self.compression_stats = {"levels_applied": 0, "tokens_saved": 0}

    def _initialize_hierarchy(self) -> list[CompressionLevel]:
        """Initialize compression hierarchy from least to most important."""
        return [
            CompressionLevel(
                name="large_dataframe_outputs",
                description="Truncate DataFrame outputs to summaries",
                compress_func=self._truncate_dataframe_outputs,
                estimated_reduction=0.3,
            ),
            CompressionLevel(
                name="repetitive_code_outputs",
                description="Consolidate similar outputs",
                compress_func=self._consolidate_repetitive_outputs,
                estimated_reduction=0.2,
            ),
            CompressionLevel(
                name="old_exploration_cells",
                description="Summarize old exploration",
                compress_func=self._summarize_exploration_cells,
                estimated_reduction=0.25,
            ),
            CompressionLevel(
                name="intermediate_calculations",
                description="Remove intermediate steps",
                compress_func=self._remove_intermediate_steps,
                estimated_reduction=0.15,
            ),
            CompressionLevel(
                name="older_message_rounds",
                description="Summarize older rounds",
                compress_func=self._summarize_old_rounds,
                estimated_reduction=0.3,
            ),
            CompressionLevel(
                name="tool_message_bodies",
                description="Aggressively truncate tool outputs",
                compress_func=self._truncate_tool_messages,
                estimated_reduction=0.4,
            ),
            CompressionLevel(
                name="current_analysis_context",
                description="Minimize current context",
                compress_func=self._minimize_current_context,
                estimated_reduction=0.5,
            ),
        ]

    def compress_messages(self, messages: list[Any], compression_level: int, reduction_target: int = 0) -> list[Any]:
        """Apply compression at specified level.

        Args:
            messages: List of messages to compress
            compression_level: Level of compression to apply (0-based)
            reduction_target: Target token reduction (optional)

        Returns:
            Compressed messages
        """
        if compression_level >= len(self.compression_hierarchy):
            logger.warning(f"Compression level {compression_level} exceeds hierarchy depth")
            llm_logger.info(f"⚠️ Compression level {compression_level} exceeds hierarchy depth")
            return messages

        level = self.compression_hierarchy[compression_level]
        logger.info(f"Applying compression: {level.name} - {level.description}")

        # Log to LLM message logger for detailed tracking
        llm_logger.info(f"\n{'🗜️' * 20} Context Compression {'🗜️' * 20}")
        llm_logger.info(f"Compression Level: {compression_level}")
        llm_logger.info(f"Strategy: {level.name}")
        llm_logger.info(f"Description: {level.description}")
        llm_logger.info(f"Estimated Reduction: {level.estimated_reduction * 100:.0f}%")

        # Log initial message stats
        total_chars = sum(len(msg.content) for msg in messages if hasattr(msg, "content") and msg.content is not None)
        llm_logger.info(f"Before Compression: {len(messages)} messages, ~{total_chars // 4} tokens")

        compressed = level.compress_func(messages, reduction_target)
        self.compression_stats["levels_applied"] = compression_level + 1

        # Log compression results
        compressed_chars = sum(len(msg.content) for msg in compressed if hasattr(msg, "content") and msg.content is not None)
        reduction_pct = (1 - compressed_chars / total_chars) * 100 if total_chars > 0 else 0
        llm_logger.info(f"After Compression: {len(compressed)} messages, ~{compressed_chars // 4} tokens")
        llm_logger.info(f"Actual Reduction: {reduction_pct:.1f}%")
        llm_logger.info(f"{'🗜️' * 50}\n")

        return compressed

    def _truncate_dataframe_outputs(self, messages: list[Any], reduction_target: int) -> list[Any]:
        """Truncate large DataFrame outputs to summaries."""
        compressed = []

        for msg in messages:
            if isinstance(msg, (AIMessage, ToolMessage)):
                content = msg.content
                # Pattern to match DataFrame outputs
                df_pattern = r"(\d+)\s+rows?\s*×\s*(\d+)\s+columns?"
                if re.search(df_pattern, content) and len(content) > 1000:
                    # Extract DataFrame info and create summary
                    match = re.search(df_pattern, content)
                    rows, cols = match.groups() if match else ("?", "?")
                    summary = f"[DataFrame output truncated: {rows} rows × {cols} columns]"

                    # Keep first few lines if present
                    lines = content.split("\n")
                    if len(lines) > 10:
                        summary = "\n".join(lines[:5]) + f"\n{summary}\n"

                    if isinstance(msg, ToolMessage):
                        new_msg = ToolMessage(content=summary, tool_call_id=msg.tool_call_id)
                    else:
                        new_msg = type(msg)(content=summary)
                    compressed.append(new_msg)
                else:
                    compressed.append(msg)
            else:
                compressed.append(msg)

        return compressed

    def _consolidate_repetitive_outputs(self, messages: list[Any], reduction_target: int) -> list[Any]:
        """Consolidate similar repetitive outputs."""
        compressed = []
        output_buffer = []
        pattern_groups = {}

        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = msg.content
                # Detect patterns like checking multiple columns
                if "isnull().sum()" in content or "describe()" in content:
                    # Group similar operations
                    pattern = self._extract_pattern(content)
                    if pattern not in pattern_groups:
                        pattern_groups[pattern] = []
                    pattern_groups[pattern].append(msg)  # Store the full message, not just content
                else:
                    compressed.append(msg)
            else:
                compressed.append(msg)

        # Consolidate pattern groups
        for pattern, msgs in pattern_groups.items():
            if len(msgs) > 2:
                summary = f"[Consolidated {len(msgs)} similar operations: {pattern}]"
                # Use tool_call_id from first message in the group
                compressed.append(ToolMessage(content=summary, tool_call_id=msgs[0].tool_call_id))
            else:
                # Add all messages if there are 2 or fewer
                compressed.extend(msgs)

        return compressed

    def _summarize_exploration_cells(self, messages: list[Any], reduction_target: int) -> list[Any]:
        """Replace old exploration with summaries."""
        compressed = []
        exploration_buffer = []

        for i, msg in enumerate(messages):
            # Keep system and initial messages
            if i < 2 or isinstance(msg, SystemMessage):
                compressed.append(msg)
                continue

            # Identify exploration patterns
            if isinstance(msg, ToolMessage) and any(
                pattern in msg.content for pattern in ["shape", "head(", "info(", "columns", "dtypes"]
            ):
                exploration_buffer.append(msg)
            else:
                # Flush exploration buffer if we have accumulated enough
                if len(exploration_buffer) > 3:
                    summary = self._create_exploration_summary(exploration_buffer)
                    # Use tool_call_id from first message in buffer, or generate new one
                    first_tool_call_id = getattr(exploration_buffer[0], "tool_call_id", str(uuid.uuid4()))
                    compressed.append(ToolMessage(content=summary, tool_call_id=first_tool_call_id))
                    exploration_buffer = []
                compressed.append(msg)

        return compressed

    def _remove_intermediate_steps(self, messages: list[Any], reduction_target: int) -> list[Any]:
        """Remove intermediate calculation steps, keeping only results."""
        compressed = []

        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage):
                # Check if this is an intermediate step
                if self._is_intermediate_calculation(msg.content):
                    # Skip if next message has the final result
                    if i + 1 < len(messages) and self._has_final_result(messages[i + 1]):
                        continue
                compressed.append(msg)
            else:
                compressed.append(msg)

        return compressed

    def _summarize_old_rounds(self, messages: list[Any], reduction_target: int) -> list[Any]:
        """Summarize older conversation rounds."""
        compressed = []
        round_threshold = 3  # Keep last N rounds intact

        # Always keep system and initial messages
        compressed.extend(messages[:2])

        # Group messages by rounds
        rounds = self._group_messages_by_rounds(messages[2:])

        # Summarize older rounds
        for i, round_msgs in enumerate(rounds):
            if i < len(rounds) - round_threshold:
                summary = self._create_round_summary(round_msgs)
                compressed.append(AIMessage(content=f"[Round {i + 1} Summary: {summary}]"))
            else:
                compressed.extend(round_msgs)

        return compressed

    def _truncate_tool_messages(self, messages: list[Any], reduction_target: int) -> list[Any]:
        """Aggressively truncate tool message outputs."""
        compressed = []
        max_tool_output = 500  # Characters

        for msg in messages:
            if isinstance(msg, ToolMessage) and len(msg.content) > max_tool_output:
                truncated = msg.content[:max_tool_output] + "\n[Output truncated...]"
                new_msg = ToolMessage(content=truncated, tool_call_id=msg.tool_call_id)
                compressed.append(new_msg)
            else:
                compressed.append(msg)

        return compressed

    def _minimize_current_context(self, messages: list[Any], reduction_target: int) -> list[Any]:
        """Keep only essential context for current analysis."""
        # This is the most aggressive compression
        compressed = []

        # Always keep system message
        compressed.append(messages[0])

        # Create minimal context
        summary = "Previous analysis completed. Key findings available. Continue with current task."
        compressed.append(HumanMessage(content=summary))

        # Keep only the most recent round
        recent_round = self._get_most_recent_round(messages)
        compressed.extend(recent_round)

        return compressed

    # Helper methods
    def _extract_pattern(self, content: str) -> str:
        """Extract operation pattern from content."""
        if "isnull().sum()" in content:
            return "null_check"
        elif "describe()" in content:
            return "statistical_summary"
        elif "value_counts()" in content:
            return "value_distribution"
        return "general_analysis"

    def _create_exploration_summary(self, exploration_msgs: list[Any]) -> str:
        """Create summary of exploration messages."""
        summary_parts = ["Data exploration summary:"]
        for msg in exploration_msgs:
            if "shape" in msg.content:
                summary_parts.append("- Dataset dimensions checked")
            elif "dtypes" in msg.content:
                summary_parts.append("- Data types analyzed")
            elif "head(" in msg.content:
                summary_parts.append("- Sample data examined")
        return "\n".join(summary_parts)

    def _is_intermediate_calculation(self, content: str) -> bool:
        """Check if content represents an intermediate calculation."""
        intermediate_patterns = [
            "Step \\d+:",
            "Intermediate result:",
            "Calculating...",
            "Processing...",
        ]
        return any(re.search(pattern, content) for pattern in intermediate_patterns)

    def _has_final_result(self, msg: Any) -> bool:
        """Check if message contains final result."""
        if not hasattr(msg, "content"):
            return False
        final_patterns = ["Final result:", "Conclusion:", "Summary:", "Total:"]
        return any(pattern in msg.content for pattern in final_patterns)

    def _group_messages_by_rounds(self, messages: list[Any]) -> list[list[Any]]:
        """Group messages into conversation rounds."""
        rounds = []
        current_round = []

        for msg in messages:
            current_round.append(msg)
            # End of round when we see an AI message without tool calls
            if isinstance(msg, AIMessage) and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                rounds.append(current_round)
                current_round = []

        if current_round:
            rounds.append(current_round)

        return rounds

    def _create_round_summary(self, round_msgs: list[Any]) -> str:
        """Create summary of a conversation round."""
        actions = []
        for msg in round_msgs:
            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                actions.append(f"Executed {len(msg.tool_calls)} tool calls")
            elif isinstance(msg, ToolMessage):
                if "error" in msg.content.lower():
                    actions.append("Encountered error")
                else:
                    actions.append("Processed tool output")
        return "; ".join(actions) if actions else "Analysis step completed"

    def _get_most_recent_round(self, messages: list[Any]) -> list[Any]:
        """Extract the most recent conversation round."""
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage) and not (
                hasattr(messages[i], "tool_calls") and messages[i].tool_calls
            ):
                return messages[i:]
        return messages[-5:]  # Fallback to last 5 messages


def estimate_compression_reduction(messages: list[Any], level: int) -> int:
    """Estimate token reduction for a compression level.

    Args:
        messages: Messages to be compressed
        level: Compression level (0-based)

    Returns:
        Estimated tokens that would be saved
    """
    compressor = HierarchicalContextCompressor()
    if level >= len(compressor.compression_hierarchy):
        return 0

    compression = compressor.compression_hierarchy[level]
    total_chars = sum(len(msg.content) for msg in messages if hasattr(msg, "content") and msg.content is not None)
    # Rough estimate: 4 chars per token
    estimated_tokens = total_chars // 4
    return int(estimated_tokens * compression.estimated_reduction)
