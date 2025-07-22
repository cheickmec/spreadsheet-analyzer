"""Base classes and protocols for the LLM-Jupyter notebook strategy layer.

This module provides the foundational abstractions for implementing different
prompt engineering and context management strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from spreadsheet_analyzer.notebook_llm.nap.protocols import NotebookDocument


class AnalysisFocus(Enum):
    """Types of analysis focus areas."""

    STRUCTURE = "structure"
    FORMULAS = "formulas"
    DATA_VALIDATION = "data_validation"
    PIVOT_TABLES = "pivot_tables"
    RELATIONSHIPS = "relationships"
    VALIDATION = "validation"
    GENERAL = "general"


class ResponseFormat(Enum):
    """Expected response formats from LLM."""

    JSON = "json"
    MARKDOWN = "markdown"
    STRUCTURED = "structured"
    NATURAL = "natural"


@dataclass
class AnalysisTask:
    """Represents a specific analysis task to be performed."""

    name: str
    description: str
    focus: AnalysisFocus
    expected_format: ResponseFormat
    focus_area: str | None = None
    additional_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextPackage:
    """Container for prepared context to be sent to LLM."""

    cells: list[dict[str, Any]]
    metadata: dict[str, Any]
    focus_hints: list[str]
    token_count: int
    compression_method: str | None = None
    additional_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cells": self.cells,
            "metadata": self.metadata,
            "focus_hints": self.focus_hints,
            "token_count": self.token_count,
            "compression_method": self.compression_method,
            **self.additional_data,
        }


@dataclass
class AnalysisResult:
    """Container for analysis results from LLM."""

    success: bool
    content: Any  # Can be dict, list, str depending on expected_format
    raw_response: str
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if the result is valid and usable."""
        return self.success and not self.errors


class LLMInterface(Protocol):
    """Protocol for LLM interaction."""

    @property
    def token_budget(self) -> int:
        """Maximum tokens available for this interaction."""
        ...

    def generate(self, prompt: str) -> str:
        """Generate response from prompt."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        ...


class AnalysisStrategy(Protocol):
    """Base protocol for all analysis strategies."""

    def prepare_context(self, notebook: NotebookDocument, focus: AnalysisFocus, token_budget: int) -> ContextPackage:
        """Prepare optimized context for LLM."""
        ...

    def format_prompt(self, context: ContextPackage, task: AnalysisTask) -> str:
        """Generate task-specific prompt."""
        ...

    def parse_response(self, response: str, expected_format: ResponseFormat) -> AnalysisResult:
        """Parse and validate LLM response."""
        ...


class BaseStrategy(ABC):
    """Abstract base class for all strategies with common functionality."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize strategy with configuration.

        Args:
            config: Strategy-specific configuration parameters
        """
        self.config = config or {}
        self.validators = self._load_validators()
        self.compressor = self._load_compressor()

    def _load_validators(self) -> list[Any]:
        """Load response validators based on configuration."""
        # Placeholder for validator loading logic
        return []

    def _load_compressor(self) -> Any | None:
        """Load context compressor based on configuration."""
        # Placeholder for compressor loading logic
        return None

    @abstractmethod
    def prepare_context(self, notebook: NotebookDocument, focus: AnalysisFocus, token_budget: int) -> ContextPackage:
        """Strategy-specific context preparation.

        Args:
            notebook: The notebook document to analyze
            focus: The analysis focus area
            token_budget: Maximum tokens to use for context

        Returns:
            Prepared context package
        """
        pass

    @abstractmethod
    def format_prompt(self, context: ContextPackage, task: AnalysisTask) -> str:
        """Strategy-specific prompt formatting.

        Args:
            context: Prepared context package
            task: Analysis task to perform

        Returns:
            Formatted prompt string
        """
        pass

    def parse_response(self, response: str, expected_format: ResponseFormat) -> AnalysisResult:
        """Parse and validate LLM response.

        Default implementation handles common formats.
        Override for strategy-specific parsing.

        Args:
            response: Raw LLM response
            expected_format: Expected response format

        Returns:
            Parsed analysis result
        """
        try:
            if expected_format == ResponseFormat.JSON:
                import json

                content = json.loads(response)
            elif expected_format == ResponseFormat.STRUCTURED:
                # Placeholder for structured parsing
                content = self._parse_structured(response)
            else:
                content = response

            return AnalysisResult(success=True, content=content, raw_response=response)
        except Exception as e:
            return AnalysisResult(success=False, content=None, raw_response=response, errors=[str(e)])

    def _parse_structured(self, response: str) -> dict[str, Any]:
        """Parse structured response format.

        Override this method for custom structured parsing.
        """
        # Default implementation - extract key-value pairs
        result = {}
        lines = response.strip().split("\n")
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
        return result

    def execute(self, notebook: NotebookDocument, task: AnalysisTask, llm: LLMInterface) -> AnalysisResult:
        """Template method for strategy execution.

        This method orchestrates the complete strategy execution flow.

        Args:
            notebook: The notebook document to analyze
            task: Analysis task to perform
            llm: LLM interface for generation

        Returns:
            Analysis result
        """
        # 1. Prepare context
        context = self.prepare_context(notebook, task.focus, llm.token_budget)

        # 2. Format prompt
        prompt = self.format_prompt(context, task)

        # 3. Call LLM
        response = llm.generate(prompt)

        # 4. Parse and validate
        result = self.parse_response(response, task.expected_format)

        # 5. Post-process
        return self.post_process(result)

    def post_process(self, result: AnalysisResult) -> AnalysisResult:
        """Post-process the analysis result.

        Override this method for strategy-specific post-processing.

        Args:
            result: Raw analysis result

        Returns:
            Post-processed result
        """
        # Default implementation - no post-processing
        return result
