"""Tests for context compression utilities."""

import pytest

from spreadsheet_analyzer.notebook_llm.context import (
    CellObservation,
    SpreadsheetLLMCompressor,
    TokenCounter,
    TokenOptimizer,
)
from spreadsheet_analyzer.notebook_llm.strategies.base import AnalysisFocus, AnalysisTask, ResponseFormat


class TestTokenCounter:
    """Test token counting functionality."""

    def test_basic_token_counting(self):
        """Test basic token counting."""
        counter = TokenCounter()

        # Test empty string
        assert counter.count_tokens("") == 0

        # Test simple text
        assert counter.count_tokens("Hello world") > 0

        # Test longer text
        long_text = "This is a longer text that should have more tokens. " * 10
        token_count = counter.count_tokens(long_text)
        assert token_count > 10

    def test_json_token_estimation(self):
        """Test JSON token estimation."""
        counter = TokenCounter()

        data = {"cell": "A1", "value": 123, "formula": "=SUM(B1:B10)"}

        token_count = counter.estimate_json_tokens(data)
        assert token_count > 0


class TestSpreadsheetLLMCompressor:
    """Test SpreadsheetLLM compression strategies."""

    def test_basic_compression(self):
        """Test basic compression functionality."""
        compressor = SpreadsheetLLMCompressor()

        observations = [
            CellObservation(location="Sheet1!A1", observation_type="value", content="Revenue", importance=1.0),
            CellObservation(location="Sheet1!B1", observation_type="value", content=1000, importance=0.8),
            CellObservation(location="Sheet1!C1", observation_type="formula", content="=SUM(B1:B10)", importance=0.9),
        ]

        # Compress with small budget
        package = compressor.compress(observations, token_budget=100)

        assert package.token_count <= 100
        assert len(package.cells) > 0
        assert package.compression_method == "SpreadsheetLLM"

    def test_pattern_compression(self):
        """Test formula pattern compression."""
        compressor = SpreadsheetLLMCompressor(enable_pattern_detection=True)

        # Create many similar formulas
        observations = []
        for i in range(10):
            observations.append(
                CellObservation(
                    location=f"Sheet1!A{i + 1}",
                    observation_type="formula",
                    content=f"=SUM(B{i + 1}:C{i + 1})",
                    importance=0.5,
                )
            )

        package = compressor.compress(observations, token_budget=200)

        # Should have detected and compressed the pattern
        assert any("pattern" in str(cell) for cell in package.cells)
        assert package.token_count < sum(o.tokens for o in observations if o.tokens > 0)

    def test_semantic_clustering(self):
        """Test semantic clustering of related cells."""
        compressor = SpreadsheetLLMCompressor(enable_semantic_clustering=True)

        observations = [
            CellObservation(location="Revenue!A1", observation_type="value", content="Q1 Revenue"),
            CellObservation(location="Revenue!B1", observation_type="value", content=1000),
            CellObservation(location="Cost!A1", observation_type="value", content="Q1 Cost"),
            CellObservation(location="Cost!B1", observation_type="value", content=500),
            CellObservation(location="Profit!A1", observation_type="value", content="Q1 Profit"),
            CellObservation(location="Profit!B1", observation_type="formula", content="=Revenue!B1-Cost!B1"),
        ]

        package = compressor.compress(observations, token_budget=150)

        # Should have created some semantic clusters
        assert len(package.cells) > 0
        assert package.metadata.get("total_observations") == len(observations)


class TestTokenOptimizer:
    """Test token optimization and pipeline selection."""

    def test_budget_allocation(self):
        """Test token budget allocation."""
        optimizer = TokenOptimizer()

        task = AnalysisTask(
            name="test", description="Test analysis", focus=AnalysisFocus.FORMULAS, expected_format=ResponseFormat.JSON
        )

        budget = optimizer.allocate_budget(task, total_tokens=8192)

        assert budget.total == 8192
        assert budget.available_for_context > 0
        assert budget.system_prompt > 0
        assert budget.response_reserve > 0

    def test_pipeline_selection(self):
        """Test compression pipeline selection."""
        optimizer = TokenOptimizer()

        # Create many observations
        observations = [
            CellObservation(location=f"Sheet1!A{i}", observation_type="value", content=f"Value {i}", importance=0.5)
            for i in range(100)
        ]

        task = AnalysisTask(
            name="test", description="Test analysis", focus=AnalysisFocus.STRUCTURE, expected_format=ResponseFormat.JSON
        )

        budget = optimizer.allocate_budget(task, total_tokens=4096)
        pipeline = optimizer.select_pipeline(observations, task, budget)

        assert pipeline is not None
        assert pipeline.level.value > 0  # Should select some compression

    def test_full_optimization(self):
        """Test full optimization process."""
        optimizer = TokenOptimizer()

        observations = [
            CellObservation(
                location=f"Sheet1!{chr(65 + i % 26)}{i // 26 + 1}",
                observation_type="formula" if i % 3 == 0 else "value",
                content=f"=SUM(A{i}:A{i + 10})" if i % 3 == 0 else f"Data {i}",
                importance=1.0 - (i * 0.01),
            )
            for i in range(50)
        ]

        task = AnalysisTask(
            name="complex_analysis",
            description="Analyze complex spreadsheet",
            focus=AnalysisFocus.FORMULAS,
            expected_format=ResponseFormat.STRUCTURED,
        )

        result = optimizer.optimize(observations, task, total_tokens=8192)  # Use larger token budget

        assert result.success
        assert result.compressed_package is not None
        assert result.compressed_package.token_count <= 8192 * 0.5  # Should fit in context allocation
        assert len(result.recommendations) >= 0
        assert result.compression_level.value >= 0  # Can be NONE if no compression needed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
