# Context Management System

This package provides composable strategies for managing context in LLM-based spreadsheet analysis.

## Overview

The `context` package implements:

- **Context strategies** for compression and optimization
- **Token budget management** to fit within LLM limits
- **Pattern detection** for efficient representation
- **Hierarchical summarization** for large datasets
- **Strategy composition** for flexible pipelines

## Problem Statement

Large spreadsheets often exceed LLM context windows:

- GPT-4: 8k-32k tokens
- Claude: 100k-200k tokens
- A medium spreadsheet can easily exceed these limits

This package provides strategies to intelligently compress and select relevant context.

## Core Concepts

### Context Package

The fundamental unit of context:

```python
from spreadsheet_analyzer.context import ContextCell, ContextPackage

# Create cells
cells = [
    ContextCell(
        location="Sheet1!A1",
        content="Revenue",
        cell_type="value",
        importance=0.8
    ),
    ContextCell(
        location="Sheet1!B1", 
        content="=SUM(B2:B100)",
        cell_type="formula",
        importance=1.0
    )
]

# Create package
package = ContextPackage.create(
    cells=cells,
    metadata={"sheet_count": 3, "total_cells": 1000},
    focus_hints=["formulas", "summary_rows"]
)
```

### Context Strategies

Strategies transform context to fit token budgets:

```python
from spreadsheet_analyzer.context import ContextStrategy
from spreadsheet_analyzer.core import Result

class CompressionStrategy:
    @property
    def name(self) -> str:
        return "pattern_compression"
    
    def apply(self, package: ContextPackage, token_budget: int) -> Result[ContextPackage, ContextError]:
        # Detect patterns and compress
        patterns = self._detect_patterns(package.cells)
        compressed_cells = self._compress_patterns(package.cells, patterns)
        
        return ok(package.with_cells(compressed_cells))
```

### Strategy Composition

Combine multiple strategies:

```python
from spreadsheet_analyzer.context import StrategyChain

# Create strategy pipeline
compression_pipeline = StrategyChain(strategies=(
    RemoveEmptyCellsStrategy(),
    DetectPatternStrategy(),
    CompressRangesStrategy(),
    TruncateTobudgetStrategy()
))

# Apply all strategies in sequence
result = compression_pipeline.apply(package, token_budget=4000)
```

## Available Strategies

### 1. Pattern Compression

Based on SpreadsheetLLM paper - detects and compresses repetitive patterns:

```python
# Input: 100 similar formulas
=SUM(A1:A10)
=SUM(B1:B10)
=SUM(C1:C10)
...

# Output: Pattern representation
Pattern: =SUM(<col>1:<col>10) 
Instances: A through CV (100 columns)
```

### 2. Range Aggregation

Combines contiguous cells:

```python
# Input: Individual cells
A1: 100, A2: 200, A3: 300, ..., A100: 10000

# Output: Range summary
A1:A100: Numeric data, sum=550000, avg=5500, min=100, max=10000
```

### 3. Hierarchical Summarization

Creates multi-level summaries:

```python
# Level 1: Individual cells
# Level 2: Row/column summaries  
# Level 3: Sheet summaries
# Level 4: Workbook summary

summarizer = HierarchicalSummarizationStrategy(levels=3)
```

### 4. Semantic Clustering

Groups related cells by meaning:

```python
# Groups cells by business concepts
Cluster "Financial": [Revenue cells, Cost cells, Profit formulas]
Cluster "Dates": [Date headers, Timeline cells]
Cluster "Metrics": [KPI formulas, Summary calculations]
```

### 5. Importance-Based Selection

Prioritizes high-value cells:

```python
selector = ImportanceBasedStrategy(
    prioritize=["formulas", "headers", "totals"],
    deprioritize=["empty", "constants"]
)
```

## Usage Example

Complete context preparation:

```python
from spreadsheet_analyzer.context import ContextManager
from spreadsheet_analyzer.context.strategies import (
    SpreadsheetLLMCompressor,
    ImportanceSelector,
    StructurePreserver
)

# Configure manager with strategies
manager = ContextManager(strategies=[
    SpreadsheetLLMCompressor(
        enable_pattern_detection=True,
        enable_range_aggregation=True
    ),
    ImportanceSelector(threshold=0.7),
    StructurePreserver()  # Maintains relationships
])

# Prepare context for LLM
query = ContextQuery(
    query_text="Find all profit calculations",
    include_formulas=True,
    relevance_threshold=0.8
)

result = manager.prepare_context(
    cells=all_cells,
    query=query, 
    token_budget=4000
)

if result.is_ok():
    package = result.unwrap()
    print(f"Compressed {len(all_cells)} cells to {len(package.cells)}")
    print(f"Token usage: {package.token_count}/{token_budget}")
```

## Compression Metrics

Track compression effectiveness:

```python
metrics = manager.get_metrics()
print(f"Compression ratio: {metrics.compression_ratio:.2f}")
print(f"Patterns detected: {metrics.patterns_detected}")
print(f"Time elapsed: {metrics.time_elapsed_ms}ms")
```

## Strategy Development

Create custom strategies:

```python
from spreadsheet_analyzer.context import ContextStrategy

class CustomStrategy:
    """Remove all cells except formulas."""
    
    @property
    def name(self) -> str:
        return "formula_only"
    
    def apply(self, package: ContextPackage, token_budget: int) -> Result[ContextPackage, ContextError]:
        formula_cells = [
            cell for cell in package.cells 
            if cell.cell_type == "formula"
        ]
        
        new_package = package.with_cells(formula_cells)
        
        if new_package.token_count <= token_budget:
            return ok(new_package)
        else:
            return err(ContextError(
                "Formula cells exceed token budget",
                token_count=new_package.token_count
            ))
```

## Best Practices

1. **Profile your data** - Understand patterns before choosing strategies
1. **Compose strategies** - Combine multiple approaches
1. **Preserve structure** - Maintain relationships between cells
1. **Test compression** - Verify important data isn't lost
1. **Monitor metrics** - Track compression effectiveness

## Performance Tips

- **Cache compressed results** for repeated queries
- **Use parallel processing** for large sheets
- **Apply strategies incrementally** - stop when budget met
- **Pre-compute importance scores** during initial analysis

## Research Foundation

Based on cutting-edge research:

- **SpreadsheetLLM** (Microsoft 2024) - Specialized encoding
- **Lost in the Middle** (2024) - Context position matters
- **Hierarchical Summarization** - Multi-level abstractions
- **Semantic Compression** - Meaning-preserving reduction
