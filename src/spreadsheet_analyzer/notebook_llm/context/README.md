# Context Compression Module

This module provides sophisticated context compression and optimization utilities for managing LLM token budgets when analyzing spreadsheets. It implements the context engineering strategies outlined in the system design document.

## Components

### 1. Token Counter (`TokenCounter`)

Provides accurate token counting for different LLM models (OpenAI, Anthropic) with fallback approximation.

```python
from spreadsheet_analyzer.notebook_llm.context import TokenCounter

counter = TokenCounter(model="gpt-4")
tokens = counter.count_tokens("Hello world")
json_tokens = counter.estimate_json_tokens({"key": "value"})
```

### 2. Base Compressor (`BaseCompressor`)

Abstract base class for implementing compression strategies. All compressors must implement the `compress` method.

### 3. SpreadsheetLLM Compressor (`SpreadsheetLLMCompressor`)

Advanced compressor implementing techniques from the SpreadsheetLLM paper:

- **Pattern Detection**: Identifies and compresses repetitive formula patterns
- **Range Aggregation**: Combines contiguous cells into range summaries
- **Semantic Clustering**: Groups related cells by business concepts
- **Hierarchical Summarization**: Creates multi-level summaries

```python
from spreadsheet_analyzer.notebook_llm.context import SpreadsheetLLMCompressor, CellObservation

compressor = SpreadsheetLLMCompressor(
    enable_pattern_detection=True,
    enable_range_aggregation=True,
    enable_semantic_clustering=True
)

observations = [
    CellObservation(
        location="Sheet1!A1",
        observation_type="formula",
        content="=SUM(B1:B10)",
        importance=0.9
    ),
    # ... more observations
]

compressed = compressor.compress(
    observations,
    token_budget=4096,
    preserve_structure=True
)
```

### 4. Token Optimizer (`TokenOptimizer`)

Intelligent token budget management with adaptive pipeline selection:

```python
from spreadsheet_analyzer.notebook_llm.context import TokenOptimizer
from spreadsheet_analyzer.notebook_llm.strategies.base import AnalysisTask, AnalysisFocus

optimizer = TokenOptimizer(model="gpt-4")

task = AnalysisTask(
    name="formula_analysis",
    description="Analyze spreadsheet formulas",
    focus=AnalysisFocus.FORMULAS,
    expected_format=ResponseFormat.JSON
)

result = optimizer.optimize(
    observations,
    task,
    total_tokens=8192,
    preserve_structure=True
)

if result.success:
    print(f"Compressed to {result.compressed_package.token_count} tokens")
    print(f"Compression level: {result.compression_level.name}")
    print(f"Recommendations: {result.recommendations}")
```

## Compression Levels

The module supports multiple compression levels that adapt to available token budgets:

1. **NONE**: No compression (large token budgets: 50k+)
1. **LIGHT**: Basic deduplication (medium budgets: 20k-50k)
1. **MODERATE**: Pattern detection + range aggregation (standard: 8k-20k)
1. **AGGRESSIVE**: Full pipeline including semantic clustering (tight: 4k-8k)
1. **EXTREME**: Maximum compression, may lose detail (very limited: 1k-4k)

## Usage Example

```python
from spreadsheet_analyzer.notebook_llm.context import (
    TokenOptimizer,
    CellObservation,
    SpreadsheetLLMCompressor
)
from spreadsheet_analyzer.notebook_llm.strategies.base import (
    AnalysisTask,
    AnalysisFocus,
    ResponseFormat
)

# Create observations from spreadsheet analysis
observations = []
for cell in worksheet:
    obs = CellObservation(
        location=f"{worksheet.title}!{cell.coordinate}",
        observation_type="formula" if cell.data_type == "f" else "value",
        content=cell.value,
        importance=calculate_importance(cell)
    )
    observations.append(obs)

# Define analysis task
task = AnalysisTask(
    name="comprehensive_analysis",
    description="Analyze spreadsheet structure and formulas",
    focus=AnalysisFocus.GENERAL,
    expected_format=ResponseFormat.STRUCTURED
)

# Optimize context
optimizer = TokenOptimizer(model="gpt-4")
result = optimizer.optimize(observations, task)

if result.success:
    # Use compressed context for LLM
    context = result.compressed_package
    print(f"Original observations: {len(observations)}")
    print(f"Compressed cells: {len(context.cells)}")
    print(f"Token usage: {context.token_count}")
    print(f"Compression ratio: {result.metrics.compression_ratio:.1%}")
```

## Best Practices

1. **Set Importance Scores**: Assign higher importance to critical cells (formulas, totals, headers)
1. **Preserve Structure**: Enable `preserve_structure=True` for complex analysis
1. **Monitor Metrics**: Check compression metrics to ensure quality
1. **Review Recommendations**: Follow optimizer recommendations for better results
1. **Test Different Levels**: Experiment with compression levels for your use case

## Performance Considerations

- Initial token counting adds ~10-50ms overhead
- Pattern detection scales O(n log n) with formula count
- Semantic clustering adds ~100-500ms for large datasets
- Memory usage is proportional to observation count

## Integration with Strategies

The context module integrates seamlessly with analysis strategies:

```python
from spreadsheet_analyzer.notebook_llm.strategies.base import BaseStrategy

class OptimizedStrategy(BaseStrategy):
    def prepare_context(self, notebook, focus, token_budget):
        # Extract observations from notebook
        observations = self._extract_observations(notebook)
        
        # Use SpreadsheetLLMCompressor
        compressor = SpreadsheetLLMCompressor()
        return compressor.compress(
            observations,
            token_budget,
            preserve_structure=True
        )
```

## Troubleshooting

### High Token Count

- Enable more aggressive compression levels
- Reduce observation importance thresholds
- Focus on specific sheet ranges

### Lost Detail

- Increase token budget if possible
- Use MODERATE instead of AGGRESSIVE compression
- Mark critical cells with higher importance

### Slow Performance

- Disable semantic clustering for simple analysis
- Process sheets in batches
- Cache compression results when possible
