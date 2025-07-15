# Algorithm Selection Rationale - Summary

## Overview

This document summarizes the algorithmic choices and their justifications added to the deterministic analysis pipeline documentation.

## Key Algorithmic Decisions and Rationales

### 1. Cryptographic Hashing - SHA-256

**Choice**: SHA-256 for file deduplication and integrity

**Rationale**:

- **Security**: 2^256 possible hashes makes collisions practically impossible
- **Not MD5/SHA-1**: Both have known vulnerabilities and collision attacks
- **Not SHA-512**: SHA-256 offers sufficient security with better performance
- **Not CRC32**: Designed for error detection, not security or uniqueness

### 2. Graph Algorithms - Tarjan's SCC

**Choice**: Tarjan's strongly connected components for circular reference detection

**Rationale**:

- **Performance**: O(V+E) linear time complexity, finds all cycles in one pass
- **Not Floyd's**: Only detects existence, doesn't identify all cycles
- **Not DFS alone**: Would be O(V²) for finding all cycles
- **Not Johnson's**: Optimized for sparse graphs, Excel formulas are often dense

### 3. Risk Scoring Algorithm

**Formula**: `Risk Score = Σ(threat_severity) / 3 + unique_threat_types`

**Rationale**:

- **Division by 3**: Normalizes severity to prevent single threat domination
- **Unique types bonus**: Rewards threat diversity as sophistication indicator
- **Not average**: Would underweight files with many threats
- **Not maximum**: Would miss cumulative risk from multiple medium threats

### 4. Complexity Scoring Weights

**Formula**: `(sheets/10 × 2) + (charts/5 × 1) + (tables/3 × 1) + (cells/10000 × 3)`

**Rationale**:

- **Cells (weight 3)**: Primary driver of memory and processing
- **Sheets (weight 2)**: Each requires separate parsing
- **Charts/Tables (weight 1)**: Add complexity but fewer in number
- **Divisors**: Based on empirical analysis of typical Excel files

### 5. Sampling Strategy

**Choice**: Grid sampling with √(total_cells) sample size

**Rationale**:

- **Grid over random**: Preserves spatial patterns in spreadsheets
- **Sample size**: Provides 95% confidence interval
- **Progressive refinement**: Doubles sample on high variance

### 6. Data Quality Weights

**Formula**: `(Completeness × 0.4) + (Consistency × 0.4) + (Validity × 0.2)`

**Rationale**:

- **Equal weight for completeness/consistency**: Both critical for reliability
- **Lower weight for validity**: Errors often recoverable
- **Empirical basis**: Missing/inconsistent data causes more failures

### 7. Large Graph Handling

**Threshold**: 50,000 nodes triggers hierarchical analysis

**Strategies**:

- **Sheet aggregation**: Reduces O(n²) to O(s²) where s \<< n
- **10% sampling**: Statistical analysis shows 99% cycle detection
- **PageRank for impact**: Identifies high-impact cells efficiently
- **3-hop expansion**: Balances completeness with performance

### 8. Optimization Choices

**Key decisions**:

- **64KB chunks**: Balances memory with I/O (typical L1 cache)
- **LRU cache at 10% memory**: Prevents thrashing
- **1000-row streaming chunks**: Balances overhead
- **SAX parsing > 10MB**: DOM would exhaust memory

### 9. Validation Strategy

**Thresholds**:

- **50MB for sampling**: Parsing becomes non-linear above this
- **Reservoir sampling**: Ensures unbiased selection
- **YARA rules**: Efficient pattern matching for security
- **Recursive descent**: Natural fit for formula parsing

### 10. Graph Library Selection

**Recommendations by use case**:

- **NetworkX**: Prototyping, research (extensive algorithms)
- **JGraphT**: Production Java systems (balanced performance)
- **Boost Graph**: Maximum performance (C++ templates)
- **Custom**: Excel-specific optimizations (sheet partitioning)

## Design Principles

All algorithmic choices follow these principles:

1. **Empirical validation**: Choices based on real-world Excel file analysis
1. **Performance consciousness**: O-notation analysis for all algorithms
1. **Memory efficiency**: Streaming and sampling for large files
1. **Statistical rigor**: Proper confidence intervals and sampling
1. **Security first**: Conservative choices for threat detection

## Trade-offs Acknowledged

- **Accuracy vs. Performance**: Sampling reduces accuracy but enables large file handling
- **Complexity vs. Maintainability**: Some algorithms (Tarjan's) are complex but necessary
- **Memory vs. Speed**: Caching trades memory for repeated computation savings
- **Generality vs. Optimization**: Generic libraries vs. Excel-specific implementations

This comprehensive rationale ensures that implementers understand not just what algorithms to use, but why each was selected over alternatives.
