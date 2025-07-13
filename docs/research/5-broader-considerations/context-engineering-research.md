# Context Engineering for LLMs: Latest Developments (2023-2024)

## Table of Contents

1. [Overview](#overview)
1. [Sliding Window Strategies and Attention Mechanisms](#sliding-window-strategies-and-attention-mechanisms)
1. [Hierarchical Summarization Techniques](#hierarchical-summarization-techniques)
1. [Graph-Based Context Compression Methods](#graph-based-context-compression-methods)
1. [Dynamic Priority Systems for Context Selection](#dynamic-priority-systems-for-context-selection)
1. [Context Window Optimization for Large Documents](#context-window-optimization-for-large-documents)
1. [Excel-Specific Context Management](#excel-specific-context-management)
1. [Latest Research Papers and Techniques](#latest-research-papers-and-techniques)
1. [Long-Context Models and Their Limitations](#long-context-models-and-their-limitations)
1. [Memory-Efficient Approaches](#memory-efficient-approaches)
1. [Practical Implementations](#practical-implementations)
1. [Benchmarks for Context Utilization](#benchmarks-for-context-utilization)

## Overview

Context engineering has emerged as "the delicate art and science of filling the context window with just the right information for the next step" (Andrej Karpathy). This discipline extends beyond prompt engineering to focus on optimizing how information is provided to LLMs within their context windows.

Key components of context in LLM applications include:

- System prompts/instructions
- User input
- Short-term memory (chat history)
- Long-term memory
- Retrieved information from knowledge bases
- Tool definitions and responses
- Structured outputs
- Global state/context

## Sliding Window Strategies and Attention Mechanisms

### Sliding Window Attention (SWA)

SWA breaks long texts into smaller, manageable segments that overlap, allowing models to process extensive contexts efficiently while maintaining coherence.

### SWAT (Sliding Window Attention Training) - 2025

- Enables efficient long-context handling by addressing the attention sink phenomenon
- Attention sink: LLMs allocate excessive attention to initial tokens due to high variance in softmax operations
- Solution: Removes normalization from attention mechanisms to eliminate the attention sink effect

### Key Characteristics

- Processes documents in overlapping windows
- Maximum attention distance constrained by window size × model depth
- Scales linearly with sequence length (vs quadratic for standard attention)

### Alternative Approaches

- **State-Space Models (SSMs)**: Forgo attention entirely for linear-time processing
- **Mamba**: Language-capable SSM handling sequences up to 1M tokens, 5× faster than Transformers on long sequences
- **Longformer**: Uses sliding window attention that scales linearly with sequence length

## Hierarchical Summarization Techniques

### Core Concept

Hierarchical compression builds a hierarchy where essential details are compressed into first sections, followed by generalized or less essential data.

### Implementation Process

1. **Topic Structure Identification**: Construct graph representations to identify distinct text sections
1. **Block Division**: Divide texts into mutually independent blocks
1. **Parallel Processing**: Process each block independently using pre-trained models
1. **Progressive Summarization**: Concatenate individual summaries and re-summarize for higher-level abstraction

### Benefits

- Extends context window by 6-8× without significant computational costs
- Preserves key ideas and context while condensing information
- Enables efficient handling of lengthy texts

### Challenges

- LLMs suffer from the "middle curse" - struggle to use information in middle of context window
- Most attention focused on beginning and end of context
- Extending beyond 4k tokens may not be justified for current inference setups

## Graph-Based Context Compression Methods

### PROMPT-SAW (2024)

- Leverages relation-aware graphs for textual prompt compression
- Combines graph-based approaches with prompt compression techniques

### AMR-Based Concept Distillation (2024)

- Uses Abstract Meaning Representation (graph-based semantic representation)
- Compresses long context for enhancing RAG systems

### Context-Aware Prompt Compression (CPC)

- Sentence-level compression with context-aware sentence encoder
- Provides relevance scores for each sentence given a question
- Up to 10.93× faster at inference compared to token-level compression

### In-Context Former (IC-Former)

- Uses cross-attention mechanisms and learnable digest tokens
- Requires only 1/32 floating-point operations during compression
- Improves processing speed by 68-112× while maintaining 90% baseline performance

### Recurrent Context Compression (RCC)

- Views attention as special state space model/RNN
- Achieves 90% accuracy in passkey retrieval up to 1000k with 8k context window encoder
- Nearly 100% performance after fine-tuning with 32k sequences

## Dynamic Priority Systems for Context Selection

### LazyLLM Dynamic Token Pruning

- Selectively computes KV cache for important tokens
- Allows dynamic selection of different token subsets in each generation step
- Progressively prunes tokens to reduce computations
- Optimizes next token prediction in each generation step

### Ascendra: Dynamic Request Prioritization

- High-priority instances for low-latency urgent requests
- LP (Low Priority) instances maximize tokens per unit time
- Improves throughput by up to 1.7× compared to vLLM
- Balances Time-to-First-Token (TTFT) and Time-Between-Tokens (TBT) SLOs

### Dynamic Context Switching

- Process files/documents independently with pre-calculated embeddings
- Callback functions analyze LLM output and inject relevant embeddings
- Tailored context management for different agent needs

### Differentiable Tool Selection

- Models tool selection as trainable function
- Runs entirely outside the LLM
- Reduces hallucinations and improves determinism
- Maintains constant context length regardless of tool depth

## Context Window Optimization for Large Documents

### Core Techniques

#### 1. Attention Mechanism Optimizations

- Sparse attention reduces computational load
- Modified Transformer-XL architectures with positional bias
- Optimized attention kernels for extended contexts

#### 2. Input Compression

- Synthetic longform instruction data generation
- Compression at different ratios for later use
- Reduces token count while preserving semantic meaning

#### 3. Architectural Innovations

- Cascading KV Cache: Organizes cache into sub-caches
- Retains critical tokens longer than sliding windows
- 12.13% average improvement in LongBench benchmarks

#### 4. Memory Management

- Cascading KV Cache reduces prefill latency by 6.8× on 1M tokens
- Takes only 14.8% of quadratic method's time

### Context Window Expansions (2024)

- Claude: 9k → 200k tokens
- Gemini 1.5: Up to 2 million tokens
- GPT-4 Turbo: 128k tokens
- OpenAI GPT-4.1 (2025): 1 million tokens

### Best Practices

1. Monitor performance metrics (speed, quality, cost)
1. Be selective about context content
1. Use clear separators (triple quotes, section headers)
1. Balance performance vs. cost considerations

## Excel-Specific Context Management

### SpreadsheetLLM Framework (2024)

Microsoft's pioneering framework for spreadsheet understanding and reasoning:

#### Key Components

1. **Vanilla Encoding**: Serializes spreadsheets with cell addresses and formats
1. **SheetCompressor**: Three-module compression system
   - Structural-anchor-based extraction
   - Inverted-index translation
   - Data-format-aware aggregation

### Challenges

- Context window overrun (2k token limits exceeded)
- Excel sheets passed as single tables break logical collections
- Default chunking strains GPU memory and timeouts
- Poor discrete value lookup on vectorized data
- Excel numeric date encoding issues

### Chain of Spreadsheet (CoS) Framework

- Extends Chain of Thought methodology to spreadsheets
- Decomposes reasoning into table detection-match-reasoning pipeline
- Transforms spreadsheet data management and analysis

### Performance Considerations

- Large spreadsheets contain homogeneous rows/columns with minimal structural value
- Formula dependencies require special handling
- Cell relationships and references need context preservation

## Latest Research Papers and Techniques

### Context Window Extension Methods

#### SelfExtend (2024)

- Extends context without fine-tuning
- Constructs bi-level attention: grouped and neighbor attention
- Minor code modifications extend existing LLMs' context windows

#### LongRoPE (2024)

- Extends context to 2048k tokens
- Only 1k fine-tuning steps at 256k training lengths
- Three innovations:
  1. Exploits non-uniformities in positional interpolation
  1. Progressive extension strategy
  1. Readjusts on 8k length for short context recovery

#### Position Interpolation (2023)

- Extends RoPE-based LLMs to 32768 tokens
- Minimal fine-tuning (within 1000 steps)
- Preserves quality on original context window tasks

### Context Management Strategies

Four main approaches:

1. **Write**: Scratchpads for saving context
1. **Select**: Choosing relevant memories
1. **Compress**: Summarization and pruning
1. **Isolate**: Splitting across sub-agents

### Semantic Compression (2023)

- Inspired by lossy source coding
- Extends context by 6-8× without fine-tuning
- Reduces semantic redundancy before LLM processing

### Information-Intensive Training (2024)

- Addresses lost-in-the-middle challenge
- Synthesized long-context Q&A datasets
- Requires fine-grained information awareness (128 tokens within 4K-32K context)

### Data Engineering for Long Context (2024)

- 500M-5B tokens sufficient for 128K context retrieval
- Emphasizes domain balance and length upsampling
- Continual pretraining effective for scaling context

## Long-Context Models and Their Limitations

### Current Model Capabilities (2024)

#### Claude 3

- 200k token context window
- Superior for legal documents, codebases
- Better at "needle in haystack" problems
- Outperforms GPT-4o on DROP benchmark

#### GPT-4 Turbo

- 128k token context window
- 8× larger than GPT-3.5 Turbo's 16k
- Strong general performance

#### Gemini 1.5 Pro

- 1 million token context window
- Handles 11 hours audio, 1 hour video, 30k+ lines of code
- 2 million token waitlist available
- Best for multimodal applications

### Limitations

- Quadratic complexity remains fundamental challenge
- Attention sink phenomenon affects performance
- Information loss inevitable in ultra-long sequences
- Middle curse: difficulty using middle context information

## Memory-Efficient Approaches

### KV Cache Quantization Techniques

#### KVQuant (NeurIPS 2024)

- Per-Channel Key Quantization
- Pre-RoPE Key Quantization
- Non-Uniform KV Cache Quantization
- Per-Vector Dense-and-Sparse Quantization
- Achieves \<0.1 perplexity degradation with 3-bit quantization
- Enables 1M context on single A100-80GB GPU

#### Coupled Quantization (CQ)

- Channel-coupled quantization for keys and values
- Exploits mutual dependency between channels
- Better quality preservation under low bit settings

#### KIVI: 2-bit Quantization

- Asymmetrical quantization without quality degradation
- Per-channel key cache, per-token value cache
- Addresses outlier patterns in keys vs values

### Framework Support

- FlexGen: 4-bit KV cache and weights
- NVIDIA TensorRT-LLM: 8-bit formats (INT8/FP8)
- vLLM: FP8 quantization since v0.3.0

### Performance Impact

- 2-3× reduction in KV cache size
- Frees tens of gigabytes of memory
- ~2.5× memory saving with int4
- Trade-off: Generation speed decreases with higher batch sizes

## Practical Implementations

### GitHub Projects

#### llm-context.py

- Smart Code Outlines for codebase structure
- .gitignore pattern integration
- Model Context Protocol (MCP) support
- Developed with Claude Sonnets collaboration

#### code-context-llm

- Generates Markdown codebase representations
- Secure content redaction
- 50% development time savings potential
- 70% documentation effort reduction

#### Context-Engineering Repository

- First-principles handbook
- Three-stage architecture for abstract reasoning
- Symbolic induction/abstraction/retrieval heads
- Cognitive tools as structured prompt templates

### Implementation Strategies

#### Four-Bucket Approach

1. **Write**: Tools or runtime state for context saving
1. **Select**: Episodic, procedural, semantic memories
1. **Compress**: LLM summarization or heuristic trimming
1. **Isolate**: Sub-agents with separate contexts

### Large-Scale Frameworks

- **Langfuse**: Open source LLM engineering platform
- **AdalFlow**: Auto-optimization library for LLM applications

## Benchmarks for Context Utilization

### Key Metrics

- Perplexity degradation
- Passkey retrieval accuracy
- Processing speed (tokens/second)
- Memory usage
- Inference latency

### Benchmark Results

- KVQuant: \<0.1 perplexity degradation at 3-bit
- RCC: 90% passkey retrieval at 1000k with 8k encoder
- IC-Former: 90% baseline performance at 68-112× speed
- Cascading KV Cache: 12.13% LongBench improvement

### Trade-offs

- Memory vs Speed: Lower quantization = slower decoding
- Context Length vs Quality: Longer contexts = attention dilution
- Cost vs Performance: Larger contexts = higher API costs

## Conclusion

Context engineering in 2024-2025 represents a critical frontier in LLM development, with advances enabling:

- Million+ token context windows
- Efficient compression maintaining semantic integrity
- Dynamic, adaptive context management
- Specialized handling for structured data (Excel)
- Practical tools and frameworks for implementation

The field continues to evolve rapidly, balancing computational efficiency, accuracy, and cost while pushing the boundaries of what's possible with large language models.
