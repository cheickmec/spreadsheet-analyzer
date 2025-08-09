# Experiments Directory

## Purpose

This directory contains **self-contained experiments** for testing and developing various LLM implementations, agents, and innovative approaches to spreadsheet analysis. Each experiment is designed to be completely independent and isolated from the main codebase.

## Key Principles

### 1. Complete Self-Containment

- **No external dependencies**: Experiments must NOT import from modules outside this directory
- **Standalone execution**: Each experiment can run independently without the main application
- **Local imports only**: All shared utilities must be duplicated within the experiments directory
- **Independent configuration**: Each experiment manages its own settings and environment

### 2. Verbose Logging

Every experiment MUST implement comprehensive logging that includes:

- **Detailed trace logs**: Every function call, decision point, and data transformation
- **LLM interactions**: Complete request/response pairs with token counts
- **Performance metrics**: Execution times, memory usage, token costs
- **Error traces**: Full stack traces and recovery attempts
- **State transitions**: Clear logging of all state changes in agents/workflows

### 3. Output and Log File Requirements

Each experiment must produce structured output files with consistent naming:

```
experiments/
  experiment_name/
    outputs/
      # File naming pattern: {module_name}_{timestamp}_{hash[:8]}_{type}.{ext}
      semantic_detector_20250807_183045_a3f2b891_main.log
      semantic_detector_20250807_183045_a3f2b891_llm_trace.log
      semantic_detector_20250807_183045_a3f2b891_metrics.json
      semantic_detector_20250807_183045_a3f2b891_errors.log
      semantic_detector_20250807_183045_a3f2b891_results.json
```

**File Naming Convention:**

- `{module_name}`: Name of the experiment module
- `{timestamp}`: YYYYMMDD_HHMMSS format (comes second for chronological sorting)
- `{hash[:8]}`: First 8 characters of the module's SHA-256 hash
- `{type}`: Log type (main, llm_trace, metrics, errors, results)
- This naming ensures files sort together by module, then chronologically, with hash for version tracking

## Directory Structure

```
experiments/
├── README.md                           # This file
├── utils.py                            # Common utilities for logging and file naming
├── outputs/                            # All experiment outputs
│   ├── semantic_zones_detector_20250807_183045_a3f2b891_main.log
│   ├── semantic_zones_detector_20250807_183045_a3f2b891_llm_trace.log
│   ├── semantic_zones_detector_20250807_183045_a3f2b891_metrics.json
│   ├── semantic_zones_detector_20250807_183045_a3f2b891_results.json
│   ├── multi_agent_analyzer_20250807_184022_b4c5d123_main.log
│   └── multi_agent_analyzer_20250807_184022_b4c5d123_results.json
├── semantic_zones_detector.py          # Self-contained experiment
├── multi_agent_analyzer.py             # Another self-contained experiment
└── prompt_engineering_test.py          # Yet another experiment
```

## Creating a New Experiment

1. **Create experiment file**: `touch experiments/your_experiment_name.py`

1. **Import and use the common logging utility**:

   ```python
   from pathlib import Path
   import sys

   # Import the common utils from same directory
   sys.path.append(str(Path(__file__).parent))
   from utils import ExperimentLogger

   # Initialize logger for your experiment
   logger = ExperimentLogger(module_path=__file__)

   # Use the various loggers
   logger.main.info("Starting experiment")
   logger.main.debug(f"Configuration: {config}")

   # Log LLM interactions
   logger.log_llm_interaction(
       model="gpt-4",
       prompt="Analyze this spreadsheet...",
       response="I found 3 tables...",
       tokens={"input": 500, "output": 200, "total": 700},
       request_id="req_123"
   )

   # Log metrics
   logger.log_metrics({
       "execution_time": 45.3,
       "total_tokens": 1500,
       "accuracy": 0.92
   })

   # Log errors with full context
   try:
       process_data()
   except Exception as e:
       logger.error.error(f"Processing failed: {e}", exc_info=True)
   ```

1. **Log everything throughout your experiment**:

   ```python
   logger.main.debug(f"Starting experiment")
   logger.main.info(f"Configuration: {json.dumps(config, indent=2)}")
   logger.main.debug(f"Entering function: process_data with args: {args}")
   logger.main.info(f"Data shape before processing: {data.shape}")
   logger.main.debug(f"Transformation applied: {transformation_details}")
   logger.main.info(f"Data shape after processing: {data.shape}")
   logger.main.warning(f"Unexpected value encountered: {value}")
   logger.error.error(f"Failed to process: {error}", exc_info=True)
   ```

## Experiment Requirements

### Must Have:

- [ ] Self-contained implementation (no external imports)
- [ ] Comprehensive logging (every decision point)
- [ ] Docstring at top of .py file with hypothesis and approach
- [ ] Timestamped log files
- [ ] Metrics collection (tokens, time, memory)
- [ ] Error handling with detailed traces
- [ ] Clear success/failure criteria

### Should Have:

- [ ] Reproducible results (seed management)
- [ ] Comparison baselines
- [ ] Visualization of results
- [ ] Cost tracking for LLM calls
- [ ] Performance profiling

### Nice to Have:

- [ ] Interactive notebooks for analysis
- [ ] Automated result summaries
- [ ] Comparative analysis tools

## Example Experiments

### 1. Semantic Zone Detection

Testing different approaches to identify semantic regions in spreadsheets without rigid table boundaries.

### 2. Multi-Agent Orchestration

Experimenting with different agent coordination patterns for complex spreadsheet analysis.

### 3. Prompt Engineering Variations

Testing various prompt structures and their impact on detection accuracy.

### 4. Context Window Optimization

Exploring strategies to maximize useful information within token limits.

### 5. Hybrid Deterministic-AI Approaches

Combining rule-based preprocessing with LLM analysis for better results.

## Best Practices

1. **Version everything**: Include model versions, prompts, and configurations in logs
1. **Track costs**: Log token usage and estimated costs for each LLM call
1. **Fail gracefully**: Experiments should handle errors and log them comprehensively
1. **Document learnings**: Update experiment README with findings and insights
1. **Clean separation**: Never let experiment code leak into main codebase until proven
1. **Comparative testing**: Run multiple variations and log differences

## Running Experiments

Each experiment should be runnable with:

```bash
cd experiments
python semantic_zones_detector.py --verbose --log-level=DEBUG
```

And should produce:

- Detailed console output
- Comprehensive log files in `outputs/` directory
- Metrics JSON file
- Result artifacts

## Important Notes

⚠️ **This directory is for experimentation only**. Code here is NOT production-ready and should NOT be imported by the main application.

⚠️ **Self-containment is critical**. Violating this principle defeats the purpose of isolated experimentation.

⚠️ **Verbose logging is mandatory**. Without detailed traces, experiments lose their value for learning and debugging.

## Contributing

When adding a new experiment:

1. Follow the structure outlined above
1. Ensure complete self-containment
1. Implement comprehensive logging
1. Document your hypothesis and approach
1. Share learnings in the experiment's README

Remember: The goal is rapid experimentation with complete visibility into what's happening, why it's happening, and what we can learn from it.
