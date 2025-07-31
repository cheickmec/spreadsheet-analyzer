# Observability with Arize Phoenix

This guide explains how to set up and use Arize Phoenix for observability and cost tracking in the Spreadsheet Analyzer.

## Overview

The Spreadsheet Analyzer integrates with [Arize Phoenix](https://github.com/Arize-ai/phoenix) to provide:

- **LLM Tracing**: Track all LLM calls, tokens, and latency
- **Agent Workflow Visualization**: See interactive flowcharts of analysis steps
- **Cost Tracking**: Monitor spending with LiteLLM's up-to-date pricing
- **Evaluation**: Test and debug your analysis workflows

## Quick Start

### 1. Local Phoenix (Easiest)

```bash
# Phoenix will launch automatically when you run the analyzer
python -m spreadsheet_analyzer.notebook_cli data.xlsx
```

Access Phoenix UI at: http://localhost:6006

### 2. Docker Phoenix (Recommended)

```bash
# Start Phoenix container
docker-compose -f docker-compose.phoenix.yml up -d

# Run analysis with Docker mode
python -m spreadsheet_analyzer.notebook_cli data.xlsx --phoenix-mode docker
```

### 3. Phoenix Cloud

```bash
# Set your API key
export PHOENIX_API_KEY=your-key-here

# Run with cloud mode
python -m spreadsheet_analyzer.notebook_cli data.xlsx --phoenix-mode cloud
```

## Configuration

### Environment Variables

Copy `phoenix.env.example` to `phoenix.env` and configure:

```bash
# Phoenix settings
PHOENIX_MODE=docker
PHOENIX_HOST=localhost
PHOENIX_PORT=6006

# Cost tracking
COST_LIMIT=10.0

# API keys
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key
```

### Command Line Options

```bash
# Set Phoenix mode
--phoenix-mode {local,docker,cloud,none}

# Configure host/port
--phoenix-host localhost
--phoenix-port 6006

# Set project name
--phoenix-project my-analysis

# Cost tracking
--cost-limit 5.0
--track-costs
```

## Features

### 1. LLM Tracing

Phoenix automatically traces:

- All LLM calls (Anthropic, OpenAI)
- Token usage (input/output)
- Latency and errors
- Tool calls and responses

View traces at: http://localhost:6006/traces

### 2. Cost Tracking

The system uses LiteLLM for accurate, up-to-date pricing:

```bash
# Set a spending limit
python -m spreadsheet_analyzer.notebook_cli data.xlsx --cost-limit 5.0

# Cost summary is shown after analysis:
💰 Cost Summary:
  Total Cost: $0.1234
  Total Tokens: 15,432
  Cost by Model:
    claude-3-5-sonnet: $0.1234
  Budget Status: ✅ Within limit ($5.00)
```

Cost tracking data is saved to: `logs/YYYYMMDD/{filename}_cost_tracking.json`

### 3. Agent Workflow Visualization

Phoenix provides an interactive flowchart showing:

- Analysis steps and decisions
- Tool calls and results
- Errors and bottlenecks

Access via the "Agents" tab in Phoenix UI.

### 4. Session Management

Each analysis creates a unique session with:

- Structured file naming
- LLM interaction logs
- Cost tracking data
- Generated notebooks

Example file structure:

```
analysis_results/20240125/
  ├── sales_data_claude_sonnet_sheet0_Dashboard.ipynb
  └── sales_data_claude_sonnet_sheet0_Dashboard_llm_log.txt
logs/20240125/
  └── sales_data_claude_sonnet_sheet0_Dashboard_cost_tracking.json
```

## Docker Deployment

### Basic Setup

```yaml
# docker-compose.phoenix.yml
services:
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"  # UI
      - "4317:4317"  # gRPC
    volumes:
      - phoenix_data:/mnt/data
```

### Production Setup (PostgreSQL)

For production, use PostgreSQL:

```yaml
services:
  phoenix:
    image: arizephoenix/phoenix:latest
    environment:
      - PHOENIX_SQL_DATABASE_URL=postgresql://phoenix:password@postgres:5432/phoenix_db
    depends_on:
      - postgres
      
  postgres:
    image: postgres:14
    environment:
      - POSTGRES_USER=phoenix
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=phoenix_db
```

## Troubleshooting

### Phoenix Won't Start

```bash
# Check if port is in use
lsof -i :6006

# View container logs
docker logs spreadsheet-analyzer-phoenix
```

### No Traces Appearing

1. Check Phoenix is running: http://localhost:6006
1. Verify mode: `--phoenix-mode docker` (not `none`)
1. Check instrumentation in logs

### Cost Tracking Issues

1. Ensure `--track-costs` is enabled (default)
1. Check cost tracking file in `logs/` directory
1. Verify model names match LiteLLM's pricing data

## Advanced Usage

### Custom Instrumentation

```python
from spreadsheet_analyzer.observability import (
    initialize_phoenix,
    instrument_all,
    get_cost_tracker
)

# Initialize Phoenix
tracer = initialize_phoenix()

# Instrument providers
instrument_all(tracer)

# Track costs
tracker = get_cost_tracker()
tracker.track_usage(
    model="gpt-4",
    input_tokens=100,
    output_tokens=200
)
```

### Exporting Data

Phoenix data can be exported for further analysis:

```bash
# Export traces (from Phoenix UI)
http://localhost:6006/exports

# Cost data (JSON format)
cat logs/YYYYMMDD/*_cost_tracking.json
```

## Best Practices

1. **Use Docker mode** for consistent deployments
1. **Set cost limits** to avoid unexpected charges
1. **Review traces** to optimize prompts and reduce tokens
1. **Monitor the Agents view** to debug complex workflows
1. **Archive Phoenix data** periodically for long-term analysis

## Resources

- [Phoenix Documentation](https://docs.arize.com/phoenix)
- [LiteLLM Pricing](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json)
- [OpenTelemetry](https://opentelemetry.io/)
