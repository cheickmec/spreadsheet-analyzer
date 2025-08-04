# Migration Guide: Phoenix Observability Integration

This guide helps you migrate from the standard notebook CLI to the Phoenix-enabled version with observability and cost tracking.

## What's New

The Phoenix integration adds:

- 🔍 **LLM Tracing**: See all LLM calls, tokens, and latency
- 📊 **Agent Visualization**: Interactive flowcharts of analysis workflows
- 💰 **Cost Tracking**: Real-time spending monitoring with LiteLLM
- 🐛 **Debugging**: Trace errors and optimize prompts

## Quick Migration

### 1. Install Dependencies

```bash
# Update dependencies
uv pip install -e .

# Or manually install:
uv pip install arize-phoenix>=4.29.0 litellm>=1.0.0 \
  openinference-instrumentation-langchain \
  openinference-instrumentation-openai \
  openinference-instrumentation-anthropic
```

### 2. Start Phoenix (Docker)

```bash
# Start Phoenix container
./scripts/start_phoenix.sh

# Or manually:
docker-compose -f docker-compose.phoenix.yml up -d
```

### 3. Update Your Commands

Replace your existing commands:

```bash
# Old command
python -m spreadsheet_analyzer.notebook_cli data.xlsx

# New command with Phoenix
python -m spreadsheet_analyzer.notebook_cli_phoenix data.xlsx --phoenix-mode docker
```

## Command Comparison

| Feature        | Old CLI                  | Phoenix CLI                      |
| -------------- | ------------------------ | -------------------------------- |
| Basic analysis | `notebook_cli data.xlsx` | `notebook_cli_phoenix data.xlsx` |
| Set model      | `--model gpt-4`          | `--model gpt-4`                  |
| Cost limit     | Not available            | `--cost-limit 5.0`               |
| Observability  | Not available            | `--phoenix-mode docker`          |
| Cost tracking  | Manual                   | Automatic with reports           |

## New Features

### Cost Tracking

```bash
# Set spending limit
python -m spreadsheet_analyzer.notebook_cli_phoenix data.xlsx --cost-limit 10.0

# Output includes cost summary:
💰 Cost Summary:
  Total Cost: $0.1234
  Total Tokens: 15,432
  Budget Status: ✅ Within limit ($10.00)
```

### Phoenix Modes

```bash
# Local mode (default - Phoenix launches automatically)
--phoenix-mode local

# Docker mode (recommended for production)
--phoenix-mode docker

# Cloud mode (requires API key)
--phoenix-mode cloud --phoenix-api-key YOUR_KEY

# Disable observability
--phoenix-mode none
```

### File Organization

New structure includes cost tracking:

```
analysis_results/20240125/
  ├── sales_data_claude_sonnet_sheet0.ipynb
  └── sales_data_claude_sonnet_sheet0_llm_log.txt
logs/20240125/
  └── sales_data_claude_sonnet_sheet0_cost_tracking.json
```

## Environment Setup

Create `phoenix.env`:

```bash
# Copy example
cp phoenix.env.example phoenix.env

# Edit with your settings
PHOENIX_MODE=docker
COST_LIMIT=10.0
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key
GEMINI_API_KEY=your-key
```

## Viewing Results

1. **Phoenix UI**: http://localhost:6006

   - Traces tab: See all LLM calls
   - Agents tab: Visualize workflows

1. **Cost Reports**: Check `logs/YYYYMMDD/*_cost_tracking.json`

1. **LLM Logs**: Review `logs/YYYYMMDD/*_llm_log.txt`

## Troubleshooting

### Phoenix not starting

```bash
# Check status
docker ps | grep phoenix

# View logs
docker logs spreadsheet-analyzer-phoenix
```

### No traces appearing

- Ensure `--phoenix-mode docker` is set
- Check Phoenix UI is accessible
- Verify API keys are set

### Cost tracking issues

- Costs appear after analysis completes
- Check cost tracking JSON file
- Ensure model names match LiteLLM's database

## Rollback

To use the old CLI without Phoenix:

```bash
# Original CLI still available
python -m spreadsheet_analyzer.notebook_cli data.xlsx
```

## Testing

Test the integration:

```bash
# Run integration test
python scripts/test_phoenix_integration.py

# Test with sample file
python -m spreadsheet_analyzer.notebook_cli_phoenix \
  tests/fixtures/excel/financial_sample.xlsx \
  --phoenix-mode docker \
  --max-rounds 2
```

## Benefits

1. **Cost Control**: Set limits, track spending
1. **Performance**: Identify slow operations
1. **Debugging**: Trace errors to specific steps
1. **Optimization**: Reduce tokens with insights
1. **Compliance**: Audit trail of all LLM usage

## Next Steps

1. Start Phoenix: `./scripts/start_phoenix.sh`
1. Run analysis with new CLI
1. Explore traces at http://localhost:6006
1. Review cost reports
1. Optimize based on insights

For more details, see [docs/observability.md](docs/observability.md).
