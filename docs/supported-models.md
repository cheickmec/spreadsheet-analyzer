# Supported LLM Models

The Spreadsheet Analyzer supports multiple LLM providers and models for analysis. Each provider requires its own API key set as an environment variable.

## Quick Reference

| Provider           | Environment Variable | Example Models                                       |
| ------------------ | -------------------- | ---------------------------------------------------- |
| Anthropic (Claude) | `ANTHROPIC_API_KEY`  | `claude-sonnet-4-20250514`, `claude-opus-4-20250514` |
| OpenAI (GPT)       | `OPENAI_API_KEY`     | `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`              |
| Google (Gemini)    | `GEMINI_API_KEY`     | `gemini-2.5-pro`, `gemini-1.5-pro`                   |
| Ollama (Local)     | None required        | `llama3.1:8b`, `mixtral:8x7b`, `qwen2.5`             |

## Provider Details

### Anthropic Claude Models

Claude models are the default and recommended choice for spreadsheet analysis due to their strong analytical capabilities.

```bash
export ANTHROPIC_API_KEY='your-api-key'

# Use Claude Sonnet (default)
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx

# Use Claude Opus for more complex analysis
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model claude-opus-4-20250514
```

### OpenAI GPT Models

GPT models provide reliable analysis with good cost-performance balance.

```bash
export OPENAI_API_KEY='your-api-key'

# Use GPT-4
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model gpt-4

# Use GPT-4 Turbo for faster responses
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model gpt-4-turbo
```

### Google Gemini Models

Gemini Pro models offer state-of-the-art performance with advanced reasoning capabilities.

```bash
export GEMINI_API_KEY='your-api-key'

# Use Gemini 2.5 Pro (recommended)
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model gemini-2.5-pro

# Use Gemini 1.5 Pro
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model gemini-1.5-pro

# Model name alias
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model gemini-pro    # Maps to gemini-2.5-pro
```

#### Gemini Model Features

- **Context Window**: Up to 2M tokens (depending on model variant)
- **Multimodal Support**: Can analyze images embedded in spreadsheets
- **Advanced Reasoning**: Supports complex analysis workflows
- **Tool Calling**: Gemini Pro models have reliable tool calling support

**Note**: Gemini Flash models are not supported due to tool calling compatibility issues.

### Ollama Local Models

Ollama models run locally without requiring API keys. Ensure Ollama is installed and running.

```bash
# Start Ollama service
ollama serve

# Pull a model
ollama pull llama3.1:8b

# Use with spreadsheet analyzer
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model llama3.1:8b
```

See [ollama-models.md](./ollama-models.md) for a complete list of supported Ollama models.

## Model Selection Guidelines

### For Best Quality Analysis

- **Claude Opus 4**: Most thorough analysis, best for complex spreadsheets
- **Gemini 2.5 Pro**: Excellent reasoning, good for financial/technical data
- **GPT-4**: Reliable all-around performance

### For Cost-Effective Analysis

- **Claude Sonnet 4**: Good balance of quality and cost (default)
- **Gemini 1.5 Pro**: More affordable than 2.5 Pro with good quality
- **GPT-3.5 Turbo**: Budget option for simpler analysis

### For Privacy/Offline Use

- **Ollama Models**: Run entirely locally
- **Llama 3.1**: Best open-source option
- **Mixtral**: Good for code-heavy spreadsheets

## Context Window Considerations

Different models have different context window sizes, affecting how much data they can analyze at once:

- **Gemini Pro Models**: Up to 2M tokens (largest)
- **Claude 3**: Up to 200K tokens
- **GPT-4 Turbo**: 128K tokens
- **GPT-4**: 8K tokens (may require context compression)
- **Ollama Models**: Varies (4K-128K depending on model)

The analyzer includes automatic context compression for models with smaller windows.

## Setting Multiple API Keys

You can set multiple API keys to switch between providers:

```bash
# In your shell profile or .env file
export ANTHROPIC_API_KEY='your-anthropic-key'
export OPENAI_API_KEY='your-openai-key'
export GEMINI_API_KEY='your-gemini-key'

# Then choose model at runtime
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model gemini-2.5-pro
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model gpt-4
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model claude-sonnet-4-20250514
```

## Cost Tracking

The analyzer includes built-in cost tracking for all API-based models. See cost reports in:

- Real-time: During analysis with `--track-costs`
- Post-analysis: In `*_cost_tracking.json` files

## Troubleshooting

### API Key Not Found

```
Error: No API key provided. Set GEMINI_API_KEY or use --api-key
```

**Solution**: Export the appropriate environment variable for your chosen model.

### Model Not Recognized

```
Error: Unsupported model: [model-name]
```

**Solution**: Check the model name spelling and ensure it's in the supported list above.

### Context Length Exceeded

```
Error: Context window exceeded
```

**Solution**: The analyzer will automatically apply compression. For GPT-4 (8K context), consider using GPT-4 Turbo or Gemini models for larger spreadsheets.
