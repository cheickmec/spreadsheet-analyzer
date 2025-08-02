# Ollama Model Support for Spreadsheet Analyzer

This document lists Ollama models that support tool/function calling, which is required for the notebook CLI's interactive analysis features.

## Supported Models

### Recommended Models

These models have been tested and work well with the spreadsheet analyzer:

1. **llama3.1:8b** (Default) - Best overall for tool calling

   - Context: 131K tokens
   - Size: ~4.7 GB
   - Excellent coding and analysis capabilities

1. **qwen2.5:7b** - Strong coding model

   - Context: 131K tokens
   - Size: ~4.4 GB
   - Good for technical analysis

1. **mistral:v0.3** - Mistral with tool support

   - Context: 32K tokens
   - Size: ~4.1 GB
   - Note: Must use v0.3 tag for tool support

1. **MFDoom/deepseek-r1-tool-calling:32b** - DeepSeek with tool support

   - Context: 131K tokens
   - Size: ~19 GB
   - Experimental but powerful for coding

### Complete List of Supported Models

#### Llama Family

- llama3.1, llama3.1:8b, llama3.1:70b, llama3.1:405b
- llama3.2, llama3.2:1b, llama3.2:3b
- llama3.3

#### Mistral Family

- mistral:v0.3 (Note: base mistral without v0.3 does NOT support tools)
- mistral-nemo
- mistral-small, mistral-small3.1, mistral-small3.2

#### Qwen Family

- qwen2, qwen2.5, qwen2.5:7b
- qwen2.5-coder
- qwen3

#### DeepSeek Family (Custom versions)

- MFDoom/deepseek-r1-tool-calling
- MFDoom/deepseek-r1-tool-calling:7b
- MFDoom/deepseek-r1-tool-calling:32b

#### Other Models

- command-r, command-r-plus, command-r7b
- firefunction-v2
- granite3-dense, granite3.1-dense, granite3.2
- nemotron, nemotron-mini
- hermes3
- phi4, phi4-mini

## Installation

To install a supported model:

```bash
# Install the default model (recommended)
ollama pull llama3.1:8b

# Install alternative models
ollama pull qwen2.5:7b
ollama pull mistral:v0.3
ollama pull MFDoom/deepseek-r1-tool-calling:32b
```

## Usage

Use a supported model with the notebook CLI:

```bash
# Using the default model
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx

# Specifying a different model
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model llama3.1:8b
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model qwen2.5:7b
uv run src/spreadsheet_analyzer/notebook_cli.py data.xlsx --model MFDoom/deepseek-r1-tool-calling:32b
```

## Troubleshooting

### Error: "Model does not support tools"

This error occurs when using a model without tool calling support. Common causes:

1. **Using base Mistral without v0.3 tag**

   - ❌ Wrong: `mistral:7b-instruct`
   - ✅ Correct: `mistral:v0.3`

1. **Using standard DeepSeek models**

   - ❌ Wrong: `deepseek-r1:32b`
   - ✅ Correct: `MFDoom/deepseek-r1-tool-calling:32b`

1. **Model not installed**

   - Check installed models: `ollama list`
   - Install a supported model: `ollama pull llama3.1:8b`

1. **Ollama service not running**

   - Start Ollama: `ollama serve`
   - Check if running: `curl http://localhost:11434/api/tags`

### Performance Considerations

- **Small models (1-8B)**: Fast, suitable for most analysis tasks
- **Medium models (8-32B)**: Better reasoning, more memory required
- **Large models (70B+)**: Best quality but require significant resources

### Context Window Sizes

Different models support different context windows:

- **Large context (128K+)**: llama3.1, llama3.2, qwen2.5, deepseek-r1
- **Medium context (32-64K)**: mistral, mixtral
- **Small context (4-8K)**: older models like llama2

Choose based on your Excel file complexity and analysis needs.
