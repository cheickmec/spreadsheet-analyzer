"""LLM interaction logic for notebook analysis.

This module handles all LLM-specific interactions, keeping them separate
from the core analysis orchestration.

CLAUDE-KNOWLEDGE: LLM interactions are isolated to make it easier to
swap providers or modify prompts without affecting core logic.
"""

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from structlog import get_logger

from ..core.types import Result, err, ok
from ..notebook_llm_interface import get_notebook_tools
from ..notebook_session import NotebookSession
from ..observability import add_session_metadata, phoenix_session
from ..prompts import load_prompt
from .context_compression import HierarchicalContextCompressor
from .notebook_analysis import AnalysisConfig, AnalysisState, save_notebook

if TYPE_CHECKING:
    from .thinking_config import ThinkingConfig

logger = get_logger(__name__)


def generate_gemini_error_message(tool_name: str, tools: list[Any]) -> str:
    """Generate an appropriate error message for Gemini tool call mistakes.

    Args:
        tool_name: Name of the incorrectly called tool
        tools: List of available tools

    Returns:
        A descriptive error message
    """
    # Special handling for common Gemini mistakes with pandas DataFrame methods
    if tool_name in ["to_markdown", "tolist", "head", "tail", "describe", "info"]:
        return (
            f"ERROR: '{tool_name}' is a pandas DataFrame method, NOT a tool!\n"
            f"To use DataFrame methods, you MUST use execute_code tool.\n"
            f'Example: execute_code(code="df.{tool_name}()")\n'
            f"Please retry using the execute_code tool."
        )
    else:
        # For other unknown tools
        return (
            f"Unknown tool '{tool_name}'. Available tools are: "
            f"{', '.join([t.name for t in tools[:5]])}... "
            f"Please use one of the available tools."
        )


def get_gemini_messages(system_prompt: str, initial_prompt: str) -> list[Any]:
    """Create Gemini-specific message sequence with tool usage clarification.

    Args:
        system_prompt: The system prompt for the analysis
        initial_prompt: The initial human prompt for the analysis

    Returns:
        A list of messages tailored for Gemini tool calling
    """
    # CLAUDE-KNOWLEDGE: Gemini models require explicit instructions about tool usage
    # because they tend to confuse DataFrame methods with callable tools. This is
    # a known issue documented in multiple GitHub issues (e.g., langchain #26083).
    # Providing concrete examples significantly improves tool calling success rate.
    tool_clarification = SystemMessage(
        content="""
CRITICAL GEMINI-SPECIFIC INSTRUCTIONS - YOU MUST FOLLOW THESE:

1. DO NOT call pandas DataFrame methods as tools. These are NOT tools:
   - to_markdown, tolist, head, tail, describe, info, etc.

2. To run ANY Python code, you MUST use the execute_code tool:
   CORRECT: execute_code(code="df.head(30)")
   WRONG: df.head(30) or head(df)

3. For multi-table detection, use this EXACT pattern:
   execute_code(code="# Multi-table detection\\nprint(f'Sheet dimensions: {df.shape}')\\nprint('\\\\n--- First 30 rows ---')\\nprint(df.head(30))")

4. CONCRETE EXAMPLE of proper tool usage:
   execute_code(code=\"\"\"
# Analyze the data structure
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\\n{df.dtypes}")

# Check for empty rows
empty_rows = df.isnull().all(axis=1)
if empty_rows.any():
    print(f"Empty rows at indices: {empty_rows[empty_rows].index.tolist()}")
\"\"\")

5. Available tools you can call:
   - execute_code: Run Python code
   - add_markdown_cell: Add documentation
   - get_formula_statistics: Get formula stats

NEVER attempt to call methods like to_markdown() or tolist() as tools!
Always use execute_code(code="...") for ALL Python operations.
"""
    )

    return [
        SystemMessage(content=system_prompt),
        tool_clarification,
        HumanMessage(content=initial_prompt),
    ]


def create_llm_instance(
    model: str, api_key: str | None = None, thinking_config: "ThinkingConfig | None" = None
) -> Result[Any, str]:
    """Create LLM instance based on model selection.

    Args:
        model: Model name
        api_key: Optional API key

    Returns:
        Result containing LLM instance or error
    """
    try:
        if "claude" in model.lower():
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return err("No API key provided. Set ANTHROPIC_API_KEY or use --api-key")

            # Prepare base parameters
            llm_kwargs = {
                "model_name": model,
                "api_key": api_key,
                "max_tokens": 4096,
            }

            # Add thinking parameters if available
            if thinking_config and thinking_config.enabled:
                # For now, we'll pass thinking parameters through model_kwargs
                # This will be used by our custom wrapper later
                llm_kwargs["model_kwargs"] = thinking_config.to_api_params()

                # Add beta headers if needed
                beta_headers = thinking_config.to_beta_headers()
                if beta_headers:
                    if "model_kwargs" not in llm_kwargs:
                        llm_kwargs["model_kwargs"] = {}
                    llm_kwargs["model_kwargs"]["extra_headers"] = beta_headers

                logger.info(f"🧠 Extended thinking enabled: {thinking_config.budget_tokens:,} tokens")
                if thinking_config.interleaved:
                    logger.info("⚡ Interleaved thinking enabled for tool use")

            llm = ChatAnthropic(**llm_kwargs)
            return ok(llm)

        elif "gpt" in model.lower():
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                return err("No API key provided. Set OPENAI_API_KEY or use --api-key")

            llm = ChatOpenAI(
                model_name=model,
                api_key=api_key,
                temperature=0,
            )
            return ok(llm)

        elif "gemini" in model.lower():
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                return err("No API key provided. Set GEMINI_API_KEY or use --api-key")

            # CLAUDE-KNOWLEDGE: Gemini 2.X models use simplified function calling where the model
            # handles function execution under the covers. They support compositional/sequential
            # function calling, allowing chaining of multiple function calls across turns.
            # Source: Google documentation and research (2024-2025)

            # Map common model names to official Gemini model IDs
            model_mapping = {
                "gemini-2.5-pro": "models/gemini-2.5-pro",
                "gemini-2.5-flash": "models/gemini-2.5-flash",  # CLAUDE-PERFORMANCE: Flash has better price/performance ratio
                "gemini-pro": "models/gemini-2.5-pro",  # Alias for 2.5 Pro
                "gemini-1.5-pro": "models/gemini-1.5-pro",
            }

            # CLAUDE-GOTCHA: Gemini API requires ALL model names to have "models/" prefix
            # Without this prefix, API calls will fail with model not found errors
            # e.g., MUST use "models/gemini-2.5-pro" not just "gemini-2.5-pro"
            actual_model = model_mapping.get(model.lower(), f"models/{model.lower()}")

            logger.info(f"Using Gemini model: {actual_model}")

            # CLAUDE-IMPORTANT: The disable_streaming parameter is CRITICAL for tool calling
            # When set to "tool_calling" (not True/False), it bypasses streaming ONLY when
            # tools are bound, preventing tool calling issues while maintaining streaming
            # for regular chat responses. This is the recommended configuration.
            #
            # CLAUDE-GOTCHA: Gemini often returns empty content when making tool calls,
            # unlike OpenAI models. Always check tool_calls first, regardless of content.
            llm = ChatGoogleGenerativeAI(
                model=actual_model,
                api_key=api_key,
                temperature=0,
                max_tokens=None,  # Let Gemini use its default
                max_retries=2,
                disable_streaming="tool_calling",  # CRITICAL: Must be exactly "tool_calling" string
            )
            return ok(llm)

        elif any(
            name in model.lower()
            for name in ["ollama", "mistral", "llama", "mixtral", "codellama", "qwen", "deepseek", "command", "phi"]
        ):
            # Ollama models
            logger.info(f"Using Ollama model: {model}")

            # Extract the model name (remove "ollama:" prefix if present)
            model_name = model.replace("ollama:", "") if model.startswith("ollama:") else model

            llm = ChatOllama(
                model=model_name,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0,
            )
            return ok(llm)

        else:
            return err(f"Unsupported model: {model}")

    except Exception as e:
        return err(f"Failed to create LLM instance: {e}")


async def track_llm_usage(response: Any, model: str) -> None:
    """Track token usage from LLM response.

    Args:
        response: LLM response object
        model: Model name
    """
    try:
        from ..observability import get_cost_tracker

        # Extract usage metadata from response
        usage_metadata = None

        # Try different ways to get usage data based on provider
        if hasattr(response, "usage_metadata"):
            usage_metadata = response.usage_metadata
        elif hasattr(response, "usage"):
            usage_metadata = response.usage
        elif hasattr(response, "response_metadata"):
            usage_metadata = response.response_metadata.get("usage", {})

        if usage_metadata:
            input_tokens = (
                usage_metadata.get("input_tokens", 0)
                or usage_metadata.get("prompt_tokens", 0)
                or usage_metadata.get("total_tokens", 0) // 2  # Rough estimate
            )
            output_tokens = (
                usage_metadata.get("output_tokens", 0)
                or usage_metadata.get("completion_tokens", 0)
                or usage_metadata.get("total_tokens", 0) // 2  # Rough estimate
            )

            if input_tokens > 0 or output_tokens > 0:
                cost_tracker = get_cost_tracker()
                cost_tracker.track_usage(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    metadata={"source": "notebook_cli"},
                )

    except Exception as e:
        logger.debug(f"Failed to track LLM usage: {e}")


def create_system_prompt(
    excel_file_name: str,
    sheet_index: int,
    sheet_name: str | None,
    notebook_state: str,
    table_boundaries: str | None = None,
) -> str:
    """Create the system prompt for LLM analysis using external template.

    Args:
        excel_file_name: Name of Excel file
        sheet_index: Sheet index
        sheet_name: Optional sheet name
        notebook_state: Current notebook state
        table_boundaries: Optional pre-detected table boundaries

    Returns:
        System prompt string
    """
    # Choose prompt based on whether table boundaries are provided
    prompt_name = "table_aware_analyst_system" if table_boundaries else "data_analyst_system"

    # Load prompt with hash validation
    result = load_prompt(prompt_name)
    if result.is_err():
        logger.error(f"Failed to load prompt '{prompt_name}': {result.err_value}")
        # Fallback to a minimal prompt if loading fails
        return f"""You are an autonomous data analyst AI conducting comprehensive spreadsheet analysis.

Analyzing Excel file: {excel_file_name}
Sheet index: {sheet_index}
Sheet name: {sheet_name or "Unknown"}

Current notebook state:
```python
{notebook_state}
```

Conduct thorough analysis and provide actionable insights."""

    prompt_data = result.unwrap()
    system_template = PromptTemplate(template=prompt_data["template"], input_variables=prompt_data["input_variables"])

    # Build kwargs based on available variables
    kwargs = {
        "excel_file_name": excel_file_name,
        "sheet_index": sheet_index,
        "sheet_name": sheet_name or "Unknown",
        "notebook_state": notebook_state,
    }

    # Add table boundaries if using table-aware prompt
    if table_boundaries:
        kwargs["table_boundaries"] = table_boundaries

    return system_template.format(**kwargs)


def create_initial_prompt(
    excel_file_name: str, sheet_index: int, sheet_info: str, notebook_state: str, has_query_interface: bool
) -> str:
    """Create the initial human prompt for LLM using external template.

    Args:
        excel_file_name: Name of Excel file
        sheet_index: Sheet index
        sheet_info: Sheet information string
        notebook_state: Current notebook state
        has_query_interface: Whether query interface is available

    Returns:
        Initial prompt string
    """
    query_interface_note = (
        "- Query interface for formula dependencies (graph-based analysis)" if has_query_interface else ""
    )

    query_instruction = (
        "Query the formula dependency graph using graph-based tools"
        if has_query_interface
        else "Look for data quality issues"
    )

    # Load prompt with hash validation
    result = load_prompt("data_analyst_initial")
    if result.is_err():
        logger.error(f"Failed to load prompt 'data_analyst_initial': {result.err_value}")
        # Fallback to a minimal prompt if loading fails
        return f"""I've loaded the Excel file '{excel_file_name}'{sheet_info} into a Jupyter notebook.

Please continue the analysis from where it left off. Focus on deeper analysis."""

    prompt_data = result.unwrap()
    initial_template = PromptTemplate(template=prompt_data["template"], input_variables=prompt_data["input_variables"])

    return initial_template.format(
        excel_file_name=excel_file_name,
        sheet_info=sheet_info,
        notebook_state=notebook_state,
        query_interface_note=query_interface_note,
        query_instruction=query_instruction,
    )


async def process_tool_calls(tools: list[Any], response: Any, llm_logger: Any) -> list[ToolMessage]:
    """Process tool calls from LLM response.

    Args:
        tools: Available tools
        response: LLM response with tool calls
        llm_logger: Logger for LLM messages

    Returns:
        List of tool output messages
    """
    tool_output_messages = []

    llm_logger.info(f"\n{'═' * 20} Tool Executions {'═' * 20}")

    for tool_call in response.tool_calls:
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        logger.info(f"LLM called tool: {tool_name}", args=tool_args)

        # Find matching tool
        tool_func = next((t for t in tools if t.name == tool_name), None)
        if tool_func:
            try:
                tool_output = await tool_func.ainvoke(tool_args)
                logger.info(f"Tool output: {tool_output}")
            except Exception as tool_error:
                # Log the error but continue analysis
                logger.exception("Tool execution failed")
                tool_output = (
                    f"Tool execution failed: {tool_error!s}. "
                    f"Continuing analysis with alternative approach. "
                    f"The analysis will proceed without this specific operation."
                )

            # Log tool call and output
            llm_logger.info(f"\nTOOL CALL: {tool_name}")
            llm_logger.info(f"Arguments: {json.dumps(tool_args, indent=2)}")
            llm_logger.info(f"Output: {tool_output}")

            tool_output_messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"]))
        else:
            logger.warning(f"LLM tried to call unknown tool: {tool_name}")

            # Special handling for common Gemini mistakes
            if tool_name in ["to_markdown", "tolist", "head", "tail", "describe", "info"]:
                error_msg = (
                    f"ERROR: '{tool_name}' is a pandas DataFrame method, NOT a tool!\n"
                    f"To use DataFrame methods, you MUST use execute_code tool.\n"
                    f'Example: execute_code(code="df.{tool_name}()")\n'
                    f"Please retry using the execute_code tool."
                )
            else:
                # For other unknown tools
                error_msg = (
                    f"Unknown tool '{tool_name}'. Available tools are: "
                    f"{', '.join([t.name for t in tools[:5]])}... "
                    f"Please use one of the available tools."
                )

            llm_logger.info(f"\nTOOL CALL ERROR: {tool_name}")
            llm_logger.info(f"Error: {error_msg}")
            tool_output_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))

    return tool_output_messages


def check_forbidden_patterns(response_content: str) -> bool:
    """Check if response contains patterns indicating user input request.

    Args:
        response_content: LLM response content

    Returns:
        True if forbidden patterns found
    """
    forbidden_patterns = [
        "would you like me to",
        "let me know if",
        "do you need",
        "should i proceed",
        "would you prefer",
        "shall i continue",
        "feel free to ask",
        "if you'd like",
        "please let me know",
    ]

    response_lower = response_content.lower()
    return any(pattern in response_lower for pattern in forbidden_patterns)


def check_analysis_complete(response_content: str) -> bool:
    """Check if analysis is marked as complete.

    Args:
        response_content: LLM response content

    Returns:
        True if analysis is complete
    """
    response_lower = response_content.lower()
    return "analysis complete" in response_lower or "📊 analysis complete" in response_lower


async def run_llm_analysis(
    session: NotebookSession,
    config: AnalysisConfig,
    state: AnalysisState,
    excel_path: Any,
    sheet_name: str | None,
    notebook_path: Path,
) -> Result[None, str]:
    """Run LLM analysis rounds.

    Args:
        session: Active notebook session
        config: Analysis configuration
        state: Analysis state
        excel_path: Path to Excel file
        sheet_name: Optional sheet name
        notebook_path: Path to save notebook for auto-save

    Returns:
        Result indicating success or failure
    """
    logger.info("Setting up LLM interaction...")

    # Get notebook tools
    tools = get_notebook_tools()

    # Create LLM instance with thinking support
    llm_result = create_llm_instance(config.model, config.api_key, config.thinking_config)
    if llm_result.is_err():
        return err(llm_result.unwrap_err())

    llm = llm_result.unwrap()

    # Bind tools to LLM
    # CLAUDE-GOTCHA: Gemini models have a known bug where bind_tools() only recognizes
    # the first tool when multiple tools are provided (GitHub langchain-google #369).
    # The workaround is to use bind(functions=tools) instead of bind_tools(tools).
    # This was fixed in later versions but we use the workaround for compatibility.
    try:
        if "gemini" in config.model.lower():
            # CLAUDE-TEST-WORKAROUND: Use bind(functions=) for Gemini to avoid single-tool bug
            logger.debug(f"Using bind(functions=) workaround for Gemini model: {config.model}")
            llm_with_tools = llm.bind(functions=tools)
        else:
            # Standard bind_tools for other models
            llm_with_tools = llm.bind_tools(tools)
        logger.info(f"Successfully bound {len(tools)} tools to {config.model}")
    except Exception as e:
        # CLAUDE-COMPLEX: If bind(functions=) fails, try bind_tools as fallback
        # Some Gemini versions may have fixed the issue
        if "gemini" in config.model.lower() and "bind" in str(e):
            try:
                logger.debug("bind(functions=) failed, trying bind_tools() fallback")
                llm_with_tools = llm.bind_tools(tools)
                logger.info(f"Fallback successful: bound {len(tools)} tools with bind_tools()")
            except Exception as fallback_error:
                logger.exception("Both binding methods failed for Gemini")
                return err(f"Failed to bind tools to Gemini LLM: {fallback_error}")
        else:
            logger.exception(f"Failed to bind tools to LLM {config.model}")
            return err(f"Failed to bind tools to LLM: {e}")

    # Get current notebook state
    try:
        notebook_state = session.toolkit.export_to_percent_format()
    except Exception:
        logger.exception("Failed to export notebook state")
        notebook_state = "# Failed to export notebook state"

    # Create initial messages
    sheet_info = f" (sheet index {config.sheet_index})" if config.sheet_index != 0 else ""

    # Check if table boundaries are provided in config
    table_boundaries = getattr(config, "table_boundaries", None)
    system_prompt = create_system_prompt(
        excel_path.name, config.sheet_index, sheet_name, notebook_state, table_boundaries
    )

    initial_prompt = create_initial_prompt(
        excel_path.name,
        config.sheet_index,
        sheet_info,
        notebook_state,
        has_query_interface=bool(
            state.pipeline_results.formula_cache_path and state.pipeline_results.formula_cache_path.exists()
        ),
    )

    # Add tool usage clarification for Gemini models
    if "gemini" in config.model.lower():
        messages = get_gemini_messages(system_prompt, initial_prompt)
    else:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=initial_prompt),
        ]

    # Wrap analysis in session tracking
    with phoenix_session(
        session_id=state.artifacts.session_id,
        user_id=None,
    ):
        # Add session metadata
        add_session_metadata(
            state.artifacts.session_id,
            {
                "excel_file": excel_path.name,
                "sheet_index": config.sheet_index,
                "sheet_name": sheet_name or f"Sheet {config.sheet_index}",
                "model": config.model,
                "max_rounds": config.max_rounds,
                "cost_limit": config.cost_limit if config.track_costs else None,
            },
        )

        # Run analysis rounds
        for round_num in range(1, config.max_rounds + 1):
            logger.info(f"Starting analysis round {round_num}/{config.max_rounds}")

            # Log round start
            if state.llm_logger:
                state.llm_logger.info(f"\n{'🔄' * 40}")
                state.llm_logger.info(f"{'🔄' * 15} ROUND {round_num} - Starting Analysis {'🔄' * 15}")
                state.llm_logger.info(f"{'🔄' * 40}\n")

                # Log messages being sent
                state.llm_logger.info(f"{'═' * 20} Messages to LLM {'═' * 20}")
                for i, msg in enumerate(messages):
                    msg_type = type(msg).__name__
                    msg_content = getattr(msg, "content", str(msg))
                    state.llm_logger.info(f"\nMessage {i + 1} ({msg_type}):\n{msg_content}")
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        state.llm_logger.info(f"Tool calls: {json.dumps(msg.tool_calls, indent=2)}")

            if config.verbose:
                logger.info("Sending messages to LLM:", messages=messages)

            # Try to call LLM with progressive compression on context errors
            response, compressed_messages = await _call_llm_with_compression(llm_with_tools, messages, config, state)

            if response is None:
                logger.error("Failed to get LLM response even with maximum compression")
                break

            # Update messages if compression was applied
            if compressed_messages is not None:
                messages = compressed_messages

            # Track usage
            await track_llm_usage(response, config.model)

            # Log response
            if state.llm_logger:
                state.llm_logger.info(f"\n{'═' * 20} LLM Response {'═' * 20}")
                state.llm_logger.info(f"Response type: {type(response).__name__}")
                state.llm_logger.info(f"Content: {response.content}")
                if hasattr(response, "tool_calls") and response.tool_calls:
                    state.llm_logger.info(f"Tool calls: {json.dumps(response.tool_calls, indent=2)}")

            if config.verbose:
                logger.info("Received response from LLM:", response=response)

            # Process response
            logger.debug(f"Response has tool_calls: {bool(getattr(response, 'tool_calls', None))}")
            logger.debug(f"Response has content: {bool(getattr(response, 'content', None))}")

            # CLAUDE-GOTCHA: Gemini often returns empty content when making tool calls,
            # unlike OpenAI models. We must check for tool_calls first, regardless of content.
            # Sometimes tool calls might be in _raw_response attribute (undocumented behavior).
            tool_calls_found = getattr(response, "tool_calls", None)

            # CLAUDE-COMPLEX: Check for tool calls in various locations due to Gemini inconsistencies
            if not tool_calls_found and "gemini" in config.model.lower() and hasattr(response, "_raw_response"):
                # Check if tool calls are hidden in raw response (Gemini quirk)
                logger.debug("Checking _raw_response for hidden tool calls (Gemini quirk)")
                # This is a known issue where Gemini puts tool calls in unexpected places

            if tool_calls_found:
                # Add the AI response to conversation
                messages.append(response)

                # Process tool calls
                tool_output_messages = await process_tool_calls(tools, response, state.llm_logger)

                # Add tool results to conversation
                messages.extend(tool_output_messages)

            # CLAUDE-PERFORMANCE: For Gemini, if no tool calls found, add retry with explicit prompt
            elif "gemini" in config.model.lower() and round_num < config.max_rounds - 1:
                # Gemini failed to use tools, add explicit instruction
                logger.warning("Gemini didn't use tools, adding explicit instruction")
                if state.llm_logger:
                    state.llm_logger.info("\n⚠️ GEMINI TOOL USAGE ISSUE - Adding explicit instruction")

                # Add explicit tool usage request
                explicit_prompt = HumanMessage(
                    content="Please use the execute_code tool to run the Python code for analysis. "
                    "Remember: ALL Python code must be run using execute_code(code='...')."
                )
                messages.append(explicit_prompt)
                continue  # Skip to next round

            elif response.content:
                logger.info(f"LLM response: {response.content}")

                # Check if this might be a context limit issue
                # If we expect tool calls but get very short content, it might be hitting token limit
                if len(response.content) < 200 and round_num < config.max_rounds:
                    logger.warning(
                        f"Received unusually short response ({len(response.content)} chars), may be hitting context limit"
                    )

                    # Try compression and retry
                    logger.info("Attempting to compress context and retry...")
                    if state.llm_logger:
                        state.llm_logger.info("\n⚠️ SHORT RESPONSE DETECTED - Possible context limit reached")
                        state.llm_logger.info(f"Response was only {len(response.content)} characters")
                        state.llm_logger.info("Attempting context compression to recover...")

                    response_retry, compressed_messages_retry = await _call_llm_with_compression(
                        llm_with_tools, messages, config, state, force_compression_level=1
                    )

                    if response_retry and hasattr(response_retry, "tool_calls") and response_retry.tool_calls:
                        logger.info("Successfully recovered with compression!")
                        response = response_retry
                        if compressed_messages_retry:
                            messages = compressed_messages_retry
                        # Process the tool calls as normal
                        messages.append(response)
                        tool_output_messages = await process_tool_calls(tools, response, state.llm_logger)
                        messages.extend(tool_output_messages)
                        continue

                # Check for forbidden patterns
                if check_forbidden_patterns(response.content):
                    logger.warning("LLM attempted to ask for user input - enforcing autonomous completion")
                    messages.append(response)
                    messages.append(
                        SystemMessage(
                            content="""
REMINDER: You must complete the analysis autonomously.
- Create a final comprehensive analysis report in markdown with "## 📊 Analysis Complete"
- Follow the required report structure (Executive Summary, Data Overview, Key Findings, etc.)
- Include all sections: findings, data quality, statistical insights, business implications, recommendations
- Then STOP - do not ask for further instructions
Complete the analysis now."""
                        )
                    )
                    continue

                # Check if analysis is complete
                if check_analysis_complete(response.content):
                    logger.info("Analysis marked as complete by LLM")
                    if state.llm_logger:
                        state.llm_logger.info(f"\n{'✅' * 40}")
                        state.llm_logger.info(f"{'✅' * 15} ROUND {round_num} Complete {'✅' * 15}")
                        state.llm_logger.info(f"{'✅' * 40}\n")
                    break

                # Log round completion
                if state.llm_logger:
                    state.llm_logger.info(f"\n{'✅' * 40}")
                    state.llm_logger.info(f"{'✅' * 15} ROUND {round_num} Complete {'✅' * 15}")
                    state.llm_logger.info(f"{'✅' * 40}\n")
                break

            else:
                logger.warning("LLM response was empty.")
                if state.llm_logger:
                    state.llm_logger.info(f"\n{'✅' * 40}")
                    state.llm_logger.info(f"{'✅' * 15} ROUND {round_num} Complete {'✅' * 15}")
                    state.llm_logger.info(f"{'✅' * 40}\n")
                break

        # CLAUDE-KNOWLEDGE: Auto-save after each round to prevent data loss
        # Save notebook after each round completes if configured
        if config.auto_save_rounds:
            logger.info(f"Auto-saving notebook after round {round_num}")
            save_result = await save_notebook(session, notebook_path)
            if save_result.is_err():
                logger.warning(f"Failed to auto-save notebook: {save_result.unwrap_err()}")
                # Continue analysis even if auto-save fails
            else:
                logger.info(f"Notebook auto-saved successfully to {notebook_path}")

                # Create checkpoint file
                checkpoint_path = notebook_path.parent / f"{notebook_path.stem}_checkpoint_round{round_num}.ipynb"
                try:
                    shutil.copy2(notebook_path, checkpoint_path)
                    logger.info(f"Created checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Failed to create checkpoint: {e}")

    return ok(None)


async def _call_llm_with_compression(llm_with_tools, messages, config, state, force_compression_level=0):
    """Call LLM with progressive compression on context errors.

    Args:
        llm_with_tools: LLM instance with tools bound
        messages: Current message list
        config: Analysis configuration
        state: Analysis state
        force_compression_level: Start at this compression level (default 0)

    Returns:
        Tuple of (LLM response, compressed messages) or (None, None) if all attempts fail
    """
    compressor = HierarchicalContextCompressor()
    compression_level = force_compression_level
    max_compression_levels = 7

    # Make a copy of messages to avoid modifying the original
    compressed_messages = messages.copy()

    while compression_level < max_compression_levels:
        try:
            # Attempt to call LLM
            logger.debug(f"Invoking LLM with {len(compressed_messages)} messages")
            response = await llm_with_tools.ainvoke(compressed_messages)
            logger.debug(f"Received response type: {type(response)}")

            if compression_level > 0:
                logger.info(f"Successfully sent request after {compression_level} compression levels")
                if state.llm_logger:
                    state.llm_logger.info(f"Applied {compression_level} compression levels to fit context window")
                return response, compressed_messages
            else:
                return response, None

        except Exception as e:
            error_str = str(e).lower()

            # Check if this is a context length error
            # Different providers have different error messages
            context_error_patterns = [
                "context length",
                "context window",
                "token limit",
                "maximum context",
                "too many tokens",
                "exceeds maximum",
                "model's maximum context",
                "reduce the length",
                "messages too long",
            ]

            is_context_error = any(pattern in error_str for pattern in context_error_patterns)

            if is_context_error and compression_level < max_compression_levels - 1:
                logger.warning(
                    f"Context window exceeded, applying compression level {compression_level}: "
                    f"{compressor.compression_hierarchy[compression_level].name}"
                )

                if state.llm_logger:
                    state.llm_logger.info(f"\n{'❌' * 20} CONTEXT ERROR DETECTED {'❌' * 20}")
                    state.llm_logger.info(f"Error: {str(e)[:200]}...")
                    state.llm_logger.info(
                        f"Applying compression level {compression_level}: "
                        f"{compressor.compression_hierarchy[compression_level].name}"
                    )

                # Apply next level of compression
                compressed_messages = compressor.compress_messages(compressed_messages, compression_level)
                compression_level += 1

            else:
                # Not a context error or max compression reached
                logger.exception("LLM API call failed")
                return None, None

    logger.error("Exhausted all compression levels")
    return None, None
