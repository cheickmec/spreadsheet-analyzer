"""LLM interaction logic for notebook analysis.

This module handles all LLM-specific interactions, keeping them separate
from the core analysis orchestration.

CLAUDE-KNOWLEDGE: LLM interactions are isolated to make it easier to
swap providers or modify prompts without affecting core logic.
"""

import json
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from structlog import get_logger

from ..core.types import Result, err, ok
from ..notebook_llm_interface import get_notebook_tools
from ..notebook_session import NotebookSession
from ..observability import add_session_metadata, phoenix_session
from .notebook_analysis import AnalysisConfig, AnalysisState

logger = get_logger(__name__)


def create_llm_instance(model: str, api_key: str | None = None) -> Result[Any, str]:
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

            llm = ChatAnthropic(
                model_name=model,
                api_key=api_key,
                max_tokens=4096,
            )
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


def create_system_prompt(excel_file_name: str, sheet_index: int, sheet_name: str | None, notebook_state: str) -> str:
    """Create the system prompt for LLM analysis.

    Args:
        excel_file_name: Name of Excel file
        sheet_index: Sheet index
        sheet_name: Optional sheet name
        notebook_state: Current notebook state

    Returns:
        System prompt string
    """
    # CLAUDE-COMPLEX: This prompt is intentionally comprehensive to guide
    # autonomous analysis without user interaction
    return f"""You are an autonomous data analyst AI conducting comprehensive spreadsheet analysis.

CRITICAL CONSTRAINT - NO VISUAL ACCESS:
- Cannot see images, plots, charts, or visualizations
- Must extract insights through textual methods only
- All analysis based on numerical summaries and textual descriptions

CONTEXT:
- Analyzing Excel file: {excel_file_name}
- Sheet index: {sheet_index}
- Sheet name: {sheet_name or "Unknown"}

CURRENT NOTEBOOK STATE:
```python
{notebook_state}
```

AUTONOMOUS ANALYSIS PROTOCOL:
1. Conduct systematic analysis without seeking user approval
2. Back all assumptions with evidence from the data
3. Reach solid conclusions based on thorough investigation
4. Document reasoning in code comments
5. Use available tools to execute code and explore data
6. Focus on actionable insights and recommendations

MULTI-TABLE DETECTION - CRITICAL FIRST STEP:
Before analyzing data, ALWAYS check if the sheet contains multiple tables using BOTH mechanical AND semantic analysis:

1. **Initial Structure Scan**
   ```python
   # First, examine the raw structure with MORE context
   print(f"Sheet dimensions: {{df.shape}}")
   print("\\n--- First 30 rows ---")
   print(df.head(30))

   # Look at ALL columns including unnamed ones
   print("\\n--- Column overview ---")
   for col in df.columns:
       non_null = df[col].notna().sum()
       print(f"{{col}}: {{non_null}} non-null values, dtype: {{df[col].dtype}}")
   ```

2. **Mechanical Detection** (empty rows/columns)
   ```python
   # Check for empty row patterns that separate tables
   empty_rows = df.isnull().all(axis=1)
   empty_row_groups = empty_rows.groupby((~empty_rows).cumsum()).sum()
   print(f"\\nEmpty row blocks: {{empty_row_groups[empty_row_groups > 0].to_dict()}}")

   # Check for empty columns (potential horizontal separators)
   empty_cols = df.isnull().all(axis=0)
   print(f"Empty columns: {{list(df.columns[empty_cols])}}")
   ```

3. **Semantic Detection** (USE HUMAN JUDGMENT)
   ```python
   # Ask yourself: What does each row represent?
   # Example: If row 1 = "Product ABC, $50" and row 100 = "John Smith, Developer"
   # These are clearly different entity types!

   # Check for semantic shifts in data
   print("\\n--- Checking for semantic table boundaries ---")

   # 1. Analyze column groupings - do they describe the same type of thing?
   # Example: ['Customer', 'Address', 'Phone'] vs ['Date', 'Amount', 'Transaction ID']

   # 2. Look for granularity changes
   # Example: 5 summary rows followed by 1000 detail rows

   # 3. Check if column names suggest different purposes
   # Example: Financial columns vs HR columns in same sheet
   ```

4. **Common Multi-Table Patterns to Recognize**
   - **Master/Detail**: Header info (few rows) + Line items (many rows)
   - **Summary/Breakdown**: Totals followed by individual components
   - **Different Domains**: Unrelated business data side-by-side
   - **Time Periods**: Current data adjacent to historical data

   Ask: "Would these naturally be separate tables in a database?"

5. **Decision Framework**
   Even WITHOUT empty rows/columns, declare multiple tables if:
   - Rows represent fundamentally different entity types
   - Column sets serve different business purposes
   - There's a clear shift in data granularity
   - A business analyst would logically separate them

6. **Multi-Table Handling Strategy**
   If multiple tables detected:
   - Document the table boundaries and what each represents
   - Analyze each table's purpose separately
   - Use `.iloc[start:end, start_col:end_col]` for extraction
   - Focus on the most relevant table(s) for insights

COMPLETION PROTOCOL - CRITICAL:
- FIRST: Always perform multi-table detection using the empty row analysis code
- Complete ALL analysis steps autonomously without asking for user input
- NEVER ask "Would you like me to..." or "Let me know if..." or "Do you need..."
- When analysis is complete, provide a final summary and STOP
- If errors occur, implement workarounds or fix code, re-run it and continue analysis
- End with definitive conclusions
- DO NOT offer to perform additional analysis - just complete what's needed

TEXTUAL DATA EXPLORATION TECHNIQUES:
- `.iloc[start:end]` or `.loc[condition]` to examine specific data regions
- `.sample(n)` for random sampling
- `.groupby()` for categorical analysis
- `.value_counts()` for frequency distributions
- `.describe()` for statistical summaries
- `.corr()` for correlation analysis
- `.isnull().sum()` for missing data analysis

TEXTUAL VISUALIZATION ALTERNATIVES (NO IMAGES):
**Distributions & Patterns:**
- `.value_counts().head(10)` to show frequency distribution
- `.describe()` to show quartiles, mean, std, min/max
- `.quantile([0.1, 0.25, 0.5, 0.75, 0.9])` for detailed percentiles
- `.hist(bins=20).value_counts()` for histogram-like data

**Trends & Relationships:**
- `.groupby().agg(['mean', 'std', 'count'])` to show patterns by category
- `.corr().round(3)` to show correlation matrix numerically
- `.pivot_table()` to show cross-tabulations
- `.rolling(window=5).mean()` for moving averages

**Outliers & Anomalies:**
- IQR method: Q1, Q3 = df.quantile([0.25, 0.75]); IQR = Q3 - Q1
- `.quantile([0.01, 0.99])` to show extreme values
- `(df > df.quantile(0.99)) | (df < df.quantile(0.01))` to identify outliers
- `.std()` and z-score calculations for statistical outliers

**Missing Data Patterns:**
- `.isnull().sum()` for column-wise missing counts
- `.isnull().sum(axis=1)` for row-wise missing patterns
- `.isnull().groupby(df['category']).sum()` for missing by category

**Data Quality Assessment:**
- `.dtypes` to check data types
- `.nunique()` to check cardinality
- `.duplicated().sum()` to find duplicates
- `.apply(lambda x: x.astype(str).str.len().max())` for string length analysis

ERROR VALIDATION REQUIREMENTS:
- **Data Type Validation**: Check for mixed data types in columns
- **Range Validation**: Identify values outside expected ranges
- **Formula Verification**: If formulas exist, verify calculations manually
- **Consistency Checks**: Look for inconsistent naming, formatting, or values
- **Missing Data Patterns**: Analyze if missing data follows patterns
- **Duplicate Detection**: Check for duplicate rows or suspicious duplicates
- **Business Logic Validation**: Verify data makes business sense
- **Cross-Reference Validation**: Check relationships between columns

EVIDENCE-BASED ANALYSIS:
- Never assume - always verify with data
- Show calculations - don't just state conclusions
- Provide confidence levels - indicate uncertainty
- Cross-validate findings - use multiple methods
- Document assumptions - clearly state what you're assuming

OUTPUT REQUIREMENTS:
- All outputs truncated at 1000 characters
- Use ONLY textual summaries and numerical descriptions
- Provide specific data examples to support conclusions
- Include error detection findings in analysis
- Reach definitive, evidence-based conclusions
- Describe patterns, trends, and relationships in words

BEST PRACTICES:
- Include reasoning in code comments
- Document analysis approach and findings
- Build upon existing analysis
- Provide clear, actionable recommendations
- Always validate findings with multiple approaches
- Use descriptive statistics to paint a picture of the data

ANALYSIS COMPLETION CRITERIA:
Mark analysis as COMPLETE when ALL of the following are achieved:
1. ✓ Multi-table detection performed (empty row analysis to identify table boundaries)
2. ✓ Data quality assessment completed (missing data, duplicates, anomalies)
3. ✓ Statistical analysis performed (distributions, correlations, patterns)
4. ✓ Business logic validated (calculations, relationships, consistency)
5. ✓ Key findings documented in markdown cells
6. ✓ Actionable recommendations provided
7. ✓ Final comprehensive analysis report created in markdown cell with title "## 📊 Analysis Complete"

When these criteria are met, create a final comprehensive analysis report in a markdown cell:

**Report Structure Required:**
# 📊 Analysis Complete

## Executive Summary
- Brief overview of the analysis performed
- Most important findings in 2-3 sentences

## Data Overview
- Dataset characteristics (size, timeframe, scope)
- Multi-table detection results
- Data quality summary

## Key Findings
1. **Finding 1**: [Detailed description with supporting evidence]
2. **Finding 2**: [Detailed description with supporting evidence]
3. **Finding 3**: [Detailed description with supporting evidence]
(Include 3-5 major findings)

## Data Quality Issues
- Missing data patterns
- Anomalies detected
- Validation concerns

## Statistical Insights
- Key distributions and patterns
- Significant correlations
- Trend analysis results

## Business Implications
- What these findings mean for business operations
- Risk factors identified
- Opportunities discovered

## Recommendations
1. **Immediate Actions**: [What should be done right away]
2. **Short-term Improvements**: [1-3 month timeline]
3. **Long-term Considerations**: [Strategic recommendations]

## Technical Notes
- Analysis methodology used
- Assumptions made
- Limitations of the analysis

Then STOP the analysis - do not ask for further instructions."""


def create_initial_prompt(
    excel_file_name: str, sheet_index: int, sheet_info: str, notebook_state: str, has_query_interface: bool
) -> str:
    """Create the initial human prompt for LLM.

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

    return f"""I've loaded the Excel file '{excel_file_name}'{sheet_info} into a Jupyter notebook.

## Current Notebook State:
```python
{notebook_state}
```

The notebook already contains:
- Pipeline analysis results (Security, Structure, Formula Analysis)
- Data loaded into DataFrame 'df' with initial exploration showing shape and first rows
{query_interface_note}
- Formula analysis tools: graph-based for dependencies, formulas library for evaluation

Please continue the analysis from where it left off. **DO NOT re-execute cells that already have output.**

You can:
1. Execute NEW Python code to explore the data further
2. Use pandas operations to analyze patterns
3. Create visualizations if helpful
4. {query_instruction}
5. Use formulas library tools for formula evaluation and what-if analysis
6. Provide insights and recommendations

Focus on deeper analysis that builds upon what's already been done.

IMPORTANT: Track your progress against the completion criteria and create a final summary when done."""


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
    session: NotebookSession, config: AnalysisConfig, state: AnalysisState, excel_path: Any, sheet_name: str | None
) -> Result[None, str]:
    """Run LLM analysis rounds.

    Args:
        session: Active notebook session
        config: Analysis configuration
        state: Analysis state
        excel_path: Path to Excel file
        sheet_name: Optional sheet name

    Returns:
        Result indicating success or failure
    """
    logger.info("Setting up LLM interaction...")

    # Get notebook tools
    tools = get_notebook_tools()

    # Create LLM instance
    llm_result = create_llm_instance(config.model, config.api_key)
    if llm_result.is_err():
        return err(llm_result.unwrap_err())

    llm = llm_result.unwrap()

    # Bind tools to LLM
    try:
        llm_with_tools = llm.bind_tools(tools)
    except Exception as e:
        return err(f"Failed to bind tools to LLM: {e}")

    # Get current notebook state
    try:
        notebook_state = session.toolkit.export_to_percent_format()
    except Exception:
        logger.exception("Failed to export notebook state")
        notebook_state = "# Failed to export notebook state"

    # Create initial messages
    sheet_info = f" (sheet index {config.sheet_index})" if config.sheet_index != 0 else ""

    system_prompt = create_system_prompt(excel_path.name, config.sheet_index, sheet_name, notebook_state)

    initial_prompt = create_initial_prompt(
        excel_path.name,
        config.sheet_index,
        sheet_info,
        notebook_state,
        has_query_interface=bool(
            state.pipeline_results.formula_cache_path and state.pipeline_results.formula_cache_path.exists()
        ),
    )

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

            try:
                # Call LLM
                response = await llm_with_tools.ainvoke(messages)

                # Track usage
                await track_llm_usage(response, config.model)

                # Log response
                if state.llm_logger:
                    state.llm_logger.info(f"\n{'═' * 20} LLM Response {'═' * 20}")
                    state.llm_logger.info(f"Response type: {type(response).__name__}")
                    state.llm_logger.info(f"Content: {response.content}")
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        state.llm_logger.info(f"Tool calls: {json.dumps(response.tool_calls, indent=2)}")

            except Exception:
                logger.exception("Model API call failed")
                break

            if config.verbose:
                logger.info("Received response from LLM:", response=response)

            # Process response
            if response.tool_calls:
                # Add the AI response to conversation
                messages.append(response)

                # Process tool calls
                tool_output_messages = await process_tool_calls(tools, response, state.llm_logger)

                # Add tool results to conversation
                messages.extend(tool_output_messages)

            elif response.content:
                logger.info(f"LLM response: {response.content}")

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

    return ok(None)
