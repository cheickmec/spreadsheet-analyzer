"""LangChain/LangGraph integration for Excel analysis workflow."""

import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_core.load.dump import dumpd
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from structlog import get_logger

from spreadsheet_analyzer.agents.kernel_manager import AgentKernelManager
from spreadsheet_analyzer.notebook_llm.nap.protocols import Cell, CellType, NotebookDocument
from spreadsheet_analyzer.pipeline.pipeline import DeterministicPipeline
from spreadsheet_analyzer.plugins.spreadsheet.io.excel_io import check_file_size
from spreadsheet_analyzer.utils import calculate_cost

# TODO: These functions need to be reimplemented using the proper architecture:
# append_llm_code_blocks, build_observation_from_notebook, create_sheet_notebook,
# inspect_notebook_quality, save_analysis_results

logger = get_logger(__name__)


# CLAUDE-PERFORMANCE: Threshold for considering a file "large"
LARGE_FILE_THRESHOLD_MB = 50
LARGE_FILE_ROW_LIMIT = 100000


def safe_json_serialize(obj: Any, max_depth: int = 5, current_depth: int = 0) -> Any:
    """Safely convert any object to a JSON-serializable format.

    Args:
        obj: Object to serialize
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth

    Returns:
        JSON-serializable version of the object
    """
    if current_depth > max_depth:
        return f"<Max depth {max_depth} exceeded>"

    # Handle None and basic types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle bytes
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes: {len(obj)} bytes>"

    # Handle datetime objects
    if hasattr(obj, "isoformat"):
        return obj.isoformat()

    # Handle UUID objects
    if hasattr(obj, "hex") and hasattr(obj, "version"):
        return str(obj)

    # Handle dictionaries
    if isinstance(obj, dict):
        return {
            safe_json_serialize(k, max_depth, current_depth + 1): safe_json_serialize(v, max_depth, current_depth + 1)
            for k, v in obj.items()
        }

    # Handle lists, tuples, sets
    if isinstance(obj, (list, tuple, set)):
        return [safe_json_serialize(item, max_depth, current_depth + 1) for item in obj]

    # Handle Pydantic models
    if hasattr(obj, "model_dump"):
        try:
            return safe_json_serialize(obj.model_dump(), max_depth, current_depth + 1)
        except Exception:
            pass

    if hasattr(obj, "dict") and callable(obj.dict):
        try:
            return safe_json_serialize(obj.dict(), max_depth, current_depth + 1)
        except Exception:
            pass

    # Handle dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        try:
            return safe_json_serialize(asdict(obj), max_depth, current_depth + 1)
        except Exception:
            pass

    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {
                k: safe_json_serialize(v, max_depth, current_depth + 1)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        except Exception:
            pass

    # Handle objects with __str__ or __repr__
    try:
        return str(obj)
    except Exception:
        try:
            return repr(obj)
        except Exception:
            return f"<{type(obj).__name__} object>"


async def execute_notebook_cells(notebook: NotebookDocument, state: dict = None) -> tuple[NotebookDocument, dict]:
    """Execute all code cells in the notebook using AgentKernelManager.

    Args:
        notebook: Notebook with code cells to execute
        state: Optional state to store kernel context for later use

    Returns:
        Tuple of (executed notebook, kernel context dict)
    """
    logger.info("Executing notebook cells", cell_count=len(notebook.cells))

    kernel_context = {}
    async with (
        AgentKernelManager(max_kernels=1) as manager,
        manager.acquire_kernel("sheet-analyzer") as (kernel_manager, session),
    ):
        # Store kernel context for later use in save operations
        kernel_context = {
            "kernel_manager": manager,
            "kernel_session": session,
        }

        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == CellType.CODE:
                try:
                    # Execute the code
                    result = await manager.execute_code(session, cell.source)

                    # Debug: Log the raw result
                    logger.debug(
                        "Cell execution result",
                        cell_index=i,
                        result_keys=list(result.keys()),
                        outputs_count=len(result.get("outputs", [])),
                        status=result.get("status"),
                    )

                    # Convert outputs to notebook format
                    outputs = []
                    for output in result.get("outputs", []):
                        logger.debug(
                            "Processing output",
                            output_type=output.get("type"),
                            output_keys=list(output.keys()),
                        )

                        if output["type"] == "stream":
                            outputs.append({"output_type": "stream", "name": "stdout", "text": output["text"]})
                        elif output["type"] == "execute_result":
                            outputs.append(
                                {
                                    "output_type": "execute_result",
                                    "execution_count": i + 1,
                                    "data": output["data"],
                                    "metadata": {},
                                }
                            )
                        elif output["type"] == "error":
                            outputs.append(
                                {
                                    "output_type": "error",
                                    "ename": output.get("ename", "Error"),
                                    "evalue": output.get("evalue", "Unknown error"),
                                    "traceback": output.get("traceback", []),
                                }
                            )

                    # Update cell with outputs - ensure they're stored as list
                    cell.outputs = outputs if outputs else []
                    cell.execution_count = i + 1

                    logger.info(
                        f"Executed cell {i + 1}/{len(notebook.cells)}",
                        outputs_count=len(outputs),
                        has_outputs=bool(outputs),
                    )

                except Exception as e:
                    logger.exception(f"Error executing cell {i + 1}")
                    cell.outputs = [
                        {"output_type": "error", "ename": type(e).__name__, "evalue": str(e), "traceback": []}
                    ]

    logger.info("Notebook execution complete")
    return notebook, kernel_context


def notebook_to_dict(notebook: NotebookDocument) -> dict:
    """Convert a NotebookDocument to a dictionary for serialization."""
    cells_data = []
    for i, cell in enumerate(notebook.cells):
        # Debug: Log cell outputs before conversion
        logger.debug(
            "Converting cell to dict",
            cell_index=i,
            cell_id=cell.id,
            has_outputs=hasattr(cell, "outputs") and cell.outputs is not None,
            outputs_count=len(cell.outputs) if hasattr(cell, "outputs") and cell.outputs else 0,
        )

        cell_dict = asdict(cell)
        # Convert CellType enum to string
        if hasattr(cell.cell_type, "value"):
            cell_dict["cell_type"] = cell.cell_type.value
        cells_data.append(cell_dict)

    return {
        "id": notebook.id,
        "cells": cells_data,
        "metadata": notebook.metadata,
        "kernel_spec": notebook.kernel_spec,
        "language_info": notebook.language_info,
    }


class SheetState(TypedDict, total=False):
    """Comprehensive state for the analysis workflow."""

    # Input
    excel_path: Path
    sheet_name: str
    skip_deterministic: bool
    provider: str
    model: str
    temperature: float

    # Intermediate results
    deterministic: dict  # Results from DeterministicPipeline
    notebook_json: dict  # Executed notebook after stage-1
    llm_response: str  # Raw LLM output
    notebook_final: dict  # Executed notebook after stage-2

    # Tracking
    messages: list[BaseMessage]  # Conversation log
    execution_errors: list[str]
    token_usage: dict[str, int]
    total_cost: float  # Running total cost in USD

    # Output
    output_path: Path
    metadata_path: Path

    # Performance tracking
    file_size_mb: float
    is_large_file: bool

    # Quality tracking
    quality_iterations: int  # Number of quality-driven iterations
    quality_reasons: list[str]  # Reasons for each iteration
    max_quality_iterations: int  # Maximum allowed quality iterations
    needs_refinement: bool  # Quality gate decision
    quality_feedback: str  # Feedback from quality gate

    # Kernel context for direct notebook saving
    kernel_manager: Any  # AgentKernelManager instance
    kernel_session: Any  # KernelSession instance


def create_llm(state: SheetState) -> Any:
    """Create LLM instance based on state configuration."""
    provider = state.get("provider", "anthropic")
    model = state.get("model", "claude-3-5-sonnet-20241022")
    temperature = state.get("temperature", 0.1)

    if provider == "anthropic":
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=8192,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=8192,
        )
    else:
        # Fallback to init_chat_model for auto-detection
        return init_chat_model(
            model=model,
            model_provider=provider,
            temperature=temperature,
            max_tokens=8192,
        )


async def deterministic_node(state: SheetState) -> dict[str, Any]:
    """Run deterministic analysis on the Excel file."""
    logger.info("Running deterministic analysis", excel_path=str(state["excel_path"]))

    if state.get("skip_deterministic"):
        logger.info("Skipping deterministic analysis as requested")
        return {}

    try:
        # Run blocking deterministic pipeline in thread pool
        pipeline = DeterministicPipeline()
        result = await asyncio.to_thread(pipeline.run, state["excel_path"])
    except Exception as e:
        logger.exception("Deterministic analysis failed")
        return {"execution_errors": [f"Deterministic analysis failed: {e!s}"]}
    else:
        return {"deterministic": result}


async def create_notebook_node(state: SheetState) -> dict[str, Any]:
    """Create initial notebook structure using shared utility."""
    logger.info(
        "Creating notebook scaffold",
        excel_path=str(state["excel_path"]),
        sheet_name=state["sheet_name"],
    )

    try:
        # Convert deterministic results if present
        deterministic_data = state.get("deterministic")
        if deterministic_data and hasattr(deterministic_data, "to_report"):
            # Convert PipelineResult to dict
            deterministic_data = deterministic_data.to_report()

        # Use shared utility
        notebook = await asyncio.to_thread(
            create_sheet_notebook,
            excel_path=state["excel_path"],
            sheet_name=state["sheet_name"],
            deterministic_results=deterministic_data,
        )
        return {"notebook_json": notebook_to_dict(notebook)}
    except Exception as e:
        logger.exception("Failed to create notebook")
        return {"execution_errors": [f"Notebook creation failed: {e!s}"]}


async def execute_initial_cells_node(state: SheetState) -> dict[str, Any]:
    """Execute initial notebook cells (data loading)."""
    logger.info("Executing initial notebook cells")

    try:
        # Reconstruct notebook from dict, converting cell dicts to Cell objects
        notebook_data = state["notebook_json"]
        cells = [Cell(**cell) if isinstance(cell, dict) else cell for cell in notebook_data["cells"]]
        notebook = NotebookDocument(
            id=notebook_data["id"],
            cells=cells,
            metadata=notebook_data["metadata"],
            kernel_spec=notebook_data["kernel_spec"],
            language_info=notebook_data["language_info"],
        )
        executed_notebook, kernel_context = await execute_notebook_cells(notebook, state)

        result = {"notebook_json": notebook_to_dict(executed_notebook)}
        # Store kernel context for later use in save operations
        result.update(kernel_context)
        return result
    except Exception as e:
        logger.exception("Initial cell execution failed")
        return {"execution_errors": [f"Initial execution failed: {e!s}"]}


def prepare_llm_prompt(state: SheetState) -> str:
    """Prepare the prompt for LLM analysis."""
    # Check if notebook exists
    if "notebook_json" not in state:
        logger.warning("No notebook_json in state for LLM prompt preparation")
        return "Error: No notebook available for analysis"

    # Build context from notebook - reconstruct from dict
    notebook_data = state["notebook_json"]
    cells = [Cell(**cell) if isinstance(cell, dict) else cell for cell in notebook_data["cells"]]
    notebook = NotebookDocument(
        id=notebook_data["id"],
        cells=cells,
        metadata=notebook_data["metadata"],
        kernel_spec=notebook_data["kernel_spec"],
        language_info=notebook_data["language_info"],
    )
    observations = build_observation_from_notebook(notebook)

    # Convert deterministic results if present
    deterministic_data = state.get("deterministic", {})
    if deterministic_data and hasattr(deterministic_data, "to_report"):
        # Convert PipelineResult to dict
        deterministic_data = deterministic_data.to_report()

    # Build prompt with context
    prompt = f"""Analyze this Excel spreadsheet data and provide insights.

Sheet: {state["sheet_name"]}

Current notebook state:
{observations}

Deterministic analysis results:
{json.dumps(deterministic_data, indent=2)}

Please provide Python code to:
1. Analyze the data structure and patterns
2. Identify key insights and anomalies
3. Create visualizations if appropriate
4. Summarize your findings

Remember to use the 'df' variable that contains the loaded data.
Only provide Python code blocks."""

    return prompt


async def llm_analysis_node(state: SheetState) -> dict[str, Any]:
    """Analyze the notebook with LLM."""
    logger.info("Running LLM analysis")

    try:
        llm = create_llm(state)

        # Build LCEL chain
        chain = (
            RunnablePassthrough()
            | RunnableLambda(prepare_llm_prompt)
            | RunnableLambda(
                lambda prompt: [
                    SystemMessage(
                        content="""You are an expert data analyst helping to analyze Excel spreadsheets.
Your task is to generate Python code that will be executed in a Jupyter notebook to analyze the data.

IMPORTANT RULES:
1. Generate ONLY executable Python code blocks
2. Each code block should be complete and self-contained
3. Use the pre-loaded variables: excel_path, sheet_name, df (if available)
4. Import any required libraries at the beginning of your code
5. Print or display results clearly
6. Handle potential errors gracefully

QUALITY CHECKLIST - Your analysis MUST include:
1. Data Profiling:
   - Basic statistics (df.describe())
   - Data types analysis (df.dtypes, df.info())
   - Null/missing value analysis (df.isna().sum())

2. Data Quality Validation:
   - Check for duplicate rows
   - Validate data types match expected formats
   - Identify outliers using IQR or Z-score methods
   - Check for data consistency issues

3. Excel-Specific Analysis:
   - Use openpyxl to check for formula errors (#DIV/0!, #N/A, etc.)
   - Identify cells with formulas vs static values
   - Validate formula consistency across rows/columns

4. Visualizations:
   - Create at least one meaningful chart (distribution, trends, relationships)
   - Use matplotlib, seaborn, or plotly for clear visualizations

5. Business Insights:
   - Summarize key findings in print statements
   - Identify potential data entry errors or anomalies
   - Provide actionable recommendations

Remember: After each analysis step, verify data quality and document any issues found."""
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            | llm
        )

        response = await chain.ainvoke(state)

        # Track token usage and cost
        token_usage = {}
        total_cost = 0.0

        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("usage", {})
            token_usage = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

            # Calculate cost
            model = state.get("model", "claude-3-5-sonnet-20241022")
            total_cost = calculate_cost(
                model=model,
                input_tokens=token_usage["prompt_tokens"],
                output_tokens=token_usage["completion_tokens"],
            )

        return {
            "llm_response": response.content,
            "messages": [*state.get("messages", []), response],
            "token_usage": token_usage,
            "total_cost": state.get("total_cost", 0.0) + total_cost,
        }
    except Exception as e:
        logger.exception("LLM analysis failed")
        return {"execution_errors": [f"LLM analysis failed: {e!s}"]}


async def execute_llm_cells_node(state: SheetState) -> dict[str, Any]:
    """Execute LLM-generated code cells."""
    logger.info("Executing LLM-generated cells")

    logger.debug(
        "Execute LLM cells entry state",
        quality_iterations=state.get("quality_iterations"),
        needs_refinement=state.get("needs_refinement"),
    )

    try:
        # Check if notebook_json exists in state
        if "notebook_json" not in state:
            logger.warning("No notebook_json in state, skipping LLM cell execution")
            return {"execution_errors": ["No notebook available for LLM cell execution"]}

        # Reconstruct notebook from dict, converting cell dicts to Cell objects
        notebook_data = state["notebook_json"]
        cells = []
        for cell in notebook_data["cells"]:
            if isinstance(cell, dict):
                # Convert cell_type string back to enum
                cell_data = cell.copy()
                if isinstance(cell_data.get("cell_type"), str):
                    cell_data["cell_type"] = CellType(cell_data["cell_type"])
                cells.append(Cell(**cell_data))
            else:
                cells.append(cell)

        notebook = NotebookDocument(
            id=notebook_data["id"],
            cells=cells,
            metadata=notebook_data["metadata"],
            kernel_spec=notebook_data["kernel_spec"],
            language_info=notebook_data["language_info"],
        )
        llm_response = state.get("llm_response", "")

        if not llm_response:
            logger.warning("No LLM response to execute")
            return {"notebook_final": notebook_to_dict(notebook)}

        # Use shared utility to append LLM code blocks
        notebook = append_llm_code_blocks(notebook, llm_response)

        # Execute the new cells
        executed_notebook, kernel_context = await execute_notebook_cells(notebook, state)

        result = {"notebook_final": notebook_to_dict(executed_notebook)}
        # Update kernel context for later use in save operations
        result.update(kernel_context)
        return result
    except Exception as e:
        logger.exception("LLM cell execution failed")
        errors = state.get("execution_errors", [])
        errors.append(f"LLM execution failed: {e!s}")
        return {
            "execution_errors": errors,
            "notebook_final": state.get("notebook_json", {}),  # Keep original if exists
        }


async def save_results_node(state: SheetState) -> dict[str, Any]:
    """Save notebook and metadata to disk."""
    logger.info("Saving results")

    # Debug: Log the state keys and types
    logger.debug("State keys: %s", list(state.keys()))
    for key, value in state.items():
        logger.debug("State[%s] type: %s", key, type(value))
        if key == "messages" and isinstance(value, list):
            logger.debug("Messages count: %d", len(value))
            for i, msg in enumerate(value):
                logger.debug("Message[%d] type: %s", i, type(msg))

    try:
        # Get notebook data, checking for both final and initial
        if "notebook_final" in state:
            notebook_data = state["notebook_final"]
        elif "notebook_json" in state:
            notebook_data = state["notebook_json"]
        else:
            logger.warning("No notebook to save")
            return {"execution_errors": ["No notebook available to save"]}

        # Reconstruct notebook from dict, converting cell dicts to Cell objects
        cells = []
        for cell in notebook_data["cells"]:
            if isinstance(cell, dict):
                # Convert cell_type string back to enum
                cell_data = cell.copy()
                if isinstance(cell_data.get("cell_type"), str):
                    cell_data["cell_type"] = CellType(cell_data["cell_type"])
                cells.append(Cell(**cell_data))
            else:
                cells.append(cell)

        notebook = NotebookDocument(
            id=notebook_data["id"],
            cells=cells,
            metadata=notebook_data["metadata"],
            kernel_spec=notebook_data["kernel_spec"],
            language_info=notebook_data["language_info"],
        )

        # Prepare metadata - convert deterministic results if present
        deterministic_data = state.get("deterministic", {})
        if deterministic_data and hasattr(deterministic_data, "to_report"):
            # Convert PipelineResult to dict
            deterministic_data = deterministic_data.to_report()

        metadata = {
            "excel_file": str(state["excel_path"]),
            "sheet_name": state["sheet_name"],
            "analysis_date": datetime.now().isoformat(),
            "model": state.get("model", "unknown"),
            "provider": state.get("provider", "unknown"),
            "temperature": state.get("temperature", 0.1),
            "skip_deterministic": state.get("skip_deterministic", False),
            "execution_errors": state.get("execution_errors", []),
            "token_usage": state.get("token_usage", {}),
            "total_cost": state.get("total_cost", 0.0),
            "deterministic_results": deterministic_data,
            "performance": {
                "file_size_mb": state.get("file_size_mb", 0),
                "is_large_file": state.get("is_large_file", False),
                "optimizations_applied": [
                    "read_only_mode",
                    "async_io",
                    "na_filter_disabled",
                    "memory_efficient_loading",
                ],
            },
        }

        # Prepare LLM logs - safely serialize all data
        llm_logs = []
        messages = state.get("messages", [])
        if messages:
            # Convert messages to dict format, handling BaseMessage objects
            formatted_messages = []
            for idx, msg in enumerate(messages):
                try:
                    # Add detailed debug logging to inspect message structure
                    logger.debug(
                        "Processing message",
                        msg_index=idx,
                        msg_type=type(msg).__name__,
                        msg_class_mro=[cls.__name__ for cls in type(msg).__mro__],
                        has_content=hasattr(msg, "content"),
                        has_dict=hasattr(msg, "__dict__"),
                        has_dict_method=hasattr(msg, "dict"),
                        has_model_dump=hasattr(msg, "model_dump"),
                        has_response_metadata=hasattr(msg, "response_metadata"),
                        has_usage_metadata=hasattr(msg, "usage_metadata"),
                    )

                    if isinstance(msg, BaseMessage):
                        # Use LangChain's built-in serialization
                        try:
                            # Use dumpd to get a serializable dict representation
                            full_dict = dumpd(msg)
                            logger.debug("Used dumpd for BaseMessage", keys=list(full_dict.keys()))

                            # Extract key fields for our simplified format
                            # dumpd format has type and content nested in kwargs
                            if "kwargs" in full_dict and isinstance(full_dict["kwargs"], dict):
                                # New dumpd format
                                kwargs = full_dict["kwargs"]
                                msg_type = kwargs.get("type", full_dict.get("type", "unknown"))
                                content = kwargs.get("content", "")
                            else:
                                # Direct format
                                msg_type = full_dict.get("type", "unknown")
                                content = full_dict.get("content", "")

                            msg_dict = {
                                "role": msg_type.replace("Message", "").lower()
                                if isinstance(msg_type, str)
                                else "unknown",
                                "content": content,
                            }

                            # Fix role names
                            if msg_dict["role"] == "ai":
                                msg_dict["role"] = "assistant"
                            elif msg_dict["role"] == "human":
                                msg_dict["role"] = "user"

                            # Extract additional fields from serialized dict
                            for key in [
                                "name",
                                "id",
                                "type",
                                "additional_kwargs",
                                "response_metadata",
                                "usage_metadata",
                            ]:
                                if full_dict.get(key):
                                    if key == "usage_metadata" and isinstance(full_dict[key], dict):
                                        # Extract token usage information
                                        usage = {}
                                        for token_key in ["input_tokens", "output_tokens", "total_tokens"]:
                                            if token_key in full_dict[key]:
                                                usage[token_key] = int(full_dict[key][token_key])
                                        if usage:
                                            msg_dict["usage"] = usage
                                    elif key == "response_metadata" and isinstance(full_dict[key], dict):
                                        # Extract model and finish reason
                                        metadata = {}
                                        for meta_key in ["model", "model_name", "finish_reason", "stop_reason"]:
                                            if meta_key in full_dict[key]:
                                                metadata[meta_key] = str(full_dict[key][meta_key])
                                        if metadata:
                                            msg_dict["metadata"] = metadata
                                    elif key == "additional_kwargs" and isinstance(full_dict[key], dict):
                                        # Extract any additional provider-specific data
                                        if full_dict[key]:
                                            msg_dict["additional_kwargs"] = {
                                                k: str(v)
                                                for k, v in full_dict[key].items()
                                                if isinstance(v, (str, int, float, bool, type(None)))
                                            }
                                    else:
                                        # Store other simple fields
                                        if isinstance(full_dict[key], (str, int, float, bool, type(None))):
                                            msg_dict[key] = full_dict[key]
                            # Extract usage metadata if available
                            if full_dict.get("usage_metadata"):
                                usage_data = full_dict["usage_metadata"]
                                msg_dict["usage"] = {
                                    "input_tokens": usage_data.get("input_tokens", 0),
                                    "output_tokens": usage_data.get("output_tokens", 0),
                                    "total_tokens": usage_data.get("total_tokens", 0),
                                }

                            # Extract response metadata if available
                            if full_dict.get("response_metadata"):
                                resp_meta = full_dict["response_metadata"]
                                msg_dict["metadata"] = {
                                    "model": resp_meta.get("model", ""),
                                    "finish_reason": resp_meta.get("finish_reason", ""),
                                }

                        except Exception as e:
                            logger.exception(f"Could not serialize BaseMessage with dumpd: {e}")
                            # Fallback to basic extraction
                            msg_dict = {
                                "role": msg.__class__.__name__.replace("Message", "").lower(),
                                "content": str(getattr(msg, "content", "")),
                            }
                            if msg_dict["role"] == "ai":
                                msg_dict["role"] = "assistant"
                            elif msg_dict["role"] == "human":
                                msg_dict["role"] = "user"

                        # Fallback: manually extract attributes if built-in methods failed
                        if "usage" not in msg_dict and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            try:
                                usage = {}
                                usage_obj = msg.usage_metadata
                                if hasattr(usage_obj, "input_tokens"):
                                    usage["input_tokens"] = int(usage_obj.input_tokens)
                                if hasattr(usage_obj, "output_tokens"):
                                    usage["output_tokens"] = int(usage_obj.output_tokens)
                                if hasattr(usage_obj, "total_tokens"):
                                    usage["total_tokens"] = int(usage_obj.total_tokens)
                                if usage:
                                    msg_dict["usage"] = usage
                            except Exception as e:
                                logger.debug(f"Could not extract usage_metadata: {e}")

                        if (
                            "metadata" not in msg_dict
                            and hasattr(msg, "response_metadata")
                            and isinstance(msg.response_metadata, dict)
                        ):
                            try:
                                metadata = {}
                                for key in ["model", "model_name", "finish_reason", "stop_reason"]:
                                    if key in msg.response_metadata:
                                        metadata[key] = str(msg.response_metadata[key])
                                if metadata:
                                    msg_dict["metadata"] = metadata
                            except Exception as e:
                                logger.debug(f"Could not extract response_metadata: {e}")

                        formatted_messages.append(msg_dict)

                    elif hasattr(msg, "__dict__"):
                        # Try to extract relevant fields from object
                        msg_dict = {
                            "role": str(getattr(msg, "type", getattr(msg, "role", "unknown"))),
                            "content": str(getattr(msg, "content", "")),
                        }
                        formatted_messages.append(msg_dict)

                    elif isinstance(msg, dict) and "content" in msg:
                        # Already formatted as dict - ensure all values are serializable
                        msg_dict = {
                            "role": str(msg.get("role", msg.get("type", "unknown"))),
                            "content": str(msg.get("content", "")),
                        }
                        if "metadata" in msg and isinstance(msg["metadata"], dict):
                            msg_dict["metadata"] = {
                                k: v
                                for k, v in msg["metadata"].items()
                                if isinstance(v, (str, int, float, bool, type(None)))
                            }
                        formatted_messages.append(msg_dict)

                    else:
                        # Log detailed info about unknown message type
                        logger.warning(
                            "Skipping non-serializable message object",
                            msg_type=type(msg).__name__,
                            msg_str=str(msg)[:200],
                            msg_repr=repr(msg)[:200],
                        )
                        formatted_messages.append(
                            {"role": "unknown", "content": f"[Unserializable message of type {type(msg).__name__}]"}
                        )

                except Exception as e:
                    logger.exception(
                        "Error serializing message",
                        msg_index=idx,
                        msg_type=type(msg).__name__,
                        error=str(e),
                    )
                    # Add a placeholder for failed serialization
                    formatted_messages.append(
                        {"role": "error", "content": f"[Failed to serialize {type(msg).__name__}: {e}]"}
                    )

            # Create log entry with all serializable data
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "request_id": str(uuid.uuid4()),
                "model": str(state.get("model", "unknown")),
                "messages": formatted_messages,
                "response": str(state.get("llm_response", ""))[:5000],  # Truncate long responses
                "token_usage": {},
            }

            # Safely add token usage
            if "token_usage" in state and isinstance(state["token_usage"], dict):
                log_entry["token_usage"] = {
                    k: v
                    for k, v in state["token_usage"].items()
                    if isinstance(v, (int, float)) and k in ["prompt_tokens", "completion_tokens", "total_tokens"]
                }

            # Safely add other fields
            log_entry["total_cost"] = float(state.get("total_cost", 0.0))
            log_entry["execution_errors"] = [str(e) for e in state.get("execution_errors", [])]
            log_entry["quality_iterations"] = int(state.get("quality_iterations", 0))
            log_entry["quality_reasons"] = [str(r) for r in state.get("quality_reasons", [])]

            # Final safety check - ensure entire log entry is JSON serializable
            try:
                # Test serialization
                json.dumps(log_entry)
                llm_logs.append(log_entry)
            except (TypeError, ValueError) as e:
                logger.warning(f"Log entry not fully serializable, applying safe serialization: {e}")
                # Apply deep serialization to ensure everything is safe
                safe_log_entry = safe_json_serialize(log_entry)
                llm_logs.append(safe_log_entry)

        # Use the existing working save method from save_analysis_results
        # The kernel context approach had scope issues, so we use the working notebook conversion
        notebook_path, metadata_path = await asyncio.to_thread(
            save_analysis_results,
            excel_path=state["excel_path"],
            sheet_name=state["sheet_name"],
            notebook=notebook,
            metadata=metadata,
            llm_logs=llm_logs,
        )

        # Save metadata separately
        if not metadata_path.exists():  # Only save if not already saved by fallback
            await asyncio.to_thread(lambda: metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8"))

        # Save LLM logs if provided
        if llm_logs:
            log_path = output_dir / f"{sheet_filename}_llm_log.json"
            await asyncio.to_thread(lambda: log_path.write_text(json.dumps(llm_logs, indent=2), encoding="utf-8"))

        return {
            "output_path": notebook_path,
            "metadata_path": metadata_path,
        }
    except Exception as e:
        logger.exception("Failed to save results")
        return {"execution_errors": [f"Save failed: {e!s}"]}


def check_execution_results(state: SheetState) -> str:
    """Check if execution had errors that need refinement."""
    errors = state.get("execution_errors", [])
    if errors and len(errors) < 3:  # Retry up to 3 times
        logger.info("Execution had errors, will refine", error_count=len(errors))
        return "refine"
    return "quality_gate"  # Go to quality gate instead of continue


async def quality_gate_node(state: SheetState) -> dict[str, Any]:
    """Check notebook quality and decide if more analysis is needed."""
    logger.info("Checking notebook quality")

    # Check iteration limit
    quality_iterations = state.get("quality_iterations", 0)
    max_iterations = state.get("max_quality_iterations", 3)

    logger.debug(
        "Quality gate entry state",
        quality_iterations=quality_iterations,
        max_iterations=max_iterations,
        needs_refinement=state.get("needs_refinement"),
        quality_reasons=state.get("quality_reasons"),
    )

    if quality_iterations >= max_iterations:
        logger.info("Maximum quality iterations reached", iterations=quality_iterations)
        return {"needs_refinement": False, "quality_feedback": "Maximum iterations reached"}

    # Check cost limit
    if state.get("total_cost", 0.0) > 5.0:  # $5 limit
        logger.info("Cost limit reached", total_cost=state["total_cost"])
        return {"needs_refinement": False, "quality_feedback": "Cost limit reached"}

    try:
        # Get the latest notebook
        notebook_data = state.get("notebook_final") or state.get("notebook_json", {})
        if not notebook_data:
            return {"needs_refinement": False, "quality_feedback": "No notebook to evaluate"}

        # Reconstruct notebook from dict
        cells = []
        for cell in notebook_data.get("cells", []):
            if isinstance(cell, dict):
                # Convert cell_type string back to enum
                cell_data = cell.copy()
                if isinstance(cell_data.get("cell_type"), str):
                    cell_data["cell_type"] = CellType(cell_data["cell_type"])
                cells.append(Cell(**cell_data))
            else:
                cells.append(cell)

        notebook = NotebookDocument(
            id=notebook_data["id"],
            cells=cells,
            metadata=notebook_data["metadata"],
            kernel_spec=notebook_data["kernel_spec"],
            language_info=notebook_data["language_info"],
        )

        # Check quality using enhanced heuristics
        needs_more, reason = inspect_notebook_quality(notebook)

        # Track quality reasons
        quality_reasons = state.get("quality_reasons", [])
        if needs_more:
            quality_reasons.append(f"Iteration {quality_iterations + 1}: {reason}")

        result = {
            "needs_refinement": needs_more,
            "quality_feedback": reason,
            "quality_iterations": quality_iterations + 1,
            "quality_reasons": quality_reasons,
        }

        logger.debug(
            "Quality gate exit state",
            needs_refinement=result["needs_refinement"],
            quality_iterations=result["quality_iterations"],
            quality_feedback=result["quality_feedback"][:100] + "..."
            if len(result["quality_feedback"]) > 100
            else result["quality_feedback"],
        )

        return result
    except Exception as e:
        logger.exception("Quality gate check failed")
        return {"needs_refinement": False, "quality_feedback": f"Quality check error: {e!s}"}


async def refine_analysis_node(state: SheetState) -> dict[str, Any]:
    """Refine analysis based on quality feedback or execution errors."""
    logger.info("Refining analysis")

    logger.debug(
        "Refine node entry state",
        quality_iterations=state.get("quality_iterations"),
        quality_reasons=state.get("quality_reasons"),
        needs_refinement=state.get("needs_refinement"),
    )

    # Get feedback source
    errors = state.get("execution_errors", [])
    quality_feedback = state.get("quality_feedback", "")

    try:
        llm = create_llm(state)

        # Build refinement prompt based on feedback type
        if errors:
            # Error-based refinement
            error_context = "\n".join(f"- {error}" for error in errors[-3:])
            refinement_prompt = f"""The previous code execution resulted in errors:

{error_context}

Please provide corrected Python code that addresses these errors.
Remember to:
1. Fix the specific errors mentioned
2. Ensure all imports are included
3. Handle edge cases gracefully
4. Use the same variables (excel_path, sheet_name, df)"""
        else:
            # Quality-based refinement
            # Build current analysis summary
            notebook_data = state.get("notebook_final") or state.get("notebook_json", {})
            cells = [Cell(**cell) if isinstance(cell, dict) else cell for cell in notebook_data.get("cells", [])]
            notebook = NotebookDocument(
                id=notebook_data["id"],
                cells=cells,
                metadata=notebook_data["metadata"],
                kernel_spec=notebook_data["kernel_spec"],
                language_info=notebook_data["language_info"],
            )
            observations = build_observation_from_notebook(notebook)

            refinement_prompt = f"""Your previous analysis needs improvement:

{quality_feedback}

Current notebook state:
{observations[:2000]}...  # Truncate for context limit

Please provide additional Python code to address the missing analysis areas. Focus on:
1. Any specific issues mentioned in the feedback
2. Adding depth to existing analysis
3. Creating visualizations if missing
4. Documenting insights in markdown cells

IMPORTANT: Only provide NEW code cells that ADD to the analysis. Do not repeat existing analyses."""

        chain = RunnablePassthrough() | RunnableLambda(lambda _: [HumanMessage(content=refinement_prompt)]) | llm

        response = await chain.ainvoke(state)

        # Track token usage
        token_usage = {}
        total_cost = 0.0

        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("usage", {})
            token_usage = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

            # Calculate cost
            model = state.get("model", "claude-3-5-sonnet-20241022")
            total_cost = calculate_cost(
                model=model,
                input_tokens=token_usage["prompt_tokens"],
                output_tokens=token_usage["completion_tokens"],
            )

        result = {
            "llm_response": response.content,
            "messages": [*state.get("messages", []), response],
            "execution_errors": [],  # Clear errors for retry
            "token_usage": token_usage,
            "total_cost": state.get("total_cost", 0.0) + total_cost,
            # IMPORTANT: Preserve quality state fields
            "quality_iterations": state.get("quality_iterations", 0),
            "quality_reasons": state.get("quality_reasons", []),
            "needs_refinement": state.get("needs_refinement", False),
            "quality_feedback": state.get("quality_feedback", ""),
        }

        logger.debug(
            "Refine node exit state",
            quality_iterations=result["quality_iterations"],
            has_llm_response=bool(result["llm_response"]),
        )

        return result
    except Exception as e:
        logger.exception("Refinement failed")
        return {"execution_errors": (errors or []) + [f"Refinement failed: {e!s}"]}


def check_quality_gate(state: SheetState) -> str:
    """Check quality gate decision."""
    needs_refinement = state.get("needs_refinement", False)
    logger.debug(
        "Check quality gate decision",
        needs_refinement=needs_refinement,
        quality_iterations=state.get("quality_iterations"),
        decision="refine" if needs_refinement else "save_results",
    )
    if needs_refinement:
        return "refine"
    return "save_results"


def create_analysis_graph(enable_tracing: bool = False) -> StateGraph:
    """Create the complete analysis workflow graph.

    Args:
        enable_tracing: Enable LangSmith tracing for observability

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Creating analysis workflow graph", tracing_enabled=enable_tracing)

    # Initialize builder
    builder = StateGraph(SheetState)

    # Add all nodes
    builder.add_node("deterministic", deterministic_node)
    builder.add_node("create_notebook", create_notebook_node)
    builder.add_node("execute_initial", execute_initial_cells_node)
    builder.add_node("llm_analysis", llm_analysis_node)
    builder.add_node("execute_llm", execute_llm_cells_node)
    builder.add_node("quality_gate", quality_gate_node)
    builder.add_node("refine", refine_analysis_node)
    builder.add_node("save_results", save_results_node)

    # Define the flow
    builder.add_edge(START, "deterministic")
    builder.add_edge("deterministic", "create_notebook")
    builder.add_edge("create_notebook", "execute_initial")
    builder.add_edge("execute_initial", "llm_analysis")
    builder.add_edge("llm_analysis", "execute_llm")

    # Conditional edge for error handling
    builder.add_conditional_edges(
        "execute_llm",
        check_execution_results,
        {
            "refine": "refine",
            "quality_gate": "quality_gate",  # Go to quality gate when no errors
        },
    )

    # Quality gate conditional edge
    builder.add_conditional_edges(
        "quality_gate",
        check_quality_gate,
        {
            "refine": "refine",
            "save_results": "save_results",
        },
    )

    # Refinement loop back to execution
    builder.add_edge("refine", "execute_llm")

    # Final edge
    builder.add_edge("save_results", END)

    # Compile with memory checkpointer and optional tracing
    checkpointer = MemorySaver()

    if enable_tracing:
        # Enable LangSmith tracing if configured
        import os

        if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
            from langsmith import Client

            try:
                client = Client()
                logger.info("LangSmith tracing enabled", project=os.environ.get("LANGCHAIN_PROJECT", "default"))
            except Exception as e:
                logger.warning("Failed to enable LangSmith tracing", error=str(e))

    return builder.compile(checkpointer=checkpointer)


async def analyze_sheet_with_langchain(
    excel_path: Path,
    sheet_name: str,
    *,
    skip_deterministic: bool = False,
    provider: str = "anthropic",
    model: str | None = None,
    temperature: float = 0.1,
    enable_tracing: bool = False,
) -> SheetState:
    """Main entry point for LangChain-based analysis.

    Args:
        excel_path: Path to Excel file to analyze
        sheet_name: Name of sheet to analyze
        skip_deterministic: Skip deterministic analysis phase
        provider: LLM provider ("anthropic" or "openai")
        model: Specific model to use (defaults based on provider)
        temperature: LLM temperature for response variability
        enable_tracing: Enable LangSmith tracing

    Returns:
        Final state containing all analysis results
    """
    logger.info(
        "Starting LangChain analysis",
        excel_path=str(excel_path),
        sheet_name=sheet_name,
        provider=provider,
        model=model,
        tracing=enable_tracing,
    )

    # Set default model based on provider
    if model is None:
        model = "claude-3-5-sonnet-20241022" if provider == "anthropic" else "gpt-4-turbo-preview"

    # Check file size for optimization decisions
    file_size_mb, is_large_file = await check_file_size(excel_path)
    if is_large_file:
        logger.warning(
            "Large Excel file detected",
            file_size_mb=f"{file_size_mb:.2f}",
            threshold_mb=LARGE_FILE_THRESHOLD_MB,
        )

    # Initialize state
    initial_state = SheetState(
        excel_path=excel_path,
        sheet_name=sheet_name,
        skip_deterministic=skip_deterministic,
        provider=provider,
        model=model,
        temperature=temperature,
        messages=[],
        execution_errors=[],
        token_usage={},
        total_cost=0.0,
        file_size_mb=file_size_mb,
        is_large_file=is_large_file,
        quality_iterations=0,
        quality_reasons=[],
        max_quality_iterations=3,
        needs_refinement=False,
        quality_feedback="",
    )

    # Create and run the graph
    graph = create_analysis_graph(enable_tracing=enable_tracing)

    # Run with a thread ID for checkpointing
    config = {"configurable": {"thread_id": f"{excel_path.stem}_{sheet_name}"}}

    # Execute the graph
    final_state = await graph.ainvoke(initial_state, config)

    return cast("SheetState", final_state)


# TEMPORARY STUBS - These need to be reimplemented using proper architecture
# TODO: Reimplement these functions properly using core_exec, plugins, and workflows


def create_sheet_notebook(
    excel_path: Path, sheet_name: str, deterministic_results: dict | None = None
) -> NotebookDocument:
    """Create initial notebook for sheet analysis.

    TODO: This should use:
    - core_exec.notebook_builder for creating the notebook
    - plugins.spreadsheet.tasks for domain-specific setup
    - workflows.notebook_workflow for orchestration
    """
    from spreadsheet_analyzer.core_exec.notebook_builder import NotebookBuilder

    builder = NotebookBuilder()
    builder.add_markdown_cell(f"# Analysis of {sheet_name} from {excel_path.name}")

    # Add data loading cell
    builder.add_code_cell(f"""import pandas as pd
excel_path = r"{excel_path}"
sheet_name = "{sheet_name}"
df = pd.read_excel(excel_path, sheet_name=sheet_name)
print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
df.head()""")

    # Add deterministic results if available
    if deterministic_results:
        builder.add_markdown_cell("## Deterministic Analysis Results")
        builder.add_code_cell(f"# Deterministic results\n{deterministic_results}")

    # Convert to NotebookDocument
    nb_dict = builder.build()
    cells = [
        Cell(
            id=f"cell_{i}",
            cell_type=CellType.MARKDOWN if cell["cell_type"] == "markdown" else CellType.CODE,
            source=cell["source"],
            metadata={},
            outputs=[],
        )
        for i, cell in enumerate(nb_dict["cells"])
    ]

    return NotebookDocument(cells=cells, metadata={})


def build_observation_from_notebook(notebook: NotebookDocument) -> str:
    """Build observation string from notebook execution results.

    TODO: This should use plugins.spreadsheet.quality for analysis
    """
    observations = []

    for i, cell in enumerate(notebook.cells):
        if cell.outputs:
            observations.append(f"Cell {i + 1} outputs:")
            for output in cell.outputs:
                if isinstance(output, dict):
                    if output.get("type") == "stream":
                        observations.append(f"  {output.get('text', '')}")
                    elif output.get("type") == "execute_result":
                        observations.append(f"  Result: {output.get('data', {})}")

    return "\n".join(observations) if observations else "No outputs yet"


def append_llm_code_blocks(notebook: NotebookDocument, llm_response: str) -> NotebookDocument:
    """Extract and append code blocks from LLM response.

    TODO: This should use:
    - core_exec.notebook_builder for adding cells
    - plugins.spreadsheet.tasks for code extraction
    """
    import re

    # Extract code blocks from LLM response
    code_blocks = re.findall(r"```python\n(.*?)\n```", llm_response, re.DOTALL)

    for code in code_blocks:
        new_cell = Cell(
            id=f"llm_cell_{len(notebook.cells)}", cell_type=CellType.CODE, source=code.strip(), metadata={}, outputs=[]
        )
        notebook.cells.append(new_cell)

    return notebook


def inspect_notebook_quality(notebook: NotebookDocument, min_insights: int = 2) -> tuple[bool, str]:
    """Check if notebook meets quality standards.

    TODO: This should use plugins.spreadsheet.quality for proper inspection
    """
    # Count cells with meaningful analysis
    code_cells = [c for c in notebook.cells if c.cell_type == CellType.CODE]
    cells_with_output = [c for c in code_cells if c.outputs]

    if len(cells_with_output) < min_insights:
        return True, f"Only {len(cells_with_output)} cells with output, need at least {min_insights}"

    # Check for errors in outputs
    for cell in notebook.cells:
        for output in cell.outputs:
            if isinstance(output, dict) and output.get("ename"):
                return True, f"Error in cell: {output.get('ename')}"

    return False, "Quality standards met"


async def save_analysis_results(
    excel_path: Path, sheet_name: str, notebook: NotebookDocument, metadata: dict, llm_logs: list[dict]
) -> tuple[Path, Path]:
    """Save notebook and metadata to disk.

    TODO: This should use:
    - core_exec.notebook_io for saving
    - workflows.notebook_workflow for orchestration
    """
    import json

    from spreadsheet_analyzer.core_exec.notebook_io import NotebookIO

    # Create output directory
    output_dir = Path("analysis_results") / excel_path.stem / sheet_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save notebook
    io = NotebookIO()
    notebook_path = output_dir / f"{sheet_name}_analysis.ipynb"
    io.save_notebook(notebook.to_nbformat(), notebook_path)

    # Save metadata
    metadata_path = output_dir / f"{sheet_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump({"metadata": metadata, "llm_logs": llm_logs}, f, indent=2)

    return notebook_path, metadata_path
