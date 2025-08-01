# OpenAI Code Interpreter API

## Executive Summary

OpenAI's Code Interpreter (now part of the Assistants API) is a hosted code execution environment that enables GPT models to write and run Python code in a sandboxed environment. It provides file handling, data analysis capabilities, and persistent sessions, making it particularly suitable for data analysis tasks including Excel file processing.

## Core Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Application                        │
│              API Calls, File Upload, Results                │
├─────────────────────────────────────────────────────────────┤
│                  OpenAI Assistants API                       │
│           Thread Management, Message Handling                │
├─────────────────────────────────────────────────────────────┤
│                   Code Interpreter Tool                      │
│         Sandboxed Python, File System, Libraries            │
├─────────────────────────────────────────────────────────────┤
│                   Execution Environment                      │
│    Jupyter-like Kernel, Pre-installed Packages, Storage     │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Assistants API**: Main interface for creating and managing assistants
1. **Threads**: Conversation sessions with persistent context
1. **Messages**: User and assistant communications
1. **Runs**: Execution instances with tool calls
1. **Files**: Uploaded files and generated outputs

## API Setup and Configuration

### Basic Setup

```python
from openai import OpenAI
client = OpenAI(api_key="your-api-key")

# Create an assistant with Code Interpreter
assistant = client.beta.assistants.create(
    name="Excel Analyst",
    instructions="You are an expert Excel analyst. Use code interpreter to analyze Excel files, create visualizations, and generate reports.",
    tools=[{"type": "code_interpreter"}],
    model="gpt-4-1106-preview"
)
```

### File Handling

```python
# Upload Excel file
file = client.files.create(
    file=open("sales_data.xlsx", "rb"),
    purpose='assistants'
)

# Create thread with file
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": "Analyze this Excel file and create a summary report",
            "file_ids": [file.id]
        }
    ]
)
```

## Usage Patterns

### 1. Basic Excel Analysis

```python
# Run analysis
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Wait for completion
import time
while run.status != 'completed':
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    time.sleep(1)

# Get results
messages = client.beta.threads.messages.list(
    thread_id=thread.id
)
```

### 2. Streaming Responses

```python
# Stream execution for real-time updates
from typing import Iterable
from openai import AssistantEventHandler

class ExcelAnalysisHandler(AssistantEventHandler):
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)
    
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
    
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)
    
    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

# Stream the run
with client.beta.threads.runs.create_and_stream(
    thread_id=thread.id,
    assistant_id=assistant.id,
    event_handler=ExcelAnalysisHandler(),
) as stream:
    stream.until_done()
```

### 3. Advanced Analysis Workflow

```python
class ExcelAnalysisAssistant:
    def __init__(self, client):
        self.client = client
        self.assistant = self._create_assistant()
    
    def _create_assistant(self):
        return self.client.beta.assistants.create(
            name="Advanced Excel Analyst",
            instructions="""You are an Excel analysis expert. For each file:
            1. Load and examine the structure
            2. Identify data types and relationships
            3. Perform statistical analysis
            4. Create appropriate visualizations
            5. Generate insights and recommendations
            """,
            tools=[{"type": "code_interpreter"}],
            model="gpt-4-1106-preview"
        )
    
    async def analyze_excel(self, file_path, analysis_type="comprehensive"):
        # Upload file
        with open(file_path, 'rb') as f:
            file = self.client.files.create(file=f, purpose='assistants')
        
        # Create thread with specific instructions
        thread = self.client.beta.threads.create()
        
        # Add message with analysis request
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Perform {analysis_type} analysis on this Excel file",
            file_ids=[file.id]
        )
        
        # Run analysis
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant.id
        )
        
        # Return thread for monitoring
        return thread, run
```

## Code Execution Environment

### Pre-installed Libraries

The Code Interpreter environment includes:

```python
# Data Analysis
import pandas as pd
import numpy as np
import scipy
import statsmodels

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly

# File Handling
import openpyxl  # Excel files
import xlrd      # Legacy Excel
import csv       # CSV files
import json      # JSON data

# Other Utilities
import datetime
import requests
import beautifulsoup4
```

### Excel Processing Example

```python
# Example code that Code Interpreter might generate
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Excel file
excel_file = pd.ExcelFile('/mnt/data/sales_data.xlsx')
sheets = excel_file.sheet_names

# Analyze each sheet
summaries = {}
for sheet in sheets:
    df = pd.read_excel(excel_file, sheet_name=sheet)
    
    # Basic statistics
    summaries[sheet] = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict()
    }
    
    # Create visualizations
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(15, 5))
        for i, col in enumerate(numeric_cols[:3]):
            df[col].hist(ax=axes[i] if len(numeric_cols) > 1 else axes)
            axes[i].set_title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig(f'/mnt/data/{sheet}_distributions.png')
        plt.close()

# Generate summary report
report = "# Excel Analysis Report\n\n"
for sheet, summary in summaries.items():
    report += f"## Sheet: {sheet}\n"
    report += f"- Dimensions: {summary['shape'][0]} rows × {summary['shape'][1]} columns\n"
    report += f"- Missing values: {sum(summary['missing'].values())} total\n\n"
```

## Integration Patterns

### 1. Notebook-Style Reports

```python
def create_notebook_from_thread(client, thread_id):
    """Convert Code Interpreter session to Jupyter notebook"""
    import nbformat as nbf
    
    # Get all messages
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    
    # Create notebook
    nb = nbf.v4.new_notebook()
    cells = []
    
    for msg in messages.data:
        if msg.role == 'user':
            cells.append(nbf.v4.new_markdown_cell(f"**User Query:**\n{msg.content[0].text.value}"))
        else:
            # Add code cells from code interpreter
            for content in msg.content:
                if hasattr(content, 'text'):
                    cells.append(nbf.v4.new_markdown_cell(content.text.value))
            
            # Extract code from annotations
            if hasattr(msg, 'run_id'):
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=msg.run_id
                )
                for step in client.beta.threads.runs.steps.list(
                    thread_id=thread_id,
                    run_id=msg.run_id
                ):
                    if step.type == 'tool_calls':
                        for tool_call in step.step_details.tool_calls:
                            if tool_call.type == 'code_interpreter':
                                code = tool_call.code_interpreter.input
                                cells.append(nbf.v4.new_code_cell(code))
    
    nb.cells = cells
    return nb
```

### 2. Comparison with Local Execution

```python
class HybridExcelAnalyzer:
    """Combines Code Interpreter with local execution"""
    
    def __init__(self, openai_client, local_kernel):
        self.openai = openai_client
        self.kernel = local_kernel
        self.assistant = self._create_assistant()
    
    def analyze_sensitive_data(self, file_path):
        """Use local execution for sensitive data"""
        # Local analysis for sensitive data
        result = self.kernel.execute(f"""
        import pandas as pd
        df = pd.read_excel('{file_path}')
        # Anonymize sensitive columns
        sensitive_cols = ['SSN', 'email', 'phone']
        for col in sensitive_cols:
            if col in df.columns:
                df[col] = 'REDACTED'
        df.to_excel('anonymized_data.xlsx')
        """)
        
        # Upload anonymized version to Code Interpreter
        with open('anonymized_data.xlsx', 'rb') as f:
            file = self.openai.files.create(file=f, purpose='assistants')
        
        # Perform analysis on anonymized data
        thread = self.openai.beta.threads.create()
        # ... continue with Code Interpreter analysis
```

### 3. Structured Output Extraction

```python
def extract_analysis_results(client, thread_id):
    """Extract structured data from Code Interpreter results"""
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    
    results = {
        'summaries': [],
        'visualizations': [],
        'insights': [],
        'code_blocks': []
    }
    
    for message in messages.data:
        if message.role == 'assistant':
            # Extract text insights
            for content in message.content:
                if content.type == 'text':
                    text = content.text.value
                    if 'insight:' in text.lower():
                        results['insights'].append(text)
                    elif 'summary:' in text.lower():
                        results['summaries'].append(text)
                elif content.type == 'image_file':
                    # Download visualization
                    file_id = content.image_file.file_id
                    results['visualizations'].append(file_id)
    
    return results
```

## Advantages and Limitations

### Advantages

1. **Sandboxed Environment**: Safe execution without local risks
1. **Pre-configured Libraries**: Common data science packages available
1. **File Persistence**: Files persist within thread session
1. **Automatic Error Handling**: GPT handles errors and retries
1. **Natural Language Interface**: No need for explicit code

### Limitations

1. **Package Limitations**: Cannot install additional packages
1. **Resource Constraints**: Memory and CPU limits
1. **Session Timeout**: Sessions expire after inactivity
1. **Network Restrictions**: No external API calls
1. **Cost**: Token and execution costs can add up

### Comparison with notebook_cli.py

| Feature                   | Code Interpreter API | notebook_cli.py      |
| ------------------------- | -------------------- | -------------------- |
| **Execution Environment** | Cloud sandboxed      | Local Jupyter kernel |
| **Package Management**    | Fixed set            | Full pip access      |
| **State Persistence**     | Thread-based         | Kernel-based         |
| **File Access**           | Upload required      | Direct local access  |
| **Security**              | Sandboxed            | Full system access   |
| **Cost**                  | Per-token pricing    | Local resources only |
| **Customization**         | Limited              | Fully customizable   |

## Best Practices

### 1. Error Handling

```python
async def robust_analysis(client, assistant_id, file_path, max_retries=3):
    """Robust analysis with retry logic"""
    for attempt in range(max_retries):
        try:
            # Upload file
            with open(file_path, 'rb') as f:
                file = client.files.create(file=f, purpose='assistants')
            
            # Create thread and run
            thread = client.beta.threads.create()
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content="Analyze this Excel file comprehensively",
                file_ids=[file.id]
            )
            
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_id
            )
            
            # Wait for completion with timeout
            timeout = 300  # 5 minutes
            start_time = time.time()
            while run.status not in ['completed', 'failed']:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Analysis timed out")
                
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                await asyncio.sleep(1)
            
            if run.status == 'completed':
                return thread
            else:
                raise Exception(f"Run failed: {run.last_error}")
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

### 2. Cost Optimization

```python
class CostOptimizedAnalyzer:
    def __init__(self, client):
        self.client = client
        self.token_count = 0
        self.file_count = 0
    
    def estimate_cost(self, file_size_mb):
        """Estimate analysis cost"""
        # Rough estimates
        tokens_per_mb = 50000  # Approximate
        cost_per_1k_tokens = 0.03  # GPT-4 pricing
        
        estimated_tokens = file_size_mb * tokens_per_mb
        estimated_cost = (estimated_tokens / 1000) * cost_per_1k_tokens
        
        return {
            'estimated_tokens': estimated_tokens,
            'estimated_cost': estimated_cost,
            'recommendation': 'Consider local analysis' if estimated_cost > 10 else 'Proceed with API'
        }
    
    def analyze_with_budget(self, file_path, budget_usd=5.0):
        """Analyze with budget constraints"""
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        estimate = self.estimate_cost(file_size)
        
        if estimate['estimated_cost'] > budget_usd:
            # Use sampling for large files
            return self._analyze_sample(file_path, budget_usd)
        else:
            return self._full_analysis(file_path)
```

### 3. Hybrid Workflow

```python
class HybridWorkflow:
    """Combine Code Interpreter with local notebook execution"""
    
    def __init__(self, openai_client, notebook_cli):
        self.openai = openai_client
        self.notebook_cli = notebook_cli
    
    async def analyze_excel(self, file_path):
        # Step 1: Quick exploration with Code Interpreter
        exploration = await self._explore_with_api(file_path)
        
        # Step 2: Detailed local analysis based on exploration
        analysis_plan = self._extract_analysis_plan(exploration)
        
        # Step 3: Execute detailed analysis locally
        notebook = await self.notebook_cli.create_analysis_notebook(
            file_path,
            plan=analysis_plan
        )
        
        # Step 4: Generate final report with Code Interpreter
        report = await self._generate_report(notebook, exploration)
        
        return report
```

## Integration Example

```python
# Complete integration example
class CodeInterpreterExcelAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.assistant = self._setup_assistant()
    
    def _setup_assistant(self):
        return self.client.beta.assistants.create(
            name="Excel Analysis Expert",
            instructions="""You are an Excel analysis expert. When analyzing files:
            1. First examine the file structure and all sheets
            2. Identify data types, relationships, and quality issues
            3. Perform appropriate statistical analysis
            4. Create meaningful visualizations
            5. Provide actionable insights
            
            Always explain your analysis process and findings clearly.""",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4-1106-preview"
        )
    
    def analyze(self, excel_path):
        # Upload file
        with open(excel_path, 'rb') as f:
            file = self.client.files.create(file=f, purpose='assistants')
        
        # Create analysis thread
        thread = self.client.beta.threads.create(
            messages=[{
                "role": "user",
                "content": "Please perform a comprehensive analysis of this Excel file",
                "file_ids": [file.id]
            }]
        )
        
        # Run analysis
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=self.assistant.id
        )
        
        # Stream results
        with self.client.beta.threads.runs.create_and_stream(
            thread_id=thread.id,
            assistant_id=self.assistant.id,
        ) as stream:
            for event in stream:
                if event.type == 'thread.message.delta':
                    print(event.data.delta.content[0].text.value, end='')
        
        return thread

# Usage
analyzer = CodeInterpreterExcelAnalyzer(api_key="sk-...")
thread = analyzer.analyze("sales_report.xlsx")
```

## Future Developments

### Expected Enhancements (2024-2025)

1. **Extended Package Support**: More pre-installed libraries
1. **Persistent Storage**: Long-term file storage
1. **Custom Environments**: User-defined environments
1. **Enhanced Security**: More granular permissions
1. **Better Integration**: Direct notebook export

## Conclusion

OpenAI's Code Interpreter API provides a powerful, managed environment for code execution that complements local solutions like notebook_cli.py. Key considerations:

**Use Code Interpreter API when:**

- Security is paramount (sandboxed environment)
- Quick analysis without setup is needed
- Working with non-sensitive data
- Cost is not a primary concern

**Use notebook_cli.py when:**

- Full control over environment is required
- Working with sensitive data
- Need custom packages or tools
- Cost optimization is important
- Integration with existing infrastructure

The ideal solution may be a hybrid approach that leverages both systems based on specific requirements.

## References

1. [OpenAI Assistants API Documentation](https://platform.openai.com/docs/assistants)
1. [Code Interpreter Guide](https://platform.openai.com/docs/assistants/tools/code-interpreter)
1. [API Reference](https://platform.openai.com/docs/api-reference/assistants)
1. [Pricing Information](https://openai.com/pricing)
1. [Community Examples](https://github.com/openai/openai-cookbook)
