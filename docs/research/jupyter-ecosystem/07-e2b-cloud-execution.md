# E2B (e2b.dev) - Cloud Code Execution for LLMs

## Executive Summary

E2B (Environments to Build) is a cloud infrastructure platform specifically designed for AI agents and LLMs to execute code safely. It provides sandboxed cloud environments that can be programmatically created, managed, and destroyed, making it ideal for building LLM applications that need to run untrusted code or perform complex computational tasks.

## Core Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Application                        │
│              SDK (Python/JS), API Calls, WebSocket          │
├─────────────────────────────────────────────────────────────┤
│                      E2B API Layer                          │
│     Sandbox Management, File System, Process Control        │
├─────────────────────────────────────────────────────────────┤
│                   Sandbox Environments                       │
│    Docker Containers, Firecracker VMs, Custom Templates     │
├─────────────────────────────────────────────────────────────┤
│                 Execution Infrastructure                     │
│       Isolated Compute, Networking, Storage, Security       │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Sandboxes**: Isolated execution environments
1. **Templates**: Pre-configured environment templates
1. **SDK**: Python and JavaScript client libraries
1. **Process Management**: Execute and monitor processes
1. **File System**: Upload, download, and manipulate files
1. **Networking**: Controlled network access

## Installation and Setup

### Basic Installation

```bash
# Install Python SDK
pip install e2b

# Install with code interpreter template
pip install e2b-code-interpreter
```

### Authentication

```python
from e2b import Sandbox
import os

# Set API key
os.environ["E2B_API_KEY"] = "your-api-key"

# Or pass directly
sandbox = Sandbox(api_key="your-api-key")
```

## Core Features

### 1. Sandbox Management

```python
from e2b import Sandbox

# Create a sandbox
sandbox = Sandbox()

# Create with specific template
sandbox = Sandbox(template="python-data-analysis")

# Sandbox lifecycle
try:
    # Use sandbox
    result = sandbox.process.start_and_wait("python script.py")
finally:
    # Always cleanup
    sandbox.close()

# Context manager pattern
with Sandbox() as sandbox:
    result = sandbox.process.start_and_wait("python script.py")
```

### 2. Code Execution

```python
# Execute Python code
python_code = """
import pandas as pd
import matplotlib.pyplot as plt

# Load Excel file
df = pd.read_excel('/tmp/data.xlsx')
print(f"Loaded {len(df)} rows")

# Create visualization
df.plot(kind='bar')
plt.savefig('/tmp/output.png')
"""

# Run code
proc = sandbox.process.start_and_wait(
    cmd=f"python -c '{python_code}'",
    timeout=30
)

print(proc.stdout)
print(proc.stderr)
```

### 3. File Operations

```python
# Upload files
with open('local_data.xlsx', 'rb') as f:
    sandbox.upload_file(f, '/tmp/data.xlsx')

# Download files
content = sandbox.download_file('/tmp/output.png')
with open('local_output.png', 'wb') as f:
    f.write(content)

# List files
files = sandbox.list('/tmp')
for file in files:
    print(f"{file.name} - {file.size} bytes")
```

## E2B Code Interpreter

### Specialized Template for Data Analysis

```python
from e2b_code_interpreter import CodeInterpreter

# Create interpreter sandbox
with CodeInterpreter() as interpreter:
    # Execute code with rich output
    result = interpreter.notebook.exec_cell("""
    import pandas as pd
    import numpy as np
    
    # Create sample data
    data = {
        'Product': ['A', 'B', 'C', 'D'],
        'Sales': [100, 150, 200, 175],
        'Profit': [20, 35, 45, 30]
    }
    df = pd.DataFrame(data)
    
    # Display
    print(df)
    df.describe()
    """)
    
    print(result.text)
    print(result.data)  # Structured output
```

### Excel Analysis Example

```python
class E2BExcelAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
    
    async def analyze_excel(self, file_path):
        with CodeInterpreter(api_key=self.api_key) as sandbox:
            # Upload Excel file
            with open(file_path, 'rb') as f:
                sandbox.upload_file(f, '/tmp/input.xlsx')
            
            # Analysis code
            analysis_code = """
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Load all sheets
            excel_file = pd.ExcelFile('/tmp/input.xlsx')
            sheets = {}
            
            for sheet_name in excel_file.sheet_names:
                sheets[sheet_name] = pd.read_excel(excel_file, sheet_name=sheet_name)
                print(f"\\nSheet: {sheet_name}")
                print(f"Shape: {sheets[sheet_name].shape}")
                print(sheets[sheet_name].head())
            
            # Generate summary statistics
            summary = {}
            for name, df in sheets.items():
                summary[name] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'numeric_columns': len(df.select_dtypes(include=['number']).columns),
                    'missing_values': df.isnull().sum().sum()
                }
            
            # Create visualizations
            fig, axes = plt.subplots(len(sheets), 1, figsize=(10, 6*len(sheets)))
            if len(sheets) == 1:
                axes = [axes]
            
            for idx, (name, df) in enumerate(sheets.items()):
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    df[numeric_cols].plot(kind='box', ax=axes[idx])
                    axes[idx].set_title(f'Numeric Distributions - {name}')
            
            plt.tight_layout()
            plt.savefig('/tmp/analysis.png')
            
            # Return summary
            summary
            """
            
            # Execute analysis
            result = sandbox.notebook.exec_cell(analysis_code)
            
            # Download results
            if sandbox.exists('/tmp/analysis.png'):
                plot_data = sandbox.download_file('/tmp/analysis.png')
                with open('excel_analysis.png', 'wb') as f:
                    f.write(plot_data)
            
            return {
                'output': result.text,
                'data': result.data,
                'files': ['excel_analysis.png']
            }
```

## Integration with LLM Workflows

### 1. LangChain Integration

```python
from langchain.tools import Tool
from e2b_code_interpreter import CodeInterpreter

class E2BTool(Tool):
    def __init__(self):
        self.name = "e2b_executor"
        self.description = "Execute Python code in a sandboxed cloud environment"
    
    def _run(self, code: str) -> str:
        with CodeInterpreter() as sandbox:
            result = sandbox.notebook.exec_cell(code)
            return f"Output: {result.text}\nData: {result.data}"
    
    async def _arun(self, code: str) -> str:
        # Async version
        return self._run(code)

# Use in LangChain
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

tools = [E2BTool()]
agent = initialize_agent(tools, OpenAI(), agent="zero-shot-react-description")
```

### 2. Custom LLM Integration

```python
class E2BCodeExecutor:
    def __init__(self, llm_client, e2b_api_key):
        self.llm = llm_client
        self.e2b_api_key = e2b_api_key
    
    async def analyze_with_code(self, user_query):
        # Get code from LLM
        prompt = f"""
        Write Python code to: {user_query}
        Include data loading, analysis, and visualization.
        Save any plots to /tmp/output.png
        """
        
        code = self.llm.generate(prompt)
        
        # Execute in E2B
        with CodeInterpreter(api_key=self.e2b_api_key) as sandbox:
            result = sandbox.notebook.exec_cell(code)
            
            # Check for outputs
            outputs = []
            if sandbox.exists('/tmp/output.png'):
                plot = sandbox.download_file('/tmp/output.png')
                outputs.append(('plot', plot))
            
            return {
                'code': code,
                'output': result.text,
                'data': result.data,
                'files': outputs
            }
```

### 3. Notebook-Style Execution

```python
class E2BNotebookSession:
    def __init__(self, template="python-data-analysis"):
        self.sandbox = None
        self.cells = []
        self.template = template
    
    def start(self):
        self.sandbox = CodeInterpreter(template=self.template)
        return self
    
    def add_cell(self, code, cell_type='code'):
        cell = {
            'type': cell_type,
            'source': code,
            'execution_count': len(self.cells) + 1
        }
        
        if cell_type == 'code':
            result = self.sandbox.notebook.exec_cell(code)
            cell['output'] = {
                'text': result.text,
                'data': result.data,
                'error': result.error
            }
        
        self.cells.append(cell)
        return cell
    
    def to_notebook(self):
        """Convert to Jupyter notebook format"""
        import nbformat as nbf
        
        nb = nbf.v4.new_notebook()
        
        for cell in self.cells:
            if cell['type'] == 'code':
                nb_cell = nbf.v4.new_code_cell(
                    source=cell['source'],
                    execution_count=cell['execution_count']
                )
                if 'output' in cell:
                    output = cell['output']
                    if output['text']:
                        nb_cell.outputs.append(
                            nbf.v4.new_output(
                                output_type='stream',
                                name='stdout',
                                text=output['text']
                            )
                        )
                nb.cells.append(nb_cell)
            else:
                nb.cells.append(nbf.v4.new_markdown_cell(cell['source']))
        
        return nb
    
    def close(self):
        if self.sandbox:
            self.sandbox.close()
```

## Advanced Features

### 1. Custom Templates

```python
# Create custom template with specific packages
template_config = {
    "dockerfile": """
    FROM python:3.11
    RUN pip install pandas numpy matplotlib seaborn openpyxl xlwings
    RUN pip install scikit-learn statsmodels
    WORKDIR /workspace
    """,
    "name": "excel-analysis-env"
}

# Use custom template
sandbox = Sandbox(template="excel-analysis-env")
```

### 2. Long-Running Processes

```python
# Start a Jupyter kernel in E2B
proc = sandbox.process.start(
    "jupyter kernel --kernel=python3",
    background=True
)

# Connect to kernel
kernel_info = json.loads(proc.stdout)
kernel_url = f"ws://localhost:{kernel_info['port']}"

# Send code to kernel
# ... WebSocket communication ...
```

### 3. Resource Management

```python
class E2BResourceManager:
    def __init__(self, max_sandboxes=5):
        self.max_sandboxes = max_sandboxes
        self.active_sandboxes = []
        self.available_sandboxes = []
    
    def get_sandbox(self):
        if self.available_sandboxes:
            return self.available_sandboxes.pop()
        elif len(self.active_sandboxes) < self.max_sandboxes:
            sandbox = Sandbox()
            self.active_sandboxes.append(sandbox)
            return sandbox
        else:
            raise Exception("Maximum sandbox limit reached")
    
    def release_sandbox(self, sandbox):
        sandbox.clear()  # Reset state
        self.available_sandboxes.append(sandbox)
    
    def cleanup(self):
        for sandbox in self.active_sandboxes + self.available_sandboxes:
            sandbox.close()
```

## Comparison with notebook_cli.py

### Architecture Comparison

| Feature                | E2B                      | notebook_cli.py    |
| ---------------------- | ------------------------ | ------------------ |
| **Execution Location** | Cloud (sandboxed)        | Local machine      |
| **Isolation**          | Complete isolation       | Process isolation  |
| **Package Management** | Template-based           | Full pip access    |
| **File Access**        | Upload/download required | Direct file system |
| **Scaling**            | Cloud scaling            | Local resources    |
| **Cost**               | Per-execution pricing    | Local compute only |

### Integration Pattern

```python
class HybridExecutor:
    """Combine E2B with local notebook execution"""
    
    def __init__(self, notebook_cli, e2b_api_key):
        self.notebook_cli = notebook_cli
        self.e2b_api_key = e2b_api_key
    
    async def execute_based_on_security(self, code, file_path):
        # Analyze code for security risks
        if self._is_safe_code(code):
            # Execute locally for better performance
            return await self.notebook_cli.execute_code(code)
        else:
            # Execute in E2B sandbox for security
            with CodeInterpreter(api_key=self.e2b_api_key) as sandbox:
                # Upload file if needed
                if file_path:
                    with open(file_path, 'rb') as f:
                        sandbox.upload_file(f, '/tmp/input.xlsx')
                
                result = sandbox.notebook.exec_cell(code)
                return result
    
    def _is_safe_code(self, code):
        # Check for potentially dangerous operations
        dangerous_patterns = [
            'os.system', 'subprocess', '__import__',
            'exec', 'eval', 'open.*w', 'shutil.rmtree'
        ]
        return not any(pattern in code for pattern in dangerous_patterns)
```

## Best Practices

### 1. Error Handling

```python
async def robust_e2b_execution(code, max_retries=3):
    for attempt in range(max_retries):
        sandbox = None
        try:
            sandbox = CodeInterpreter()
            result = sandbox.notebook.exec_cell(code)
            
            if result.error:
                if attempt < max_retries - 1:
                    # Retry with modified code
                    code = fix_common_errors(code, result.error)
                    continue
                else:
                    raise Exception(f"Execution failed: {result.error}")
            
            return result
            
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
        finally:
            if sandbox:
                sandbox.close()
```

### 2. Performance Optimization

```python
class E2BConnectionPool:
    """Reuse sandboxes for better performance"""
    
    def __init__(self, pool_size=3):
        self.pool = []
        self.in_use = set()
        self.pool_size = pool_size
    
    async def acquire(self):
        # Try to get from pool
        for sandbox in self.pool:
            if sandbox not in self.in_use:
                self.in_use.add(sandbox)
                return sandbox
        
        # Create new if under limit
        if len(self.pool) < self.pool_size:
            sandbox = CodeInterpreter()
            self.pool.append(sandbox)
            self.in_use.add(sandbox)
            return sandbox
        
        # Wait for available sandbox
        while True:
            await asyncio.sleep(0.1)
            for sandbox in self.pool:
                if sandbox not in self.in_use:
                    self.in_use.add(sandbox)
                    return sandbox
    
    def release(self, sandbox):
        sandbox.notebook.exec_cell("globals().clear()")  # Clear state
        self.in_use.remove(sandbox)
```

### 3. Security Considerations

```python
class SecureE2BExecutor:
    def __init__(self):
        self.blocked_imports = ['os', 'subprocess', 'sys']
        self.blocked_functions = ['exec', 'eval', '__import__']
    
    def sanitize_code(self, code):
        """Remove potentially dangerous code"""
        lines = code.split('\n')
        sanitized = []
        
        for line in lines:
            # Check for blocked imports
            if any(f"import {mod}" in line for mod in self.blocked_imports):
                sanitized.append(f"# BLOCKED: {line}")
            # Check for blocked functions
            elif any(func in line for func in self.blocked_functions):
                sanitized.append(f"# BLOCKED: {line}")
            else:
                sanitized.append(line)
        
        return '\n'.join(sanitized)
    
    def execute_safely(self, code):
        sanitized_code = self.sanitize_code(code)
        
        with CodeInterpreter() as sandbox:
            # Set resource limits
            sandbox.set_limits(
                max_memory="512MB",
                max_cpu="1",
                timeout=60
            )
            
            result = sandbox.notebook.exec_cell(sanitized_code)
            return result
```

## Use Cases

### 1. Multi-Tenant Analysis Platform

```python
class MultiTenantExcelAnalyzer:
    def __init__(self, e2b_api_key):
        self.api_key = e2b_api_key
        self.tenant_sandboxes = {}
    
    async def analyze_for_tenant(self, tenant_id, excel_file):
        # Ensure isolation between tenants
        sandbox_key = f"tenant_{tenant_id}"
        
        if sandbox_key not in self.tenant_sandboxes:
            self.tenant_sandboxes[sandbox_key] = CodeInterpreter(
                api_key=self.api_key,
                metadata={"tenant_id": tenant_id}
            )
        
        sandbox = self.tenant_sandboxes[sandbox_key]
        
        # Tenant-specific analysis
        with open(excel_file, 'rb') as f:
            sandbox.upload_file(f, f"/tmp/{tenant_id}_data.xlsx")
        
        result = sandbox.notebook.exec_cell(f"""
        import pandas as pd
        df = pd.read_excel('/tmp/{tenant_id}_data.xlsx')
        # Analysis specific to tenant...
        """)
        
        return result
```

### 2. Untrusted Code Execution

```python
async def execute_user_provided_code(user_code, timeout=30):
    """Safely execute user-provided analysis code"""
    
    # Wrap in safety checks
    wrapped_code = f"""
import signal
import sys

def timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

try:
    {user_code}
finally:
    signal.alarm(0)
"""
    
    with CodeInterpreter() as sandbox:
        result = sandbox.notebook.exec_cell(wrapped_code)
        return result
```

## Future Developments

### Expected Features (2024-2025)

1. **GPU Support**: For ML workloads
1. **Persistent Sandboxes**: Long-running environments
1. **Custom Networking**: VPC integration
1. **Advanced Templates**: More pre-built environments
1. **WebSocket Support**: Real-time communication

## Conclusion

E2B provides a robust cloud-based code execution platform that complements local solutions like notebook_cli.py. Key advantages:

**Use E2B when:**

- Security is critical (untrusted code)
- Multi-tenant isolation is required
- Scaling beyond local resources
- Standardized environments needed

**Use notebook_cli.py when:**

- Low latency is critical
- Custom package requirements
- Direct file system access needed
- Cost sensitivity

The ideal architecture often combines both:

- E2B for untrusted/experimental code
- Local execution for trusted, performance-critical paths

## References

1. [E2B Documentation](https://e2b.dev/docs)
1. [E2B Python SDK](https://github.com/e2b-dev/e2b-python)
1. [Code Interpreter SDK](https://e2b.dev/docs/code-interpreter/overview)
1. [API Reference](https://e2b.dev/docs/api-reference)
1. [Pricing](https://e2b.dev/pricing)
1. [Security Model](https://e2b.dev/docs/security)
