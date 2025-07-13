# Code Examples for Prompt Engineering in Spreadsheet Analysis

## Table of Contents

1. [Basic Implementation Examples](#basic-implementation-examples)
1. [Chain-of-Thought Implementation](#chain-of-thought-implementation)
1. [Structured Output Generation](#structured-output-generation)
1. [Anti-Hallucination Implementations](#anti-hallucination-implementations)
1. [Excel-Specific Integrations](#excel-specific-integrations)
1. [Advanced Techniques](#advanced-techniques)
1. [Testing and Validation](#testing-and-validation)

## Basic Implementation Examples

### Simple Formula Analysis with OpenAI

```python
import openai
from typing import Dict, Any
import json

class ExcelFormulaAnalyzer:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    def analyze_formula(self, formula: str, context: str = "") -> Dict[str, Any]:
        """Analyze an Excel formula and return structured insights."""

        prompt = f"""
        <task>
        Analyze the following Excel formula and provide a detailed explanation.
        </task>

        <formula>
        {formula}
        </formula>

        <context>
        {context if context else "No additional context provided."}
        </context>

        <output_requirements>
        Provide your analysis in the following structure:
        1. Formula purpose
        2. Components breakdown
        3. Expected output type
        4. Potential issues
        5. Optimization suggestions
        </output_requirements>
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an Excel formula expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2  # Lower temperature for more consistent analysis
        )

        return self._parse_response(response.choices[0].message.content)

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the response into structured format."""
        # Implementation would parse the text response into structured data
        sections = {
            "purpose": "",
            "components": [],
            "output_type": "",
            "issues": [],
            "optimizations": []
        }
        # Parsing logic here
        return sections

# Usage example
analyzer = ExcelFormulaAnalyzer(api_key="your-api-key")
result = analyzer.analyze_formula(
    "=VLOOKUP(A2,Sheet2!$A$1:$D$100,4,FALSE)",
    context="Looking up product prices from a master price list"
)
```

### Claude Implementation with XML Tags

```python
import anthropic
from typing import List, Dict
import xml.etree.ElementTree as ET

class ClaudeSpreadsheetAnalyzer:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def analyze_data_patterns(self, data_sample: List[List[Any]]) -> Dict[str, Any]:
        """Analyze patterns in spreadsheet data using Claude."""

        # Convert data to string representation
        data_str = "\n".join(["\t".join(map(str, row)) for row in data_sample])

        prompt = f"""
        <role>
        You are a data analyst specializing in spreadsheet pattern recognition.
        </role>

        <data>
        {data_str}
        </data>

        <analysis_tasks>
        <task>Identify data types for each column</task>
        <task>Find patterns and regularities</task>
        <task>Detect anomalies</task>
        <task>Suggest data quality improvements</task>
        </analysis_tasks>

        <output_format>
        <patterns>
            <pattern type="[type]" confidence="[high/medium/low]">
                <description>[Pattern description]</description>
                <evidence>[Supporting evidence]</evidence>
            </pattern>
        </patterns>
        <anomalies>
            <anomaly row="[row]" column="[column]">
                <value>[anomalous value]</value>
                <reason>[Why it's anomalous]</reason>
            </anomaly>
        </anomalies>
        </output_format>

        Please provide your analysis in the XML format specified above.
        """

        message = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            temperature=0,
            system="You are an expert data analyst. Always structure your responses using the provided XML tags.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return self._parse_xml_response(message.content[0].text)

    def _parse_xml_response(self, xml_text: str) -> Dict[str, Any]:
        """Parse XML response from Claude."""
        try:
            # Extract XML content
            start = xml_text.find('<patterns>')
            end = xml_text.rfind('</anomalies>') + len('</anomalies>')
            xml_content = xml_text[start:end]

            # Wrap in root element for parsing
            xml_content = f"<analysis>{xml_content}</analysis>"

            root = ET.fromstring(xml_content)

            patterns = []
            for pattern in root.findall('.//pattern'):
                patterns.append({
                    'type': pattern.get('type'),
                    'confidence': pattern.get('confidence'),
                    'description': pattern.find('description').text,
                    'evidence': pattern.find('evidence').text
                })

            anomalies = []
            for anomaly in root.findall('.//anomaly'):
                anomalies.append({
                    'row': anomaly.get('row'),
                    'column': anomaly.get('column'),
                    'value': anomaly.find('value').text,
                    'reason': anomaly.find('reason').text
                })

            return {
                'patterns': patterns,
                'anomalies': anomalies
            }
        except Exception as e:
            return {'error': f'Failed to parse XML: {str(e)}'}
```

## Chain-of-Thought Implementation

### Zero-Shot CoT for Complex Calculations

```python
class ChainOfThoughtAnalyzer:
    def __init__(self, llm_client):
        self.client = llm_client

    def analyze_complex_formula(self, formula: str) -> Dict[str, Any]:
        """Use zero-shot chain-of-thought for complex formula analysis."""

        prompt = f"""
        Analyze this complex Excel formula: {formula}

        Let's think step by step:

        1. First, identify the outermost function
        2. Then, work through each nested function from inside out
        3. Trace the data flow through each component
        4. Calculate an example with sample values
        5. Identify potential edge cases

        Show your complete reasoning process.
        """

        response = self.client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=2000
        )

        return self._extract_steps(response)

    def multi_step_data_transformation(self,
                                     source_data: str,
                                     target_format: str) -> List[str]:
        """Generate multi-step data transformation plan."""

        prompt = f"""
        I need to transform data from this format:
        {source_data}

        To this format:
        {target_format}

        Let's think step by step:

        Step 1: Analyze the source data structure
        - What columns do we have?
        - What are the data types?
        - Are there any special patterns?

        Step 2: Understand the target requirements
        - What's the desired structure?
        - What transformations are needed?
        - Are there any constraints?

        Step 3: Plan the transformation steps
        - What Excel functions will we use?
        - In what order should we apply them?
        - How do we handle edge cases?

        Step 4: Generate the formulas
        - Create each formula with explanations
        - Show intermediate results
        - Validate the output

        Please work through each step systematically.
        """

        response = self.client.generate(prompt=prompt)
        return self._extract_transformation_steps(response)
```

### Few-Shot CoT Implementation

```python
class FewShotCoTAnalyzer:
    def __init__(self, llm_client):
        self.client = llm_client

    def analyze_with_examples(self, formula: str) -> str:
        """Use few-shot CoT with examples."""

        prompt = f"""
        Analyze Excel formulas by breaking them down step-by-step.

        Example 1:
        Formula: =IF(A1>10,B1*2,B1)

        Step-by-step analysis:
        1. This is an IF function with three arguments
        2. Condition: A1>10 checks if cell A1 is greater than 10
        3. If TRUE: Returns B1*2 (doubles the value in B1)
        4. If FALSE: Returns B1 (unchanged)
        5. Purpose: Conditionally doubles a value based on another cell

        Example 2:
        Formula: =SUMIF(A:A,">0",B:B)

        Step-by-step analysis:
        1. SUMIF function sums values based on criteria
        2. Range: Column A (A:A) is checked for criteria
        3. Criteria: ">0" means positive values only
        4. Sum_range: Column B (B:B) values are summed
        5. Purpose: Sum B column values where corresponding A value is positive

        Now analyze this formula:
        Formula: {formula}

        Step-by-step analysis:
        """

        return self.client.generate(prompt=prompt, temperature=0.2)
```

## Structured Output Generation

### JSON Schema Enforcement

```python
import json
from typing import Optional
from pydantic import BaseModel, Field, validator

class FormulaAnalysis(BaseModel):
    """Structured model for formula analysis results."""

    formula: str
    complexity_score: int = Field(ge=1, le=10)
    functions_used: List[Dict[str, str]]
    cell_references: Dict[str, List[str]]
    is_volatile: bool
    optimization_suggestions: Optional[List[str]] = None

    @validator('functions_used')
    def validate_functions(cls, v):
        for func in v:
            if 'name' not in func or 'purpose' not in func:
                raise ValueError('Each function must have name and purpose')
        return v

class StructuredOutputGenerator:
    def __init__(self, llm_client):
        self.client = llm_client

    def analyze_formula_structured(self, formula: str) -> FormulaAnalysis:
        """Generate structured analysis with schema validation."""

        schema = FormulaAnalysis.schema_json(indent=2)

        prompt = f"""
        Analyze this Excel formula and return the results in JSON format.

        Formula: {formula}

        Required JSON schema:
        {schema}

        Ensure your response is valid JSON matching this schema exactly.
        Include all required fields and follow the type constraints.
        """

        response = self.client.generate(
            prompt=prompt,
            temperature=0,
            response_format={"type": "json_object"}  # If supported by API
        )

        try:
            # Parse and validate response
            data = json.loads(response)
            return FormulaAnalysis(**data)
        except Exception as e:
            # Fallback parsing logic
            return self._parse_unstructured(response, formula)

    def generate_xml_report(self, data: List[List[Any]]) -> str:
        """Generate XML report for spreadsheet data."""

        prompt = f"""
        Convert this spreadsheet data to XML format:

        Data:
        {data}

        Use this structure:
        <spreadsheet>
            <metadata>
                <rows>{len(data)}</rows>
                <columns>{len(data[0]) if data else 0}</columns>
            </metadata>
            <headers>
                <header index="0">Column Name</header>
            </headers>
            <data>
                <row id="1">
                    <cell column="0">Value</cell>
                </row>
            </data>
        </spreadsheet>

        Ensure valid XML with proper escaping.
        """

        return self.client.generate(prompt=prompt, temperature=0)
```

### Grammar-Constrained Generation

```python
class GrammarConstrainedGenerator:
    def __init__(self, llm_client):
        self.client = llm_client

    def generate_formula_with_constraints(self,
                                        requirement: str,
                                        allowed_functions: List[str],
                                        max_nesting: int = 3) -> str:
        """Generate formula with specific constraints."""

        grammar_rules = f"""
        Grammar constraints for Excel formula generation:

        1. Formula must start with =
        2. Only use functions from: {', '.join(allowed_functions)}
        3. Maximum nesting depth: {max_nesting}
        4. All cell references must be absolute ($A$1 style)
        5. String literals must be in double quotes
        6. No spaces except after commas

        Valid formula patterns:
        - =FUNCTION(arg1,arg2,...)
        - =FUNCTION1(FUNCTION2(...),arg)
        - =value1+value2
        """

        prompt = f"""
        {grammar_rules}

        Requirement: {requirement}

        Generate a valid Excel formula following ALL the grammar constraints above.
        Explain why your formula follows each rule.
        """

        response = self.client.generate(prompt=prompt, temperature=0.1)

        # Validate the generated formula
        formula = self._extract_formula(response)
        if self._validate_grammar(formula, allowed_functions, max_nesting):
            return formula
        else:
            raise ValueError("Generated formula violates constraints")
```

## Anti-Hallucination Implementations

### RAG-Enhanced Analysis

```python
import numpy as np
from typing import List, Tuple
import pandas as pd

class RAGSpreadsheetAnalyzer:
    def __init__(self, llm_client, knowledge_base):
        self.client = llm_client
        self.kb = knowledge_base  # Vector database with Excel documentation

    def analyze_with_retrieval(self,
                             formula: str,
                             top_k: int = 5) -> Dict[str, Any]:
        """Analyze formula with retrieval-augmented generation."""

        # Retrieve relevant documentation
        relevant_docs = self.kb.search(formula, top_k=top_k)

        # Build context from retrieved documents
        context = self._build_context(relevant_docs)

        prompt = f"""
        Based on the following Excel documentation, analyze this formula:

        Formula: {formula}

        Documentation:
        {context}

        Instructions:
        1. Only use information from the provided documentation
        2. If something is not covered in the docs, say "Not found in documentation"
        3. Quote specific sections when making claims
        4. Distinguish between documented behavior and assumptions

        Analysis:
        """

        response = self.client.generate(prompt=prompt, temperature=0)

        return {
            'analysis': response,
            'sources': [doc['source'] for doc in relevant_docs],
            'confidence': self._calculate_confidence(response, relevant_docs)
        }

    def _build_context(self, docs: List[Dict]) -> str:
        """Build context from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(docs):
            context_parts.append(f"""
            Source {i+1}: {doc['source']}
            Content: {doc['content']}
            Relevance Score: {doc['score']}
            """)
        return "\n".join(context_parts)

    def _calculate_confidence(self, response: str, docs: List[Dict]) -> float:
        """Calculate confidence based on source quality and relevance."""
        # Implementation would calculate confidence score
        return 0.85
```

### Fact-Checking Implementation

```python
class FactCheckingAnalyzer:
    def __init__(self, llm_client):
        self.client = llm_client
        self.excel_facts = self._load_excel_facts()

    def analyze_with_fact_checking(self,
                                 formula: str,
                                 claims: List[str]) -> Dict[str, Any]:
        """Analyze formula and fact-check any claims."""

        prompt = f"""
        Analyze this formula and verify the following claims:

        Formula: {formula}

        Claims to verify:
        {chr(10).join(f'{i+1}. {claim}' for i, claim in enumerate(claims))}

        For each claim:
        1. State whether it's TRUE, FALSE, or UNVERIFIABLE
        2. Provide evidence from the formula
        3. If FALSE, explain the correct behavior
        4. If UNVERIFIABLE, explain what information is missing

        Use only what can be directly observed in the formula.
        Do not make assumptions about cell contents.
        """

        response = self.client.generate(prompt=prompt, temperature=0)

        return self._parse_fact_check_response(response)

    def validate_formula_behavior(self,
                                formula: str,
                                test_cases: List[Dict]) -> List[Dict]:
        """Validate formula behavior with test cases."""

        results = []
        for test in test_cases:
            prompt = f"""
            Given this formula: {formula}

            With these values:
            {chr(10).join(f'{ref}: {val}' for ref, val in test['inputs'].items())}

            According to Excel's documented behavior:
            1. What should the formula return?
            2. Show the step-by-step calculation
            3. Note any potential errors or edge cases

            Do not guess. If you're unsure about Excel's behavior, say so.
            """

            response = self.client.generate(prompt=prompt, temperature=0)

            results.append({
                'test_case': test,
                'analysis': response,
                'expected': test.get('expected'),
                'llm_result': self._extract_result(response)
            })

        return results
```

## Excel-Specific Integrations

### SpreadsheetLLM-Style Implementation

```python
class SpreadsheetCompressor:
    """Implementation inspired by Microsoft's SpreadsheetLLM approach."""

    def __init__(self, llm_client):
        self.client = llm_client

    def compress_spreadsheet(self,
                           data: pd.DataFrame,
                           compression_ratio: float = 0.1) -> Dict[str, Any]:
        """Compress spreadsheet data for LLM processing."""

        # Identify structural anchors
        anchors = self._identify_anchors(data)

        # Detect homogeneous regions
        regions = self._detect_homogeneous_regions(data)

        # Create compressed representation
        compressed = {
            'anchors': anchors,
            'regions': regions,
            'sample_data': self._sample_data(data, compression_ratio),
            'metadata': {
                'original_shape': data.shape,
                'column_types': data.dtypes.to_dict(),
                'compression_ratio': compression_ratio
            }
        }

        return compressed

    def analyze_compressed_sheet(self, compressed_data: Dict) -> str:
        """Analyze compressed spreadsheet representation."""

        prompt = f"""
        Analyze this compressed spreadsheet representation:

        Structural Anchors:
        {compressed_data['anchors']}

        Homogeneous Regions:
        {compressed_data['regions']}

        Sample Data:
        {compressed_data['sample_data']}

        Metadata:
        {compressed_data['metadata']}

        Based on this compressed view:
        1. Identify the spreadsheet's purpose and structure
        2. Detect patterns in the data organization
        3. Suggest formulas for common calculations
        4. Identify potential data quality issues
        """

        return self.client.generate(prompt=prompt)

    def _identify_anchors(self, data: pd.DataFrame) -> List[Dict]:
        """Identify structural anchor points in spreadsheet."""
        anchors = []

        # Headers
        anchors.extend([
            {'type': 'header', 'location': (0, col), 'value': str(data.columns[col])}
            for col in range(len(data.columns))
        ])

        # Detect subtotals, formulas, etc.
        # Implementation would identify key structural elements

        return anchors
```

### Excel Formula Builder

```python
class ExcelFormulaBuilder:
    def __init__(self, llm_client):
        self.client = llm_client
        self.formula_templates = self._load_templates()

    def build_formula_interactive(self,
                                requirement: str,
                                data_context: Dict) -> Dict[str, Any]:
        """Build Excel formula with interactive refinement."""

        # Initial formula generation
        initial_prompt = f"""
        Create an Excel formula for: {requirement}

        Data context:
        - Available columns: {data_context.get('columns', [])}
        - Data types: {data_context.get('types', {})}
        - Row count: {data_context.get('row_count', 'unknown')}

        Generate:
        1. The formula
        2. Explanation of each component
        3. Example with sample data
        4. Potential limitations
        """

        initial_response = self.client.generate(initial_prompt)
        formula = self._extract_formula(initial_response)

        # Validate and refine
        validation_result = self._validate_formula(formula, data_context)

        if not validation_result['valid']:
            refinement_prompt = f"""
            The formula {formula} has issues:
            {validation_result['issues']}

            Please fix these issues while maintaining the original requirement:
            {requirement}
            """

            refined_response = self.client.generate(refinement_prompt)
            formula = self._extract_formula(refined_response)

        # Generate comprehensive documentation
        doc_prompt = f"""
        Document this Excel formula comprehensively:

        Formula: {formula}
        Purpose: {requirement}

        Include:
        1. Complete breakdown of syntax
        2. Step-by-step execution flow
        3. Performance considerations
        4. Common errors and solutions
        5. Alternative approaches
        """

        documentation = self.client.generate(doc_prompt)

        return {
            'formula': formula,
            'documentation': documentation,
            'validation': validation_result,
            'alternatives': self._generate_alternatives(requirement, formula)
        }
```

## Advanced Techniques

### Tree of Thoughts Implementation

```python
class TreeOfThoughtsAnalyzer:
    def __init__(self, llm_client, max_depth: int = 3):
        self.client = llm_client
        self.max_depth = max_depth

    def analyze_complex_problem(self,
                              problem: str,
                              data_context: str) -> Dict[str, Any]:
        """Use Tree of Thoughts for complex spreadsheet problems."""

        # Initialize root thought
        root = {
            'thought': problem,
            'children': [],
            'evaluations': {},
            'depth': 0
        }

        # Explore thought tree
        self._explore_thoughts(root, data_context)

        # Find best path
        best_path = self._find_best_path(root)

        # Generate solution from best path
        solution = self._generate_solution(best_path, data_context)

        return {
            'solution': solution,
            'thought_tree': root,
            'best_path': best_path,
            'alternative_paths': self._get_alternative_paths(root)
        }

    def _explore_thoughts(self, node: Dict, context: str):
        """Recursively explore thought branches."""

        if node['depth'] >= self.max_depth:
            return

        # Generate next thoughts
        prompt = f"""
        Current thought: {node['thought']}
        Context: {context}

        Generate 3 different approaches to solve this:
        1. [Approach 1]
        2. [Approach 2]
        3. [Approach 3]

        For each approach, briefly explain the strategy.
        """

        response = self.client.generate(prompt)
        approaches = self._parse_approaches(response)

        # Evaluate each approach
        for approach in approaches:
            evaluation = self._evaluate_approach(approach, context)

            child = {
                'thought': approach,
                'children': [],
                'evaluations': evaluation,
                'depth': node['depth'] + 1
            }

            node['children'].append(child)

            # Continue exploration if promising
            if evaluation['score'] > 0.7:
                self._explore_thoughts(child, context)

    def _evaluate_approach(self, approach: str, context: str) -> Dict:
        """Evaluate the quality of an approach."""

        prompt = f"""
        Evaluate this approach for solving a spreadsheet problem:

        Approach: {approach}
        Context: {context}

        Rate on:
        1. Feasibility (0-1)
        2. Efficiency (0-1)
        3. Accuracy (0-1)
        4. Maintainability (0-1)

        Provide brief reasoning for each score.
        """

        response = self.client.generate(prompt, temperature=0)
        return self._parse_evaluation(response)
```

### Multi-Expert Prompting

```python
class MultiExpertAnalyzer:
    def __init__(self, llm_client):
        self.client = llm_client
        self.experts = {
            'formula_expert': "You are an Excel formula optimization expert.",
            'data_analyst': "You are a data analysis expert for spreadsheets.",
            'performance_expert': "You are a spreadsheet performance optimization expert.",
            'error_handler': "You are an Excel error handling and debugging expert."
        }

    def analyze_with_experts(self,
                           problem: str,
                           data: Any) -> Dict[str, Any]:
        """Get analysis from multiple expert perspectives."""

        expert_opinions = {}

        for expert_name, expert_role in self.experts.items():
            prompt = f"""
            {expert_role}

            Problem: {problem}
            Data: {data}

            Provide your expert analysis focusing on your specialty.
            Include specific recommendations and potential concerns.
            """

            response = self.client.generate(
                prompt=prompt,
                system=expert_role,
                temperature=0.3
            )

            expert_opinions[expert_name] = response

        # Synthesize expert opinions
        synthesis_prompt = f"""
        Multiple experts have analyzed this spreadsheet problem:

        {self._format_expert_opinions(expert_opinions)}

        Synthesize their insights into:
        1. Consensus recommendations
        2. Points of disagreement
        3. Combined best approach
        4. Risk assessment
        5. Implementation plan
        """

        synthesis = self.client.generate(synthesis_prompt)

        return {
            'expert_opinions': expert_opinions,
            'synthesis': synthesis,
            'consensus_score': self._calculate_consensus(expert_opinions)
        }
```

## Testing and Validation

### Automated Prompt Testing Framework

```python
import unittest
from typing import List, Callable
import time

class PromptTestCase:
    def __init__(self,
                 name: str,
                 input_data: Any,
                 expected_output: Any,
                 validator: Callable):
        self.name = name
        self.input_data = input_data
        self.expected_output = expected_output
        self.validator = validator

class PromptTestingFramework:
    def __init__(self, llm_client):
        self.client = llm_client
        self.test_results = []

    def test_formula_analysis_prompts(self,
                                    test_cases: List[PromptTestCase]) -> Dict:
        """Test formula analysis prompts with various inputs."""

        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }

        for test_case in test_cases:
            start_time = time.time()

            try:
                # Generate analysis
                response = self._run_analysis(test_case.input_data)

                # Validate response
                is_valid = test_case.validator(response, test_case.expected_output)

                execution_time = time.time() - start_time

                result = {
                    'test_name': test_case.name,
                    'passed': is_valid,
                    'execution_time': execution_time,
                    'response': response
                }

                if is_valid:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    result['failure_reason'] = self._analyze_failure(
                        response,
                        test_case.expected_output
                    )

                results['details'].append(result)

            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'test_name': test_case.name,
                    'passed': False,
                    'error': str(e)
                })

        return results

    def benchmark_prompts(self,
                         prompts: List[str],
                         metrics: List[str]) -> pd.DataFrame:
        """Benchmark different prompts on various metrics."""

        results = []

        for prompt in prompts:
            metrics_data = {
                'prompt': prompt[:50] + '...',  # Truncate for display
                'tokens': self._count_tokens(prompt),
            }

            # Test each metric
            for metric in metrics:
                if metric == 'accuracy':
                    score = self._test_accuracy(prompt)
                elif metric == 'consistency':
                    score = self._test_consistency(prompt)
                elif metric == 'speed':
                    score = self._test_speed(prompt)
                else:
                    score = None

                metrics_data[metric] = score

            results.append(metrics_data)

        return pd.DataFrame(results)

    def _test_consistency(self, prompt: str, runs: int = 5) -> float:
        """Test response consistency across multiple runs."""

        responses = []
        for _ in range(runs):
            response = self.client.generate(prompt, temperature=0)
            responses.append(response)

        # Calculate consistency score
        # Implementation would compare responses
        return 0.95  # Placeholder
```

### Validation Utilities

```python
class SpreadsheetPromptValidator:
    @staticmethod
    def validate_formula_syntax(formula: str) -> Dict[str, Any]:
        """Validate Excel formula syntax."""

        issues = []

        # Check basic structure
        if not formula.startswith('='):
            issues.append("Formula must start with =")

        # Check parentheses balance
        if formula.count('(') != formula.count(')'):
            issues.append("Unbalanced parentheses")

        # Check quotes balance
        quote_count = formula.count('"')
        if quote_count % 2 != 0:
            issues.append("Unbalanced quotes")

        # Check for common function names
        import re
        functions = re.findall(r'[A-Z]+(?=\()', formula)
        excel_functions = {'SUM', 'IF', 'VLOOKUP', 'INDEX', 'MATCH', 'COUNT',
                          'AVERAGE', 'MAX', 'MIN', 'CONCATENATE', 'LEFT',
                          'RIGHT', 'MID', 'LEN', 'TRIM', 'ROUND'}

        unknown_functions = set(functions) - excel_functions
        if unknown_functions:
            issues.append(f"Unknown functions: {unknown_functions}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'functions_used': functions
        }

    @staticmethod
    def validate_structured_output(output: str,
                                 expected_format: str) -> Dict[str, Any]:
        """Validate structured output format."""

        if expected_format == 'json':
            try:
                parsed = json.loads(output)
                return {'valid': True, 'parsed': parsed}
            except json.JSONDecodeError as e:
                return {'valid': False, 'error': str(e)}

        elif expected_format == 'xml':
            try:
                root = ET.fromstring(output)
                return {'valid': True, 'root': root}
            except ET.ParseError as e:
                return {'valid': False, 'error': str(e)}

        return {'valid': False, 'error': f'Unknown format: {expected_format}'}
```

## Usage Examples and Best Practices

### Complete Analysis Pipeline

```python
class SpreadsheetAnalysisPipeline:
    def __init__(self, api_key: str):
        self.formula_analyzer = ExcelFormulaAnalyzer(api_key)
        self.pattern_analyzer = ClaudeSpreadsheetAnalyzer(api_key)
        self.cot_analyzer = ChainOfThoughtAnalyzer(openai.OpenAI(api_key=api_key))
        self.validator = SpreadsheetPromptValidator()

    def analyze_spreadsheet(self,
                          file_path: str,
                          analysis_depth: str = 'standard') -> Dict:
        """Complete spreadsheet analysis pipeline."""

        # Load spreadsheet
        df = pd.read_excel(file_path)

        results = {
            'file': file_path,
            'summary': self._generate_summary(df),
            'formulas': [],
            'patterns': {},
            'recommendations': []
        }

        # Extract and analyze formulas
        formulas = self._extract_formulas(df)
        for cell_ref, formula in formulas.items():
            if self.validator.validate_formula_syntax(formula)['valid']:
                if analysis_depth == 'deep':
                    # Use chain-of-thought for complex formulas
                    analysis = self.cot_analyzer.analyze_complex_formula(formula)
                else:
                    # Standard analysis
                    analysis = self.formula_analyzer.analyze_formula(
                        formula,
                        context=f"Cell {cell_ref}"
                    )

                results['formulas'].append({
                    'cell': cell_ref,
                    'formula': formula,
                    'analysis': analysis
                })

        # Analyze data patterns
        if len(df) > 0:
            sample_data = df.head(100).values.tolist()
            results['patterns'] = self.pattern_analyzer.analyze_data_patterns(
                sample_data
            )

        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)

        return results

    def _generate_recommendations(self, analysis_results: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""

        recommendations = []

        # Check for volatile functions
        volatile_count = sum(
            1 for f in analysis_results['formulas']
            if any(vol in f['formula'] for vol in ['INDIRECT', 'OFFSET', 'NOW'])
        )
        if volatile_count > 0:
            recommendations.append(
                f"Found {volatile_count} volatile functions. "
                "Consider replacing with non-volatile alternatives for better performance."
            )

        # Check for patterns
        if 'patterns' in analysis_results and analysis_results['patterns']:
            pattern_count = len(analysis_results['patterns'].get('patterns', []))
            recommendations.append(
                f"Identified {pattern_count} data patterns. "
                "Consider creating summary tables or pivot tables to visualize these patterns."
            )

        return recommendations

# Usage
pipeline = SpreadsheetAnalysisPipeline(api_key="your-api-key")
results = pipeline.analyze_spreadsheet("sales_data.xlsx", analysis_depth="deep")
print(json.dumps(results, indent=2))
```

These code examples demonstrate practical implementations of various prompt engineering techniques specifically tailored for spreadsheet analysis tasks. They can be adapted and extended based on specific requirements and use cases.
