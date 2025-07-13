# Prompt Templates for Spreadsheet Analysis

## Table of Contents

1. [Basic Excel Formula Analysis](#basic-excel-formula-analysis)
1. [Advanced Formula Understanding](#advanced-formula-understanding)
1. [Data Pattern Recognition](#data-pattern-recognition)
1. [Error Detection and Debugging](#error-detection-and-debugging)
1. [Data Transformation](#data-transformation)
1. [Structured Output Templates](#structured-output-templates)
1. [Anti-Hallucination Templates](#anti-hallucination-templates)
1. [Chain-of-Thought Templates](#chain-of-thought-templates)

## Basic Excel Formula Analysis

### Formula Explanation Template

```
<task>
Analyze the following Excel formula and explain its purpose, components, and expected output.
</task>

<formula>
{formula_here}
</formula>

<context>
This formula is used in cell {cell_reference} of a spreadsheet that tracks {spreadsheet_purpose}.
</context>

<instructions>
1. Break down each function used in the formula
2. Explain the purpose of each argument
3. Describe what the formula calculates
4. Identify any potential issues or edge cases
5. Suggest improvements if applicable
</instructions>

<output_format>
Structure your response with the following sections:
- Formula Overview
- Component Breakdown
- Calculation Logic
- Expected Results
- Potential Issues
- Improvement Suggestions
</output_format>
```

### Formula Generation Template

```
<role>
You are an Excel formula expert specializing in creating efficient and accurate formulas for business data analysis.
</role>

<task>
Create an Excel formula that accomplishes the following:
{task_description}
</task>

<data_context>
- Data range: {data_range}
- Column headers: {headers}
- Data types: {data_types}
- Special considerations: {considerations}
</data_context>

<requirements>
- The formula must be compatible with {excel_version}
- Avoid volatile functions if possible
- Include error handling
- Make the formula maintainable and readable
</requirements>

<output>
Provide:
1. The complete formula
2. Step-by-step explanation
3. Example with sample data
4. Alternative approaches if applicable
</output>
```

## Advanced Formula Understanding

### Complex Formula Decomposition Template

```
<system>
You are analyzing complex Excel formulas. Think step by step through each component.
</system>

<formula>
{complex_formula}
</formula>

<analysis_framework>
1. Identify the main function and its purpose
2. List all nested functions in order of execution
3. Trace the data flow through each function
4. Identify dependencies and references
5. Evaluate the formula with example values
</analysis_framework>

<think_step_by_step>
Let's work through this formula systematically:

Step 1: Main function identification
Step 2: Nested function analysis
Step 3: Parameter evaluation
Step 4: Execution order
Step 5: Result calculation
</think_step_by_step>

<output_requirements>
- Use clear, non-technical language where possible
- Include a visual representation of the formula structure
- Provide concrete examples
- Highlight any unusual or advanced techniques
</output_requirements>
```

### Array Formula Analysis Template

```
<context>
Analyzing an array formula that processes multiple cells simultaneously.
</context>

<formula>
{array_formula}
</formula>

<analysis_approach>
1. Identify the array dimensions
2. Explain the array operation
3. Show how each element is processed
4. Demonstrate the output array structure
</analysis_approach>

<example_data>
Input array:
{input_array}

Expected output:
{output_structure}
</example_data>

<special_considerations>
- Dynamic array behavior (if Excel 365)
- CSE (Ctrl+Shift+Enter) requirements
- Performance implications
- Compatibility notes
</special_considerations>
```

## Data Pattern Recognition

### Pattern Detection Template

```
<role>
You are a data analyst specializing in identifying patterns and anomalies in spreadsheet data.
</role>

<data_sample>
{data_sample}
</data_sample>

<analysis_tasks>
1. Identify recurring patterns in the data
2. Detect anomalies or outliers
3. Recognize data types and formats
4. Find relationships between columns
5. Identify missing or inconsistent data
</analysis_tasks>

<pattern_categories>
- Temporal patterns (dates, time series)
- Numerical patterns (sequences, distributions)
- Categorical patterns (groupings, hierarchies)
- Relational patterns (correlations, dependencies)
</pattern_categories>

<output_format>
## Pattern Analysis Report

### Data Overview
- Total records:
- Column count:
- Data quality score:

### Identified Patterns
1. [Pattern Type]: [Description]
   - Evidence: [Supporting data]
   - Confidence: [High/Medium/Low]

### Anomalies Detected
- [Anomaly description and location]

### Recommendations
- [Data cleaning suggestions]
- [Analysis opportunities]
</output_format>
```

## Error Detection and Debugging

### Formula Error Diagnosis Template

```
<problem>
The following formula is producing an error:
{formula_with_error}

Error type: {error_type}
Cell reference: {cell_ref}
</problem>

<diagnostic_steps>
1. Check for syntax errors
2. Verify all cell references exist
3. Confirm data types match expected inputs
4. Test each function component separately
5. Check for circular references
6. Verify named ranges are defined
</diagnostic_steps>

<context_needed>
- Sample data from referenced cells
- Named range definitions
- Excel version being used
- Any recent changes to the spreadsheet
</context_needed>

<solution_format>
## Error Diagnosis

### Root Cause
[Specific cause of the error]

### Solution
[Step-by-step fix]

### Prevention
[How to avoid this error in the future]

### Alternative Approach
[Different formula that achieves the same result]
</solution_format>
```

### Data Validation Template

```
<task>
Validate the data in the following spreadsheet range for consistency and accuracy.
</task>

<data_rules>
{validation_rules}
</data_rules>

<validation_checks>
1. Data type consistency
2. Range boundaries
3. Required fields
4. Format compliance
5. Logical relationships
6. Duplicate detection
</validation_checks>

<report_format>
## Validation Report

### Summary
- Records checked:
- Errors found:
- Warnings:

### Critical Errors
| Row | Column | Error Type | Value | Expected |
|-----|--------|-----------|-------|----------|

### Warnings
| Row | Column | Warning | Suggestion |
|-----|--------|---------|------------|

### Data Quality Score
[Score]/100

### Remediation Steps
1. [Specific fix for each error type]
</report_format>
```

## Data Transformation

### Data Cleaning Template

````
<task>
Clean and standardize the following dataset according to best practices.
</task>

<messy_data>
{data_sample}
</messy_data>

<cleaning_requirements>
1. Standardize date formats to {date_format}
2. Remove duplicate entries
3. Handle missing values according to {missing_value_strategy}
4. Normalize text fields (trim, proper case)
5. Validate numerical ranges
6. Fix common data entry errors
</cleaning_requirements>

<transformation_steps>
For each issue found:
1. Identify the problem
2. Propose the fix
3. Show before/after example
4. Provide the Excel formula or method
</transformation_steps>

<output>
## Data Cleaning Plan

### Issues Identified
1. [Issue]: [Count] occurrences
   - Example: [Sample]
   - Fix: [Method/Formula]

### Cleaning Formulas
```excel
' For issue 1:
=FORMULA_HERE

' For issue 2:
=FORMULA_HERE
````

### Validation Checks

[Formulas to verify data is clean]
</output>

```

### Pivot Table Design Template
```

<objective>
Design a pivot table to analyze {analysis_goal} from the following data structure.
</objective>

\<data_structure>
Columns: {column_list}
Row count: {row_count}
Key metrics: {metrics}
\</data_structure>

\<analysis_requirements>

1. Primary grouping by {dimension1}
1. Secondary grouping by {dimension2}
1. Calculate {aggregations}
1. Include {filters}
1. Sort by {sort_criteria}
   \</analysis_requirements>

\<pivot_design>

## Pivot Table Configuration

### Row Fields

- {field1} (grouped by {grouping})
- {field2}

### Column Fields

- {field3}

### Values

- {metric1}: {aggregation_type}
- {metric2}: {aggregation_type}

### Filters

- {filter_field}: {filter_criteria}

### Calculated Fields

```
Field Name: {name}
Formula: {formula}
Purpose: {purpose}
```

### Formatting Recommendations

- Number format: {format}
- Conditional formatting rules: {rules}
- Chart type: {chart_recommendation}
  \</pivot_design>

```

## Structured Output Templates

### JSON Output Template for Formula Analysis
```

<instructions>
Analyze the Excel formula and return the results in the following JSON structure:
</instructions>

<formula>
{formula}
</formula>

\<json_schema>
{
"formula_analysis": {
"original_formula": "string",
"formula_type": "string",
"complexity_score": "number (1-10)",
"functions_used": \[
{
"function_name": "string",
"purpose": "string",
"arguments": ["string"]
}
\],
"cell_references": {
"absolute": ["string"],
"relative": ["string"],
"named_ranges": ["string"]
},
"dependencies": {
"upstream_cells": ["string"],
"downstream_impact": ["string"]
},
"performance_considerations": {
"is_volatile": "boolean",
"calculation_time": "string",
"optimization_suggestions": ["string"]
},
"potential_errors": \[
{
"error_type": "string",
"likelihood": "string",
"prevention": "string"
}
\]
}
}
\</json_schema>

<requirements>
- All fields must be populated
- Use null for non-applicable fields
- Ensure valid JSON syntax
- Include explanatory comments where helpful
</requirements>
```

### XML Template for Data Structure

```
<task>
Convert the spreadsheet data structure into XML format for integration purposes.
</task>

<data_sample>
{sample_data}
</data_sample>

<xml_structure>
<?xml version="1.0" encoding="UTF-8"?>
<spreadsheet_data>
  <metadata>
    <source_file>{filename}</source_file>
    <sheet_name>{sheetname}</sheet_name>
    <row_count>{count}</row_count>
    <column_count>{count}</column_count>
    <last_modified>{timestamp}</last_modified>
  </metadata>

  <column_definitions>
    <column id="{id}" name="{name}" type="{type}" nullable="{boolean}"/>
  </column_definitions>

  <data_rows>
    <row id="{row_number}">
      <cell column="{column_id}" value="{value}" formula="{formula_if_any}"/>
    </row>
  </data_rows>

  <data_validation_rules>
    <rule column="{column_id}" type="{validation_type}" parameters="{params}"/>
  </data_validation_rules>
</spreadsheet_data>
</xml_structure>
```

## Anti-Hallucination Templates

### Fact-Grounded Analysis Template

```
<instructions>
Analyze this Excel data based ONLY on the information provided. Do not make assumptions or add information not present in the data.
</instructions>

<data>
{actual_data}
</data>

<analysis_constraints>
1. Only reference data points that exist in the provided sample
2. If information is missing, explicitly state "Data not provided"
3. Use phrases like "Based on the available data..." or "The provided sample shows..."
4. Avoid generalizations beyond the scope of the data
5. Clearly distinguish between facts and interpretations
</analysis_constraints>

<required_sections>
## Factual Observations
[Only what can be directly observed in the data]

## Data Limitations
[What information is missing or unclear]

## Supported Conclusions
[Only conclusions directly supported by the data]
According to the provided data...

## Additional Data Needed
[What additional information would be helpful]
</required_sections>
```

### Verification Template

```
<task>
Verify the accuracy of the following Excel formula interpretation.
</task>

<formula>
{formula}
</formula>

<claimed_interpretation>
{interpretation}
</claimed_interpretation>

<verification_steps>
1. Parse formula syntax
2. Verify each function's documented behavior
3. Check argument validity
4. Test with example values
5. Compare results with claimed interpretation
</verification_steps>

<response_format>
## Verification Report

### Formula Components
According to Excel documentation:
- Function X: [Official description]
- Function Y: [Official description]

### Step-by-Step Execution
1. [What happens first]
2. [What happens next]
[Show actual calculation]

### Verification Result
☐ Interpretation is correct
☐ Interpretation has errors: [List specific errors]
☐ Interpretation is partially correct: [Explain]

### Evidence
[Cite specific Excel documentation or test results]
</response_format>
```

## Chain-of-Thought Templates

### Complex Problem Solving Template

```
<problem>
{complex_spreadsheet_problem}
</problem>

<thinking_framework>
Let's think step by step:

1. Understanding the Problem
   - What is being asked?
   - What data do we have?
   - What is the desired outcome?

2. Breaking Down the Components
   - Component A: [Description]
   - Component B: [Description]
   - How do they relate?

3. Identifying the Approach
   - Possible method 1: [Description]
   - Possible method 2: [Description]
   - Best approach because: [Reasoning]

4. Implementation Steps
   - Step 1: [Specific action]
   - Step 2: [Specific action]
   - Step 3: [Specific action]

5. Validation
   - How to verify the solution works
   - Edge cases to consider
   - Expected results
</thinking_framework>

<solution>
Based on the step-by-step analysis:

### Recommended Solution
[Complete solution with formulas]

### Why This Works
[Explanation referencing the thinking steps]

### Alternative Approaches
[Other valid solutions considered]
</solution>
```

### Multi-Step Formula Creation Template

```
<objective>
Create a formula to {complex_calculation_need}
</objective>

<available_data>
{data_description}
</available_data>

<step_by_step_approach>
Let's build this formula step by step:

Step 1: Identify the Core Calculation
- What is the main operation needed?
- Formula component: {component1}

Step 2: Add Data Validation
- What errors could occur?
- Error handling: {error_handling}

Step 3: Incorporate Edge Cases
- Special scenarios: {scenarios}
- Adjustments needed: {adjustments}

Step 4: Optimize for Performance
- Current approach efficiency: {analysis}
- Optimization: {optimization}

Step 5: Combine All Components
- Integration method: {method}
- Final formula structure: {structure}
</step_by_step_approach>

<final_formula>
=COMPLETE_FORMULA_HERE

### Component Breakdown
- Part 1: {explanation}
- Part 2: {explanation}
- Part 3: {explanation}

### Test Cases
| Input | Expected Output | Formula Result |
|-------|----------------|----------------|
| {test1} | {expected1} | {result1} |
| {test2} | {expected2} | {result2} |
</final_formula>
```

## Usage Guidelines

### When to Use Each Template

1. **Basic Analysis**: For simple formulas and straightforward tasks
1. **Advanced Analysis**: For complex, nested formulas or array operations
1. **Pattern Recognition**: When exploring data for insights
1. **Error Detection**: For debugging and troubleshooting
1. **Data Transformation**: For cleaning and restructuring data
1. **Structured Output**: When integration with other systems is needed
1. **Anti-Hallucination**: For high-stakes analysis requiring accuracy
1. **Chain-of-Thought**: For complex problems requiring reasoning

### Customization Tips

1. Adjust the level of detail based on user expertise
1. Include relevant Excel version-specific features
1. Add domain-specific context when applicable
1. Modify output formats to match requirements
1. Incorporate few-shot examples for better results

### Best Practices

1. Always include clear instructions
1. Provide sufficient context
1. Specify the desired output format
1. Use appropriate technical level
1. Include validation steps
1. Request explanations for transparency
1. Build in error handling
1. Test templates with various inputs
