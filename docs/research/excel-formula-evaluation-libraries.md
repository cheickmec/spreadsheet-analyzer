# Comprehensive Research: Excel Formula Evaluation Libraries

*Research Date: July 30, 2025*
*For: Spreadsheet Analyzer Project*

## Executive Summary

This comprehensive research examines Python libraries capable of parsing and evaluating Excel formulas without requiring Excel itself. The analysis reveals a diverse ecosystem with xlcalculator emerging as the most modern solution, building upon koala2's proven foundation. Key findings include the critical limitation of openpyxl (no formula evaluation), the maturity of graph-based approaches, and the trade-offs between performance, feature completeness, and maintenance activity across different libraries.

## Table of Contents

1. [Overview](#overview)
1. [xlcalculator](#xlcalculator)
1. [koala2](#koala2)
1. [formulas](#formulas)
1. [pycel](#pycel)
1. [Comparison Matrix](#comparison-matrix)
1. [Recommendations](#recommendations)

## Overview

### Current Limitation

Our spreadsheet analyzer currently uses openpyxl, which **does not evaluate formulas**. It only reads the last calculated value stored in the Excel file. This limits our ability to:

- Understand formula dependencies dynamically
- Recalculate values when inputs change
- Fully analyze spreadsheet logic without Excel

### Key Requirements

For our spreadsheet analyzer, we need a library that can:

1. Parse Excel formulas accurately
1. Build dependency graphs
1. Evaluate formulas without Excel
1. Handle complex formulas (VLOOKUP, INDEX/MATCH, etc.)
1. Support cross-sheet references
1. Scale to large workbooks (thousands of formulas)

## xlcalculator

### Overview

xlcalculator is a modernization of the koala2 library that converts MS Excel formulas to Python and evaluates them.

### Key Features

- **AST-based evaluation**: Builds proper AST from formula nodes, each supporting an eval() function
- **Extensible function support**: Excel functions can be easily added via xlfunctions.xl.register() decorator
- **Model extraction**: Can focus on specific cell addresses or defined names
- **Native Excel data types**: Supports Excel's data type system
- **No code generation**: Unlike koala2, doesn't generate Python code

### Installation

```bash
pip install xlcalculator
```

### Architecture

```
Formula → Parser → AST → Evaluation
                     ↓
              Function Registry
```

### Pros

- Modern architecture with proper AST
- Active development (modernization of koala2)
- Extensible function system
- Good performance for formula networks

### Cons

- Relatively newer, less battle-tested than koala2
- Documentation could be more comprehensive
- Limited examples for complex scenarios

### Example Usage

```python
from xlcalculator import ModelCompiler
from xlcalculator import Model
from xlcalculator import Evaluator

filename = "example.xlsx"
compiler = ModelCompiler()
new_model = compiler.read_and_parse_archive(filename)
evaluator = Evaluator(new_model)

# Set inputs
evaluator.set_cell_value("Sheet1!A1", 10)

# Evaluate
result = evaluator.evaluate("Sheet1!D1")
```

## koala2

### Overview

koala2 transposes Excel calculations into Python for better performance and scaling. It's been used on workbooks with 100,000+ formulas.

### Key Features

- **Graph-based approach**: Converts workbook into a dependency graph
- **Lazy evaluation**: Only evaluates when parent cells are modified
- **Graph persistence**: Can dump/load graphs to avoid recompilation
- **Formula modification**: Can change formulas at runtime
- **Named range support**: Full support for Excel named ranges

### Installation

```bash
pip install koala2
```

### Architecture

```
Excel File → Graph Builder → Dependency Graph → Evaluator
                                    ↓
                            Serialization (gzip)
```

### Pros

- Battle-tested on large workbooks (100k+ formulas)
- Graph serialization for performance
- Can create spreadsheets from scratch
- Supports formula modification

### Cons

- Early development stage (per documentation)
- Graph conversion can be time-consuming
- Less modern architecture than newer alternatives

### Example Usage

```python
from koala.ExcelCompiler import ExcelCompiler
from koala.Spreadsheet import Spreadsheet

# Load and compile
sp = Spreadsheet("example.xlsx")

# Optimize graph for specific inputs/outputs
sp = sp.gen_graph(inputs=['Sheet1!A1'], outputs=['Sheet1!D1'])

# Set values and evaluate
sp.cell_set_value('Sheet1!A1', 10)
result = sp.cell_evaluate('Sheet1!D1')

# Save compiled graph
sp.dump('compiled_graph.gzip')
```

## formulas

### Overview

formulas implements an interpreter for Excel formulas, parsing and compiling Excel formula expressions. It can compile entire workbooks to Python without needing Excel.

### Key Features

- **Full formula parsing**: Complete Excel formula parser with AST generation
- **Workbook compilation**: Compiles Excel workbooks to Python
- **Circular reference support**: Handles circular references when enabled
- **Visualization**: Can plot formula AST and dependency graphs
- **JSON export/import**: Readable format for version control

### Installation

```bash
pip install formulas[all]  # Includes excel and plot extras
```

### Architecture

```
Formula String → Parser → AST → Compiler → Python Function
                           ↓
                    Visualization (plot)
                           ↓
                    JSON Export/Import
```

### Pros

- Comprehensive formula parser
- Excellent visualization capabilities
- JSON export for version control
- Handles circular references
- No Excel dependency

### Cons

- Not all Excel formulas implemented
- May be slower than compiled approaches
- Complex API for simple use cases

### Example Usage

```python
import formulas

# Parse single formula
func = formulas.Parser().ast('=(1 + 1) + B3 / A2')[1].compile()
result = func(A2=1, B3=5)  # Returns 7.0

# Compile workbook
xl_model = formulas.ExcelModel().loads("example.xlsx").finish()
xl_model.calculate(inputs={"Sheet1!A1": 10})
output = xl_model.dsp.value("Sheet1!D1")

# Export to JSON
xl_model.save("model.json")
```

## pycel

### Overview

pycel compiles Excel spreadsheets to Python code and visualizes them as graphs. It's been tested on spreadsheets with 10+ sheets and 10,000+ formulas.

### Key Features

- **Graph visualization**: Visualizes spreadsheet dependencies
- **Fast execution**: Graph-based code is optimized for speed
- **Extensive function support**: Mathematical functions, ranges, MIN/MAX, INDEX, LOOKUP, LINEST
- **Multiple serialization formats**: pkl, yaml, json
- **PyXLL integration**: Can run as Excel add-in

### Installation

```bash
pip install pycel
```

### Architecture

```
Excel → COM/Direct Parse → Graph Compiler → Python Code
                                    ↓
                            Dependency Graph
                                    ↓
                            Serialization
```

### Pros

- Proven on large spreadsheets (10k+ formulas)
- Good performance (50ms for 10k formulas)
- Small, understandable codebase
- Multiple serialization options

### Cons

- Development driven by specific needs
- Limited function coverage
- No VBA support
- Cell reference limitations with OFFSET

### Example Usage

```python
from pycel import ExcelCompiler

# Compile spreadsheet
compiler = ExcelCompiler(filename="example.xlsx")

# Set inputs
compiler.set_value('Sheet1!A1', 10)

# Evaluate
result = compiler.evaluate('Sheet1!D1')

# Save compiled model
compiler.save("compiled_model.pkl")
```

## Comparison Matrix

| Feature                 | xlcalculator | koala2 | formulas | pycel   | openpyxl  |
| ----------------------- | ------------ | ------ | -------- | ------- | --------- |
| **Formula Evaluation**  | ✅           | ✅     | ✅       | ✅      | ❌        |
| **AST Generation**      | ✅           | ❌     | ✅       | ❌      | ❌        |
| **Dependency Graph**    | ✅           | ✅     | ✅       | ✅      | ❌        |
| **Large Scale (100k+)** | ✅           | ✅     | ❓       | ✅      | N/A       |
| **Visualization**       | ❌           | ❌     | ✅       | ✅      | ❌        |
| **JSON Export**         | ❌           | ❌     | ✅       | ✅      | ❌        |
| **Circular Refs**       | ❓           | ❓     | ✅       | ❓      | N/A       |
| **Function Coverage**   | Good         | Good   | Partial  | Partial | N/A       |
| **Active Development**  | ✅           | ❓     | ✅       | ❓      | ✅        |
| **Documentation**       | Fair         | Fair   | Good     | Fair    | Excellent |
| **License**             | MIT          | GPL    | EUPL     | GPL     | MIT       |

## Performance Considerations

Based on available benchmarks:

- **pycel**: 50ms for 10,000 formulas
- **koala2**: Proven on 100,000+ formulas
- **xlcalculator**: Performance testing shows competitive results
- **formulas**: May be slower due to interpretation overhead

## Integration Considerations

### With Our Current Stack

1. **Dependency Graph Enhancement**: All libraries provide dependency graphs that could enhance our current analysis
1. **Formula Evaluation**: Would add capability to recalculate values dynamically
1. **Validation**: Could validate our static formula parsing against dynamic evaluation

### Architecture Impact

```
Current: Excel → openpyxl → Static Analysis → Graph
Proposed: Excel → openpyxl → Static Analysis → Graph
                      ↓
              Formula Evaluator (xlcalculator/koala2/etc.)
                      ↓
              Dynamic Analysis
```

## Recommendations

### Primary Recommendation: xlcalculator

**Reasons:**

1. Modern architecture with proper AST
1. Active development and modernization of proven koala2
1. MIT license (compatible with our project)
1. Extensible function system
1. Good balance of features and performance

### Secondary Option: formulas

**Use if:**

- Visualization is critical
- JSON export/import needed
- Circular reference handling required
- Full AST manipulation needed

### For Specific Use Cases:

- **Large-scale proven solution**: koala2
- **Speed critical with visualization**: pycel
- **Maximum flexibility**: formulas

### Implementation Strategy

1. **Phase 1**: Integrate xlcalculator for formula evaluation
1. **Phase 2**: Enhance dependency graph with evaluation results
1. **Phase 3**: Add dynamic analysis features
1. **Phase 4**: Consider formulas for visualization if needed

## Conclusion

While openpyxl serves well for reading Excel files, adding a formula evaluation library would significantly enhance our spreadsheet analyzer's capabilities. xlcalculator provides the best balance of modern architecture, performance, and compatibility for our needs. The ability to evaluate formulas would enable:

- Dynamic dependency validation
- What-if analysis
- Formula correctness checking
- Complete spreadsheet logic understanding

The integration effort is justified by the substantial increase in analysis capabilities.
