______________________________________________________________________

## name: Performance Issue about: Report slow analysis or high resource usage title: '[PERF] ' labels: ['performance', 'needs-triage'] assignees: ''

# ⚡ Performance Issue

## 📋 Description

<!-- Describe the performance issue you're experiencing -->

## 📊 File Characteristics

- **File Size**: [e.g., 50MB]
- **Number of Sheets**: [e.g., 100]
- **Total Cells with Data**: [approximate]
- **Number of Formulas**: [approximate]
- **Complex Features**:
  - [ ] Pivot Tables
  - [ ] Charts/Graphs
  - [ ] Array Formulas
  - [ ] External References
  - [ ] Macros/VBA
  - [ ] Data Validation Rules
  - [ ] Conditional Formatting

## ⏱️ Performance Metrics

### Observed Performance

- **Total Analysis Time**: [e.g., 5 minutes]
- **Memory Usage**: [e.g., 2GB peak]
- **CPU Usage**: [e.g., 100% single core]

### Expected Performance

Based on the [performance targets](../../docs/design/comprehensive-system-design.md#performance-targets-and-benchmarks):

- File Upload (< 10MB): < 2 seconds
- Basic Analysis (< 10 sheets, < 10K cells): < 5 seconds
- Deep AI Analysis (< 50 sheets, < 100K cells): < 30 seconds

## 🔍 Analysis Details

### Command Used

```bash
# Exact command with all flags
spreadsheet-analyzer analyze file.xlsx --deep --verbose
```

### Analysis Phase

<!-- Where does the slowdown occur? -->

- [ ] File loading
- [ ] Structural analysis
- [ ] Formula parsing
- [ ] Pattern detection
- [ ] AI analysis
- [ ] Report generation
- [ ] Unknown

## 🖥️ System Information

- **OS**: [e.g., macOS 14.0 on M2 Max]
- **CPU**: [e.g., Apple M2 Max, Intel i9-12900K]
- **RAM**: [e.g., 32GB]
- **Storage**: [SSD/HDD]
- **Python Version**: [e.g., 3.12.0]
- **Running in Docker**: Yes/No
- **Other Running Applications**: [anything consuming significant resources]

## 📊 Profiling Data

<!-- If possible, run with profiling enabled -->

<details>
<summary>Memory Profile</summary>

```
# Run with: uv run python -m memory_profiler spreadsheet_analyzer ...
<!-- Paste memory profile output -->
```

</details>

<details>
<summary>CPU Profile</summary>

```
# Run with: uv run python -m cProfile -o profile.stats spreadsheet_analyzer ...
<!-- Paste relevant profile output -->
```

</details>

## 🔄 Reproducibility

- [ ] Issue occurs consistently
- [ ] Issue occurs intermittently
- [ ] Issue occurs only with specific files
- [ ] Issue occurs after extended usage

## 📎 Sample File

<!-- Can you provide a file that demonstrates the issue? -->

- [ ] I can provide the actual file
- [ ] I can provide a similar file with anonymized data
- [ ] I cannot share the file but can describe its structure
- [ ] I can create a synthetic file that reproduces the issue

## 🚀 Workarounds Tried

<!-- What have you tried to improve performance? -->

- [ ] Using `--quick` mode instead of `--deep`
- [ ] Processing sheets individually
- [ ] Reducing file size
- [ ] Increasing system resources
- [ ] Other: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

## 💡 Suggestions

<!-- Any ideas for performance improvements? -->

## 📋 Checklist

- [ ] I've checked that I'm using the latest version
- [ ] I've reviewed the performance documentation
- [ ] I've tried basic optimization steps
- [ ] I've included all relevant metrics above
- [ ] I've run with `--profile` flag if available
