# Design Document

**Spreadsheet Deep-Analysis Agent System (Microsoft AutoGen)**

______________________________________________________________________

## 1 Context & Motivation

Spreadsheets remain the **dominant self-service analytics tool in business**, yet they routinely evolve into semi-programs: dozens of worksheets, thousands of formulas, opaque links to external files, and multiple unrelated tables jammed into a single grid. Finance teams, auditors, data scientists, and regulators all struggle with the same pain points:

| Pain Point                                                          | Why It Matters                                           |
| ------------------------------------------------------------------- | -------------------------------------------------------- |
| **Hidden logic** – business rules live in formulas nobody remembers | Bugs propagate unnoticed; Sox/FDA audits fail            |
| **Multiple tables per sheet** – no explicit schema                  | Automated loaders mis-parse, leading to data loss        |
| **Cross-sheet & cross-file links**                                  | A single broken external workbook silently corrupts KPIs |
| **Manual, error-prone review**                                      | High-value analysts spend hours eyeballing cells         |

Existing tools attack fragments of the problem (table inference heuristics, separate lineage linters, basic descriptive stats) but **no end-to-end system "understands" the entire workbook** and produces a reproducible, code-backed analysis.

**Vision** Architect an *AI-native pipeline* that ingests any Excel workbook and, through a team of specialised AutoGen agents—each with its own isolated Jupyter kernel—**discovers structure, maps dependencies, analyses every dataset, and emits a human-readable + machine-consumable explanation of the file**.
The outcome is a turnkey artefact you can hand to auditors, BI teams, or downstream ETL jobs to eliminate manual spelunking and boost trust in spreadsheet-driven processes.

______________________________________________________________________

## 2 Goals & Non-Goals

| Goals                                                                               | Out of Scope                                                  |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Detect every rectangular table and classify it semantically (LLM-only).             | Detecting charts, images, or embedded VBA code (future work). |
| Build a *whole-workbook* formula dependency graph, including external links.        | Executing Excel macros.                                       |
| Run deep statistical/ML analysis on each detected table—no size thresholds.         | Real-time collaborative editing while agents run.             |
| Produce a version-controlled Jupyter notebook per agent + a unified final report.   | Custom UI dashboards (left to consuming apps).                |
| Provide observability, guardrails, and artefact persistence without kernel sharing. | Multi-language (non-Python) kernels.                          |

______________________________________________________________________

## 3 Proposed Architecture (High Level)

```
Excel Workbook
      │
      ▼
Supervisor Agent ──▶  Sheet-Executor (one per sheet)
      │                   ├── Multi-Table-Detection Agent  (LLM-only)
      │                   ├── Formula-Graph Agent          (code)
      │                   └── N × Data-Analysis Agents     (code, per table)
      │
      ▼
Workbook-Synthesiser Agent  ──▶ Final Markdown + JSON + lineage graphs
```

Every **agent runs in its own Docker-backed Jupyter kernel**, guaranteeing isolation while persisting artefacts (JSON, pickle, parquet) to a shared object store for downstream consumers.

______________________________________________________________________

## 4 Agent Catalogue

### 4.1 Supervisor Agent

*Purpose* Entry point; spawns one **Sheet-Executor** per worksheet, aggregates status.
*Inputs* Excel binary. *Outputs* Global run log, error map.

### 4.2 Sheet-Executor (Wrapper)

*Flow* Sequentially invokes Section 4.3 → 4.4 → 4.5 agents for its sheet.
*Kernel* Ephemeral Docker image (`python:3.12-slim`, pandas, openpyxl).

### 4.3 Multi-Table-Detection Agent (LLM-only)

*Prompt highlights*

- "Identify every contiguous rectangular region that forms a coherent table."
- "Validate by sampling rows for consistent column counts and data types."
- Output JSON: `{table_id, range, columns[], semantic_hint}`.
  *No heuristics or thresholds*; reasoning is delegated entirely to the LLM.
  *Artefacts* `tables.json` per sheet.

### 4.4 Formula-Graph Agent

*Scope* Parses *all* formulas across the workbook **once**, expands cross-sheet & external references, detects cycles, flags volatile functions.
*Stack* `formulas` → `excel-dependency-graph` → NetworkX.
*Artefacts* `graph.pkl` and `graph_adj.json` for random-access lineage queries.
*Runs* Immediately after 4.3 so Data-Analysis Agents can query lineage.

### 4.5 Data-Analysis Agent

*Per table* Loads the table, performs schema inference, descriptive stats, optional ML (clustering/regression) **regardless of size**.
*Uses* Formula graph helper to explain derived cells ("Column E is `C*D`").
*Outputs* Notebook cells + `analysis_{table_id}.json`.

### 4.6 Workbook-Synthesiser Agent

*Aggregates* All artefacts, builds a narrative markdown report, embeds SVG of formula graph, lists external link health, and emits a machine-readable manifest.

### 4.7 Prompt-Manager (Service)

Version-controlled prompt store; injects context into 4.3 and 4.5 agents.

### 4.8 Observability & Guardrails

Cross-cutting AutoGen tracing, OpenTelemetry export, Docker resource limits, Llama-Guard for PII in prompts.

______________________________________________________________________

## 5 Execution Flow (Swimlane)

```
Supervisor ─┬─▶ Sheet‑Exec ──▶ Multi‑Table‑Detect ─▶ Formula‑Graph ─▶ Data‑Analysis (×N)
            └─▶ Workbook‑Synthesiser
```

Artefacts flow downward; no shared Python memory between kernels.

______________________________________________________________________

## 6 Technical Considerations & Trade‑offs

|                              |                           |                                                                        |
| ---------------------------- | ------------------------- | ---------------------------------------------------------------------- |
| Topic                        | Decision                  | Trade‑off                                                              |
| Kernel isolation             | One Docker per agent      | ↑ Memory & spin‑up time vs. ↓ state bleed & reproducibility.           |
| Formula‑Graph timing         | Build **before** analysis | Guarantees lineage info; small upfront cost.                           |
| LLM‑only table detection     | No heuristics in code     | Simpler codebase, relies on prompt quality & model tokens.             |
| Artefact FS vs shared memory | File hand‑off             | Slower I/O but works with isolated kernels & allows post‑run auditing. |

## 7 Risks & Mitigations

| Risk                                | Impact              | Mitigation                                                |
| ----------------------------------- | ------------------- | --------------------------------------------------------- |
| LLM misses a table                  | Incomplete analysis | Prompt tuning + random row audits + manual override CLI.  |
| Docker spawn latency on large files | Longer wall‑time    | Warm‑pool of kernel containers; parallel Sheet‑Executors. |
| Large external‑link graphs          | Memory blow‑up      | Lazy graph construction + prune unused precedents.        |

______________________________________________________________________

## 9 Milestones & Success Metrics

| Date      | Deliverable                                      | Acceptance                                      |
| --------- | ------------------------------------------------ | ----------------------------------------------- |
| T + 2 wks | MVP: Supervisor + single-sheet pipeline          | All artefacts present; unit tests pass          |
| T + 4 wks | Formula-Graph Agent integrated                   | Graph resolves cross-sheet edges on sample file |
| T + 6 wks | Parallel Sheet-Executors + Data-Analysis Agents  | 80 % runtime ≤ 10 min for 20-sheet workbook     |
| T + 8 wks | Observability, guardrails, final report template | Trace coverage ≥ 95 %, no PII leaks             |

______________________________________________________________________

## 10 Risks & Mitigations

| Risk                                          | Mitigation                                                                      |
| --------------------------------------------- | ------------------------------------------------------------------------------- |
| LLM misidentifies table boundaries            | Self-reflection directive + random-row audit in prompt; manual override option. |
| External links unreachable → graph incomplete | Flag nodes `external_missing:true`; Synthesiser surfaces warnings.              |
| Container sprawl                              | Time-to-live + CI stress test to tune resource limits.                          |
| Cost blow-up on large workbooks               | Token-usage monitoring in tracing; pricing alert thresholds.                    |

______________________________________________________________________

## 11 References

1. AutoGen JupyterCodeExecutor docs
1. AutoGen Group‑Chat pattern
1. *formulas* PyPI parser
1. *excel‑dependency‑graph* library
1. Microsoft docs — external workbook links & 3‑D references
1. Langfuse + AutoGen tracing guide

______________________________________________________________________
