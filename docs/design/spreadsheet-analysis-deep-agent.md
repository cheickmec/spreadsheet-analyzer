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

| Goals (MVP)                                                     | Phase 2                                        | Out of Scope                     |
| --------------------------------------------------------------- | ---------------------------------------------- | -------------------------------- |
| Detect rectangular tables and classify semantically (LLM-only). | Chart/image detection and relationship mapping | Executing Excel macros.          |
| Build formula dependency graph, catalog external links.         | Pivot table source analysis                    | Real-time collaborative editing. |
| Conditional ML analysis based on table characteristics.         | Data validation rule extraction                | Custom UI dashboards.            |
| Jupyter notebook per agent + unified final report.              | External workbook auto-resolution              | Multi-language kernels.          |
| Observability, guardrails, flexible kernel strategies.          |                                                | VBA code execution.              |

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
- "Table criteria: ≥2 columns, ≥3 rows (including header), consistent data types per column."
- "Exclude: single-cell values, sparse matrices, pivot table outputs (detect source ranges only)."
- "Validate by sampling rows for consistent column counts and data types."
- Output JSON: `{table_id, range, columns[], semantic_hint, table_type}`.
  *No heuristics or thresholds*; reasoning is delegated entirely to the LLM.
  *Artefacts* `tables.json` per sheet.

### 4.4 Formula-Graph Agent

*Scope* Parses *all* formulas across the workbook **once**, expands cross-sheet references, **catalogs but doesn't resolve** external `[Workbook]Sheet` links.
*External Links* **Single-level traversal only** - catalogs immediate external `[Workbook]Sheet` references with metadata (file path, accessibility status) but doesn't auto-resolve or follow recursive links.
*Stack* `formulas` → `excel-dependency-graph` → NetworkX.
*Formula Limits* Handles standard functions; **defers complex array formulas and nested LAMBDA expressions** to manual review (logs to `complex_formulas.csv`).
*Artefacts* `graph.pkl` and `graph_adj.json` for random-access lineage queries.
*Runs* Immediately after 4.3 so Data-Analysis Agents can query lineage.

### 4.5 Data-Analysis Agent

*Per table* Loads the table, performs schema inference, descriptive stats, and **conditional ML analysis**.
*ML Criteria* Runs ML only if: ≥10 rows, \<80% categorical data, not a lookup/reference table.
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

## 6 Execution Environment & Kernel Strategy

**Primary Strategy**: **DockerJupyterServer** per agent → **JupyterCodeExecutor**.
**Alternative**: Shared kernel with namespace isolation for lightweight agents.

| Approach         | Pros                             | Cons                          | Recommended For                    |
| ---------------- | -------------------------------- | ----------------------------- | ---------------------------------- |
| Per-agent Docker | Complete isolation, reproducible | Higher memory, slower startup | Formula-Graph, Data-Analysis       |
| Shared kernel    | Faster, lower resource usage     | Potential state pollution     | Multi-Table-Detection, Synthesiser |

**External Link Handling**: Catalog external `[Workbook]Sheet` references with metadata (file path, reachability) but **do not auto-resolve** to avoid authentication and network complexity in MVP.

______________________________________________________________________

## 7 Trade-offs & Technical Decisions

| Decision                 | Alternatives         | Rationale                                                                  |
| ------------------------ | -------------------- | -------------------------------------------------------------------------- |
| Conditional ML analysis  | Always/never run ML  | Avoids noise on lookup tables while capturing insights from real datasets. |
| External link cataloging | Full resolution      | Simpler MVP; defer auth/network complexity to Phase 2.                     |
| LLM-only table detection | Hybrid heuristic+LLM | Aligns with "treat LLM like human analyst" requirement.                    |
| AutoGen framework        | LangGraph, CrewAI    | Bundles Jupyter executor and group-chat orchestration out-of-box.          |

## 7 Risks & Mitigations

| Risk                                | Impact              | Mitigation                                                |
| ----------------------------------- | ------------------- | --------------------------------------------------------- |
| LLM misses a table                  | Incomplete analysis | Prompt tuning + random row audits + manual override CLI.  |
| Docker spawn latency on large files | Longer wall‑time    | Warm‑pool of kernel containers; parallel Sheet‑Executors. |
| Large external‑link graphs          | Memory blow‑up      | Lazy graph construction + prune unused precedents.        |

______________________________________________________________________

## 9 Milestones & Success Metrics

| Phase   | Date       | Deliverable                                         | Acceptance                                      |
| ------- | ---------- | --------------------------------------------------- | ----------------------------------------------- |
| MVP     | T + 2 wks  | Supervisor + single-sheet pipeline + conditional ML | All artefacts present; ML skips lookup tables   |
| MVP     | T + 4 wks  | Formula-Graph with external link cataloging         | Graph maps cross-sheet, catalogs externals      |
| MVP     | T + 6 wks  | Parallel Sheet-Executors + hybrid kernel strategy   | 80% runtime ≤ 8 min for 20-sheet workbook       |
| MVP     | T + 8 wks  | Observability, guardrails, final report template    | Trace coverage ≥ 95%, formula complexity limits |
| Phase 2 | T + 12 wks | Chart detection, pivot analysis, data validation    | Visual elements mapped to source tables         |

______________________________________________________________________

## 10 Risks & Mitigations

| Risk                                          | Mitigation                                                                      |
| --------------------------------------------- | ------------------------------------------------------------------------------- |
| LLM misidentifies table boundaries            | Self-reflection directive + random-row audit in prompt; manual override option. |
| External links unreachable → graph incomplete | Catalog-only approach; flag nodes `external_missing:true` in Phase 2.           |
| Conditional ML logic too complex              | Simple heuristics (row count, categorical %) with override flags.               |
| Complex formulas break parser                 | Graceful fallback: log to `complex_formulas.csv`, mark nodes as `unparseable`.  |
| Mixed kernel strategy coordination            | Clear agent-to-kernel mapping; shared kernel only for stateless agents.         |
| Cost blow-up on large workbooks               | Token-usage monitoring + conditional ML reduces unnecessary LLM calls.          |

______________________________________________________________________

## 11 References

1. AutoGen JupyterCodeExecutor docs
1. AutoGen Group‑Chat pattern
1. *formulas* PyPI parser
1. *excel‑dependency‑graph* library
1. Microsoft docs — external workbook links & 3‑D references
1. Langfuse + AutoGen tracing guide

______________________________________________________________________
