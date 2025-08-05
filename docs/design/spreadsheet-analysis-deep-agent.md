# Spreadsheet Analysis Deep Agent — **Design Document**

______________________________________________________________________

## 1 Context & Motivation

Excel files often combine **multiple disparate tables per sheet** and **complex cross‑sheet / cross‑workbook formulas**. Analysts need accurate lineage and robust profiling, yet existing loaders (e.g., `pandas.read_excel`) flatten structure and ignore formulas. We propose an **AutoGen‑based multi‑agent pipeline** that fully explains any workbook — tables, formulas, statistics — while keeping every code step reproducible in notebooks.

______________________________________________________________________

## 2 Goals & Non‑Goals

| Goals                                                      | Out‑of‑Scope                              |
| ---------------------------------------------------------- | ----------------------------------------- |
| Detect every table via an LLM agent (no rule heuristics).  | Extract charts, images, or VBA.           |
| Build a whole‑workbook formula graph incl. external links. | Execute Excel macros.                     |
| Isolate **each agent in its own Docker Jupyter kernel**.   | Real‑time co‑editing while pipeline runs. |
| Persist artefacts so agents communicate via files.         | Threshold‑based ML gating (removed).      |

______________________________________________________________________

## 3 High‑Level Architecture

```
User → Supervisor Agent
        ├── Sheet‑Executor (one per sheet, light wrapper)
        │     ├── Multi‑Table‑Detection Agent   (LLM‑only)
        │     ├── Formula‑Graph Agent           (Python)
        │     └── N × Data‑Analysis Agents      (Python, per table)
        └── Workbook‑Synthesiser Agent          (Python)

All agents --> Artefact Store (table_meta.json, graph.pkl, analysis/*.json)
Tracing & Guardrails (OpenTelemetry, Langfuse, Docker resource limits)
```

______________________________________________________________________

## 4 Agent Catalogue (Detailed)

Each subsection follows **Purpose → Inputs → Outputs → Internal Flow → Kernel / Runtime**.

### 4.1 Supervisor Agent

- **Purpose** – Orchestrate the run: load workbook, spawn Sheet‑Executors, collate results.

- **Inputs** – File path (xlsx/xlsm), config flags.

- **Outputs** – Return status, path to final artefacts.

- **Flow** –

  1. Validate file integrity.
  1. For each worksheet → spin up Sheet‑Executor.
  1. After all sheets finish → launch Workbook‑Synthesiser.

- **Kernel** – Lightweight Python process (no Jupyter kernel needed).

### 4.2 Sheet‑Executor *(Wrapper)*

- **Purpose** – Sequentially run the three per‑sheet agents; handle artefact paths.
- **Inputs** – Worksheet object.
- **Outputs** – Aggregated per‑sheet folder under artefact store.
- **Kernel** – None (delegates to child agents).

### 4.3 Multi‑Table‑Detection Agent

- **Purpose** – Locate every rectangular table using LLM reasoning.

- **Inputs** – Sheet dataframe (CSV serialised), system prompt with deep‑scan checklist.

- **Outputs** – `table_meta.json` with `{table_id, range, columns, semantic_hint}` per table.

- **Internal Flow** –

  1. Prompt instructs model to examine headers, data types, blank row/col patterns.
  1. Model self‑audits by sampling three random rows; if schema mismatches, it revises ranges.
  1. Emits final JSON.

- **Kernel** – Docker Jupyter kernel via `JupyterCodeExecutor` (hosts helper code only; LLM decides boundaries).

### 4.4 Formula‑Graph Agent

- **Purpose** – Build/merge a **whole‑workbook** dependency DAG capturing cross‑sheet & external edges.

- **Inputs** – Worksheet object, existing `graph.pkl` (may be empty).

- **Outputs** – Updated `graph.pkl` (NetworkX), `adjacency.json` (lightweight view).

- **Internal Flow** –

  1. Parse all formulas via `formulas` lib.
  1. Resolve references: same sheet, cross sheet, `[Workbook]Sheet` external, 3‑D ranges, named ranges.
  1. Tag nodes: `external:true`, `external_missing:true`, `volatile:true`.
  1. Detect cycles; write `cycles.csv` for future Error‑Fix agent.

- **Kernel** – Dedicated Docker Jupyter kernel (Python scripts only).

### 4.5 Data‑Analysis Agent *(one per table)*

- **Purpose** – Profile and model the table.

- **Inputs** – Table range, full dataframe slice, `graph.pkl` for lineage.

- **Outputs** – `analysis/{table_id}.json`, markdown narrative, charts embedded in notebook.

- **Internal Flow** –

  1. Schema inference (dtype, nulls, uniqueness).
  1. Descriptive stats & visualisations.
  1. ML module: if numeric target exists → regression; if categorical → classification/clustering. **Runs unconditionally**.
  1. Use lineage helpers (`get_precedents(cell)`) to explain derived metrics.

- **Kernel** – Own Docker Jupyter; heavy libs (pandas, sklearn) installed.

### 4.6 Workbook‑Synthesiser Agent

- **Purpose** – Consolidate per‑sheet/table artefacts and compile final deliverables.

- **Inputs** – All artefacts + `graph.pkl`.

- **Outputs** – `final_report.md`, `influence_diagram.svg`, `manifest.json`.

- **Internal Flow** –

  1. Load table metadata, analysis summaries, and graph.
  1. Generate high‑level description (workbook purpose, sheet relationships).
  1. Highlight *Broken External Links* and *Circular References* (red edges).
  1. Save files; notify supervisor.

- **Kernel** – Docker Jupyter (lightweight).

### 4.7 Prompt Manager (Supporting Service)

- Stores versioned JSON prompts; ships Operator Glossary snapshot; updates glossary online if allowed.

### 4.8 Observability & Guardrails (Cross‑cutting)

- OpenTelemetry instrumentation; Langfuse sink.
- Docker CPU/RAM quotas; import blacklist pre‑exec.
- PII redaction before LLM calls.

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

## 9 References

1. AutoGen JupyterCodeExecutor docs
1. AutoGen Group‑Chat pattern
1. *formulas* PyPI parser
1. *excel‑dependency‑graph* library
1. Microsoft docs — external workbook links & 3‑D references
1. Langfuse + AutoGen tracing guide

______________________________________________________________________
