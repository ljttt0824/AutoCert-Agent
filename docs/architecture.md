# Architecture

## Agent orchestration

`src/agents/agent.py` constructs one LangChain `create_agent` backed by LangGraph. The system prompt selects a working mode and calls one of three tools. This is a unified Agent with specialized tools, not four separately deployed Agent processes.

```text
Request → Agent prompt routing
           ├─ knowledge_base_query
           ├─ parameter_semantic_check
           └─ progress_risk_scan
        → response with evidence or review flags
```

Material drafting is prompt-guided generation and has no dedicated deterministic tool.

## Knowledge retrieval

`src/storage/knowledge/kb.py` reads Markdown files under `assets/docs/`. It imports them through Coze `KnowledgeClient`, then combines semantic results with a local keyword-search fallback. Result text includes the source filename. The repository uses a distinct demo dataset name to avoid colliding with other Coze knowledge bases.

The bundled corpus is synthetic. Replace it only with materials that are authorized for the target deployment, then verify chunking, retrieval, citations, and retention settings in the Coze workspace.

## Parameter comparison

`src/tools/vehicle_cert_tools.py` parses tabular parameter text, maps known aliases to canonical labels, applies a small set of explicit conversions, and sorts severe differences first. `Wh` to `kWh`, hours to minutes, and `kV` to `V` have direct conversions. `mAh` cannot be compared to `kWh` without voltage, so the tool returns a manual-review result instead of guessing.

This is a small rule set, not a universal vehicle schema. Unrecognized names and domain-specific units need human review.

## Schedule risk analysis

The tool reads `today` and `submission_deadline` from the task JSON for reproducible runs. If `today` is omitted, it uses the host date; a valid deadline is required. Tasks are classified using a seven-day near-deadline window. The longest dependency chain is calculated from the task graph; cycles suppress that chain instead of presenting a misleading path. The result is a planning aid and does not calculate duration-based critical paths.

## Runtime and state

`src/main.py` exposes Coze-compatible HTTP and streaming endpoints. The server listens on `127.0.0.1` by default; set `COZE_BIND_HOST=0.0.0.0` only when a trusted platform proxy requires it. No authentication middleware is included.

`src/storage/memory/memory_saver.py` prefers the PostgreSQL URL supplied by Coze and falls back to in-memory checkpoints. The fallback loses conversation state on process restart. `src/storage/s3/` contains reusable S3-compatible storage helpers; the sample workflows do not require object storage.

## External requirements

- Linux Coze Coding workspace with Python 3.12
- Coze workload identity and an available model endpoint
- Coze KnowledgeClient service for semantic retrieval
- Optional PostgreSQL for persistent checkpoints
- `uv` for installing the locked dependency set

Local keyword search can be inspected from source, but full Agent execution and hosted retrieval cannot be reproduced outside a configured Coze workspace.
