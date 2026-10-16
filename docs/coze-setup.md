# Coze Coding Setup

1. Create or select a Linux Coze Coding workspace with Python 3.12, then clone this repository into that workspace.
2. Install the locked dependencies with `uv sync --frozen`.
3. Configure the workspace managed identity so the process can read `COZE_WORKLOAD_IDENTITY_API_KEY` and `COZE_INTEGRATION_MODEL_BASE_URL`. Configure `PGDATABASE_URL` only when persistent PostgreSQL checkpoints are required.
4. Edit `config/agent_llm_config.json` and select a model available to the workspace. The sample model name is not a credential or an availability guarantee.
5. Review the synthetic Markdown files in `assets/docs/`. On first run, the Agent imports them into the dedicated `autocert_agent_demo_knowledge` dataset. The sample is intentionally not a source of official regulations.
6. Start a demo request with `bash scripts/local_run.sh -m agent`.
7. For a private platform HTTP deployment, set `COZE_BIND_HOST=0.0.0.0` in the managed environment and run `bash scripts/http_run.sh -p 5000`. Keep the service behind the platform proxy and access controls; the application itself does not authenticate callers.

Do not paste secrets into JSON, shell command arguments, logs, Git, or this document. Use Coze managed identity for credentials. Before importing company or customer materials, confirm authorization, data residency, model retention, and knowledge-base access controls.
