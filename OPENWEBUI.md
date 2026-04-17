# Open WebUI Gateway

This repo exposes the OpenFabric planner stack through an OpenAI-compatible
gateway so Open WebUI can be used as the chat interface.

## Start

Keep vLLM running on port `8000`, then start the gateway:

```bash
bash scripts/start_openwebui_gateway.sh
```

The gateway listens on `0.0.0.0:8310`, autostarts the OpenFabric planner agents
from `agent_library/specs/ops_assistant_llm.yml`, and forwards planner LLM calls
to:

```text
http://127.0.0.1:8000/v1
```

Default model:

```text
deepseek-ai/deepseek-coder-6.7b-instruct
```

Override defaults with environment variables:

```bash
OPENFABRIC_GATEWAY_PORT=8311 \
LLM_OPS_BASE_URL=http://127.0.0.1:8000/v1 \
LLM_OPS_MODEL=your-vllm-model \
bash scripts/start_openwebui_gateway.sh
```

## Open WebUI Connection

In Open WebUI, add an OpenAI-compatible connection:

```text
URL: http://host.docker.internal:8310/v1
API Key: dummy
Model: openfabric-planner
```

If Open WebUI is running on Linux Docker and cannot resolve
`host.docker.internal`, start the container with:

```bash
--add-host=host.docker.internal:host-gateway
```

If Open WebUI runs directly on the host, use:

```text
URL: http://127.0.0.1:8310/v1
```

## Open WebUI Background Prompts

Open WebUI may send extra prompts after a chat response to generate follow-up
questions, chat titles, and tags. The gateway detects those housekeeping prompts
and forwards them directly to the backing LLM instead of routing them through
the OpenFabric planner.

## SQL Agent

The active planner includes read-only SQL agents. Configure the database URL
before starting the gateway:

```bash
export SQL_AGENT_DSN=sqlite:////absolute/path/to/database.db
# or
export SQL_DATABASE_URL=postgresql://user:password@host:5432/database
# or
export SQL_DATABASE_URL=mysql://user:password@host:3306/database
```

Optional:

```bash
export SQL_AGENT_ROW_LIMIT=100
```

SQLite works with the Python standard library. PostgreSQL requires `psycopg2`
or `psycopg` to be installed. MySQL/MariaDB requires `pymysql`.

The SQL agent introspects schemas, tables, columns, and foreign-key
relationships, generates read-only SQL for database questions, rejects mutating
SQL, executes the query, and returns rows for synthesis.

Multiple SQL databases are defined in `agent_library/specs/ops_assistant_llm.yml`
under:

```yaml
arguments:
  agents:
    sql_runner:
      - name: mydb
        port: 8307
        env:
          SQL_AGENT_DSN: "${SQL_DATABASE_URL:-postgresql://admin:admin123@127.0.0.1:5432/mydb}"
          SQL_AGENT_NAME: "mydb"
        metadata:
          database_name: "mydb"
```

Each entry instantiates a separate `sql_runner_<name>` agent with its own port,
DSN, and capability metadata. The planner sees the database name in
`system.capabilities` and can route requests to the matching SQL agent.
