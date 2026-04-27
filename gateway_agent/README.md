# Gateway Agent

`gateway_agent/` is a standalone node-local execution service for OpenFabric shell commands.

This v2 agent:
- runs on one host
- executes commands locally on that host
- only accepts requests whose `node` matches its configured logical node name

## API

- `GET /healthz` -> `{ "status": "ok", "node": "<configured node>" }`
- `GET /capabilities` -> `{ "node": "<configured node>", "version": "0.4.0", "capabilities": [...] }`
- `POST /exec` with `{ "node": str, "command": str }`
- `POST /exec/stream` with `{ "node": str, "command": str }`
- `POST /exec` returns `{ "stdout": str, "stderr": str, "exit_code": int }`
- `POST /exec/stream` returns `text/event-stream` with `stdout`, `stderr`, and `completed` events

Command failures return `200` with a non-zero `exit_code`. Request validation problems return `4xx`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r gateway_agent/requirements.txt
```

## Configuration

- `GATEWAY_NODE_NAME` logical node served by this agent. Default: `localhost`
- `GATEWAY_BIND_HOST` bind host. Default: `127.0.0.1`
- `GATEWAY_BIND_PORT` bind port. Default: `8787`
- `GATEWAY_EXEC_TIMEOUT_SECONDS` per-command timeout. Default: `30`
- `GATEWAY_TRACE_COMMANDS` log each requested command on the gateway server. Default: `0`
- `GATEWAY_WORKDIR` optional working directory for command execution

## Run

```bash
export GATEWAY_NODE_NAME=localhost
export GATEWAY_BIND_HOST=127.0.0.1
export GATEWAY_BIND_PORT=8787

uvicorn gateway_agent.app:app --host "${GATEWAY_BIND_HOST}" --port "${GATEWAY_BIND_PORT}"
```

Or use the helper script:

```bash
./gateway_agent/startup.sh
```

To make the gateway print each requested command on the server side:

```bash
./gateway_agent/startup.sh --trace-commands
```

or:

```bash
export GATEWAY_TRACE_COMMANDS=1
./gateway_agent/startup.sh
```

## OpenFabric Wiring

The example runtime spec already includes:

```yaml
nodes:
  default: localhost
  endpoints:
    - name: localhost
      url: http://127.0.0.1:8787/exec
```

Then run a shell task through OpenFabric:

```bash
aor chat examples/general_purpose_assistant.yaml
```

If you want to use a different runtime spec, add a similar `nodes:` section there or fall back to the `AOR_GATEWAY_URL`, `AOR_AVAILABLE_NODES`, and `AOR_DEFAULT_NODE` environment variables.

This service does not include authentication in v1. Keep it bound to loopback or behind a trusted proxy/network boundary.
