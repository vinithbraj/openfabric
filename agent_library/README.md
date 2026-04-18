Agent Library

Reusable HTTP agents that follow the ASDL event contract style (`/handle` endpoint with `event`, `payload`, and `emits` response).

Contents

- `agents/planner.py`: converts `user.ask` into `research.query` and `task.plan`
- `agents/retriever.py`: handles `research.query` and emits `research.result`
- `agents/calculator.py`: handles `task.plan` and emits `task.result`
- `agents/synthesizer.py`: turns `research.result` or `task.result` into `answer.final`
- `agents/operations_planner.py`: detects file/shell/notify tasks from `user.ask`
- `agents/llm_operations_planner.py`: LLM-driven planner with deterministic fallback
- `agents/filesystem.py`: reads workspace files from `file.read`
- `agents/shell_runner.py`: executes allowlisted shell commands from `shell.exec`
- `agents/notifier.py`: handles `notify.send` and emits `notify.result`
- `specs/research_task_assistant.yml`: composable ASDL blueprint using these agents
- `specs/ops_assistant.yml`: composable ASDL blueprint for operational tasks
- `specs/ops_assistant_llm.yml`: ops assistant using `llm_operations_planner`

Run

```bash
source .venv/bin/activate
python cli.py agent_library/specs/research_task_assistant.yml
python cli.py agent_library/specs/ops_assistant.yml
python cli.py agent_library/specs/ops_assistant_llm.yml
python cli.py agent_library/specs/ops_assistant_llm.yml --list-agents
python cli.py agent_library/specs/ops_assistant_llm.yml --agent shell_runner
python cli.py agent_library/specs/ops_assistant_llm.yml --agent sql_runner --agent synthesizer
```

`--agent` works like a runtime service filter. It can match:

- the concrete agent name, for example `sql_runner_mydb`
- the template agent name, for example `sql_runner`
- the argument instance name, for example `mydb`

LLM planner environment

`llm_operations_planner` reads:

- `LLM_OPS_API_KEY` (required for LLM mode)
- `LLM_OPS_BASE_URL` (default: `https://api.openai.com/v1`)
- `LLM_OPS_MODEL` (default: `gpt-4o-mini`)
- `LLM_OPS_SYNTH_MODEL` (optional override for final-answer synthesis; defaults to `LLM_OPS_MODEL`)
- `LLM_OPS_TIMEOUT_SECONDS` (default: `300`)

If LLM config/request fails, it falls back to deterministic planning.

Runtime capability discovery

In `specs/ops_assistant_llm.yml`, runtime emits `system.capabilities` at setup.
`llm_operations_planner` subscribes to this event and dynamically restricts planning
to tools/events that are actually available in the current ASDL topology.

Agent metadata contract for capability routing

Agents can expose richer `AGENT_METADATA` keys to improve planner matching:

- `capability_domains`: high-level domains (for example `shell`, `filesystem`, `notification`)
- `action_verbs`: verbs the agent can perform (for example `read`, `execute`, `notify`)
- `side_effect_policy`: behavior class (`read_only`, `allow_non_destructive_side_effects`, etc.)
- `safety_enforced_by_agent`: whether the agent applies its own runtime safety checks

These fields are published in `system.capabilities` and consumed by `llm_operations_planner`.

Reuse pattern

1. Add a new agent module under `agent_library/agents/`.
2. Implement `/handle` returning `{"emits": [...]}`.
3. Reference the app in spec with:

```yaml
runtime:
  adapter: http
  endpoint: "http://localhost:<port>/handle"
  autostart:
    app: agent_library.agents.<module>:app
```
