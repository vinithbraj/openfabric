# Shell Intelligence

The runtime supports a small read-only shell intelligence layer for system inspection and explicit user-provided commands.

Shell intelligence is native-first:

- Filesystem prompts such as `list all files in this folder` use `fs.*`.
- SQL prompts use `sql.*`.
- SLURM prompts use `slurm.*`.
- Shell is used for explicit terminal commands or system inspections that do not have a safer native tool.

## Safety Model

Shell commands are classified before execution.

Modes:

- `AOR_SHELL_MODE=disabled`: no shell execution.
- `AOR_SHELL_MODE=read_only`: default; only read-only inspection commands execute.
- `AOR_SHELL_MODE=approval_required`: risky commands return approval-required/refusal responses.
- `AOR_SHELL_MODE=permissive`: local development only; forbidden commands remain blocked.

Default settings:

- `AOR_SHELL_MODE=read_only`
- `AOR_SHELL_ALLOW_MUTATION_WITH_APPROVAL=false`
- `AOR_SHELL_MAX_OUTPUT_CHARS=20000`
- `AOR_SHELL_COMMAND_TIMEOUT_SECONDS=30`

Read-only examples include `ls`, `df`, `du`, `free`, `ps`, `lsof`, `ss`, `findmnt`, `uptime`, `hostname`, and `systemctl status`.

Mutating or risky examples such as `rm`, `kill`, `chmod`, `chown`, service restart/stop/start, redirects, command substitution, and package installs are not run automatically. Forbidden examples such as `rm -rf /`, `sudo`, `mkfs`, `dd` to devices, shutdown/reboot, and `curl | bash` are blocked.

## Natural-Language Inspections

Supported natural-language inspections compile to fixed read-only templates, for example:

- `check disk usage` -> `df -h`
- `show memory usage` -> `free -h`
- `what process is using port 8310` -> `lsof -i :8310`
- `show listening ports` -> `ss -tuln`
- `show uptime` -> `uptime`
- `show service status for ssh` -> `systemctl status ssh --no-pager`

The LLM is not allowed to invent shell commands or execution plans. Optional LLM intent extraction may only produce typed shell intents.

## Presentation

User-mode responses show:

- the result output
- `Command Used`
- execution metadata such as tool, risk, and status

Raw gateway internals, lifecycle noise, and hidden telemetry are not shown in normal user responses.
