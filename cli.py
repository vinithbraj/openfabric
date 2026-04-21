import argparse
import json
import sys

from runtime.console import log_error
from runtime.loader import list_spec_agents, load_spec
from runtime.semantic_validator import validate_semantics
from runtime.engine import Engine
from runtime.run_store import RunStore


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec_path", nargs="?", help="Path to the ASDL spec file")
    parser.add_argument("question", nargs="*", help="Optional one-shot question")
    parser.add_argument(
        "--timeout",
        type=float,
        dest="timeout_seconds",
        help="Global timeout override in seconds for HTTP agent calls and autostart readiness",
    )
    parser.add_argument(
        "--agent",
        action="append",
        dest="selected_agents",
        help="Start only the named agent(s). Repeat the flag or pass a comma-separated list. Matches agent name, template name, or argument instance name.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--list-agents",
        action="store_true",
        help="List available agent selectors for the spec and exit.",
    )
    mode_group.add_argument(
        "--list-runs",
        action="store_true",
        help="List persisted workflow runs and exit.",
    )
    mode_group.add_argument(
        "--show-run",
        help="Show the persisted inspection payload for a run id and exit.",
    )
    mode_group.add_argument(
        "--show-run-graph",
        help="Render the persisted workflow graph for a run id and exit.",
    )
    mode_group.add_argument(
        "--serve-runs-ui",
        action="store_true",
        help="Serve a browser UI for persisted runs and execution graphs.",
    )
    parser.add_argument(
        "--run-status",
        help="Optional status filter for --list-runs.",
    )
    parser.add_argument(
        "--run-limit",
        type=int,
        default=20,
        help="Maximum number of runs to show with --list-runs.",
    )
    parser.add_argument(
        "--graph-format",
        choices=("mermaid", "json"),
        default="mermaid",
        help="Output format for --show-run-graph.",
    )
    parser.add_argument(
        "--ui-host",
        default="127.0.0.1",
        help="Host bind address for --serve-runs-ui.",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=8787,
        help="Port for --serve-runs-ui.",
    )
    return parser


def _format_run_summary(summary):
    parts = [
        str(summary.get("run_id") or ""),
        str(summary.get("status") or "unknown"),
        f"attempts={summary.get('attempt_count', 0)}",
    ]
    updated_at = summary.get("updated_at") or summary.get("created_at")
    if updated_at:
        parts.append(f"updated={updated_at}")
    task = str(summary.get("task") or "").strip()
    if task:
        parts.append(f"task={task}")
    active_step_id = summary.get("active_step_id")
    if active_step_id:
        parts.append(f"active_step={active_step_id}")
    return " ".join(parts)


def main():
    engine = None
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.list_runs or args.show_run or args.show_run_graph or args.serve_runs_ui:
            if args.serve_runs_ui:
                from runtime.run_visualizer import serve_run_visualizer

                serve_run_visualizer(host=args.ui_host, port=args.ui_port)
                return
            store = RunStore()
            if args.list_runs:
                for summary in store.list_runs(limit=args.run_limit, status=args.run_status):
                    print(_format_run_summary(summary))
                return
            if args.show_run:
                inspection = store.inspect(args.show_run, include_timeline=True)
                if not isinstance(inspection, dict):
                    raise ValueError(f"Run '{args.show_run}' was not found.")
                print(json.dumps(inspection, ensure_ascii=True, indent=2, sort_keys=True))
                return
            inspection = store.inspect(args.show_run_graph, include_timeline=False)
            if not isinstance(inspection, dict):
                raise ValueError(f"Run '{args.show_run_graph}' was not found.")
            if args.graph_format == "json":
                print(json.dumps(inspection.get("graph") or {}, ensure_ascii=True, indent=2, sort_keys=True))
                return
            print(str(inspection.get("graph_mermaid") or "flowchart TD"))
            return

        if args.list_agents:
            if not args.spec_path:
                parser.error("spec_path is required with --list-agents")
            for agent in list_spec_agents(args.spec_path):
                aliases = agent.get("aliases") or []
                alias_text = f" aliases: {', '.join(aliases)}" if aliases else ""
                description = agent.get("description", "")
                detail = f" - {description}" if description else ""
                print(f"{agent['name']}{alias_text}{detail}")
            return

        if not args.spec_path:
            parser.error("spec_path is required unless you are using --list-runs, --show-run, --show-run-graph, or --serve-runs-ui")

        spec = load_spec(args.spec_path, selected_agents=args.selected_agents)
        validate_semantics(spec)

        engine = Engine(spec, global_timeout_seconds=args.timeout_seconds)
        engine.setup()

        question = " ".join(args.question).strip() if args.question else None
        if question:
            engine.emit("user.ask", {"question": question})
            return

        print("Interactive mode. Type a prompt and press Enter. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_input = input("ask> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                break

            engine.emit("user.ask", {"question": user_input})
    except Exception as exc:
        log_error(f"{exc}")
        sys.exit(1)
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
