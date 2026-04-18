import argparse
import sys

from runtime.console import log_error
from runtime.loader import list_spec_agents, load_spec
from runtime.semantic_validator import validate_semantics
from runtime.engine import Engine


def _build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec_path", help="Path to the ASDL spec file")
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
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available agent selectors for the spec and exit.",
    )
    return parser


def main():
    engine = None
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.list_agents:
            for agent in list_spec_agents(args.spec_path):
                aliases = agent.get("aliases") or []
                alias_text = f" aliases: {', '.join(aliases)}" if aliases else ""
                description = agent.get("description", "")
                detail = f" - {description}" if description else ""
                print(f"{agent['name']}{alias_text}{detail}")
            return

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
