import argparse
import sys

from runtime.console import log_error
from runtime.loader import load_spec
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
    return parser


def main():
    engine = None
    parser = _build_parser()
    args = parser.parse_args()

    try:
        spec = load_spec(args.spec_path)
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
