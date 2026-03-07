import sys
from runtime.loader import load_spec
from runtime.semantic_validator import validate_semantics
from runtime.engine import Engine


def main():
    engine = None

    if len(sys.argv) < 2:
        print("Usage: python cli.py <spec.yml> [question]")
        sys.exit(1)

    spec_path = sys.argv[1]
    question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None

    try:
        spec = load_spec(spec_path)
        validate_semantics(spec)

        engine = Engine(spec)
        engine.setup()

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
        print(f"Error: {exc}")
        sys.exit(1)
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
