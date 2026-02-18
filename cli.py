import sys
from runtime.loader import load_spec
from runtime.semantic_validator import validate_semantics
from runtime.engine import Engine


def main():

    if len(sys.argv) < 2:
        print("Usage: python cli.py <spec.yml>")
        sys.exit(1)

    spec_path = sys.argv[1]

    spec = load_spec(spec_path)
    validate_semantics(spec)

    engine = Engine(spec)
    engine.setup()

    # Initial trigger
    engine.emit("user.ask", {"question": "What is Artificial Intelligence?"})


if __name__ == "__main__":
    main()
