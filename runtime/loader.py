import yaml

def load_spec(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
