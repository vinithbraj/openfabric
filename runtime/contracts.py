from typing import Any

try:
    from jsonschema import validate
except ImportError:  # pragma: no cover - lightweight environments
    def validate(*, instance: Any, schema: dict[str, Any]) -> None:
        expected_type = schema.get("type")
        if expected_type == "object" and not isinstance(instance, dict):
            raise ValueError("Payload must be an object.")
        required = schema.get("required")
        if isinstance(required, list) and isinstance(instance, dict):
            missing = [key for key in required if key not in instance]
            if missing:
                raise ValueError(f"Missing required contract fields: {', '.join(str(key) for key in missing)}")

class ContractRegistry:

    def __init__(self, contracts: dict):
        self.contracts = contracts

    def validate_payload(self, contract_name: str, payload: dict):

        if contract_name not in self.contracts:
            raise ValueError(f"Unknown contract: {contract_name}")

        contract = self.contracts[contract_name]

        if contract["type"] != "object":
            raise ValueError("Only object contracts supported in v0.1")

        schema = self._build_object_schema(contract)

        validate(instance=payload, schema=schema)

    def _build_object_schema(self, contract: dict) -> dict:
        schema = {"type": "object"}
        for key in (
            "required",
            "properties",
            "additionalProperties",
            "minProperties",
            "maxProperties",
        ):
            if key in contract:
                schema[key] = contract[key]
        return schema
