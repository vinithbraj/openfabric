from jsonschema import validate

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
