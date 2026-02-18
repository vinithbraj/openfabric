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

        schema = {
            "type": "object",
            "required": contract.get("required", []),
            "properties": {
                k: {"type": v["type"]}
                for k, v in contract["properties"].items()
            }
        }

        validate(instance=payload, schema=schema)
