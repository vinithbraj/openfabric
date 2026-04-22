import copy
import unittest

from runtime.contracts import ContractRegistry
from runtime.engine import Engine
from runtime.registry import ADAPTER_REGISTRY


class _CaptureReducerAdapter:
    last_payload = None

    def __init__(self, _config):
        pass

    def handle(self, event_name, payload):
        if event_name != "data.reduce":
            return []
        self.__class__.last_payload = copy.deepcopy(payload)
        return [
            (
                "data.reduced",
                {
                    "step_id": payload.get("step_id"),
                    "task": payload.get("task", ""),
                    "original_task": payload.get("original_task", ""),
                    "target_agent": payload.get("target_agent", ""),
                    "source_event": payload.get("source_event", ""),
                    "detail": "Reduced step output.",
                    "reduced_result": "There are 8 tables in the dicom schema.",
                    "strategy": "summary",
                    "attempts": 1,
                },
            )
        ]


TEST_SPEC = {
    "contracts": {
        "DataReduceRequest": {
            "type": "object",
            "required": ["task", "step_id", "source_event"],
            "properties": {
                "run_id": {"type": "string"},
                "attempt": {"type": "number"},
                "task": {"type": "string"},
                "original_task": {"type": "string"},
                "step_id": {"type": "string"},
                "target_agent": {"type": "string"},
                "source_event": {"type": "string"},
                "reduction_request": {"type": "object"},
                "local_reduction_command": {"type": "string"},
                "existing_reduced_result": {},
                "input_data": {},
                "source_value": {},
                "source_payload": {"type": "object"},
                "available_context": {"type": "object"},
                "node": {"type": "object"},
            },
        },
        "DataReducedResult": {
            "type": "object",
            "required": ["step_id", "source_event", "strategy"],
            "properties": {
                "step_id": {"type": "string"},
                "task": {"type": "string"},
                "original_task": {"type": "string"},
                "target_agent": {"type": "string"},
                "source_event": {"type": "string"},
                "detail": {"type": "string"},
                "reduced_result": {},
                "strategy": {"type": "string"},
                "attempts": {"type": "number"},
                "local_reduction_command": {"type": "string"},
                "error": {"type": "string"},
                "node": {"type": "object"},
            },
        },
    },
    "events": {
        "data.reduce": {"contract": "DataReduceRequest"},
        "data.reduced": {"contract": "DataReducedResult"},
    },
    "agents": {
        "data_reducer": {
            "runtime": {"adapter": "test_capture_reducer"},
            "subscribes_to": ["data.reduce"],
            "emits": ["data.reduced"],
        }
    },
}


class ReducerContractTests(unittest.TestCase):
    def setUp(self):
        ADAPTER_REGISTRY["test_capture_reducer"] = _CaptureReducerAdapter
        _CaptureReducerAdapter.last_payload = None

    def tearDown(self):
        ADAPTER_REGISTRY.pop("test_capture_reducer", None)

    def test_contract_registry_accepts_absent_optional_reducer_fields(self):
        registry = ContractRegistry(TEST_SPEC["contracts"])
        registry.validate_payload(
            "DataReduceRequest",
            {
                "task": "List the tables in the dicom schema and tell me how many there are.",
                "step_id": "step1",
                "source_event": "sql.result",
                "target_agent": "sql_runner_dicom_mock",
                "reduction_request": {
                    "kind": "sql.summary",
                    "task": "List the tables in the dicom schema and tell me how many there are.",
                },
            },
        )
        registry.validate_payload(
            "DataReducedResult",
            {
                "step_id": "step1",
                "source_event": "sql.result",
                "detail": "Reduced step output.",
                "reduced_result": "There are 8 tables in the dicom schema.",
                "strategy": "summary",
                "attempts": 1,
            },
        )

    def test_engine_omits_empty_reducer_command_for_sql_summary_request(self):
        engine = Engine(TEST_SPEC)
        engine.setup()

        reduction_result = engine._reduce_step_output(
            workflow_payload={
                "run_id": "run1",
                "attempt": 1,
                "task": "List the tables in the dicom schema and tell me how many there are.",
            },
            step_id="step1",
            step_payload={
                "id": "step1",
                "task": "List the qualifying rows with requested details",
                "target_agent": "sql_runner_dicom_mock",
            },
            primary_event="sql.result",
            primary_payload={
                "detail": "SQL query executed.",
                "sql": "select table_name from information_schema.tables where table_schema = 'dicom'",
                "result": {
                    "columns": ["table_name"],
                    "rows": [{"table_name": "patients"}],
                    "row_count": 8,
                    "total_matching_rows": 8,
                },
                "reduction_request": {
                    "kind": "sql.summary",
                    "task": "List the tables in the dicom schema and tell me how many there are.",
                    "source_sql": "select table_name from information_schema.tables where table_schema = 'dicom'",
                    "columns": ["table_name"],
                    "row_count": 8,
                    "sample_rows": [{"table_name": "patients"}],
                    "input_format": "json",
                },
            },
            primary_value={
                "columns": ["table_name"],
                "rows": [{"table_name": "patients"}],
                "row_count": 8,
                "total_matching_rows": 8,
            },
            context={},
            depth=0,
        )

        self.assertIsNotNone(reduction_result)
        self.assertEqual(reduction_result["strategy"], "summary")
        self.assertIsNotNone(_CaptureReducerAdapter.last_payload)
        self.assertIn("reduction_request", _CaptureReducerAdapter.last_payload)
        self.assertNotIn("local_reduction_command", _CaptureReducerAdapter.last_payload)


if __name__ == "__main__":
    unittest.main()
