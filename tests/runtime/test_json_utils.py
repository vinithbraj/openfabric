from __future__ import annotations

from aor_runtime.core.utils import extract_json_object


def test_extract_json_object_repairs_invalid_string_escape_in_sql() -> None:
    payload = extract_json_object(
        """
        {
          "steps": [
            {
              "id": 1,
              "action": "sql.query",
              "args": {
                "database": "dicom",
                "query": "SELECT patient_id, name, dob FROM patient WHERE dob <= CURRENT_DATE - INTERVAL \\'45 years\\' ORDER BY dob"
              }
            }
          ]
        }
        """
    )

    assert payload["steps"][0]["args"]["query"] == (
        "SELECT patient_id, name, dob FROM patient WHERE dob <= CURRENT_DATE - INTERVAL '45 years' ORDER BY dob"
    )


def test_extract_json_object_preserves_valid_escapes_while_repairing_invalid_ones() -> None:
    payload = extract_json_object('{"note":"line1\\nline2 and \\'"'"'quote\\'"'"'"}')

    assert payload["note"] == "line1\nline2 and 'quote'"
