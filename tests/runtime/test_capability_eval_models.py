from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aor_runtime.runtime.capabilities.eval import (
    CapabilityEvalCase,
    CapabilityEvalPack,
    ensure_unique_case_ids,
    load_capability_eval_packs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKS_DIR = REPO_ROOT / "evals" / "capabilities"


def test_eval_case_model_validates() -> None:
    case = CapabilityEvalCase(id="case_1", prompt="read line 2 from notes.txt", expected="budget")
    assert case.id == "case_1"
    assert case.expect_llm_calls == 0


def test_eval_pack_model_validates() -> None:
    pack = CapabilityEvalPack(
        capability="filesystem",
        cases=[CapabilityEvalCase(id="filesystem_case_1", prompt="read line 2", expected="budget")],
        strict_threshold=0.95,
        semantic_threshold=1.0,
        max_llm_fallbacks=0,
    )
    assert pack.capability == "filesystem"


def test_invalid_threshold_fails() -> None:
    with pytest.raises(ValidationError):
        CapabilityEvalPack(
            capability="filesystem",
            cases=[CapabilityEvalCase(id="filesystem_case_1", prompt="read line 2", expected="budget")],
            strict_threshold=1.2,
        )


def test_missing_required_id_or_prompt_fails() -> None:
    with pytest.raises(ValidationError):
        CapabilityEvalCase(id="", prompt="read line 2", expected="budget")
    with pytest.raises(ValidationError):
        CapabilityEvalCase(id="case_1", prompt="", expected="budget")


def test_json_eval_files_load_successfully() -> None:
    packs = load_capability_eval_packs(PACKS_DIR)
    assert [pack.capability for pack in packs] == ["compound", "fetch", "filesystem", "shell", "sql", "text_transform"]
    assert all(pack.cases for pack in packs)


def test_all_eval_case_ids_are_unique_across_all_packs() -> None:
    packs = load_capability_eval_packs(PACKS_DIR)
    ensure_unique_case_ids(packs)
