from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import StepLog, ValidationCheck, ValidationReport
from aor_runtime.tools.filesystem import fs_exists, fs_list, fs_read, resolve_path


class RuntimeValidator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate(self, history: list[StepLog]) -> ValidationReport:
        checks: list[ValidationCheck] = []
        for item in history:
            checks.append(self._validate_step(item))
        success = all(check.success for check in checks)
        detail = "validated" if success else "one or more checks failed"
        return ValidationReport(success=success, checks=checks, detail=detail)

    def _validate_step(self, item: StepLog) -> ValidationCheck:
        step = item.step
        if not item.success:
            return ValidationCheck(name=f"step_{step.id}_{step.action}", success=False, detail=item.error or "step failed")

        try:
            if step.action == "fs.exists":
                current = fs_exists(self.settings, str(step.args["path"]))
                return ValidationCheck(
                    name=f"step_{step.id}_{step.action}",
                    success=bool(current["exists"]),
                    detail=f"exists={current['exists']} path={current['path']}",
                )

            if step.action == "fs.copy":
                src = str(step.args["src"])
                dst = str(step.args["dst"])
                src_check = fs_exists(self.settings, src)
                dst_check = fs_exists(self.settings, dst)
                if not src_check["exists"] or not dst_check["exists"]:
                    return ValidationCheck(
                        name=f"step_{step.id}_{step.action}",
                        success=False,
                        detail="source or destination missing after copy",
                    )
                src_content = str(fs_read(self.settings, src)["content"])
                dst_content = str(fs_read(self.settings, dst)["content"])
                return ValidationCheck(
                    name=f"step_{step.id}_{step.action}",
                    success=src_content == dst_content,
                    detail="copied file matches source exactly" if src_content == dst_content else "copied file content mismatch",
                )

            if step.action == "fs.read":
                actual = str(fs_read(self.settings, str(step.args["path"]))["content"])
                observed = str(item.result.get("content", ""))
                return ValidationCheck(
                    name=f"step_{step.id}_{step.action}",
                    success=actual == observed,
                    detail="read content matches filesystem" if actual == observed else "read content mismatch",
                )

            if step.action == "fs.write":
                actual = str(fs_read(self.settings, str(step.args["path"]))["content"])
                expected = str(step.args["content"])
                return ValidationCheck(
                    name=f"step_{step.id}_{step.action}",
                    success=actual == expected,
                    detail="write content exact match" if actual == expected else "write content mismatch",
                )

            if step.action == "fs.mkdir":
                resolved = resolve_path(self.settings, str(step.args["path"]))
                return ValidationCheck(
                    name=f"step_{step.id}_{step.action}",
                    success=resolved.exists() and resolved.is_dir(),
                    detail=f"directory exists at {resolved}",
                )

            if step.action == "fs.list":
                actual = list(fs_list(self.settings, str(step.args["path"]))["entries"])
                observed = [str(entry) for entry in item.result.get("entries", [])]
                return ValidationCheck(
                    name=f"step_{step.id}_{step.action}",
                    success=actual == observed,
                    detail="directory listing matches filesystem" if actual == observed else "directory listing mismatch",
                )

            if step.action == "shell.exec":
                returncode = int(item.result.get("returncode", 0))
                return ValidationCheck(
                    name=f"step_{step.id}_{step.action}",
                    success=returncode == 0,
                    detail=f"returncode={returncode}",
                )

            if step.action == "python.exec":
                return ValidationCheck(
                    name=f"step_{step.id}_{step.action}",
                    success="result" in item.result,
                    detail="python.exec returned structured result",
                )
        except Exception as exc:  # noqa: BLE001
            return ValidationCheck(name=f"step_{step.id}_{step.action}", success=False, detail=str(exc))

        return ValidationCheck(name=f"step_{step.id}_{step.action}", success=False, detail="unknown action")
