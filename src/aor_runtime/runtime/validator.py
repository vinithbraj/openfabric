from __future__ import annotations

from pathlib import Path

from aor_runtime.config import Settings, get_settings
from aor_runtime.core.contracts import StepLog, ValidationResult
from aor_runtime.core.utils import extract_json_object
from aor_runtime.tools.filesystem import fs_exists, fs_list, fs_read, resolve_path


class RuntimeValidator:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate(self, history: list[StepLog]) -> tuple[ValidationResult, list[dict[str, str | bool]]]:
        checks: list[dict[str, str | bool]] = []
        for item in history:
            checks.append(self._validate_step(item))
        failed = [check for check in checks if not bool(check["success"])]
        if failed:
            first = failed[0]
            return ValidationResult(success=False, reason=str(first["detail"])), checks
        return ValidationResult(success=True, reason=None), checks

    def _validate_step(self, item: StepLog) -> dict[str, str | bool]:
        step = item.step
        if not item.success:
            return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": item.error or "step failed"}

        try:
            if step.action == "fs.exists":
                current = fs_exists(self.settings, str(step.args["path"]))
                return {"name": f"step_{step.id}_{step.action}", "success": bool(current["exists"]), "detail": f"exists={current['exists']} path={current['path']}"}

            if step.action == "fs.copy":
                src = str(step.args["src"])
                dst = str(step.args["dst"])
                src_check = fs_exists(self.settings, src)
                dst_check = fs_exists(self.settings, dst)
                if not src_check["exists"] or not dst_check["exists"]:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "source or destination missing after copy"}
                src_content = str(fs_read(self.settings, src)["content"])
                dst_content = str(fs_read(self.settings, dst)["content"])
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": src_content == dst_content,
                    "detail": "copied file matches source exactly" if src_content == dst_content else "copied file content mismatch",
                }

            if step.action == "fs.read":
                actual = str(fs_read(self.settings, str(step.args["path"]))["content"])
                observed = str(item.result.get("content", ""))
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": actual == observed,
                    "detail": "read content matches filesystem" if actual == observed else "read content mismatch",
                }

            if step.action == "fs.write":
                actual = str(fs_read(self.settings, str(step.args["path"]))["content"])
                expected = str(step.args["content"])
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": actual == expected,
                    "detail": "write content exact match" if actual == expected else "write content mismatch",
                }

            if step.action == "fs.mkdir":
                resolved = resolve_path(self.settings, str(step.args["path"]))
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": resolved.exists() and resolved.is_dir(),
                    "detail": f"directory exists at {resolved}",
                }

            if step.action == "fs.list":
                actual = list(fs_list(self.settings, str(step.args["path"]))["entries"])
                observed = [str(entry) for entry in item.result.get("entries", [])]
                return {
                    "name": f"step_{step.id}_{step.action}",
                    "success": actual == observed,
                    "detail": "directory listing matches filesystem" if actual == observed else "directory listing mismatch",
                }

            if step.action == "shell.exec":
                returncode = int(item.result.get("returncode", 0))
                return {"name": f"step_{step.id}_{step.action}", "success": returncode == 0, "detail": f"returncode={returncode}"}

            if step.action == "python.exec":
                success = bool(item.result.get("success", False))
                detail = str(item.result.get("error") or "python.exec returned structured result")
                if not success:
                    return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": detail}
                output = item.result.get("output")
                if isinstance(output, str):
                    manifest_check = self._validate_python_manifest(output)
                    if manifest_check is not None:
                        return {"name": f"step_{step.id}_{step.action}", "success": manifest_check[0], "detail": manifest_check[1]}
                return {"name": f"step_{step.id}_{step.action}", "success": True, "detail": "python.exec returned structured result"}
        except Exception as exc:  # noqa: BLE001
            return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": str(exc)}

        return {"name": f"step_{step.id}_{step.action}", "success": False, "detail": "unknown action"}

    def _validate_python_manifest(self, output: str) -> tuple[bool, str] | None:
        try:
            payload = extract_json_object(output)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(payload, dict):
            return None

        operation = payload.get("operation")
        if operation != "bulk_copy":
            return None

        src_dir = payload.get("src_dir")
        dst_dir = payload.get("dst_dir")
        copied_files = payload.get("copied_files")
        if not isinstance(src_dir, str) or not isinstance(dst_dir, str) or not isinstance(copied_files, list):
            return False, "python.exec bulk_copy manifest is missing required fields"

        src_entries = [name for name in fs_list(self.settings, src_dir)["entries"] if str(name).endswith(".txt")]
        dst_entries = fs_list(self.settings, dst_dir)["entries"]
        normalized_copied = [str(name) for name in copied_files]
        if sorted(src_entries) != sorted(normalized_copied):
            return False, "python.exec bulk_copy manifest does not match source txt files"
        for name in src_entries:
            if name not in dst_entries:
                return False, f"bulk copy missing destination file {name}"
            src_content = fs_read(self.settings, f"{src_dir}/{name}")["content"]
            dst_content = fs_read(self.settings, f"{dst_dir}/{name}")["content"]
            if src_content != dst_content:
                return False, f"bulk copy content mismatch for {name}"
        return True, "python.exec bulk_copy manifest verified"
