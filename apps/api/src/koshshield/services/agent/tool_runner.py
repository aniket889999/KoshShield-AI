import hashlib
import json
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol


class ToolRunnerError(RuntimeError):
    code = "TOOL_RUNNER_ERROR"


class ToolRunnerUnavailableError(ToolRunnerError):
    code = "TOOL_RUNNER_UNAVAILABLE"


class ToolExecutionError(ToolRunnerError):
    code = "TOOL_EXECUTION_FAILED"


class ToolOutputVerificationError(ToolRunnerError):
    code = "TOOL_OUTPUT_INVALID"


@dataclass(frozen=True)
class ToolRunnerResult:
    payload: dict[str, object]
    output_hash: str
    sandbox: dict[str, object]


class ToolRunner(Protocol):
    def execute(self, *, tool_name: str, payload: dict[str, object]) -> ToolRunnerResult: ...


class DockerToolRunner:
    def __init__(
        self,
        *,
        image: str,
        timeout_seconds: int = 10,
        max_output_bytes: int = 64 * 1024,
        docker_binary: str = "docker",
    ) -> None:
        self.image = image
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.docker_binary = docker_binary

    def execute(self, *, tool_name: str, payload: dict[str, object]) -> ToolRunnerResult:
        request_bytes = json.dumps(
            {"tool": tool_name, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        if len(request_bytes) > 16 * 1024:
            raise ToolExecutionError("tool input exceeds the sandbox limit")

        command = [
            self.docker_binary,
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--log-driver",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "128m",
            "--cpus",
            "0.50",
            "--user",
            "65534:65534",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--interactive",
            self.image,
        ]
        try:
            process = subprocess.run(
                command,
                input=request_bytes,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as err:
            raise ToolRunnerUnavailableError("Docker CLI is unavailable") from err
        except subprocess.TimeoutExpired as err:
            raise ToolExecutionError("sandbox execution timed out") from err
        except OSError as err:
            raise ToolRunnerUnavailableError("Docker runtime is unavailable") from err

        if process.returncode != 0:
            raise ToolExecutionError("sandbox rejected or failed the tool execution")
        if len(process.stdout) > self.max_output_bytes:
            raise ToolOutputVerificationError("sandbox output exceeds the verification limit")

        try:
            output = json.loads(process.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise ToolOutputVerificationError("sandbox returned invalid JSON") from err
        if not isinstance(output, dict):
            raise ToolOutputVerificationError("sandbox output must be a JSON object")
        self._verify_output(tool_name=tool_name, output=output)

        canonical = json.dumps(output, sort_keys=True, separators=(",", ":"))
        return ToolRunnerResult(
            payload=output,
            output_hash=hashlib.sha256(canonical.encode()).hexdigest(),
            sandbox={
                "network_disabled": True,
                "read_only_root": True,
                "capabilities_dropped": True,
                "image": self.image,
                "timeout_seconds": self.timeout_seconds,
            },
        )

    @staticmethod
    def _verify_output(*, tool_name: str, output: dict[str, object]) -> None:
        if output.get("ok") is not True or output.get("tool") != tool_name:
            raise ToolOutputVerificationError("sandbox output identity check failed")
        result = output.get("result")
        if not isinstance(result, dict):
            raise ToolOutputVerificationError("sandbox result is missing")

        if tool_name == "calculator":
            value = result.get("value")
            if not isinstance(value, str) or not value or len(value) > 128:
                raise ToolOutputVerificationError("calculator result is invalid")
            try:
                numeric_value = Decimal(value)
            except InvalidOperation as err:
                raise ToolOutputVerificationError("calculator result is not numeric") from err
            if not numeric_value.is_finite() or abs(numeric_value) > Decimal("1e18"):
                raise ToolOutputVerificationError("calculator result exceeds the allowed range")
            return

        if tool_name == "document_report":
            content = result.get("content")
            if (
                not isinstance(content, str)
                or not content
                or len(content.encode()) > 32 * 1024
                or any(ord(char) < 32 and char not in "\n\t" for char in content)
                or result.get("media_type") != "text/markdown"
            ):
                raise ToolOutputVerificationError("document report result is invalid")
            return

        raise ToolOutputVerificationError("sandbox returned an unknown tool result")
