import json
import subprocess
import sys
from pathlib import Path

from koshshield.services.agent.tool_runner import DockerToolRunner

RUNNER_SCRIPT = Path(__file__).parents[3] / "tools" / "runner" / "koshshield_tool_runner.py"


def invoke_local_runner(request: dict[str, object]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, "-I", str(RUNNER_SCRIPT)],
        input=json.dumps(request).encode(),
        capture_output=True,
        check=False,
    )


def test_restricted_runner_calculates_without_eval_or_shell() -> None:
    process = invoke_local_runner(
        {"tool": "calculator", "payload": {"expression": "(50 + 25) * 2"}}
    )

    assert process.returncode == 0
    assert json.loads(process.stdout) == {
        "ok": True,
        "tool": "calculator",
        "result": {"value": "150"},
    }


def test_restricted_runner_rejects_code_and_unknown_tools() -> None:
    code = invoke_local_runner(
        {"tool": "calculator", "payload": {"expression": "__import__('os').system('id')"}}
    )
    network = invoke_local_runner(
        {"tool": "network_request", "payload": {"destination": "outside"}}
    )

    assert code.returncode == 2
    assert network.returncode == 2
    assert json.loads(code.stdout) == {"ok": False, "error": "RESTRICTED_TOOL_REJECTED"}
    assert json.loads(network.stdout) == {"ok": False, "error": "RESTRICTED_TOOL_REJECTED"}


def test_restricted_runner_generates_bounded_document_report() -> None:
    process = invoke_local_runner(
        {
            "tool": "document_report",
            "payload": {
                "document": {
                    "document_id": "9a18f91a-2bee-4ff6-883c-3c626d0c8c9d",
                    "evidence_hash": "a" * 64,
                    "filename": "procurement-report.pdf",
                    "status": "INDEXED",
                    "page_count": 3,
                    "redaction_count": 7,
                    "chunk_count": 5,
                }
            },
        }
    )

    assert process.returncode == 0
    output = json.loads(process.stdout)
    assert output["tool"] == "document_report"
    assert output["result"]["media_type"] == "text/markdown"
    assert "procurement-report.pdf" in output["result"]["content"]
    assert "Active retrieval chunks: 5" in output["result"]["content"]


def test_docker_runner_enforces_network_and_resource_boundaries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=b'{"ok":true,"tool":"calculator","result":{"value":"4"}}',
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = DockerToolRunner(image="koshshield-tool-runner:0.1.0").execute(
        tool_name="calculator",
        payload={"expression": "2 + 2"},
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--log-driver") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--user") + 1] == "65534:65534"
    assert captured["kwargs"]["timeout"] == 10
    assert result.sandbox["network_disabled"] is True
