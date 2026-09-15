"""Offline inspection of operator-supplied Python wheels. Never extracts or imports them."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import BadZipFile, ZipFile

import tomllib
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

MAX_METADATA_BYTES = 1024 * 1024
EVIDENCE_PATH = Path("data/runtime/dependency-resolution.json")
MODEL_REQUIREMENTS = (
    "FlagEmbedding",
    "transformers",
    "torch",
    "paddlepaddle",
    "paddleocr",
)

OFFLINE_PIP_BOOTSTRAP = """import runpy, socket, sys
def deny_network(*args, **kwargs):
    raise OSError("OFFLINE_NETWORK_FORBIDDEN")
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network
socket.getaddrinfo = deny_network
sys.argv = ["pip", *sys.argv[1:]]
runpy.run_module("pip", run_name="__main__")
"""


class DependencyPreparationError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WheelInfo:
    filename: str
    name: str
    version: str
    size_bytes: int
    sha256: str
    requires_python: str
    requirements: tuple[str, ...]
    tags: frozenset[str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wheel(path: Path, wheelhouse: Path) -> WheelInfo:
    root = wheelhouse.resolve()
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or not path.is_file()
    ):
        raise DependencyPreparationError("WHEEL_PATH_UNSAFE")
    try:
        name, version, _, tags = parse_wheel_filename(path.name)
        with ZipFile(path) as archive:
            entries = archive.infolist()
            metadata = [
                entry
                for entry in entries
                if entry.filename.count("/") == 1
                and entry.filename.endswith(".dist-info/METADATA")
            ]
            if len(metadata) != 1 or metadata[0].file_size > MAX_METADATA_BYTES:
                raise DependencyPreparationError("WHEEL_METADATA_INVALID")
            message = BytesParser(policy=policy.default).parsebytes(
                archive.read(metadata[0])
            )
            if (
                len(message.get_all("Name", [])) != 1
                or len(message.get_all("Version", [])) != 1
            ):
                raise DependencyPreparationError("WHEEL_METADATA_INVALID")
            if (
                canonicalize_name(str(message["Name"])) != name
                or Version(str(message["Version"])) != version
            ):
                raise DependencyPreparationError("WHEEL_IDENTITY_MISMATCH")
            return WheelInfo(
                filename=path.name,
                name=name,
                version=str(version),
                size_bytes=path.stat().st_size,
                sha256=file_sha256(path),
                requires_python=str(message.get("Requires-Python", "")),
                requirements=tuple(
                    str(item) for item in message.get_all("Requires-Dist", [])
                ),
                tags=frozenset(str(tag) for tag in tags),
            )
    except DependencyPreparationError:
        raise
    except (ValueError, OSError, BadZipFile, KeyError, RuntimeError) as exc:
        raise DependencyPreparationError("WHEEL_METADATA_INVALID") from exc


def inventory_wheels(wheelhouse: Path) -> list[WheelInfo]:
    if not wheelhouse.is_dir():
        raise DependencyPreparationError("WHEELHOUSE_MISSING")
    paths = sorted(wheelhouse.glob("*.whl"))
    if not paths:
        raise DependencyPreparationError("WHEELHOUSE_EMPTY")
    return [inspect_wheel(path, wheelhouse) for path in paths]


def local_requirement(text: str) -> Requirement:
    try:
        requirement = Requirement(text)
    except InvalidRequirement as exc:
        raise DependencyPreparationError("REQUIREMENT_INVALID") from exc
    if requirement.url is not None:
        raise DependencyPreparationError("DIRECT_REFERENCE_FORBIDDEN")
    return requirement


def runtime_requirements(repo_root: Path) -> tuple[str, ...]:
    project = tomllib.loads(
        (repo_root / "apps/api/pyproject.toml").read_text(encoding="utf-8")
    )
    dependencies = project["project"]["dependencies"]
    if not isinstance(dependencies, list) or not all(
        isinstance(item, str) for item in dependencies
    ):
        raise DependencyPreparationError("PROJECT_DEPENDENCIES_INVALID")
    return tuple(
        sorted(
            {
                str(local_requirement(item))
                for item in [*dependencies, *MODEL_REQUIREMENTS]
            }
        )
    )


def compatible_wheels(
    wheels: list[WheelInfo],
    supported_tags: set[str] | None = None,
    python_version: str | None = None,
) -> list[WheelInfo]:
    tags = (
        supported_tags
        if supported_tags is not None
        else {str(tag) for tag in sys_tags()}
    )
    version = python_version or platform.python_version()
    compatible = []
    for wheel in wheels:
        for requirement in wheel.requirements:
            local_requirement(requirement)
        try:
            python_compatible = SpecifierSet(wheel.requires_python).contains(
                version, prereleases=True
            )
        except InvalidSpecifier as exc:
            raise DependencyPreparationError("REQUIRES_PYTHON_INVALID") from exc
        if wheel.tags & tags and python_compatible:
            compatible.append(wheel)
    return compatible


def missing_root_requirements(
    wheels: list[WheelInfo], roots: tuple[str, ...]
) -> list[str]:
    environment = {**default_environment(), "extra": ""}
    missing = []
    for text in roots:
        requirement = local_requirement(text)
        if requirement.marker and not requirement.marker.evaluate(environment):
            continue
        if not any(
            wheel.name == canonicalize_name(requirement.name)
            and requirement.specifier.contains(wheel.version, prereleases=True)
            for wheel in wheels
        ):
            missing.append(str(requirement))
    return sorted(missing)


def resolve_offline(
    wheelhouse: Path, roots: tuple[str, ...], timeout: int = 120
) -> dict:
    """Run the installed pip resolver against local wheels, without installing anything."""
    if not roots:
        raise DependencyPreparationError("ROOT_REQUIREMENTS_EMPTY")
    roots = tuple(sorted({str(local_requirement(item)) for item in roots}))
    wheels = inventory_wheels(wheelhouse)
    candidates = compatible_wheels(wheels)
    if missing_root_requirements(candidates, roots):
        raise DependencyPreparationError("ROOT_WHEELS_MISSING")
    environment = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "TMPDIR", "SYSTEMROOT", "LANG")
        if key in os.environ
    }
    environment["PIP_CONFIG_FILE"] = os.devnull
    with tempfile.TemporaryDirectory(prefix="koshshield-resolve-") as scratch:
        report_path = Path(scratch) / "report.json"
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            OFFLINE_PIP_BOOTSTRAP,
            "--isolated",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--no-index",
            "--only-binary=:all:",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-input",
            "--find-links",
            str(wheelhouse.resolve()),
            "--report",
            str(report_path),
            "--",
            *roots,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=environment,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DependencyPreparationError("OFFLINE_RESOLUTION_TIMEOUT") from exc
        if result.returncode != 0 or not report_path.is_file():
            raise DependencyPreparationError("OFFLINE_RESOLUTION_FAILED")
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise DependencyPreparationError("RESOLVER_REPORT_INVALID") from exc
    if (
        not isinstance(report, dict)
        or report.get("version") != "1"
        or not isinstance(report.get("install"), list)
    ):
        raise DependencyPreparationError("RESOLVER_REPORT_INVALID")
    return report


def validate_selected_dependencies(
    selected: list[WheelInfo], roots: tuple[str, ...]
) -> None:
    """Check a fixed pip selection against wheel metadata; this does not select versions."""
    environment = default_environment()
    by_name = {wheel.name: wheel for wheel in selected}
    pending = [local_requirement(item) for item in roots]
    pending = [
        req
        for req in pending
        if not req.marker or req.marker.evaluate({**environment, "extra": ""})
    ]
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    visited = set()
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        key = (name, str(requirement.specifier), tuple(sorted(requirement.extras)))
        if key in seen:
            continue
        seen.add(key)
        wheel = by_name.get(name)
        if wheel is None or not requirement.specifier.contains(
            wheel.version, prereleases=True
        ):
            raise DependencyPreparationError("DEPENDENCY_SELECTION_INCOMPLETE")
        visited.add(name)
        for text in wheel.requirements:
            child = local_requirement(text)
            if not child.marker or any(
                child.marker.evaluate({**environment, "extra": extra})
                for extra in requirement.extras or {""}
            ):
                pending.append(child)
    if visited != set(by_name):
        raise DependencyPreparationError("DEPENDENCY_SELECTION_UNEXPECTED")


def build_resolution_evidence(
    wheelhouse: Path, roots: tuple[str, ...], report: dict
) -> tuple[str, dict]:
    """Bind an actual pip report to current wheel bytes, roots and interpreter environment."""
    normalized_roots = tuple(sorted({str(local_requirement(item)) for item in roots}))
    if not normalized_roots:
        raise DependencyPreparationError("ROOT_REQUIREMENTS_EMPTY")
    inventory = {
        wheel.filename: wheel
        for wheel in compatible_wheels(inventory_wheels(wheelhouse))
    }
    selected = []
    seen = set()
    try:
        if report["version"] != "1" or not isinstance(report["pip_version"], str):
            raise DependencyPreparationError("RESOLVER_REPORT_INVALID")
        if report["environment"] != default_environment():
            raise DependencyPreparationError("RESOLVER_ENVIRONMENT_MISMATCH")
        for item in report["install"]:
            download = item["download_info"]
            url = urlsplit(download["url"])
            if url.scheme != "file" or url.netloc or url.query or url.fragment:
                raise DependencyPreparationError("RESOLVER_SOURCE_UNSAFE")
            source = Path(unquote(url.path))
            wheel = inventory.get(source.name)
            if (
                wheel is None
                or source.resolve() != (wheelhouse / wheel.filename).resolve()
            ):
                raise DependencyPreparationError("RESOLVER_SOURCE_UNSAFE")
            if (
                download["archive_info"]["hashes"]["sha256"].lower() != wheel.sha256
                or canonicalize_name(item["metadata"]["name"]) != wheel.name
                or Version(item["metadata"]["version"]) != Version(wheel.version)
                or wheel.name in seen
            ):
                raise DependencyPreparationError("RESOLVER_WHEEL_MISMATCH")
            selected.append(wheel)
            seen.add(wheel.name)
    except (KeyError, TypeError, AttributeError) as exc:
        raise DependencyPreparationError("RESOLVER_REPORT_INVALID") from exc
    validate_selected_dependencies(selected, normalized_roots)
    selected.sort(key=lambda wheel: wheel.name)
    lock = "".join(
        f"{wheel.name}=={wheel.version} --hash=sha256:{wheel.sha256}\n"
        for wheel in selected
    )
    evidence = {
        "schema_version": 1,
        "source": "pip-dry-run-no-index",
        "roots": list(normalized_roots),
        "environment": default_environment(),
        "lock_sha256": hashlib.sha256(lock.encode()).hexdigest(),
        "wheels": [
            {
                "filename": wheel.filename,
                "name": wheel.name,
                "version": wheel.version,
                "size_bytes": wheel.size_bytes,
                "sha256": wheel.sha256,
            }
            for wheel in selected
        ],
        "pip_report": report,
    }
    return lock, evidence


def verify_dependency_evidence(repo_root: Path, lock_file: Path) -> None:
    """Read-only consistency check of recorded local resolution, not publisher authentication."""
    try:
        evidence_path = repo_root / EVIDENCE_PATH
        if (
            evidence_path.stat().st_size > 8 * MAX_METADATA_BYTES
            or lock_file.stat().st_size > MAX_METADATA_BYTES
        ):
            raise DependencyPreparationError("DEPENDENCY_EVIDENCE_TOO_LARGE")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if (
            type(evidence["schema_version"]) is not int
            or evidence["schema_version"] != 1
            or evidence["source"] != "pip-dry-run-no-index"
        ):
            raise DependencyPreparationError("DEPENDENCY_EVIDENCE_INVALID")
        relative = Path(evidence["wheelhouse"])
        if relative.is_absolute() or ".." in relative.parts:
            raise DependencyPreparationError("WHEELHOUSE_PATH_UNSAFE")
        wheelhouse = (repo_root / relative).resolve()
        if not wheelhouse.is_relative_to(repo_root.resolve()):
            raise DependencyPreparationError("WHEELHOUSE_PATH_UNSAFE")
        roots = runtime_requirements(repo_root)
        lock, rebuilt = build_resolution_evidence(
            wheelhouse, roots, evidence["pip_report"]
        )
        if any(evidence.get(key) != value for key, value in rebuilt.items()):
            raise DependencyPreparationError("DEPENDENCY_EVIDENCE_STALE")
        if lock_file.read_text(encoding="utf-8") != lock:
            raise DependencyPreparationError("DEPENDENCY_LOCK_MISMATCH")
    except DependencyPreparationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise DependencyPreparationError("DEPENDENCY_EVIDENCE_INVALID") from exc
