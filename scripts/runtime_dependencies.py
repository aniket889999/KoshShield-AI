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
from zipfile import BadZipFile, ZipFile

import tomllib
from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

MAX_METADATA_BYTES = 1024 * 1024
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
