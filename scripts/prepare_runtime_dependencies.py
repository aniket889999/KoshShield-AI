"""Inspect local wheels or explicitly prepare a resolved lock. Never installs packages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from . import runtime_dependencies as dependencies
else:
    import runtime_dependencies as dependencies

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--wheelhouse", type=Path, default=Path("data/wheelhouse"))
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Run offline pip resolution and write a lock and evidence, without installing",
    )
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    wheelhouse = (root / args.wheelhouse).resolve()
    try:
        if not wheelhouse.is_relative_to(root):
            raise dependencies.DependencyPreparationError("WHEELHOUSE_PATH_UNSAFE")
        roots = dependencies.runtime_requirements(root)
        inventory = dependencies.inventory_wheels(wheelhouse)
        compatible = dependencies.compatible_wheels(inventory)
        missing = dependencies.missing_root_requirements(compatible, roots)
        result = {
            "status": "BLOCKED" if missing else "INVENTORIED",
            "wheel_count": len(inventory),
            "compatible_wheel_count": len(compatible),
            "wheel_bytes": sum(wheel.size_bytes for wheel in inventory),
            "missing_root_requirements": missing,
            "resolver_executed": False,
            "packages_installed": False,
        }
        if args.prepare and not missing:
            report = dependencies.resolve_offline(wheelhouse, roots)
            dependencies.write_resolution_bundle(root, wheelhouse, roots, report)
            dependencies.verify_dependency_evidence(
                root, root / "requirements-lock.txt"
            )
            result.update(status="RESOLVED_NOT_INSTALLED", resolver_executed=True)
        print(json.dumps(result, indent=2))
        return 1 if missing else 0
    except dependencies.DependencyPreparationError as error:
        code = error.code
    except (OSError, ValueError, KeyError, TypeError):
        code = "DEPENDENCY_PREPARATION_FAILED"
    print(
        json.dumps(
            {"status": "BLOCKED", "failure_code": code, "packages_installed": False}
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
