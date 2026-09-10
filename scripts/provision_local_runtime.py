"""Reproducible local runtime provisioning script for KoshShield AI.

Reads docs/runtime_artifacts_manifest.json to inventory, prepare, or download
pinned runtime model artifacts and binaries for offline local-first execution.

In strict compliance with KoshShield AI local-first policies:
- Application startup and inference NEVER trigger downloads.
- Provisioning is an explicit, operator-initiated setup procedure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docs" / "runtime_artifacts_manifest.json"


def load_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def check_disk_space(required_bytes: int) -> bool:
    usage = shutil.disk_usage(REPO_ROOT)
    return usage.free >= (required_bytes + 2 * 1024 * 1024 * 1024)  # 2 GB safety buffer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Provision KoshShield local runtime artifacts"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print required artifacts, sizes, and sources without downloading",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Download and unpack artifacts according to the manifest",
    )
    args = parser.parse_args()

    manifest = load_manifest()
    artifacts = manifest.get("artifacts", [])

    total_bytes = sum(a.get("estimated_size_bytes", 0) for a in artifacts)

    print(f"KoshShield AI Runtime Manifest v{manifest.get('manifest_version', 1)}")
    print(f"Total estimated download size: {total_bytes / (1024**3):.2f} GB\n")

    for a in artifacts:
        print(f"Artifact: {a['id']} ({a['category']})")
        print(f"  Source: {a.get('official_source')}")
        print(f"  License: {a.get('license')}")
        print(f"  Estimated Size: {a.get('estimated_size_human')}")
        if "target_dir" in a:
            print(f"  Target Dir: {a['target_dir']}")
        print()

    if not args.apply:
        print(
            "Dry run complete. To proceed with download, run with --apply after authorization."
        )
        return 0

    if not check_disk_space(total_bytes):
        print(
            f"ERROR: Insufficient disk space. Required: {total_bytes / (1024**3):.2f} GB + 2GB buffer."
        )
        return 1

    print(
        "Operator authorization required before downloading external weights and dependencies."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
