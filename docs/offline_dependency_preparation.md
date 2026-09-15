# Offline Python dependency preparation

Stage 0D provides preparation tooling, not an installed AI runtime. It uses
operator-supplied wheels and the existing interpreter. It does not download,
install, import candidate packages, start services, or acquire model weights.

## Inspect without writing

From the repository root, inspect an existing wheel directory:

```sh
make dependency-inventory WHEELHOUSE=.temp_requirements
```

The default is `data/wheelhouse`. The selected directory must remain inside the
repository; symlinked wheels and paths escaping that root are rejected. Do not
move or delete existing files to satisfy this command automatically.

Inspection reads wheel metadata and SHA-256 hashes without extraction. It checks
filename/metadata identity, interpreter tags, `Requires-Python`, and availability
of direct requirements from `apps/api/pyproject.toml` plus the local model
providers. Direct URL requirements are rejected, including those in wheel
metadata. Existing application constraints are retained; no versions are guessed.

`BLOCKED` with exit code 1 means required wheels are missing or invalid.
`INVENTORIED` with exit code 0 means only that the inventory and direct checks
passed. It does not establish transitive resolution, native-library loading,
publisher authenticity, or inference readiness.

On 2026-09-15, inspection of `.temp_requirements` found 17 compatible wheels
totaling 180,301,742 bytes and 13 missing direct requirements, including
FlagEmbedding, Transformers, PaddleOCR, Qdrant client and database dependencies.
Transitive requirements may add further missing wheels. No real project lock
was generated from this incomplete cache.

## Explicitly resolve a complete local wheel set

After approved artifact acquisition has supplied the complete wheel set:

```sh
make prepare-dependencies WHEELHOUSE=data/wheelhouse
```

This separate, explicit command runs the installed pip resolver with
`--dry-run --ignore-installed --no-index --only-binary=:all:`. Pip configuration
and inherited index variables are disabled, caching is disabled, and Python
socket connection/name-resolution entry points are blocked in the child process.
This is not a claim of OS-level network isolation against a compromised pip.

The resolver checks transitive constraints without installing packages or
building source distributions. Its report is checked against current wheel
bytes, package identities, requested roots, selected dependency constraints and
the interpreter environment. Inconsistent or incomplete selections fail closed.

A successful preparation writes:

- `requirements-lock.txt`: exact selected versions and wheel SHA-256 hashes.
- `data/runtime/dependency-resolution.json`: local pip report and consistency evidence.

Publication creates complete individual files without overwriting differing
operator files. Identical outputs are idempotent. An interrupted two-file
publication may leave a partial bundle, but readiness rejects missing or
inconsistent evidence. Repeating an identical preparation can complete it.
There is no automatic replacement or deletion of an existing different lock.

The success status is `RESOLVED_NOT_INSTALLED`. Model provisioning remains a
separate operation, and `provision_local_runtime.py --apply` remains unimplemented.

## Evidence boundary

Readiness consumes the recorded evidence without launching pip. It rechecks
root constraints, environment, wheel contents and the lock. Changing any of
those invalidates the bundle. A comment or invented version list cannot satisfy
the check.

This is consistency evidence from local resolution, not a cryptographic
attestation of who generated it or a publisher signature. A wheel hash computed
locally does not prove trusted upstream origin. Artifact acquisition and source
verification remain separate operator responsibilities before installation.

The receipt contains local wheel paths and stays under ignored `data/`; wheel
caches remain ignored as well. Do not stage receipts, wheels, models, credentials
or operational data. A genuine reviewed dependency lock may be committed later.

## Verification

```sh
make test-api
make lint-api
.venv/bin/pytest apps/api/tests/test_runtime_dependencies.py
```

Tests use tiny synthetic wheels, execute real offline pip resolution, assert
installed distributions are unchanged, and cover conflicts, URL rejection,
tampered evidence, stale inputs, path restrictions and output preservation.
They do not prove BGE-M3, OCR, Qwen or native extension execution.

`make test` and `make lint` still include frontend gates. The separate API targets
allow accurate backend results when frontend tooling is unavailable; do not
report an unexecuted frontend gate as passed. Python lint now includes `scripts/`.

## Remaining Stage 0 work

Acquire the missing wheels and model artifacts only after approval; address the
unresolved artifact provenance and storage requirements; then install and run
the real native-PDF and scanned-PDF smoke checks separately. Until those pass,
Stage 0 and real inference remain incomplete.
