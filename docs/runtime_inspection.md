# Runtime inspection and explicit probes

Stage 0 is still incomplete. File layout, a reachable server and passing unit tests
are not evidence of successful real-model inference. This checkpoint improves
diagnostics without acquiring models, installing packages or starting services.

## Commands and effects

Run from the repository root using the existing `.venv`:

| Command | Effects |
| --- | --- |
| `make runtime-preflight` | Reads local artifact headers/configuration and installed package metadata. No service calls, model loading or scratch storage. |
| `.venv/bin/python -B -m koshshield.runtime_preflight --probe-services` | Also queries configured local llama.cpp and Qdrant. Does not start services, invoke generation or modify collections/indexes. |
| `.venv/bin/python -B -m koshshield.runtime_preflight --probe-storage` | Also creates an owned temporary SQLite database and encrypted vault, disposes the connection and removes only its scratch directory. Does not test the deployment database or vault. |
| `make runtime-probes` | Enables both optional probe flags. |
| `make provision-runtime` | Separate manifest, dependency-resolution evidence and storage-capacity assessment. It does not acquire assets. |
| `make smoke-local` | Separate execution harness with real adapters and isolated synthetic resources. May invoke configured models; not an inspection command. |

Python `-B` prevents interpreter bytecode writes. Direct Python invocations without
it can create `__pycache__` files even though inspection code does not write data.
Local service URLs are not proof of host-level network isolation; DNS, proxies,
process permissions and outbound traffic require separate deployment validation.

## Interpreting the JSON report

- `READY` applies only to the checks actually described. It is not an approval to
  provision, deploy or process confidential records.
- Default inspection returns `NOT_READY` (exit 1) because its service/storage
  checks are deliberately `NOT_EXECUTED`. This is expected, not a passed runtime.
- `missing_categories` includes failed and intentionally unexecuted checks.
- `service_probes_enabled` and `storage_probe_enabled` show the requested scope.
- `generation_executed` and `runtime_verified` remain `false`, even if all
  prerequisite probes pass. Only the separate smoke/evaluation runs test inference.
- `metadata.validation_scope` distinguishes headers, bundle layout, module
  discovery and temporary storage. `integrity_verified: false` is intentional:
  these checks do not calculate model digests or authenticate a publisher.
- `metadata.failure_code` is safe to share. Raw exception strings and paths are
  not included. One failed check does not prevent independent checks completing.

The CLI emits JSON and exits 0 only when every prerequisite check passes. Invalid
configuration emits `CONFIGURATION_INVALID` and exits 1. Invalid CLI arguments
use argparse's normal usage message and exit 2.

## Supported local artifact layouts

### BGE-M3

The supported unsharded FlagEmbedding bundle includes:

```text
config.json                  # bounded JSON; positive integer hidden_size
tokenizer.json               # bounded JSON; tokenizer model and vocabulary
tokenizer_config.json
special_tokens_map.json
sentencepiece.bpe.model
pytorch_model.bin            # or model.safetensors; ONNX-only is unsupported
sparse_linear.pt
colbert_linear.pt
```

Both trained heads are required by the inspected FlagEmbedding loader: if either
is absent it can initialize heads rather than load trained parameters. The shared
bundle check now blocks that incomplete layout before provider initialization.
It does not deserialize pickle weights or prove the files contain correct models.
Sharded weight layouts are not supported by this inspection contract.

**Manifest follow-up:** the current provisioning manifest does not yet include
`sparse_linear.pt` and `colbert_linear.pt`. Add their exact sizes and independently
checked digest provenance at the pinned revision before approving acquisition;
do not infer them from names or mark them locally verified without actual bytes.
The stricter runtime gate intentionally remains blocked on an incomplete bundle.

### Qwen GGUF and projector

Both paths must identify distinct regular files, not final-component symlinks or
two hardlinks to the same inode. Inspection reads only the 24-byte header, checks
little-endian GGUF v2/v3 magic/version/count bounds and rejects empty/truncated
files. It does not parse tensors, authenticate Qwen identity, match a projector
to a model, or establish llama.cpp compatibility. A plausible header can still
belong to an invalid model. Use manifest hashes and a real generation test too.

### PaddleOCR

The current adapter uses the PaddleOCR 2.x API. Package discovery requires both
`paddleocr` and `paddle`, and installed PaddleOCR metadata must identify major
version 2. This is a necessary API check, not proof of Paddle/Python/CPU binary
compatibility. Do not automatically downgrade or install anything on failure.
Offline dependency preparation now constrains `paddleocr>=2.7,<3` to avoid
selecting the unsupported 3.x API. This does not pin a tested Paddle binary
combination; the full compatible lock and native smoke tests are still required.

Each detection/recognition directory, and any explicitly configured classifier
directory, must contain a matching nonempty pair:

- `model.pdmodel` and `model.pdiparams`; or
- `inference.pdmodel` and `inference.pdiparams`.

Empty folders, mixed prefixes and missing classifier files fail. PaddleOCR 3.x
requires a separately implemented/tested adapter, not a changed version label.

## Service and failure checks

Qdrant inspection obtains the expected dimension from the bounded local embedding
configuration, not a 1024 fallback. It checks dense/sparse vector names, cosine
distance and all required payload index types. Missing collections or indexes
remain failures; diagnostics never create or repair them. Initial collection
creation stays with the existing indexing workflow, while smoke uses owned
scratch collections. The probe closes its own client, not a singleton provider.

Common codes and actions:

| Code | Action |
| --- | --- |
| `PROBE_NOT_REQUESTED` | Explicitly enable the relevant probe after reviewing its effects. |
| `ARTIFACT_MISSING`, `OCR_INFERENCE_FILES_MISSING` | Check the approved local bundle and configured paths. |
| `ARTIFACT_EMPTY`, `GGUF_HEADER_TRUNCATED` | Investigate an incomplete transfer; do not try loading it. |
| `EMBEDDING_DIMENSION_INVALID` | Correct the local model configuration; do not guess dimensions. |
| `OCR_API_VERSION_UNSUPPORTED` | Resolve adapter/version compatibility during approved provisioning. |
| `QDRANT_SCHEMA_MISMATCH` | Inspect the collection through an authorized maintenance workflow; no auto-recreation. |
| `LLAMA_CONTRACT_MISMATCH` | Compare configured and actual build/alias/vision capabilities. |
| `CHECK_FAILED`, `SCRATCH_STORAGE_FAILED` | Diagnose local access/resource failures without sharing sensitive exception output. |

## Verification and remaining work

Tests cover malformed/empty model files, trained-head absence, invalid dimensions,
OCR layout/version mismatch, read-only Qdrant checks, independent failure reporting
and CLI behavior under a no-network/no-write Python audit guard. These are
synthetic contract tests, not real-model or kernel isolation evidence.

Still required: approved artifact acquisition and provenance, sufficient measured
disk capacity, a complete compatible dependency bundle, actual native-library
loading, live llama.cpp/Qdrant/OCR tests and the held-out procurement evaluation.
No benchmark improvement or deployment readiness is claimed by this checkpoint.

### Verification snapshot (2026-09-16)

- Backend: `pytest apps/api/tests -rs` -> 341 passed, 3 skipped.
- Skips: local llama.cpp offline, Docker Qdrant absent, real BGE-M3 weights absent.
- Backend/scripts Ruff lint and format checks passed (107 files).
- Existing frontend toolchain: 5 tests passed; ESLint and production build passed.
- Default CLI: `NOT_READY`; artifact paths unconfigured; service/storage probes
  correctly reported `NOT_EXECUTED`.
- Provisioning: `BLOCKED` by unverified digests, incomplete dependency lock and
  insufficient capacity. The inspected snapshot had 15.88 GiB free against a
  22.33 GiB requirement, before adding the missing BGE-M3 heads. Re-measure at
  provisioning time; this is not a permanent capacity figure.
- No model/package acquisition, installation or service startup was performed.

Implementation references: [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md),
[FlagEmbedding M3 loader](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/finetune/embedder/encoder_only/m3/runner.py),
[PaddleOCR 2.7 inference loader](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/tools/infer/utility.py).
Recheck the actual pinned dependency versions during provisioning.
