# Runtime Artifacts Provenance and Inventory (Stage 0C)

**Document Version**: 1.0.0  
**Retrieval & Verification Date**: 2026-09-12  
**Target Environment**: `darwin-arm64` (Apple Silicon, macOS)  
**Security Boundary**: Strictly local-first and air-gapped (`AGENTS.md`). No network access at runtime or during application startup.

---

## 1. Executive Summary & Verification Boundary

In Stage 0C, all local runtime artifact identities, versions, filenames, byte sizes, and checksums were reconciled against official upstream public metadata and release registries (Hugging Face API, GitHub Releases API, and Docker Hub Registry API).

### Critical Operational Guarantees
1. **Metadata-Only Investigation**: No artifact payloads, model weights, wheels, or container images were downloaded or pulled. Installation remains **strictly blocked**.
2. **Distinction Between Recorded Metadata and Local Verification**: An artifact's integrity status in `docs/runtime_artifacts_manifest.json` reflects whether an authoritative SHA-256 digest is established from upstream records. **No artifact has been locally verified merely because its digest is recorded**. All local filesystem targets remain empty until an operator approves an explicit, verified acquisition step.
3. **No Fabricated Checksums or Format Mismatches**:
   - For Hugging Face models, Git LFS pointer SHA-256 digests (`oid sha256:...`) are distinguished from 40-hex Git blob SHA-1 identifiers.
   - For container images, the multi-platform index digest is distinguished from the platform-specific `linux/arm64` image manifest digest.
   - HTTP ETags (MD5) and Git commit IDs are **never** labeled as published SHA-256 digests.
   - Upstream archives lacking published SHA-256 digests retain `integrity_status: "UNVERIFIED"`.
4. **Honest Reporting of Nonexistent Artifacts**: The manifest's previously guessed detection URL (`en_PP-OCRv4_det_infer.tar`) returns **HTTP 404 (Not Found)** upstream. This is reported explicitly below without silent substitution.

---

## 2. Authoritative Artifact Inventory Table

| Component / ID | Upstream Repository & Immutable Revision | Target Path / Reference | Architecture / Quantization | Exact Published Size | Integrity Digest (SHA-256) | Checksum Source / Provenance | Integrity Status |
|---|---|---|---|---|---|---|---|
| **BGE-M3** (Weights) | `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181` | `data/models/bge-m3/pytorch_model.bin` | XLM-RoBERTa dense+sparse+colbert | 2,271,145,830 bytes (~2.115 GiB) | `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38` | Official Git LFS pointer OID | **VERIFIED** (Upstream Metadata) |
| **BGE-M3** (Fast Tokenizer) | `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181` | `data/models/bge-m3/tokenizer.json` | Tokenizer JSON definition | 17,098,108 bytes (~16.306 MiB) | `21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08` | Official Git LFS pointer OID | **VERIFIED** (Upstream Metadata) |
| **BGE-M3** (SentencePiece) | `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181` | `data/models/bge-m3/sentencepiece.bpe.model` | BPE Tokenizer model | 5,069,051 bytes (~4.834 MiB) | `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865` | Official Git LFS pointer OID | **VERIFIED** (Upstream Metadata) |
| **BGE-M3** (Config) | `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181` | `data/models/bge-m3/config.json` | JSON Config | 687 bytes | *None published* | Git blob SHA-1 `e6eda1c72da8f9dc30fdd9b69c73d35af3b7a7ad` | **UNVERIFIED** (Upstream SHA-256 Absent) |
| **BGE-M3** (Tokenizer Config) | `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181` | `data/models/bge-m3/tokenizer_config.json` | JSON Config | 444 bytes | *None published* | Git blob SHA-1 `dc69ac559dcba2694012009aaa108c614541789a` | **UNVERIFIED** (Upstream SHA-256 Absent) |
| **BGE-M3** (Special Tokens) | `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181` | `data/models/bge-m3/special_tokens_map.json` | JSON Config | 964 bytes | *None published* | Git blob SHA-1 `b1879d702821e753ffe4245048eee415d54a9385` | **UNVERIFIED** (Upstream SHA-256 Absent) |
| **Qwen3-VL-4B** (LLM Backbone) | `Qwen/Qwen3-VL-4B-Instruct-GGUF` @ `1cd86afb9a95c410a6038ab3b40d8b578c892266` | `data/models/qwen3-vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf` | GGUF Q4_K_M (Metal / ARM64) | 2,497,281,664 bytes (~2.326 GiB) | `66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a` | Official Git LFS pointer OID | **VERIFIED** (Upstream Metadata) |
| **Qwen3-VL-4B** (Vision Projector) | `Qwen/Qwen3-VL-4B-Instruct-GGUF` @ `1cd86afb9a95c410a6038ab3b40d8b578c892266` | `data/models/qwen3-vl/mmproj-Qwen3VL-4B-Instruct-F16.gguf` | GGUF F16 Vision Projector | 836,180,256 bytes (~797.44 MiB) | `256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331` | Official Git LFS pointer OID | **VERIFIED** (Upstream Metadata) |
| **PaddleOCR** (Detection Archive) | `PaddlePaddle/PaddleOCR` (`en_PP-OCRv4_det_infer.tar`) | `data/models/paddleocr/det/en_PP-OCRv4_det_infer.tar` | PP-OCRv4 DB detection | 4,894,720 bytes (from candidate `ch_`) | *None published* | HTTP 404 upstream; candidate ETag/MD5 `bbc541a36ae5fe2f5e873d1e16f81c20` | **UNVERIFIED** (404 / Nonexistent URL) |
| **PaddleOCR** (Recognition Archive) | `PaddlePaddle/PaddleOCR` (`en_PP-OCRv4_rec_infer.tar`) | `data/models/paddleocr/rec/en_PP-OCRv4_rec_infer.tar` | PP-OCRv4 SVTR recognition | 10,240,000 bytes (~9.766 MiB) | *None published* | Baidu BOS ETag/MD5 `88bb7268e04b2f76b445551ce7a0b524` | **UNVERIFIED** (Upstream SHA-256 Absent) |
| **llama-server** | `ggml-org/llama.cpp` @ tag `b10809` (commit `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`) | `llama-server` (from `llama-b10809-bin-macos-arm64.tar.gz`) | darwin-arm64 (macOS Apple Silicon) | 11,123,196 bytes (~10.608 MiB) | *None published* | GitHub release b10809 assets contain no SHA-256 | **UNVERIFIED** (Upstream SHA-256 Absent) |
| **Qdrant** | `qdrant/qdrant:v1.15.4` on Docker Hub | `qdrant/qdrant:v1.15.4` | linux/arm64 container image | 65,266,145 bytes (~62.243 MiB compressed) | `cc374f58d68768be3b83a78c2d57782eb1661e5c9c63e7c495d40c76c539f469` | Docker Hub OCI manifest digest for linux/arm64 | **VERIFIED** (Upstream Registry Digest) |

---

## 3. Detailed Provenance Analysis by Component

### 3.1 BGE-M3 (`BAAI/bge-m3`)
- **Upstream Repository**: `https://huggingface.co/BAAI/bge-m3`
- **Immutable Commit**: `5617a9f61b028005a4858fdac845db406aefb181`
- **Weights File Identity Correction**:
  - The previous manifest guessed `model.safetensors` (`2,238,610,584` bytes).
  - Inspection of the official Hugging Face tree API for commit `5617a9f6` reveals that `model.safetensors` **does not exist** in the repository.
  - The official PyTorch weights file is `pytorch_model.bin` (`2,271,145,830` bytes).
  - `runtime_preflight.py` (`check_bge_m3`) already supports `pytorch_model.bin` as an authorized weight file.
  - Git LFS pointer OID publishes the exact SHA-256 digest: `b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38`.
- **Tokenizer Files**:
  - `tokenizer.json`: Size `17,098,108` bytes, LFS SHA-256 `21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08`.
  - `sentencepiece.bpe.model`: Size `5,069,051` bytes, LFS SHA-256 `cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865`.
- **Non-LFS Git Configuration Files**:
  - `config.json` (687 bytes), `tokenizer_config.json` (444 bytes), and `special_tokens_map.json` (964 bytes) are small plain-text files tracked directly in the Git tree.
  - Git tracks these files using Git blob SHA-1 object identifiers (`e6eda1c72da8f9dc30fdd9b69c73d35af3b7a7ad`, `dc69ac559dcba2694012009aaa108c614541789a`, `b1879d702821e753ffe4245048eee415d54a9385`).
  - Hugging Face does not publish a standalone SHA-256 digest for non-LFS files.
  - Per instructions (*"Do not label an ETag, Git SHA-1 or locally invented value a published SHA-256"*), these files retain `integrity_status: "UNVERIFIED"`.

### 3.2 Qwen3-VL-4B-Instruct-GGUF (`Qwen/Qwen3-VL-4B-Instruct-GGUF`)
- **Upstream Repository**: `https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF`
- **Immutable Commit**: `1cd86afb9a95c410a6038ab3b40d8b578c892266`
- **Filename Corrections**:
  - Guessed filename: `Qwen3-VL-4B-Instruct-Q4_K_M.gguf` (with hyphen). Official upstream filename: `Qwen3VL-4B-Instruct-Q4_K_M.gguf` (no hyphen between 3 and VL). Size: `2,497,281,664` bytes.
  - Guessed projector: `mmproj-Qwen3-VL-4B-Instruct-f16.gguf` (with hyphen, lowercase f). Official upstream filename: `mmproj-Qwen3VL-4B-Instruct-F16.gguf` (capital F16, no hyphen). Size: `836,180,256` bytes.
- **Integrity**:
  - Both files are tracked via Git LFS. Their LFS pointer OIDs are authoritative published SHA-256 digests:
    - Base model: `66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a`
    - Vision projector: `256f3a43bd4205ffef48d6b92715e1e70b5b0e9aef06522584967513a9985331`

### 3.3 PaddleOCR (`PaddlePaddle/PaddleOCR`)
- **Nonexistent Detection Model**:
  - Current manifest specifies `https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_det_infer.tar`.
  - HTTP verification returns **HTTP 404 (Not Found)**.
  - Examination of the official PaddleOCR codebase (`paddleocr/paddleocr.py`) reveals that PP-OCRv4 detection is language-agnostic and only published under the Chinese model hierarchy: `https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar` (size `4,894,720` bytes).
  - This nonexistent artifact is retained with `integrity_status: "UNVERIFIED"` and documented here rather than silently substituted.
- **Recognition Model**:
  - `https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar` returns **HTTP 200 OK** (size `10,240,000` bytes).
- **Checksum Provenance**:
  - Baidu BOS publishes HTTP ETags which are 32-character hexadecimal MD5 hashes (`bbc541a36ae5fe2f5e873d1e16f81c20` and `88bb7268e04b2f76b445551ce7a0b524`).
  - No SHA-256 digests are published by the Baidu BOS hosting infrastructure.
  - Retained as `UNVERIFIED` fail-closed.

### 3.4 llama.cpp (`ggml-org/llama.cpp`)
- **Pinned Release**: `v0.4.0` (tag `b10809`, commit `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`)
- **Target Asset**: `llama-b10809-bin-macos-arm64.tar.gz` (exact published size `11,123,196` bytes).
- **Checksum Reality**:
  - GitHub release assets for `b10809` contain no `.sha256` or `SHA256SUMS` files.
  - The release notes reference GitHub Attestations, but no static SHA-256 string is published on the release page.
  - Retained as `UNVERIFIED` fail-closed.

### 3.5 Qdrant Container Image (`qdrant/qdrant:v1.15.4`)
- **Multi-Platform Index Digest**:
  - `sha256:6ac4807063bbecddca0250bfbcff52acf18c22263b904d12919349e6d0a408f1`
  - Published in Docker Hub response header `docker-content-digest` and matches the SHA-256 of the OCI index JSON payload.
- **Selected Platform (`linux/arm64`) Image Manifest Digest**:
  - `sha256:cc374f58d68768be3b83a78c2d57782eb1661e5c9c63e7c495d40c76c539f469`
  - Referenced in the OCI index for platform `os: linux`, `architecture: arm64`.
- **Exact Published Size**:
  - Sum of 14 compressed OCI layer blobs + image config = `65,266,145` bytes (~62.243 MiB).
- **Integrity Status**: `VERIFIED` (recorded upstream manifest digest).

---

## 4. Capacity Assessment & Footprint Reconciliation

All byte totals are calculated using exact integer arithmetic and 1024-based binary units (GiB):

| Category | Integer Bytes | GiB Equivalent | Notes |
|---|---|---|---|
| **BGE-M3 Artifacts** | 2,293,315,084 | 2.136 GiB | Exact published sizes of weights, tokenizers, and configs |
| **Qwen3-VL Artifacts** | 3,333,461,920 | 3.104 GiB | Q4_K_M GGUF (`2,497,281,664`) + F16 mmproj (`836,180,256`) |
| **PaddleOCR Archives** | 15,134,720 | 0.014 GiB | Detection candidate (`4,894,720`) + Recognition (`10,240,000`) |
| **llama.cpp macOS Asset** | 11,123,196 | 0.010 GiB | Exact published `tar.gz` archive size |
| **Qdrant Container Layers** | 65,266,145 | 0.061 GiB | Published compressed layer payloads |
| **Net Published Artifact Total** | **5,718,301,065** | **5.326 GiB** | Sum of all required runtime payloads |
| **Python Dependencies (Installed)** | 2,684,354,560 | 2.500 GiB | [ESTIMATED] PyTorch, PaddlePaddle, FlagEmbedding, etc. |
| **Temporary Extraction Space** | 2,147,483,648 | 2.000 GiB | [ESTIMATED] Extraction buffer for archives and wheels |
| **Docker Storage Growth** | 1,610,612,736 | 1.500 GiB | [ESTIMATED] Uncompressed image layers and container state |
| **Runtime Caches** | 1,073,741,824 | 1.000 GiB | [ESTIMATED] HuggingFace / PaddleOCR execution cache |
| **Total Estimated Overhead** | **7,516,192,768** | **7.000 GiB** | Integer sum of estimated operational overhead |
| **Total Peak Footprint** | **13,234,493,833** | **12.326 GiB** | Published artifacts + Estimated overhead |
| **Mandatory Headroom** | **10,737,418,240** | **10.000 GiB** | Enforced local policy minimum |
| **Total Required Capacity** | **23,971,912,073** | **22.326 GiB** | Peak footprint + Headroom |
| **Current Available Space** | **15,822,802,944** | **14.736 GiB** | Statfs free bytes on target volume |
| **Capacity Deficit** | **8,149,109,129** | **7.589 GiB** | **INSUFFICIENT SPACE** (Blocks provisioning) |

---

## 5. Active Blockers & Next Concrete Prerequisites

Running `python3 scripts/provision_local_runtime.py --dry-run` reports **BLOCKED** with 8 active blockers:

1. **`ARTIFACT_INTEGRITY_UNVERIFIED` (6 items)**:
   - `bge-m3/config.json`, `bge-m3/tokenizer_config.json`, `bge-m3/special_tokens_map.json`: Upstream Git repository does not publish SHA-256 digests for plain-text Git blobs.
   - `paddleocr-v4-en/en_PP-OCRv4_det_infer.tar`: Guessed URL returns HTTP 404 upstream; upstream Baidu BOS does not publish SHA-256.
   - `paddleocr-v4-en/en_PP-OCRv4_rec_infer.tar`: Baidu BOS does not publish SHA-256.
   - `llama-server`: GitHub release `b10809` assets do not publish SHA-256 digests.
2. **`DEPENDENCY_LOCK_INCOMPLETE`**: Authoritative hash-pinned `requirements-lock.txt` for `darwin-arm64-cp312` is not yet generated.
3. **`CAPACITY_INSUFFICIENT`**: Available disk space on the primary filesystem is `14.74 GiB`, leaving an estimated deficit of `7.59 GiB` against the `22.33 GiB` required threshold (including 10 GiB mandatory headroom).

### Next Concrete Prerequisite
Before requesting download or installation approval:
1. Formally resolve the nonexistent detection model reference (decide whether to adopt `ch_PP-OCRv4_det_infer.tar` as the official upstream multilingual detection archive).
2. Authorize an operator hash-verification step to calculate SHA-256 digests for the archives lacking upstream published checksums (Baidu BOS and llama.cpp).
3. Generate a genuine, verified `requirements-lock.txt` resolving all transitive dependencies for `darwin-arm64-cp312`.
4. Reconcile disk capacity by freeing at least `8.0 GiB` on the local volume or mounting an external volume for `--model-dir`.
