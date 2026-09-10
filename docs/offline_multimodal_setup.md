# Offline Multimodal Setup: Qwen3-VL-4B with llama.cpp

This guide details the manual, air-gapped setup for local multimodal retrieval-augmented generation (RAG) in KoshShield AI using Qwen3-VL-4B and `llama.cpp` (pinned supported version: **`llama.cpp build b4600`**).

In compliance with KoshShield AI's local-first security policy (`AGENTS.md`):
- **Never** download model weights or projector files at application runtime.
- **Never** contact Hugging Face, remote model hubs, or external endpoints.
- **Never** commit model weights or GGUF files to Git.

---

## 1. Required Artifacts

Acquire the following quantized GGUF artifacts via a verified offline transfer (e.g., secure internal artifact repository or encrypted media):

| Artifact | File Name | Description |
|---|---|---|
| **Base Model** | `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | 4-bit quantized Qwen3-VL-4B instruction-tuned text backbone |
| **Multimodal Projector** | `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf` | 8-bit quantized vision encoder projector for visual tokens |

Store these files in a dedicated local directory outside the repository (for example, `/opt/models/qwen3-vl/` or `~/.cache/koshshield/models/`).

---

## 2. Integrity Verification (Operator Signed Manifest)

Integrity verification requires a trusted, operator-supplied signed SHA-256 manifest. Checksums must never be invented or assumed.

Operators must verify the SHA-256 digests of downloaded model artifacts against their organization's cryptographically signed manifest prior to starting the inference server:

```bash
# Generate SHA-256 checksums of local artifact files
shasum -a 256 Qwen3VL-4B-Instruct-Q4_K_M.gguf
shasum -a 256 mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf

# Verify against the operator's signed release manifest
gpg --verify SHA256SUMS.sig SHA256SUMS
shasum -a 256 -c SHA256SUMS
```

Do not proceed if any digest fails validation against the authoritative signed manifest.

---

## 3. Starting the Local `llama-server`

KoshShield AI requires `llama-server` from **`llama.cpp build b4600`** running locally on loopback (`127.0.0.1` or `localhost`) or an explicit container service name (`llama-server`).

### Command Line Invocation

```bash
llama-server \
  --model /opt/models/qwen3-vl/Qwen3VL-4B-Instruct-Q4_K_M.gguf \
  --mmproj /opt/models/qwen3-vl/mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --alias Qwen3VL-4B-Instruct \
  --ctx-size 8192 \
  --n-gpu-layers 33 \
  --temperature 0.0
```

### Critical Server Parameters
- Supported build: `llama.cpp build b4600`
- `--host 127.0.0.1`: Bind strictly to loopback to prevent external network access.
- `--alias Qwen3VL-4B-Instruct`: Match the model ID configured in `KOSHSHIELD_LLAMA_CPP_MODEL_ID` exactly.
- `--mmproj`: Enables vision embedding capabilities; the server must report `architecture.input_modalities` containing both `text` and `image` via `/v1/models`.
- `--temperature 0.0`: Deterministic responses for grounded evidence verification.

---

## 4. KoshShield AI Configuration

Enable multimodal answering in `apps/api/.env` or process environment:

```env
# Enable experimental multimodal answering
KOSHSHIELD_ENABLE_MULTIMODAL_ANSWERING=true

# Strict local loopback URL (SSRF protection rejects remote hosts, single labels, and redirects)
KOSHSHIELD_LLAMA_BASE_URL=http://127.0.0.1:8080/v1

# Configured model identifier checked against /v1/models (must match exactly)
KOSHSHIELD_LLAMA_CPP_MODEL_ID=Qwen3VL-4B-Instruct

# Resource bounds
KOSHSHIELD_LLAMA_CPP_TIMEOUT_SECONDS=30.0
KOSHSHIELD_LLAMA_CPP_MAX_TOKENS=512
```

---

## 5. Security & Privacy Guarantees

1. **Anti-SSRF Protection**: `LlamaCppMultimodalClient` only permits connections to `127.0.0.1`, `localhost`, `::1`, and the explicitly configured service name (`llama-server`). Remote IP addresses, AWS/GCP metadata endpoints (`169.254.169.254`), proxy environment variables, and redirects are blocked.
2. **Privacy-Masked Visual Evidence**: The server never passes original page images to the model. Only vault-decrypted, redacted PNG derivatives (`[REDACTED]` bounding boxes) are encoded as base64 data URLs. If unlocated PII exists on a page, visual evidence is blocked (`BLOCKED_UNLOCATED_PII`).
3. **Strict Grounding**: The model cannot hallucinate citations. The server validates every returned `cited_chunk_ids` against authoritative DB chunk records for the active index version. Substantive answers without retrieved citations are rejected as `insufficient_evidence=true`.
4. **Residual PII Protection**: Model output is scanned with `IndianPiiDetector` before returning to the caller. Any residual Aadhaar, PAN, phone number, or sensitive identifier is replaced with redacted placeholders.
5. **No Model Data in Audit**: Audit logs record only metadata (`actor_id`, `tenant_id`, `model_id`, `query_length`, selected chunk IDs, execution duration). No raw queries, answer text, prompt strings, or image bytes are persisted.
