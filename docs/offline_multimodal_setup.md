# Offline Multimodal Setup: Qwen3-VL-4B with llama.cpp

This guide details the manual, air-gapped setup for local multimodal retrieval-augmented generation (RAG) in KoshShield AI using Qwen3-VL-4B and `llama.cpp` (pinned supported target contract: **release `v0.4.0`**, **build `b10809`**, **commit `5266f24`**).

> **Integration Status Notice**:
> The Milestone 4 multimodal code path is fully implemented and security-hardened with fail-closed runtime preflight and strict schema validation. Real Qwen3-VL plus llama.cpp integration remains **NOT EXECUTED** in this environment because local inference server processes and multi-gigabyte model weights are intentionally absent.

In strict compliance with KoshShield AI's local-first security policy (`AGENTS.md`):
- **Never** download model weights or projector files at application runtime.
- **Never** use `-hf` flags, contact Hugging Face, remote model hubs, or external endpoints.
- **Never** commit model weights or GGUF files to Git.

---

## 1. Required Artifacts

Acquire the following quantized GGUF artifacts via a verified offline physical transfer (e.g., secure internal artifact repository or encrypted media):

| Artifact | File Name | Description |
|---|---|---|
| **Base Model** | `Qwen3-VL-4B-Instruct-Q4_K_M.gguf` | 4-bit quantized Qwen3-VL-4B instruction-tuned text backbone |
| **Multimodal Projector** | `mmproj-Qwen3-VL-4B-Instruct-Q8_0.gguf` | 8-bit quantized vision encoder projector for visual tokens |

Store these files in a dedicated local directory outside the repository (for example, `/opt/models/qwen3-vl/` or `~/.cache/koshshield/models/`).

---

## 2. Integrity Verification (Operator Signed Manifest)

Integrity verification requires a trusted, operator-supplied signed SHA-256 manifest. Checksums must never be invented or assumed.

Operators must verify the SHA-256 digests of downloaded model artifacts against their organization's cryptographically signed manifest prior to starting the inference server:

```bash
# Generate SHA-256 checksums of local artifact files
shasum -a 256 Qwen3-VL-4B-Instruct-Q4_K_M.gguf
shasum -a 256 mmproj-Qwen3-VL-4B-Instruct-Q8_0.gguf

# Verify against the operator's signed release manifest
gpg --verify SHA256SUMS.sig SHA256SUMS
shasum -a 256 -c SHA256SUMS
```

Do not proceed if any digest fails validation against the authoritative signed manifest.

---

## 3. Starting the Local `llama-server`

KoshShield AI requires `llama-server` from **`llama.cpp` release `v0.4.0` (build `b10809`, commit `5266f24`)** running locally on loopback (`127.0.0.1` or `localhost`) or an explicit container service name (`llama-server`).

### Command Line Invocation

```bash
llama-server \
  --model /opt/models/qwen3-vl/Qwen3-VL-4B-Instruct-Q4_K_M.gguf \
  --mmproj /opt/models/qwen3-vl/mmproj-Qwen3-VL-4B-Instruct-Q8_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --alias qwen3-vl-4b-instruct \
  --ctx-size 8192 \
  --n-gpu-layers 33 \
  --temperature 0.0
```

### Critical Server Parameters
- Supported target contract: **release `v0.4.0`**, **build `b10809`**, **commit `5266f24`**
- `--host 127.0.0.1`: Bind strictly to loopback to prevent external network access.
- `--alias qwen3-vl-4b-instruct`: Match the model alias configured in `KOSHSHIELD_LLAMA_CPP_MODEL_ID` exactly.
- `--model` and `--mmproj`: Local file paths only. Never use `-hf` or remote download options.
- Vision Modality: The server must report `modalities.vision: true` and matching `build_info` via native `GET /props?model=qwen3-vl-4b-instruct`.
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
KOSHSHIELD_LLAMA_CPP_MODEL_ID=qwen3-vl-4b-instruct

# Allowlisted runtime contract
KOSHSHIELD_LLAMA_CPP_RELEASE=v0.4.0
KOSHSHIELD_LLAMA_CPP_BUILD=b10809
KOSHSHIELD_LLAMA_CPP_COMMIT=5266f24

# Resource bounds
KOSHSHIELD_LLAMA_CPP_TIMEOUT_SECONDS=30.0
KOSHSHIELD_LLAMA_CPP_MAX_TOKENS=512
```

---

## 5. Security & Privacy Guarantees

1. **Anti-SSRF Protection**: `LlamaCppMultimodalClient` only permits connections to `127.0.0.1`, `localhost`, `::1`, and the explicitly configured service name (`llama-server`). Remote IP addresses, AWS/GCP metadata endpoints (`169.254.169.254`), proxy environment variables, and redirects are blocked.
2. **Fail-Closed Runtime Preflight**: Before inference, the client queries both `GET /v1/models` and `GET /props?model={safe_model}`, strictly verifying `modalities.vision: true` and `build_info` matching the target contract (`b10809` / `5266f24`). Any build mismatch or missing vision modality aborts immediately.
3. **Privacy-Masked Visual Evidence**: The server never passes original page images to the model. Only vault-decrypted, redacted PNG derivatives (`[REDACTED]` bounding boxes) are encoded as base64 data URLs. If unlocated PII exists on a page, visual evidence is blocked (`BLOCKED_UNLOCATED_PII`).
4. **Strict Schema Validation**: The client validates the complete llama.cpp chat completion response against strict Pydantic schemas (`choices`, `message`, `content`, `usage`) with extra fields forbidden and bounded tokens, mapping any unexpected shapes to sanitized HTTP 502 (`MODEL_RESPONSE_INVALID`).
5. **Strict Grounding**: The model cannot hallucinate citations. The server validates every returned `cited_chunk_ids` against authoritative DB chunk records for the active index version. Substantive answers without retrieved citations are rejected as `insufficient_evidence=true`.
6. **Residual PII Protection**: Model output is scanned with `IndianPiiDetector` before returning to the caller. Any residual Aadhaar, PAN, phone number, or sensitive identifier is replaced with redacted placeholders.
7. **No Model Data in Audit or Logs**: Audit logs and application logs record only metadata (`actor_id`, `tenant_id`, `model_id`, `query_length`, selected chunk IDs, execution duration, stable failure codes). No raw queries, answer text, prompt strings, paths, or image bytes are persisted or logged.
