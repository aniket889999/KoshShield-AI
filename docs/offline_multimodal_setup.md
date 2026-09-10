# Offline Multimodal Setup: Qwen3-VL-4B with llama.cpp

This guide details the manual, air-gapped setup for local multimodal retrieval-augmented generation (RAG) in KoshShield AI using Qwen3-VL-4B and `llama.cpp`.

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

## 2. Integrity Verification (SHA-256 Checksums)

Verify the integrity of downloaded artifacts against locally recorded checksums before loading:

```bash
# Generate SHA-256 checksums
shasum -a 256 Qwen3VL-4B-Instruct-Q4_K_M.gguf
shasum -a 256 mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
```

Expected locally recorded checksum baseline:
- `Qwen3VL-4B-Instruct-Q4_K_M.gguf`:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` *(example recorded baseline)*
- `mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf`:
  `d41d8cd98f00b204e9800998ecf8427e04a9d700325d77ae224b7447477174e3` *(example recorded baseline)*

Verify that the output matches your organization's recorded signed digest before launching `llama-server`.

---

## 3. Starting the Local `llama-server`

KoshShield AI requires `llama-server` from `llama.cpp` running locally on loopback (`127.0.0.1` or `localhost`) or an explicit container service name (`llama-server`).

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
- `--host 127.0.0.1`: Bind strictly to loopback to prevent external network access.
- `--alias Qwen3VL-4B-Instruct`: Match the model ID configured in `KOSHSHIELD_LLAMA_CPP_MODEL_ID`.
- `--mmproj`: Enables vision embedding capabilities; if omitted, KoshShield will detect absence of vision and return HTTP 503.
- `--temperature 0.0`: Deterministic responses for grounded evidence verification.

---

## 4. KoshShield AI Configuration

Enable multimodal answering in `apps/api/.env` or process environment:

```env
# Enable experimental multimodal answering
KOSHSHIELD_ENABLE_MULTIMODAL_ANSWERING=true

# Strict local loopback URL (SSRF protection rejects remote hosts, single labels, and redirects)
KOSHSHIELD_LLAMA_CPP_URL=http://127.0.0.1:8080

# Configured model identifier checked against /v1/models
KOSHSHIELD_LLAMA_CPP_MODEL_ID=Qwen3VL-4B-Instruct

# Resource bounds
KOSHSHIELD_LLAMA_CPP_TIMEOUT_SECONDS=60.0
KOSHSHIELD_LLAMA_CPP_MAX_TOKENS=1024
```

---

## 5. Security & Privacy Guarantees

1. **Anti-SSRF Protection**: `LlamaCppMultimodalClient` only permits connections to `127.0.0.1`, `localhost`, `::1`, and the explicitly configured service name (`llama-server`). Remote IP addresses, AWS/GCP metadata endpoints (`169.254.169.254`), proxy environment variables, and redirects are blocked.
2. **Privacy-Masked Visual Evidence**: The server never passes original page images to the model. Only vault-decrypted, redacted PNG derivatives (`[REDACTED]` bounding boxes) are encoded as base64 data URLs. If unlocated PII exists on a page, visual evidence is blocked (`BLOCKED_UNLOCATED_PII`).
3. **Strict Grounding**: The model cannot hallucinate citations. The server validates every returned `cited_chunk_ids` against authoritative DB chunk records for the active index version. Substantive answers without retrieved citations are rejected as `insufficient_evidence=true`.
4. **Residual PII Protection**: Model output is scanned with `IndianPiiDetector` before returning to the caller. Any residual Aadhaar, PAN, phone number, or sensitive identifier is replaced with redacted placeholders.
5. **No Model Data in Audit**: Audit logs record only metadata (`actor_id`, `tenant_id`, `model_id`, `query_length`, selected chunk IDs, execution duration). No raw queries, answer text, prompt strings, or image bytes are persisted.
