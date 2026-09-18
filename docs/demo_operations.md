# Demo operations and fail-closed readiness guide

This guide documents the operational architecture, verification procedures, and
demonstration boundaries for KoshShield AI during Hackathon evaluations.

## 1. Operational boundary & honesty principle

KoshShield AI is built local-first and operates without hosted AI APIs, cloud
databases, remote OCR, external analytics, or telemetry.

When running on workstations without dedicated AI accelerators, model weights,
or background daemon services:
- The application **does not fabricate** simulated or hallucinated model answers.
- The application **fails closed** with standardized, sanitized failure codes.
- The operations console clearly distinguishes between `PASSED`, `FAILED`, and
  `NOT_EXECUTED` states.

All privacy and governance controls are designed as **DPDP-aligned** mechanisms
(incorporating purpose limitation, data minimization, consent/approval gating,
and tamper-evident audit trails). This does **not** constitute a statutory claim
of DPDP compliance or an air-gapped security certification.

---

## 2. What is operational offline without model weights

The following subsystems are fully operational without local GPU weights:

| Operational Subsystem | Implemented Mechanism | Verification Command |
|---|---|---|
| **Encrypted Vault** | AES-256-GCM encrypted document intake with unique file initialization vectors | `pytest apps/api/tests/test_vault.py` |
| **Deterministic Extraction** | PDF structural parsing, layout geometry extraction, and image bounding | `pytest apps/api/tests/test_extraction.py` |
| **Indian PII Recognizers** | Verhoeff-verified Aadhaar, format-verified PAN, Phone, Email, IFSC, and Gov ID | `pytest apps/api/tests/test_pii_detection.py` |
| **Review Queue** | Human-in-the-loop review interface for accepting or rejecting sensitive findings | Interactive Web UI (`/` -> Review queue) |
| **Governance Approval Gate** | Multi-role approval required before document becomes eligible for indexing | `pytest apps/api/tests/test_document_evidence_api.py` |
| **Lifecycle Timeline** | Auditable 7-stage document timeline with cryptographic event verification | `pytest apps/api/tests/test_document_lifecycle_api.py` |
| **Procurement Oracle** | Read-only synthetic reference evaluation and supplier comparison | `make evaluate-fixtures` |
| **Tamper-Evident Audit** | Hash-chained event store tracking actor actions with SHA-256 links | `pytest apps/api/tests/test_audit.py` |

---

## 3. What fails closed when model weights or services are absent

When local AI models or daemon services are unavailable, the application strictly
refuses to guess:

| Component | Offline Failure Mode | Safe Failure Code | Status in UI |
|---|---|---|---|
| **llama.cpp / Qwen3-VL-4B** | Inference server unreachable on loopback endpoint | `LLAMA_CPP_OFFLINE` / `GGUF_PATHS_UNCONFIGURED` | `SERVICE_UNAVAILABLE` |
| **BGE-M3 Dense Embeddings** | Safetensors weights or FlagEmbedding package absent | `BGE_PATH_UNCONFIGURED` / `ARTIFACT_MISSING` | `MISSING_ARTIFACT` |
| **PaddleOCR Neural OCR** | Inference bundles or C++ runtime unconfigured | `OCR_PATHS_UNCONFIGURED` / `ARTIFACT_MISSING` | `NOT_CONFIGURED` |
| **Qdrant Vector Database** | Local vector store container offline | `QDRANT_OFFLINE` | `SERVICE_UNAVAILABLE` |
| **Live Query Answering** | Model client unavailable | `LLAMA_UNAVAILABLE` (HTTP 503) | Fail-Closed Error Notice |
| **Evidence Inspection** | Document unapproved or unindexed | `DOCUMENT_NOT_APPROVED` (HTTP 403) | Restricted Gate Warning |

---

## 4. Operator demonstration walkthrough

To demonstrate secure operational controls to evaluators:

### Step 1: Inspect demo readiness
Navigate to **Readiness & Lifecycle** in the web console or execute:
```bash
curl -H "X-Tenant-ID: default" -H "X-Actor-ID: operator" -H "X-Roles: admin" \
  http://localhost:8000/api/v1/system/readiness
```
Observe that:
- Core storage (`relational_db` and `encrypted_vault`) is `READY`.
- Missing AI weights report `DEMO_RESTRICTED` rather than pretending full intelligence.
- Details are sanitized and contain no local directory paths or secrets.

### Step 2: Ingest a document and verify lifecycle
Upload a PDF via the console or REST API:
```bash
curl -X POST -F "file=@sample.pdf" \
  -H "X-Tenant-ID: default" -H "X-Actor-ID: operator" \
  http://localhost:8000/api/v1/documents
```
Inspect the lifecycle timeline:
```bash
curl -H "X-Tenant-ID: default" -H "X-Actor-ID: operator" \
  http://localhost:8000/api/v1/documents/{document_id}/timeline
```
Observe that:
- Stage `UPLOAD` is `PASSED`.
- Downstream stages (`EXTRACTION`, `PII_REVIEW`, `APPROVAL`, `INDEXING`, `RETRIEVAL`) are
  accurately marked `NOT_EXECUTED` until processed.
- The document cannot bypass stages out of order.

### Step 3: Verify governance evidence gating
Attempt to retrieve evidence chunks before approval:
```bash
curl -H "X-Tenant-ID: default" -H "X-Actor-ID: operator" \
  http://localhost:8000/api/v1/documents/{document_id}/evidence
```
Expected response:
```json
{
  "detail": "Evidence catalog restricted: document has not completed governance approval and indexing..."
}
```
**HTTP status**: `403 Forbidden`. The system fails closed.

### Step 4: Run reference procurement evaluation
Recompute synthetic fixture calculations deterministically:
```bash
make evaluate-fixtures
```
Verify exit code `0` and metadata-only output confirming manifest integrity,
CSV parsing, and supplier comparison checks without loading any neural weights.

---

## 5. Security and sanitization guarantees

- **No Absolute Paths**: All endpoints strip filesystem prefixes (`/Users/...`, `/home/...`).
- **No Vault Decryption for Evidence**: Evidence endpoints read only approved masked text derivatives, never original encrypted vault files.
- **Tenant Isolation**: Any cross-tenant resource request returns HTTP 404 to eliminate resource enumeration.
- **Residual PII Trap**: All output snippets pass through runtime PII detectors; any unmasked pattern triggers immediate fail-closed termination.
