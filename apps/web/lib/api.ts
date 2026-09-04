export type DependencyStatus = "ready" | "unavailable" | "not_configured";

export interface DependencyState {
  status: DependencyStatus;
  endpoint?: string | null;
}

export interface SystemStatus {
  application: string;
  environment: string;
  processing_boundary: string;
  external_ai_enabled: boolean;
  metadata_backend: string;
  vault: DependencyState;
  metadata_store: DependencyState;
  vector_store: DependencyState;
  local_model: DependencyState;
  ocr: DependencyState;
  embedding: DependencyState;
}


export interface DocumentRecord {
  id: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  sha256: string;
  status: string;
  version?: number;
  created_at: string;
}

export interface ExtractionJob {
  id: string;
  document_id: string;
  status: string;
  pages_processed: number;
  total_pages: number;
  extraction_method?: string | null;
  error_message?: string | null;
  created_at: string;
  completed_at?: string | null;
}

export interface RedactionFinding {
  id: string;
  document_id: string;
  page_number: number;
  finding_type: string;
  confidence: number;
  detection_source: string;
  start_offset: number;
  end_offset: number;
  bbox_json?: { bbox?: [number, number, number, number] } | null;
  salted_value_hash: string;
  masked_context: string;
  status: "PENDING" | "ACCEPTED" | "REJECTED";
  reviewer_id?: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface DocumentPagePreview {
  page_number: number;
  width: number;
  height: number;
  extraction_method: string;
  masked_text?: string | null;
  findings: RedactionFinding[];
}

export interface DocumentRedactions {
  document_id: string;
  status: string;
  total_pages: number;
  total_findings: number;
  unresolved_count: number;
  findings: RedactionFinding[];
  pages: DocumentPagePreview[];
}

export interface ReviewQueueItem {
  document_id: string;
  filename: string;
  status: string;
  total_pages: number;
  total_findings: number;
  pending_findings: number;
  accepted_findings: number;
  rejected_findings: number;
  created_at: string;
}

export interface AuditEvent {
  id: string;
  actor_id: string;
  event_type: string;
  resource_type: string;
  resource_id?: string | null;
  event_hash: string;
  created_at: string;
}

export interface AuditIntegrity {
  valid: boolean;
  event_count: number;
  head_hash?: string | null;
  first_invalid_event_id?: string | null;
}

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    cache: "no-store",
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export function getSystemStatus() {
  return request<SystemStatus>("/system/status");
}

export function listDocuments() {
  return request<DocumentRecord[]>("/documents");
}

export function listAuditEvents() {
  return request<AuditEvent[]>("/audit/events?limit=5");
}

export function getAuditIntegrity() {
  return request<AuditIntegrity>("/audit/integrity");
}

export function uploadDocument(file: File) {
  const form = new FormData();
  form.append("file", file);
  return request<DocumentRecord>("/documents", {
    method: "POST",
    headers: { "X-Actor-ID": "local-demo-user" },
    body: form,
  });
}

export function startExtraction(documentId: string) {
  return request<ExtractionJob>(`/documents/${documentId}/extraction`, {
    method: "POST",
    headers: { "X-Actor-ID": "local-demo-user" },
  });
}

export function getExtractionStatus(documentId: string) {
  return request<ExtractionJob>(`/documents/${documentId}/extraction`);
}

export function getReviewQueue() {
  return request<ReviewQueueItem[]>("/review");
}

export function getDocumentRedactions(documentId: string) {
  return request<DocumentRedactions>(`/documents/${documentId}/redactions`);
}

export function updateRedactionDecision(
  documentId: string,
  findingId: string,
  decision: "ACCEPTED" | "REJECTED",
  version: number
) {
  return request<RedactionFinding>(`/documents/${documentId}/redactions/${findingId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      "X-Actor-ID": "local-demo-user",
    },
    body: JSON.stringify({ decision, version }),
  });
}

export function acceptHighConfidenceRedactions(documentId: string) {
  return request<{ accepted_count: number }>(
    `/documents/${documentId}/redactions/accept-high-confidence`,
    {
      method: "POST",
      headers: { "X-Actor-ID": "local-demo-user" },
    }
  );
}

export function approveRedactions(documentId: string) {
  return request<DocumentRecord>(`/documents/${documentId}/redactions/approve`, {
    method: "POST",
    headers: { "X-Actor-ID": "local-demo-user" },
  });
}

export interface IndexingStatus {
  document_id: string;
  status: string;
  chunk_count: number;
  redaction_version: number;
  tenant_id: string;
  collection_name: string;
  completed_at?: string | null;
}

export interface RetrievalEvidenceItem {
  rank: number;
  fused_score: number;
  sources: string[];
  masked_snippet: string;
  document_id: string;
  document_filename: string;
  page_number: number;
  chunk_id: string;
  evidence_hash: string;
  redaction_version: number;
  citation_label: string;
}

export interface RetrievalResponse {
  query_hash: string;
  tenant_id: string;
  top_k: number;
  total_found: number;
  results: RetrievalEvidenceItem[];
}

export interface RetrievalStatus {
  collection_name: string;
  vector_store_status: DependencyStatus;
  embedding_model_status: DependencyStatus;
  embedding_model_reason: string;
  total_chunks: number;
  indexed_documents_count: number;
}

export function indexDocument(documentId: string) {
  return request<IndexingStatus>(`/documents/${documentId}/index`, {
    method: "POST",
    headers: {
      "X-Actor-ID": "local-demo-user",
      "X-Tenant-ID": "default",
    },
  });
}

export function getDocumentIndexing(documentId: string) {
  return request<IndexingStatus>(`/documents/${documentId}/indexing`, {
    headers: { "X-Tenant-ID": "default" },
  });
}

export function getRetrievalStatus() {
  return request<RetrievalStatus>("/retrieval/status");
}

export function searchRetrieval(params: {
  query: string;
  top_k?: number;
  permitted_document_ids?: string[];
  classification?: string;
}) {
  return request<RetrievalResponse>("/retrieval/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Actor-ID": "local-demo-user",
      "X-Tenant-ID": "default",
    },
    body: JSON.stringify(params),
  });
}
