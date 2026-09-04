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
}

export interface DocumentRecord {
  id: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  sha256: string;
  status: string;
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
