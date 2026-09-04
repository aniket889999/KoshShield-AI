"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  Archive,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Database,
  FileCheck2,
  FileLock2,
  FileSearch,
  Gauge,
  History,
  KeyRound,
  LockKeyhole,
  Menu,
  Network,
  Search,
  Server,
  ShieldCheck,
  Upload,
  UserCheck,
  X,
} from "lucide-react";
import { useRef, useState, type ComponentType } from "react";

import {
  getAuditIntegrity,
  getSystemStatus,
  listAuditEvents,
  listDocuments,
  uploadDocument,
  type DependencyStatus,
} from "@/lib/api";
import { formatBytes, formatTime, shortHash } from "@/lib/format";

const navigation = [
  { label: "Overview", icon: Gauge, active: true },
  { label: "Documents", icon: FileLock2 },
  { label: "Review queue", icon: FileCheck2 },
  { label: "Intelligence", icon: FileSearch },
  { label: "Approvals", icon: UserCheck },
  { label: "Audit trail", icon: History },
];

const statusLabels: Record<DependencyStatus, string> = {
  ready: "Ready",
  unavailable: "Unavailable",
  not_configured: "Not configured",
};

function StatusBadge({ status }: { status: DependencyStatus }) {
  return (
    <span className={`status-badge status-${status}`}>
      <span className="status-dot" />
      {statusLabels[status]}
    </span>
  );
}

function ServiceTile({
  icon: Icon,
  label,
  detail,
  status,
}: {
  icon: ComponentType<{ size?: number; strokeWidth?: number }>;
  label: string;
  detail: string;
  status: DependencyStatus;
}) {
  return (
    <div className="service-tile">
      <div className="service-icon">
        <Icon size={20} strokeWidth={1.8} />
      </div>
      <div className="service-copy">
        <span>{label}</span>
        <strong>{detail}</strong>
      </div>
      <StatusBadge status={status} />
    </div>
  );
}

export function OperationsConsole() {
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const statusQuery = useQuery({
    queryKey: ["system-status"],
    queryFn: getSystemStatus,
    refetchInterval: 10_000,
  });
  const documentsQuery = useQuery({ queryKey: ["documents"], queryFn: listDocuments });
  const auditQuery = useQuery({ queryKey: ["audit"], queryFn: listAuditEvents });
  const auditIntegrityQuery = useQuery({
    queryKey: ["audit-integrity"],
    queryFn: getAuditIntegrity,
  });
  const uploadMutation = useMutation({
    mutationFn: uploadDocument,
    onSuccess: async () => {
      setSelectedFile(null);
      if (inputRef.current) inputRef.current.value = "";
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["documents"] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
        queryClient.invalidateQueries({ queryKey: ["audit-integrity"] }),
      ]);
    },
  });

  const system = statusQuery.data;
  const systemReady = system?.metadata_store.status === "ready";
  const documentCount = documentsQuery.data?.length ?? 0;

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark"><ShieldCheck size={23} /></div>
          <div>
            <strong>KoshShield AI</strong>
            <span>Sovereign workbench</span>
          </div>
          <button className="icon-button mobile-only" onClick={() => setSidebarOpen(false)} aria-label="Close navigation">
            <X size={19} />
          </button>
        </div>

        <nav aria-label="Primary navigation">
          <span className="nav-label">Workspace</span>
          {navigation.map(({ label, icon: Icon, active }) => (
            <button className={`nav-item ${active ? "nav-active" : ""}`} key={label}>
              <Icon size={18} />
              <span>{label}</span>
              {active && <ChevronRight size={16} className="nav-arrow" />}
            </button>
          ))}
        </nav>

        <div className="boundary-panel">
          <div><Network size={17} /><strong>Data boundary</strong></div>
          <p>No external AI endpoints are permitted by configuration.</p>
          <span><LockKeyhole size={14} /> Local-only processing</span>
        </div>
      </aside>

      {sidebarOpen && <button className="sidebar-scrim" onClick={() => setSidebarOpen(false)} aria-label="Close navigation overlay" />}

      <main>
        <header className="topbar">
          <button className="icon-button mobile-only" onClick={() => setSidebarOpen(true)} aria-label="Open navigation">
            <Menu size={20} />
          </button>
          <div className="search-shell">
            <Search size={17} />
            <span>Search secured documents</span>
            <kbd>⌘ K</kbd>
          </div>
          <div className="topbar-status">
            <span className={`connection-dot ${systemReady ? "connection-ready" : ""}`} />
            {statusQuery.isError ? "Backend offline" : systemReady ? "Local services connected" : "Checking services"}
          </div>
          <button className="profile-button" aria-label="Current user">AK</button>
        </header>

        <div className="workspace">
          <section className="page-heading">
            <div>
              <span className="eyebrow">Secure operations</span>
              <h1>Operations overview</h1>
              <p>Inspect the local trust boundary and accept documents into encrypted storage.</p>
            </div>
            <div className="mode-chip"><ShieldCheck size={17} /> Air-gapped mode</div>
          </section>

          {statusQuery.isError && (
            <div className="notice notice-error">
              <CircleAlert size={20} />
              <div><strong>Local API is unavailable</strong><span>Start the configured FastAPI service, then retry.</span></div>
              <button onClick={() => statusQuery.refetch()}>Retry</button>
            </div>
          )}

          {system && system.vault.status === "not_configured" && (
            <div className="notice notice-warning">
              <KeyRound size={20} />
              <div><strong>Encrypted intake is locked</strong><span>Configure KOSHSHIELD_MASTER_KEY_BASE64 before uploading documents.</span></div>
            </div>
          )}

          <section className="service-grid" aria-label="Local service status">
            <ServiceTile icon={LockKeyhole} label="Processing boundary" detail="Local only" status={system ? "ready" : "unavailable"} />
            <ServiceTile icon={Archive} label="Encrypted vault" detail="AES-256-GCM" status={system?.vault.status ?? "unavailable"} />
            <ServiceTile icon={Database} label="Metadata store" detail={system?.metadata_backend ?? "Not connected"} status={system?.metadata_store.status ?? "unavailable"} />
            <ServiceTile icon={Server} label="Local model" detail="Qwen3-VL-4B" status={system?.local_model.status ?? "unavailable"} />
          </section>

          <div className="content-grid">
            <section className="panel intake-panel">
              <div className="panel-heading">
                <div><span className="section-kicker">Document intake</span><h2>Accept into encrypted vault</h2></div>
                <span className="supported-types">PDF · PNG · JPEG</span>
              </div>

              <label className={`upload-zone ${selectedFile ? "upload-selected" : ""}`}>
                <input
                  ref={inputRef}
                  type="file"
                  accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg"
                  onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                />
                <div className="upload-icon"><Upload size={24} /></div>
                {selectedFile ? (
                  <div><strong>{selectedFile.name}</strong><span>{formatBytes(selectedFile.size)} · Ready for encrypted intake</span></div>
                ) : (
                  <div><strong>Select a confidential document</strong><span>Original bytes are encrypted before acceptance is recorded.</span></div>
                )}
              </label>

              <div className="upload-actions">
                <div className="intake-proof"><ShieldCheck size={16} /><span>Signature validation · SHA-256 evidence hash · encrypted storage</span></div>
                <button
                  className="primary-button"
                  disabled={!selectedFile || uploadMutation.isPending || system?.vault.status !== "ready"}
                  onClick={() => selectedFile && uploadMutation.mutate(selectedFile)}
                >
                  {uploadMutation.isPending ? "Encrypting…" : "Accept document"}
                </button>
              </div>
              {uploadMutation.isError && <p className="form-error">{uploadMutation.error.message}</p>}
              {uploadMutation.isSuccess && <p className="form-success"><CheckCircle2 size={15} /> Document encrypted and audit event recorded.</p>}
            </section>

            <section className="panel assurance-panel">
              <div className="panel-heading">
                <div><span className="section-kicker">Trust controls</span><h2>Current guarantees</h2></div>
                <Activity size={19} />
              </div>
              <div className="assurance-list">
                <div><ShieldCheck size={19} /><span><strong>External AI disabled</strong><small>Runtime configuration rejects public service URLs.</small></span></div>
                <div><FileLock2 size={19} /><span><strong>Originals encrypted first</strong><small>Plain document bytes are never written to application storage.</small></span></div>
                <div><History size={19} /><span><strong>Evidence is chained</strong><small>Each audit event includes the previous event hash.</small></span></div>
              </div>
            </section>
          </div>

          <div className="content-grid lower-grid">
            <section className="panel documents-panel">
              <div className="panel-heading">
                <div><span className="section-kicker">Recent intake</span><h2>Encrypted documents</h2></div>
                <span className="record-count">
                  {documentCount} {documentCount === 1 ? "record" : "records"}
                </span>
              </div>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Document</th><th>Status</th><th>Evidence hash</th><th>Accepted</th></tr></thead>
                  <tbody>
                    {documentsQuery.data?.map((document) => (
                      <tr key={document.id}>
                        <td><span className="document-cell"><FileLock2 size={17} /><span><strong>{document.filename}</strong><small>{formatBytes(document.size_bytes)}</small></span></span></td>
                        <td><span className="encrypted-label"><LockKeyhole size={13} /> Encrypted</span></td>
                        <td><code>{shortHash(document.sha256)}</code></td>
                        <td>{formatTime(document.created_at)}</td>
                      </tr>
                    ))}
                    {!documentsQuery.data?.length && (
                      <tr><td colSpan={4} className="empty-cell">No documents have entered the encrypted vault.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="panel audit-panel">
              <div className="panel-heading">
                <div><span className="section-kicker">Audit snapshot</span><h2>Latest events</h2></div>
                <StatusBadge
                  status={auditIntegrityQuery.data?.valid ? "ready" : "unavailable"}
                />
              </div>
              <div className="audit-list">
                {auditQuery.data?.map((event) => (
                  <div className="audit-row" key={event.id}>
                    <div className="audit-icon"><FileCheck2 size={16} /></div>
                    <span><strong>{event.event_type.replace(".", " ")}</strong><small>{event.actor_id} · {formatTime(event.created_at)}</small></span>
                    <code>{event.event_hash.slice(0, 7)}</code>
                  </div>
                ))}
                {!auditQuery.data?.length && <p className="empty-audit">Audit events appear after a secured action.</p>}
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}
