"use client";

import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  Cpu,
  Database,
  HardDrive,
  Layers,
  Loader2,
  RefreshCw,
  ScanLine,
  Server,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { useState } from "react";

import {
  getDemoReadiness,
  getDocumentEvidenceCatalog,
  getDocumentTimeline,
  listDocuments,
  type ComponentReadiness,
  type DemoReadinessReport,
  type DocumentRecord,
  type LifecycleStageRecord,
} from "@/lib/api";
import {
  formatBytes,
  formatReadinessStatus,
  formatStageStatus,
  formatTime,
  shortHash,
} from "@/lib/format";

function categoryIcon(category: string) {
  switch (category) {
    case "EMBEDDING":
      return <Sparkles size={18} />;
    case "INFERENCE_SERVER":
      return <Server size={18} />;
    case "LLM_WEIGHTS":
      return <Cpu size={18} />;
    case "OCR":
      return <ScanLine size={18} />;
    case "VECTOR_DATABASE":
      return <Layers size={18} />;
    case "RELATIONAL_DATABASE":
      return <Database size={18} />;
    case "ENCRYPTED_VAULT":
      return <HardDrive size={18} />;
    default:
      return <Activity size={18} />;
  }
}

function readinessBadgeClass(status: string) {
  switch (status) {
    case "READY":
      return "status-badge status-ready";
    case "MISSING_ARTIFACT":
    case "SERVICE_UNAVAILABLE":
    case "ERROR":
      return "status-badge status-unavailable";
    case "NOT_CONFIGURED":
    case "NOT_EXECUTED":
    default:
      return "status-badge status-not_configured";
  }
}

function stageBadgeClass(status: string) {
  switch (status) {
    case "PASSED":
      return "status-badge status-ready";
    case "FAILED":
      return "status-badge status-unavailable";
    case "IN_PROGRESS":
      return "status-badge status-ready";
    case "PENDING":
      return "status-badge status-not_configured";
    case "NOT_EXECUTED":
    default:
      return "status-badge status-not_configured";
  }
}

export function DemoOperationsWorkspace() {
  const [probeActive, setProbeActive] = useState(false);
  const [selectedDocId, setSelectedDocId] = useState<string>("");

  const readinessQuery = useQuery({
    queryKey: ["demo-readiness", probeActive],
    queryFn: () => getDemoReadiness(probeActive),
    refetchInterval: 10_000,
  });

  const documentsQuery = useQuery({
    queryKey: ["documents"],
    queryFn: listDocuments,
  });

  const activeDocId = selectedDocId || documentsQuery.data?.[0]?.id || "";

  const timelineQuery = useQuery({
    queryKey: ["document-timeline", activeDocId],
    queryFn: () => getDocumentTimeline(activeDocId),
    enabled: Boolean(activeDocId),
  });

  const evidenceQuery = useQuery({
    queryKey: ["document-evidence", activeDocId],
    queryFn: () => getDocumentEvidenceCatalog(activeDocId),
    enabled: Boolean(activeDocId),
    retry: false,
  });

  const report: DemoReadinessReport | undefined = readinessQuery.data;
  const timeline = timelineQuery.data;
  const evidence = evidenceQuery.data;

  return (
    <div className="workspace-stack" style={{ display: "grid", gap: "24px" }}>
      {/* 1. Header and Overall Status */}
      <section className="panel">
        <div className="panel-heading" style={{ alignItems: "center" }}>
          <div>
            <span className="section-kicker">Demo Operations Layer</span>
            <h2>Runtime Readiness & Integrity Boundary</h2>
          </div>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            <button
              type="button"
              className="action-button secondary-action"
              onClick={() => {
                setProbeActive((prev) => !prev);
              }}
              style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px" }}
            >
              <RefreshCw size={14} className={readinessQuery.isFetching ? "spin-loader" : ""} />
              <span>{probeActive ? "Probing Local Services (On)" : "Probe Local Services"}</span>
            </button>
            <span
              className={
                report?.overall_status === "READY"
                  ? "mode-chip"
                  : report?.overall_status === "DEMO_RESTRICTED"
                  ? "notice-chip-amber"
                  : "notice-chip-red"
              }
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                padding: "6px 12px",
                borderRadius: "6px",
                fontSize: "11px",
                fontWeight: 800,
              }}
            >
              <ShieldCheck size={14} />
              {report?.overall_status ?? "CHECKING…"}
            </span>
          </div>
        </div>

        {report && (
          <div
            style={{
              padding: "16px",
              background: "#f8fafc",
              border: "1px solid var(--line)",
              borderRadius: "8px",
              marginTop: "12px",
            }}
          >
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
                gap: "16px",
                marginBottom: "14px",
              }}
            >
              <div>
                <span
                  style={{
                    fontSize: "11px",
                    fontWeight: 700,
                    textTransform: "uppercase",
                    color: "var(--ink-500)",
                  }}
                >
                  Offline Demonstration
                </span>
                <div style={{ fontSize: "14px", fontWeight: 700, marginTop: "4px" }}>
                  {report.can_run_offline_demo
                    ? "✓ Operational (Vault, Rules, Audit, Governance)"
                    : "✗ Inactive"}
                </div>
              </div>
              <div>
                <span
                  style={{
                    fontSize: "11px",
                    fontWeight: 700,
                    textTransform: "uppercase",
                    color: "var(--ink-500)",
                  }}
                >
                  Live AI Inference
                </span>
                <div style={{ fontSize: "14px", fontWeight: 700, marginTop: "4px" }}>
                  {report.can_run_live_inference
                    ? "✓ Operational (Qwen, BGE-M3, Qdrant)"
                    : "Fail-Closed / Model Weights Offline"}
                </div>
              </div>
              <div>
                <span
                  style={{
                    fontSize: "11px",
                    fontWeight: 700,
                    textTransform: "uppercase",
                    color: "var(--ink-500)",
                  }}
                >
                  Local Isolation Boundary
                </span>
                <div style={{ fontSize: "14px", fontWeight: 700, marginTop: "4px" }}>
                  Strict Local-Only (Zero External Network Calls)
                </div>
              </div>
            </div>

            <p style={{ fontSize: "12px", color: "var(--ink-700)", margin: "8px 0 0" }}>
              {report.summary}
            </p>
            <div
              style={{
                fontSize: "11px",
                color: "var(--ink-500)",
                marginTop: "8px",
                fontStyle: "italic",
              }}
            >
              {report.disclaimer}
            </div>
          </div>
        )}
      </section>

      {/* 2. Seven Core Components Matrix */}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="section-kicker">Component Integrity</span>
            <h3>Required Local Components ({report?.executable_components_count ?? 0}/{report?.total_components_count ?? 7} Executable)</h3>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: "14px",
            marginTop: "14px",
          }}
        >
          {report?.components.map((comp: ComponentReadiness) => (
            <div
              key={comp.component_id}
              style={{
                border: "1px solid var(--line)",
                borderRadius: "8px",
                padding: "14px",
                background: "white",
                display: "flex",
                flexDirection: "column",
                gap: "8px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ color: "var(--navy-900)" }}>{categoryIcon(comp.category)}</span>
                  <strong style={{ fontSize: "13px" }}>{comp.display_name}</strong>
                </div>
                <span className={readinessBadgeClass(comp.status)}>
                  <span className="status-dot" />
                  {formatReadinessStatus(comp.status)}
                </span>
              </div>

              <div style={{ fontSize: "12px", color: "var(--ink-700)", lineHeight: 1.4 }}>
                {comp.details}
              </div>

              <div
                style={{
                  marginTop: "auto",
                  paddingTop: "8px",
                  borderTop: "1px solid var(--line)",
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "11px",
                  color: "var(--ink-500)",
                }}
              >
                <span>
                  Execution: <strong>{comp.executable ? "EXECUTABLE" : "NON_EXECUTABLE"}</strong>
                </span>
                {comp.failure_code && (
                  <span style={{ fontFamily: "monospace", color: "var(--red-700)" }}>
                    {comp.failure_code}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 3. Document Lifecycle Timeline Explorer */}
      <section className="panel">
        <div className="panel-heading" style={{ alignItems: "center" }}>
          <div>
            <span className="section-kicker">Auditable Verification</span>
            <h3>Document Lifecycle Timeline</h3>
          </div>
          {documentsQuery.data && documentsQuery.data.length > 0 && (
            <select
              value={activeDocId}
              onChange={(e) => setSelectedDocId(e.target.value)}
              style={{
                padding: "6px 12px",
                borderRadius: "6px",
                border: "1px solid var(--line)",
                fontSize: "12px",
                background: "white",
              }}
            >
              {documentsQuery.data.map((doc: DocumentRecord) => (
                <option key={doc.id} value={doc.id}>
                  {doc.filename} ({doc.status})
                </option>
              ))}
            </select>
          )}
        </div>

        {timelineQuery.isPending && (
          <div style={{ padding: "24px", textAlign: "center", color: "var(--ink-500)" }}>
            <Loader2 size={24} className="spin-loader" />
            <div style={{ marginTop: "8px", fontSize: "12px" }}>Loading lifecycle timeline…</div>
          </div>
        )}

        {timeline && (
          <div>
            {/* Metadata Summary Banner */}
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "18px",
                padding: "12px 16px",
                background: "#f1f5f9",
                borderRadius: "6px",
                marginBottom: "18px",
                fontSize: "12px",
              }}
            >
              <div>
                File: <strong>{timeline.filename}</strong>
              </div>
              <div>
                Size: <strong>{formatBytes(timeline.size_bytes)}</strong>
              </div>
              <div>
                SHA-256: <code>{shortHash(timeline.sha256)}</code>
              </div>
              <div>
                Status: <strong>{timeline.current_status}</strong>
              </div>
              <div>
                Tenant: <code>{timeline.tenant_id}</code>
              </div>
            </div>

            {/* Stages Stepper */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))",
                gap: "10px",
                marginBottom: "20px",
              }}
            >
              {timeline.stages.map((stage: LifecycleStageRecord, idx: number) => (
                <div
                  key={stage.stage_id}
                  style={{
                    border: "1px solid var(--line)",
                    borderRadius: "6px",
                    padding: "10px",
                    background: stage.status === "PASSED" ? "#f0fdf4" : "white",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      marginBottom: "6px",
                    }}
                  >
                    <span style={{ fontSize: "11px", fontWeight: 700, color: "var(--ink-500)" }}>
                      {idx + 1}. {stage.display_name}
                    </span>
                    <span className={stageBadgeClass(stage.status)} style={{ fontSize: "9px" }}>
                      {formatStageStatus(stage.status)}
                    </span>
                  </div>
                  <div style={{ fontSize: "11px", color: "var(--ink-700)", lineHeight: 1.3 }}>
                    {stage.summary}
                  </div>
                </div>
              ))}
            </div>

            {/* Cryptographic Audit Events for this Document */}
            <div>
              <h4 style={{ fontSize: "13px", fontWeight: 700, marginBottom: "10px" }}>
                Tamper-Evident Audit Records ({timeline.audit_events.length})
              </h4>
              <div
                style={{
                  border: "1px solid var(--line)",
                  borderRadius: "6px",
                  overflow: "hidden",
                  fontSize: "11px",
                }}
              >
                {timeline.audit_events.length === 0 ? (
                  <div style={{ padding: "12px", color: "var(--ink-500)" }}>
                    No audit records registered for this resource.
                  </div>
                ) : (
                  timeline.audit_events.map((ev, i) => (
                    <div
                      key={ev.event_id}
                      style={{
                        display: "grid",
                        gridTemplateColumns: "140px 180px 120px 1fr",
                        gap: "10px",
                        padding: "8px 12px",
                        borderBottom: i < timeline.audit_events.length - 1 ? "1px solid var(--line)" : "none",
                        background: i % 2 === 0 ? "#ffffff" : "#f8fafc",
                        alignItems: "center",
                      }}
                    >
                      <span style={{ color: "var(--ink-500)" }}>{formatTime(ev.timestamp)}</span>
                      <strong>{ev.event_type}</strong>
                      <span>Actor: {ev.actor_id}</span>
                      <span style={{ fontFamily: "monospace", color: "var(--teal-700)" }}>
                        Hash: {shortHash(ev.event_hash)}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </section>

      {/* 4. Secure Evidence & Citation Inspector */}
      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="section-kicker">Governance & Citations</span>
            <h3>Privacy-Gated Evidence Catalog</h3>
          </div>
        </div>

        {evidenceQuery.isError && (
          <div
            style={{
              padding: "16px",
              background: "#fff7ed",
              border: "1px solid #fed7aa",
              borderRadius: "6px",
              display: "flex",
              alignItems: "center",
              gap: "12px",
              color: "#9a3412",
              fontSize: "12px",
            }}
          >
            <AlertTriangle size={20} />
            <div>
              <strong>Evidence Catalog Restricted</strong>
              <div>
                Document must complete PII review and governance approval before chunks and citations are eligible for retrieval.
              </div>
            </div>
          </div>
        )}

        {evidenceQuery.isSuccess && evidence && (
          <div>
            <div
              style={{
                padding: "10px 14px",
                background: "#ecfdf5",
                border: "1px solid #a7f3d0",
                borderRadius: "6px",
                marginBottom: "14px",
                fontSize: "12px",
                color: "#065f46",
              }}
            >
              ✓ <strong>{evidence.privacy_gate_verdict}</strong>: Original document bytes remain in the AES-256-GCM encrypted vault. Only authorized, privacy-masked chunks ({evidence.chunk_count}) are viewable.
            </div>

            <div style={{ display: "grid", gap: "10px" }}>
              {evidence.chunks.map((chunk) => (
                <div
                  key={chunk.chunk_id}
                  style={{
                    border: "1px solid var(--line)",
                    borderRadius: "6px",
                    padding: "12px",
                    background: "white",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginBottom: "6px",
                      fontSize: "12px",
                    }}
                  >
                    <strong>{chunk.citation_label}</strong>
                    <span style={{ color: "var(--ink-500)", fontFamily: "monospace", fontSize: "11px" }}>
                      Hash: {shortHash(chunk.masked_content_hash)}
                    </span>
                  </div>
                  <div
                    style={{
                      padding: "8px 10px",
                      background: "#f8fafc",
                      borderRadius: "4px",
                      fontSize: "12px",
                      color: "var(--ink-900)",
                      fontFamily: "monospace",
                      lineHeight: 1.4,
                    }}
                  >
                    {chunk.masked_snippet}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
