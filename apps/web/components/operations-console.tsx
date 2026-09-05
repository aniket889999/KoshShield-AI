"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Activity,
  Archive,
  BookOpen,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Copy,
  Cpu,
  CreditCard,
  Database,
  Eye,
  FileCheck2,
  FileCode,
  FileLock2,
  FileSearch,
  FileText,
  Fingerprint,
  Gauge,
  History,
  Image as ImageIcon,
  KeyRound,
  Layers,
  Loader2,
  LockKeyhole,
  Mail,
  Menu,
  Network,
  Phone,
  ScanLine,
  Search,
  Server,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
  UserCheck,
  X,
} from "lucide-react";
import NextImage from "next/image";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ComponentType,
  type CSSProperties,
} from "react";

import {
  acceptHighConfidenceRedactions,
  approveRedactions,
  fetchVisualEvidenceImage,
  getAuditIntegrity,
  getDocumentRedactions,
  getRetrievalStatus,
  getReviewQueue,
  getSystemStatus,
  indexDocument,
  listAuditEvents,
  listDocuments,
  searchRetrieval,
  startExtraction,
  updateRedactionDecision,
  uploadDocument,
  type DependencyStatus,
  type DocumentRecord,
  type RedactionFinding,
  type RetrievalEvidenceItem,
  type RetrievalVisualRegion,
} from "@/lib/api";


import {
  formatBytes,
  formatConfidence,
  formatFindingLabel,
  formatStatusLabel,
  formatTime,
  shortHash,
} from "@/lib/format";

type NavTab = "Overview" | "Documents" | "Review queue" | "Intelligence" | "Approvals" | "Audit trail";

const navigation: { label: NavTab; icon: ComponentType<{ size?: number }> }[] = [
  { label: "Overview", icon: Gauge },
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

function FindingTypeIcon({ type }: { type: string }) {
  switch (type.toUpperCase()) {
    case "AADHAAR":
      return <Fingerprint size={15} />;
    case "PAN":
      return <CreditCard size={15} />;
    case "PHONE":
      return <Phone size={15} />;
    case "EMAIL":
      return <Mail size={15} />;
    case "BANK_ACCOUNT":
    case "IFSC":
      return <FileText size={15} />;
    case "PASSPORT":
    case "GOV_ID":
      return <FileCode size={15} />;
    default:
      return <ShieldAlert size={15} />;
  }
}

interface VisualEvidenceSelection {
  item: RetrievalEvidenceItem;
  region: RetrievalVisualRegion;
  imageUrl: string;
}

function regionLabel(regionType: string) {
  return regionType
    .toLowerCase()
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function sourceLabel(source: string) {
  if (source === "dense") return "Dense vector";
  if (source === "sparse") return "Sparse lexical";
  if (source === "visual-caption") return "Visual caption";
  return source;
}

function regionHighlightStyle(region: RetrievalVisualRegion): CSSProperties | undefined {
  const [x0, y0, x1, y1] = region.bbox ?? [];
  const pageWidth = region.page_width ?? 0;
  const pageHeight = region.page_height ?? 0;
  if (!region.bbox || pageWidth <= 0 || pageHeight <= 0) return undefined;

  return {
    left: `${(x0 / pageWidth) * 100}%`,
    top: `${(y0 / pageHeight) * 100}%`,
    width: `${((x1 - x0) / pageWidth) * 100}%`,
    height: `${((y1 - y0) / pageHeight) * 100}%`,
  };
}

export function OperationsConsole() {
  const queryClient = useQueryClient();
  const inputRef = useRef<HTMLInputElement>(null);
  const [activeNav, setActiveNav] = useState<NavTab>("Overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Review workspace state
  const [selectedReviewDocId, setSelectedReviewDocId] = useState<string | null>(null);
  const [selectedPageNumber, setSelectedPageNumber] = useState<number>(1);
  const [categoryFilter, setCategoryFilter] = useState<string>("ALL");
  const [actionNotice, setActionNotice] = useState<string | null>(null);

  // Queries
  const statusQuery = useQuery({
    queryKey: ["system-status"],
    queryFn: getSystemStatus,
    refetchInterval: 10_000,
  });
  const documentsQuery = useQuery({ queryKey: ["documents"], queryFn: listDocuments });
  const reviewQueueQuery = useQuery({
    queryKey: ["review-queue"],
    queryFn: getReviewQueue,
    refetchInterval: 5_000,
  });
  const auditQuery = useQuery({ queryKey: ["audit"], queryFn: listAuditEvents });
  const auditIntegrityQuery = useQuery({
    queryKey: ["audit-integrity"],
    queryFn: getAuditIntegrity,
  });

  // Active review document redactions query
  const redactionsQuery = useQuery({
    queryKey: ["document-redactions", selectedReviewDocId],
    queryFn: () => (selectedReviewDocId ? getDocumentRedactions(selectedReviewDocId) : null),
    enabled: Boolean(selectedReviewDocId),
  });

  // Mutations
  const uploadMutation = useMutation({
    mutationFn: uploadDocument,
    onSuccess: async (doc) => {
      setSelectedFile(null);
      if (inputRef.current) inputRef.current.value = "";
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["documents"] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
        queryClient.invalidateQueries({ queryKey: ["audit-integrity"] }),
      ]);
      setActionNotice(`Document "${doc.filename}" successfully encrypted in vault.`);
    },
  });

  const extractMutation = useMutation({
    mutationFn: (docId: string) => startExtraction(docId),
    onSuccess: async (_, docId) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["documents"] }),
        queryClient.invalidateQueries({ queryKey: ["review-queue"] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
        queryClient.invalidateQueries({ queryKey: ["document-redactions", docId] }),
      ]);
      setSelectedReviewDocId(docId);
      setActiveNav("Review queue");
      setActionNotice("Extraction complete. Review detected Indian PII findings.");
    },
  });

  const updateDecisionMutation = useMutation({
    mutationFn: ({
      docId,
      findingId,
      decision,
      version,
    }: {
      docId: string;
      findingId: string;
      decision: "ACCEPTED" | "REJECTED";
      version: number;
    }) => updateRedactionDecision(docId, findingId, decision, version),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["document-redactions", selectedReviewDocId] }),
        queryClient.invalidateQueries({ queryKey: ["review-queue"] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
      ]);
    },
  });

  const acceptHighConfidenceMutation = useMutation({
    mutationFn: (docId: string) => acceptHighConfidenceRedactions(docId),
    onSuccess: async (res) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["document-redactions", selectedReviewDocId] }),
        queryClient.invalidateQueries({ queryKey: ["review-queue"] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
      ]);
      setActionNotice(`Accepted ${res.accepted_count} high-confidence finding(s).`);
    },
  });

  const approveMutation = useMutation({
    mutationFn: (docId: string) => approveRedactions(docId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["document-redactions", selectedReviewDocId] }),
        queryClient.invalidateQueries({ queryKey: ["review-queue"] }),
        queryClient.invalidateQueries({ queryKey: ["documents"] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
      ]);
      setActionNotice("Redactions approved. Document is marked INDEX_READY with deterministic masked text.");
    },
  });

  // Intelligence state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchTopK, setSearchTopK] = useState(5);
  const [searchDocFilter, setSearchDocFilter] = useState("all");
  const [selectedIndexDocId, setSelectedIndexDocId] = useState("");
  const [copiedCitation, setCopiedCitation] = useState<string | null>(null);
  const [visualEvidence, setVisualEvidence] = useState<VisualEvidenceSelection | null>(null);
  const [visualEvidenceLoading, setVisualEvidenceLoading] = useState<string | null>(null);
  const [visualEvidenceError, setVisualEvidenceError] = useState<string | null>(null);

  const retrievalStatusQuery = useQuery({
    queryKey: ["retrieval-status"],
    queryFn: getRetrievalStatus,
    refetchInterval: 5000,
  });

  const indexMutation = useMutation({
    mutationFn: (docId: string) => indexDocument(docId),
    onSuccess: async (data) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["retrieval-status"] }),
        queryClient.invalidateQueries({ queryKey: ["documents"] }),
        queryClient.invalidateQueries({ queryKey: ["audit"] }),
      ]);
      setActionNotice(`Document indexed into Qdrant (${data.chunk_count} chunks generated).`);
    },
  });

  const searchMutation = useMutation({
    mutationFn: (params: {
      query: string;
      top_k: number;
      permitted_document_ids?: string[];
    }) => searchRetrieval(params),
  });

  const handleCopyCitation = (citation: string) => {
    navigator.clipboard.writeText(citation);
    setCopiedCitation(citation);
    setTimeout(() => setCopiedCitation(null), 2500);
  };

  useEffect(() => {
    return () => {
      if (visualEvidence?.imageUrl) {
        URL.revokeObjectURL(visualEvidence.imageUrl);
      }
    };
  }, [visualEvidence?.imageUrl]);

  const handleOpenVisualEvidence = async (
    item: RetrievalEvidenceItem,
    region: RetrievalVisualRegion
  ) => {
    setVisualEvidenceError(null);
    setVisualEvidenceLoading(region.region_id);
    try {
      const imageBlob = await fetchVisualEvidenceImage(item.chunk_id);
      const imageUrl = URL.createObjectURL(imageBlob);
      setVisualEvidence((previous) => {
        if (previous?.imageUrl) URL.revokeObjectURL(previous.imageUrl);
        return { item, region, imageUrl };
      });
    } catch (error) {
      setVisualEvidenceError(error instanceof Error ? error.message : "Visual evidence unavailable");
    } finally {
      setVisualEvidenceLoading(null);
    }
  };

  const handleCloseVisualEvidence = () => {
    setVisualEvidence((previous) => {
      if (previous?.imageUrl) URL.revokeObjectURL(previous.imageUrl);
      return null;
    });
  };

  const handleSearchSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) return;
    handleCloseVisualEvidence();
    setVisualEvidenceError(null);
    const permittedIds =
      searchDocFilter === "all" ? undefined : [searchDocFilter];
    searchMutation.mutate({
      query: searchQuery.trim(),
      top_k: searchTopK,
      permitted_document_ids: permittedIds,
    });
  };


  const system = statusQuery.data;
  const systemReady = system?.metadata_store.status === "ready";
  const documentCount = documentsQuery.data?.length ?? 0;
  const reviewQueueCount = reviewQueueQuery.data?.filter((q) => q.status === "REVIEW_REQUIRED").length ?? 0;

  // Active review item
  const reviewQueueItems = useMemo(
    () => reviewQueueQuery.data ?? [],
    [reviewQueueQuery.data]
  );
  const currentReviewDoc = useMemo(() => {
    if (!selectedReviewDocId) return reviewQueueItems[0] ?? null;
    return reviewQueueItems.find((d) => d.document_id === selectedReviewDocId) ?? null;
  }, [selectedReviewDocId, reviewQueueItems]);

  const activeDocId = currentReviewDoc?.document_id ?? selectedReviewDocId;
  const redactionsData = redactionsQuery.data;

  // Filter findings for active page and category
  const activeFindings = useMemo(() => {
    if (!redactionsData) return [];
    return redactionsData.findings.filter((f) => {
      const pageMatches = f.page_number === selectedPageNumber;
      const categoryMatches = categoryFilter === "ALL" || f.finding_type === categoryFilter;
      return pageMatches && categoryMatches;
    });
  }, [redactionsData, selectedPageNumber, categoryFilter]);

  // Distinct finding categories for the document
  const categories = useMemo(() => {
    if (!redactionsData) return [];
    const set = new Set(redactionsData.findings.map((f) => f.finding_type));
    return Array.from(set);
  }, [redactionsData]);

  const activePagePreview = useMemo(() => {
    if (!redactionsData) return null;
    return redactionsData.pages.find((p) => p.page_number === selectedPageNumber) ?? null;
  }, [redactionsData, selectedPageNumber]);

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark">
            <ShieldCheck size={23} />
          </div>
          <div>
            <strong>KoshShield AI</strong>
            <span>Sovereign workbench</span>
          </div>
          <button
            className="icon-button mobile-only"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close navigation"
          >
            <X size={19} />
          </button>
        </div>

        <nav aria-label="Primary navigation">
          <span className="nav-label">Workspace</span>
          {navigation.map(({ label, icon: Icon }) => {
            const isActive = activeNav === label;
            return (
              <button
                className={`nav-item ${isActive ? "nav-active" : ""}`}
                key={label}
                onClick={() => {
                  setActiveNav(label);
                  setSidebarOpen(false);
                }}
              >
                <Icon size={18} />
                <span>{label}</span>
                {label === "Review queue" && reviewQueueCount > 0 && (
                  <span className="nav-badge">{reviewQueueCount}</span>
                )}
                {isActive && <ChevronRight size={16} className="nav-arrow" />}
              </button>
            );
          })}
        </nav>

        <div className="boundary-panel">
          <div>
            <Network size={17} />
            <strong>Data boundary</strong>
          </div>
          <p>No external AI endpoints or cloud models are permitted.</p>
          <span>
            <LockKeyhole size={14} /> Air-gapped / Local only
          </span>
        </div>
      </aside>

      {sidebarOpen && (
        <button
          className="sidebar-scrim"
          onClick={() => setSidebarOpen(false)}
          aria-label="Close navigation overlay"
        />
      )}

      <main>
        <header className="topbar">
          <button
            className="icon-button mobile-only"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open navigation"
          >
            <Menu size={20} />
          </button>
          <div className="search-shell">
            <Search size={17} />
            <span>Search secured documents</span>
            <kbd>⌘ K</kbd>
          </div>
          <div className="topbar-status">
            <span className={`connection-dot ${systemReady ? "connection-ready" : ""}`} />
            {statusQuery.isError
              ? "Backend offline"
              : systemReady
                ? "Local services connected"
                : "Checking services"}
          </div>
          <button className="profile-button" aria-label="Current user">
            AK
          </button>
        </header>

        <div className="workspace">
          {actionNotice && (
            <div className="notice notice-info">
              <CheckCircle2 size={18} />
              <span>{actionNotice}</span>
              <button onClick={() => setActionNotice(null)}>Dismiss</button>
            </div>
          )}

          {/* ========================================================= */}
          {/* OVERVIEW TAB */}
          {/* ========================================================= */}
          {activeNav === "Overview" && (
            <>
              <section className="page-heading">
                <div>
                  <span className="eyebrow">Secure operations</span>
                  <h1>Operations overview</h1>
                  <p>Inspect the local trust boundary, system services, and accept encrypted intake.</p>
                </div>
                <div className="mode-chip">
                  <ShieldCheck size={17} /> Air-gapped mode
                </div>
              </section>

              {statusQuery.isError && (
                <div className="notice notice-error">
                  <CircleAlert size={20} />
                  <div>
                    <strong>Local API is unavailable</strong>
                    <span>Start the configured FastAPI service, then retry.</span>
                  </div>
                  <button onClick={() => statusQuery.refetch()}>Retry</button>
                </div>
              )}

              {system && system.vault.status === "not_configured" && (
                <div className="notice notice-warning">
                  <KeyRound size={20} />
                  <div>
                    <strong>Encrypted intake is locked</strong>
                    <span>Configure KOSHSHIELD_MASTER_KEY_BASE64 before uploading documents.</span>
                  </div>
                </div>
              )}

              <section className="service-grid service-grid-5" aria-label="Local service status">
                <ServiceTile
                  icon={LockKeyhole}
                  label="Processing boundary"
                  detail="Local only"
                  status={system ? "ready" : "unavailable"}
                />
                <ServiceTile
                  icon={Archive}
                  label="Encrypted vault"
                  detail="AES-256-GCM"
                  status={system?.vault.status ?? "unavailable"}
                />
                <ServiceTile
                  icon={Database}
                  label="Metadata store"
                  detail={system?.metadata_backend ?? "Not connected"}
                  status={system?.metadata_store.status ?? "unavailable"}
                />
                <ServiceTile
                  icon={ScanLine}
                  label="Local OCR"
                  detail="PaddleOCR"
                  status={system?.ocr.status ?? "unavailable"}
                />
                <ServiceTile
                  icon={Cpu}
                  label="Embeddings"
                  detail="BGE-M3"
                  status={system?.embedding?.status ?? "unavailable"}
                />
                <ServiceTile
                  icon={Server}
                  label="Local model"
                  detail="Qwen3-VL-4B"
                  status={system?.local_model.status ?? "unavailable"}
                />
              </section>


              <div className="content-grid">
                <section className="panel intake-panel">
                  <div className="panel-heading">
                    <div>
                      <span className="section-kicker">Document intake</span>
                      <h2>Accept into encrypted vault</h2>
                    </div>
                    <span className="supported-types">PDF · PNG · JPEG</span>
                  </div>

                  <label className={`upload-zone ${selectedFile ? "upload-selected" : ""}`}>
                    <input
                      ref={inputRef}
                      type="file"
                      accept=".pdf,.png,.jpg,.jpeg,application/pdf,image/png,image/jpeg"
                      onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                    />
                    <div className="upload-icon">
                      <Upload size={24} />
                    </div>
                    {selectedFile ? (
                      <div>
                        <strong>{selectedFile.name}</strong>
                        <span>{formatBytes(selectedFile.size)} · Ready for encrypted intake</span>
                      </div>
                    ) : (
                      <div>
                        <strong>Select a confidential document</strong>
                        <span>Original bytes are encrypted before acceptance is recorded.</span>
                      </div>
                    )}
                  </label>

                  <div className="upload-actions">
                    <div className="intake-proof">
                      <ShieldCheck size={16} />
                      <span>Signature validation · SHA-256 evidence hash · encrypted storage</span>
                    </div>
                    <button
                      className="primary-button"
                      disabled={!selectedFile || uploadMutation.isPending || system?.vault.status !== "ready"}
                      onClick={() => selectedFile && uploadMutation.mutate(selectedFile)}
                    >
                      {uploadMutation.isPending ? "Encrypting…" : "Accept document"}
                    </button>
                  </div>
                  {uploadMutation.isError && <p className="form-error">{uploadMutation.error.message}</p>}
                  {uploadMutation.isSuccess && (
                    <p className="form-success">
                      <CheckCircle2 size={15} /> Document encrypted and audit event recorded.
                    </p>
                  )}
                </section>

                <section className="panel assurance-panel">
                  <div className="panel-heading">
                    <div>
                      <span className="section-kicker">Trust controls</span>
                      <h2>Milestone 2 guarantees</h2>
                    </div>
                    <Activity size={19} />
                  </div>
                  <div className="assurance-list">
                    <div>
                      <ShieldCheck size={19} />
                      <span>
                        <strong>Air-gapped extraction</strong>
                        <small>Native PDF parsing with PyMuPDF; no network calls.</small>
                      </span>
                    </div>
                    <div>
                      <FileLock2 size={19} />
                      <span>
                        <strong>Raw text in vault only</strong>
                        <small>Unmasked extracted text is never stored in DB or logs.</small>
                      </span>
                    </div>
                    <div>
                      <UserCheck size={19} />
                      <span>
                        <strong>Human redaction review</strong>
                        <small>Indexing is blocked until all PII findings are resolved.</small>
                      </span>
                    </div>
                    <div>
                      <History size={19} />
                      <span>
                        <strong>Chained audit trail</strong>
                        <small>Every state transition and redaction decision is cryptographically chained.</small>
                      </span>
                    </div>
                  </div>
                </section>
              </div>

              <div className="content-grid lower-grid">
                <section className="panel documents-panel">
                  <div className="panel-heading">
                    <div>
                      <span className="section-kicker">Recent intake</span>
                      <h2>Vault documents</h2>
                    </div>
                    <button
                      className="text-button"
                      onClick={() => setActiveNav("Documents")}
                    >
                      View all ({documentCount})
                    </button>
                  </div>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Document</th>
                          <th>Status</th>
                          <th>Evidence hash</th>
                          <th>Actions</th>
                        </tr>
                      </thead>
                      <tbody>
                        {documentsQuery.data?.slice(0, 5).map((document) => (
                          <tr key={document.id}>
                            <td>
                              <span className="document-cell">
                                <FileLock2 size={17} />
                                <span>
                                  <strong>{document.filename}</strong>
                                  <small>{formatBytes(document.size_bytes)}</small>
                                </span>
                              </span>
                            </td>
                            <td>
                              <span className={`doc-status-badge status-tag-${document.status.toLowerCase()}`}>
                                {formatStatusLabel(document.status)}
                              </span>
                            </td>
                            <td>
                              <code>{shortHash(document.sha256)}</code>
                            </td>
                            <td>
                              {document.status === "ENCRYPTED" && (
                                <button
                                  className="action-button-small"
                                  disabled={extractMutation.isPending}
                                  onClick={() => extractMutation.mutate(document.id)}
                                >
                                  {extractMutation.isPending ? "Extracting…" : "Extract PII"}
                                </button>
                              )}
                              {document.status === "REVIEW_REQUIRED" && (
                                <button
                                  className="action-button-primary-small"
                                  onClick={() => {
                                    setSelectedReviewDocId(document.id);
                                    setActiveNav("Review queue");
                                  }}
                                >
                                  Review findings
                                </button>
                              )}
                              {document.status === "INDEX_READY" && (
                                <span className="label-index-ready">
                                  <Check size={13} /> Ready
                                </span>
                              )}
                            </td>
                          </tr>
                        ))}
                        {!documentsQuery.data?.length && (
                          <tr>
                            <td colSpan={4} className="empty-cell">
                              No documents have entered the encrypted vault.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </section>

                <section className="panel audit-panel">
                  <div className="panel-heading">
                    <div>
                      <span className="section-kicker">Audit snapshot</span>
                      <h2>Latest events</h2>
                    </div>
                    <StatusBadge
                      status={auditIntegrityQuery.data?.valid ? "ready" : "unavailable"}
                    />
                  </div>
                  <div className="audit-list">
                    {auditQuery.data?.map((event) => (
                      <div className="audit-row" key={event.id}>
                        <div className="audit-icon">
                          <FileCheck2 size={16} />
                        </div>
                        <span>
                          <strong>{event.event_type.replace(/[._]/g, " ")}</strong>
                          <small>
                            {event.actor_id} · {formatTime(event.created_at)}
                          </small>
                        </span>
                        <code>{event.event_hash.slice(0, 7)}</code>
                      </div>
                    ))}
                    {!auditQuery.data?.length && (
                      <p className="empty-audit">Audit events appear after a secured action.</p>
                    )}
                  </div>
                </section>
              </div>
            </>
          )}

          {/* ========================================================= */}
          {/* DOCUMENTS TAB */}
          {/* ========================================================= */}
          {activeNav === "Documents" && (
            <>
              <section className="page-heading">
                <div>
                  <span className="eyebrow">Document inventory</span>
                  <h1>Vault documents</h1>
                  <p>All files stored in the AES-256-GCM encrypted vault with explicit processing states.</p>
                </div>
                <button
                  className="primary-button"
                  onClick={() => setActiveNav("Overview")}
                >
                  <Upload size={14} style={{ marginRight: 6 }} /> Intake new document
                </button>
              </section>

              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <span className="section-kicker">State ledger</span>
                    <h2>All vault records ({documentCount})</h2>
                  </div>
                  <span className="supported-types">Local encrypted store</span>
                </div>

                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Document</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Evidence hash</th>
                        <th>Intake timestamp</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {documentsQuery.data?.map((doc: DocumentRecord) => (
                        <tr key={doc.id}>
                          <td>
                            <span className="document-cell">
                              <FileLock2 size={18} />
                              <span>
                                <strong>{doc.filename}</strong>
                                <small>{formatBytes(doc.size_bytes)}</small>
                              </span>
                            </span>
                          </td>
                          <td>
                            <span className="type-tag">{doc.media_type.split("/")[1]?.toUpperCase() ?? "PDF"}</span>
                          </td>
                          <td>
                            <span className={`doc-status-badge status-tag-${doc.status.toLowerCase()}`}>
                              {formatStatusLabel(doc.status)}
                            </span>
                          </td>
                          <td>
                            <code>{shortHash(doc.sha256)}</code>
                          </td>
                          <td>{formatTime(doc.created_at)}</td>
                          <td>
                            {doc.status === "ENCRYPTED" && (
                              <button
                                className="action-button-small"
                                disabled={extractMutation.isPending}
                                onClick={() => extractMutation.mutate(doc.id)}
                              >
                                {extractMutation.isPending ? "Extracting…" : "Extract PII"}
                              </button>
                            )}
                            {doc.status === "EXTRACTION_FAILED" && (
                              <button
                                className="action-button-small"
                                disabled={extractMutation.isPending}
                                onClick={() => extractMutation.mutate(doc.id)}
                              >
                                Retry extraction
                              </button>
                            )}
                            {doc.status === "REVIEW_REQUIRED" && (
                              <button
                                className="action-button-primary-small"
                                onClick={() => {
                                  setSelectedReviewDocId(doc.id);
                                  setActiveNav("Review queue");
                                }}
                              >
                                Review findings
                              </button>
                            )}
                            {doc.status === "INDEX_READY" && (
                              <button
                                className="action-button-outline-small"
                                onClick={() => {
                                  setSelectedReviewDocId(doc.id);
                                  setActiveNav("Review queue");
                                }}
                              >
                                <Eye size={12} style={{ marginRight: 4 }} /> View masked
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                      {!documentsQuery.data?.length && (
                        <tr>
                          <td colSpan={6} className="empty-cell">
                            No documents in vault yet. Use Document Intake on the Overview tab.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}

          {/* ========================================================= */}
          {/* REVIEW QUEUE TAB */}
          {/* ========================================================= */}
          {activeNav === "Review queue" && (
            <div className="review-workspace">
              <section className="page-heading">
                <div>
                  <span className="eyebrow">Human-in-the-loop</span>
                  <h1>Redaction review workspace</h1>
                  <p>Inspect detected Indian PII findings, accept or reject redactions, and finalize privacy masking.</p>
                </div>
                <div className="review-header-stats">
                  {currentReviewDoc && (
                    <span className={`doc-status-badge status-tag-${currentReviewDoc.status.toLowerCase()}`}>
                      {formatStatusLabel(currentReviewDoc.status)}
                    </span>
                  )}
                </div>
              </section>

              {/* Document selection bar */}
              <div className="review-doc-bar">
                <span className="doc-bar-label">Document:</span>
                <div className="doc-pills">
                  {reviewQueueItems.map((item) => (
                    <button
                      key={item.document_id}
                      className={`doc-pill ${item.document_id === activeDocId ? "doc-pill-active" : ""}`}
                      onClick={() => {
                        setSelectedReviewDocId(item.document_id);
                        setSelectedPageNumber(1);
                        setCategoryFilter("ALL");
                      }}
                    >
                      <FileText size={14} />
                      <span>{item.filename}</span>
                      {item.pending_findings > 0 && (
                        <span className="pill-badge">{item.pending_findings} pending</span>
                      )}
                    </button>
                  ))}
                  {!reviewQueueItems.length && (
                    <span className="empty-queue-note">
                      No documents currently require review. Extract a document from the Documents tab.
                    </span>
                  )}
                </div>
              </div>

              {activeDocId && redactionsData && (
                <>
                  {/* Summary Bar */}
                  <div className="review-stats-grid">
                    <div className="stat-card">
                      <span className="stat-label">Total findings</span>
                      <strong className="stat-value">{redactionsData.total_findings}</strong>
                    </div>
                    <div className="stat-card stat-pending">
                      <span className="stat-label">Pending review</span>
                      <strong className="stat-value">{redactionsData.unresolved_count}</strong>
                    </div>
                    <div className="stat-card stat-accepted">
                      <span className="stat-label">Accepted</span>
                      <strong className="stat-value">
                        {redactionsData.findings.filter((f) => f.status === "ACCEPTED").length}
                      </strong>
                    </div>
                    <div className="stat-card stat-rejected">
                      <span className="stat-label">Rejected</span>
                      <strong className="stat-value">
                        {redactionsData.findings.filter((f) => f.status === "REJECTED").length}
                      </strong>
                    </div>
                    <div className="stat-card">
                      <span className="stat-label">Pages</span>
                      <strong className="stat-value">{redactionsData.total_pages}</strong>
                    </div>
                  </div>

                  {/* Page selector bar */}
                  {redactionsData.total_pages > 1 && (
                    <div className="page-bar">
                      <span className="page-bar-label">Pages:</span>
                      <div className="page-tabs">
                        {Array.from({ length: redactionsData.total_pages }, (_, i) => i + 1).map((pg) => {
                          const pageFindingsCount = redactionsData.findings.filter((f) => f.page_number === pg).length;
                          return (
                            <button
                              key={pg}
                              className={`page-tab ${pg === selectedPageNumber ? "page-tab-active" : ""}`}
                              onClick={() => setSelectedPageNumber(pg)}
                            >
                              Page {pg}
                              {pageFindingsCount > 0 && <span className="tab-pill">{pageFindingsCount}</span>}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Dual pane workspace */}
                  <div className="workspace-dual-pane">
                    {/* Left: Masked text preview */}
                    <div className="preview-pane panel">
                      <div className="panel-heading">
                        <div>
                          <span className="section-kicker">Page {selectedPageNumber} preview</span>
                          <h2>Masked text output</h2>
                        </div>
                        <span className="extraction-method-tag">
                          <Layers size={13} style={{ marginRight: 4 }} />
                          {activePagePreview?.extraction_method === "native_pdf"
                            ? "Native PDF (PyMuPDF)"
                            : activePagePreview?.extraction_method ?? "Extracted"}
                        </span>
                      </div>

                      <div className="preview-text-box">
                        {activePagePreview?.masked_text ? (
                          <pre className="masked-pre">{activePagePreview.masked_text}</pre>
                        ) : (
                          <div className="masked-context-stream">
                            <p className="stream-caption">
                              Privacy preview with active redaction placeholders:
                            </p>
                            {activeFindings.length > 0 ? (
                              <div className="context-snippets-list">
                                {activeFindings.map((f) => (
                                  <div className="snippet-row" key={f.id}>
                                    <span className="snippet-type">{formatFindingLabel(f.finding_type)}</span>
                                    <code className="snippet-code">{f.masked_context}</code>
                                  </div>
                                ))}
                              </div>
                            ) : (
                              <p className="no-findings-page">No findings on page {selectedPageNumber}.</p>
                            )}
                          </div>
                        )}
                      </div>

                      <div className="preview-footer-note">
                        <LockKeyhole size={13} />
                        <span>Raw original text remains encrypted in vault; only masked content is shown.</span>
                      </div>
                    </div>

                    {/* Right: Findings list & review actions */}
                    <div className="findings-pane panel">
                      <div className="panel-heading">
                        <div>
                          <span className="section-kicker">PII Detections</span>
                          <h2>Findings ({activeFindings.length})</h2>
                        </div>
                        {redactionsData.status === "REVIEW_REQUIRED" && (
                          <button
                            className="bulk-accept-button"
                            disabled={acceptHighConfidenceMutation.isPending}
                            onClick={() => acceptHighConfidenceMutation.mutate(activeDocId)}
                            title="Accept all findings with ≥85% confidence"
                          >
                            <Sparkles size={14} />
                            <span>Accept high-confidence</span>
                          </button>
                        )}
                      </div>

                      {/* Category filters */}
                      {categories.length > 1 && (
                        <div className="category-chips">
                          <button
                            className={`chip ${categoryFilter === "ALL" ? "chip-active" : ""}`}
                            onClick={() => setCategoryFilter("ALL")}
                          >
                            All
                          </button>
                          {categories.map((cat) => (
                            <button
                              key={cat}
                              className={`chip ${categoryFilter === cat ? "chip-active" : ""}`}
                              onClick={() => setCategoryFilter(cat)}
                            >
                              {cat}
                            </button>
                          ))}
                        </div>
                      )}

                      {/* Findings cards */}
                      <div className="findings-list">
                        {activeFindings.map((finding: RedactionFinding) => (
                          <div
                            key={finding.id}
                            className={`finding-card finding-${finding.status.toLowerCase()}`}
                          >
                            <div className="finding-header">
                              <div className="finding-title">
                                <FindingTypeIcon type={finding.finding_type} />
                                <strong>{formatFindingLabel(finding.finding_type)}</strong>
                              </div>
                              <div className="finding-meta">
                                <span className="confidence-chip">
                                  {formatConfidence(finding.confidence)} confidence
                                </span>
                                <span className={`decision-badge decision-${finding.status.toLowerCase()}`}>
                                  {finding.status}
                                </span>
                              </div>
                            </div>

                            <div className="context-box">
                              <code>{finding.masked_context}</code>
                            </div>

                            <div className="finding-footer">
                              <span className="source-tag">{finding.detection_source}</span>

                              {redactionsData.status === "REVIEW_REQUIRED" && (
                                <div className="decision-actions">
                                  <button
                                    className={`btn-accept ${finding.status === "ACCEPTED" ? "btn-active" : ""}`}
                                    disabled={updateDecisionMutation.isPending}
                                    onClick={() =>
                                      updateDecisionMutation.mutate({
                                        docId: activeDocId,
                                        findingId: finding.id,
                                        decision: "ACCEPTED",
                                        version: finding.version,
                                      })
                                    }
                                  >
                                    <Check size={13} /> Accept
                                  </button>
                                  <button
                                    className={`btn-reject ${finding.status === "REJECTED" ? "btn-active" : ""}`}
                                    disabled={updateDecisionMutation.isPending}
                                    onClick={() =>
                                      updateDecisionMutation.mutate({
                                        docId: activeDocId,
                                        findingId: finding.id,
                                        decision: "REJECTED",
                                        version: finding.version,
                                      })
                                    }
                                  >
                                    <X size={13} /> Reject
                                  </button>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}

                        {!activeFindings.length && (
                          <div className="empty-findings">
                            <CheckCircle2 size={24} style={{ color: "var(--green-700)" }} />
                            <strong>No findings to review</strong>
                            <span>No detections matching the current filters on this page.</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Approval action banner */}
                  <div className="approval-banner panel">
                    <div className="approval-status-info">
                      {redactionsData.unresolved_count > 0 ? (
                        <div className="unresolved-warning">
                          <CircleAlert size={20} />
                          <div>
                            <strong>
                              {redactionsData.unresolved_count} unresolved finding(s) require human decision
                            </strong>
                            <span>
                              All findings must be accepted or rejected before privacy redaction can be approved.
                            </span>
                          </div>
                        </div>
                      ) : redactionsData.status === "REVIEW_REQUIRED" ? (
                        <div className="resolved-ready">
                          <CheckCircle2 size={20} />
                          <div>
                            <strong>All findings resolved</strong>
                            <span>
                              Ready to finalize redactions, produce deterministic masked text, and mark INDEX_READY.
                            </span>
                          </div>
                        </div>
                      ) : (
                        <div className="resolved-ready">
                          <CheckCircle2 size={20} />
                          <div>
                            <strong>Document is INDEX_READY</strong>
                            <span>Masked copy generated and approved. Ready for local indexing in Milestone 3.</span>
                          </div>
                        </div>
                      )}
                    </div>

                    {redactionsData.status === "REVIEW_REQUIRED" && (
                      <button
                        className="approve-primary-button"
                        disabled={redactionsData.unresolved_count > 0 || approveMutation.isPending}
                        onClick={() => approveMutation.mutate(activeDocId)}
                      >
                        {approveMutation.isPending ? (
                          <>
                            <Loader2 size={16} className="spin-loader" /> Finalizing…
                          </>
                        ) : (
                          <>
                            <CheckCircle2 size={16} style={{ marginRight: 6 }} /> Approve redactions & Mark INDEX_READY
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Intelligence Workspace Tab */}
          {activeNav === "Intelligence" && (
            <div className="intelligence-workspace">
              {/* Status and telemetry bar */}
              <div className="retrieval-telemetry-grid">
                <div className="telemetry-card">
                  <span className="telemetry-kicker">Vector Store</span>
                  <div className="telemetry-content">
                    <Database size={20} />
                    <div>
                      <strong>Qdrant Local</strong>
                      <span className="telemetry-sub">{retrievalStatusQuery.data?.collection_name ?? "koshshield_masked_docs"}</span>
                    </div>
                  </div>
                  <div className="telemetry-status">
                    <StatusBadge status={retrievalStatusQuery.data?.vector_store_status ?? "unavailable"} />
                  </div>
                </div>

                <div className="telemetry-card">
                  <span className="telemetry-kicker">Embedding Model</span>
                  <div className="telemetry-content">
                    <Cpu size={20} />
                    <div>
                      <strong>BGE-M3 Hybrid</strong>
                      <span className="telemetry-sub">Dense (1024) + Lexical</span>
                    </div>
                  </div>
                  <div className="telemetry-status">
                    <StatusBadge status={retrievalStatusQuery.data?.embedding_model_status ?? "unavailable"} />
                  </div>
                </div>

                <div className="telemetry-card">
                  <span className="telemetry-kicker">Indexed Corpus</span>
                  <div className="telemetry-content">
                    <Layers size={20} />
                    <div>
                      <strong>{retrievalStatusQuery.data?.indexed_documents_count ?? 0} Documents</strong>
                      <span className="telemetry-sub">INDEXED in vector store</span>
                    </div>
                  </div>
                </div>

                <div className="telemetry-card">
                  <span className="telemetry-kicker">Vector Points</span>
                  <div className="telemetry-content">
                    <Fingerprint size={20} />
                    <div>
                      <strong>{retrievalStatusQuery.data?.total_chunks ?? 0} Chunks</strong>
                      <span className="telemetry-sub">Masked passage points</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Indexing Command Panel */}
              <section className="panel index-command-panel">
                <div className="panel-heading">
                  <div>
                    <span className="section-kicker">Corpus Indexing</span>
                    <h2>Vectorize Approved Documents</h2>
                  </div>
                  <span className="privacy-pill">
                    <ShieldCheck size={14} /> Privacy Gate Enforced
                  </span>
                </div>

                <p className="panel-instruction">
                  Only documents in <strong>INDEX_READY</strong> or <strong>INDEXED</strong> status with zero unresolved findings are eligible for local Qdrant vectorization. Original documents are never decrypted; only approved masked text is chunked and embedded.
                </p>

                <div className="index-action-row">
                  <select
                    className="doc-select-input"
                    value={selectedIndexDocId}
                    onChange={(e) => setSelectedIndexDocId(e.target.value)}
                  >
                    <option value="">-- Select an approved document to index --</option>
                    {documentsQuery.data
                      ?.filter((d) => d.status === "INDEX_READY" || d.status === "INDEXED")
                      .map((d) => (
                        <option key={d.id} value={d.id}>
                          {d.filename} ({formatStatusLabel(d.status)} · {d.size_bytes} B)
                        </option>
                      ))}
                  </select>

                  <button
                    className="primary-button"
                    disabled={!selectedIndexDocId || indexMutation.isPending}
                    onClick={() => selectedIndexDocId && indexMutation.mutate(selectedIndexDocId)}
                  >
                    {indexMutation.isPending ? (
                      <>
                        <Loader2 size={16} className="spin-loader" /> Vectorizing…
                      </>
                    ) : (
                      <>
                        <Layers size={16} style={{ marginRight: 6 }} /> Index into Qdrant
                      </>
                    )}
                  </button>
                </div>

                {indexMutation.isError && (
                  <p className="form-error" style={{ marginTop: "0.75rem" }}>
                    <CircleAlert size={15} style={{ verticalAlign: "middle", marginRight: 4 }} />
                    {indexMutation.error.message}
                  </p>
                )}
                {indexMutation.isSuccess && (
                  <p className="form-success" style={{ marginTop: "0.75rem" }}>
                    <CheckCircle2 size={15} /> Document indexed successfully ({indexMutation.data.chunk_count} chunks). Audit record emitted.
                  </p>
                )}
              </section>

              {/* Hybrid Search Panel */}
              <section className="panel hybrid-search-panel">
                <div className="panel-heading">
                  <div>
                    <span className="section-kicker">Hybrid Retrieval</span>
                    <h2>Dense & Sparse Search with Citations</h2>
                  </div>
                  <span className="privacy-pill">
                    <Fingerprint size={14} /> Cryptographic Citations
                  </span>
                </div>

                <form onSubmit={handleSearchSubmit} className="search-form">
                  <div className="search-input-wrapper">
                    <Search size={18} className="search-icon-left" />
                    <input
                      type="text"
                      className="search-input"
                      placeholder="Search confidential intelligence (e.g., procurement guidelines, tender deadlines, budget allocation...)"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                    <button
                      type="submit"
                      className="search-submit-button"
                      disabled={!searchQuery.trim() || searchMutation.isPending}
                    >
                      {searchMutation.isPending ? (
                        <>
                          <Loader2 size={16} className="spin-loader" /> Searching…
                        </>
                      ) : (
                        "Search Evidence"
                      )}
                    </button>
                  </div>

                  <div className="search-filters-row">
                    <div className="filter-group">
                      <label htmlFor="topk-select">Top Results:</label>
                      <select
                        id="topk-select"
                        value={searchTopK}
                        onChange={(e) => setSearchTopK(Number(e.target.value))}
                        className="filter-select"
                      >
                        <option value={3}>Top 3</option>
                        <option value={5}>Top 5</option>
                        <option value={10}>Top 10</option>
                        <option value={20}>Top 20</option>
                      </select>
                    </div>

                    <div className="filter-group">
                      <label htmlFor="docfilter-select">Scope:</label>
                      <select
                        id="docfilter-select"
                        value={searchDocFilter}
                        onChange={(e) => setSearchDocFilter(e.target.value)}
                        className="filter-select"
                      >
                        <option value="all">All Permitted Documents</option>
                        {documentsQuery.data
                          ?.filter((d) => d.status === "INDEXED")
                          .map((d) => (
                            <option key={d.id} value={d.id}>
                              {d.filename}
                            </option>
                          ))}
                      </select>
                    </div>

                    <div className="filter-hint">
                      <span>Reciprocal Rank Fusion (RRF: k=60) · Strict Tenant Isolation</span>
                    </div>
                  </div>
                </form>

                {/* Search Feedback & Results Area */}
                <div className="search-results-area">
                  {searchMutation.isPending && (
                    <div className="search-loading-state">
                      <Loader2 size={28} className="spin-loader" />
                      <strong>Generating BGE-M3 query representations…</strong>
                      <span>Executing dual dense + sparse search and fusing ranked candidates with RRF.</span>
                    </div>
                  )}

                  {searchMutation.isError && (
                    <div className="search-error-state">
                      <CircleAlert size={28} />
                      <strong>Retrieval query failed</strong>
                      <span>{searchMutation.error.message}</span>
                    </div>
                  )}

                  {searchMutation.isSuccess && searchMutation.data.results.length === 0 && (
                    <div className="search-empty-state">
                      <FileSearch size={32} />
                      <strong>No evidence matches found</strong>
                      <span>No chunks matching the query were found within your authorized scope. Try expanding keywords or indexing additional documents.</span>
                    </div>
                  )}

                  {searchMutation.isSuccess && searchMutation.data.results.length > 0 && (
                    <div className="evidence-results-list">
                      <div className="evidence-summary-bar">
                        <span>
                          <strong>{searchMutation.data.total_found}</strong> verified evidence passage(s) retrieved
                        </span>
                        <div className="evidence-query-meta">
                          <span>Query: <code>{searchMutation.data.query_length} chars ({searchMutation.data.duration_ms}ms)</code></span>
                          <span className="fusion-pill">RRF Fused</span>
                        </div>
                      </div>

                      {searchMutation.data.results.map((item) => (
                        <article key={item.chunk_id} className="evidence-card">
                          <header className="evidence-card-header">
                            <div className="evidence-rank-group">
                              <span className="rank-badge">#{item.rank}</span>
                              <span className="fused-score-badge">Score: {item.fused_score}</span>
                              <div className="source-badges-group">
                                {item.sources.map((s) => (
                                  <span key={s} className={`source-badge source-${s}`}>
                                    {sourceLabel(s)}
                                  </span>
                                ))}
                              </div>
                            </div>

                            <div className="citation-container">
                              <code className="citation-code">{item.citation_label}</code>
                              <button
                                type="button"
                                className="copy-citation-btn"
                                title="Copy citation to clipboard"
                                onClick={() => handleCopyCitation(item.citation_label)}
                              >
                                {copiedCitation === item.citation_label ? (
                                  <Check size={14} style={{ color: "var(--green-700)" }} />
                                ) : (
                                  <Copy size={14} />
                                )}
                              </button>
                            </div>
                          </header>

                          <div className="evidence-text-snippet">
                            <p>{item.masked_snippet}</p>
                          </div>

                          {item.visual_regions.length > 0 && (
                            <div className="visual-region-strip">
                              <div className="visual-region-strip-label">
                                <ImageIcon size={14} />
                                <span>{item.visual_regions.length} visual region(s)</span>
                              </div>
                              <div className="visual-region-buttons">
                                {item.visual_regions.map((region) => (
                                  <button
                                    type="button"
                                    key={region.region_id}
                                    className="visual-region-button"
                                    disabled={
                                      !region.image_available ||
                                      visualEvidenceLoading === region.region_id
                                    }
                                    onClick={() => handleOpenVisualEvidence(item, region)}
                                  >
                                    {visualEvidenceLoading === region.region_id ? (
                                      <Loader2 size={13} className="spin-loader" />
                                    ) : (
                                      <Eye size={13} />
                                    )}
                                    <span>{regionLabel(region.region_type)}</span>
                                  </button>
                                ))}
                              </div>
                            </div>
                          )}

                          <footer className="evidence-card-footer">
                            <div className="meta-item">
                              <FileText size={13} />
                              <span>{item.document_filename}</span>
                            </div>
                            <div className="meta-item">
                              <BookOpen size={13} />
                              <span>Page {item.page_number}</span>
                            </div>
                            <div className="meta-item">
                              <Fingerprint size={13} />
                              <span>Evidence: <code>{item.evidence_hash.slice(0, 16)}…</code></span>
                            </div>
                            <div className="meta-item">
                              <ShieldCheck size={13} />
                              <span>Redaction v{item.redaction_version}</span>
                            </div>
                          </footer>
                        </article>
                      ))}

                      {visualEvidenceError && (
                        <div className="visual-evidence-error">
                          <CircleAlert size={17} />
                          <span>{visualEvidenceError}</span>
                        </div>
                      )}

                      {visualEvidence && (
                        <section className="visual-evidence-viewer" aria-label="Visual evidence">
                          <div className="visual-evidence-copy">
                            <span className="section-kicker">Authorized page evidence</span>
                            <h3>{regionLabel(visualEvidence.region.region_type)}</h3>
                            <p>{visualEvidence.region.caption}</p>
                            <div className="visual-evidence-meta">
                              <span>{visualEvidence.item.document_filename}</span>
                              <span>Page {visualEvidence.region.page_number}</span>
                              <span>
                                Image: <code>{visualEvidence.region.image_sha256?.slice(0, 16)}...</code>
                              </span>
                            </div>
                          </div>
                          <button
                            type="button"
                            className="visual-evidence-close"
                            onClick={handleCloseVisualEvidence}
                            aria-label="Close visual evidence"
                          >
                            <X size={16} />
                          </button>
                          <div className="visual-evidence-image-frame">
                            <NextImage
                              src={visualEvidence.imageUrl}
                              alt={`Page ${visualEvidence.region.page_number} evidence from ${visualEvidence.item.document_filename}`}
                              width={Math.round(visualEvidence.region.page_width ?? 900)}
                              height={Math.round(visualEvidence.region.page_height ?? 1200)}
                              unoptimized
                            />
                            {regionHighlightStyle(visualEvidence.region) && (
                              <span
                                className="visual-region-highlight"
                                style={regionHighlightStyle(visualEvidence.region)}
                              />
                            )}
                          </div>
                        </section>
                      )}
                    </div>
                  )}

                  {!searchMutation.isPending && !searchMutation.isSuccess && !searchMutation.isError && (
                    <div className="search-idle-state">
                      <FileSearch size={36} />
                      <strong>Intelligence evidence workspace ready</strong>
                      <span>
                        Enter a keyword or natural language query above to retrieve dense and sparse candidate passages from authorized documents.
                      </span>
                    </div>
                  )}
                </div>
              </section>
            </div>
          )}

          {/* Fallback for other tabs */}
          {(activeNav === "Approvals" || activeNav === "Audit trail") && (
            <section className="panel placeholder-panel">
              <div className="panel-heading">
                <div>
                  <span className="section-kicker">Roadmap boundary</span>
                  <h2>{activeNav}</h2>
                </div>
              </div>
              <p className="placeholder-text">
                {activeNav === "Approvals" &&
                  "Policy-gated agent tools and Docker sandbox approvals are scheduled for Milestone 5."}
                {activeNav === "Audit trail" &&
                  "Full audit chain inspection dashboard is scheduled for Milestone 7; recent audit events are available on Overview."}
              </p>
              <button className="primary-button" onClick={() => setActiveNav("Overview")}>
                Return to Overview
              </button>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
