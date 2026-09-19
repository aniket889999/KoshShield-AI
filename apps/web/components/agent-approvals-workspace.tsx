"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Ban,
  Calculator,
  Check,
  CircleAlert,
  Clock3,
  Code2,
  FileCheck2,
  FileOutput,
  FileSpreadsheet,
  FileText,
  Fingerprint,
  Loader2,
  LockKeyhole,
  Play,
  ShieldCheck,
  Sparkles,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";

import {
  decideAgentApproval,
  executeAgentAction,
  getAgentRunDeliverable,
  listAgentRuns,
  listDocuments,
  proposeAgentAction,
  type AgentRun,
} from "@/lib/api";
import { formatStatusLabel, formatTime, shortHash } from "@/lib/format";

type AllowedTool =
  | "calculator"
  | "calculate_procurement_comparison"
  | "draft_approval_note"
  | "draft_report"
  | "generate_verified_code"
  | "document_report"
  | "summarize_document";

type Classification = "INTERNAL" | "CONFIDENTIAL" | "RESTRICTED";

function toolLabel(toolName: string) {
  if (toolName === "calculator") return "Calculator";
  if (toolName === "calculate_procurement_comparison") return "Procurement comparison";
  if (toolName === "draft_approval_note") return "Draft approval note";
  if (toolName === "draft_report") return "Draft report";
  if (toolName === "generate_verified_code") return "Verified code generator";
  if (toolName === "document_report") return "Document report";
  if (toolName === "summarize_document") return "Summarize document (Neural)";
  return formatStatusLabel(toolName);
}

function toolIcon(toolName: string) {
  if (toolName === "calculator") return <Calculator size={17} />;
  if (toolName === "calculate_procurement_comparison") return <FileSpreadsheet size={17} />;
  if (toolName === "draft_approval_note") return <FileCheck2 size={17} />;
  if (toolName === "draft_report" || toolName === "document_report") return <FileOutput size={17} />;
  if (toolName === "generate_verified_code") return <Code2 size={17} />;
  if (toolName === "summarize_document") return <Sparkles size={17} />;
  return <Ban size={17} />;
}

function resultText(run: AgentRun) {
  if (run.result?.output.value) return `Result: ${run.result.output.value}`;
  if (run.result?.output.content) return run.result.output.content;
  return null;
}

function DeliverableCard({ runId }: { runId: string }) {
  const query = useQuery({
    queryKey: ["agent-deliverable", runId],
    queryFn: () => getAgentRunDeliverable(runId),
  });

  if (query.isLoading) {
    return (
      <div className="agent-deliverable-box">
        <span style={{ fontSize: "10px", color: "var(--ink-500)" }}>
          <Loader2 size={12} className="spin-loader" style={{ display: "inline", marginRight: "4px" }} />
          Retrieving verified deliverable...
        </span>
      </div>
    );
  }

  if (query.error || !query.data) {
    return (
      <div className="agent-deliverable-box">
        <span style={{ fontSize: "10px", color: "var(--red-700)" }}>
          {query.error instanceof Error ? query.error.message : "Deliverable unavailable"}
        </span>
      </div>
    );
  }

  const deliverable = query.data;

  return (
    <div className="agent-deliverable-box">
      <div className="agent-deliverable-header">
        <span className="agent-deliverable-title">{deliverable.title}</span>
        <span className="agent-deliverable-badge">{deliverable.provenance.verification_status}</span>
      </div>
      <div className="agent-deliverable-meta">
        <span>Action: {deliverable.action_name}</span>
        <span>Approval: {shortHash(deliverable.approval_id)}</span>
        <span>Evidence items: {deliverable.provenance.evidence_identities?.length ?? 0}</span>
        <span>Generated: {formatTime(deliverable.provenance.generated_at)}</span>
      </div>
      {deliverable.citations?.length > 0 && (
        <div className="agent-deliverable-citations">
          {deliverable.citations.map((c, i) => (
            <div key={i} className="agent-citation-card">
              <strong>Chunk {shortHash(c.chunk_id)} · Page {c.page_number}</strong>
              <p>&ldquo;{c.masked_snippet}&rdquo;</p>
            </div>
          ))}
        </div>
      )}
      <p className="agent-deliverable-disclaimer">{deliverable.provenance.disclaimer}</p>
    </div>
  );
}

export function AgentApprovalsWorkspace() {
  const queryClient = useQueryClient();
  const [tool, setTool] = useState<AllowedTool>("calculator");
  const [classification, setClassification] = useState<Classification>("CONFIDENTIAL");
  const [expression, setExpression] = useState("1250 * 18 / 100");
  const [documentId, setDocumentId] = useState("");
  const [procurementBase, setProcurementBase] = useState("50000");
  const [noteSubject, setNoteSubject] = useState("Sanction for IT Hardware Procurement");
  const [reportTitle, setReportTitle] = useState("Audit Compliance Progress Review");
  const [codeTemplate, setCodeTemplate] = useState("audit_hasher");
  const [expandedDeliverableRunId, setExpandedDeliverableRunId] = useState<string | null>(null);

  const runsQuery = useQuery({
    queryKey: ["agent-runs"],
    queryFn: listAgentRuns,
    refetchInterval: 4_000,
  });
  const documentsQuery = useQuery({ queryKey: ["documents"], queryFn: listDocuments });
  const indexedDocuments = useMemo(
    () => documentsQuery.data?.filter((document) => document.status === "INDEXED") ?? [],
    [documentsQuery.data]
  );

  const refreshAgentState = async () => {
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: ["agent-runs"] }),
      queryClient.invalidateQueries({ queryKey: ["audit"] }),
      queryClient.invalidateQueries({ queryKey: ["audit-integrity"] }),
    ]);
  };

  const buildToolArguments = (): Record<string, unknown> => {
    if (tool === "calculator") {
      return { expression };
    }
    if (tool === "calculate_procurement_comparison") {
      const baseNum = Number(procurementBase) || 50000;
      return {
        document_id: documentId,
        base_amount: baseNum,
        comparison_items: [
          { name: "Vendor Alpha (L1)", amount: Math.round(baseNum * 0.95) },
          { name: "Vendor Beta (L2)", amount: Math.round(baseNum * 1.05) },
        ],
      };
    }
    if (tool === "draft_approval_note") {
      return {
        document_id: documentId,
        note_subject: noteSubject || "Administrative Approval",
        requester_department: "Finance & Operations",
      };
    }
    if (tool === "draft_report") {
      return {
        document_id: documentId,
        report_title: reportTitle || "Governance Report",
        sections: ["Executive Summary", "Findings", "Recommendations"],
      };
    }
    if (tool === "generate_verified_code") {
      return {
        template_name: codeTemplate,
        module_name: "sanitized_ingest",
      };
    }
    if (tool === "summarize_document") {
      return {
        document_id: documentId,
        max_length_words: 200,
      };
    }
    return { document_id: documentId };
  };

  const proposeMutation = useMutation({
    mutationFn: (blockedTool?: string) =>
      proposeAgentAction(
        blockedTool
          ? {
              tool_name: blockedTool,
              classification: "RESTRICTED",
              arguments: { destination: "external-network" },
            }
          : {
              tool_name: tool,
              classification,
              arguments: buildToolArguments(),
            }
      ),
    onSuccess: refreshAgentState,
  });

  const approvalMutation = useMutation({
    mutationFn: ({ run, decision }: { run: AgentRun; decision: "APPROVED" | "REJECTED" }) =>
      decideAgentApproval(run.id, decision, run.version),
    onSuccess: refreshAgentState,
  });

  const executeMutation = useMutation({
    mutationFn: (run: AgentRun) => executeAgentAction(run.id, run.version),
    onSuccess: refreshAgentState,
  });

  const runs = runsQuery.data ?? [];
  const pendingCount = runs.filter((run) => run.status === "APPROVAL_PENDING").length;
  const completedCount = runs.filter((run) => run.status === "COMPLETED").length;
  const rejectedCount = runs.filter((run) => run.status === "REJECTED").length;
  const mutationError =
    proposeMutation.error ?? approvalMutation.error ?? executeMutation.error ?? runsQuery.error;

  const canSubmit =
    tool === "calculator"
      ? Boolean(expression.trim())
      : tool === "generate_verified_code"
      ? Boolean(codeTemplate)
      : Boolean(documentId);

  return (
    <div className="agent-workspace">
      <section className="page-heading">
        <div>
          <span className="eyebrow">Policy-gated agent</span>
          <h1>Approvals</h1>
          <p>Review and execute restricted local actions through deterministic policy and human approval.</p>
        </div>
        <span className="mode-chip">
          <ShieldCheck size={15} /> Policy active
        </span>
      </section>

      <div className="agent-metrics">
        <div>
          <Clock3 size={18} />
          <span>Pending</span>
          <strong>{pendingCount}</strong>
        </div>
        <div>
          <Check size={18} />
          <span>Completed</span>
          <strong>{completedCount}</strong>
        </div>
        <div>
          <Ban size={18} />
          <span>Rejected</span>
          <strong>{rejectedCount}</strong>
        </div>
        <div>
          <LockKeyhole size={18} />
          <span>Execution</span>
          <strong>Sandbox / allowlisted</strong>
        </div>
      </div>

      {mutationError && (
        <div className="notice notice-error">
          <CircleAlert size={18} />
          <div>
            <strong>Agent action failed</strong>
            <span>{mutationError instanceof Error ? mutationError.message : "Request failed"}</span>
          </div>
        </div>
      )}

      <div className="agent-control-grid">
        <section className="panel agent-request-panel">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">New action</span>
              <h2>Request tool execution</h2>
            </div>
          </div>

          <label className="agent-field">
            <span>Action type</span>
            <select
              value={tool}
              onChange={(event) => setTool(event.target.value as AllowedTool)}
            >
              <option value="calculator">Calculator (Pure arithmetic)</option>
              <option value="calculate_procurement_comparison">Procurement comparison (Deterministic)</option>
              <option value="draft_approval_note">Draft approval note (Evidence-bound)</option>
              <option value="draft_report">Draft report (Evidence-bound)</option>
              <option value="generate_verified_code">Verified code generator (Template allowlisted)</option>
              <option value="document_report">Document report (Legacy format)</option>
              <option value="summarize_document">Summarize document (Neural - fails closed offline)</option>
            </select>
          </label>

          <label className="agent-field">
            <span>Classification</span>
            <select
              value={classification}
              onChange={(event) => setClassification(event.target.value as Classification)}
            >
              <option value="INTERNAL">Internal</option>
              <option value="CONFIDENTIAL">Confidential</option>
              <option value="RESTRICTED">Restricted</option>
            </select>
          </label>

          {tool === "calculator" && (
            <label className="agent-field">
              <span>Arithmetic expression</span>
              <input
                value={expression}
                maxLength={200}
                onChange={(event) => setExpression(event.target.value)}
                placeholder="1250 * 18 / 100"
              />
            </label>
          )}

          {tool === "generate_verified_code" && (
            <label className="agent-field">
              <span>Code template</span>
              <select
                value={codeTemplate}
                onChange={(event) => setCodeTemplate(event.target.value)}
              >
                <option value="audit_hasher">Audit Hasher (SHA-256 integrity hash)</option>
                <option value="csv_parser">CSV Parser (Sanitized CSV ingestion)</option>
                <option value="data_validator">Data Validator (Safe boundary validation)</option>
              </select>
            </label>
          )}

          {tool !== "calculator" && tool !== "generate_verified_code" && (
            <label className="agent-field">
              <span>Indexed document</span>
              <select value={documentId} onChange={(event) => setDocumentId(event.target.value)}>
                <option value="">Select an approved document</option>
                {indexedDocuments.map((document) => (
                  <option value={document.id} key={document.id}>
                    {document.filename} (v{document.version})
                  </option>
                ))}
              </select>
            </label>
          )}

          {tool === "calculate_procurement_comparison" && (
            <label className="agent-field">
              <span>Base amount (INR)</span>
              <input
                type="number"
                value={procurementBase}
                onChange={(event) => setProcurementBase(event.target.value)}
                placeholder="50000"
              />
            </label>
          )}

          {tool === "draft_approval_note" && (
            <label className="agent-field">
              <span>Note subject</span>
              <input
                value={noteSubject}
                maxLength={200}
                onChange={(event) => setNoteSubject(event.target.value)}
                placeholder="Sanction for IT Hardware Procurement"
              />
            </label>
          )}

          {tool === "draft_report" && (
            <label className="agent-field">
              <span>Report title</span>
              <input
                value={reportTitle}
                maxLength={200}
                onChange={(event) => setReportTitle(event.target.value)}
                placeholder="Audit Compliance Progress Review"
              />
            </label>
          )}

          {tool === "summarize_document" && (
            <p style={{ margin: "4px 0 0", fontSize: "10px", color: "var(--ink-500)", fontStyle: "italic" }}>
              Note: This action requires local neural model runtime. When offline, policy evaluates safely and execution truthfully returns NOT_EXECUTED.
            </p>
          )}

          <div className="agent-request-actions">
            <button
              type="button"
              className="primary-button"
              disabled={!canSubmit || proposeMutation.isPending}
              onClick={() => proposeMutation.mutate(undefined)}
            >
              {proposeMutation.isPending ? (
                <Loader2 size={15} className="spin-loader" />
              ) : (
                <LockKeyhole size={15} />
              )}
              Request approval
            </button>
            <button
              type="button"
              className="agent-blocked-button"
              disabled={proposeMutation.isPending}
              onClick={() => proposeMutation.mutate("network_request")}
            >
              <Ban size={15} /> Submit blocked network action
            </button>
          </div>
        </section>

        <section className="panel agent-policy-panel">
          <div className="panel-heading">
            <div>
              <span className="section-kicker">Enforced boundary</span>
              <h2>Execution policy</h2>
            </div>
          </div>
          <dl>
            <div><dt>Allowlist</dt><dd>7 strict typed action contracts</dd></div>
            <div><dt>Approval</dt><dd>Independent reviewer required</dd></div>
            <div><dt>Network</dt><dd>Completely disabled (Air-gapped)</dd></div>
            <div><dt>Filesystem</dt><dd>Encrypted vault only (No shell/SQL)</dd></div>
            <div><dt>Offline neural</dt><dd>Fails closed (Truthful NOT_EXECUTED)</dd></div>
            <div><dt>Deliverables</dt><dd>Masked citations + Provenance</dd></div>
          </dl>
        </section>
      </div>

      <section className="panel agent-queue-panel">
        <div className="panel-heading">
          <div>
            <span className="section-kicker">Persisted workflow</span>
            <h2>Agent run queue</h2>
          </div>
          <span className="record-count">{runs.length} runs</span>
        </div>

        <div className="agent-run-list">
          {runs.map((run) => {
            const output = resultText(run);
            const busy = approvalMutation.isPending || executeMutation.isPending;
            const hasDeliverable = run.status === "COMPLETED";
            const isDeliverableExpanded = expandedDeliverableRunId === run.id;

            return (
              <article className="agent-run-row" key={run.id}>
                <header>
                  <div className="agent-run-title">
                    {toolIcon(run.tool_name)}
                    <div>
                      <strong>{toolLabel(run.tool_name)}</strong>
                      <span>{run.argument_summary}</span>
                    </div>
                  </div>
                  <span className={`agent-run-status agent-status-${run.status.toLowerCase()}`}>
                    {formatStatusLabel(run.status)}
                  </span>
                </header>

                <div className="agent-state-track" aria-label="Agent state history">
                  {run.state_history.map((state, index) => (
                    <span key={`${state}-${index}`}>{formatStatusLabel(state)}</span>
                  ))}
                </div>

                <div className="agent-run-meta">
                  <span>Actor: {run.actor_id}</span>
                  <span>{run.classification}</span>
                  <span>{formatTime(run.created_at)}</span>
                  <span><Fingerprint size={12} /> {shortHash(run.arguments_hash)}</span>
                  {run.approval && (
                    <span>Approval: {run.approval.decision}</span>
                  )}
                </div>

                <p className="agent-policy-reason">{run.policy_reason}</p>

                {output && <pre className="agent-result-output">{output}</pre>}
                {run.failure_code && (
                  <div className="agent-failure-code">
                    <CircleAlert size={14} /> Safe failure code: {run.failure_code}
                  </div>
                )}

                {hasDeliverable && isDeliverableExpanded && (
                  <DeliverableCard runId={run.id} />
                )}

                <footer>
                  {run.status === "APPROVAL_PENDING" && (
                    <>
                      <button
                        type="button"
                        className="agent-approve-button"
                        disabled={busy}
                        onClick={() => approvalMutation.mutate({ run, decision: "APPROVED" })}
                      >
                        <Check size={14} /> Approve
                      </button>
                      <button
                        type="button"
                        className="agent-reject-button"
                        disabled={busy}
                        onClick={() => approvalMutation.mutate({ run, decision: "REJECTED" })}
                      >
                        <X size={14} /> Reject
                      </button>
                    </>
                  )}
                  {(run.status === "APPROVED" || run.status === "FAILED") && (
                    <button
                      type="button"
                      className="agent-execute-button"
                      disabled={busy}
                      onClick={() => executeMutation.mutate(run)}
                    >
                      {executeMutation.isPending ? (
                        <Loader2 size={14} className="spin-loader" />
                      ) : (
                        <Play size={14} />
                      )}
                      {run.status === "FAILED" ? "Retry execution" : "Execute restricted tool"}
                    </button>
                  )}
                  {hasDeliverable && (
                    <button
                      type="button"
                      className="agent-deliverable-toggle"
                      onClick={() =>
                        setExpandedDeliverableRunId(isDeliverableExpanded ? null : run.id)
                      }
                    >
                      <FileText size={13} />
                      {isDeliverableExpanded ? "Hide deliverable" : "View cited deliverable"}
                    </button>
                  )}
                  {run.approval?.reviewer_id && (
                    <span className="agent-reviewer">Reviewed by {run.approval.reviewer_id}</span>
                  )}
                </footer>
              </article>
            );
          })}

          {!runs.length && !runsQuery.isLoading && (
            <div className="agent-empty-state">
              <LockKeyhole size={28} />
              <strong>No agent runs</strong>
              <span>Requested actions appear here for policy review.</span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
