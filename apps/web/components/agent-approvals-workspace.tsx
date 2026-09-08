"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Ban,
  Calculator,
  Check,
  CircleAlert,
  Clock3,
  FileOutput,
  Fingerprint,
  Loader2,
  LockKeyhole,
  Play,
  ShieldCheck,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";

import {
  decideAgentApproval,
  executeAgentAction,
  listAgentRuns,
  listDocuments,
  proposeAgentAction,
  type AgentRun,
} from "@/lib/api";
import { formatStatusLabel, formatTime, shortHash } from "@/lib/format";

type AllowedTool = "calculator" | "document_report";
type Classification = "INTERNAL" | "CONFIDENTIAL" | "RESTRICTED";

function toolLabel(toolName: string) {
  if (toolName === "calculator") return "Calculator";
  if (toolName === "document_report") return "Document report";
  return formatStatusLabel(toolName);
}

function resultText(run: AgentRun) {
  if (run.result?.output.value) return `Result: ${run.result.output.value}`;
  if (run.result?.output.content) return run.result.output.content;
  return null;
}

export function AgentApprovalsWorkspace() {
  const queryClient = useQueryClient();
  const [tool, setTool] = useState<AllowedTool>("calculator");
  const [classification, setClassification] = useState<Classification>("CONFIDENTIAL");
  const [expression, setExpression] = useState("1250 * 18 / 100");
  const [documentId, setDocumentId] = useState("");

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
              arguments:
                tool === "calculator" ? { expression } : { document_id: documentId },
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
    tool === "calculator" ? Boolean(expression.trim()) : Boolean(documentId);

  return (
    <div className="agent-workspace">
      <section className="page-heading">
        <div>
          <span className="eyebrow">Policy-gated agent</span>
          <h1>Approvals</h1>
          <p>Review and execute restricted local tools through the human authorization queue.</p>
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
          <strong>Docker only</strong>
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

          <div className="agent-tool-tabs" role="tablist" aria-label="Allowed agent tools">
            <button
              type="button"
              className={tool === "calculator" ? "agent-tool-active" : ""}
              onClick={() => setTool("calculator")}
            >
              <Calculator size={16} /> Calculator
            </button>
            <button
              type="button"
              className={tool === "document_report" ? "agent-tool-active" : ""}
              onClick={() => setTool("document_report")}
            >
              <FileOutput size={16} /> Document report
            </button>
          </div>

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

          {tool === "calculator" ? (
            <label className="agent-field">
              <span>Arithmetic expression</span>
              <input
                value={expression}
                maxLength={200}
                onChange={(event) => setExpression(event.target.value)}
                placeholder="1250 * 18 / 100"
              />
            </label>
          ) : (
            <label className="agent-field">
              <span>Indexed document</span>
              <select value={documentId} onChange={(event) => setDocumentId(event.target.value)}>
                <option value="">Select a document</option>
                {indexedDocuments.map((document) => (
                  <option value={document.id} key={document.id}>
                    {document.filename}
                  </option>
                ))}
              </select>
            </label>
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
            <div><dt>Allowlist</dt><dd>Calculator, document report</dd></div>
            <div><dt>Approval</dt><dd>Independent reviewer</dd></div>
            <div><dt>Network</dt><dd>Disabled</dd></div>
            <div><dt>Root filesystem</dt><dd>Read-only</dd></div>
            <div><dt>Linux capabilities</dt><dd>Dropped</dd></div>
            <div><dt>Output</dt><dd>Schema and hash verified</dd></div>
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
            return (
              <article className="agent-run-row" key={run.id}>
                <header>
                  <div className="agent-run-title">
                    {run.tool_name === "calculator" ? (
                      <Calculator size={17} />
                    ) : run.tool_name === "document_report" ? (
                      <FileOutput size={17} />
                    ) : (
                      <Ban size={17} />
                    )}
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
                  <span>{run.actor_id}</span>
                  <span>{run.classification}</span>
                  <span>{formatTime(run.created_at)}</span>
                  <span><Fingerprint size={12} /> {shortHash(run.arguments_hash)}</span>
                </div>

                <p className="agent-policy-reason">{run.policy_reason}</p>

                {output && <pre className="agent-result-output">{output}</pre>}
                {run.failure_code && (
                  <div className="agent-failure-code">
                    <CircleAlert size={14} /> {run.failure_code}
                  </div>
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
                      {run.status === "FAILED" ? "Retry in sandbox" : "Execute in sandbox"}
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
