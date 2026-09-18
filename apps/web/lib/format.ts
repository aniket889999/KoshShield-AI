export function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

export function shortHash(value: string): string {
  return `${value.slice(0, 8)}…${value.slice(-6)}`;
}

export function formatTime(value: string): string {
  return new Intl.DateTimeFormat("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function formatConfidence(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function formatFindingLabel(type: string): string {
  switch (type.toUpperCase()) {
    case "AADHAAR":
      return "Aadhaar number";
    case "PAN":
      return "PAN card";
    case "PHONE":
      return "Mobile number";
    case "EMAIL":
      return "Email address";
    case "BANK_ACCOUNT":
      return "Bank account";
    case "IFSC":
      return "IFSC code";
    case "PASSPORT":
      return "Passport number";
    case "GOV_ID":
      return "Government / Employee ID";
    default:
      return type.replace(/_/g, " ").toLowerCase();
  }
}

export function formatStatusLabel(status: string): string {
  switch (status.toUpperCase()) {
    case "ENCRYPTED":
      return "Encrypted";
    case "EXTRACTION_QUEUED":
      return "Extraction queued";
    case "EXTRACTING":
      return "Extracting…";
    case "REVIEW_REQUIRED":
      return "Review required";
    case "REDACTION_APPROVED":
      return "Redaction approved";
    case "INDEX_READY":
      return "Index ready";
    case "INDEXING":
      return "Indexing…";
    case "INDEXED":
      return "Indexed";
    case "INDEX_FAILED":
      return "Index failed";
    case "EXTRACTION_FAILED":
      return "Extraction failed";
    default:
      return status.replace(/_/g, " ");
  }
}

export function formatStageStatus(status: string): string {
  switch (status.toUpperCase()) {
    case "PASSED":
      return "Passed";
    case "FAILED":
      return "Failed";
    case "IN_PROGRESS":
      return "In progress";
    case "PENDING":
      return "Pending";
    case "SKIPPED":
      return "Skipped";
    case "NOT_EXECUTED":
      return "Not executed";
    default:
      return status.replace(/_/g, " ");
  }
}

export function formatReadinessStatus(status: string): string {
  switch (status.toUpperCase()) {
    case "READY":
      return "Ready";
    case "MISSING_ARTIFACT":
      return "Missing artifact";
    case "SERVICE_UNAVAILABLE":
      return "Service offline";
    case "CONTRACT_MISMATCH":
      return "Contract mismatch";
    case "NOT_CONFIGURED":
      return "Not configured";
    case "NOT_EXECUTED":
      return "Not executed";
    case "ERROR":
      return "Error";
    default:
      return status.replace(/_/g, " ");
  }
}
