import { describe, expect, it } from "vitest";

import {
  formatBytes,
  formatConfidence,
  formatFindingLabel,
  formatStatusLabel,
  shortHash,
} from "./format";

describe("format helpers", () => {
  it("formats document sizes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
  });

  it("shortens evidence hashes", () => {
    expect(shortHash("1234567890abcdefghijklmnop")).toBe("12345678…klmnop");
  });

  it("formats confidence percentages", () => {
    expect(formatConfidence(0.98)).toBe("98%");
    expect(formatConfidence(0.852)).toBe("85%");
  });

  it("formats finding labels", () => {
    expect(formatFindingLabel("AADHAAR")).toBe("Aadhaar number");
    expect(formatFindingLabel("PAN")).toBe("PAN card");
    expect(formatFindingLabel("PHONE")).toBe("Mobile number");
    expect(formatFindingLabel("BANK_ACCOUNT")).toBe("Bank account");
  });

  it("formats status labels", () => {
    expect(formatStatusLabel("REVIEW_REQUIRED")).toBe("Review required");
    expect(formatStatusLabel("INDEX_READY")).toBe("Index ready");
    expect(formatStatusLabel("INDEXING")).toBe("Indexing…");
    expect(formatStatusLabel("INDEXED")).toBe("Indexed");
    expect(formatStatusLabel("INDEX_FAILED")).toBe("Index failed");
    expect(formatStatusLabel("ENCRYPTED")).toBe("Encrypted");
  });
});
