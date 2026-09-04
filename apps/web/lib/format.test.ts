import { describe, expect, it } from "vitest";

import { formatBytes, shortHash } from "./format";

describe("format helpers", () => {
  it("formats document sizes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
  });

  it("shortens evidence hashes", () => {
    expect(shortHash("1234567890abcdefghijklmnop")).toBe("12345678…klmnop");
  });
});
