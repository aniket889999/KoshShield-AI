import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass
class MaskedChunk:
    chunk_id: str
    tenant_id: str
    document_id: str
    page_number: int
    redaction_version: int
    chunk_sequence: int
    masked_text: str
    char_start: int
    char_end: int
    masked_content_hash: str
    document_evidence_hash: str
    classification: str
    document_filename: str
    indexed_at: str


class DeterministicMaskedChunker:
    """Deterministic, token-aware chunker for approved masked text.

    Guarantees:
    - Strictly preserves page boundaries for accurate page citations.
    - Never splits redaction placeholders ([REDACTED_*]).
    - Prefers paragraph or sentence boundaries.
    - Produces deterministic, content-hashed chunk IDs.
    """

    def __init__(
        self,
        target_tokens: int = 450,
        overlap_tokens: int = 70,
        chars_per_token: int = 4,
    ) -> None:
        self.target_chars = target_tokens * chars_per_token
        self.overlap_chars = overlap_tokens * chars_per_token

    @staticmethod
    def _find_protected_spans(text: str) -> list[tuple[int, int]]:
        """Finds spans of redaction placeholders that must never be severed."""
        spans: list[tuple[int, int]] = []
        for match in re.finditer(r"\[REDACTED_[A-Z0-9_]+\]", text):
            spans.append((match.start(), match.end()))
        return spans

    def _adjust_split_point(
        self, text: str, ideal_end: int, protected_spans: list[tuple[int, int]]
    ) -> int:
        """Adjusts the split point to respect paragraph/sentence boundaries
        and avoid splitting redaction tokens."""

        # 1. First, check if ideal_end falls inside a protected redaction span
        for start, end in protected_spans:
            if start < ideal_end < end:
                # Adjust to before the placeholder starts
                ideal_end = start
                break

        if ideal_end <= 0 or ideal_end >= len(text):
            return len(text)

        # 2. Look backwards up to 35% of target_chars for clean boundary
        search_window_start = max(0, ideal_end - int(self.target_chars * 0.35))
        candidate_text = text[search_window_start:ideal_end]

        # Check for paragraph break
        para_idx = candidate_text.rfind("\n\n")
        if para_idx != -1:
            candidate_split = search_window_start + para_idx + 2
            if not any(s < candidate_split < e for s, e in protected_spans):
                return candidate_split

        # Check for newline
        line_idx = candidate_text.rfind("\n")
        if line_idx != -1:
            candidate_split = search_window_start + line_idx + 1
            if not any(s < candidate_split < e for s, e in protected_spans):
                return candidate_split

        # Check for sentence end
        for punct in [". ", "? ", "! "]:
            p_idx = candidate_text.rfind(punct)
            if p_idx != -1:
                candidate_split = search_window_start + p_idx + 2
                if not any(s < candidate_split < e for s, e in protected_spans):
                    return candidate_split

        # Check for whitespace
        space_idx = candidate_text.rfind(" ")
        if space_idx != -1:
            candidate_split = search_window_start + space_idx + 1
            if not any(s < candidate_split < e for s, e in protected_spans):
                return candidate_split

        return ideal_end

    def chunk_page(
        self,
        page_text: str,
        document_id: str,
        page_number: int,
        document_filename: str,
        document_evidence_hash: str,
        redaction_version: int,
        tenant_id: str = "default",
        classification: str = "CONFIDENTIAL",
    ) -> list[MaskedChunk]:
        clean_text = page_text.strip()
        if not clean_text:
            return []

        protected_spans = self._find_protected_spans(clean_text)
        chunks: list[MaskedChunk] = []
        text_len = len(clean_text)
        start = 0
        seq = 0
        now_iso = datetime.now(UTC).isoformat()

        while start < text_len:
            if text_len - start <= self.target_chars:
                end = text_len
            else:
                ideal_end = start + self.target_chars
                end = self._adjust_split_point(clean_text, ideal_end, protected_spans)
                if end <= start:
                    # Guard against zero-step loop
                    end = min(start + self.target_chars, text_len)

            chunk_content = clean_text[start:end].strip()
            if chunk_content:
                content_hash = hashlib.sha256(chunk_content.encode("utf-8")).hexdigest()
                # Deterministic stable chunk ID
                chunk_id_raw = f"{document_id}:p{page_number}:seq{seq}:{content_hash}"
                chunk_id = hashlib.sha256(chunk_id_raw.encode("utf-8")).hexdigest()[:32]

                chunks.append(
                    MaskedChunk(
                        chunk_id=chunk_id,
                        tenant_id=tenant_id,
                        document_id=document_id,
                        page_number=page_number,
                        redaction_version=redaction_version,
                        chunk_sequence=seq,
                        masked_text=chunk_content,
                        char_start=start,
                        char_end=end,
                        masked_content_hash=content_hash,
                        document_evidence_hash=document_evidence_hash,
                        classification=classification,
                        document_filename=document_filename,
                        indexed_at=now_iso,
                    )
                )
                seq += 1

            if end >= text_len:
                break

            # Calculate next start with overlap
            next_start = max(start + 1, end - self.overlap_chars)
            # Ensure next_start does not start in the middle of a protected placeholder
            for p_start, p_end in protected_spans:
                if p_start < next_start < p_end:
                    next_start = p_end
                    break
            start = next_start

        return chunks
