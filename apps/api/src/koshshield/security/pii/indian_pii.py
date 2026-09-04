import hashlib
import re

from koshshield.security.pii.interfaces import DetectedPii, PiiDetector
from koshshield.security.pii.verhoeff import validate_verhoeff

REDACTION_PLACEHOLDERS: dict[str, str] = {
    "AADHAAR": "[AADHAAR_REDACTED]",
    "PAN": "[PAN_REDACTED]",
    "PHONE": "[PHONE_REDACTED]",
    "EMAIL": "[EMAIL_REDACTED]",
    "BANK_ACCOUNT": "[BANK_ACCOUNT_REDACTED]",
    "IFSC": "[IFSC_REDACTED]",
    "PASSPORT": "[PASSPORT_REDACTED]",
    "GOV_ID": "[GOV_ID_REDACTED]",
}


def hash_pii_value(value: str, salt: str) -> str:
    """Computes a secure salted SHA-256 hash of a sensitive value."""
    normalized = "".join(value.split()).strip()
    return hashlib.sha256(f"{salt}:{normalized}".encode()).hexdigest()


def generate_masked_context(
    text: str, start: int, end: int, placeholder: str, window: int = 40
) -> str:
    """Creates a privacy-safe context snippet with the sensitive span replaced."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)

    prefix = text[ctx_start:start]
    suffix = text[end:ctx_end]

    prefix_ellipsis = "…" if ctx_start > 0 else ""
    suffix_ellipsis = "…" if ctx_end < len(text) else ""

    return f"{prefix_ellipsis}{prefix}{placeholder}{suffix}{suffix_ellipsis}"


def generate_safe_contexts_for_page(
    page_text: str,
    findings: list[DetectedPii],
    window: int = 40,
) -> list[str]:
    """Generates context snippets where ALL detected PII findings on the page are masked,
    preventing any sensitive identifier from leaking into neighboring context snippets.
    """
    if not findings:
        return []

    sorted_findings = sorted(findings, key=lambda f: f.start)

    new_positions = []
    shift = 0
    for f in sorted_findings:
        new_start = f.start + shift
        placeholder = REDACTION_PLACEHOLDERS.get(f.finding_type, "[REDACTED]")
        new_end = new_start + len(placeholder)
        new_positions.append((f, new_start, new_end, placeholder))
        shift += len(placeholder) - (f.end - f.start)

    masked_page_text = page_text
    for f in sorted(sorted_findings, key=lambda f: f.start, reverse=True):
        ph = REDACTION_PLACEHOLDERS.get(f.finding_type, "[REDACTED]")
        masked_page_text = masked_page_text[: f.start] + ph + masked_page_text[f.end :]

    contexts_by_id: dict[int, str] = {}
    for f, new_start, new_end, _ in new_positions:
        ctx_start = max(0, new_start - window)
        ctx_end = min(len(masked_page_text), new_end + window)
        prefix_ellipsis = "…" if ctx_start > 0 else ""
        suffix_ellipsis = "…" if ctx_end < len(masked_page_text) else ""
        ctx = f"{prefix_ellipsis}{masked_page_text[ctx_start:ctx_end]}{suffix_ellipsis}"
        contexts_by_id[id(f)] = ctx

    return [contexts_by_id[id(f)] for f in findings]


class IndianPiiDetector(PiiDetector):
    """Deterministic local recognizers for Indian PII identifiers."""

    # Aadhaar: 12 digits, first digit 2-9, optional 4-4-4 spacing
    AADHAAR_PATTERN = re.compile(r"\b([2-9]\d{3}[ -]?\d{4}[ -]?\d{4})\b")

    # PAN: 5 uppercase letters (4th is entity category), 4 digits, 1 letter
    PAN_PATTERN = re.compile(r"\b([A-Z]{3}[CPHFATBLJG][A-Z]\d{4}[A-Z])\b")

    # Indian Mobile: +91 or 0 prefix optional, 10 digits starting with 6-9
    PHONE_PATTERN = re.compile(r"(?:(?:\+91[\s\-]?)|\b0)?([6-9]\d{4}[\s\-]?\d{5}\b|[6-9]\d{9}\b)")

    # Email: RFC-compliant structure
    EMAIL_PATTERN = re.compile(r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

    # Bank Account: preceded by account context keyword, 9 to 18 digits
    BANK_ACCOUNT_PATTERN = re.compile(
        r"(?i)(?:a/c|acct|account(?:\s*no\.?|\s*number)?|sb\s*a/c|current\s*a/c)[\s:#-]*([0-9]{9,18})\b"
    )

    # IFSC Code: 4 letters, 0, 6 alphanumeric
    IFSC_PATTERN = re.compile(r"\b([A-Z]{4}0[A-Z0-9]{6})\b")

    # Passport: 1 letter + 7 digits (with or without passport keyword)
    PASSPORT_WITH_KW_PATTERN = re.compile(
        r"(?i)(?:passport(?:\s*no\.?|\s*number)?[\s:#-]*)([A-Za-z][0-9]{7})\b"
    )
    PASSPORT_STANDALONE_PATTERN = re.compile(r"\b([A-Za-z][0-9]{7})\b")

    # Government / Employee ID / Voter EPIC
    GOV_ID_PATTERN = re.compile(
        r"\b(?:(?:GOV|EMP|UID|NIC|OFFICER|STAFF|EMPLOYEE|VOTER|EPIC)[-_/][A-Z0-9]{4,12}|[A-Z]{3}[0-9]{7})\b"
    )

    def __init__(self, salt: str = "koshshield-default-dev-salt") -> None:
        self.salt = salt

    def detect(
        self,
        text: str,
        page_number: int = 1,
        blocks: list[tuple[float, float, float, float, str]] | None = None,
    ) -> list[DetectedPii]:
        findings: list[DetectedPii] = []

        # 1. Aadhaar detection with Verhoeff verification
        for match in self.AADHAAR_PATTERN.finditer(text):
            raw_val = match.group(1)
            digits_only = re.sub(r"\D", "", raw_val)
            if len(digits_only) == 12 and validate_verhoeff(digits_only):
                findings.append(
                    DetectedPii(
                        finding_type="AADHAAR",
                        value=raw_val,
                        start=match.start(1),
                        end=match.end(1),
                        confidence=0.98,
                        detection_source="verhoeff_checksum_validator",
                        page_number=page_number,
                    )
                )

        # 2. PAN detection
        for match in self.PAN_PATTERN.finditer(text):
            raw_val = match.group(1)
            findings.append(
                DetectedPii(
                    finding_type="PAN",
                    value=raw_val,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.95,
                    detection_source="pan_structural_pattern",
                    page_number=page_number,
                )
            )

        # 3. Email detection
        for match in self.EMAIL_PATTERN.finditer(text):
            raw_val = match.group(1)
            findings.append(
                DetectedPii(
                    finding_type="EMAIL",
                    value=raw_val,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.95,
                    detection_source="email_rfc_pattern",
                    page_number=page_number,
                )
            )

        # 4. Bank account detection (context-sensitive)
        for match in self.BANK_ACCOUNT_PATTERN.finditer(text):
            account_digits = match.group(1)
            findings.append(
                DetectedPii(
                    finding_type="BANK_ACCOUNT",
                    value=account_digits,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.90,
                    detection_source="bank_account_context_pattern",
                    page_number=page_number,
                )
            )

        # 5. IFSC detection
        for match in self.IFSC_PATTERN.finditer(text):
            raw_val = match.group(1)
            findings.append(
                DetectedPii(
                    finding_type="IFSC",
                    value=raw_val,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.95,
                    detection_source="ifsc_structural_pattern",
                    page_number=page_number,
                )
            )

        # 6. Passport detection
        found_passports: set[int] = set()
        for match in self.PASSPORT_WITH_KW_PATTERN.finditer(text):
            raw_val = match.group(1)
            found_passports.add(match.start(1))
            findings.append(
                DetectedPii(
                    finding_type="PASSPORT",
                    value=raw_val,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.95,
                    detection_source="passport_context_pattern",
                    page_number=page_number,
                )
            )
        for match in self.PASSPORT_STANDALONE_PATTERN.finditer(text):
            if match.start(1) not in found_passports:
                findings.append(
                    DetectedPii(
                        finding_type="PASSPORT",
                        value=match.group(1),
                        start=match.start(1),
                        end=match.end(1),
                        confidence=0.85,
                        detection_source="passport_pattern",
                        page_number=page_number,
                    )
                )

        # 7. Government / Employee ID
        for match in self.GOV_ID_PATTERN.finditer(text):
            raw_val = match.group(0)
            findings.append(
                DetectedPii(
                    finding_type="GOV_ID",
                    value=raw_val,
                    start=match.start(0),
                    end=match.end(0),
                    confidence=0.85,
                    detection_source="gov_employee_id_pattern",
                    page_number=page_number,
                )
            )

        # 8. Indian Mobile Phone detection
        for match in self.PHONE_PATTERN.finditer(text):
            # group 1 contains the 10-digit number
            digits = re.sub(r"\D", "", match.group(1))
            if len(digits) == 10 and digits[0] in "6789":
                findings.append(
                    DetectedPii(
                        finding_type="PHONE",
                        value=match.group(0).strip(),
                        start=match.start(0),
                        end=match.end(0),
                        confidence=0.90,
                        detection_source="indian_phone_normalizer",
                        page_number=page_number,
                    )
                )

        # Deduplicate overlapping findings (higher confidence and longer spans win)
        deduped = self._deduplicate_findings(findings)

        # Attach bounding boxes if block coordinate data is available
        if blocks:
            self._attach_bounding_boxes(deduped, blocks)

        return deduped

    def _deduplicate_findings(self, findings: list[DetectedPii]) -> list[DetectedPii]:
        if not findings:
            return []

        # Sort: higher confidence first, longer span second, earlier start third
        sorted_findings = sorted(
            findings,
            key=lambda f: (f.confidence, f.end - f.start, -f.start),
            reverse=True,
        )

        selected: list[DetectedPii] = []
        for candidate in sorted_findings:
            overlaps = False
            for existing in selected:
                # Check for overlap: max(start) < min(end)
                if max(candidate.start, existing.start) < min(candidate.end, existing.end):
                    overlaps = True
                    break
            if not overlaps:
                selected.append(candidate)

        # Return sorted by appearance order (start offset)
        return sorted(selected, key=lambda f: f.start)

    def _attach_bounding_boxes(
        self,
        findings: list[DetectedPii],
        blocks: list[tuple[float, float, float, float, str]],
    ) -> None:
        for finding in findings:
            val = finding.value.strip()
            for bbox_item in blocks:
                # bbox_item: (x0, y0, x1, y1, text)
                block_text = bbox_item[4]
                if val in block_text:
                    finding.bbox = (bbox_item[0], bbox_item[1], bbox_item[2], bbox_item[3])
                    break
