from koshshield.security.pii.indian_pii import IndianPiiDetector
from koshshield.security.pii.verhoeff import generate_verhoeff_check_digit, validate_verhoeff


def test_verhoeff_checksum_validation() -> None:
    # Known valid Aadhaar numbers (Verhoeff checksums verified)
    # Using synthetic valid 12-digit numbers
    base_11 = "28374619283"
    check_digit = generate_verhoeff_check_digit(base_11)
    valid_aadhaar = f"{base_11}{check_digit}"

    assert validate_verhoeff(valid_aadhaar) is True

    # Single digit alteration must fail
    wrong_digit = (check_digit + 1) % 10
    invalid_single = f"{base_11}{wrong_digit}"
    assert validate_verhoeff(invalid_single) is False

    # Adjacent digit transposition must fail
    # Swap last two digits
    transposed = f"{base_11[:-1]}{check_digit}{base_11[-1]}"
    assert validate_verhoeff(transposed) is False


def test_pan_detection_and_validation() -> None:
    detector = IndianPiiDetector(salt="test-salt")

    # Valid PANs for various taxpayer categories
    text = """
    Individual PAN: ABCPE1234F
    Company PAN: AABCC5678G
    Trust PAN: AAATT9876H
    Invalid format 1: 12345ABCDE
    Invalid format 2: ABCDE12345
    Invalid 4th char: ABCXE1234F
    """
    findings = detector.detect(text)
    pan_findings = [f for f in findings if f.finding_type == "PAN"]

    assert len(pan_findings) == 3
    pan_values = {f.value for f in pan_findings}
    assert "ABCPE1234F" in pan_values
    assert "AABCC5678G" in pan_values
    assert "AAATT9876H" in pan_values
    assert "ABCXE1234F" not in pan_values


def test_aadhaar_detection_with_verhoeff() -> None:
    detector = IndianPiiDetector(salt="test-salt")

    # Generate two valid synthetic Aadhaar numbers
    base1 = "36759834123"
    aadhaar1 = f"{base1}{generate_verhoeff_check_digit(base1)}"

    base2 = "98765432109"
    aadhaar2 = f"{base2}{generate_verhoeff_check_digit(base2)}"

    # Format with spaces and dashes
    text = f"""
    Applicant Aadhaar: {aadhaar1[:4]} {aadhaar1[4:8]} {aadhaar1[8:]}
    Spouse Aadhaar: {aadhaar2[:4]}-{aadhaar2[4:8]}-{aadhaar2[8:]}
    Invalid Aadhaar (bad check digit): 3675 9834 1230
    Starting with 0: 0123 4567 8901
    Starting with 1: 1234 5678 9012
    """
    findings = detector.detect(text)
    aadhaar_findings = [f for f in findings if f.finding_type == "AADHAAR"]

    assert len(aadhaar_findings) == 2
    assert any(aadhaar1[:4] in f.value for f in aadhaar_findings)
    assert any(aadhaar2[:4] in f.value for f in aadhaar_findings)
    assert not any("3675 9834 1230" in f.value for f in aadhaar_findings)


def test_phone_normalization_and_detection() -> None:
    detector = IndianPiiDetector(salt="test-salt")

    text = """
    Contact numbers:
    Direct: 9876543210
    With +91: +91 9876543210
    With dash: +91-98765-43210
    With zero: 09876543210
    Invalid (starts with 5): 5876543210
    Invalid (too short): 987654321
    """
    findings = detector.detect(text)
    phone_findings = [f for f in findings if f.finding_type == "PHONE"]

    # 4 valid phone numbers
    assert len(phone_findings) == 4
    for p in phone_findings:
        assert p.confidence == 0.90
        assert p.detection_source == "indian_phone_normalizer"


def test_email_ifsc_passport_and_bank_account_detection() -> None:
    detector = IndianPiiDetector(salt="test-salt")

    text = """
    Employee Email: officer.dept@nic.in
    Bank Details:
    State Bank of India IFSC: SBIN0001234
    Account Number: 123456789012
    Passport Number: Z1234567
    Government ID: GOV-849204
    """
    findings = detector.detect(text)
    types = {f.finding_type: f for f in findings}

    assert "EMAIL" in types
    assert types["EMAIL"].value == "officer.dept@nic.in"

    assert "IFSC" in types
    assert types["IFSC"].value == "SBIN0001234"

    assert "BANK_ACCOUNT" in types
    assert types["BANK_ACCOUNT"].value == "123456789012"

    assert "PASSPORT" in types
    assert types["PASSPORT"].value == "Z1234567"

    assert "GOV_ID" in types
    assert types["GOV_ID"].value == "GOV-849204"


def test_overlapping_finding_deduplication() -> None:
    detector = IndianPiiDetector(salt="test-salt")

    # In a case where a sequence could match both account number and phone
    text = "Account: 98765432101234"
    findings = detector.detect(text)

    # Overlapping spans must be deduplicated
    spans = [(f.start, f.end) for f in findings]
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            assert max(spans[i][0], spans[j][0]) >= min(spans[i][1], spans[j][1]), (
                f"Overlap detected between {findings[i]} and {findings[j]}"
            )
