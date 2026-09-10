"""Build a deterministic synthetic case and independently check its answer key."""

import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = A4
NAVY = colors.HexColor("#15344B")
TEAL = colors.HexColor("#087F8C")
INK = colors.HexColor("#1E2933")
MUTED = colors.HexColor("#52616B")
PALE = colors.HexColor("#EDF5F7")


def read_json(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_json(relative, payload):
    (ROOT / relative).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def base_page(pdf, number, subtitle):
    pdf.setFillColor(NAVY)
    pdf.rect(0, HEIGHT - 105, WIDTH, 105, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(42, HEIGHT - 34, "KOSHSHIELD / SYNTHETIC EVALUATION CASE")
    pdf.setFont("Helvetica-Bold", 22)
    pdf.drawString(42, HEIGHT - 67, "PROC-001 | Pump supply")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(42, HEIGHT - 88, subtitle)
    pdf.setStrokeColor(colors.HexColor("#CAD6DC"))
    pdf.line(42, 52, WIDTH - 42, 52)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(42, 37, "FICTIONAL TEST DATA | No purchasing or engineering authority")
    pdf.drawRightString(WIDTH - 42, 37, f"Page {number} of 2")


def section(pdf, text, y):
    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(42, y, text)


def line(pdf, text, y, bold=False):
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold" if bold else "Helvetica", 11)
    if pdf.stringWidth(text, "Helvetica-Bold" if bold else "Helvetica", 11) > WIDTH - 84:
        raise ValueError("Fixture line exceeds printable width")
    pdf.drawString(42, y, text)


def table(pdf, headers, rows, widths, y):
    height = 37
    for row_number, row in enumerate([headers, *rows]):
        top = y - row_number * height
        pdf.setFillColor(TEAL if row_number == 0 else PALE if row_number % 2 else colors.white)
        pdf.rect(42, top - height, sum(widths), height, stroke=0, fill=1)
        x = 42
        for text, width in zip(row, widths, strict=True):
            font = "Helvetica-Bold" if row_number == 0 else "Helvetica"
            if pdf.stringWidth(str(text), font, 10) > width - 14:
                raise ValueError("Fixture table cell exceeds printable width")
            pdf.setFillColor(colors.white if row_number == 0 else INK)
            pdf.setFont(font, 10)
            pdf.drawString(x + 7, top - 23, str(text))
            x += width
    return y - (len(rows) + 1) * height


def make_native(case, output):
    pdf = canvas.Canvas(str(output), pagesize=A4, invariant=1, pageCompression=1)
    pdf.setTitle("PROC-001 synthetic pump supply evaluation")
    pdf.setAuthor("KoshShield synthetic test fixture")
    base_page(pdf, 1, "Tender requirements and privacy test fields")
    section(pdf, "Purpose", 704)
    line(
        pdf,
        "Compare three fictional supplier proposals against the stated requirements.",
        680,
    )
    line(
        pdf,
        "Use only the values supplied. Missing information must remain unknown.",
        661,
    )
    section(pdf, "Required technical and commercial terms", 620)
    rows = [
        [r["id"], r["field"].capitalize(), r["operator"], r["value"], r["unit"]]
        for r in case["requirements"]
    ]
    table(
        pdf,
        ["ID", "Requirement", "Rule", "Value", "Unit"],
        rows,
        [45, 166, 70, 85, 145],
        601,
    )
    section(pdf, "Fictional contact: mandatory privacy fixture", 367)
    line(pdf, "These strings are invented solely to exercise identifier detection.", 343)
    line(pdf, f"PAN: {case['contact']['pan']}", 316)
    line(pdf, f"Phone: {case['contact']['phone']}", 292)
    line(pdf, f"Email: {case['contact']['email']}", 268)
    section(pdf, "Review conditions", 219)
    line(
        pdf,
        "Assess the supplied evidence; do not infer missing vendor commitments.",
        195,
    )
    line(
        pdf,
        "The second page contains the proposals. No purchase prices are supplied.",
        175,
    )
    line(
        pdf,
        "All values and identities are synthetic and must not be used operationally.",
        155,
    )
    pdf.showPage()
    base_page(pdf, 2, "Supplier proposal table and measurement context")
    section(pdf, "Supplier proposal summary", 704)
    rows = [
        [
            p["supplier"],
            p["flow"],
            p["pressure"],
            "Not provided" if p["delivery"] is None else p["delivery"],
            p["warranty"],
        ]
        for p in case["proposals"]
    ]
    table(
        pdf,
        ["Supplier", "Flow m3/h", "Pressure bar", "Delivery days", "Warranty months"],
        rows,
        [90, 85, 92, 115, 129],
        679,
    )
    section(pdf, "Evidence limitations", 488)
    line(pdf, "Supplier C did not provide a delivery period.", 463)
    line(pdf, "Purchase prices are not present anywhere in this case.", 441)
    line(
        pdf,
        "A listed number is a supplier claim, not an independent test certificate.",
        419,
    )
    section(pdf, "Synthetic measurement extract", 374)
    with (ROOT / "inputs/proc001-measurements.csv").open(newline="", encoding="utf-8") as stream:
        measurements = list(csv.DictReader(stream))
    table(
        pdf,
        ["Equipment", "Measured flow m3/h", "Measured pressure bar"],
        [[r["equipment_id"], r["flow_m3_h"], r["pressure_bar"]] for r in measurements],
        [115, 186, 210],
        352,
    )
    line(pdf, "Measurement rows are separate from the supplier proposal claims.", 171)
    line(pdf, "The CSV repeats these measurements for a later generated-code task.", 151)
    pdf.save()


def make_scan(native, scanned, preview_dir):
    executable = shutil.which("pdftoppm")
    if not executable:
        raise RuntimeError("pdftoppm is required; no automatic installation is performed")
    subprocess.run(
        [executable, "-r", "150", "-png", str(native), str(preview_dir / "page")],
        check=True,
        capture_output=True,
    )
    images = sorted(preview_dir.glob("page-*.png"))
    if len(images) != 2:
        raise ValueError("Native PDF must render as exactly two pages")
    pdf = canvas.Canvas(str(scanned), pagesize=A4, invariant=1, pageCompression=1)
    pdf.setTitle("PROC-001 synthetic raster scan; no embedded text")
    pdf.setAuthor("KoshShield synthetic test fixture")
    for path in images:
        pdf.drawImage(str(path), 0, 0, width=WIDTH, height=HEIGHT)
        pdf.showPage()
    pdf.save()


def validate(case, native, scanned):
    truth = read_json("expected/proc001-ground-truth.json")
    native_doc, scan_doc = PdfReader(native), PdfReader(scanned)
    assert len(native_doc.pages) == len(scan_doc.pages) == truth["expected_pages"] == 2
    texts = [page.extract_text() or "" for page in native_doc.pages]
    assert all(len(text) > 200 for text in texts)
    assert all(not (page.extract_text() or "").strip() for page in scan_doc.pages)
    assert all(len(page.images) == 1 for page in scan_doc.pages)
    for pii in truth["pii"]:
        assert pii["value"] in texts[pii["page"] - 1]
    assert "Not provided" in texts[1]
    assert "Purchase prices are not present" in texts[1]
    assert "80" in texts[0] and "m3/h" in texts[0]
    comparisons = {}
    for proposal in case["proposals"]:
        results = {}
        for requirement in case["requirements"]:
            value = proposal[requirement["field"]]
            if value is None:
                verdict = "MISSING"
            else:
                meets = (
                    value >= requirement["value"]
                    if requirement["operator"] == ">="
                    else value <= requirement["value"]
                )
                verdict = "SUPPORTED" if meets else "CONTRADICTED"
            results[requirement["id"]] = verdict
        comparisons[proposal["supplier"]] = results
    assert comparisons == truth["comparison_expected"]
    measurements = []
    with (ROOT / "inputs/proc001-measurements.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            failures = []
            if Decimal(row["flow_m3_h"]) < Decimal(80):
                failures.append("R1")
            if Decimal(row["pressure_bar"]) < Decimal(12):
                failures.append("R2")
            measurements.append(
                {"equipment_id": row["equipment_id"], "failed_requirements": failures}
            )
    assert measurements == truth["measurement_expected"]
    return {
        "case_id": case["case_id"],
        "synthetic": True,
        "fixture_checks": "PASSED",
        "native_pages": len(native_doc.pages),
        "scanned_pages": len(scan_doc.pages),
        "scan_has_embedded_text": False,
        "ground_truth_comparisons_checked": len(comparisons) * len(case["requirements"]),
        "measurement_rows_checked": len(measurements),
        "live_ocr": "NOT_EXECUTED",
        "live_embeddings": "NOT_EXECUTED",
        "live_qdrant": "NOT_EXECUTED",
        "live_qwen": "NOT_EXECUTED",
    }


def main():
    case = read_json("source_case.json")
    native = ROOT / "inputs/proc001-native.pdf"
    scanned = ROOT / "inputs/proc001-scanned.pdf"
    make_native(case, native)
    with tempfile.TemporaryDirectory(prefix="koshshield-fixture-") as directory:
        make_scan(native, scanned, Path(directory))
    result = validate(case, native, scanned)
    write_json("fixture-validation.json", result)
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        entries.append(
            {
                "path": relative,
                "role": "input" if relative.startswith("inputs/") else "evaluation_only",
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    write_json("manifest.json", {"case_id": "PROC-001", "synthetic": True, "files": entries})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
