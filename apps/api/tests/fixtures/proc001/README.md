# PROC-001 synthetic smoke fixture

Every supplier, identifier, document and measurement here is invented test data. No confidential document, personal record or government database was used. The fixture is a simple known-answer smoke case, not an independent benchmark, real tender or engineering recommendation.

## Runtime inputs

- `inputs/proc001-native.pdf`: two-page PDF with native text and a proposal table.
- `inputs/proc001-scanned.pdf`: the same two pages rasterized at 150 DPI, with no embedded text. This must exercise real OCR.
- `inputs/proc001-measurements.csv`: reserved for the later coding task; not an input to this round's PDF smoke test.

Run the PDF variants in isolated namespaces. Indexing both together lets the native version hide a broken scanned-document path.

Page 1 defines four requirements and includes three deliberately synthetic identifier strings. Page 2 contains three supplier proposals, an explicitly absent delivery term and no prices. Expected facts, citation pages, missing-evidence outcomes and prohibited output strings are in `expected/proc001-ground-truth.json`.

## Evaluation isolation

Never ingest the expected-answer file, source JSON, README, generator, verification scripts, manifest or validation reports into the application. Only manifest-listed input PDFs go through upload, extraction, review, indexing and answering. The answer key is used by a separate checker after output is returned.

Do not mark the document approved by updating its database row. Use ordinary demo-scoped review actions. A failed OCR/privacy gate is a failed or unavailable test stage, not a reason to bypass the gate. Confirm raw synthetic identifiers are absent from all released derivative surfaces. Native text contains real detector inputs, although the data itself is fictional.

## Generation and validation

Run `python generate_fixtures.py` with ReportLab, pypdf and `pdftoppm` available. The generator writes the two PDFs and validation report. It verifies page counts, absence of text in the rasterized PDF, expected source tokens and independently recomputed comparison/measurement outcomes.

The generator also writes `manifest.json`, including a SHA-256 for every package file except the manifest itself. These hashes check transport integrity, not factual correctness. Codex separately reviews the rendered pages before publishing the outer `PACKAGE_READY.json` marker.

`fixture-validation.json` records file and ground-truth checks only. It never means live OCR, Qdrant, BGE-M3 or Qwen passed. Those are Gemini's actual runtime tests.

Use the pinned environment tested for the project when running the application. Creating these fixture PDFs does not download any model or start a service. Generated raster previews are QA intermediates and are not part of the imported fixture package.
