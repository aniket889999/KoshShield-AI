# Synthetic procurement evaluation

This checkpoint improves the trustworthiness of the PROC-001 demonstration and
adds reusable reference calculations. It does not complete real-model Stage 0,
make a procurement decision, or establish production accuracy.

## Two separate commands

`make evaluate-fixtures` performs read-only deterministic checks with existing
Python dependencies. It verifies the fixture manifest, validates source/answer-key
schemas, recomputes 12 supplier/requirement verdicts and checks 3 measurement rows.
It does not initialize models, contact Qdrant, start services, install anything,
regenerate PDFs or write reports. Python `-B` prevents bytecode writes.

`make smoke-local` uses actual local adapters and isolated scratch resources for
native and scanned PDFs. It can load configured models and exercise local services.
It is not the read-only reference command. Missing providers remain
`NOT_EXECUTED`, not a successful AI demonstration.

The reference CLI supports a local `--fixture-dir` for a copied synthetic package:

```bash
.venv/bin/python -B -m koshshield.evaluation.reference --fixture-dir /path/to/synthetic-package
```

Its JSON is metadata-only. Exit 0 means the declared reference calculations agree
with the stored answer key; exit 1 means a validation or comparison failed. No
contact fields, prompts, source text or local paths are included in the result.

## What is verified

- Manifest paths are bounded and relative; escaping links, duplicate entries,
  size/hash mismatches, missing files and non-synthetic packages are rejected.
- Files are snapshotted after verification. Later disk changes do not replace the
  bytes supplied to the model. Only listed `input` PDFs can enter document intake.
- Evaluation-only JSON, the generator and README never enter model context.
- Answer keys require nonempty, unique variants/questions, valid page bounds,
  finite typed quantities, supported units and consistent missing-evidence flags.
- Comparison values come from the source case, not expected labels. Missing
  proposal terms remain `MISSING`; no imputation or supplier ranking is performed.
- Numeric comparisons use Decimal and explicit `>=`/`<=` rules. Proposal fields
  carry the schema's declared units; no arbitrary unit conversion is attempted.
- CSV columns declare `m3/h` and `bar`. Cells require bounded, nonnegative decimal
  notation; malformed, infinite, blank, formula-like and ragged input fails.
  Duplicate equipment IDs and empty datasets fail rather than disappearing.
- Runtime context is checked against the expected tenant, document, index and
  redaction versions, document hash, page bounds and masked-content hash.
- Citation checks evaluate the chunk IDs returned by the model. Retrieving the
  correct page without citing it no longer earns citation coverage.
- A vision probe requires the intended page's approved current masked derivative.
  Plaintext digest and PNG signature are checked before the existing model client
  performs its image decoding checks. Missing images cannot fall back to text-only.

Manifest digests establish consistency with the checked-in package, not external
publisher authenticity. The initial rollout exposed a stale generator checksum
from the original fixture import; that metadata was reconciled to the reviewed
checked-in generator. PDFs, source values and expected answers were not regenerated.

## Answer-grading limits

Numeric presence checks now match complete quantities, including units and both
proposed/required values. A substring like `180 m3/h` cannot satisfy `80 m3/h`.
Common forms such as `80.0 m3/h` and a superscript cubic unit are accepted.

These checks cannot determine negation, supplier attribution, contradictions in
prose, or whether a citation entails an answer. Reports therefore keep
`supported_facts_matched: null`, `semantic_support_verified: false` and set
`answer_validation_scope` to `deterministic_checks_only_not_semantic_entailment`.
For unknown facts, numeric checks are null; the insufficient-evidence flag is
tested, but an invented value hidden in prose still requires semantic review.
Even a fully passing smoke report is not a semantic accuracy result.

The synthetic-identifier check covers known fixture strings, with Unicode and
case normalization. It is not a universal PII detector or privacy guarantee.
Separate production privacy gates remain in place.

## Reporting and reproducibility

The reference report includes comparison/measurement check and mismatch counts,
missing proposal values, manifest consistency, and explicit
`model_evaluation_executed: false` / `pdf_content_semantics_checked: false` flags.
It compares record identities as well as values, so omitted or extra predictions
cannot inflate a score. A modified, correctly rehashed but incorrect answer key
still fails reference comparison.

Smoke question results separate retrieved-page coverage, cited-page coverage,
citation validity, numeric checks, insufficient-evidence agreement and actual
visual input. Missing stage records and empty question results cannot produce a
vacuous pass. Question failure codes contain no raw model text.

Tests use synthetic local fixtures and mocks where runtime services are absent.
The read-only CLI is also exercised with Python audit guards rejecting network,
subprocess and file-write attempts. This is not kernel isolation evidence.

Remaining work: real OCR/BGE-M3/Qdrant/Qwen execution; semantic review of returned
claims; held-out cases; authenticated case APIs; real agent-generated Office/code
deliverables. The reference evaluator supplies trusted expected calculations for
that later workflow, but does not represent it as already implemented.
