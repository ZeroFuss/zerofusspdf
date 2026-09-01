# Quality gates

ZFP is built around **testable gates rather than subjective demonstrations**. Every subsystem
below has a metric, and every metric has a threshold that CI enforces against the synthetic
corpus.

## The dashboard

| System | Metric | Implemented in |
|---|---|---|
| Field geometry | IoU, center error, per-edge coordinate error | `qa.metrics.field_geometry_metrics` |
| Field recall | fraction of true fields detected | `qa.metrics.recall_precision` |
| False positives | non-input regions made fillable | `qa.metrics.recall_precision` |
| Field type | macro-F1 across text/check/radio/choice/signature/date | `qa.metrics.type_macro_f1` |
| Label association | correct label→field rate | `qa.metrics.label_association_rate` |
| Canonical semantics | exact canonical-key accuracy | `qa.metrics.canonical_accuracy` |
| OCR | character and word error rate | `qa.metrics.ocr_cer_wer` |
| Auto-fill | exact-value match rate | `qa.metrics.autofill_exact_match` |
| Validation | fraction of values satisfying inferred constraints | `resolver.validators` |
| Repeat consistency | agreement across repeated fields | `qa.metrics.repeat_consistency` |
| PDF integrity | parse/repair errors after save | `qa.verify.check_integrity` |
| Viewer compatibility | field round trip after reopen | `qa.verify.check_fields_roundtrip` |
| Visual preservation | pixel difference outside edited regions | `qa.renderdiff.visual_preservation_report` |
| Forms | appearance/value consistency after reopen+save | `qa.verify.check_fields_roundtrip` |
| Security | unauthorized encrypted/signed mutation count | `agents.specialists.SecurityGateAgent` |
| Privacy | unapproved PII egress count | `council.redaction` + policy assertions |
| Automation | documents completed without user field placement | `pipeline.batch` summary |

## Enforced thresholds

`tests/integration/test_quality_gates.py` fails the build below these, measured on the
deterministic synthetic corpus:

| Corpus | Recall | Mean IoU | Type F1 | Canonical accuracy |
|---|---|---|---|---|
| `underline` | ≥ 0.90 | ≥ 0.80 | ≥ 0.85 | ≥ 0.75 |
| `boxed` | ≥ 0.90 | ≥ 0.80 | ≥ 0.85 | ≥ 0.75 |
| `checkbox` | ≥ 0.85 | ≥ 0.70 | ≥ 0.85 | — |
| `comb` | ≥ 0.85 | ≥ 0.75 | — | — |
| `mixed` | ≥ 0.80 | ≥ 0.72 | ≥ 0.80 | ≥ 0.65 |

Plus absolute invariants, which have no tolerance at all:

- The original file's bytes are a **literal prefix** of an incrementally written output.
- Every written widget rectangle lies inside its page's crop box.
- Reopening the output yields exactly the fields that were written, with the same values.
- No two widgets on a page overlap by more than 10% IoU.
- An encrypted or signed document is never mutated without explicit authorization.
- No `secret`-sensitivity value ever appears in a council context or a log record.

## Why synthetic data is the primary corpus

The generator knows the perfect ground truth by construction — it *placed* the label, the
rule, and the blank, so it knows the exact rectangle, type, and canonical key. That gives
essentially unlimited labelled geometry with no hand-drawn boxes.

`zfp.synth` deliberately randomizes fonts, line thickness, field spacing, checkbox shapes,
underlines vs. borderless blanks, tables, comb cells, multi-column layouts, page rotation,
multi-page structure, repeated fields, and (through the raster path) paper texture, scan
noise, skew, perspective, blur, JPEG artifacts and DPI.

## The regression corpus

Synthetic data cannot cover pathology. The production regression corpus must contain real
native PDFs, scanned PDFs, hybrid files, **damaged files**, encrypted examples, rotated
pages, existing AcroForms, static forms, static XFA, dynamic XFA, signed documents, complex
tables, and forms from multiple jurisdictions. AcroForm and XFA variants are distinct
fixtures, never assumed equivalent.

Damaged files matter more than they look: `PdfFile.rebuild_xref()` exists because real
documents arrive with broken cross-reference tables, and a form engine that cannot open them
is not a form engine.

## Running the gates

```
PYTHONPATH=src python3 -m pytest tests -q            # everything
PYTHONPATH=src python3 -m pytest tests/integration -q # gates only
PYTHONPATH=src python3 -m zfp.cli.main doctor         # adapter availability
PYTHONPATH=src python3 -m zfp.cli.main metrics pred.json truth.json
```
