# Zero PDF (ZFP)

**A geometry-first PDF understanding engine.** It converts visual intent into native PDF
interactivity, resolves the semantics of each field against an authorized user-data graph,
fills everything it can deterministically, uses AI only to resolve ambiguity, and proves
afterward that nothing outside the intended regions changed.

Clean-room, local-first, Apache-2.0. **The core has zero third-party runtime dependencies.**

---

## The problem

Hand a normal tool a flat PDF form — a scanned rental application, a vendor onboarding
packet, a government permit — and it gives you a picture. You can annotate on top of it, but
nothing on the page is a *field*. Hand it a 400-page document and it gives up.

The usual answer is to point a large model at the page and ask where the blanks are. That is
slow, irreproducible, and geometrically wrong: a model asked for coordinates guesses
coordinates.

## The approach

ZFP inverts the order:

> **deterministic structure first, computer vision second, machine learning third,
> generative AI last.**

The page already contains the answer. A born-digital form has the underline as a real vector
path with exact endpoints. A scan has the same underline as ink a Hough transform finds in
milliseconds. Either way the rectangle is *measured*, never predicted. A model is only ever
asked what a field **means** — never where it is.

Then the original page is left completely untouched. ZFP adds a precisely aligned interactive
AcroForm layer over it and writes it as an incremental update, so the original bytes remain a
literal prefix of the output. The result is a genuinely fillable PDF in Acrobat, in browsers,
and in every compliant reader — not an overlay that only works inside one app.

## What it does

```
ANY STATIC FORM
      ↓  automatic classification      (existing form? native? scan? hybrid? XFA? signed?)
      ↓  automatic detection           (lines, boxes, circles, combs, blanks, table cells)
      ↓  native fillable fields        (real AcroForm widgets with appearance streams)
      ↓  automatic semantic naming     (label → canonical key, 240+ key ontology)
      ↓  automatic profile matching    (encrypted vault, provenance-tracked)
      ↓  automatic validation          (normalizers + validators + repeat consistency)
      ↓
FINISHED INTEROPERABLE PDF  + a proof that nothing else changed
```

## Install

```bash
pip install -e .                      # core: no third-party dependencies
pip install -e ".[vision,render]"     # optional: OpenCV/NumPy geometry, page rendering
pip install -e ".[ocr]"               # optional: Tesseract OCR for scans
pip install -e ".[all]"               # everything optional
```

Optional adapters are discovered at runtime. A missing one degrades a capability; it never
crashes a run.

```bash
zfp doctor      # exactly which adapters and binaries this machine has
```

## Use

```bash
zfp preflight  form.pdf                      # what kind of PDF is this, page by page
zfp detect     form.pdf --json               # candidate fields, rects, types, confidences
zfp build      form.pdf -o fillable.pdf      # detect and write real AcroForm fields
zfp fill       fillable.pdf -o done.pdf --vault me.zfpv
zfp auto       form.pdf -o done.pdf --vault me.zfpv   # the whole deployment
zfp verify     form.pdf done.pdf             # integrity, prefix, round trip, visual diff
zfp agents                                   # print the deployment tree
zfp synth out.pdf --kind mixed --seed 7      # a synthetic form with exact ground truth
```

```python
from zfp.pipeline.run import process
from zfp.vault.store import ProfileVault

vault = ProfileVault.load("me.zfpv", password="…")
report = process("form.pdf", out="done.pdf", vault=vault)

print(report.summary())
for v in report.fill_report.values:
    print(f"{v.field_name:32} {v.status:14} {v.value!r}  ({v.confidence:.3f})")
```

## Architecture

Nine layers, documented in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md):

| | |
|---|---|
| `zfp.pdfio` | Dependency-free PDF lexer, parser, xref/object streams, filters, incremental writer, base-14 metrics |
| `zfp.preflight` | Per-page triage: encrypted / signed / AcroForm / XFA / native / raster |
| `zfp.native` · `zfp.raster` · `zfp.ocr` | Content-stream interpretation; render, deskew, denoise; OCR cascade that **never OCRs text the PDF already contains** |
| `zfp.vision` · `zfp.candidates` · `zfp.fusion` | Geometric primitives → eleven archetype detectors → rectangles snapped to real ink |
| `zfp.semantics` · `zfp.ontology` | Spatial graph, label linking, sections, type inference, 240+ canonical keys and 800+ aliases |
| `zfp.council` | Multi-member council for unresolved semantics only; strict JSON schema, local members by default |
| `zfp.vault` · `zfp.resolver` | Encrypted provenance-bearing profile graph; normalization, validation, repeat propagation |
| `zfp.acroform` · `zfp.appearance` | Real fields, flags, appearance streams, comb/multiline/choice/check/radio |
| `zfp.qa` | Integrity, prefix preservation, round trip, render diff, the metric dashboard |

## The agent mesh

A run is a **deployment**. An orchestrator deploys stages of specialized agents in parallel;
they deploy sub-agents per page shard; facilitators reconcile their competing proposals; a
council settles what is still ambiguous. All of it is deterministic Python — two runs of the
same document produce byte-identical output and an identical trace.

See [`docs/AGENTS.md`](docs/AGENTS.md) for the full topology and the default stage plan.

Long documents are page-sharded. Page count is a scheduling parameter, never a capability
boundary — there is no 25-page limit anywhere in the system.

## Autonomy, honestly

"Zero touch" must never mean "silently hallucinate."

| Mode | Behaviour |
|---|---|
| `conservative` *(default)* | Fills only what clears the confidence and validation thresholds. Everything else is reported as **data unavailable** — with no manual box placement asked of the user. |
| `completion` | Uses council consensus and best-evidence resolution to fill every resolvable field, with confidence and provenance attached. |
| `off` | Detection and field creation only. |

Finding a field and knowing a truthful answer are different problems. ZFP is built to be
excellent at the first and honest about the second. Signature fields are a policy boundary,
not a text field: ZFP creates them and refuses to sign without an explicit `SigningPolicy`.

## Privacy

Local-first by default. `PrivacyConfig.allow_external_inference` is **False** out of the box,
and with it off no byte of the document leaves the machine. When egress is enabled, only a
redacted structural question is sent — never the document, never a page image, never a value
from the vault, and never anything tagged `secret`. See [`docs/PRIVACY.md`](docs/PRIVACY.md).

## Proof, not demos

Every subsystem has a metric and every metric has an enforced threshold
([`docs/QA.md`](docs/QA.md)) — geometry IoU, field recall, type macro-F1, label association,
canonical accuracy, OCR CER/WER, autofill exact match, repeat consistency, PDF integrity,
visual preservation. The primary corpus is synthetic, because the generator knows the perfect
ground truth by construction: it placed the label, the rule and the blank, so it knows the
exact rectangle, type and canonical key.

Invariants with no tolerance at all: the original bytes stay a literal prefix of the output;
every widget lies inside its crop box; reopening yields exactly the fields written; encrypted
and signed documents are never mutated without authorization.

## Clean room

ZFP is developed clean-room from public specifications, published product behaviour, and
general document-analysis technique. No proprietary source, no patent claim-language
transcription, no extracted resources. A formal freedom-to-operate review is a precondition
of commercialisation. See [`docs/CLEANROOM.md`](docs/CLEANROOM.md). Nothing there is legal
advice.

## Licence

Apache-2.0. The default install carries no other licence. MuPDF is deliberately **not** a
dependency — see the AGPL decision in [`docs/LICENSING.md`](docs/LICENSING.md).
