# ZFP Architecture

> **deterministic structure first, computer vision second, machine learning third,
> generative AI last.**

Using a generative model as the first stage makes the system slower, less reproducible and
less geometrically precise. Using it as an ambiguity resolver makes it extremely useful.

## The one-line description

ZFP is a geometry-first PDF understanding engine that converts visual intent into native PDF
interactivity, resolves the semantics of each field against an authorized user-data graph,
fills everything it can deterministically, uses AI only to resolve ambiguity, and proves
afterward that nothing outside the intended regions changed.

## Why an overlay, not a rebuild

For a static form, the original page is an **immutable visual substrate**. ZFP does not
reconstruct the page as editable text in order to fill it; it adds a precisely aligned
interactive AcroForm layer over it and writes that layer as an *incremental update*, so the
original bytes remain a literal prefix of the output file. That is both the highest-fidelity
result and the cheapest thing to prove correct.

Genuine content editing of a scan is a **different pathway** (`zfp.services.scanedit`):
OCR, localized background reconstruction, font/style estimation, text replacement.

## Routing table

| PDF condition | ZFP behaviour |
|---|---|
| Existing AcroForm | Read widget rectangles and field dictionaries directly; never re-detect |
| Flat born-digital form | Parse text/vector geometry, identify blank fields, overlay widgets |
| Scanned form | OCR + graphics detection + semantic association, then overlay widgets |
| Hybrid PDF | Process each page/region by whether native text or raster content is present |
| Scan needing wording changes | OCR into editable regions, reconstruct only affected text areas |
| XFA form | Route through a dedicated compatibility layer |
| Encrypted / restricted | Honour the security state and authorized credentials |
| Signed document | Preserve the signed revision unless a signing workflow makes a new one |

## Stack

```
   ┌──────────────┐
   │ File Ingress │
   └──────┬───────┘
          ▼
   ┌─────────────────────┐
   │ PDF Preflight       │  security / forms / XFA / text / raster
   └──────┬──────────────┘
   ┌──────┴──────────────────────────────────┐
   ▼                                         ▼
Native PDF path                        Raster / Scan path
 text + glyphs                          preprocessing
 vector primitives                            │
 annotations                                 OCR
 AcroForm                                     │
   └───────────────────┬──────────────────────┘
                       ▼
            ┌─────────────────────┐
            │ Candidate Generator │  lines / boxes / circles / blanks
            └──────────┬──────────┘
            ┌──────────▼──────────┐
            │ Geometry Fusion     │  exact PDF rects
            └──────────┬──────────┘
            ┌──────────▼──────────┐
            │ Semantic Linker     │  label / context / type
            └──────────┬──────────┘
                Ambiguous?  ──yes──►  AI semantic council (structured JSON only)
                       │ no                     │
                       └───────────┬────────────┘
            ┌──────────────────────▼──┐
            │ Canonical Form Schema   │
            └──────────┬──────────────┘
            ┌──────────▼──────────────┐
            │ User Data Resolver      │  + normalizers + validators
            └──────────┬──────────────┘
            ┌──────────▼──────────────┐
            │ AcroForm Writer         │  (or Content Editor)
            └──────────┬──────────────┘
            ┌──────────▼──────────────┐
            │ Render + QA + Diff      │
            └──────────┬──────────────┘
                       ▼
                   FINAL PDF
```

## Layer map

| Layer | Package | Responsibility |
|---|---|---|
| Object layer | `zfp.pdfio` | Dependency-free PDF lexer, parser, xref/objstm, filters, incremental writer, Document facade, base-14 metrics |
| Triage | `zfp.preflight` | Encrypted / signed / AcroForm / XFA / native / raster classification, per page |
| Native perception | `zfp.native` | Content-stream interpreter → text spans with glyph boxes, vector primitives |
| Raster perception | `zfp.raster`, `zfp.ocr` | Render, deskew/denoise/binarize, OCR cascade with per-word confidence |
| Geometry | `zfp.vision`, `zfp.candidates`, `zfp.fusion` | Rules, boxes, circles, blanks, comb cells, table cells → candidate rectangles snapped to real primitives |
| Semantics | `zfp.semantics`, `zfp.ontology` | Spatial graph, label linking, sections, type inference, canonical keys |
| Escalation | `zfp.council` | Multi-member council; external models only for unresolved semantics, under a privacy policy |
| Data | `zfp.vault`, `zfp.resolver` | Encrypted profile graph with provenance; normalization, validation, repeat propagation |
| Output | `zfp.acroform`, `zfp.appearance` | Real AcroForm fields, flags, appearance streams, comb/multiline/choice/check/radio |
| Proof | `zfp.qa` | Integrity, prefix preservation, field round trip, render/structural diff, metrics dashboard |
| Deployment | `zfp.agents` | Orchestrators, facilitators, councils, specialists, page-shard sub-agents |
| Surface | `zfp.pipeline`, `zfp.cli`, `zfp.api`, `zfp.services` | End-to-end runs, CLI, HTTP API, broader PDF services |

## Dependency policy

The entire core path — parse, detect, write, fill, verify — runs on a **bare CPython
standard library**. Every third-party library is an *optional adapter* resolved at runtime
through `zfp.core.optional`. A missing adapter degrades a capability; it never crashes a run.
`zfp doctor` prints exactly which adapters are present.

This is a deliberate architectural choice, not an aesthetic one: it keeps the licensing
surface small (see `docs/LICENSING.md`), it makes the geometry path reproducible on any
machine, and it means the quality gates measure ZFP's own algorithms rather than a vendor's.

## Coordinate discipline

An LLM is never ZFP's ruler. Rectangles crossing a module boundary are always PDF user
space, y-up, page origin, floats in points. Pixel space never escapes `zfp.raster` /
`zfp.ocr` / `zfp.vision`; those modules convert through `PageGeometry`, which applies the
render scale, the crop-box origin and `/Rotate` explicitly:

```
PDF_x = pixel_x / render_scale + crop_origin_x
PDF_y = page_height - pixel_y / render_scale + crop_origin_y     (rotation applied first)
```

Round-tripping a rectangle through `user_to_pixel` and back is exact to 1e-6 for all four
rotations. That property is asserted in the unit tests, because field-placement quality is
won or lost here.

## Candidate scoring

Each candidate carries independent evidence rather than one opaque score:

```
C(field) = 0.30*geometric + 0.20*blank_region + 0.15*nearby_label
         + 0.10*layout_consistency + 0.10*repeated_pattern
         + 0.10*semantic_type + 0.05*model_consensus
```

Weights live in `ScoringWeights` and are meant to be calibrated against the synthetic corpus,
not frozen. Confidence is reported as four separate numbers — geometry, label link, semantic
type, autofill value — so a run can be conservative about *meaning* while being certain about
*placement*.

## No 25-page limit

Long documents are page-sharded across sub-agents and reconciled globally afterwards
(`zfp.agents.subagents`). Page count is a scheduling parameter, never a capability boundary.
