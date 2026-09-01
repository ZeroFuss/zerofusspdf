# Licensing architecture

Licensing is an architectural constraint in ZFP, not an afterthought. The dependency policy
in `docs/ARCHITECTURE.md` — a stdlib-only core with optional adapters — exists partly so
that the licence surface of a *default* installation is trivially small.

## ZFP itself

Apache License 2.0 (`LICENSE`). Permissive, patent-grant bearing, and compatible with the
components below.

## Default installation

**No third-party runtime dependencies.** The core path — parse, detect, write, fill,
verify — is pure CPython standard library. A default install carries the Python licence and
Apache-2.0, nothing else.

## Optional adapters, by licence class

| Extra | Component | Licence | Effect on a closed-source distribution |
|---|---|---|---|
| `pdfbackends` | qpdf / pikepdf | Apache-2.0 / MPL-2.0 | Safe. MPL-2.0 is file-level copyleft; ZFP does not modify pikepdf files. |
| `pdfbackends` | pypdf | BSD-3-Clause | Safe. |
| `ocr` | Tesseract | Apache-2.0 | Safe. |
| `ocr` | Pillow | MIT-CMU | Safe. |
| `ocr` | OCRmyPDF | MPL-2.0 (check the pinned version) | Safe when unmodified and separately distributed. |
| `paddle` | PaddleOCR | Apache-2.0 | Safe; model weights carry their own terms — check per model. |
| `vision` | OpenCV | Apache-2.0 (4.5+) | Safe. |
| `vision` | NumPy | BSD-3-Clause | Safe. |
| `render` | pypdfium2 / PDFium | BSD-3-Clause / Apache-2.0 | Safe. |
| `api` | FastAPI, uvicorn | MIT / BSD-3-Clause | Safe. |

## The MuPDF decision

MuPDF (and `PyMuPDF`) is **AGPL-3.0 or commercial**. AGPL's network clause reaches a hosted
SaaS deployment, which is exactly how a PDF product is usually delivered.

**Policy:** MuPDF is never a default dependency and is not listed in any extra. ZFP detects
`fitz` at runtime only as a *user-supplied* rendering backend, because a user who already has
PyMuPDF installed under their own licence terms may legitimately use it. `zfp doctor` labels
it explicitly as AGPL/commercial so nobody adopts it by accident.

The default renderer preference order is therefore: `pypdfium2` → `pdftoppm` (Poppler,
GPL-2.0, invoked as a separate *process*, which is a distribution question rather than a
linking one) → embedded-image extraction (no dependency at all) → `fitz` only if the user
explicitly enables it.

## Model weights and hosted inference

- Locally run model weights carry their own licences, frequently *not* OSI-approved and
  frequently restricting commercial use. Any weight shipped or auto-downloaded must be
  recorded here with its licence before it is added.
- Hosted inference (the optional council member) is a **service term** question, not a
  licence question: retention, training-on-inputs, and sub-processor policy. See
  `docs/PRIVACY.md`.

## Rules for contributors

1. A new runtime dependency in the default install requires an explicit decision recorded
   in this file. The default answer is no.
2. No AGPL or GPL component may become a default or an extra. Process-boundary use must be
   documented as such.
3. Any vendored code carries its original licence header and an entry in `NOTICE`.
4. Model weights are never committed to this repository.
