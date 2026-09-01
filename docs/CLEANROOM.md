# Clean-room policy and freedom-to-operate

ZFP is developed **clean-room** from public specifications, published product behaviour, and
general document-analysis technique. This document states the rules the project works under.

## What the design is derived from

- The ISO 32000 / PDF object model as documented publicly: `/AcroForm`, field dictionaries,
  widget annotations, appearance streams, incremental updates, cross-reference streams.
- Publicly documented *behaviour* of commercial products — that Acrobat exposes an
  "automatically detect form fields" preference, that Acrobat Sign scans visual cues and
  predicts types near a signature, that scanned PDFs get OCR applied automatically when
  edited. Behaviour is observable; observing it is not copying.
- An Adobe-assigned patent on recognition and population of form fields, read as **prior art
  to design around and cite**, not as a blueprint to implement. Its architectural
  separation — field recognition, field suggestion, object detection — is a natural
  decomposition that this project reached independently and describes in its own terms.
- Open research and datasets for document understanding (layout models, form-understanding
  benchmarks).

## What is forbidden in this repository

1. **No proprietary source.** No decompiled, disassembled, or leaked code from any
   commercial PDF product, in any form, including "translated to Python".
2. **No claim-language transcription.** Patent claim text is not to be pasted into code,
   comments, docstrings, or design documents as an implementation instruction.
3. **No reproduction of proprietary data files** — font programs, trained model weights,
   dictionaries, or resource bundles extracted from a commercial product.
4. **No trade-dress copying.** Product names, icons, and UI strings of other vendors do not
   appear in the user-facing surface.

## What is expressly permitted

- Reading public vendor documentation and marketing pages to build the capability surface.
- Implementing published standards and file formats.
- Using permissively licensed open-source components within their licence terms.
- Independently arriving at an architecture that resembles a competitor's, because the
  problem constrains the solution. Two people asked to place a widget on a printed
  underline will both find the underline first.

## Freedom-to-operate

A formal FTO review is a **precondition of commercialisation**, not of development. Before
ZFP is distributed commercially:

- A patent search over form-field recognition, automatic field placement, form auto-fill
  from stored prior responses, and document-layout analysis must be run by qualified counsel.
- Any claim that reads on a ZFP subsystem must be resolved by design-around, licence, or
  removal of that subsystem — recorded in `docs/FTO.md` (created at that time).
- Google Patents legal-status information is explicitly not a legal conclusion and must not
  be relied on as one.

Nothing in this document is legal advice.

## Attribution discipline in the codebase

Where a technique is standard, cite the standard. Where a behaviour was inferred from a
product, say "observed behaviour" and name what was observed. Never write "this is how
$VENDOR does it" in a source comment — the project does not know that, and the claim creates
exactly the impression a clean-room process exists to avoid.
