"""Stage wiring: the perception packages driven as one system.

:mod:`zfp.pipeline.detect` is the perception entry point.  It profiles a document, picks
the native or the raster sensing path for each page, and returns the fused field
candidates -- the whole of `preflight -> native | raster+ocr+vision -> candidates ->
fusion` behind three functions.

    >>> from zfp.pdfio.document import Document
    >>> from zfp.pipeline import detect_document
    >>> from zfp.synth import SynthOptions, generate
    >>> profile, candidates = detect_document(
    ...     Document.open(generate(SynthOptions(kind="boxed", seed=1)).pdf_bytes)
    ... )
    >>> profile.doc_class.value
    'flat_native_form'
    >>> all(c.rect.width > 0 for c in candidates)
    True
"""

from __future__ import annotations

from .detect import (
    NATIVE_MODES,
    RASTER_MODES,
    PageSensing,
    detect_document,
    detect_page,
    page_context,
    page_sensing,
    wants_raster,
)

__all__ = [
    "NATIVE_MODES",
    "RASTER_MODES",
    "PageSensing",
    "detect_document",
    "detect_page",
    "page_context",
    "page_sensing",
    "wants_raster",
]
