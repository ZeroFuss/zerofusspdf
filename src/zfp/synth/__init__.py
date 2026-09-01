"""Synthetic form corpus: real PDFs with exact ground truth.

``zfp.synth`` draws printed forms the way a word processor would -- rules, boxes,
checkboxes, comb cells, ruled tables -- and hands back both the PDF bytes and the precise
rectangle, canonical key and expected value of every field it drew.  It is the yardstick
the whole detection stack is measured with, and the source of every fixture in the test
suite (no binary blobs in git).

    >>> from zfp.synth import SynthOptions, generate
    >>> form = generate(SynthOptions(kind="underline", seed=7))
    >>> form.pdf_bytes.startswith(b"%PDF-")
    True
    >>> all(f.canonical_key for f in form.fields)
    True
"""

from __future__ import annotations

from .content import ContentBuilder, attach_page_content, font_dictionary
from .generator import (
    KINDS,
    ROTATIONS,
    GroundTruthField,
    SyntheticForm,
    SynthOptions,
    generate,
    generate_corpus,
)
from .layouts import (
    PAGE_KINDS,
    Canvas,
    FieldMark,
    LabelSpec,
    PageDraw,
    Style,
    build_style,
    draw_page,
    spec_for,
    specs_for,
)

__all__ = [
    "ContentBuilder",
    "attach_page_content",
    "font_dictionary",
    "KINDS",
    "PAGE_KINDS",
    "ROTATIONS",
    "GroundTruthField",
    "SyntheticForm",
    "SynthOptions",
    "generate",
    "generate_corpus",
    "Canvas",
    "FieldMark",
    "LabelSpec",
    "PageDraw",
    "Style",
    "build_style",
    "draw_page",
    "spec_for",
    "specs_for",
]
