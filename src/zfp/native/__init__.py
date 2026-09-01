"""``zfp.native`` -- reading a born-digital page.

Three layers, bottom up:

``encoding``
    Font dictionaries into ``(charcode, unicode)`` pairs and per-code advance widths.
``content``
    The content-stream interpreter: text spans with per-glyph boxes, vector primitives
    with real user-space coordinates, image boxes and white fills.
``text``
    Layout analysis over the resulting spans: lines, merged runs, columns, reading order.

Detection quality on a native PDF is decided almost entirely here, because everything
downstream -- rule detection, blank regions, label linking -- consumes these coordinates.

Nothing in this package imports a third-party library, and nothing in it raises on
malformed input: a broken content stream yields fewer primitives and a non-zero
:attr:`~zfp.native.content.ContentResult.errors`, never an exception.
"""

from __future__ import annotations

from .content import (
    MAX_FORM_DEPTH,
    WHITE_LUMINANCE,
    ContentResult,
    ContentState,
    ContentStreamInterpreter,
    analyze_page,
)
from .encoding import FontProgram, decode_string, font_widths, load_font
from .text import (
    detect_columns,
    group_spans_into_lines,
    merge_adjacent_spans,
    reading_order,
)

__all__ = [
    "ContentState",
    "ContentResult",
    "ContentStreamInterpreter",
    "analyze_page",
    "WHITE_LUMINANCE",
    "MAX_FORM_DEPTH",
    "FontProgram",
    "decode_string",
    "font_widths",
    "load_font",
    "group_spans_into_lines",
    "merge_adjacent_spans",
    "detect_columns",
    "reading_order",
]
