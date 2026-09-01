"""A miniature page painter: raw PDF content-stream operators, by hand.

The synthetic corpus has to draw the *visual substrate* of a form -- rules, boxes,
checkboxes, comb cells, labels -- and it has to do so with no third-party writer and no
guesswork about where the ink landed.  :class:`ContentBuilder` is that painter: every
method appends operators to a buffer and returns ``self``, so a layout reads as one
fluent chain, and :func:`attach_page_content` bolts the finished stream onto a page
together with the base-14 font dictionaries it references.

Nothing here knows what a *field* is.  Coordinates are PDF user space (y-up, page
origin, points) exactly as they appear in the content stream, which is what makes the
ground-truth rectangles in :mod:`zfp.synth.layouts` exact by construction rather than
by measurement.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

from ..core.errors import ValidationError
from ..pdfio.document import Document
from ..pdfio.filters import encode_flate
from ..pdfio.fonts import escape_pdf_text, resolve_base_font, text_width
from ..pdfio.objects import PdfArray, PdfDict, PdfName, PdfRef, PdfStream
from ..pdfio.writer import format_number

__all__ = [
    "ContentBuilder",
    "attach_page_content",
    "font_dictionary",
    "BEZIER_K",
]

#: Magic constant for approximating a quarter circle with a cubic Bezier.
BEZIER_K: float = 0.5522847498307936

#: The two base-14 faces whose built-in encoding must never be overridden.
_SYMBOLIC = ("Symbol", "ZapfDingbats")

FontMap = Union[Mapping[str, str], Iterable[Tuple[str, str]], None]


def _n(value: float) -> bytes:
    """Format a number the way the serializer does: at most six decimals, no exponent."""
    return format_number(float(value))


def _gray(value: float) -> float:
    """Clamp a gray level onto ``0.0..1.0``."""
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


class ContentBuilder:
    """Accumulates PDF content-stream operators for one page.

    Every drawing method returns ``self`` so calls chain.  Graphics state changes made
    by :meth:`rect`, :meth:`circle` and :meth:`dotted_line` are wrapped in ``q``/``Q``,
    so they never leak into whatever is drawn next; :meth:`gray` and
    :meth:`setlinewidth` deliberately do leak, because that is what a caller asking for
    them wants.

    Attributes:
        fonts_used: Resource name (``"F1"``) -> canonical base font name
            (``"Helvetica"``), filled in by :meth:`text`.  Hand this straight to
            :func:`attach_page_content`.
    """

    __slots__ = ("_ops", "fonts_used")

    def __init__(self) -> None:
        self._ops: List[bytes] = []
        self.fonts_used: Dict[str, str] = {}

    # -- raw ---------------------------------------------------------------------------
    def op(self, operators: bytes) -> "ContentBuilder":
        """Append a pre-formatted operator line verbatim."""
        if operators:
            self._ops.append(bytes(operators))
        return self

    # -- graphics state ----------------------------------------------------------------
    def save(self) -> "ContentBuilder":
        """Push the graphics state (``q``)."""
        return self.op(b"q")

    def restore(self) -> "ContentBuilder":
        """Pop the graphics state (``Q``)."""
        return self.op(b"Q")

    def gray(self, v: float) -> "ContentBuilder":
        """Set both the fill and the stroke colour to gray level ``v`` (0 = black)."""
        level = _n(_gray(v))
        return self.op(level + b" g " + level + b" G")

    def setlinewidth(self, w: float) -> "ContentBuilder":
        """Set the stroke width in points (``w``)."""
        return self.op(_n(max(0.0, float(w))) + b" w")

    # -- text --------------------------------------------------------------------------
    def text(
        self,
        x: float,
        y: float,
        s: str,
        font_res: str = "F1",
        size: float = 10.0,
        base_font: str = "Helvetica",
    ) -> "ContentBuilder":
        """Draw ``s`` with its baseline origin at ``(x, y)``.

        The text matrix is set absolutely (``1 0 0 1 x y Tm``) rather than relatively, so
        a caller never has to reason about leftover text-state from an earlier run.

        Args:
            x: Baseline origin x, user space.
            y: Baseline origin y, user space.
            s: The string.  Encoded as WinAnsi and escaped by
                :func:`~zfp.pdfio.fonts.escape_pdf_text`.
            font_res: The ``/Font`` resource name to select, without the slash.
            size: Font size in points.
            base_font: Any font name; resolved onto one of the base-14 faces.

        Returns:
            ``self``.
        """
        if not s:
            return self
        canonical = resolve_base_font(base_font)
        name = str(font_res).lstrip("/") or "F1"
        self.fonts_used[name] = canonical
        payload = escape_pdf_text(str(s))
        return self.op(
            b"BT /" + name.encode("ascii", "replace") + b" " + _n(size) + b" Tf 1 0 0 1 "
            + _n(x) + b" " + _n(y) + b" Tm (" + payload + b") Tj ET"
        )

    def width_of(self, s: str, base_font: str = "Helvetica", size: float = 10.0) -> float:
        """Return the advance width of ``s`` -- the same metric :meth:`text` will use."""
        return text_width(str(s), resolve_base_font(base_font), float(size))

    # -- paths -------------------------------------------------------------------------
    def line(self, x0: float, y0: float, x1: float, y1: float, width: float = 0.6) -> "ContentBuilder":
        """Stroke a single straight segment."""
        return self.op(
            b"q " + _n(width) + b" w " + _n(x0) + b" " + _n(y0) + b" m "
            + _n(x1) + b" " + _n(y1) + b" l S Q"
        )

    def rect(
        self,
        x0: float,
        y0: float,
        w: float,
        h: float,
        width: float = 0.6,
        fill: Optional[float] = None,
    ) -> "ContentBuilder":
        """Draw an axis-aligned rectangle whose lower-left corner is ``(x0, y0)``.

        Args:
            x0: Lower-left x.
            y0: Lower-left y.
            w: Width in points.
            h: Height in points.
            width: Stroke width; ``0`` draws a fill-only rectangle.
            fill: Gray level to fill with, or ``None`` for no fill.
        """
        parts = [b"q", _n(width) + b" w"]
        if fill is not None:
            parts.append(_n(_gray(fill)) + b" g")
        parts.append(
            _n(x0) + b" " + _n(y0) + b" " + _n(w) + b" " + _n(h) + b" re"
        )
        if fill is not None:
            parts.append(b"B" if width > 0.0 else b"f")
        else:
            parts.append(b"S")
        parts.append(b"Q")
        return self.op(b" ".join(parts))

    def circle(
        self,
        cx: float,
        cy: float,
        r: float,
        width: float = 0.6,
        fill: Optional[float] = None,
    ) -> "ContentBuilder":
        """Draw a circle as four cubic Bezier quadrants -- the PDF idiom for an ellipse."""
        r = abs(float(r))
        if r <= 0.0:
            return self
        k = BEZIER_K * r
        parts = [b"q", _n(width) + b" w"]
        if fill is not None:
            parts.append(_n(_gray(fill)) + b" g")
        parts.append(_n(cx + r) + b" " + _n(cy) + b" m")
        quadrants = (
            (cx + r, cy + k, cx + k, cy + r, cx, cy + r),
            (cx - k, cy + r, cx - r, cy + k, cx - r, cy),
            (cx - r, cy - k, cx - k, cy - r, cx, cy - r),
            (cx + k, cy - r, cx + r, cy - k, cx + r, cy),
        )
        for x1, y1, x2, y2, x3, y3 in quadrants:
            parts.append(
                _n(x1) + b" " + _n(y1) + b" " + _n(x2) + b" " + _n(y2) + b" "
                + _n(x3) + b" " + _n(y3) + b" c"
            )
        parts.append(b"h")
        if fill is not None:
            parts.append(b"B" if width > 0.0 else b"f")
        else:
            parts.append(b"S")
        parts.append(b"Q")
        return self.op(b" ".join(parts))

    def dotted_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        width: float = 0.6,
        dash: Tuple[float, ...] = (1.0, 2.0),
    ) -> "ContentBuilder":
        """Stroke a dashed/dotted segment.  ``dash`` is the ``d`` operator's array."""
        pattern = b" ".join(_n(v) for v in (dash or (1.0, 2.0)))
        return self.op(
            b"q [" + pattern + b"] 0 d " + _n(width) + b" w "
            + _n(x0) + b" " + _n(y0) + b" m " + _n(x1) + b" " + _n(y1) + b" l S Q"
        )

    # -- output ------------------------------------------------------------------------
    def is_empty(self) -> bool:
        """True when nothing has been drawn yet."""
        return not self._ops

    def build(self) -> bytes:
        """Return the content stream bytes, one operator group per line."""
        if not self._ops:
            return b""
        return b"\n".join(self._ops) + b"\n"

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "ContentBuilder(ops=%d, fonts=%r)" % (len(self._ops), sorted(self.fonts_used))


# --------------------------------------------------------------------------------------
# Attaching a stream to a page
# --------------------------------------------------------------------------------------


def font_dictionary(base_font: str) -> PdfDict:
    """Return a ``/Type /Font`` dictionary for a base-14 face.

    ``/Encoding /WinAnsiEncoding`` is added for the text faces (it is what
    :func:`~zfp.pdfio.fonts.escape_pdf_text` encodes to) but never for Symbol or
    ZapfDingbats, whose built-in encodings must not be overridden.
    """
    canonical = resolve_base_font(base_font)
    entries = {
        "Type": PdfName("Font"),
        "Subtype": PdfName("Type1"),
        "BaseFont": PdfName(canonical),
    }
    if canonical not in _SYMBOLIC:
        entries["Encoding"] = PdfName("WinAnsiEncoding")
    return PdfDict(entries)


def _normalize_fonts(fonts_used: FontMap) -> Dict[str, str]:
    """Coerce the ``fonts_used`` argument into ``{resource_name: base_font}``."""
    if fonts_used is None:
        return {}
    if isinstance(fonts_used, (str, bytes, bytearray)):
        raise ValidationError(
            "fonts_used must be a mapping or (resource, font) pairs, got %r" % (fonts_used,)
        )
    items: Iterable[Tuple[str, str]]
    if isinstance(fonts_used, Mapping):
        items = fonts_used.items()
    else:
        items = fonts_used
    out: Dict[str, str] = {}
    for entry in items:
        # A bare string unpacks into characters, which would silently produce nonsense.
        if isinstance(entry, (str, bytes, bytearray)):
            raise ValidationError(
                "fonts_used entries must be (resource, font) pairs, got %r" % (entry,)
            )
        try:
            name, base = entry
        except (TypeError, ValueError):
            raise ValidationError(
                "fonts_used must be a mapping or (resource, font) pairs, got %r" % (entry,)
            ) from None
        key = str(name).lstrip("/")
        if not key:
            raise ValidationError("a font resource name may not be empty")
        out[key] = resolve_base_font(str(base))
    return out


def attach_page_content(
    doc: Document,
    page_index: int,
    content_bytes: bytes,
    fonts_used: FontMap = None,
    *,
    compress: bool = False,
) -> PdfRef:
    """Install ``content_bytes`` as page ``page_index``'s content stream.

    Replaces ``/Contents`` (reusing the existing content object number when there is
    one, so object numbering stays stable and no orphan is left behind), rebuilds
    ``/Resources`` with a ``/Font`` entry per name in ``fonts_used``, and stages every
    touched object with the document's writer.

    Args:
        doc: The open document; it must already have the page.
        page_index: Zero-based page index.
        content_bytes: The operators, as produced by :meth:`ContentBuilder.build`.
        fonts_used: ``{resource_name: font_name}`` for every font the stream selects.
        compress: Flate-compress the stream.  Deterministic, and transparent to
            :meth:`~zfp.pdfio.document.Page.content_bytes`.

    Returns:
        The reference to the content stream object.

    Raises:
        ValidationError: ``fonts_used`` is malformed, or the page index is out of range.
    """
    page = doc.page(page_index)
    writer = doc.writer
    fonts = _normalize_fonts(fonts_used)

    payload = bytes(content_bytes or b"")
    stream_dict = PdfDict()
    if compress and payload:
        payload = encode_flate(payload)
        stream_dict["Filter"] = PdfName("FlateDecode")
    stream_dict["Length"] = len(payload)
    stream = PdfStream(stream_dict, payload)

    existing_contents = page.dict.get("Contents")
    if isinstance(existing_contents, PdfRef):
        ref = existing_contents
        writer.set_object(ref.num, stream)
    else:
        ref = writer.add_object(stream)
        page.dict["Contents"] = ref

    # -- resources ---------------------------------------------------------------------
    resources = PdfDict()
    inherited = doc.resolve(page.dict.get("Resources")) if "Resources" in page.dict else None
    if isinstance(inherited, PdfDict):
        resources.update(inherited)

    font_dir = PdfDict()
    current_fonts = doc.resolve(resources.get("Font")) if "Font" in resources else None
    if isinstance(current_fonts, PdfDict):
        font_dir.update(current_fonts)

    for name in sorted(fonts):
        canonical = fonts[name]
        present = doc.resolve(font_dir.get(name)) if name in font_dir else None
        if isinstance(present, PdfDict) and present.get_name("BaseFont") == canonical:
            continue
        font_dir[name] = writer.add_object(font_dictionary(canonical))

    if len(font_dir):
        resources["Font"] = font_dir
    resources["ProcSet"] = PdfArray([PdfName("PDF"), PdfName("Text")])
    page.dict["Resources"] = resources
    page.touch()
    return ref
