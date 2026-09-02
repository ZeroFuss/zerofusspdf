"""Text placement mathematics for form-field appearances.

Pure geometry and metrics: nothing here emits PDF syntax.  :mod:`zfp.appearance.streams`
turns these layouts into content streams.

Every coordinate produced here is in the *appearance XObject's own space* -- origin at
``(0, 0)``, extending to ``(rect.width, rect.height)``.  That is the space a form
XObject's ``/BBox`` establishes, and placing text in page coordinates instead is the
single most common appearance-stream bug: the glyphs land hundreds of points off-screen
and the field renders blank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..core.geometry import Rect
from ..core.types import FieldSpec, FieldType
from ..pdfio import fonts

#: Multiple of the font size used as the distance between consecutive baselines.
LEADING_FACTOR = 1.16

#: Default inset applied inside a widget rectangle before text is placed.
DEFAULT_PADDING = 2.0

#: Fraction of a radio widget's radius filled by the selected dot.
RADIO_DOT_RATIO = 0.42


@dataclass
class TextLayout:
    """Where each line of a value sits inside an appearance XObject."""

    lines: List[str] = field(default_factory=list)
    font_size: float = 0.0
    line_height: float = 0.0
    origins: List[Tuple[float, float]] = field(default_factory=list)
    clipped: bool = False
    alignment: int = 0
    base_font: str = "Helvetica"

    @property
    def baselines(self) -> List[float]:
        """The y coordinate of each line's baseline."""
        return [y for _x, y in self.origins]

    @property
    def origin_x(self) -> float:
        """The x coordinate of the first line, or 0.0 when there are none."""
        return self.origins[0][0] if self.origins else 0.0


@dataclass
class CombLayout:
    """Per-character placement for a comb field."""

    characters: List[str] = field(default_factory=list)
    positions: List[float] = field(default_factory=list)
    baseline: float = 0.0
    font_size: float = 0.0
    cell_width: float = 0.0
    cells: int = 0
    clipped: bool = False
    base_font: str = "Helvetica"


def _padding_for(spec: FieldSpec, padding: Optional[float]) -> float:
    if padding is not None:
        return max(0.0, float(padding))
    return max(DEFAULT_PADDING, float(spec.border_width))


def _is_multiline(spec: FieldSpec, override: Optional[bool]) -> bool:
    if override is not None:
        return bool(override)
    return bool(spec.multiline) or spec.field_type is FieldType.MULTILINE_TEXT


def resolve_size(value: str, spec: FieldSpec, rect: Rect, padding: float,
                 *, multiline: bool = False) -> float:
    """The font size to draw ``value`` at: the spec's, or an auto-fit one.

    A fixed ``spec.font_size`` is honoured exactly -- an over-long value is clipped
    rather than silently shrunk, because a form whose text size jumps around between
    fields looks broken.  Size ``0`` means auto, which is what the PDF specification
    means by it too.
    """
    if spec.font_size and spec.font_size > 0:
        return round(float(spec.font_size), 2)
    if multiline:
        # Wrapping absorbs width, so only the height constrains the choice; cap at a
        # size that leaves room for at least two lines in a tall box.
        usable = max(1.0, rect.height - 2 * padding)
        return round(max(4.0, min(12.0, usable / (2 * LEADING_FACTOR))), 2)
    return fonts.fit_font_size(value or "M", spec.font_name, rect,
                               max_size=12.0, min_size=4.0, padding=padding)


def _aligned_x(line: str, base_font: str, size: float, inner_width: float,
               padding: float, alignment: int) -> float:
    width = fonts.text_width(line, base_font, size)
    if alignment == 1:
        return padding + max(0.0, (inner_width - width) / 2.0)
    if alignment == 2:
        return padding + max(0.0, inner_width - width)
    return padding


def _single_line_baseline(rect: Rect, base_font: str, size: float, padding: float) -> float:
    """Vertically centre one line using real font metrics.

    The visual centre of a line of text is the midpoint of its ascent-to-descent span,
    not the midpoint of the em box, so the baseline sits at::

        (height - (ascent + descent)) / 2 - descent

    with ``descent`` negative.  Using ``height/2`` instead leaves text noticeably low.
    """
    ascent = fonts.font_ascent(base_font) * size / 1000.0
    descent = fonts.font_descent(base_font) * size / 1000.0  # negative
    baseline = (rect.height - (ascent + descent)) / 2.0 - descent
    lower = padding
    upper = max(padding, rect.height - padding - ascent)
    return round(min(max(baseline, lower), upper), 3)


def layout_text(value: str, spec: FieldSpec, rect: Rect, *,
                multiline: Optional[bool] = None,
                padding: Optional[float] = None) -> TextLayout:
    """Place ``value`` inside ``rect``, in the appearance XObject's own space."""
    rect = rect.normalized()
    pad = _padding_for(spec, padding)
    base_font = fonts.resolve_base_font(spec.font_name)
    ml = _is_multiline(spec, multiline)
    text = value or ""
    size = resolve_size(text, spec, rect, pad, multiline=ml)
    inner_width = max(0.0, rect.width - 2 * pad)
    line_height = round(size * LEADING_FACTOR, 3)

    if not text:
        return TextLayout(lines=[], font_size=size, line_height=line_height,
                          origins=[], clipped=False, alignment=spec.alignment,
                          base_font=base_font)

    if ml:
        lines = fonts.wrap_text(text, base_font, size, inner_width) if inner_width > 0 else [text]
        ascent = fonts.font_ascent(base_font) * size / 1000.0
        top = rect.height - pad - ascent
        origins: List[Tuple[float, float]] = []
        for i, line in enumerate(lines):
            y = top - i * line_height
            origins.append((_aligned_x(line, base_font, size, inner_width, pad,
                                       spec.alignment), round(y, 3)))
        # A line whose baseline drops below the padding would be drawn outside the box.
        visible = [i for i, (_x, y) in enumerate(origins) if y >= pad - line_height * 0.25]
        clipped = len(visible) < len(lines)
        return TextLayout(lines=lines, font_size=size, line_height=line_height,
                          origins=origins, clipped=clipped, alignment=spec.alignment,
                          base_font=base_font)

    line = text.replace("\n", " ").replace("\r", " ")
    baseline = _single_line_baseline(rect, base_font, size, pad)
    x = _aligned_x(line, base_font, size, inner_width, pad, spec.alignment)
    clipped = fonts.text_width(line, base_font, size) > inner_width + 1e-6
    return TextLayout(lines=[line], font_size=size, line_height=line_height,
                      origins=[(round(x, 3), baseline)], clipped=clipped,
                      alignment=spec.alignment, base_font=base_font)


def layout_comb(value: str, spec: FieldSpec, rect: Rect,
                *, padding: Optional[float] = None) -> CombLayout:
    """Distribute ``value`` one character per cell across a comb field.

    A comb divides the widget into ``comb_cells`` equal columns and centres one glyph in
    each.  Characters beyond the cell count are dropped -- the field physically cannot
    hold them -- and ``clipped`` records that.
    """
    rect = rect.normalized()
    pad = _padding_for(spec, padding)
    base_font = fonts.resolve_base_font(spec.font_name)
    cells = int(spec.comb_cells or spec.max_length or 0)
    if cells <= 0:
        cells = max(1, len(value or ""))
    cell_width = rect.width / cells if cells else rect.width

    text = (value or "").replace("\n", "").replace("\r", "")
    clipped = len(text) > cells
    chars = list(text[:cells])

    if spec.font_size and spec.font_size > 0:
        size = round(float(spec.font_size), 2)
    else:
        widest = max((fonts.text_width(c, base_font, 10.0) for c in chars), default=6.0)
        by_width = (cell_width * 0.72) / (widest / 10.0) if widest > 0 else 12.0
        by_height = max(1.0, rect.height - 2 * pad) / LEADING_FACTOR
        size = round(max(4.0, min(12.0, by_width, by_height)), 2)

    baseline = _single_line_baseline(rect, base_font, size, pad)
    positions: List[float] = []
    for i, ch in enumerate(chars):
        centre = (i + 0.5) * cell_width
        positions.append(round(centre - fonts.text_width(ch, base_font, size) / 2.0, 3))

    return CombLayout(characters=chars, positions=positions, baseline=baseline,
                      font_size=size, cell_width=round(cell_width, 4), cells=cells,
                      clipped=clipped, base_font=base_font)


def layout_choice(value: str, spec: FieldSpec, rect: Rect,
                  *, padding: Optional[float] = None) -> TextLayout:
    """Single-line placement with room reserved on the right for a dropdown arrow."""
    rect = rect.normalized()
    arrow = min(12.0, max(6.0, rect.height * 0.7))
    inner = Rect(rect.x0, rect.y0, max(rect.x0, rect.x1 - arrow), rect.y1)
    return layout_text(value, spec, Rect(0.0, 0.0, inner.width, rect.height),
                       multiline=False, padding=padding)


def arrow_path(rect: Rect) -> List[Tuple[str, Tuple[float, ...]]]:
    """A small downward triangle at the right edge, for combo boxes."""
    rect = rect.normalized()
    size = min(7.0, max(3.5, rect.height * 0.28))
    cx = rect.width - size - 3.0
    cy = rect.height / 2.0
    return [
        ("m", (cx, cy + size * 0.35)),
        ("l", (cx + size, cy + size * 0.35)),
        ("l", (cx + size / 2.0, cy - size * 0.45)),
        ("h", ()),
    ]


def check_mark_path(rect: Rect, style: str = "check") -> List[Tuple[str, Tuple[float, ...]]]:
    """Drawing operations for a check-state mark, in XObject space.

    Returned as ``(operator, operands)`` pairs using only ``m``, ``l``, ``c`` and ``h``,
    so the caller decides whether to stroke or fill.  Drawing the mark as a real path
    means a checkbox needs no font at all -- one less thing to go wrong in a viewer that
    resolves resources differently.
    """
    rect = rect.normalized()
    w, h = rect.width, rect.height
    if w <= 0 or h <= 0:
        return []
    inset = min(w, h) * 0.22
    x0, y0 = inset, inset
    x1, y1 = w - inset, h - inset
    cx, cy = w / 2.0, h / 2.0
    r = min(w, h) / 2.0 - inset

    style = (style or "check").lower()
    if style == "cross":
        return [("m", (x0, y0)), ("l", (x1, y1)), ("m", (x0, y1)), ("l", (x1, y0))]
    if style == "square":
        return [("m", (x0, y0)), ("l", (x1, y0)), ("l", (x1, y1)), ("l", (x0, y1)), ("h", ())]
    if style == "diamond":
        return [("m", (cx, y0)), ("l", (x1, cy)), ("l", (cx, y1)), ("l", (x0, cy)), ("h", ())]
    if style == "circle":
        return circle_path(cx, cy, max(0.5, r))
    if style == "star":
        import math
        pts: List[Tuple[str, Tuple[float, ...]]] = []
        for i in range(10):
            radius = r if i % 2 == 0 else r * 0.42
            angle = math.pi / 2 + i * math.pi / 5
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            pts.append(("m" if i == 0 else "l", (round(x, 3), round(y, 3))))
        pts.append(("h", ()))
        return pts
    # Default: a tick whose elbow sits low-left of centre.
    return [
        ("m", (x0 + (x1 - x0) * 0.06, y0 + (y1 - y0) * 0.52)),
        ("l", (x0 + (x1 - x0) * 0.38, y0 + (y1 - y0) * 0.16)),
        ("l", (x1, y1)),
    ]


def circle_path(cx: float, cy: float, r: float) -> List[Tuple[str, Tuple[float, ...]]]:
    """A closed circle built from four cubic Bezier arcs."""
    k = 0.5523 * r
    return [
        ("m", (round(cx - r, 4), round(cy, 4))),
        ("c", (round(cx - r, 4), round(cy + k, 4), round(cx - k, 4), round(cy + r, 4),
               round(cx, 4), round(cy + r, 4))),
        ("c", (round(cx + k, 4), round(cy + r, 4), round(cx + r, 4), round(cy + k, 4),
               round(cx + r, 4), round(cy, 4))),
        ("c", (round(cx + r, 4), round(cy - k, 4), round(cx + k, 4), round(cy - r, 4),
               round(cx, 4), round(cy - r, 4))),
        ("c", (round(cx - k, 4), round(cy - r, 4), round(cx - r, 4), round(cy - k, 4),
               round(cx - r, 4), round(cy, 4))),
        ("h", ()),
    ]


#: ZapfDingbats characters for the check styles, for callers that prefer the font route.
ZAPF_STYLES = {
    "check": "4", "cross": "8", "diamond": "u", "circle": "l", "square": "n", "star": "H",
}


def zapf_char_for(style: str) -> str:
    """The ZapfDingbats character conventionally used for a check style."""
    return ZAPF_STYLES.get((style or "check").lower(), "4")


def measure(layout: TextLayout) -> Tuple[float, float]:
    """Overall (width, height) occupied by a laid-out value."""
    if not layout.lines:
        return (0.0, 0.0)
    width = max(fonts.text_width(line, layout.base_font, layout.font_size)
                for line in layout.lines)
    return (round(width, 3), round(len(layout.lines) * layout.line_height, 3))


def fits(layout: TextLayout, rect: Rect) -> bool:
    """Whether a layout sits entirely inside ``rect`` without clipping."""
    if layout.clipped:
        return False
    w, h = measure(layout)
    return w <= rect.normalized().width + 1e-6 and h <= rect.normalized().height + 1e-6


__all__ = [
    "LEADING_FACTOR", "DEFAULT_PADDING", "RADIO_DOT_RATIO", "ZAPF_STYLES",
    "TextLayout", "CombLayout",
    "layout_text", "layout_comb", "layout_choice", "resolve_size",
    "check_mark_path", "circle_path", "arrow_path", "zapf_char_for", "measure", "fits",
]
