"""Layout analysis over extracted text spans: lines, runs, columns, reading order.

A content stream emits one span per show operation, which is an artefact of how the
producer wrote the file and not how a person reads the page.  ``Name:`` and its trailing
colon can be two spans; a whole paragraph can be one; a two-column form interleaves the
two columns in stream order.  This module rebuilds the human view:

* :func:`group_spans_into_lines` clusters spans onto text lines by baseline;
* :func:`merge_adjacent_spans` glues the fragments of one visual run back together,
  inserting a space only where the geometry says there was one;
* :func:`detect_columns` finds vertical gutters by projecting spans onto the x axis;
* :func:`reading_order` walks columns left to right and lines top to bottom.

Everything here is pure geometry over :class:`~zfp.core.types.TextSpan`; nothing touches
a PDF.  All four functions are deterministic, tolerate ``baseline=None`` (falling back to
the box centre, which is what an OCR span usually gives) and never mutate their input --
:func:`merge_adjacent_spans` returns new spans.

Rotated pages are handled by the caller: spans arrive in PDF user space, so a page with
``/Rotate 90`` still has horizontal lines in this coordinate system.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..core.geometry import PageGeometry, Rect
from ..core.logging import get_logger
from ..core.types import TextSpan

__all__ = [
    "LINE_TOLERANCE_RATIO",
    "SPACE_GAP_RATIO",
    "GUTTER_RATIO",
    "WIDE_SPAN_RATIO",
    "baseline_of",
    "span_size",
    "group_spans_into_lines",
    "merge_adjacent_spans",
    "detect_columns",
    "reading_order",
]

_log = get_logger(__name__)

#: Two spans share a line when their baselines differ by less than this times the
#: median font size on the page.
LINE_TOLERANCE_RATIO = 0.5
#: A horizontal gap wider than this times the font size means a real space character.
SPACE_GAP_RATIO = 0.18
#: A vertical gap wider than this times the page width is a column gutter.
GUTTER_RATIO = 0.04
#: Spans wider than this times the page width span columns and are ignored when
#: projecting for gutters (a banner headline must not weld two columns together).
WIDE_SPAN_RATIO = 0.6
#: Used when a span carries neither a font size nor a measurable height.
_FALLBACK_SIZE = 10.0


def baseline_of(span: TextSpan) -> float:
    """Return a span's baseline, falling back to the vertical centre of its box.

    Args:
        span: Any text span, native or OCR.

    Returns:
        The baseline y in user space.

    Examples:
        >>> from zfp.core.geometry import Rect
        >>> baseline_of(TextSpan("a", Rect(0, 10, 5, 20), 0))
        15.0
    """
    if span.baseline is not None:
        return float(span.baseline)
    return float(span.rect.center.y)


def span_size(span: TextSpan) -> float:
    """Return a usable font size for a span: ``font_size``, else its box height.

    Args:
        span: Any text span.

    Returns:
        A strictly positive size in points.
    """
    if span.font_size and span.font_size > 0.0:
        return float(span.font_size)
    height = span.rect.height
    if height > 0.0:
        return float(height)
    return _FALLBACK_SIZE


def _median(values: Sequence[float]) -> float:
    """Median of a non-empty sequence; ``0.0`` for an empty one."""
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _line_sort_key(span: TextSpan) -> Tuple[float, float, str]:
    """Left-to-right order within a line, with deterministic tie-breaking."""
    return (span.rect.x0, -baseline_of(span), span.text)


def group_spans_into_lines(spans: Sequence[TextSpan]) -> List[List[TextSpan]]:
    """Cluster spans onto text lines by baseline, top of the page first.

    The clustering tolerance is ``0.5 * median font size`` over the whole input, so a
    page of 10 pt body text tolerates 5 pt of baseline jitter -- enough for superscripts
    and mixed fonts on one line, tight enough not to swallow the next line.  Spans with
    no baseline use the vertical centre of their box.

    Args:
        spans: The spans to group; order does not matter.

    Returns:
        Lines from top to bottom, each sorted left to right.  Never contains an empty
        line, and every input span appears exactly once.

    Examples:
        >>> from zfp.core.geometry import Rect
        >>> a = TextSpan("b", Rect(50, 100, 60, 110), 0, font_size=10, baseline=100)
        >>> b = TextSpan("a", Rect(10, 100, 20, 110), 0, font_size=10, baseline=100)
        >>> c = TextSpan("c", Rect(10, 60, 20, 70), 0, font_size=10, baseline=60)
        >>> [[s.text for s in line] for line in group_spans_into_lines([a, b, c])]
        [['a', 'b'], ['c']]
    """
    items = [span for span in spans if span is not None]
    if not items:
        return []
    tolerance = LINE_TOLERANCE_RATIO * _median([span_size(span) for span in items])
    if tolerance <= 0.0:
        tolerance = LINE_TOLERANCE_RATIO * _FALLBACK_SIZE

    ordered = sorted(items, key=lambda s: (-baseline_of(s), s.rect.x0, s.text))
    lines: List[List[TextSpan]] = []
    reference = 0.0
    for span in ordered:
        base = baseline_of(span)
        if lines and abs(base - reference) <= tolerance:
            lines[-1].append(span)
            continue
        lines.append([span])
        reference = base
    for line in lines:
        line.sort(key=_line_sort_key)
    return lines


def _clone(span: TextSpan) -> TextSpan:
    """Copy a span, including an independent ``glyph_rects`` list."""
    return TextSpan(
        text=span.text,
        rect=span.rect,
        page=span.page,
        font_name=span.font_name,
        font_size=span.font_size,
        source=span.source,
        confidence=span.confidence,
        glyph_rects=list(span.glyph_rects),
        baseline=span.baseline,
    )


def _mergeable(left: TextSpan, right: TextSpan) -> bool:
    """True when two spans may legally be joined into one run.

    Native and OCR text never merge, and neither do a visible span and an invisible one:
    an OCR layer hidden behind a scan sits on exactly the same baseline as the picture of
    the words, and welding them would produce doubled text.
    """
    if left.page != right.page or left.source != right.source:
        return False
    return (left.confidence > 0.0) == (right.confidence > 0.0)


def merge_adjacent_spans(spans: Sequence[TextSpan], gap_ratio: float = 0.3) -> List[TextSpan]:
    """Join spans that sit on one line and are close enough to be one visual run.

    Two spans merge when the horizontal gap between them is at most
    ``gap_ratio * font_size``; a space is inserted into the joined text when that gap
    exceeds ``0.18 * font_size``, which is roughly the width of a thin space and reliably
    separates ``"Name" ":"`` (no space) from ``"First" "Name"`` (space).  ``glyph_rects``
    are concatenated so per-character geometry survives the merge; the inserted space
    carries the gap itself as its rectangle, keeping ``len(glyph_rects) == len(text)``.

    Args:
        spans: The spans to merge; grouped into lines internally, so any order works.
        gap_ratio: Maximum gap, as a multiple of the font size, that still merges.

    Returns:
        New spans -- the inputs are never mutated -- ordered by line (top to bottom) and
        then left to right.

    Examples:
        >>> from zfp.core.geometry import Rect
        >>> a = TextSpan("First", Rect(10, 90, 40, 100), 0, font_size=10, baseline=90)
        >>> b = TextSpan("Name", Rect(43, 90, 75, 100), 0, font_size=10, baseline=90)
        >>> [s.text for s in merge_adjacent_spans([a, b])]
        ['First Name']
    """
    ratio = float(gap_ratio)
    out: List[TextSpan] = []
    for line in group_spans_into_lines(spans):
        current: Optional[TextSpan] = None
        for span in line:
            if current is None:
                current = _clone(span)
                continue
            size = max(span_size(current), span_size(span))
            gap = span.rect.x0 - current.rect.x1
            if not _mergeable(current, span) or gap > ratio * size:
                out.append(current)
                current = _clone(span)
                continue
            aligned = len(current.glyph_rects) == len(current.text) and len(
                span.glyph_rects
            ) == len(span.text)
            if gap > SPACE_GAP_RATIO * size:
                current.text += " "
                if aligned:
                    current.glyph_rects.append(
                        Rect(
                            current.rect.x1, span.rect.y0, span.rect.x0, span.rect.y1
                        ).normalized()
                    )
            current.text += span.text
            current.glyph_rects.extend(span.glyph_rects)
            current.rect = current.rect.union(span.rect)
            current.font_size = max(current.font_size, span.font_size)
            current.confidence = min(current.confidence, span.confidence)
            if not current.font_name:
                current.font_name = span.font_name
            if current.baseline is None:
                current.baseline = span.baseline
        if current is not None:
            out.append(current)
    return out


def _page_bounds(spans: Sequence[TextSpan], page_geometry: Optional[PageGeometry]) -> Optional[Rect]:
    """Return the box columns are measured against: the crop box, else the spans' hull."""
    if page_geometry is not None:
        crop = page_geometry.crop_box
        if crop.width > 0.0 and crop.height > 0.0:
            return crop
    return Rect.bounding([span.rect for span in spans])


def detect_columns(
    spans: Sequence[TextSpan], page_geometry: Optional[PageGeometry] = None
) -> List[Rect]:
    """Find the page's text columns by projecting span x-ranges onto the x axis.

    Every span contributes its horizontal extent to a coverage map; a run of x with no
    coverage at all, wider than ``0.04 * page width``, is a gutter, and the covered runs
    between gutters are the columns.  Spans wider than ``0.6 * page width`` are excluded
    from the projection -- a full-width heading physically bridges every gutter and would
    otherwise collapse a two-column page into one -- but they still count towards the
    vertical extent of whichever column they overlap most.

    Args:
        spans: The page's text spans.
        page_geometry: The page's geometry, used for the page width.  ``None`` falls back
            to the bounding box of the spans themselves.

    Returns:
        One :class:`~zfp.core.geometry.Rect` per column, left to right.  Each rectangle
        spans the column's own horizontal extent and the vertical extent of the text in
        it.  A single-column page returns exactly one rectangle; an empty input returns
        an empty list.

    Examples:
        >>> from zfp.core.geometry import Rect
        >>> left = TextSpan("l", Rect(50, 700, 250, 710), 0, font_size=10)
        >>> right = TextSpan("r", Rect(350, 700, 550, 710), 0, font_size=10)
        >>> [round(c.x0) for c in detect_columns([left, right])]
        [50, 350]
    """
    items = [span for span in spans if span is not None and span.rect.width >= 0.0]
    if not items:
        return []
    bounds = _page_bounds(items, page_geometry)
    if bounds is None or bounds.width <= 0.0:
        return [Rect.bounding([span.rect for span in items])]  # pragma: no cover
    page_width = bounds.width
    threshold = GUTTER_RATIO * page_width

    narrow = [span for span in items if span.rect.width <= WIDE_SPAN_RATIO * page_width]
    projected = narrow if narrow else items
    intervals = sorted((span.rect.x0, span.rect.x1) for span in projected)

    runs: List[List[float]] = []
    for x0, x1 in intervals:
        if runs and x0 - runs[-1][1] < threshold:
            if x1 > runs[-1][1]:
                runs[-1][1] = x1
            continue
        runs.append([x0, x1])
    if not runs:  # pragma: no cover - intervals is non-empty here
        return []

    members: List[List[TextSpan]] = [[] for _ in runs]
    for span in items:
        members[_best_run(span, runs)].append(span)

    columns: List[Rect] = []
    for run, group in zip(runs, members):
        boxes = [span.rect for span in group]
        hull = Rect.bounding(boxes)
        y0 = hull.y0 if hull is not None else bounds.y0
        y1 = hull.y1 if hull is not None else bounds.y1
        columns.append(Rect(run[0], y0, run[1], y1))
    return columns


def _best_run(span: TextSpan, runs: Sequence[Sequence[float]]) -> int:
    """Index of the run a span belongs to: most horizontal overlap, else nearest."""
    best_index = 0
    best_overlap = -1.0
    for index, run in enumerate(runs):
        overlap = min(span.rect.x1, run[1]) - max(span.rect.x0, run[0])
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    if best_overlap > 0.0:
        return best_index
    centre = span.rect.center.x
    best_index = 0
    best_distance = float("inf")
    for index, run in enumerate(runs):
        distance = abs(centre - (run[0] + run[1]) / 2.0)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return best_index


def reading_order(
    spans: Sequence[TextSpan], page_geometry: Optional[PageGeometry] = None
) -> List[TextSpan]:
    """Return the spans in the order a person reads them.

    Columns first, left to right; within a column, lines top to bottom; within a line,
    left to right.  On a single-column page this reduces to plain line order.

    A column layout is only *believed* when at least two of the detected columns carry
    two or more text lines each.  :func:`detect_columns` reports any gutter it finds, and
    on a sparse page (three stray spans, a header and a page number) that can be a gutter
    between things a reader still reads across.  Ordering a page column-first on that
    evidence would be worse than not doing it at all, so the fallback is plain line
    order.

    Args:
        spans: The page's text spans.
        page_geometry: The page's geometry, for the column projection.

    Returns:
        A flat list holding every input span exactly once.

    Examples:
        >>> from zfp.core.geometry import Rect
        >>> a = TextSpan("top", Rect(50, 700, 100, 710), 0, font_size=10, baseline=700)
        >>> b = TextSpan("bottom", Rect(50, 600, 100, 610), 0, font_size=10, baseline=600)
        >>> [s.text for s in reading_order([b, a])]
        ['top', 'bottom']
    """
    items = [span for span in spans if span is not None]
    if not items:
        return []
    columns = detect_columns(items, page_geometry)
    if len(columns) > 1:
        runs = [(column.x0, column.x1) for column in columns]
        buckets: Dict[int, List[TextSpan]] = {index: [] for index in range(len(runs))}
        for span in items:
            buckets[_best_run(span, runs)].append(span)
        grouped = [group_spans_into_lines(buckets[index]) for index in range(len(runs))]
        if sum(1 for lines in grouped if len(lines) >= 2) >= 2:
            return [span for lines in grouped for line in lines for span in line]
    return [span for line in group_spans_into_lines(items) for span in line]
