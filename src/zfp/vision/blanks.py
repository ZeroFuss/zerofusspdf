"""Blank-region analysis: where on the page there is *nothing*, and why that matters.

Half of the fields in a real-world form have no rule and no box.  They are a gap in the
page: a label, then eighty points of paper.  Finding those honestly means building an
occupancy model of the page and then asking for the maximal empty rectangles inside it
-- and then throwing away the ones that are merely margin.

The occupancy model is a coarse grid (2pt cells, capped at 400x400 so a poster-sized
page costs the same as a letter one).  Emptiness is decided per cell and conservatively:
a cell touched by any ink at all is occupied, so a blank region is genuinely blank.

Only *filled* shapes occupy their interior.  A stroked rectangle -- a table border, a
section frame -- occupies its four edges and nothing else, which is exactly why a
borderless field inside a framed section is still findable.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from ..core.geometry import EPS, PageGeometry, Rect
from ..core.logging import get_logger
from ..core.types import TextSpan, VectorPrimitive
from .primitives import ROUND_DIGITS, _as_rect, _detection, _painted, reading_key

__all__ = [
    "OccupancyGrid",
    "occupancy_grid",
    "maximal_empty_cells",
    "blank_regions",
    "suppress_redundant",
    "whitespace_profile",
    "line_gaps",
]

_log = get_logger(__name__)

#: Target occupancy cell size, in points.
CELL_PT = 2.0

#: Hard cap on grid resolution per axis; cells grow on a very large page.
MAX_CELLS_PER_AXIS = 400

#: Never return more blanks than this from one page.
MAX_BLANKS = 200

#: Enumeration cap for the histogram sweep.
_MAX_CANDIDATES = 20000

#: A filled shape larger than this fraction of the page is a background wash; only its
#: border is treated as ink, otherwise one grey panel would swallow the whole page.
_BACKGROUND_AREA_RATIO = 0.25

#: A blank counts as "below" a label only within this many line heights.
_BELOW_LINE_FACTOR = 4.0

#: Two blanks overlapping more than this are the same blank.
_BLANK_DEDUP_IOU = 0.8

#: A gap taller than this multiple of the median line height is a borderless field.
LINE_GAP_FACTOR = 1.6


# ======================================================================================
# Occupancy grid
# ======================================================================================


class OccupancyGrid:
    """A coarse boolean raster of where a page has ink.

    Cell ``(col, row)`` covers ``[x0 + col*cell_w, x0 + (col+1)*cell_w)`` horizontally
    and ``[y0 + row*cell_h, y0 + (row+1)*cell_h)`` vertically, so row 0 is the *bottom*
    of the page in PDF user space.  :meth:`cells_to_rect` converts back.
    """

    __slots__ = ("bounds", "cols", "rows", "cell_w", "cell_h", "data")

    def __init__(self, bounds: Rect, cols: int, rows: int) -> None:
        self.bounds = bounds
        self.cols = cols
        self.rows = rows
        self.cell_w = max(bounds.width, EPS) / float(cols)
        self.cell_h = max(bounds.height, EPS) / float(rows)
        self.data = bytearray(cols * rows)

    # -- writing ----------------------------------------------------------------------
    def mark(self, rect: Rect) -> None:
        """Mark every cell that ``rect`` touches, however slightly."""
        c0, r0, c1, r1 = self.cell_span(rect)
        if c1 < c0 or r1 < r0:
            return
        width = c1 - c0 + 1
        filler = b"\x01" * width
        for row in range(r0, r1 + 1):
            base = row * self.cols
            self.data[base + c0 : base + c1 + 1] = filler

    def mark_border(self, rect: Rect, thickness: float) -> None:
        """Mark only the four edges of ``rect`` (a stroked outline, not a filled panel)."""
        band = max(thickness, self.cell_w, self.cell_h)
        self.mark(Rect(rect.x0, rect.y0, rect.x1, min(rect.y1, rect.y0 + band)))
        self.mark(Rect(rect.x0, max(rect.y0, rect.y1 - band), rect.x1, rect.y1))
        self.mark(Rect(rect.x0, rect.y0, min(rect.x1, rect.x0 + band), rect.y1))
        self.mark(Rect(max(rect.x0, rect.x1 - band), rect.y0, rect.x1, rect.y1))

    # -- geometry ---------------------------------------------------------------------
    def cell_span(self, rect: Rect) -> Tuple[int, int, int, int]:
        """Inclusive ``(c0, r0, c1, r1)`` cell range covered by ``rect``, clamped."""
        norm = rect.normalized()
        c0 = int(math.floor((norm.x0 - self.bounds.x0) / self.cell_w))
        c1 = int(math.ceil((norm.x1 - self.bounds.x0) / self.cell_w)) - 1
        r0 = int(math.floor((norm.y0 - self.bounds.y0) / self.cell_h))
        r1 = int(math.ceil((norm.y1 - self.bounds.y0) / self.cell_h)) - 1
        c1 = max(c0, c1)
        r1 = max(r0, r1)
        if c1 < 0 or r1 < 0 or c0 >= self.cols or r0 >= self.rows:
            return (0, 0, -1, -1)
        return (max(0, c0), max(0, r0), min(self.cols - 1, c1), min(self.rows - 1, r1))

    def cells_to_rect(self, c0: int, r0: int, c1: int, r1: int) -> Rect:
        """Convert an inclusive cell range back to a user-space rectangle."""
        return Rect(
            self.bounds.x0 + c0 * self.cell_w,
            self.bounds.y0 + r0 * self.cell_h,
            self.bounds.x0 + (c1 + 1) * self.cell_w,
            self.bounds.y0 + (r1 + 1) * self.cell_h,
        ).rounded(ROUND_DIGITS)

    def occupied_ratio(self) -> float:
        """Fraction of cells carrying ink."""
        total = self.cols * self.rows
        if total <= 0:
            return 0.0
        return sum(self.data) / float(total)


def occupancy_grid(
    rects: Iterable[Rect], bounds: Rect, cell_pt: float = CELL_PT, cap: int = MAX_CELLS_PER_AXIS
) -> OccupancyGrid:
    """Rasterize ``rects`` into a coarse occupancy grid over ``bounds``."""
    box = bounds.normalized()
    step = max(float(cell_pt), EPS)
    cols = max(1, min(cap, int(math.ceil(max(box.width, EPS) / step))))
    rows = max(1, min(cap, int(math.ceil(max(box.height, EPS) / step))))
    grid = OccupancyGrid(box, cols, rows)
    for rect in rects:
        if rect is not None:
            grid.mark(rect)
    return grid


def _is_maximal(grid: OccupancyGrid, c0: int, c_end: int, row: int) -> bool:
    """True when a rectangle ending at ``row`` cannot simply grow into ``row + 1``.

    Without this test the sweep emits one rectangle per row of every empty column, all
    of them contained in the tallest one.  Checking the next row costs a single C-level
    slice and keeps the result set to the genuinely maximal rectangles.
    """
    if row + 1 >= grid.rows:
        return True
    base = (row + 1) * grid.cols
    return any(grid.data[base + c0 : base + c_end])


def maximal_empty_cells(
    grid: OccupancyGrid, min_cols: int, min_rows: int, limit: int = _MAX_CANDIDATES
) -> List[Tuple[int, int, int, int]]:
    """Enumerate maximal empty cell rectangles with the largest-rectangle sweep.

    For every grid row we maintain, per column, the run of free cells ending there, and
    pop the classic monotone stack.  Each pop yields one rectangle that cannot grow
    sideways at its height -- the maximal empty rectangles of the page, in
    ``(c0, r0, c1, r1)`` inclusive cell coordinates.
    """
    results: List[Tuple[int, int, int, int]] = []
    cols, rows = grid.cols, grid.rows
    if cols <= 0 or rows <= 0 or min_cols > cols or min_rows > rows:
        return results
    heights = [0] * cols
    for row in range(rows):
        base = row * cols
        for col in range(cols):
            heights[col] = 0 if grid.data[base + col] else heights[col] + 1
        stack: List[Tuple[int, int]] = []
        for col in range(cols + 1):
            current = heights[col] if col < cols else 0
            start = col
            while stack and stack[-1][1] >= current:
                s, height = stack.pop()
                width = col - s
                # A bar exactly as tall as its successor is not finished: the rectangle
                # can still grow to the right, so it is not maximal yet.
                if (
                    height > current
                    and height >= min_rows
                    and width >= min_cols
                    and _is_maximal(grid, s, col, row)
                ):
                    results.append((s, row - height + 1, col - 1, row))
                    if len(results) >= limit:
                        _log.warning("vision: blank enumeration hit the cap of %d", limit)
                        return results
                start = s
            if current > 0:
                stack.append((start, current))
    return results


# ======================================================================================
# Blank regions
# ======================================================================================


def _ink_rects(
    spans: Sequence[TextSpan],
    prims: Sequence[VectorPrimitive],
    images: Optional[Sequence[Rect]],
    bounds: Rect,
) -> Tuple[List[Rect], List[Tuple[Rect, float]]]:
    """Split the page's ink into solid rectangles and outline-only ones."""
    solid: List[Rect] = []
    outlines: List[Tuple[Rect, float]] = []
    page_area = max(bounds.area, EPS)
    for span in spans or []:
        if span is None:
            continue
        if getattr(span, "is_blank", None) is not None and span.is_blank():
            continue
        rect = _as_rect(span)
        if rect is not None:
            solid.append(rect)
    for prim in prims or []:
        if prim is None or not _painted(prim):
            continue
        rect = _as_rect(prim)
        if rect is None:
            continue
        filled = bool(getattr(prim, "filled", False))
        if filled and rect.area <= _BACKGROUND_AREA_RATIO * page_area:
            solid.append(rect)
        else:
            outlines.append((rect, float(getattr(prim, "stroke_width", 0.0) or 0.0)))
    for rect in images or []:
        if isinstance(rect, Rect):
            solid.append(rect.normalized())
    return solid, outlines


def _anchored(
    rect: Rect,
    spans: Sequence[Tuple[float, float, float, float]],
    beside: float,
    below: float,
) -> bool:
    """True when a blank sits beside or under text, rather than out in the margin.

    The spans arrive as plain coordinate tuples and the overlap arithmetic is inlined:
    this runs once per candidate per span, which on a dense page is a few hundred
    thousand times, and allocating a normalized :class:`Rect` for each would dominate
    the whole detector.
    """
    rx0, ry0, rx1, ry1 = rect.x0, rect.y0, rect.x1, rect.y1
    for sx0, sy0, sx1, sy1 in spans:
        if min(ry1, sy1) - max(ry0, sy0) > 0.0:
            if -EPS <= rx0 - sx1 <= beside:
                return True
            if -EPS <= sx0 - rx1 <= beside:
                return True
        if min(rx1, sx1) - max(rx0, sx0) > 0.0:
            if -EPS <= sy0 - ry1 <= below:
                return True
    return False


def suppress_redundant(rects: Sequence[Rect], limit: int = MAX_BLANKS) -> List[Rect]:
    """Sort blanks by ``(-area, y0, x0)`` and drop the ones another already covers.

    A rectangle contained in a kept one, or overlapping it by more than
    :data:`_BLANK_DEDUP_IOU`, says nothing new.  Sorting by area first means the survivor
    of any such pair is always the larger, which is what a caller wants to search inside.
    """
    ordered = sorted(rects, key=lambda r: (-r.area, r.y0, r.x0))
    kept: List[Rect] = []
    for rect in ordered:
        redundant = False
        for other in kept:
            if other.contains_rect(rect) or other.iou(rect) > _BLANK_DEDUP_IOU:
                redundant = True
                break
        if redundant:
            continue
        kept.append(rect)
        if len(kept) >= limit:
            break
    return kept


def blank_regions(
    spans: Sequence[TextSpan],
    prims: Sequence[VectorPrimitive],
    geometry: Optional[PageGeometry],
    config: Any = None,
    images: Optional[Sequence[Rect]] = None,
) -> List[Rect]:
    """Find the maximal empty rectangles that could hold a borderless field.

    The page is rasterized into an occupancy grid from the text spans, the ink
    primitives and any image rectangles; the maximal empty rectangles of that grid are
    enumerated; and then the geometry is filtered by meaning:

    * a region must be at least ``blank_min_width_pt`` x ``blank_min_height_pt``;
    * it must be horizontally adjacent to a text span (within ``label_max_distance_pt``)
      or sit directly below one.  A blank in the middle of nowhere is margin, not a
      field, and emitting it would poison every downstream score.

    Results are deduplicated (containment and IoU > 0.8), sorted by ``(-area, y0, x0)``
    and capped at :data:`MAX_BLANKS`.
    """
    det = _detection(config)
    if geometry is None:
        return []
    bounds = geometry.crop_box.normalized()
    if bounds.width <= EPS or bounds.height <= EPS:
        return []

    solid, outlines = _ink_rects(spans, prims, images, bounds)
    grid = occupancy_grid(solid, bounds)
    for rect, stroke in outlines:
        grid.mark_border(rect, max(stroke, det.max_line_thickness_pt))

    min_cols = max(1, int(math.ceil((det.blank_min_width_pt - EPS) / grid.cell_w)))
    min_rows = max(1, int(math.ceil((det.blank_min_height_pt - EPS) / grid.cell_h)))
    cells = maximal_empty_cells(grid, min_cols, min_rows)
    if not cells:
        return []

    span_boxes = [
        (r.x0, r.y0, r.x1, r.y1)
        for r in (_as_rect(s) for s in (spans or []))
        if r is not None
    ]
    if not span_boxes:
        return []
    beside = det.label_max_distance_pt
    below = max(_BELOW_LINE_FACTOR * det.blank_min_height_pt, 12.0)

    candidates: List[Rect] = []
    seen = set()
    for c0, r0, c1, r1 in cells:
        rect = grid.cells_to_rect(c0, r0, c1, r1)
        rect = geometry.clamp(rect).rounded(ROUND_DIGITS)
        if rect.width + EPS < det.blank_min_width_pt or rect.height + EPS < det.blank_min_height_pt:
            continue
        key = (rect.x0, rect.y0, rect.x1, rect.y1)
        if key in seen:
            continue
        seen.add(key)
        if not _anchored(rect, span_boxes, beside, below):
            continue
        candidates.append(rect)

    return suppress_redundant(candidates)


# ======================================================================================
# Profiles
# ======================================================================================


def whitespace_profile(
    spans: Sequence[TextSpan], geometry: Optional[PageGeometry]
) -> Tuple[List[float], List[float]]:
    """Ink coverage per horizontal band and per vertical band.

    Returns ``(rows, columns)``.  ``rows[0]`` is the band at the **top** of the page and
    ``columns[0]`` the band at its left edge; every value is the fraction of that band
    carrying text, from 0.0 (blank) to 1.0 (solid).  The layout stage reads the row
    profile to find gutters between blocks and the column profile to find columns.
    """
    if geometry is None:
        return ([], [])
    bounds = geometry.crop_box.normalized()
    if bounds.width <= EPS or bounds.height <= EPS:
        return ([], [])
    rects = []
    for span in spans or []:
        if span is None:
            continue
        if getattr(span, "is_blank", None) is not None and span.is_blank():
            continue
        rect = _as_rect(span)
        if rect is not None:
            rects.append(rect)
    grid = occupancy_grid(rects, bounds)
    cols, rows = grid.cols, grid.rows
    row_profile: List[float] = []
    for row in range(rows - 1, -1, -1):  # top band first
        base = row * cols
        row_profile.append(sum(grid.data[base : base + cols]) / float(cols))
    col_profile: List[float] = []
    for col in range(cols):
        total = 0
        for row in range(rows):
            total += grid.data[row * cols + col]
        col_profile.append(total / float(rows))
    return (row_profile, col_profile)


def _text_lines(spans: Sequence[TextSpan]) -> List[Rect]:
    """Group spans into line bands, top of the page first."""
    rects = []
    for span in spans or []:
        if span is None:
            continue
        if getattr(span, "is_blank", None) is not None and span.is_blank():
            continue
        rect = _as_rect(span)
        if rect is not None and rect.height > 0.0:
            rects.append(rect)
    if not rects:
        return []
    rects.sort(key=lambda r: (-r.center.y, r.x0))
    lines: List[Rect] = []
    for rect in rects:
        if lines:
            current = lines[-1]
            overlap = current.vertical_overlap(rect)
            if overlap >= 0.4 * min(current.height, rect.height):
                lines[-1] = current.union(rect)
                continue
        lines.append(rect)
    return lines


def line_gaps(spans: Sequence[TextSpan], config: Any = None) -> List[Rect]:
    """Vertical gaps between consecutive text lines that are too tall to be leading.

    A gap taller than :data:`LINE_GAP_FACTOR` times the median line height is where a
    borderless field lives -- the writer left room on purpose.  The rectangle returned
    spans the gap vertically and the union of the two neighbouring lines horizontally.
    """
    _detection(config)
    lines = _text_lines(spans)
    if len(lines) < 2:
        return []
    heights = sorted(line.height for line in lines)
    middle = len(heights) // 2
    if len(heights) % 2:
        median = heights[middle]
    else:
        median = (heights[middle - 1] + heights[middle]) / 2.0
    if median <= EPS:
        return []
    threshold = LINE_GAP_FACTOR * median
    out: List[Rect] = []
    for index in range(len(lines) - 1):
        upper = lines[index]
        lower = lines[index + 1]
        gap = upper.y0 - lower.y1
        if gap <= threshold + EPS:
            continue
        x0 = min(upper.x0, lower.x0)
        x1 = max(upper.x1, lower.x1)
        if x1 - x0 <= EPS:
            continue
        out.append(Rect(x0, lower.y1, x1, upper.y0).rounded(ROUND_DIGITS))
    out.sort(key=reading_key)
    return out
