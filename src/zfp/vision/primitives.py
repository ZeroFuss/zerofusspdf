"""Geometric primitive detection over vector paths -- the native, pure-python path.

A one-point horizontal rule in a clean form is not a machine-learning problem: it is a
run of collinear segments with a length, a thickness and two endpoints.  This module is
the whole of that reasoning, expressed over :class:`~zfp.core.types.VectorPrimitive`
objects produced by :mod:`zfp.native.content`.

Everything here is deterministic and dependency-free.  Every rectangle that leaves the
module is PDF user space (y-up, page origin, points), rounded to three decimals so that
two runs over the same document produce byte-identical output.

Vocabulary
----------
rule
    A primitive whose bounding box is long on one axis and thinner than
    ``DetectionConfig.max_line_thickness_pt`` on the other.  Underlines, table borders,
    box sides and leader rules are all rules.
box
    Either a painted ``"rect"`` primitive, or four rules whose endpoints meet at four
    corners.
lattice
    The set of horizontal and vertical *levels* (rules clustered by their perpendicular
    coordinate) from which table cells are reconstructed.
"""

from __future__ import annotations

import bisect
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.config import DetectionConfig
from ..core.geometry import EPS, PageGeometry, Point, Rect
from ..core.logging import get_logger
from ..core.types import TextSpan, VectorPrimitive

__all__ = [
    "GLYPH_CHECKBOXES",
    "ZAPF_BOX_CHARS",
    "normalize_primitives",
    "horizontal_rules",
    "vertical_rules",
    "merge_collinear",
    "boxes_from_rules",
    "detect_boxes",
    "detect_circles",
    "detect_checkbox_glyphs",
    "detect_table_cells",
    "detect_comb_cells",
    "rule_spans",
    "reading_key",
    "dedupe_rects",
    "blank_regions",
]

_log = get_logger(__name__)

#: Rounding applied to every rectangle that leaves this module.
ROUND_DIGITS = 3

#: A primitive smaller than this on *both* axes is a stray point, not a mark.
_MIN_EXTENT_PT = 0.05

#: Explicit ``"line"`` primitives qualify as rules at this fraction of the configured
#: minimum length: a 15pt initials underline is a field, a 15pt table tick is not.
_LINE_LENGTH_RELAX = 0.6

#: A fragment shorter than this is noise, never a dash of a longer rule.
_FRAGMENT_MIN_PT = 1.0

#: A painted rectangle needs both sides above this to be a box rather than a mark.
_BOX_MIN_DIM_PT = 4.0

#: Two boxes overlapping by more than this are the same box.
_BOX_DEDUP_IOU = 0.8

#: Work caps.  Detection is a fixed cost per page, never a combinatorial one.
_MAX_BOX_RULES = 120
_MAX_BOXES = 2000
_MAX_LATTICE_RULES = 200
_MAX_CELL_SPAN = 8
#: How many lattice levels past a cell's own edge the search may look at while trying
#: to find its closing edge.  Only levels actually drawn at the cell (see
#: :meth:`_Level.reaches`) count against :data:`_MAX_CELL_SPAN`; the rest are skipped,
#: and this bounds how many of those skips one search will pay for.
_MAX_CELL_SCAN = 64
_CELL_WORK_BUDGET = 1_500_000
_MAX_CELLS = 4000
_MINIMALITY_LIMIT = 400

#: Text that *is* a checkbox all by itself.  A run of underscores is not one of them.
GLYPH_CHECKBOXES = frozenset(
    {
        "□",  # WHITE SQUARE
        "☐",  # BALLOT BOX
        "☑",  # BALLOT BOX WITH CHECK
        "☒",  # BALLOT BOX WITH X
        "❑",  # LOWER RIGHT SHADOWED WHITE SQUARE
        "❒",  # UPPER RIGHT SHADOWED WHITE SQUARE
        "❏",  # LOWER RIGHT DROP-SHADOWED WHITE SQUARE
        "❐",  # UPPER RIGHT DROP-SHADOWED WHITE SQUARE
        "◻",  # WHITE MEDIUM SQUARE
        "⬜",  # WHITE LARGE SQUARE
        "○",  # WHITE CIRCLE
        "◎",  # BULLSEYE
        "◯",  # LARGE CIRCLE
        "[]",
        "[ ]",
        "()",
        "( )",
    }
)

#: ZapfDingbats codes for the square/circle family (a72-a78).  The black bullet ``l``
#: and the check/cross marks are deliberately absent: a bullet list is not a form.
ZAPF_BOX_CHARS = frozenset("mnopqrs")


# ======================================================================================
# Shared helpers (also used by zfp.vision.blanks and zfp.vision.raster_shapes)
# ======================================================================================


def _detection(config: Any) -> DetectionConfig:
    """Coerce ``config`` -- a :class:`ZfpConfig`, a :class:`DetectionConfig` or ``None``
    -- into a :class:`DetectionConfig`."""
    if isinstance(config, DetectionConfig):
        return config
    inner = getattr(config, "detection", None)
    if isinstance(inner, DetectionConfig):
        return inner
    return DetectionConfig()


def _as_rect(obj: Any) -> Optional[Rect]:
    """Return the normalized rectangle of a ``Rect`` or of anything carrying ``.rect``."""
    if obj is None:
        return None
    if isinstance(obj, Rect):
        return obj.normalized()
    inner = getattr(obj, "rect", None)
    if isinstance(inner, Rect):
        return inner.normalized()
    return None


def reading_key(rect: Rect) -> Tuple[float, float, float, float]:
    """Sort key giving page reading order: top band first, then left to right."""
    return (
        -round(rect.y1, ROUND_DIGITS),
        round(rect.x0, ROUND_DIGITS),
        -round(rect.y0, ROUND_DIGITS),
        round(rect.x1, ROUND_DIGITS),
    )


def dedupe_rects(rects: Sequence[Rect], iou_threshold: float = _BOX_DEDUP_IOU) -> List[Rect]:
    """Drop near-duplicate rectangles, keeping the first in reading order."""
    ordered = sorted((r.rounded(ROUND_DIGITS) for r in rects), key=reading_key)
    kept: List[Rect] = []
    for rect in ordered:
        duplicate = False
        for other in kept:
            if other.iou(rect) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(rect)
    return kept


def _sample_down(items: List[Any], limit: int, what: str) -> List[Any]:
    """Evenly thin ``items`` down to ``limit`` entries, logging that we did."""
    total = len(items)
    if total <= limit:
        return items
    _log.warning("vision: %d %s exceeds the cap of %d, sampling down", total, what, limit)
    step = total / float(limit)
    return [items[min(total - 1, int(index * step))] for index in range(limit)]


def _rule_endpoints(rect: Rect) -> List[Point]:
    """The two endpoints of the centre line of a rule-shaped rectangle."""
    if rect.width >= rect.height:
        y = round((rect.y0 + rect.y1) / 2.0, ROUND_DIGITS)
        return [Point(round(rect.x0, ROUND_DIGITS), y), Point(round(rect.x1, ROUND_DIGITS), y)]
    x = round((rect.x0 + rect.x1) / 2.0, ROUND_DIGITS)
    return [Point(x, round(rect.y0, ROUND_DIGITS)), Point(x, round(rect.y1, ROUND_DIGITS))]


def _painted(prim: VectorPrimitive) -> bool:
    """True when the primitive puts ink on the page."""
    return bool(getattr(prim, "filled", False) or getattr(prim, "stroked", True))


def _thickness(thin_extent: float, prim: VectorPrimitive) -> float:
    """Visual thickness of a rule: its bounding box, or the pen width for a flat line."""
    if thin_extent <= EPS:
        width = float(getattr(prim, "stroke_width", 0.0) or 0.0)
        return max(0.0, width)
    return thin_extent


def _explicit_line(prim: VectorPrimitive, axis: str, tol: float) -> bool:
    """True when ``prim`` is a drawn line whose endpoints share the axis coordinate."""
    if str(getattr(prim, "kind", "")).lower() != "line":
        return False
    points = list(getattr(prim, "points", None) or [])
    if len(points) < 2:
        return False
    values = [p.y for p in points] if axis == "h" else [p.x for p in points]
    return (max(values) - min(values)) <= tol + EPS


# ======================================================================================
# Normalization
# ======================================================================================


def normalize_primitives(
    prims: Sequence[VectorPrimitive],
    config: Any = None,
    geometry: Optional[PageGeometry] = None,
) -> List[VectorPrimitive]:
    """Clean up raw content-stream primitives for the detectors downstream.

    Four things happen, in order:

    1. degenerate primitives (smaller than :data:`_MIN_EXTENT_PT` on *both* axes, or
       painted with neither fill nor stroke) are dropped;
    2. rectangles are clamped to the page when a ``geometry`` is supplied, so a path
       that bleeds off the media box cannot invent a field in the margin;
    3. every coordinate is rounded to :data:`ROUND_DIGITS` decimals;
    4. a ``"rect"`` thinner than ``max_line_thickness_pt`` on one axis is re-labelled as
       the ``"line"`` it visually is, with endpoints on its centre line.  This is how a
       ``re``-drawn underline becomes a rule.

    Args:
        prims: Raw primitives, typically ``ContentResult.primitives``.
        config: A :class:`ZfpConfig` or :class:`DetectionConfig` (``None`` = defaults).
        geometry: Optional page geometry used to clamp to the crop box.

    Returns:
        A new list of new :class:`VectorPrimitive` objects, in input order.  The inputs
        are never mutated.
    """
    det = _detection(config)
    max_thick = det.max_line_thickness_pt
    out: List[VectorPrimitive] = []
    for prim in prims or []:
        if prim is None:
            continue
        rect = _as_rect(prim)
        if rect is None:
            continue
        if not _painted(prim):
            continue
        if geometry is not None:
            rect = geometry.clamp(rect)
        if rect.width < _MIN_EXTENT_PT and rect.height < _MIN_EXTENT_PT:
            continue
        rect = rect.rounded(ROUND_DIGITS)
        kind = str(getattr(prim, "kind", "path") or "path").lower()
        points = [
            Point(round(p.x, ROUND_DIGITS), round(p.y, ROUND_DIGITS))
            for p in (getattr(prim, "points", None) or [])
        ]
        if geometry is not None and points:
            crop = geometry.crop_box
            points = [
                Point(
                    round(min(max(p.x, crop.x0), crop.x1), ROUND_DIGITS),
                    round(min(max(p.y, crop.y0), crop.y1), ROUND_DIGITS),
                )
                for p in points
            ]
        if kind == "rect":
            thin = min(rect.width, rect.height)
            long_side = max(rect.width, rect.height)
            if thin <= max_thick + EPS and long_side > max_thick + EPS:
                kind = "line"
                points = _rule_endpoints(rect)
        out.append(
            VectorPrimitive(
                kind=kind,
                rect=rect,
                page=int(getattr(prim, "page", 0) or 0),
                stroke_width=float(getattr(prim, "stroke_width", 0.0) or 0.0),
                filled=bool(getattr(prim, "filled", False)),
                stroked=bool(getattr(prim, "stroked", True)),
                points=points,
            )
        )
    return out


# ======================================================================================
# Rules
# ======================================================================================


def _rule_geometry(
    prim: VectorPrimitive, det: DetectionConfig, axis: str
) -> Optional[Tuple[Rect, float, float]]:
    """Return ``(rect, length, minimum_length)`` when ``prim`` could be a rule.

    "Could be" means only: painted, not a circle, and thinner than
    ``max_line_thickness_pt`` across the axis.  The length threshold is *returned*
    rather than applied, because a fragment too short on its own may still be one dash
    of a dashed rule -- see :func:`_select_rules`.
    """
    rect = _as_rect(prim)
    if rect is None or not _painted(prim):
        return None
    if str(getattr(prim, "kind", "")).lower() == "circle":
        return None
    if axis == "h":
        length, thin = rect.width, rect.height
    else:
        length, thin = rect.height, rect.width
    thickness = _thickness(thin, prim)
    if thickness > det.max_line_thickness_pt + EPS:
        return None
    if length <= thickness or length < _FRAGMENT_MIN_PT:
        return None
    minimum = det.min_line_length_pt
    if _explicit_line(prim, axis, det.line_merge_tolerance_pt):
        minimum *= _LINE_LENGTH_RELAX
    return (rect, length, minimum)


def _cross_of(rect: Rect, axis: str) -> float:
    """The coordinate perpendicular to ``axis`` -- what collinearity is measured on."""
    return rect.center.y if axis == "h" else rect.center.x


def _start_of(rect: Rect, axis: str) -> float:
    """Where a rule begins along its own axis."""
    return rect.x0 if axis == "h" else rect.y0


def _end_of(rect: Rect, axis: str) -> float:
    """Where a rule ends along its own axis."""
    return rect.x1 if axis == "h" else rect.y1


def _group_collinear(rects: Sequence[Rect], axis: str, tol: float) -> List[List[int]]:
    """Group rectangle indices into collinear runs separated by at most ``2 * tol``.

    Two passes: cluster by the perpendicular coordinate (within ``tol``), then split each
    cluster wherever the gap along the axis is too wide to be one rule.
    """
    if not rects:
        return []
    gap_limit = 2.0 * tol
    order = sorted(
        range(len(rects)), key=lambda i: (_cross_of(rects[i], axis), _start_of(rects[i], axis))
    )
    groups: List[List[int]] = []

    def flush(cluster: List[int]) -> None:
        if not cluster:
            return
        cluster.sort(key=lambda i: (_start_of(rects[i], axis), _end_of(rects[i], axis)))
        run = [cluster[0]]
        reach = _end_of(rects[cluster[0]], axis)
        for index in cluster[1:]:
            if _start_of(rects[index], axis) <= reach + gap_limit + EPS:
                run.append(index)
                reach = max(reach, _end_of(rects[index], axis))
            else:
                groups.append(run)
                run = [index]
                reach = _end_of(rects[index], axis)
        groups.append(run)

    cluster: List[int] = []
    anchor = 0.0
    for index in order:
        cross = _cross_of(rects[index], axis)
        if cluster and (cross - anchor) <= tol + EPS:
            cluster.append(index)
            continue
        flush(cluster)
        cluster = [index]
        anchor = cross
    flush(cluster)
    return groups


def _select_rules(
    prims: Sequence[VectorPrimitive], det: DetectionConfig, axis: str
) -> List[VectorPrimitive]:
    """Select the primitives that read as rules along ``axis``, dashes included.

    A fragment qualifies either on its own length, or because it belongs to a collinear
    run whose combined extent reaches ``min_line_length_pt``.  Without that second test a
    dashed rule would be discarded here and :func:`merge_collinear` would never see it.
    """
    entries: List[Tuple[VectorPrimitive, Rect, float, float]] = []
    for prim in prims or []:
        if prim is None:
            continue
        geometry = _rule_geometry(prim, det, axis)
        if geometry is None:
            continue
        entries.append((prim, geometry[0], geometry[1], geometry[2]))
    if not entries:
        return []
    rects = [entry[1] for entry in entries]
    keep = [entry[2] >= entry[3] - EPS for entry in entries]
    if not all(keep):
        tol = max(det.line_merge_tolerance_pt, 0.0)
        for group in _group_collinear(rects, axis, tol):
            if len(group) < 2 or all(keep[i] for i in group):
                continue
            bounds = Rect.bounding([rects[i] for i in group])
            if bounds is None:
                continue
            extent = bounds.width if axis == "h" else bounds.height
            if extent >= det.min_line_length_pt - EPS:
                for index in group:
                    keep[index] = True
    return [entries[pos][0] for pos in range(len(entries)) if keep[pos]]


def horizontal_rules(prims: Sequence[VectorPrimitive], config: Any = None) -> List[VectorPrimitive]:
    """Select the primitives that read as horizontal rules.

    A primitive qualifies when its bounding box is at least ``min_line_length_pt`` wide
    and no thicker than ``max_line_thickness_pt``.  An explicit ``"line"`` primitive
    whose endpoints share a *y* (within ``line_merge_tolerance_pt``) qualifies at
    ``0.6 * min_line_length_pt``, because a drawn line is unambiguous evidence in a way
    that a thin rectangle is not.  A short fragment also qualifies when it is one dash of
    a collinear run that is long enough overall.

    The input order is preserved and the primitives are returned by reference.
    """
    return _select_rules(prims, _detection(config), "h")


def vertical_rules(prims: Sequence[VectorPrimitive], config: Any = None) -> List[VectorPrimitive]:
    """Select the primitives that read as vertical rules (see :func:`horizontal_rules`)."""
    return _select_rules(prims, _detection(config), "v")


def merge_collinear(
    rules: Sequence[VectorPrimitive], config: Any = None
) -> List[VectorPrimitive]:
    """Fuse collinear rule fragments into single rules.

    Rules are clustered by their perpendicular coordinate within
    ``line_merge_tolerance_pt``, sorted along their own axis, and merged whenever the
    gap between consecutive segments is under ``2 * line_merge_tolerance_pt``.  This is
    what turns a dashed or segmented underline -- which a content stream may emit as
    forty separate ``re`` operators -- into the one field it looks like on paper.

    Horizontal and vertical rules are clustered independently, so a mixed list is safe.
    The result is sorted into reading order: horizontals first, then verticals.
    """
    det = _detection(config)
    tol = max(det.line_merge_tolerance_pt, 0.0)
    horizontals: List[VectorPrimitive] = []
    verticals: List[VectorPrimitive] = []
    for prim in rules or []:
        rect = _as_rect(prim)
        if rect is None:
            continue
        (horizontals if rect.width >= rect.height else verticals).append(prim)
    out: List[VectorPrimitive] = []
    out.extend(_merge_axis(horizontals, "h", tol))
    out.extend(_merge_axis(verticals, "v", tol))
    return out


def _merge_axis(prims: List[VectorPrimitive], axis: str, tol: float) -> List[VectorPrimitive]:
    """Merge one axis' rules into collinear runs, in reading order."""
    entries: List[Tuple[VectorPrimitive, Rect]] = []
    for prim in prims:
        rect = _as_rect(prim)
        if rect is not None:
            entries.append((prim, rect))
    if not entries:
        return []
    rects = [entry[1] for entry in entries]
    out: List[VectorPrimitive] = []
    for group in _group_collinear(rects, axis, tol):
        bounds = Rect.bounding([rects[i] for i in group])
        if bounds is None:
            continue
        bounds = bounds.rounded(ROUND_DIGITS)
        members = [entries[i][0] for i in group]
        out.append(
            VectorPrimitive(
                kind="line",
                rect=bounds,
                page=int(getattr(members[0], "page", 0) or 0),
                stroke_width=max(float(getattr(p, "stroke_width", 0.0) or 0.0) for p in members),
                filled=any(bool(getattr(p, "filled", False)) for p in members),
                stroked=any(bool(getattr(p, "stroked", True)) for p in members),
                points=_rule_endpoints(bounds),
            )
        )
    out.sort(key=lambda p: reading_key(p.rect))
    return out


# ======================================================================================
# Boxes and circles
# ======================================================================================


def boxes_from_rules(
    h_rects: Sequence[Any],
    v_rects: Sequence[Any],
    tolerance: float,
    min_dim: float = _BOX_MIN_DIM_PT,
    thickness: float = 0.0,
    limit: int = _MAX_BOXES,
) -> List[Rect]:
    """Rebuild rectangles from two horizontal and two vertical rules that *meet*.

    A box drawn as four strokes has four corners: the two horizontals share their x
    extent, the two verticals share their y extent, and each vertical's endpoints land
    on the horizontals (all within ``tolerance``, widened by half of ``thickness`` so a
    2pt border still closes).  Grid intersections, where rules cross and continue, are
    deliberately *not* boxes -- see :func:`detect_table_cells` for those.

    The function is unit-agnostic: :func:`detect_boxes` calls it in points and
    :mod:`zfp.vision.raster_shapes` calls it in pixels.
    """
    tol = max(float(tolerance), EPS) + 0.5 * max(float(thickness), 0.0)
    hs = [r for r in (_as_rect(x) for x in (h_rects or [])) if r is not None]
    vs = [r for r in (_as_rect(x) for x in (v_rects or [])) if r is not None]
    if not hs or not vs:
        return []
    hs.sort(key=reading_key)
    vs.sort(key=reading_key)
    hs = _sample_down(hs, _MAX_BOX_RULES, "horizontal rules for box detection")
    vs = _sample_down(vs, _MAX_BOX_RULES, "vertical rules for box detection")

    by_x = sorted(vs, key=lambda r: r.center.x)
    xs = [r.center.x for r in by_x]

    out: List[Rect] = []
    for i in range(len(hs)):
        a = hs[i]
        for j in range(i + 1, len(hs)):
            b = hs[j]
            if abs(a.x0 - b.x0) > tol or abs(a.x1 - b.x1) > tol:
                continue
            y_top = max(a.center.y, b.center.y)
            y_bot = min(a.center.y, b.center.y)
            if y_top - y_bot < min_dim:
                continue
            left_ref = (a.x0 + b.x0) / 2.0
            right_ref = (a.x1 + b.x1) / 2.0
            if right_ref - left_ref < min_dim:
                continue
            left = _find_vertical(by_x, xs, left_ref, y_bot, y_top, tol)
            if left is None:
                continue
            right = _find_vertical(by_x, xs, right_ref, y_bot, y_top, tol)
            if right is None:
                continue
            rect = a.union(b).union(left).union(right)
            if rect.width < min_dim or rect.height < min_dim:
                continue
            out.append(rect.rounded(ROUND_DIGITS))
            if len(out) >= limit:
                _log.warning("vision: box detection hit the cap of %d", limit)
                return dedupe_rects(out, _BOX_DEDUP_IOU)
    return dedupe_rects(out, _BOX_DEDUP_IOU)


def _find_vertical(
    by_x: List[Rect], xs: List[float], x: float, y_bot: float, y_top: float, tol: float
) -> Optional[Rect]:
    """Find a vertical rule at ``x`` whose endpoints meet ``y_bot`` and ``y_top``."""
    lo = bisect.bisect_left(xs, x - tol)
    hi = bisect.bisect_right(xs, x + tol)
    best: Optional[Rect] = None
    best_error = 0.0
    for index in range(lo, hi):
        rect = by_x[index]
        if abs(rect.y0 - y_bot) > tol or abs(rect.y1 - y_top) > tol:
            continue
        error = abs(rect.center.x - x) + abs(rect.y0 - y_bot) + abs(rect.y1 - y_top)
        if best is None or error < best_error:
            best, best_error = rect, error
    return best


def _unique_rects(rects: Sequence[Rect]) -> List[Rect]:
    """Exact-duplicate removal that keeps the first occurrence."""
    out: List[Rect] = []
    seen = set()
    for rect in rects:
        key = (
            round(rect.x0, ROUND_DIGITS),
            round(rect.y0, ROUND_DIGITS),
            round(rect.x1, ROUND_DIGITS),
            round(rect.y1, ROUND_DIGITS),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rect)
    return out


def detect_boxes(prims: Sequence[VectorPrimitive], config: Any = None) -> List[Rect]:
    """Find rectangular regions: painted ``"rect"`` primitives and four-rule boxes.

    Both sides must exceed 4pt -- below that a rectangle is a mark (a bullet, a glyph
    fragment), not a region anything can be typed into.  Results are deduplicated at an
    IoU of 0.8, so a box drawn *and* stroked as four rules is reported once.
    """
    det = _detection(config)
    found: List[Rect] = []
    for prim in prims or []:
        if prim is None or not _painted(prim):
            continue
        if str(getattr(prim, "kind", "")).lower() != "rect":
            continue
        rect = _as_rect(prim)
        if rect is None:
            continue
        if rect.width > _BOX_MIN_DIM_PT and rect.height > _BOX_MIN_DIM_PT:
            found.append(rect.rounded(ROUND_DIGITS))
    raw_h = horizontal_rules(prims, det)
    raw_v = vertical_rules(prims, det)
    h_rects = _unique_rects([p.rect for p in raw_h] + [p.rect for p in merge_collinear(raw_h, det)])
    v_rects = _unique_rects([p.rect for p in raw_v] + [p.rect for p in merge_collinear(raw_v, det)])
    found.extend(
        boxes_from_rules(
            h_rects,
            v_rects,
            det.line_merge_tolerance_pt,
            _BOX_MIN_DIM_PT,
            det.max_line_thickness_pt,
        )
    )
    return dedupe_rects(found, _BOX_DEDUP_IOU)


def _looks_like_curve(prim: VectorPrimitive, rect: Rect) -> bool:
    """True when a ``"path"`` primitive traces a closed curve rather than a rectangle.

    A four-Bezier circle flattens to points that leave the bounding box border; an
    axis-aligned rectangle never does.
    """
    points = list(getattr(prim, "points", None) or [])
    if len(points) < 4:
        return False
    band_x = max(rect.width * 0.08, 0.25)
    band_y = max(rect.height * 0.08, 0.25)
    for point in points:
        inside_x = (point.x - rect.x0) > band_x and (rect.x1 - point.x) > band_x
        inside_y = (point.y - rect.y0) > band_y and (rect.y1 - point.y) > band_y
        if inside_x and inside_y:
            return True
    return False


def detect_circles(prims: Sequence[VectorPrimitive], config: Any = None) -> List[Rect]:
    """Find circular marks: ``"circle"`` primitives plus near-square Bezier paths.

    :mod:`zfp.native.content` already labels a four-Bezier near-square subpath as a
    circle; the path fallback here catches producers that do not, which is how radio
    buttons drawn as raw curves are recovered.
    """
    _detection(config)
    found: List[Rect] = []
    for prim in prims or []:
        if prim is None or not _painted(prim):
            continue
        rect = _as_rect(prim)
        if rect is None or rect.width <= 0.0 or rect.height <= 0.0:
            continue
        kind = str(getattr(prim, "kind", "")).lower()
        if kind == "circle":
            found.append(rect.rounded(ROUND_DIGITS))
            continue
        if kind != "path":
            continue
        if min(rect.width, rect.height) < 2.0:
            continue
        ratio = rect.width / rect.height
        if abs(ratio - 1.0) > 0.25:
            continue
        if _looks_like_curve(prim, rect):
            found.append(rect.rounded(ROUND_DIGITS))
    return dedupe_rects(found, _BOX_DEDUP_IOU)


def _is_checkbox_sized(rect: Rect, det: DetectionConfig) -> bool:
    """True when a rectangle has the size and squareness of a checkbox."""
    width, height = rect.width, rect.height
    if width < det.checkbox_min_pt - EPS or width > det.checkbox_max_pt + EPS:
        return False
    if height < det.checkbox_min_pt - EPS or height > det.checkbox_max_pt + EPS:
        return False
    if height <= 0.0:
        return False
    return abs(width / height - 1.0) <= det.checkbox_aspect_tolerance + EPS


def _checkbox_glyph_text(span: TextSpan) -> bool:
    """True when a span's text *is* a checkbox glyph."""
    raw = (getattr(span, "text", "") or "").strip()
    if not raw:
        return False
    if raw in GLYPH_CHECKBOXES:
        return True
    collapsed = " ".join(raw.split())
    if collapsed in GLYPH_CHECKBOXES:
        return True
    squeezed = "".join(raw.split())
    if squeezed in GLYPH_CHECKBOXES:
        return True
    font = str(getattr(span, "font_name", "") or "").lower()
    if ("zapf" in font or "dingbat" in font) and len(raw) == 1 and raw in ZAPF_BOX_CHARS:
        return True
    return False


def detect_checkbox_glyphs(
    prims: Sequence[VectorPrimitive],
    spans: Sequence[TextSpan],
    config: Any = None,
) -> List[Rect]:
    """Find checkbox-shaped marks, drawn *or* typed.

    Two independent sources:

    * geometry -- boxes and circles whose sides are both inside
      ``[checkbox_min_pt, checkbox_max_pt]`` and whose aspect ratio is within
      ``checkbox_aspect_tolerance`` of square;
    * typography -- a text span that is exactly a ballot/box character
      (:data:`GLYPH_CHECKBOXES`) or a ZapfDingbats box glyph.  A run of underscores is
      *not* one: that is an underline field, and a different detector owns it.
    """
    det = _detection(config)
    found: List[Rect] = []
    for rect in detect_boxes(prims, det):
        if _is_checkbox_sized(rect, det):
            found.append(rect)
    for rect in detect_circles(prims, det):
        if _is_checkbox_sized(rect, det):
            found.append(rect)
    sanity = det.checkbox_max_pt * 4.0
    for span in spans or []:
        if span is None or not _checkbox_glyph_text(span):
            continue
        glyph = _as_rect(span)
        if glyph is None or glyph.width <= 0.0 or glyph.height <= 0.0:
            continue
        if glyph.width > sanity or glyph.height > sanity:
            continue
        found.append(glyph.rounded(ROUND_DIGITS))
    return dedupe_rects(found, _BOX_DEDUP_IOU)


# ======================================================================================
# Table lattice
# ======================================================================================


class _Level:
    """One lattice level: a coordinate plus the merged segments drawn along it."""

    __slots__ = ("coord", "segments")

    def __init__(self, coord: float, segments: List[Tuple[float, float]]) -> None:
        self.coord = coord
        self.segments = segments

    def covers(self, lo: float, hi: float, tol: float) -> bool:
        """True when one merged segment spans ``[lo, hi]`` within ``tol``."""
        for start, end in self.segments:
            if start <= lo + tol and end >= hi - tol:
                return True
        return False

    def reaches(self, coord: float, tol: float) -> bool:
        """True when one merged segment contains ``coord`` within ``tol``.

        Weaker than :meth:`covers`: it asks only whether the level is *drawn* at that
        coordinate at all, which is what decides whether it could take part in a cell
        there.
        """
        for start, end in self.segments:
            if start - tol <= coord <= end + tol:
                return True
        return False


def _build_levels(rects: List[Rect], axis: str, tol: float) -> List[_Level]:
    """Cluster rules into levels; each level's segments are merged with ``tol`` gaps."""
    if not rects:
        return []
    entries = []
    for rect in rects:
        cross = rect.center.y if axis == "h" else rect.center.x
        span = (rect.x0, rect.x1) if axis == "h" else (rect.y0, rect.y1)
        entries.append((cross, span))
    entries.sort(key=lambda e: (e[0], e[1][0]))

    levels: List[_Level] = []
    bucket: List[Tuple[float, Tuple[float, float]]] = []
    anchor = 0.0
    for entry in entries:
        if bucket and (entry[0] - anchor) <= tol + EPS:
            bucket.append(entry)
            continue
        if bucket:
            levels.append(_finish_level(bucket, tol))
        bucket = [entry]
        anchor = entry[0]
    if bucket:
        levels.append(_finish_level(bucket, tol))
    return levels


def _finish_level(bucket: List[Tuple[float, Tuple[float, float]]], tol: float) -> _Level:
    """Turn one cluster into a level with merged segments."""
    coord = sum(entry[0] for entry in bucket) / float(len(bucket))
    spans = sorted(entry[1] for entry in bucket)
    merged: List[List[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + tol + EPS:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return _Level(coord, [(s, e) for s, e in merged])


def detect_table_cells(
    h_rules: Sequence[Any], v_rules: Sequence[Any], config: Any = None
) -> List[Rect]:
    """Rebuild the *minimal* cells of a ruled table from its rules.

    The lattice is the set of horizontal levels (top to bottom) and vertical levels
    (left to right).  For every level pair ``(top, left)`` the smallest right edge -- and
    for that, the smallest bottom edge -- closing a rectangle on all four sides wins.
    Emitting only the minimal bounded rectangle is what makes merged cells behave: a cell
    spanning two columns is emitted once, at its true width, because the column rule
    between its ends does not reach down to its bottom.

    Work is capped: more than :data:`_MAX_LATTICE_RULES` rules on an axis are sampled
    down, cells spanning more than :data:`_MAX_CELL_SPAN` *participating* levels are
    not considered, the search looks at most :data:`_MAX_CELL_SCAN` levels past an
    edge, and the whole thing abandons after :data:`_CELL_WORK_BUDGET` checks (all
    logged).  A level that is not drawn at the cell at all -- a comb's separators
    ruled below the table, an underline elsewhere on the page -- is skipped without
    counting against the span, so unrelated geometry between two real table rules
    cannot hide the cell they bound.
    """
    det = _detection(config)
    tol = max(det.line_merge_tolerance_pt, EPS)
    hs = [r for r in (_as_rect(x) for x in (h_rules or [])) if r is not None]
    vs = [r for r in (_as_rect(x) for x in (v_rules or [])) if r is not None]
    if len(hs) < 2 or len(vs) < 2:
        return []
    hs.sort(key=reading_key)
    vs.sort(key=reading_key)
    hs = _sample_down(hs, _MAX_LATTICE_RULES, "horizontal rules for the table lattice")
    vs = _sample_down(vs, _MAX_LATTICE_RULES, "vertical rules for the table lattice")

    rows = _build_levels(hs, "h", tol)
    cols = _build_levels(vs, "v", tol)
    rows.sort(key=lambda lv: -lv.coord)  # top of the page first
    cols.sort(key=lambda lv: lv.coord)  # left first
    if len(rows) < 2 or len(cols) < 2:
        return []

    min_dim = max(2.0 * tol, 1.0)
    cells: List[Rect] = []
    budget = _CELL_WORK_BUDGET
    for i in range(len(rows) - 1):
        top = rows[i]
        for a in range(len(cols) - 1):
            left = cols[a]
            x_left = left.coord
            b_limit = min(len(cols), a + 1 + _MAX_CELL_SCAN)
            closed = False
            columns_left = _MAX_CELL_SPAN
            for b in range(a + 1, b_limit):
                right = cols[b]
                x_right = right.coord
                if x_right - x_left < min_dim:
                    continue
                if not right.reaches(top.coord, tol):
                    # A column rule that is not even drawn at this row cannot close a
                    # cell in it, and it is not one of the cell's columns either: a comb
                    # field's separators, ruled further down the same page, are exactly
                    # this and there can easily be a dozen of them between two real
                    # table columns.  Skipping them without spending the span budget is
                    # what keeps the column after them reachable.
                    continue
                columns_left -= 1
                if columns_left < 0:
                    break
                budget -= 1
                if budget <= 0:
                    _log.warning("vision: table cell search exhausted its work budget")
                    return _finish_cells(cells)
                if not top.covers(x_left, x_right, tol):
                    break  # the top edge is too short; a wider cell cannot help
                j_limit = min(len(rows), i + 1 + _MAX_CELL_SCAN)
                rows_left = _MAX_CELL_SPAN
                for j in range(i + 1, j_limit):
                    bottom = rows[j]
                    y_top = top.coord
                    y_bot = bottom.coord
                    if y_top - y_bot < min_dim:
                        continue
                    if not bottom.reaches(x_left, tol):
                        continue  # not drawn at this column; see the note above
                    rows_left -= 1
                    if rows_left < 0:
                        break
                    budget -= 1
                    if budget <= 0:
                        _log.warning("vision: table cell search exhausted its work budget")
                        return _finish_cells(cells)
                    if not left.covers(y_bot, y_top, tol):
                        break  # the left edge stops short; a taller cell cannot help
                    if not right.covers(y_bot, y_top, tol):
                        continue
                    if not bottom.covers(x_left, x_right, tol):
                        continue
                    cells.append(Rect(x_left, y_bot, x_right, y_top).rounded(ROUND_DIGITS))
                    closed = True
                    break
                if closed:
                    break
            if len(cells) >= _MAX_CELLS:
                _log.warning("vision: table cell detection hit the cap of %d", _MAX_CELLS)
                return _finish_cells(cells)
    return _finish_cells(cells)


def _finish_cells(cells: List[Rect]) -> List[Rect]:
    """Deduplicate, drop non-minimal cells, and sort into reading order."""
    unique: List[Rect] = []
    seen = set()
    for rect in cells:
        key = (rect.x0, rect.y0, rect.x1, rect.y1)
        if key in seen:
            continue
        seen.add(key)
        unique.append(rect)
    unique.sort(key=reading_key)
    if len(unique) > _MINIMALITY_LIMIT:
        return unique
    minimal: List[Rect] = []
    for rect in unique:
        contains_other = False
        for other in unique:
            if other is rect:
                continue
            if other.area + EPS >= rect.area:
                continue
            if (
                other.x0 >= rect.x0 - EPS
                and other.y0 >= rect.y0 - EPS
                and other.x1 <= rect.x1 + EPS
                and other.y1 <= rect.y1 + EPS
            ):
                contains_other = True
                break
        if not contains_other:
            minimal.append(rect)
    return minimal


# ======================================================================================
# Comb cells
# ======================================================================================


def detect_comb_cells(boxes: Sequence[Any], config: Any = None) -> List[List[Rect]]:
    """Group boxes into comb runs: equal cells, equally spaced, on one row.

    This is the "one character per box" archetype -- account numbers, dates, postcodes.
    A run needs at least three cells whose widths and inter-cell gaps agree within
    ``comb_cell_tolerance_pt``; runs are maximal and never overlap.

    Returns:
        A list of runs, each a left-to-right list of cell rectangles.  Runs are ordered
        top of page first, then left to right.
    """
    det = _detection(config)
    tol = max(det.comb_cell_tolerance_pt, 0.0)
    rects = [r for r in (_as_rect(x) for x in (boxes or [])) if r is not None]
    if len(rects) < 3:
        return []
    rects.sort(key=lambda r: (-r.center.y, r.x0, r.x1))

    rows: List[List[Rect]] = []
    for rect in rects:
        placed = False
        for row in rows:
            first = row[0]
            if abs(first.center.y - rect.center.y) <= tol + EPS and abs(
                first.height - rect.height
            ) <= tol + EPS:
                row.append(rect)
                placed = True
                break
        if not placed:
            rows.append([rect])

    runs: List[List[Rect]] = []
    for row in rows:
        row.sort(key=lambda r: (r.x0, r.x1))
        runs.extend(_comb_runs(row, tol))
    runs.sort(key=lambda run: reading_key(run[0]))
    return runs


def _comb_runs(row: List[Rect], tol: float) -> List[List[Rect]]:
    """Maximal runs of >= 3 equal-width, equally-spaced cells in one row."""
    out: List[List[Rect]] = []
    count = len(row)
    if count < 3:
        return out
    start = 0
    while start + 2 < count:
        width = row[start].width
        if abs(row[start + 1].width - width) > tol + EPS:
            start += 1
            continue
        gap = row[start + 1].x0 - row[start].x1
        end = start + 1
        while end + 1 < count:
            nxt = row[end + 1]
            if abs(nxt.width - width) > tol + EPS:
                break
            if abs((nxt.x0 - row[end].x1) - gap) > tol + EPS:
                break
            end += 1
        if end - start + 1 >= 3:
            out.append([r.rounded(ROUND_DIGITS) for r in row[start : end + 1]])
            start = end + 1
        else:
            start += 1
    return out


# ======================================================================================
# Rules that already carry text
# ======================================================================================


def rule_spans(
    rules: Sequence[Any], spans: Sequence[TextSpan], config: Any = None
) -> Dict[int, List[TextSpan]]:
    """Map each rule to the text spans sitting *on* it.

    A rule with text immediately above it (within ``underline_gap_pt``, overlapping at
    least half of the span's width) is an underlined heading or an already-filled value,
    not an empty field.  The candidate detectors need that distinction: it is the
    difference between "Employment History" with a rule under it and a blank to write on.

    Returns:
        ``{index in rules: [spans, left to right]}``, sparse -- rules with no text on
        them are absent, so callers should use ``.get(index, [])``.
    """
    det = _detection(config)
    gap_limit = max(det.underline_gap_pt, 0.0)
    slack = max(det.line_merge_tolerance_pt, 0.0)
    out: Dict[int, List[TextSpan]] = {}
    prepared = []
    for span in spans or []:
        rect = _as_rect(span)
        if rect is None or span is None:
            continue
        if getattr(span, "is_blank", None) is not None and span.is_blank():
            continue
        prepared.append((span, rect))
    if not prepared:
        return out
    for index, rule in enumerate(rules or []):
        rect = _as_rect(rule)
        if rect is None:
            continue
        hits: List[Tuple[float, TextSpan]] = []
        for span, span_rect in prepared:
            gap = span_rect.y0 - rect.y1
            if gap < -slack - EPS or gap > gap_limit + EPS:
                continue
            overlap = rect.horizontal_overlap(span_rect)
            if span_rect.width > 0.0 and overlap < 0.5 * span_rect.width:
                continue
            if span_rect.width <= 0.0 and overlap <= 0.0:
                continue
            hits.append((span_rect.x0, span))
        if hits:
            hits.sort(key=lambda item: item[0])
            out[index] = [span for _, span in hits]
    return out


# ======================================================================================
# Contract re-export
# ======================================================================================


def blank_regions(
    spans: Sequence[TextSpan],
    prims: Sequence[VectorPrimitive],
    geometry: Optional[PageGeometry],
    config: Any = None,
    images: Optional[Sequence[Rect]] = None,
) -> List[Rect]:
    """Blank-region analysis; see :func:`zfp.vision.blanks.blank_regions`.

    The contract lists this name in ``zfp.vision.primitives``; the implementation lives
    in :mod:`zfp.vision.blanks` and is imported lazily so the two modules can share
    helpers without a cycle.
    """
    from .blanks import blank_regions as _impl

    return _impl(spans, prims, geometry, config, images=images)
