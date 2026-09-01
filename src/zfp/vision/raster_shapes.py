"""Shape detection on a rasterized page -- the scan path.

A scanned form has no vector primitives at all: its rules, boxes and radio buttons are
runs of dark pixels.  This module recovers the same geometry :mod:`zfp.vision.primitives`
recovers from a content stream, from the gray bytes of a
:class:`~zfp.raster.render.RenderedPage`.

Two backends, one contract:

* **numpy + OpenCV** when both are installed: morphological opening with long horizontal
  and vertical kernels isolates the rules, connected components find the boxes, and
  ``HoughCircles`` finds the radio buttons.
* **pure CPython otherwise** -- the case that actually runs in a default install.  Each
  row is run-length scanned for dark runs longer than ``min_line_length_pt * scale`` and
  thinner than ``max_line_thickness_pt * scale``; the same is done column-wise; runs are
  merged across adjacent rows into bands; boxes come from the rule lattice with the very
  same corner logic the native path uses; circles are connected components whose bounding
  box is near-square and whose fill ratio says "ring", not "blob".

  The scanning is done with :meth:`bytes.translate` and :meth:`bytes.find`, so the hot
  loop runs at C speed over a binarized copy of the page and the Python level only ever
  sees runs.

Pixel space never escapes: every rectangle is converted through
:meth:`~zfp.core.geometry.PageGeometry.pixel_rect_to_user` exactly once, at the end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.config import DetectionConfig
from ..core.geometry import EPS, PageGeometry, Rect
from ..core.logging import get_logger
from ..core.optional import optional_import
from ..core.types import VectorPrimitive
from .blanks import (
    CELL_PT,
    MAX_BLANKS,
    OccupancyGrid,
    maximal_empty_cells,
    occupancy_grid,
    suppress_redundant,
)
from .primitives import (
    ROUND_DIGITS,
    _detection,
    boxes_from_rules,
    dedupe_rects,
    reading_key,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..raster.render import RenderedPage

__all__ = ["RasterShapes", "detect_shapes_from_image", "binarize_ink"]

_log = get_logger(__name__)

#: Below this gray-level spread a page carries no ink worth analysing.
_MIN_CONTRAST = 16

#: Histogram subsampling stride for very large rasters (deterministic).
_HIST_STRIDE = 7
_HIST_LIMIT = 500_000

#: Two runs in adjacent rows join a band when they overlap by this much of the shorter.
_BAND_OVERLAP = 0.6

#: A ring, not a blob: acceptable ink/bbox ratio for a raster circle.
_CIRCLE_FILL_MIN = 0.25
_CIRCLE_FILL_MAX = 0.85

#: Work caps.
_MAX_RUNS = 400_000
_MAX_RULES_PER_AXIS = 400


# ======================================================================================
# Result
# ======================================================================================


@dataclass
class RasterShapes:
    """Everything the raster path found, **in PDF user space**.

    All five members are user-space rectangles, converted once from pixels through the
    page geometry.  ``h_rules`` and ``v_rules`` keep their (thin) thickness, so a caller
    can tell a hairline from a 2pt border.
    """

    h_rules: List[Rect] = field(default_factory=list)
    v_rules: List[Rect] = field(default_factory=list)
    boxes: List[Rect] = field(default_factory=list)
    circles: List[Rect] = field(default_factory=list)
    blanks: List[Rect] = field(default_factory=list)
    backend: str = "none"

    def is_empty(self) -> bool:
        """True when nothing at all was detected."""
        return not (self.h_rules or self.v_rules or self.boxes or self.circles or self.blanks)

    def all_rects(self) -> List[Rect]:
        """Every rectangle found, in one list (rules, boxes, circles, blanks)."""
        return list(self.h_rules) + list(self.v_rules) + list(self.boxes) + list(
            self.circles
        ) + list(self.blanks)

    def as_primitives(self, page: int = 0) -> List[VectorPrimitive]:
        """Express the rules and boxes as :class:`VectorPrimitive` objects.

        This lets the fusion stage treat a scanned page exactly like a native one.
        """
        out: List[VectorPrimitive] = []
        for rect in self.h_rules + self.v_rules:
            out.append(
                VectorPrimitive(
                    kind="line",
                    rect=rect,
                    page=page,
                    stroke_width=min(rect.width, rect.height),
                    filled=True,
                    stroked=True,
                )
            )
        for rect in self.boxes:
            out.append(VectorPrimitive(kind="rect", rect=rect, page=page, stroked=True))
        for rect in self.circles:
            out.append(VectorPrimitive(kind="circle", rect=rect, page=page, stroked=True))
        return out

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "h_rules": [r.as_list() for r in self.h_rules],
            "v_rules": [r.as_list() for r in self.v_rules],
            "boxes": [r.as_list() for r in self.boxes],
            "circles": [r.as_list() for r in self.circles],
            "blanks": [r.as_list() for r in self.blanks],
            "backend": self.backend,
        }


# ======================================================================================
# Binarization
# ======================================================================================


def _otsu(gray: bytes) -> int:
    """Otsu's threshold over a (possibly subsampled) gray buffer."""
    sample = gray[::_HIST_STRIDE] if len(gray) > _HIST_LIMIT else gray
    histogram = [sample.count(level) for level in range(256)]
    total = len(sample)
    if total <= 0:
        return 128
    sum_all = 0.0
    for level in range(256):
        sum_all += level * histogram[level]
    sum_b = 0.0
    weight_b = 0
    best_value = -1.0
    best_threshold = 128
    for level in range(256):
        weight_b += histogram[level]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += level * histogram[level]
        mean_b = sum_b / weight_b
        mean_f = (sum_all - sum_b) / weight_f
        between = weight_b * weight_f * (mean_b - mean_f) * (mean_b - mean_f)
        if between > best_value:
            best_value = between
            best_threshold = level
    return best_threshold


def binarize_ink(gray: bytes) -> Tuple[bytes, int]:
    """Return ``(mask, threshold)`` where ``mask`` holds 1 for ink and 0 for paper.

    The mask is produced with :meth:`bytes.translate`, which is a single C-level pass.
    """
    if not gray:
        return (b"", 128)
    low, high = min(gray), max(gray)
    if high - low < _MIN_CONTRAST:
        return (b"\x00" * len(gray), 0)
    threshold = max(low, min(_otsu(gray), high - 1))
    table = bytes(1 if level <= threshold else 0 for level in range(256))
    return (gray.translate(table), threshold)


# ======================================================================================
# Run-length machinery (pure python path)
# ======================================================================================


def _raw_runs(buf: bytes) -> List[Tuple[int, int]]:
    """All maximal runs of ink in ``buf`` as half-open ``(start, end)`` pairs."""
    out: List[Tuple[int, int]] = []
    size = len(buf)
    pos = 0
    while pos < size:
        start = buf.find(1, pos)
        if start < 0:
            break
        end = buf.find(0, start + 1)
        if end < 0:
            end = size
        out.append((start, end))
        pos = end + 1
    return out


def _merge_runs(runs: Sequence[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    """Join runs separated by at most ``gap`` blank pixels (a dashed rule is one rule)."""
    merged: List[List[int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _bands(
    per_line: Sequence[List[Tuple[int, int]]], max_thickness: int
) -> List[Tuple[int, int, int, int]]:
    """Group per-line runs into bands: ``(line_start, line_end, start, end)`` inclusive.

    A run continues a band when it overlaps the band's current extent by at least
    :data:`_BAND_OVERLAP` of the shorter of the two -- which is how a two-pixel-thick
    rule scanned onto three rows stays one rule.
    """
    finished: List[Tuple[int, int, int, int]] = []
    active: List[List[int]] = []  # [line_start, line_end, start, end]
    for index, runs in enumerate(per_line):
        used = [False] * len(runs)
        carried: List[List[int]] = []
        for band in active:
            match = -1
            for position, (start, end) in enumerate(runs):
                if used[position]:
                    continue
                overlap = min(end, band[3]) - max(start, band[2])
                if overlap <= 0:
                    continue
                shorter = min(end - start, band[3] - band[2])
                if shorter <= 0 or overlap >= _BAND_OVERLAP * shorter:
                    match = position
                    break
            if match < 0:
                finished.append((band[0], band[1], band[2], band[3]))
                continue
            used[match] = True
            start, end = runs[match]
            band[1] = index
            band[2] = min(band[2], start)
            band[3] = max(band[3], end)
            carried.append(band)
        for position, (start, end) in enumerate(runs):
            if not used[position]:
                carried.append([index, index, start, end])
        active = carried
    for band in active:
        finished.append((band[0], band[1], band[2], band[3]))
    return [b for b in finished if (b[1] - b[0] + 1) <= max_thickness]


def _rules_pure(
    mask: bytes, width: int, height: int, min_len: int, max_thick: int, gap: int
) -> Tuple[List[Rect], List[Rect]]:
    """Row- and column-scan the ink mask for rules, in pixel space."""
    rows: List[List[Tuple[int, int]]] = []
    for y in range(height):
        runs = _merge_runs(_raw_runs(mask[y * width : (y + 1) * width]), gap)
        rows.append([(s, e) for s, e in runs if e - s >= min_len])
    h_rects = [
        Rect(float(b[2]), float(b[0]), float(b[3]), float(b[1] + 1))
        for b in _bands(rows, max_thick)
    ]

    columns: List[List[Tuple[int, int]]] = []
    for x in range(width):
        runs = _merge_runs(_raw_runs(mask[x::width]), gap)
        columns.append([(s, e) for s, e in runs if e - s >= min_len])
    v_rects = [
        Rect(float(b[0]), float(b[2]), float(b[1] + 1), float(b[3]))
        for b in _bands(columns, max_thick)
    ]
    if len(h_rects) > _MAX_RULES_PER_AXIS:
        _log.warning("vision: %d raster horizontal rules, truncating", len(h_rects))
        h_rects.sort(key=lambda r: -r.width)
        h_rects = h_rects[:_MAX_RULES_PER_AXIS]
    if len(v_rects) > _MAX_RULES_PER_AXIS:
        _log.warning("vision: %d raster vertical rules, truncating", len(v_rects))
        v_rects.sort(key=lambda r: -r.height)
        v_rects = v_rects[:_MAX_RULES_PER_AXIS]
    return (h_rects, v_rects)


class _UnionFind:
    """The smallest union-find that does the job, over run indices."""

    __slots__ = ("parent",)

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self.parent
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _circles_pure(
    mask: bytes, width: int, height: int, det: DetectionConfig, scale: float
) -> List[Rect]:
    """Connected components that look like a drawn ring rather than a filled blob."""
    runs: List[Tuple[int, int, int]] = []  # (row, start, end)
    per_row: List[Tuple[int, int]] = []  # (first index, count)
    for y in range(height):
        row_runs = _raw_runs(mask[y * width : (y + 1) * width])
        per_row.append((len(runs), len(row_runs)))
        for start, end in row_runs:
            runs.append((y, start, end))
        if len(runs) > _MAX_RUNS:
            _log.warning("vision: raster circle search abandoned, %d runs", len(runs))
            return []
    if not runs:
        return []

    union = _UnionFind(len(runs))
    for y in range(1, height):
        base_a, count_a = per_row[y - 1]
        base_b, count_b = per_row[y]
        i = j = 0
        while i < count_a and j < count_b:
            a = runs[base_a + i]
            b = runs[base_b + j]
            if a[1] < b[2] + 1 and b[1] < a[2] + 1:
                union.union(base_a + i, base_b + j)
            if a[2] <= b[2]:
                i += 1
            else:
                j += 1

    boxes: Dict[int, List[float]] = {}
    for index, (y, start, end) in enumerate(runs):
        root = union.find(index)
        stat = boxes.get(root)
        if stat is None:
            boxes[root] = [start, y, end, y + 1, end - start]
            continue
        stat[0] = min(stat[0], start)
        stat[1] = min(stat[1], y)
        stat[2] = max(stat[2], end)
        stat[3] = max(stat[3], y + 1)
        stat[4] += end - start

    min_side = max(3.0, det.checkbox_min_pt * scale * 0.5)
    max_side = det.checkbox_max_pt * scale * 2.0
    tolerance = max(det.checkbox_aspect_tolerance, 0.05)
    out: List[Rect] = []
    for stat in boxes.values():
        w = stat[2] - stat[0]
        h = stat[3] - stat[1]
        if w < min_side or h < min_side or w > max_side or h > max_side:
            continue
        if h <= 0 or abs(w / float(h) - 1.0) > tolerance:
            continue
        area = float(w * h)
        if area <= 0.0:
            continue
        ratio = stat[4] / area
        if ratio < _CIRCLE_FILL_MIN or ratio > _CIRCLE_FILL_MAX:
            continue
        out.append(Rect(float(stat[0]), float(stat[1]), float(stat[2]), float(stat[3])))
    return out


def _blanks_pure(
    mask: bytes, width: int, height: int, det: DetectionConfig, scale: float
) -> List[Rect]:
    """Maximal empty rectangles of the raster, anchored to nearby ink."""
    bounds = Rect(0.0, 0.0, float(width), float(height))
    cell_px = max(CELL_PT * scale, 1.0)
    grid = occupancy_grid([], bounds, cell_pt=cell_px)
    gap = max(1, int(round(grid.cell_w)))
    for y in range(height):
        for start, end in _merge_runs(_raw_runs(mask[y * width : (y + 1) * width]), gap):
            grid.mark(Rect(float(start), float(y), float(end), float(y + 1)))
    min_cols = max(1, int(math.ceil((det.blank_min_width_pt * scale - EPS) / grid.cell_w)))
    min_rows = max(1, int(math.ceil((det.blank_min_height_pt * scale - EPS) / grid.cell_h)))
    cells = maximal_empty_cells(grid, min_cols, min_rows)
    if not cells:
        return []
    left_cells = max(1, int(round(det.label_max_distance_pt * scale / grid.cell_w)))
    above_cells = max(1, int(round(4.0 * det.blank_min_height_pt * scale / grid.cell_h)))
    out: List[Rect] = []
    seen = set()
    for c0, r0, c1, r1 in cells:
        if not _ink_nearby(grid, c0, r0, c1, r1, left_cells, above_cells):
            continue
        rect = grid.cells_to_rect(c0, r0, c1, r1)
        key = (rect.x0, rect.y0, rect.x1, rect.y1)
        if key in seen:
            continue
        seen.add(key)
        out.append(rect)
    return out


def _ink_nearby(
    grid: OccupancyGrid, c0: int, r0: int, c1: int, r1: int, left: int, above: int
) -> bool:
    """True when ink sits just left of, or just above, a blank (pixel space: y down)."""
    data, cols = grid.data, grid.cols
    for col in range(max(0, c0 - left), c0):
        for row in range(r0, r1 + 1):
            if data[row * cols + col]:
                return True
    for row in range(max(0, r0 - above), r0):
        base = row * cols
        if any(data[base + c0 : base + c1 + 1]):
            return True
    return False


# ======================================================================================
# OpenCV path
# ======================================================================================


def _detect_with_cv2(
    gray: bytes,
    width: int,
    height: int,
    threshold: int,
    det: DetectionConfig,
    scale: float,
    np_mod: Any,
    cv_mod: Any,
) -> Optional[Tuple[List[Rect], List[Rect], List[Rect], List[Rect]]]:
    """Morphology/Hough implementation used when numpy and OpenCV are both installed.

    Returns pixel-space ``(h_rules, v_rules, boxes, circles)`` or ``None`` when anything
    at all goes wrong -- the caller then falls back to the pure-python path.
    """
    try:
        min_len = max(3, int(det.min_line_length_pt * scale))
        max_thick = max(1, int(round(det.max_line_thickness_pt * scale)))
        arr = np_mod.frombuffer(gray, dtype=np_mod.uint8).reshape((height, width))
        binary = np_mod.where(arr <= threshold, np_mod.uint8(255), np_mod.uint8(0))

        def _open(kernel_size):
            kernel = cv_mod.getStructuringElement(cv_mod.MORPH_RECT, kernel_size)
            return cv_mod.morphologyEx(binary, cv_mod.MORPH_OPEN, kernel)

        def _rects(image, horizontal):
            found = cv_mod.findContours(
                image, cv_mod.RETR_EXTERNAL, cv_mod.CHAIN_APPROX_SIMPLE
            )
            contours = found[0] if len(found) == 2 else found[1]
            out = []
            for contour in contours:
                x, y, w, h = cv_mod.boundingRect(contour)
                if horizontal and (w < min_len or h > max_thick):
                    continue
                if not horizontal and (h < min_len or w > max_thick):
                    continue
                out.append(Rect(float(x), float(y), float(x + w), float(y + h)))
            return out

        h_image = _open((min_len, 1))
        v_image = _open((1, min_len))
        h_rects = _rects(h_image, True)
        v_rects = _rects(v_image, False)

        boxes = boxes_from_rules(
            h_rects,
            v_rects,
            det.line_merge_tolerance_pt * scale,
            4.0 * scale,
            det.max_line_thickness_pt * scale,
        )
        lattice = cv_mod.bitwise_or(h_image, v_image)
        count, _labels, stats, _centroids = cv_mod.connectedComponentsWithStats(lattice, 8)
        min_dim = 4.0 * scale
        for index in range(1, int(count)):
            x, y, w, h, area = (int(v) for v in stats[index][:5])
            if w < min_dim or h < min_dim:
                continue
            if area >= 0.5 * float(w * h):
                continue  # solid blob, not a box outline
            boxes.append(Rect(float(x), float(y), float(x + w), float(y + h)))

        circles: List[Rect] = []
        min_radius = max(2, int(det.checkbox_min_pt * scale * 0.4))
        max_radius = max(min_radius + 1, int(det.checkbox_max_pt * scale))
        detected = cv_mod.HoughCircles(
            arr,
            cv_mod.HOUGH_GRADIENT,
            dp=1,
            minDist=float(max(4, min_radius * 2)),
            param1=100,
            param2=20,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if detected is not None:
            for circle in np_mod.reshape(detected, (-1, 3)):
                cx, cy, radius = float(circle[0]), float(circle[1]), float(circle[2])
                circles.append(Rect(cx - radius, cy - radius, cx + radius, cy + radius))
        return (h_rects, v_rects, dedupe_rects(boxes), circles)
    except Exception as exc:  # pragma: no cover - depends on absent optional backends
        _log.warning("vision: OpenCV shape detection failed (%s), using the pure path", exc)
        return None


# ======================================================================================
# Entry point
# ======================================================================================


def detect_shapes_from_image(
    page: Optional["RenderedPage"],
    geometry: Optional[PageGeometry] = None,
    config: Any = None,
    scale: Optional[float] = None,
) -> RasterShapes:
    """Detect rules, boxes, circles and blanks on a rasterized page.

    Args:
        page: A :class:`~zfp.raster.render.RenderedPage` (anything exposing ``width``,
            ``height``, ``gray`` and ``scale`` will do).  ``None`` yields empty results.
        geometry: The page's :class:`PageGeometry`, used for the single pixel-to-user
            conversion at the end.  ``None`` yields empty results.
        config: A :class:`ZfpConfig` or :class:`DetectionConfig`.
        scale: Pixels per point; defaults to ``page.scale``.  Passing the scale as the
            third positional argument (the older contract ordering) is also accepted.

    Returns:
        A :class:`RasterShapes` whose every rectangle is in PDF user space.  A page with
        no ink -- or no page at all -- gives empty lists, never an exception.
    """
    if isinstance(config, (int, float)) and not isinstance(config, bool):
        config, scale = scale, float(config)
    det = _detection(config)
    if page is None or geometry is None:
        return RasterShapes()
    width = int(getattr(page, "width", 0) or 0)
    height = int(getattr(page, "height", 0) or 0)
    gray = getattr(page, "gray", b"") or b""
    if width <= 0 or height <= 0 or len(gray) != width * height:
        return RasterShapes()
    if scale is None:
        scale = float(getattr(page, "scale", 1.0) or 1.0)
    scale = float(scale)
    if scale <= 0.0:
        return RasterShapes()

    mask, threshold = binarize_ink(gray)
    if not mask or mask.find(1) < 0:
        return RasterShapes()

    min_len = max(2, int(round(det.min_line_length_pt * scale)))
    max_thick = max(1, int(round(det.max_line_thickness_pt * scale)))
    gap = max(1, int(round(det.line_merge_tolerance_pt * scale)))

    backend = "pure"
    result = None
    numpy_module = optional_import("numpy")
    cv_module = optional_import("cv2")
    if numpy_module and cv_module:
        result = _detect_with_cv2(
            gray, width, height, threshold, det, scale, numpy_module.module, cv_module.module
        )
        if result is not None:
            backend = "opencv"
    if result is None:
        h_px, v_px = _rules_pure(mask, width, height, min_len, max_thick, gap)
        boxes_px = boxes_from_rules(
            h_px,
            v_px,
            det.line_merge_tolerance_pt * scale,
            4.0 * scale,
            float(max_thick),
        )
        circles_px = _circles_pure(mask, width, height, det, scale)
    else:
        h_px, v_px, boxes_px, circles_px = result
    blanks_px = _blanks_pure(mask, width, height, det, scale)

    # Orientation is re-decided in user space: on a page with /Rotate 90 a raster-
    # horizontal rule is a *vertical* rule once it lands back in PDF coordinates.
    rules = _to_user(h_px, geometry, scale) + _to_user(v_px, geometry, scale)
    shapes = RasterShapes(
        h_rules=[r for r in rules if r.width >= r.height],
        v_rules=[r for r in rules if r.width < r.height],
        boxes=_to_user(boxes_px, geometry, scale),
        circles=_to_user(circles_px, geometry, scale),
        blanks=_to_user(blanks_px, geometry, scale),
        backend=backend,
    )
    shapes.h_rules.sort(key=reading_key)
    shapes.v_rules.sort(key=reading_key)
    shapes.boxes = dedupe_rects(shapes.boxes)
    shapes.circles = dedupe_rects(shapes.circles)
    shapes.blanks = suppress_redundant(shapes.blanks, MAX_BLANKS)
    return shapes


def _to_user(rects: Iterable[Rect], geometry: PageGeometry, scale: float) -> List[Rect]:
    """Convert pixel rectangles to user space -- the one and only conversion."""
    out: List[Rect] = []
    for rect in rects:
        user = geometry.clamp(geometry.pixel_rect_to_user(rect, scale)).rounded(ROUND_DIGITS)
        if user.width <= 0.0 and user.height <= 0.0:
            continue
        out.append(user)
    return out
