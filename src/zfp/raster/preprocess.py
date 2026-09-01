"""Scan preprocessing: contrast, binarization, denoising, deskew and orientation.

Everything here is pure CPython operating on :attr:`RenderedPage.gray`.  No numpy, no
OpenCV, no Pillow -- a default ZFP install has none of them, and the OCR cascade still
has to be handed a clean bilevel page.

The pipeline :func:`preprocess` runs is orientation -> deskew -> contrast -> denoise ->
binarize, and every decision it takes is recorded in a :class:`PreprocessReport` so a
later stage (or a human) can see why a page came out the way it did.

Conventions
-----------
* Ink is dark.  A binarized page holds only 0 (ink) and 255 (paper).
* A positive skew angle means the content is rotated **counter-clockwise as displayed**;
  :func:`deskew` applies a rotation, so straightening is ``deskew(page, -angle)``.
"""

from __future__ import annotations

import math
import operator
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import accumulate
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..core.errors import ValidationError
from ..core.logging import get_logger
from .render import RenderedPage, _box_shrink

__all__ = [
    "PreprocessReport",
    "normalize_contrast",
    "binarize",
    "otsu_threshold",
    "denoise",
    "estimate_skew",
    "deskew",
    "rotate_quarter",
    "detect_orientation",
    "preprocess",
    "histogram",
    "ink_mask",
    "BINARIZE_METHODS",
]

_log = get_logger(__name__)

_INK = 0
_PAPER = 255

#: Accepted values of the ``method`` argument of :func:`binarize`.
BINARIZE_METHODS = ("sauvola", "otsu")

_SAUVOLA_WINDOW = 31
_SAUVOLA_K = 0.2
_SAUVOLA_R = 128.0
_MIN_COMPONENT_PIXELS = 4
#: Sauvola reports its mean local threshold from every Nth row (a diagnostic only).
_THRESHOLD_SAMPLE_ROWS = 32
_SKEW_LIMIT = 5.0
_SKEW_STEP = 0.25
_SKEW_MAX_POINTS = 40000

_SQUARES = tuple(value * value for value in range(256))


@dataclass
class PreprocessReport:
    """What :func:`preprocess` did, and why.

    Attributes:
        steps: The operations actually applied, in order.
        orientation: Quarter-turn correction applied, in degrees (0 or 90).
        skew_angle: Skew detected before correction, in degrees.
        threshold: The global threshold, or the mean local threshold for Sauvola.
        removed_components: Connected ink blobs discarded by :func:`denoise`.
        method: Binarization method used.
        ink_ratio: Fraction of the final page that is ink, in ``[0, 1]``.
        notes: Free-form remarks (skipped steps, degenerate pages).
    """

    steps: List[str] = field(default_factory=list)
    orientation: int = 0
    skew_angle: float = 0.0
    threshold: int = 0
    removed_components: int = 0
    method: str = "sauvola"
    ink_ratio: float = 0.0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain JSON-ready dictionary."""
        return {
            "steps": list(self.steps),
            "orientation": self.orientation,
            "skew_angle": round(self.skew_angle, 4),
            "threshold": self.threshold,
            "removed_components": self.removed_components,
            "method": self.method,
            "ink_ratio": round(self.ink_ratio, 6),
            "notes": list(self.notes),
        }


# ======================================================================================
# Small shared helpers
# ======================================================================================


def histogram(data: bytes) -> List[int]:
    """Return the 256-bin gray histogram of ``data``."""
    bins = [0] * 256
    for value, count in Counter(data).items():
        bins[value] = count
    return bins


def _replace_gray(
    page: RenderedPage, gray: Union[bytes, bytearray], width: int = -1, height: int = -1
) -> RenderedPage:
    """Return a copy of ``page`` carrying new pixels (and optionally a new size)."""
    return replace(
        page,
        gray=bytes(gray),
        width=page.width if width < 0 else width,
        height=page.height if height < 0 else height,
    )


def _is_binary(bins: Sequence[int]) -> bool:
    """True when the histogram only holds the two extreme levels."""
    return sum(bins) == bins[0] + bins[255]


def ink_mask(page: RenderedPage, threshold: Optional[int] = None) -> Tuple[bytes, int]:
    """Return ``(mask, threshold)`` where the mask holds 1 for ink and 0 for paper.

    ``threshold`` defaults to Otsu's, computed on the page itself.  An already binarized
    page is split at 128, which is exact for the 0/255 values :func:`binarize` produces.
    """
    bins = histogram(page.gray)
    if threshold is None:
        threshold = 127 if _is_binary(bins) else otsu_threshold(bins)
    table = bytes(1 if i <= threshold else 0 for i in range(256))
    return (page.gray.translate(table), int(threshold))


def _mask_for_analysis(page: RenderedPage, max_side: int = 768) -> Tuple[bytes, int, int]:
    """Return ``(mask, width, height)`` for skew/orientation analysis.

    Two things separate this from :func:`ink_mask`: the page is shrunk so a 300 dpi scan
    costs the same as a thumbnail (skew and orientation are global properties, and a
    uniform shrink preserves both), and a page that is not already bilevel is thresholded
    *locally*.  A global threshold on a shaded scan classifies the dark half of the paper
    as ink, which would drown the very profile these functions measure.
    """
    gray, width, height = page.gray, page.width, page.height
    if width <= 0 or height <= 0:
        return (b"", 0, 0)
    longest = max(width, height)
    if longest > max_side:
        factor = int(math.ceil(longest / float(max_side)))
        gray, width, height = _box_shrink(gray, width, height, factor)
    if not _is_binary(histogram(gray)):
        shrunk = RenderedPage(
            page=page.page, width=width, height=height, scale=page.scale,
            gray=gray, backend=page.backend,
        )
        gray, _threshold = _binarize_sauvola(shrunk)
    table = bytes(1 if i <= 127 else 0 for i in range(256))
    return (gray.translate(table), width, height)


def otsu_threshold(bins: Sequence[int]) -> int:
    """Return Otsu's global threshold for a 256-bin histogram.

    Pixels at or below the returned level are the dark (ink) class.
    """
    total = sum(bins)
    if total <= 0:
        return 127
    weighted = sum(index * count for index, count in enumerate(bins))
    background = 0
    sum_background = 0
    best_variance = -1.0
    low = high = 127
    for level in range(256):
        background += bins[level]
        if background == 0:
            continue
        foreground = total - background
        if foreground == 0:
            break
        sum_background += level * bins[level]
        mean_b = sum_background / background
        mean_f = (weighted - sum_background) / foreground
        delta = mean_b - mean_f
        variance = background * foreground * delta * delta
        if variance > best_variance:
            best_variance = variance
            low = high = level
        elif variance == best_variance:
            # Empty bins leave the variance untouched; a strictly bimodal image ties
            # across its whole valley, and the middle of that plateau is the threshold
            # a reader means by "the valley".
            high = level
    return (low + high) // 2


# ======================================================================================
# Contrast
# ======================================================================================


def normalize_contrast(page: RenderedPage, low: float = 0.02, high: float = 0.98) -> RenderedPage:
    """Stretch the histogram between its 2nd and 98th percentiles.

    A scan whose paper sits at 200 and whose ink sits at 90 becomes a page whose paper is
    white and whose ink is black, which is what every downstream threshold assumes.  A
    flat page (no spread between the percentiles) is returned unchanged.
    """
    if page.width <= 0 or page.height <= 0:
        return page
    bins = histogram(page.gray)
    total = sum(bins)
    if total <= 0:
        return page
    low_target = total * float(low)
    high_target = total * float(high)
    cumulative = 0
    lo = 0
    hi = 255
    found_low = False
    for level, count in enumerate(bins):
        cumulative += count
        if not found_low and cumulative >= low_target:
            lo = level
            found_low = True
        if cumulative >= high_target:
            hi = level
            break
    if hi <= lo:
        return page
    span = float(hi - lo)
    table = bytearray(256)
    for value in range(256):
        scaled = int(round((value - lo) * 255.0 / span))
        table[value] = 0 if scaled < 0 else (255 if scaled > 255 else scaled)
    return _replace_gray(page, page.gray.translate(bytes(table)))


# ======================================================================================
# Binarization
# ======================================================================================


def _binarize_otsu(page: RenderedPage) -> Tuple[bytes, int]:
    """Global Otsu binarization; returns ``(gray, threshold)``."""
    threshold = otsu_threshold(histogram(page.gray))
    table = bytes(_INK if i <= threshold else _PAPER for i in range(256))
    return (page.gray.translate(table), threshold)


def _binarize_sauvola(
    page: RenderedPage,
    window: int = _SAUVOLA_WINDOW,
    k: float = _SAUVOLA_K,
    r: float = _SAUVOLA_R,
) -> Tuple[bytes, int]:
    """Sauvola local binarization; returns ``(gray, mean local threshold)``.

    ``T(x, y) = m * (1 + k * (s / R - 1))`` over a ``window x window`` neighbourhood.
    The local mean and variance come from summed-area (integral image) arithmetic, kept
    as a rolling band of column sums so the memory cost is O(width) rather than
    O(width * height) -- the same O(n) work, without a 70 MB table for a 300 dpi page.
    The interior of each row, where the window never clips, is evaluated as one
    comprehension over pre-differenced prefix sums; only the two edge strips need the
    clipped arithmetic.

    The reported threshold is the mean of the local thresholds, sampled every
    :data:`_THRESHOLD_SAMPLE_ROWS` rows -- it is a diagnostic for the report, not a value
    the binarization itself depends on.
    """
    width, height = page.width, page.height
    gray = page.gray
    if width <= 0 or height <= 0:
        return (gray, 127)
    window = max(3, int(window) | 1)
    radius = window // 2
    squares = _SQUARES
    sqrt = math.sqrt
    subtract = operator.sub
    one_minus_k = 1.0 - k
    k_over_r = k / r

    column_sum = [0] * width
    column_sq = [0] * width
    rows_in = min(height, radius + 1)
    for y in range(rows_in):
        row = gray[y * width : (y + 1) * width]
        column_sum = [a + b for a, b in zip(column_sum, row)]
        column_sq = [a + b for a, b in zip(column_sq, map(squares.__getitem__, row))]

    out = bytearray(width * height)
    sampled_total = 0.0
    sampled_count = 0
    for y in range(height):
        if y > 0:
            entering = y + radius
            if entering < height:
                row = gray[entering * width : (entering + 1) * width]
                column_sum = [a + b for a, b in zip(column_sum, row)]
                column_sq = [a + b for a, b in zip(column_sq, map(squares.__getitem__, row))]
                rows_in += 1
            leaving = y - radius - 1
            if leaving >= 0:
                row = gray[leaving * width : (leaving + 1) * width]
                column_sum = [a - b for a, b in zip(column_sum, row)]
                column_sq = [a - b for a, b in zip(column_sq, map(squares.__getitem__, row))]
                rows_in -= 1
        prefix = [0]
        prefix.extend(accumulate(column_sum))
        prefix_sq = [0]
        prefix_sq.extend(accumulate(column_sq))
        source = gray[y * width : (y + 1) * width]
        base = y * width

        if width > window:
            inverse = 1.0 / (window * rows_in)
            totals = map(subtract, prefix[window:], prefix[: width - window + 1])
            totals_sq = map(subtract, prefix_sq[window:], prefix_sq[: width - window + 1])
            # ``mean`` and ``variance`` are bound mid-expression so the comprehension
            # stays one pass; it runs once per interior pixel of a 300 dpi page.
            out[base + radius : base + width - radius] = bytes(
                _INK
                if value
                < (mean := total * inverse) * one_minus_k
                + k_over_r
                * mean
                * sqrt(variance if (variance := total_sq * inverse - mean * mean) > 0.0 else 0.0)
                else _PAPER
                for value, total, total_sq in zip(
                    source[radius : width - radius], totals, totals_sq
                )
            )
            edges: Iterable[int] = list(range(radius)) + list(range(width - radius, width))
        else:
            edges = range(width)

        sample = (y % _THRESHOLD_SAMPLE_ROWS) == 0
        for x in edges:
            left = x - radius
            right = x + radius + 1
            if left < 0:
                left = 0
            if right > width:
                right = width
            count = (right - left) * rows_in
            if count <= 0:
                out[base + x] = _PAPER
                continue
            total = prefix[right] - prefix[left]
            total_sq = prefix_sq[right] - prefix_sq[left]
            mean = total / count
            variance = total_sq / count - mean * mean
            deviation = sqrt(variance) if variance > 0.0 else 0.0
            local = mean * (1.0 + k * (deviation / r - 1.0))
            out[base + x] = _INK if source[x] < local else _PAPER
        if sample:
            for x in range(radius, width - radius, 4):
                left = x - radius
                right = x + radius + 1
                count = (right - left) * rows_in
                total = prefix[right] - prefix[left]
                total_sq = prefix_sq[right] - prefix_sq[left]
                mean = total / count
                variance = total_sq / count - mean * mean
                deviation = sqrt(variance) if variance > 0.0 else 0.0
                sampled_total += mean * (1.0 + k * (deviation / r - 1.0))
                sampled_count += 1

    if sampled_count:
        mean_threshold = int(round(sampled_total / sampled_count))
    else:
        mean_threshold = 127
    return (bytes(out), max(0, min(255, mean_threshold)))


def binarize(page: RenderedPage, method: str = "sauvola") -> RenderedPage:
    """Binarize a page to 0 (ink) and 255 (paper).

    Args:
        page: The raster to threshold.
        method: ``"sauvola"`` (default) for local adaptive thresholding with a 31 pixel
            window and ``k = 0.2`` -- the right choice for a shaded or unevenly lit scan
            -- or ``"otsu"`` for a single global threshold.

    Raises:
        ValidationError: ``method`` is not one of :data:`BINARIZE_METHODS`.
    """
    name = str(method).strip().lower()
    if name in ("adaptive", "local"):
        name = "sauvola"
    if name in ("global",):
        name = "otsu"
    if name not in BINARIZE_METHODS:
        raise ValidationError(
            "unknown binarization method %r; use one of %s" % (method, ", ".join(BINARIZE_METHODS))
        )
    gray, _threshold = _binarize_sauvola(page) if name == "sauvola" else _binarize_otsu(page)
    return _replace_gray(page, gray)


def _binarize_with_threshold(page: RenderedPage, method: str) -> Tuple[RenderedPage, int]:
    """Like :func:`binarize` but also returns the threshold, for the report."""
    name = "otsu" if str(method).strip().lower() in ("otsu", "global") else "sauvola"
    gray, threshold = _binarize_sauvola(page) if name == "sauvola" else _binarize_otsu(page)
    return (_replace_gray(page, gray), threshold)


# ======================================================================================
# Denoising
# ======================================================================================


def _median3(gray: bytes, width: int, height: int) -> bytes:
    """3x3 median filter.  The one-pixel border is left as it was.

    The nine neighbours of a whole row are produced by zipping three offset slices of
    each of the three source rows, which keeps the per-pixel work down to one ``sorted``
    of a nine-tuple with no index arithmetic at all.
    """
    if width < 3 or height < 3:
        return gray
    out = bytearray(gray)
    binary = _is_binary(histogram(gray))
    span = width - 2
    for y in range(1, height - 1):
        above = gray[(y - 1) * width : y * width]
        middle = gray[y * width : (y + 1) * width]
        below = gray[(y + 1) * width : (y + 2) * width]
        start = y * width + 1
        if binary:
            # 0/255 only, so the median is simply the majority of the nine.
            out[start : start + span] = bytes(
                _PAPER if (a + b + c + d + e + f + g + h + i) >= 1275 else _INK
                for a, b, c, d, e, f, g, h, i in zip(
                    above, above[1:], above[2:],
                    middle, middle[1:], middle[2:],
                    below, below[1:], below[2:],
                )
            )
            continue
        out[start : start + span] = bytes(
            sorted(neighbourhood)[4]
            for neighbourhood in zip(
                above, above[1:], above[2:],
                middle, middle[1:], middle[2:],
                below, below[1:], below[2:],
            )
        )
    return bytes(out)


def _small_component_runs(
    mask: bytes, width: int, height: int, min_size: int
) -> Tuple[List[Tuple[int, int, int]], int]:
    """Find ink blobs below ``min_size`` pixels.

    Two-pass run-based connected-component labelling with 8-connectivity, backed by a
    union-find over *runs* rather than pixels, so the memory cost is proportional to the
    amount of ink instead of the page area.

    Returns:
        ``(runs, component_count)`` where ``runs`` is the ``(y, x0, x1)`` spans that
        belong to a component smaller than ``min_size``.
    """
    parent: List[int] = []

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> int:
        root_a, root_b = find(a), find(b)
        if root_a == root_b:
            return root_a
        if root_a > root_b:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        return root_a

    runs: List[Tuple[int, int, int, int]] = []
    previous: List[Tuple[int, int, int]] = []
    for y in range(height):
        row = mask[y * width : (y + 1) * width]
        current: List[Tuple[int, int, int]] = []
        cursor = 0
        while True:
            start = row.find(1, cursor)
            if start < 0:
                break
            stop = row.find(0, start)
            if stop < 0:
                stop = width
            label: Optional[int] = None
            for prev_start, prev_stop, prev_label in previous:
                if prev_start <= stop and start <= prev_stop:
                    label = find(prev_label) if label is None else union(label, prev_label)
            if label is None:
                label = len(parent)
                parent.append(label)
            current.append((start, stop, label))
            runs.append((y, start, stop, label))
            cursor = stop + 1
        previous = current

    sizes: Dict[int, int] = {}
    for _y, start, stop, label in runs:
        root = find(label)
        sizes[root] = sizes.get(root, 0) + (stop - start)
    small = set(root for root, size in sizes.items() if size < min_size)
    removed = [(y, start, stop) for y, start, stop, label in runs if find(label) in small]
    return (removed, len(small))


def _denoise_counted(
    page: RenderedPage, min_component: int = _MIN_COMPONENT_PIXELS
) -> Tuple[RenderedPage, int]:
    """:func:`denoise`, additionally reporting how many blobs were removed."""
    if page.width <= 0 or page.height <= 0:
        return (page, 0)
    filtered = _median3(page.gray, page.width, page.height)
    intermediate = _replace_gray(page, filtered)
    mask, _threshold = ink_mask(intermediate)
    removed, count = _small_component_runs(mask, page.width, page.height, int(min_component))
    if not removed:
        return (intermediate, 0)
    out = bytearray(filtered)
    for y, start, stop in removed:
        base = y * page.width
        out[base + start : base + stop] = b"\xff" * (stop - start)
    return (_replace_gray(page, out), count)


def denoise(page: RenderedPage, min_component: int = _MIN_COMPONENT_PIXELS) -> RenderedPage:
    """Median-filter the page and drop ink blobs smaller than ``min_component`` pixels.

    The 3x3 median kills salt-and-pepper speckle without softening strokes the way a
    blur would; the connected-component pass then removes the isolated dust that
    survives it (scanner grit, dot-matrix noise, JPEG mosquito noise).
    """
    return _denoise_counted(page, min_component)[0]


# ======================================================================================
# Skew and orientation
# ======================================================================================


def _ink_points(
    mask: bytes, width: int, height: int, max_points: int = _SKEW_MAX_POINTS
) -> List[Tuple[int, int]]:
    """Return a bounded, evenly spread sample of ink pixel coordinates."""
    total = mask.count(1)
    if total == 0:
        return []
    stride = 1
    if total > max_points:
        stride = int(math.ceil(total / float(max_points)))
    points: List[Tuple[int, int]] = []
    seen = 0
    for y in range(height):
        row = mask[y * width : (y + 1) * width]
        if 1 not in row:
            continue
        cursor = 0
        while True:
            x = row.find(1, cursor)
            if x < 0:
                break
            if seen % stride == 0:
                points.append((x, y))
            seen += 1
            cursor = x + 1
    return points


def estimate_skew(
    page: RenderedPage, limit: float = _SKEW_LIMIT, step: float = _SKEW_STEP
) -> float:
    """Estimate the page skew in degrees, searching ``[-limit, +limit]``.

    The ink is projected onto the y axis at each candidate angle and the angle whose
    row-sum profile has the largest variance wins: text lines only stack into sharp
    peaks when the projection runs along the baselines.  Because the total ink count is
    the same at every angle and the profile length is held constant, maximizing the
    variance is exactly maximizing the sum of squared bucket counts.

    Returns:
        The skew in degrees, positive when the content is rotated counter-clockwise as
        displayed.  Straighten with ``deskew(page, -angle)``.  Ties resolve toward 0.
    """
    mask, width, height = _mask_for_analysis(page)
    points = _ink_points(mask, width, height)
    if len(points) < 16:
        return 0.0
    limit = abs(float(limit))
    step = abs(float(step)) or _SKEW_STEP
    shear = math.tan(math.radians(limit))
    # Every angle projects into the same [0, height + 2 * offset] band of buckets, which
    # is what makes the scores comparable across angles.
    offset = int(math.ceil(width * shear)) + 1

    steps = int(round(2.0 * limit / step))
    best_score = -1.0
    best_angle = 0.0
    for index in range(steps + 1):
        angle = -limit + index * step
        tangent = math.tan(math.radians(angle))
        counts = Counter(
            int(y + x * tangent) + offset for x, y in points
        )
        score = 0.0
        for value in counts.values():
            score += value * value
        if score > best_score or (score == best_score and abs(angle) < abs(best_angle)):
            best_score = score
            best_angle = angle
    return round(best_angle, 4)


def deskew(page: RenderedPage, angle: float) -> RenderedPage:
    """Rotate the raster by ``angle`` degrees about its centre, bilinearly sampled.

    Positive angles rotate counter-clockwise as displayed, matching the sign
    :func:`estimate_skew` returns; the canvas keeps its size and anything rotated in
    from outside the original page is white.
    """
    value = float(angle)
    if abs(value) < 1e-9 or page.width <= 0 or page.height <= 0:
        return page
    width, height = page.width, page.height
    gray = page.gray
    radians = math.radians(value)
    cos = math.cos(radians)
    sin = math.sin(radians)
    cx = width / 2.0
    cy = height / 2.0
    out = bytearray(b"\xff" * (width * height))
    max_x = width - 1
    max_y = height - 1
    for py in range(height):
        dy = py + 0.5 - cy
        # Inverse rotation: source = centre + (dx*cos - dy*sin, dx*sin + dy*cos)
        sx = cx - (cx - 0.5) * cos - dy * sin - 0.5
        sy = cy - (cx - 0.5) * sin + dy * cos - 0.5
        base = py * width
        for px in range(width):
            if 0.0 <= sx <= max_x and 0.0 <= sy <= max_y:
                ix = int(sx)
                iy = int(sy)
                fx = sx - ix
                fy = sy - iy
                ix1 = ix + 1 if ix < max_x else ix
                iy1 = iy + 1 if iy < max_y else iy
                top = iy * width
                bottom = iy1 * width
                p00 = gray[top + ix]
                p01 = gray[top + ix1]
                p10 = gray[bottom + ix]
                p11 = gray[bottom + ix1]
                top_value = p00 + (p01 - p00) * fx
                bottom_value = p10 + (p11 - p10) * fx
                out[base + px] = int(top_value + (bottom_value - top_value) * fy + 0.5)
            sx += cos
            sy += sin
    return _replace_gray(page, bytes(out))


def rotate_quarter(page: RenderedPage, degrees: int) -> RenderedPage:
    """Rotate by an exact multiple of 90 degrees (clockwise as displayed), losslessly."""
    turns = int(degrees) // 90 % 4
    width, height = page.width, page.height
    gray = page.gray
    if turns == 0 or width <= 0 or height <= 0:
        return page
    if turns == 2:
        return _replace_gray(page, gray[::-1])
    if turns == 1:  # 90 clockwise: new row y' is old column y', bottom to top
        rows = [gray[x::width][::-1] for x in range(width)]
        return _replace_gray(page, b"".join(rows), width=height, height=width)
    rows = [gray[width - 1 - x :: width] for x in range(width)]
    return _replace_gray(page, b"".join(rows), width=height, height=width)


def _profile_score(counts: Sequence[int]) -> float:
    """Return the squared coefficient of variation of a projection profile.

    Normalizing by the mean makes profiles of different lengths comparable, which is
    what the horizontal-versus-vertical comparison needs.
    """
    n = len(counts)
    if n == 0:
        return 0.0
    total = float(sum(counts))
    if total <= 0.0:
        return 0.0
    mean = total / n
    variance = sum((value - mean) ** 2 for value in counts) / n
    return variance / (mean * mean)


def detect_orientation(page: RenderedPage) -> int:
    """Return 0 or 90: the quarter turn that puts the text lines horizontal.

    Text lines produce a strongly peaked projection *perpendicular* to the baseline, so
    comparing the horizontal and vertical projections separates upright pages from
    sideways ones.  It cannot tell 0 from 180, or 90 from 270 -- that needs glyph
    evidence -- so the answer is deliberately restricted to ``{0, 90}`` and the OCR stage
    is expected to confirm the remaining half turn.
    """
    if page.width <= 0 or page.height <= 0:
        return 0
    mask, width, height = _mask_for_analysis(page)
    if not mask or mask.count(1) == 0:
        return 0
    rows = [mask[y * width : (y + 1) * width].count(1) for y in range(height)]
    columns = [mask[x::width].count(1) for x in range(width)]
    row_score = _profile_score(rows)
    column_score = _profile_score(columns)
    return 0 if row_score >= column_score else 90


# ======================================================================================
# The pipeline
# ======================================================================================


def preprocess(
    page: RenderedPage, config: Any = None, method: str = "sauvola"
) -> Tuple[RenderedPage, PreprocessReport]:
    """Run the full scan cleanup and report every decision.

    The order is orientation -> deskew -> contrast -> denoise -> binarize: the geometric
    corrections come first so the photometric ones see square text, and binarization
    comes last so it sees the cleanest gray available.

    Args:
        page: The raster to clean up.
        config: A :class:`~zfp.core.config.ZfpConfig` (or ``None``).  Accepted for
            contract compatibility; the preprocessing constants are not yet exposed as
            configuration, so only the presence of the object is honoured.
        method: Binarization method, see :func:`binarize`.

    Returns:
        ``(page, report)`` -- the binarized page and its :class:`PreprocessReport`.
    """
    report = PreprocessReport(method="otsu" if str(method).lower() == "otsu" else "sauvola")
    if page.width <= 0 or page.height <= 0:
        report.notes.append("empty raster")
        return (page, report)

    orientation = detect_orientation(page)
    report.orientation = orientation
    if orientation:
        page = rotate_quarter(page, orientation)
        report.steps.append("orientation:%d" % orientation)
    else:
        report.steps.append("orientation:0")

    skew = estimate_skew(page)
    report.skew_angle = skew
    if abs(skew) >= 0.1:
        page = deskew(page, -skew)
        report.steps.append("deskew:%.2f" % -skew)
    else:
        report.notes.append("skew below the 0.1 degree correction floor")

    page = normalize_contrast(page)
    report.steps.append("normalize_contrast")

    page, removed = _denoise_counted(page)
    report.removed_components = removed
    report.steps.append("denoise:%d" % removed)

    page, threshold = _binarize_with_threshold(page, report.method)
    report.threshold = threshold
    report.steps.append("binarize:%s" % report.method)

    ink = page.gray.count(_INK)
    report.ink_ratio = ink / float(page.width * page.height)
    return (page, report)
