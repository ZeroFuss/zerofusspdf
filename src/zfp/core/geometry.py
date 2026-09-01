"""Geometry primitives shared by every ZFP module.

All rectangles that cross a module boundary are expressed in **PDF user space**:
y-up, origin at the page origin, ``x0 <= x1``, ``y0 <= y1``, floats in points.
Raster/pixel space is confined to the raster/ocr/vision layers, which convert with
:class:`PageGeometry` before returning anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .errors import ValidationError

__all__ = ["EPS", "Point", "Rect", "Matrix", "PageGeometry"]

EPS: float = 1e-6

_VALID_ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class Point:
    """An immutable point in PDF user space."""

    x: float
    y: float

    def translated(self, dx: float, dy: float) -> Point:
        """Return a copy shifted by ``(dx, dy)``."""
        return Point(self.x + dx, self.y + dy)

    def distance_to(self, other: Point) -> float:
        """Euclidean distance to ``other``."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def as_tuple(self) -> Tuple[float, float]:
        """Return ``(x, y)``."""
        return (self.x, self.y)


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in PDF user space.

    Instances are *not* auto-normalized (construction stays cheap and lossless);
    call :meth:`normalized` when the ordering of the corners is not guaranteed.
    Factory helpers such as :meth:`from_list` and :meth:`from_points` normalize.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    # ---------------------------------------------------------------- factories
    @staticmethod
    def from_points(a: Point, b: Point) -> Rect:
        """Build the normalized rectangle spanned by two corner points."""
        return Rect(min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y))

    @staticmethod
    def from_list(v: Sequence[float]) -> Rect:
        """Build a normalized rectangle from ``[x0, y0, x1, y1]``."""
        if v is None or len(v) < 4:
            raise ValidationError("Rect.from_list needs at least 4 numbers, got %r" % (v,))
        x0, y0, x1, y1 = (float(v[0]), float(v[1]), float(v[2]), float(v[3]))
        return Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    @staticmethod
    def bounding(rects: Iterable[Rect]) -> Optional[Rect]:
        """Return the bounding box of ``rects`` or ``None`` when the iterable is empty."""
        out: Optional[Rect] = None
        for r in rects:
            n = r.normalized()
            if out is None:
                out = n
            else:
                out = Rect(
                    min(out.x0, n.x0), min(out.y0, n.y0), max(out.x1, n.x1), max(out.y1, n.y1)
                )
        return out

    # -------------------------------------------------------------- properties
    @property
    def width(self) -> float:
        """Width in points (always >= 0)."""
        return abs(self.x1 - self.x0)

    @property
    def height(self) -> float:
        """Height in points (always >= 0)."""
        return abs(self.y1 - self.y0)

    @property
    def area(self) -> float:
        """Area in square points (always >= 0)."""
        return self.width * self.height

    @property
    def center(self) -> Point:
        """Geometric center."""
        return Point((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    # ------------------------------------------------------------- derivations
    def normalized(self) -> Rect:
        """Return a copy with ``x0 <= x1`` and ``y0 <= y1``."""
        return Rect(
            min(self.x0, self.x1),
            min(self.y0, self.y1),
            max(self.x0, self.x1),
            max(self.y0, self.y1),
        )

    def inflated(self, dx: float, dy: Optional[float] = None) -> Rect:
        """Grow the rectangle by ``dx`` horizontally and ``dy`` vertically (``dy=dx`` by default)."""
        d_y = dx if dy is None else dy
        n = self.normalized()
        return Rect(n.x0 - dx, n.y0 - d_y, n.x1 + dx, n.y1 + d_y)

    def translated(self, dx: float, dy: float) -> Rect:
        """Return a copy shifted by ``(dx, dy)``."""
        return Rect(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def scaled(self, sx: float, sy: Optional[float] = None) -> Rect:
        """Scale all coordinates about the user-space origin (``sy=sx`` by default)."""
        s_y = sx if sy is None else sy
        return Rect(self.x0 * sx, self.y0 * s_y, self.x1 * sx, self.y1 * s_y).normalized()

    def union(self, other: Rect) -> Rect:
        """Return the smallest rectangle containing both rectangles."""
        a, b = self.normalized(), other.normalized()
        return Rect(min(a.x0, b.x0), min(a.y0, b.y0), max(a.x1, b.x1), max(a.y1, b.y1))

    def intersection(self, other: Rect) -> Optional[Rect]:
        """Return the overlapping rectangle, or ``None`` when the rectangles are disjoint.

        Closed-set semantics: rectangles that merely touch (or degenerate rules with
        zero thickness) yield a zero-area rectangle rather than ``None``.
        """
        a, b = self.normalized(), other.normalized()
        x0, y0 = max(a.x0, b.x0), max(a.y0, b.y0)
        x1, y1 = min(a.x1, b.x1), min(a.y1, b.y1)
        if x1 < x0 or y1 < y0:
            return None
        return Rect(x0, y0, x1, y1)

    def iou(self, other: Rect) -> float:
        """Intersection-over-union in ``[0, 1]``; ``0.0`` when both rectangles have no area."""
        inter = self.intersection(other)
        inter_area = 0.0 if inter is None else inter.area
        union_area = self.area + other.area - inter_area
        if union_area <= EPS:
            return 0.0
        return inter_area / union_area

    # --------------------------------------------------------------- predicates
    def contains_point(self, p: Point) -> bool:
        """True when ``p`` lies inside or on the boundary."""
        n = self.normalized()
        return n.x0 <= p.x <= n.x1 and n.y0 <= p.y <= n.y1

    def contains_rect(self, other: Rect) -> bool:
        """True when ``other`` lies entirely inside (boundary inclusive)."""
        a, b = self.normalized(), other.normalized()
        return a.x0 <= b.x0 and a.y0 <= b.y0 and b.x1 <= a.x1 and b.y1 <= a.y1

    def intersects(self, other: Rect) -> bool:
        """True when :meth:`intersection` is not ``None``."""
        return self.intersection(other) is not None

    def horizontal_overlap(self, other: Rect) -> float:
        """Length in points over which the two x-ranges overlap (0 when disjoint)."""
        a, b = self.normalized(), other.normalized()
        return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))

    def vertical_overlap(self, other: Rect) -> float:
        """Length in points over which the two y-ranges overlap (0 when disjoint)."""
        a, b = self.normalized(), other.normalized()
        return max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))

    # ------------------------------------------------------------ serialization
    def as_list(self) -> List[float]:
        """Return ``[x0, y0, x1, y1]``."""
        return [self.x0, self.y0, self.x1, self.y1]

    def rounded(self, ndigits: int = 3) -> Rect:
        """Return a copy with every coordinate rounded to ``ndigits`` decimals."""
        return Rect(
            round(self.x0, ndigits),
            round(self.y0, ndigits),
            round(self.x1, ndigits),
            round(self.y1, ndigits),
        )


@dataclass(frozen=True)
class Matrix:
    """A PDF affine transform ``[a b 0; c d 0; e f 1]`` applied to row vectors.

    ``x' = a*x + c*y + e`` and ``y' = b*x + d*y + f``.
    """

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    # ---------------------------------------------------------------- factories
    @staticmethod
    def identity() -> Matrix:
        """The identity transform."""
        return Matrix()

    @staticmethod
    def translation(tx: float, ty: float) -> Matrix:
        """Translate by ``(tx, ty)``."""
        return Matrix(1.0, 0.0, 0.0, 1.0, tx, ty)

    @staticmethod
    def scaling(sx: float, sy: float) -> Matrix:
        """Scale by ``(sx, sy)`` about the origin."""
        return Matrix(sx, 0.0, 0.0, sy, 0.0, 0.0)

    @staticmethod
    def rotation(degrees: float) -> Matrix:
        """Rotate counter-clockwise by ``degrees`` about the origin (PDF convention)."""
        rad = math.radians(degrees)
        cos, sin = math.cos(rad), math.sin(rad)
        # Snap the exact quarter turns so round trips stay bit-clean.
        if abs(cos) < 1e-15:
            cos = 0.0
        if abs(sin) < 1e-15:
            sin = 0.0
        return Matrix(cos, sin, -sin, cos, 0.0, 0.0)

    # ---------------------------------------------------------------- algebra
    def concat(self, other: Matrix) -> Matrix:
        """Return ``self`` **then** ``other`` — the PDF matrix product ``self x other``."""
        return Matrix(
            a=self.a * other.a + self.b * other.c,
            b=self.a * other.b + self.b * other.d,
            c=self.c * other.a + self.d * other.c,
            d=self.c * other.b + self.d * other.d,
            e=self.e * other.a + self.f * other.c + other.e,
            f=self.e * other.b + self.f * other.d + other.f,
        )

    def apply(self, p: Point) -> Point:
        """Transform a point."""
        x, y = self.apply_xy(p.x, p.y)
        return Point(x, y)

    def apply_xy(self, x: float, y: float) -> Tuple[float, float]:
        """Transform raw coordinates."""
        return (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)

    def transform_rect(self, r: Rect) -> Rect:
        """Return the axis-aligned bounding box of the four transformed corners."""
        n = r.normalized()
        corners = (
            self.apply_xy(n.x0, n.y0),
            self.apply_xy(n.x1, n.y0),
            self.apply_xy(n.x1, n.y1),
            self.apply_xy(n.x0, n.y1),
        )
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return Rect(min(xs), min(ys), max(xs), max(ys))

    def determinant(self) -> float:
        """Determinant of the linear part."""
        return self.a * self.d - self.b * self.c

    def inverted(self) -> Matrix:
        """Return the inverse transform.

        Raises:
            ValidationError: when the matrix is singular.
        """
        det = self.determinant()
        if abs(det) < 1e-12:
            raise ValidationError("Matrix is singular and cannot be inverted: %r" % (self,))
        ia = self.d / det
        ib = -self.b / det
        ic = -self.c / det
        id_ = self.a / det
        ie = -(self.e * ia + self.f * ic)
        if_ = -(self.e * ib + self.f * id_)
        return Matrix(ia, ib, ic, id_, ie, if_)

    def as_tuple(self) -> Tuple[float, ...]:
        """Return ``(a, b, c, d, e, f)``."""
        return (self.a, self.b, self.c, self.d, self.e, self.f)


def _normalize_rotation(value: float) -> int:
    """Snap an arbitrary /Rotate value onto 0/90/180/270."""
    try:
        quarter = int(round(float(value) / 90.0))
    except (TypeError, ValueError):
        return 0
    return (quarter % 4) * 90


@dataclass(frozen=True)
class PageGeometry:
    """Everything needed to move between raster pixels and PDF user space."""

    index: int
    media_box: Rect
    crop_box: Rect
    rotation: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "media_box", self.media_box.normalized())
        object.__setattr__(self, "crop_box", self.crop_box.normalized())
        object.__setattr__(self, "rotation", _normalize_rotation(self.rotation))

    # -------------------------------------------------------------- properties
    @property
    def width(self) -> float:
        """Crop box width in points."""
        return self.crop_box.width

    @property
    def height(self) -> float:
        """Crop box height in points."""
        return self.crop_box.height

    @property
    def display_size(self) -> Tuple[float, float]:
        """Displayed ``(width, height)`` in points, swapped for 90/270 rotation."""
        if self.rotation in (90, 270):
            return (self.height, self.width)
        return (self.width, self.height)

    # ------------------------------------------------------------- transforms
    def render_matrix(self, scale: float) -> Matrix:
        """Return the user-space -> pixel-space transform for a raster at ``scale``.

        The target raster has a top-left origin, y-down, and size
        ``(display_w * scale, display_h * scale)``. The crop box origin is subtracted
        first, then ``rotation`` is applied so the visible page appears rotated
        **clockwise** by ``rotation`` degrees.
        """
        s = float(scale)
        cx0, cy0 = self.crop_box.x0, self.crop_box.y0
        cx1, cy1 = self.crop_box.x1, self.crop_box.y1
        if self.rotation == 90:
            # px = s*(y - cy0); py = s*(x - cx0)
            return Matrix(0.0, s, s, 0.0, -s * cy0, -s * cx0)
        if self.rotation == 180:
            # px = s*(cx1 - x); py = s*(y - cy0)
            return Matrix(-s, 0.0, 0.0, s, s * cx1, -s * cy0)
        if self.rotation == 270:
            # px = s*(cy1 - y); py = s*(cx1 - x)
            return Matrix(0.0, -s, -s, 0.0, s * cy1, s * cx1)
        # rotation == 0: px = s*(x - cx0); py = s*(cy1 - y)
        return Matrix(s, 0.0, 0.0, -s, -s * cx0, s * cy1)

    def pixel_size(self, scale: float) -> Tuple[int, int]:
        """Raster size in whole pixels for ``scale`` (ceil, never below 1)."""
        dw, dh = self.display_size
        return (max(1, int(math.ceil(dw * scale - EPS))), max(1, int(math.ceil(dh * scale - EPS))))

    def pixel_to_user(self, px: float, py: float, scale: float) -> Point:
        """Convert a pixel coordinate back to PDF user space."""
        x, y = self.render_matrix(scale).inverted().apply_xy(px, py)
        return Point(x, y)

    def user_to_pixel(self, x: float, y: float, scale: float) -> Tuple[float, float]:
        """Convert a user-space coordinate to pixel space."""
        return self.render_matrix(scale).apply_xy(x, y)

    def pixel_rect_to_user(self, rect: Rect, scale: float) -> Rect:
        """Convert a pixel-space rectangle to a normalized user-space rectangle."""
        return self.render_matrix(scale).inverted().transform_rect(rect).normalized()

    def user_rect_to_pixel(self, rect: Rect, scale: float) -> Rect:
        """Convert a user-space rectangle to a normalized pixel-space rectangle."""
        return self.render_matrix(scale).transform_rect(rect).normalized()

    def clamp(self, r: Rect) -> Rect:
        """Clip ``r`` to the crop box, clamping each coordinate independently."""
        n = r.normalized()
        c = self.crop_box
        x0 = min(max(n.x0, c.x0), c.x1)
        x1 = min(max(n.x1, c.x0), c.x1)
        y0 = min(max(n.y0, c.y0), c.y1)
        y1 = min(max(n.y1, c.y0), c.y1)
        return Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
