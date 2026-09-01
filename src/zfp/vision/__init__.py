"""Geometric perception: the shapes a form is made of.

Three modules, all pure CPython:

* :mod:`zfp.vision.primitives` -- rules, boxes, circles, checkbox glyphs, table cells and
  comb runs, reasoned out of the :class:`~zfp.core.types.VectorPrimitive` objects the
  content-stream interpreter produced.  This is the native path, and it is exact.
* :mod:`zfp.vision.blanks` -- the occupancy model of a page and the maximal empty
  rectangles inside it, which is where borderless fields live.
* :mod:`zfp.vision.raster_shapes` -- the same shapes recovered from the pixels of a
  scan, with an OpenCV fast path and a pure-python one that needs nothing at all.

Every rectangle leaving this package is PDF user space, y-up, page origin, points,
rounded to three decimals.  Pixel space stops at :func:`detect_shapes_from_image`.
"""

from __future__ import annotations

from .blanks import (
    OccupancyGrid,
    blank_regions,
    line_gaps,
    maximal_empty_cells,
    occupancy_grid,
    whitespace_profile,
)
from .primitives import (
    GLYPH_CHECKBOXES,
    ZAPF_BOX_CHARS,
    boxes_from_rules,
    dedupe_rects,
    detect_boxes,
    detect_checkbox_glyphs,
    detect_circles,
    detect_comb_cells,
    detect_table_cells,
    horizontal_rules,
    merge_collinear,
    normalize_primitives,
    reading_key,
    rule_spans,
    vertical_rules,
)
from .raster_shapes import RasterShapes, binarize_ink, detect_shapes_from_image

__all__ = [
    "GLYPH_CHECKBOXES",
    "OccupancyGrid",
    "RasterShapes",
    "ZAPF_BOX_CHARS",
    "binarize_ink",
    "blank_regions",
    "boxes_from_rules",
    "dedupe_rects",
    "detect_boxes",
    "detect_checkbox_glyphs",
    "detect_circles",
    "detect_comb_cells",
    "detect_shapes_from_image",
    "detect_table_cells",
    "horizontal_rules",
    "line_gaps",
    "maximal_empty_cells",
    "merge_collinear",
    "normalize_primitives",
    "occupancy_grid",
    "reading_key",
    "rule_spans",
    "vertical_rules",
    "whitespace_profile",
]
