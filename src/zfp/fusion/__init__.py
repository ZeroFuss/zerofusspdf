"""Geometry fusion and candidate scoring.

Detectors propose; fusion decides.  This package takes the overlapping, contradictory,
differently-measured proposals of every archetype detector, snaps them onto the exact
vector geometry where exact geometry exists, merges the ones that are describing the
same field, resolves what is left of the overlaps, and returns one ranked, calibrated
list of candidates in reading order.

* :mod:`zfp.fusion.geometry_fusion` -- snapping, merging, suppression, scoring, and the
  :func:`~zfp.fusion.geometry_fusion.fuse` pipeline that runs them in the right order.
* :mod:`zfp.fusion.calibration` -- per-type padding and size sanity, plus the learned
  per-edge :class:`~zfp.fusion.calibration.Calibrator` and the coordinate-ascent
  :func:`~zfp.fusion.calibration.calibrate_weights`, so the numbers can be fitted to a
  corpus instead of being hard-coded forever.

Every rectangle entering or leaving this package is PDF user space, y-up, page origin,
points.  Everything here is pure CPython and deterministic.
"""

from __future__ import annotations

from .calibration import (
    DEFAULT_PADDING,
    MAX_SIZE,
    MIN_SIZE,
    Calibrator,
    EdgeAdjustment,
    FieldPadding,
    calibrate,
    calibrate_weights,
    padding_for,
    size_bounds,
)
from .geometry_fusion import (
    DEFAULT_SNAP_TOL_PT,
    MAX_WIDGET_IOU,
    MIN_OVERLAP_IOU,
    SNAP_EXPANSION_FACTOR,
    TYPE_SPECIFICITY,
    calibrate_rect,
    deduplicate,
    fuse,
    fused_score,
    merge_cluster,
    rank,
    score_candidates,
    snap_to_primitive,
    snap_tolerance,
    suppress_overlaps,
)

__all__ = [
    "DEFAULT_PADDING",
    "DEFAULT_SNAP_TOL_PT",
    "MAX_SIZE",
    "MAX_WIDGET_IOU",
    "MIN_OVERLAP_IOU",
    "MIN_SIZE",
    "SNAP_EXPANSION_FACTOR",
    "TYPE_SPECIFICITY",
    "Calibrator",
    "EdgeAdjustment",
    "FieldPadding",
    "calibrate",
    "calibrate_rect",
    "calibrate_weights",
    "deduplicate",
    "fuse",
    "fused_score",
    "merge_cluster",
    "padding_for",
    "rank",
    "score_candidates",
    "size_bounds",
    "snap_to_primitive",
    "snap_tolerance",
    "suppress_overlaps",
]
