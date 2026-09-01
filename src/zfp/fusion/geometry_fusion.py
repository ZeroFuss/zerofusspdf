"""Geometry fusion: many detectors, one field.

Eleven archetype detectors, a vector pass, an OCR pass and a blank-region pass all look
at the same piece of paper and all describe the same field slightly differently.  An OCR
engine says an underline runs from about x=601 to x=1127; the vector detector, which
read the actual path operators, says 598 to 1130.  Both are "right".  Only one of them
is *exact*, and this module is where that gets decided.

The rules that make it work:

* **Snapping corrects estimates, never measurements.**  A candidate whose evidence is
  already exact vector geometry -- a rule, a stroked box, a table cell, a comb cell, a
  checkbox glyph -- has *already* had the page's own coordinates applied to it, through
  whichever convention its archetype uses (the inside of the ink for a box, a
  ``underline_gap_pt`` gap above a rule for an underline, the glyph box for a check).
  Re-snapping such a rectangle onto the raw path coordinates does not refine it, it
  undoes the convention.  So :func:`fuse` snaps only the candidates built from
  estimates -- blank regions, layout inference, OCR geometry -- and
  :func:`has_exact_geometry` is the predicate that decides.
* **Snapping is per edge, not per rectangle.**  A horizontal rule tells you exactly
  where a field starts and ends horizontally.  It tells you nothing whatsoever about the
  field's height.  So :func:`snap_to_primitive` adopts coordinates one edge at a time
  and leaves every edge it has no evidence for alone.
* **Agreement is merged, never dropped.**  Three detectors finding the same field is the
  single strongest signal available; :func:`deduplicate` fuses them into one candidate
  whose geometry confidence is *higher* than any of the three, because independent
  corroboration is worth more than any single observation.
* **Overlap is a hard invariant.**  ``docs/QA.md`` states that no two widgets on a page
  may overlap by more than 10% IoU.  :func:`suppress_overlaps` guarantees it by
  construction: a candidate is emitted only after its final rectangle has been checked
  against every rectangle already emitted.

Where the fused score lives
---------------------------
:class:`~zfp.core.types.FieldCandidate` has no ``score`` attribute, and inventing one
would put a number in the record that nothing else in the contract knows how to read.
So the composite is handled two ways at once:

* :func:`fused_score` recomputes it on demand, deterministically, from the candidate's
  own evidence.  It is a pure function -- call it anywhere, it always agrees.
* :func:`score_candidates` folds it into ``confidence.semantic_type`` so that
  :meth:`Confidence.overall` reflects it.  ``confidence.geometry`` is **never** touched
  here: geometry confidence belongs to the detectors that measured the geometry, and to
  :func:`deduplicate`'s corroboration bonus.  The fold never lowers a value another
  stage already established, so running fusion after semantics is safe.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.config import DetectionConfig, ScoringWeights
from ..core.geometry import EPS, PageGeometry, Rect
from ..core.logging import get_logger
from ..core.types import (
    Confidence,
    Evidence,
    EvidenceKind,
    FieldCandidate,
    FieldConstraints,
    FieldType,
    VectorPrimitive,
)
from .calibration import ROUND_DIGITS, calibrate

__all__ = [
    "DEFAULT_SNAP_TOL_PT",
    "EXACT_GEOMETRY_EVIDENCE",
    "MAX_WIDGET_IOU",
    "MIN_OVERLAP_IOU",
    "SNAP_EXPANSION_FACTOR",
    "TYPE_SPECIFICITY",
    "calibrate_rect",
    "deduplicate",
    "fuse",
    "has_exact_geometry",
    "fused_score",
    "merge_cluster",
    "rank",
    "score_candidates",
    "snap_to_primitive",
    "snap_tolerance",
    "suppress_overlaps",
]

LOG = get_logger(__name__)

#: Absolute invariant from ``docs/QA.md``: no two emitted widgets may overlap by more.
MAX_WIDGET_IOU = 0.10
#: Overlaps at or below this are ignored entirely by :func:`suppress_overlaps`.
MIN_OVERLAP_IOU = MAX_WIDGET_IOU
#: A snapped rectangle may never extend further than ``tol * this`` beyond the original.
SNAP_EXPANSION_FACTOR = 3.0
#: Default per-edge snapping tolerance in points; see :func:`snap_tolerance`.
DEFAULT_SNAP_TOL_PT = 4.0
#: A primitive this thin (in its short dimension) is a rule, not a box, whatever its
#: aspect ratio says.  Matches :attr:`DetectionConfig.max_line_thickness_pt`.
_RULE_MAX_THICKNESS_PT = 3.0
#: An x-edge donation needs the donor within this much vertically (and vice versa);
#: expressed as a multiple of the tolerance, floored by the rectangle's own height.
_ASSOC_TOL_FACTOR = 3.0
#: Bound on the shrink/re-check loop in :func:`suppress_overlaps`.
_SUPPRESS_MAX_PASSES = 8

#: Evidence kinds whose rectangle came straight off the page's own geometry rather than
#: from an estimate.  A candidate carrying any of them has already been placed by its
#: archetype's convention -- the inside of a stroked box, ``underline_gap_pt`` above a
#: rule, the bounding box of a check glyph -- and must not be re-snapped onto the raw
#: path coordinates, which would undo exactly that convention.
EXACT_GEOMETRY_EVIDENCE = frozenset(
    (
        EvidenceKind.VECTOR_LINE,
        EvidenceKind.VECTOR_RECT,
        EvidenceKind.VECTOR_CIRCLE,
        EvidenceKind.TABLE_CELL,
        EvidenceKind.COMB_CELL,
        EvidenceKind.CHECKBOX_GLYPH,
        EvidenceKind.EXISTING_WIDGET,
    )
)

#: How *specific* a field type is.  A merge keeps the most specific type of its members:
#: a detector that says CHECKBOX looked at a small square and measured its aspect ratio,
#: while a detector that says TEXT merely failed to say anything else, and UNKNOWN is the
#: absence of an opinion.  Higher wins.
TYPE_SPECIFICITY: Dict[FieldType, int] = {
    FieldType.UNKNOWN: 0,
    FieldType.TEXT: 10,
    FieldType.MULTILINE_TEXT: 20,
    FieldType.BUTTON: 25,
    FieldType.NUMBER: 30,
    FieldType.CURRENCY: 32,
    FieldType.EMAIL: 34,
    FieldType.PHONE: 36,
    FieldType.DATE: 38,
    FieldType.CHOICE: 40,
    FieldType.LISTBOX: 42,
    FieldType.COMB: 50,
    FieldType.SIGNATURE: 55,
    FieldType.CHECKBOX: 60,
    FieldType.RADIO: 65,
}


# ======================================================================================
# Small helpers
# ======================================================================================


def _detection(config: Any) -> DetectionConfig:
    """Coerce ``config`` -- a ``ZfpConfig``, a :class:`DetectionConfig` or ``None`` --
    into a :class:`DetectionConfig`."""
    if isinstance(config, DetectionConfig):
        return config
    inner = getattr(config, "detection", None)
    if isinstance(inner, DetectionConfig):
        return inner
    return DetectionConfig()


def _scoring(config: Any) -> ScoringWeights:
    """Coerce ``config`` into a :class:`ScoringWeights` (contract defaults when absent)."""
    if isinstance(config, ScoringWeights):
        return config
    inner = getattr(config, "scoring", None)
    if isinstance(inner, ScoringWeights):
        return inner
    return ScoringWeights()


def _clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _specificity(field_type: Any) -> int:
    """Return the specificity rank of ``field_type`` (unknown types rank lowest)."""
    if not isinstance(field_type, FieldType):
        try:
            field_type = FieldType(str(field_type))
        except (ValueError, TypeError):
            return 0
    return TYPE_SPECIFICITY.get(field_type, 0)


def _clone(cand: FieldCandidate) -> FieldCandidate:
    """Return a copy that shares no mutable state with ``cand``."""
    return replace(
        cand,
        sources=list(cand.sources),
        parent_context=list(cand.parent_context),
        evidence=list(cand.evidence),
        confidence=replace(cand.confidence),
        constraints=replace(cand.constraints, choices=list(cand.constraints.choices)),
    )


def _reading_key(cand: FieldCandidate) -> Tuple[int, float, float, float, str]:
    """Deterministic reading-order key: page, then top-down, then left-to-right."""
    r = cand.rect
    return (int(cand.page), -r.y1, r.x0, r.x1, str(cand.id))


def _gap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    """Distance between two 1-D spans; ``0.0`` when they overlap or touch."""
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


# ======================================================================================
# Snapping
# ======================================================================================


def has_exact_geometry(cand: FieldCandidate) -> bool:
    """True when ``cand`` was measured off the page rather than estimated.

    Membership is decided by :data:`EXACT_GEOMETRY_EVIDENCE`: a candidate built from a
    rule, a stroked box, a table cell, a comb cell, a check glyph or an existing widget
    already carries the page's own coordinates, mapped through its archetype's
    convention.  Everything else -- a blank region, a layout inference, an OCR word box
    -- is an estimate, and estimates are what :func:`snap_to_primitive` exists for.

    Args:
        cand: The candidate to classify.

    Returns:
        Whether the candidate's rectangle is a measurement.
    """
    if cand is None:
        return False
    for item in cand.evidence or ():
        if getattr(item, "kind", None) in EXACT_GEOMETRY_EVIDENCE:
            return True
    return False


def snap_tolerance(config: Any = None) -> float:
    """Return the per-edge snapping tolerance in points for ``config``.

    Two line-merge tolerances is the natural scale -- if two rules that far apart are
    considered the same rule, a candidate edge that far from a rule is describing that
    rule -- floored at :data:`DEFAULT_SNAP_TOL_PT` so an OCR estimate a few points off
    still finds its vector.
    """
    det = _detection(config)
    return max(DEFAULT_SNAP_TOL_PT, 2.0 * float(det.line_merge_tolerance_pt))


def _donations(prim: VectorPrimitive) -> List[Tuple[str, float, str]]:
    """Return the ``(edge, value, axis)`` coordinates ``prim`` is entitled to donate.

    A **horizontal rule** knows exactly where the field starts and stops horizontally,
    and where its bottom edge sits (either side of the stroke, whichever the candidate
    is closer to) -- and nothing about the top.  A **vertical rule** is one side of the
    field: it donates its x to whichever x edge is nearer, plus its extent vertically.
    A **box** donates all four edges, corner to corner.
    """
    rect = prim.rect.normalized()
    w, h = rect.width, rect.height
    if w <= EPS and h <= EPS:
        return []
    kind = str(getattr(prim, "kind", "") or "")
    thin = min(w, h) <= _RULE_MAX_THICKNESS_PT
    is_rule = kind == "line" or thin
    if is_rule and w >= h:
        return [
            ("x0", rect.x0, "x"),
            ("x1", rect.x1, "x"),
            ("y0", rect.y0, "y"),
            ("y0", rect.y1, "y"),
        ]
    if is_rule:
        return [
            ("x0", rect.x0, "x"),
            ("x0", rect.x1, "x"),
            ("x1", rect.x0, "x"),
            ("x1", rect.x1, "x"),
            ("y0", rect.y0, "y"),
            ("y1", rect.y1, "y"),
        ]
    return [
        ("x0", rect.x0, "x"),
        ("x1", rect.x1, "x"),
        ("y0", rect.y0, "y"),
        ("y1", rect.y1, "y"),
    ]


def _donor_priority(prim: VectorPrimitive) -> int:
    """Rank donors for tie-breaking: a box beats a rule beats anything else."""
    rect = prim.rect.normalized()
    kind = str(getattr(prim, "kind", "") or "")
    if kind == "rect" and min(rect.width, rect.height) > _RULE_MAX_THICKNESS_PT:
        return 3
    if kind in ("line", "rect"):
        return 2
    if kind == "circle":
        return 1
    return 0


def snap_to_primitive(rect: Rect, prims: Sequence[VectorPrimitive], tol: float) -> Rect:
    """Adopt the coordinates of nearby vector primitives, one edge at a time.

    For every edge of ``rect`` the closest primitive coordinate within ``tol`` wins and
    is adopted verbatim -- the vector path is exact, the estimate is not.  Edges with no
    primitive within ``tol`` are left exactly as they were, which is the whole point: an
    underline pins the x range and the bottom of a field and says nothing about its
    height, so snapping to one must not change the height.

    A donation is only considered when the donor is actually *associated* with the
    rectangle: an x coordinate may only come from a primitive that is vertically nearby,
    and a y coordinate only from one that is horizontally nearby.  Without that, a rule
    on the far side of the page at the right height would silently capture an edge.

    The result never extends more than ``SNAP_EXPANSION_FACTOR * tol`` beyond the input
    on any side, and is returned unchanged (the identical object) when nothing is close
    enough.

    Args:
        rect: The candidate rectangle, PDF user space.
        prims: Primitives to snap to; already filtered to this page by the caller.
        tol: Maximum per-edge distance, in points.

    Returns:
        The snapped rectangle, or ``rect`` itself when no edge moved.
    """
    if tol is None or tol <= 0.0 or not prims:
        return rect
    base = rect.normalized()
    assoc = max(_ASSOC_TOL_FACTOR * float(tol), base.height)
    current = {"x0": base.x0, "y0": base.y0, "x1": base.x1, "y1": base.y1}
    best: Dict[str, Tuple[Tuple[float, int, float, int], float]] = {}

    for index, prim in enumerate(prims):
        if prim is None or getattr(prim, "rect", None) is None:
            continue
        prect = prim.rect.normalized()
        priority = _donor_priority(prim)
        for edge, value, axis in _donations(prim):
            if axis == "x":
                if _gap_1d(base.y0, base.y1, prect.y0, prect.y1) > assoc + EPS:
                    continue
            elif _gap_1d(base.x0, base.x1, prect.x0, prect.x1) > assoc + EPS:
                continue
            distance = abs(value - current[edge])
            if distance > float(tol) + EPS:
                continue
            key = (round(distance, 9), -priority, round(value, 9), index)
            found = best.get(edge)
            if found is None or key < found[0]:
                best[edge] = (key, value)

    if not best:
        return rect

    out = dict(current)
    for edge, (_key, value) in best.items():
        out[edge] = value

    # Never expand beyond the allowed envelope, whatever tolerance the caller passed.
    limit = SNAP_EXPANSION_FACTOR * float(tol)
    out["x0"] = max(out["x0"], base.x0 - limit)
    out["y0"] = max(out["y0"], base.y0 - limit)
    out["x1"] = min(out["x1"], base.x1 + limit)
    out["y1"] = min(out["y1"], base.y1 + limit)

    # An axis that collapsed keeps its original span: a snap must never invert a field.
    if out["x1"] - out["x0"] <= EPS:
        out["x0"], out["x1"] = base.x0, base.x1
    if out["y1"] - out["y0"] <= EPS:
        out["y0"], out["y1"] = base.y0, base.y1

    snapped = Rect(out["x0"], out["y0"], out["x1"], out["y1"])
    if snapped.as_list() == base.as_list():
        return rect
    return snapped


# ======================================================================================
# Calibration entry point
# ======================================================================================


def calibrate_rect(rect: Rect, field_type: FieldType, config: Any = None) -> Rect:
    """Return ``rect`` padded for ``field_type`` and clamped to a sane size.

    Thin wrapper over :func:`zfp.fusion.calibration.calibrate` so callers of the fusion
    stage never have to reach into the calibration tables themselves.
    """
    return calibrate(rect, field_type, config)


# ======================================================================================
# Scoring
# ======================================================================================


def fused_score(cand: FieldCandidate, weights: Optional[ScoringWeights] = None) -> float:
    """Return the weighted composite of a candidate's evidence, in ``[0, 1]``.

    Pure and deterministic: it depends only on ``cand.evidence`` (collapsed into the
    seven contract buckets by :meth:`FieldCandidate.evidence_scores`) and on the
    weights.  Recomputing it later always yields the same number, which is why the
    composite does not need to be stored on the candidate at all.
    """
    if cand is None:
        return 0.0
    return _clamp01((weights or ScoringWeights()).score(cand.evidence_scores()))


def score_candidates(
    cands: Sequence[FieldCandidate], weights: Optional[ScoringWeights] = None
) -> List[FieldCandidate]:
    """Return copies of ``cands`` with the fused score folded into their confidence.

    The composite lands in ``confidence.semantic_type``, so that
    :meth:`Confidence.overall` -- the number the rest of the pipeline thresholds on --
    reflects the weighted evidence.  ``confidence.geometry`` is deliberately left alone:
    it is the detectors' measurement of *where* the field is, and only
    :func:`deduplicate`'s corroboration bonus is allowed to raise it.

    The fold takes the maximum of the existing value and the fused score, so a semantic
    confidence established by a later stage is never silently lowered by re-running
    fusion.  Inputs are not mutated.
    """
    w = weights or ScoringWeights()
    out: List[FieldCandidate] = []
    for cand in cands or []:
        if cand is None:
            continue
        clone = _clone(cand)
        value = fused_score(clone, w)
        clone.confidence.semantic_type = max(_clamp01(clone.confidence.semantic_type), value)
        out.append(clone)
    return out


def rank(
    cands: Sequence[FieldCandidate], weights: Optional[ScoringWeights] = None
) -> List[FieldCandidate]:
    """Return ``cands`` sorted by :func:`fused_score`, strongest first.

    Ties fall back to reading order (page, top-down, left-to-right, then id), so the
    ordering is total and reproducible across runs and machines.
    """
    w = weights or ScoringWeights()
    return sorted(
        [c for c in (cands or []) if c is not None],
        key=lambda c: (-fused_score(c, w),) + _reading_key(c),
    )


# ======================================================================================
# Deduplication / merging
# ======================================================================================


def _evidence_signature(cand: FieldCandidate) -> Tuple[str, ...]:
    """Return the sorted set of evidence kinds behind a candidate.

    Two candidates with the same signature saw the same *kind* of thing, so they are not
    independent observations and must not corroborate each other.
    """
    kinds = set()
    for item in cand.evidence or []:
        kind = item.kind.value if isinstance(item.kind, EvidenceKind) else str(item.kind)
        kinds.add(kind)
    if not kinds:
        kinds = {str(s) for s in (cand.sources or [])}
    if not kinds:
        return ("",)
    return tuple(sorted(kinds))


def _merge_evidence(members: Sequence[FieldCandidate]) -> List[Evidence]:
    """Concatenate member evidence, dropping exact ``(kind, score, detail)`` repeats."""
    seen = set()
    out: List[Evidence] = []
    for cand in members:
        for item in cand.evidence or []:
            kind = item.kind.value if isinstance(item.kind, EvidenceKind) else str(item.kind)
            key = (kind, round(float(item.score), 3), str(item.detail))
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
    return out


def _merge_constraints(
    primary: FieldCandidate, members: Sequence[FieldCandidate]
) -> FieldConstraints:
    """Take the primary's constraints, filling unset slots from the other members."""
    base = replace(primary.constraints, choices=list(primary.constraints.choices))
    for cand in members:
        other = cand.constraints
        if base.max_chars_estimate is None and other.max_chars_estimate is not None:
            base.max_chars_estimate = other.max_chars_estimate
        if base.comb_cells is None and other.comb_cells is not None:
            base.comb_cells = other.comb_cells
        if base.pattern is None and other.pattern is not None:
            base.pattern = other.pattern
        if base.format_hint is None and other.format_hint is not None:
            base.format_hint = other.format_hint
        if not base.choices and other.choices:
            base.choices = list(other.choices)
        # Boolean flags are unions: any detector asserting them is asserting evidence.
        base.required = base.required or other.required
        base.multiline = base.multiline or other.multiline
        base.read_only = base.read_only or other.read_only
    return base


def _corroborated_geometry(members: Sequence[FieldCandidate]) -> float:
    """Return the fused geometry confidence of a cluster.

    Members are grouped by evidence signature; the strongest of each *distinct* group is
    combined with a noisy-OR, ``1 - prod(1 - c_i)``, capped at ``0.999``.  Two detectors
    that both looked at vector lines are one observation and get no bonus; a vector
    line, an OCR word and a blank region are three, and the field is far more likely to
    be real than any one of them alone claimed.
    """
    per_signature: Dict[Tuple[str, ...], float] = {}
    for cand in members:
        signature = _evidence_signature(cand)
        value = _clamp01(cand.confidence.geometry)
        if value > per_signature.get(signature, -1.0):
            per_signature[signature] = value
    if not per_signature:
        return 0.0
    strongest = max(per_signature.values())
    if len(per_signature) < 2:
        return strongest
    product = 1.0
    for signature in sorted(per_signature):
        product *= 1.0 - per_signature[signature]
    fused = min(0.999, 1.0 - product)
    return max(fused, strongest)


def merge_cluster(members: Sequence[FieldCandidate]) -> FieldCandidate:
    """Fuse a cluster of overlapping candidates into a single one.

    The survivor keeps the rectangle of the member with the highest
    ``confidence.geometry`` (its geometry is the best-measured one), the most specific
    field type of any member, the first non-``None`` visible label, the union of sources
    and parent contexts, the concatenation of all distinct evidence, the per-axis
    maximum of every confidence component, and -- on ``geometry`` -- the corroboration
    bonus from :func:`_corroborated_geometry`.  Its ``id`` is the primary member's, so a
    fused candidate stays traceable back to the detector that placed it best.
    """
    ordered = [c for c in members if c is not None]
    if not ordered:
        raise ValueError("merge_cluster needs at least one candidate")
    if len(ordered) == 1:
        return _clone(ordered[0])

    primary = min(
        ordered,
        key=lambda c: (-_clamp01(c.confidence.geometry),) + _reading_key(c),
    )
    merged = _clone(primary)

    # Field type: most specific wins, then best geometry, then reading order.
    type_owner = min(
        ordered,
        key=lambda c: (
            -_specificity(c.field_type),
            -_clamp01(c.confidence.geometry),
        )
        + _reading_key(c),
    )
    merged.field_type = type_owner.field_type

    sources: List[str] = []
    contexts: List[str] = []
    for cand in ordered:
        for name in cand.sources or []:
            if name not in sources:
                sources.append(name)
        for name in cand.parent_context or []:
            if name not in contexts:
                contexts.append(name)
    merged.sources = sources
    merged.parent_context = contexts

    merged.evidence = _merge_evidence(ordered)
    merged.constraints = _merge_constraints(primary, [c for c in ordered if c is not primary])

    merged.confidence = Confidence(
        geometry=_corroborated_geometry(ordered),
        label_link=max(_clamp01(c.confidence.label_link) for c in ordered),
        semantic_type=max(_clamp01(c.confidence.semantic_type) for c in ordered),
        autofill_value=max(_clamp01(c.confidence.autofill_value) for c in ordered),
    )

    for cand in ordered:
        if merged.visible_label is None and cand.visible_label is not None:
            merged.visible_label = cand.visible_label
        if merged.canonical_key is None and cand.canonical_key is not None:
            merged.canonical_key = cand.canonical_key
        if merged.group_id is None and cand.group_id is not None:
            merged.group_id = cand.group_id
        if merged.export_value is None and cand.export_value is not None:
            merged.export_value = cand.export_value
    merged.order = min(int(c.order) for c in ordered)
    return merged


def deduplicate(
    cands: Sequence[FieldCandidate], iou_threshold: float = 0.55
) -> List[FieldCandidate]:
    """Cluster same-page candidates overlapping above ``iou_threshold`` and merge them.

    Clustering is transitive: A overlapping B and B overlapping C puts all three in one
    cluster even when A and C barely touch, because they are all describing one field.
    Nothing is discarded -- see :func:`merge_cluster` for what the survivor inherits.

    Args:
        cands: Candidates from any number of detectors and pages.
        iou_threshold: Strict lower bound; IoU must *exceed* it to cluster.

    Returns:
        One candidate per cluster, in reading order.
    """
    items = [c for c in (cands or []) if c is not None]
    if len(items) < 2:
        return [_clone(c) for c in items]

    ordered = sorted(items, key=_reading_key)
    parent = list(range(len(ordered)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            if a.page != b.page:
                continue
            if a.rect.iou(b.rect) > float(iou_threshold) + EPS:
                union(i, j)

    clusters: Dict[int, List[FieldCandidate]] = {}
    for index, cand in enumerate(ordered):
        clusters.setdefault(find(index), []).append(cand)

    merged = [merge_cluster(clusters[root]) for root in sorted(clusters)]
    merged.sort(key=_reading_key)
    return merged


# ======================================================================================
# Overlap suppression
# ======================================================================================


def _remainder(loser: Rect, winner: Rect) -> Optional[Rect]:
    """Return the largest axis-aligned part of ``loser`` that ``winner`` does not cover.

    Four strips are possible -- left of, right of, below and above the winner -- and the
    biggest one wins, with a fully deterministic tie-break on coordinates.
    """
    a = loser.normalized()
    b = winner.normalized()
    strips: List[Rect] = []
    left = min(a.x1, b.x0)
    if left - a.x0 > EPS:
        strips.append(Rect(a.x0, a.y0, left, a.y1))
    right = max(a.x0, b.x1)
    if a.x1 - right > EPS:
        strips.append(Rect(right, a.y0, a.x1, a.y1))
    below = min(a.y1, b.y0)
    if below - a.y0 > EPS:
        strips.append(Rect(a.x0, a.y0, a.x1, below))
    above = max(a.y0, b.y1)
    if a.y1 - above > EPS:
        strips.append(Rect(a.x0, above, a.x1, a.y1))
    strips = [s for s in strips if s.width > EPS and s.height > EPS]
    if not strips:
        return None
    strips.sort(key=lambda s: (-s.area, s.x0, s.y0, s.x1, s.y1))
    return strips[0]


def _is_viable(rect: Rect, field_type: FieldType, det: DetectionConfig) -> bool:
    """True when a shrunken rectangle is still a usable field of this type."""
    # Signed spans: Rect.width/height are absolute and would hide an inverted rect.
    if (rect.x1 - rect.x0) <= EPS or (rect.y1 - rect.y0) <= EPS:
        return False
    if field_type in (FieldType.CHECKBOX, FieldType.RADIO):
        return rect.width + EPS >= det.checkbox_min_pt and rect.height + EPS >= det.checkbox_min_pt
    return (
        rect.width + EPS >= det.blank_min_width_pt and rect.height + EPS >= det.blank_min_height_pt
    )


def suppress_overlaps(
    cands: Sequence[FieldCandidate], config: Any = None
) -> List[FieldCandidate]:
    """Resolve residual overlaps so no two emitted widgets exceed 10% IoU.

    Candidates are processed strongest first (:func:`fused_score`, then reading order).
    A weaker candidate overlapping an already-accepted one by more than
    :data:`MAX_WIDGET_IOU` is shrunk to the largest part of itself the winner does not
    cover; if what is left is no longer a usable field -- narrower than
    ``blank_min_width_pt``, or shorter than ``blank_min_height_pt``; for checkboxes and
    radios, smaller than ``checkbox_min_pt`` -- it is dropped instead.

    Shrinking can raise the IoU with a *third* rectangle (the union shrinks while the
    intersection does not), so the check repeats until the rectangle is stable, and a
    final verification pass against every accepted rectangle decides whether it may be
    emitted at all.  The 10% invariant therefore holds by construction, not by hope.
    """
    det = _detection(config)
    weights = _scoring(config)
    ordered = rank([c for c in (cands or []) if c is not None], weights)

    kept: List[FieldCandidate] = []
    for cand in ordered:
        rect: Optional[Rect] = cand.rect.normalized()
        for _ in range(_SUPPRESS_MAX_PASSES):
            changed = False
            for other in kept:
                if other.page != cand.page or rect is None:
                    continue
                if rect.iou(other.rect) <= MAX_WIDGET_IOU + EPS:
                    continue
                shrunk = _remainder(rect, other.rect)
                if shrunk is None or not _is_viable(shrunk, cand.field_type, det):
                    LOG.debug(
                        "suppressing candidate %s on page %d: no viable remainder",
                        cand.id,
                        cand.page,
                    )
                    rect = None
                    break
                rect = shrunk
                changed = True
            if rect is None or not changed:
                break
        if rect is None:
            continue
        rect = rect.rounded(ROUND_DIGITS)
        if any(
            o.page == cand.page and rect.iou(o.rect) > MAX_WIDGET_IOU + EPS for o in kept
        ):
            LOG.debug("suppressing candidate %s on page %d: still overlapping", cand.id, cand.page)
            continue
        survivor = _clone(cand)
        survivor.rect = rect
        kept.append(survivor)

    kept.sort(key=_reading_key)
    return kept


# ======================================================================================
# The pipeline
# ======================================================================================


def fuse(
    candidates: Sequence[FieldCandidate],
    config: Any = None,
    primitives_by_page: Optional[Mapping[int, Sequence[VectorPrimitive]]] = None,
    geometry_by_page: Optional[Mapping[int, PageGeometry]] = None,
) -> List[FieldCandidate]:
    """Run the whole fusion stage over a document's raw candidates.

    The order of operations is not arbitrary:

    1. **Normalize** every rectangle and clip it to its page, when a
       :class:`~zfp.core.geometry.PageGeometry` is supplied for that page.
    2. **Snap** the *estimated* candidates to the page's vector primitives, so exact
       geometry beats estimates before anything is compared (skipped when
       ``primitives_by_page`` is ``None``, and skipped per candidate when
       :func:`has_exact_geometry` says the rectangle is already a measurement).
    3. **Calibrate** -- per-type padding and the sanity clamp -- so the rectangles being
       compared are the ones that will actually be written.
    4. **Deduplicate**, merging agreeing detectors and applying the corroboration bonus.
    5. **Score**, folding the weighted evidence into ``confidence.semantic_type``.
    6. **Suppress** the residual overlaps, enforcing the 10% IoU invariant.
    7. **Filter** below ``config.min_candidate_confidence``, measured on
       :meth:`Confidence.overall`.
    8. **Sort** into reading order and renumber ``order``.

    Snapping happens before calibration because padding is a deliberate offset from the
    true geometry; snapping afterwards would try to undo it.  Deduplication happens
    after calibration because two detectors' rectangles agree far better once both have
    been calibrated the same way.

    Args:
        candidates: Raw candidates from any number of detectors and pages.
        config: A ``ZfpConfig`` or :class:`DetectionConfig`; defaults when ``None``.
        primitives_by_page: Vector primitives keyed by page index, for snapping.
        geometry_by_page: Page geometries keyed by page index, for clipping.

    Returns:
        The fused candidates, sorted by ``(page, -rect.y1, rect.x0)``.
    """
    det = _detection(config)
    weights = _scoring(config)
    tol = snap_tolerance(det)

    prepared: List[FieldCandidate] = []
    for cand in candidates or []:
        if cand is None:
            continue
        work = _clone(cand)
        geometry = None
        if geometry_by_page:
            geometry = geometry_by_page.get(int(work.page))
        rect = work.rect.normalized()
        if geometry is not None:
            rect = geometry.clamp(rect)
        if primitives_by_page and not has_exact_geometry(work):
            prims = primitives_by_page.get(int(work.page)) or []
            if prims:
                rect = snap_to_primitive(rect, list(prims), tol)
        rect = calibrate_rect(rect, work.field_type, det)
        if geometry is not None:
            rect = geometry.clamp(rect)
        work.rect = rect.rounded(ROUND_DIGITS)
        prepared.append(work)

    merged = deduplicate(prepared, det.dedup_iou_threshold)
    scored = score_candidates(merged, weights)
    kept = suppress_overlaps(scored, config)

    minimum = float(det.min_candidate_confidence)
    surviving = [c for c in kept if c.confidence.overall() + EPS >= minimum]
    surviving.sort(key=_reading_key)
    for index, cand in enumerate(surviving):
        cand.order = index
    return surviving
