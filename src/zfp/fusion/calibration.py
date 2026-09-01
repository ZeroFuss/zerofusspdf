"""Rectangle calibration: turning a *detected* region into a *fillable* rectangle.

A detector reports what it saw -- the extent of a rule, the hole in the ink, the four
sides of a box.  A widget rectangle is a different thing: it is where a viewer will draw
a caret and paint a glyph.  The two can differ by a small, systematic, per-type offset.

**The default offset is zero, and that is a decision, not an omission.**  Every
archetype in :mod:`zfp.candidates` already applies the convention that turns its
measurement into the writable area: a stroked box contributes its *inside* (the ink is
not writable), an underline contributes the band ``DetectionConfig.underline_gap_pt``
above the rule and ``DetectionConfig.field_height_pt`` tall, a check glyph contributes
its own bounding box, a comb contributes the union of its cells.  Those conventions are
what ``zfp.synth`` draws against and what ``docs/QA.md`` measures IoU against.  A
constant, unlearned padding applied on top of them does not refine the rectangle -- it
moves the widget off the geometry that every other module agreed on, and it costs IoU on
every field on the page.  So :data:`DEFAULT_PADDING` is an identity table, and a real
offset has to be *earned* from ground truth by :class:`Calibrator`.

This module owns three separate things, deliberately kept apart:

* :data:`DEFAULT_PADDING` -- the per-type padding table applied by :func:`calibrate`.
  Zero everywhere by default; a deployment that wants a house style edits it, and every
  entry still round-trips through :meth:`FieldPadding.as_dict`.
* :data:`MIN_SIZE` / :data:`MAX_SIZE` -- sanity clamps.  A "field" 700pt tall is not a
  field, it is a detection error, and it must not reach the writer.
* :class:`Calibrator` -- *learned* per-edge corrections fitted against ground truth.
  The research note that "the exact weights should be calibrated rather than hard-coded
  permanently" is made real here: :meth:`Calibrator.fit` measures the mean signed error
  of a corpus of predictions and :meth:`Calibrator.apply` subtracts it back out.

Everything is deterministic and stdlib-only.  A :class:`Calibrator` fitted on nothing is
an exact no-op, so an uncalibrated pipeline behaves precisely as if this class did not
exist -- and, with the identity padding table, so does :func:`calibrate` apart from its
clamps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.config import SCORING_WEIGHT_NAMES, DetectionConfig, ScoringWeights
from ..core.errors import ValidationError
from ..core.geometry import EPS, Rect
from ..core.logging import get_logger
from ..core.types import FieldCandidate, FieldType

__all__ = [
    "DEFAULT_PADDING",
    "MAX_SIZE",
    "MIN_SIZE",
    "ROUND_DIGITS",
    "Calibrator",
    "EdgeAdjustment",
    "FieldPadding",
    "calibrate",
    "calibrate_weights",
    "padding_for",
    "size_bounds",
]

LOG = get_logger(__name__)

#: Every rectangle leaving this module is rounded to this many decimals.
ROUND_DIGITS = 3

#: Learned adjustments are rounded to this many decimals before they are stored, so a
#: serialized calibration round-trips byte-for-byte.
_ADJUST_DIGITS = 6


# ======================================================================================
# Padding
# ======================================================================================


@dataclass(frozen=True)
class FieldPadding:
    """Per-edge outward padding in points.

    Positive values grow the rectangle: ``left`` moves ``x0`` left, ``right`` moves
    ``x1`` right, ``bottom`` moves ``y0`` down and ``top`` moves ``y1`` up (PDF user
    space is y-up).  Negative values shrink it, which is legal and occasionally useful
    for glyph-sized fields.
    """

    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return ``(left, right, top, bottom)``."""
        return (self.left, self.right, self.top, self.bottom)

    def applied(self, rect: Rect) -> Rect:
        """Return ``rect`` grown by this padding (normalized, never inverted)."""
        n = rect.normalized()
        out = Rect(n.x0 - self.left, n.y0 - self.bottom, n.x1 + self.right, n.y1 + self.top)
        # Negative padding must never turn a rectangle inside out.
        if out.x1 < out.x0 or out.y1 < out.y0:
            return n
        return out

    def as_dict(self) -> Dict[str, float]:
        """Return a JSON-ready dictionary."""
        return {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> FieldPadding:
        """Rebuild a :class:`FieldPadding` from :meth:`as_dict` output."""
        return FieldPadding(
            left=float(d.get("left", 0.0) or 0.0),
            right=float(d.get("right", 0.0) or 0.0),
            top=float(d.get("top", 0.0) or 0.0),
            bottom=float(d.get("bottom", 0.0) or 0.0),
        )


_NONE = FieldPadding()

#: Per-type padding applied by :func:`calibrate` before any clamping.
#:
#: Identity by default.  The reasoning that has always been written against
#: ``CHECKBOX`` -- "the glyph *is* the field" -- and against ``COMB`` -- "padding would
#: break the pitch" -- is true of every other type too, because every detector already
#: reports the writable area rather than the ink: a box field is the inside of its
#: rectangle, an underline field is the band above the rule, a table cell is the inside
#: of the grid.  Padding those a second time only pushes the widget back onto the ink.
#: Real per-edge offsets are fitted from ground truth by :class:`Calibrator`.
DEFAULT_PADDING: Dict[FieldType, FieldPadding] = {
    FieldType.TEXT: _NONE,
    FieldType.MULTILINE_TEXT: _NONE,
    FieldType.CHECKBOX: _NONE,
    FieldType.RADIO: _NONE,
    FieldType.CHOICE: _NONE,
    FieldType.LISTBOX: _NONE,
    FieldType.SIGNATURE: _NONE,
    FieldType.DATE: _NONE,
    FieldType.NUMBER: _NONE,
    FieldType.CURRENCY: _NONE,
    FieldType.EMAIL: _NONE,
    FieldType.PHONE: _NONE,
    FieldType.COMB: _NONE,
    FieldType.BUTTON: _NONE,
    FieldType.UNKNOWN: _NONE,
}

#: Smallest sane ``(width, height)`` in points, per type.
MIN_SIZE: Dict[FieldType, Tuple[float, float]] = {
    FieldType.TEXT: (8.0, 6.0),
    FieldType.MULTILINE_TEXT: (24.0, 12.0),
    FieldType.CHECKBOX: (4.0, 4.0),
    FieldType.RADIO: (4.0, 4.0),
    FieldType.CHOICE: (16.0, 8.0),
    FieldType.LISTBOX: (16.0, 12.0),
    FieldType.SIGNATURE: (24.0, 10.0),
    FieldType.DATE: (8.0, 6.0),
    FieldType.NUMBER: (8.0, 6.0),
    FieldType.CURRENCY: (8.0, 6.0),
    FieldType.EMAIL: (8.0, 6.0),
    FieldType.PHONE: (8.0, 6.0),
    FieldType.COMB: (6.0, 6.0),
    FieldType.BUTTON: (8.0, 6.0),
    FieldType.UNKNOWN: (4.0, 4.0),
}

#: Largest sane ``(width, height)`` in points, per type.  Widths are generous (a table
#: row on a landscape page really can be 700pt wide); heights are not, because a tall
#: rectangle is the classic signature of a runaway blank-region detection.
MAX_SIZE: Dict[FieldType, Tuple[float, float]] = {
    FieldType.TEXT: (720.0, 40.0),
    FieldType.MULTILINE_TEXT: (720.0, 400.0),
    FieldType.CHECKBOX: (28.0, 28.0),
    FieldType.RADIO: (28.0, 28.0),
    FieldType.CHOICE: (720.0, 40.0),
    FieldType.LISTBOX: (720.0, 300.0),
    FieldType.SIGNATURE: (500.0, 90.0),
    FieldType.DATE: (300.0, 40.0),
    FieldType.NUMBER: (400.0, 40.0),
    FieldType.CURRENCY: (400.0, 40.0),
    FieldType.EMAIL: (720.0, 40.0),
    FieldType.PHONE: (400.0, 40.0),
    FieldType.COMB: (720.0, 40.0),
    FieldType.BUTTON: (400.0, 60.0),
    FieldType.UNKNOWN: (720.0, 120.0),
}

_FALLBACK_MIN = (4.0, 4.0)
_FALLBACK_MAX = (720.0, 400.0)


def _as_field_type(value: Any) -> FieldType:
    """Coerce ``value`` into a :class:`FieldType`, defaulting to ``UNKNOWN``."""
    if isinstance(value, FieldType):
        return value
    try:
        return FieldType(str(value))
    except (ValueError, TypeError):
        return FieldType.UNKNOWN


def _detection(config: Any) -> DetectionConfig:
    """Coerce ``config`` -- a ``ZfpConfig``, a :class:`DetectionConfig` or ``None`` --
    into a :class:`DetectionConfig`."""
    if isinstance(config, DetectionConfig):
        return config
    inner = getattr(config, "detection", None)
    if isinstance(inner, DetectionConfig):
        return inner
    return DetectionConfig()


def padding_for(field_type: Any) -> FieldPadding:
    """Return the padding for ``field_type`` (``UNKNOWN``'s padding when unmapped)."""
    return DEFAULT_PADDING.get(_as_field_type(field_type), DEFAULT_PADDING[FieldType.UNKNOWN])


def size_bounds(
    field_type: Any, config: Any = None
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return ``((min_w, min_h), (max_w, max_h))`` for ``field_type``.

    Checkbox and radio bounds are taken from :class:`DetectionConfig` when one is
    supplied, so the clamp agrees with the detector that produced the candidate.
    """
    ft = _as_field_type(field_type)
    lo = MIN_SIZE.get(ft, _FALLBACK_MIN)
    hi = MAX_SIZE.get(ft, _FALLBACK_MAX)
    if ft in (FieldType.CHECKBOX, FieldType.RADIO) and config is not None:
        det = _detection(config)
        lo = (min(lo[0], det.checkbox_min_pt), min(lo[1], det.checkbox_min_pt))
        hi = (max(hi[0], det.checkbox_max_pt), max(hi[1], det.checkbox_max_pt))
    return lo, hi


def _clamp_span(
    lo: float, hi: float, minimum: float, maximum: float, anchor: str
) -> Tuple[float, float]:
    """Clamp the 1-D span ``[lo, hi]`` into ``[minimum, maximum]`` length.

    Growth is always symmetric about the centre -- there is no information telling us
    which side is short.  Shrinking honours ``anchor``: ``"center"`` keeps the midpoint,
    ``"high"`` keeps ``hi`` fixed, ``"low"`` keeps ``lo`` fixed.
    """
    span = hi - lo
    if span < minimum - EPS:
        mid = 0.5 * (lo + hi)
        half = 0.5 * minimum
        return mid - half, mid + half
    if span > maximum + EPS:
        if anchor == "high":
            return hi - maximum, hi
        if anchor == "low":
            return lo, lo + maximum
        mid = 0.5 * (lo + hi)
        half = 0.5 * maximum
        return mid - half, mid + half
    return lo, hi


def calibrate(rect: Rect, field_type: Any, config: Any = None) -> Rect:
    """Return ``rect`` padded for ``field_type`` and clamped to a sane size.

    Padding comes first (:data:`DEFAULT_PADDING`), the clamp second
    (:data:`MIN_SIZE` / :data:`MAX_SIZE`).  Undersized rectangles grow about their
    centre.  Oversized ones shrink about their centre horizontally, but keep their
    **top** edge vertically: a spuriously tall region is nearly always a blank-region
    detection that ran downwards past the end of the field into the whitespace below
    it, so the top edge is the trustworthy one.
    """
    ft = _as_field_type(field_type)
    padded = padding_for(ft).applied(rect)
    (min_w, min_h), (max_w, max_h) = size_bounds(ft, config)
    x0, x1 = _clamp_span(padded.x0, padded.x1, min_w, max_w, "center")
    y0, y1 = _clamp_span(padded.y0, padded.y1, min_h, max_h, "high")
    return Rect(x0, y0, x1, y1).rounded(ROUND_DIGITS)


# ======================================================================================
# Learned per-edge calibration
# ======================================================================================


@dataclass(frozen=True)
class EdgeAdjustment:
    """The mean signed error ``predicted - truth`` of each of the four edges."""

    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    count: int = 0

    def is_zero(self) -> bool:
        """True when the adjustment would not move a rectangle at all."""
        return (
            abs(self.x0) <= EPS
            and abs(self.y0) <= EPS
            and abs(self.x1) <= EPS
            and abs(self.y1) <= EPS
        )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return ``(x0, y0, x1, y1)``."""
        return (self.x0, self.y0, self.x1, self.y1)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "count": self.count}

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> EdgeAdjustment:
        """Rebuild an :class:`EdgeAdjustment` from :meth:`as_dict` output."""
        return EdgeAdjustment(
            x0=float(d.get("x0", 0.0) or 0.0),
            y0=float(d.get("y0", 0.0) or 0.0),
            x1=float(d.get("x1", 0.0) or 0.0),
            y1=float(d.get("y1", 0.0) or 0.0),
            count=int(d.get("count", 0) or 0),
        )


def _split_pair(item: Any) -> Tuple[Optional[FieldType], Optional[Rect]]:
    """Extract ``(field_type, rect)`` from a Rect / candidate / ``(type, rect)`` pair."""
    if item is None:
        return None, None
    if isinstance(item, Rect):
        return None, item
    if isinstance(item, FieldCandidate):
        return _as_field_type(item.field_type), item.rect
    if isinstance(item, (tuple, list)) and len(item) == 2:
        a, b = item
        if isinstance(a, Rect) and not isinstance(b, Rect):
            return _as_field_type(b), a
        if isinstance(b, Rect):
            return _as_field_type(a), b
        return None, None
    rect = getattr(item, "rect", None)
    if isinstance(rect, Rect):
        return _as_field_type(getattr(item, "field_type", FieldType.UNKNOWN)), rect
    return None, None


@dataclass
class Calibrator:
    """Learned per-edge corrections, fitted from a corpus and applied to new rectangles.

    ``adjustments`` maps a :class:`FieldType` to the mean signed error of predictions of
    that type; ``overall`` is the same statistic pooled over every type, used as the
    fallback for a type the corpus never contained.  Both are ``predicted - truth``, so
    :meth:`apply` **subtracts** them.

    A default-constructed (or empty-fitted) calibrator is an exact no-op.
    """

    adjustments: Dict[FieldType, EdgeAdjustment] = field(default_factory=dict)
    overall: EdgeAdjustment = field(default_factory=EdgeAdjustment)

    # ---------------------------------------------------------------------- fitting
    @classmethod
    def fit(
        cls,
        pred_rects: Sequence[Any],
        true_rects: Sequence[Any],
        field_types: Optional[Sequence[Any]] = None,
    ) -> Calibrator:
        """Fit a calibrator from paired predictions and ground truth.

        ``pred_rects`` and ``true_rects`` are parallel sequences.  Each element may be a
        :class:`~zfp.core.geometry.Rect`, a :class:`~zfp.core.types.FieldCandidate`, a
        ``(field_type, rect)`` pair or any object exposing ``.rect``.  The field type of
        a pair is taken from the prediction, then from the truth, then from the optional
        ``field_types`` sequence, then falls back to ``UNKNOWN``.

        Raises:
            ValidationError: when the two sequences have different lengths.
        """
        preds = list(pred_rects or [])
        truths = list(true_rects or [])
        if len(preds) != len(truths):
            raise ValidationError(
                "Calibrator.fit needs parallel sequences (%d predictions, %d truths)"
                % (len(preds), len(truths))
            )
        types = list(field_types or [])

        sums: Dict[FieldType, List[float]] = {}
        counts: Dict[FieldType, int] = {}
        pooled = [0.0, 0.0, 0.0, 0.0]
        pooled_n = 0

        for index, (pred_item, true_item) in enumerate(zip(preds, truths)):
            pred_ft, pred_rect = _split_pair(pred_item)
            true_ft, true_rect = _split_pair(true_item)
            if pred_rect is None or true_rect is None:
                continue
            ft = pred_ft or true_ft
            if ft is None and index < len(types):
                ft = _as_field_type(types[index])
            if ft is None:
                ft = FieldType.UNKNOWN
            p = pred_rect.normalized()
            t = true_rect.normalized()
            deltas = (p.x0 - t.x0, p.y0 - t.y0, p.x1 - t.x1, p.y1 - t.y1)
            bucket = sums.setdefault(ft, [0.0, 0.0, 0.0, 0.0])
            for axis in range(4):
                bucket[axis] += deltas[axis]
                pooled[axis] += deltas[axis]
            counts[ft] = counts.get(ft, 0) + 1
            pooled_n += 1

        adjustments: Dict[FieldType, EdgeAdjustment] = {}
        # Sorted by the enum's value so the mapping's iteration order is deterministic.
        for ft in sorted(sums, key=lambda f: f.value):
            n = counts[ft]
            values = [round(v / n, _ADJUST_DIGITS) for v in sums[ft]]
            adjustments[ft] = EdgeAdjustment(*values, count=n)
        if pooled_n:
            pooled_values = [round(v / pooled_n, _ADJUST_DIGITS) for v in pooled]
            overall = EdgeAdjustment(*pooled_values, count=pooled_n)
        else:
            overall = EdgeAdjustment()
        return cls(adjustments=adjustments, overall=overall)

    # ---------------------------------------------------------------------- applying
    def adjustment_for(self, field_type: Any) -> EdgeAdjustment:
        """Return the adjustment used for ``field_type`` (pooled when unseen)."""
        ft = _as_field_type(field_type)
        found = self.adjustments.get(ft)
        if found is not None:
            return found
        return self.overall

    def apply(self, rect: Rect, field_type: Any = FieldType.UNKNOWN) -> Rect:
        """Return ``rect`` with the learned error subtracted from every edge.

        Returns ``rect`` unchanged when nothing was learned for the type (or at all),
        and when the correction would invert the rectangle.
        """
        adj = self.adjustment_for(field_type)
        if adj.count <= 0 or adj.is_zero():
            return rect
        n = rect.normalized()
        out = Rect(n.x0 - adj.x0, n.y0 - adj.y0, n.x1 - adj.x1, n.y1 - adj.y1)
        # Signed spans: Rect.width/height are absolute and would hide an inversion.
        if (out.x1 - out.x0) <= EPS or (out.y1 - out.y0) <= EPS:
            LOG.debug("calibration adjustment would invert %s; skipped", n.as_list())
            return rect
        return out.rounded(ROUND_DIGITS)

    def is_empty(self) -> bool:
        """True when this calibrator cannot move any rectangle."""
        if self.overall.count > 0 and not self.overall.is_zero():
            return False
        return all(a.count <= 0 or a.is_zero() for a in self.adjustments.values())

    # ----------------------------------------------------------------- serialization
    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "adjustments": {
                ft.value: adj.as_dict()
                for ft, adj in sorted(self.adjustments.items(), key=lambda kv: kv[0].value)
            },
            "overall": self.overall.as_dict(),
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Calibrator:
        """Rebuild a :class:`Calibrator` from :meth:`as_dict` output."""
        raw = d.get("adjustments") or {}
        adjustments: Dict[FieldType, EdgeAdjustment] = {}
        for key in sorted(raw, key=str):
            adjustments[_as_field_type(key)] = EdgeAdjustment.from_dict(raw[key] or {})
        overall = EdgeAdjustment.from_dict(d.get("overall") or {})
        return Calibrator(adjustments=adjustments, overall=overall)


# ======================================================================================
# Weight calibration
# ======================================================================================


def _truth_by_page(truth: Iterable[Any]) -> Dict[int, List[Rect]]:
    """Group ground-truth rectangles by page (a bare ``Rect`` is assumed page 0)."""
    out: Dict[int, List[Rect]] = {}
    for item in truth or []:
        if item is None:
            continue
        page = 0
        rect: Optional[Rect] = None
        if isinstance(item, Rect):
            rect = item
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            a, b = item
            if isinstance(b, Rect):
                page = int(a)
                rect = b
            elif isinstance(a, Rect):
                rect = a
                page = int(b)
        else:
            maybe = getattr(item, "rect", None)
            if isinstance(maybe, Rect):
                rect = maybe
                page = int(getattr(item, "page", 0) or 0)
        if rect is None:
            continue
        out.setdefault(page, []).append(rect.normalized())
    return out


def _f1(
    scored: Sequence[Tuple[FieldCandidate, Dict[str, float]]],
    truth_pages: Mapping[int, List[Rect]],
    weights: ScoringWeights,
    accept_threshold: float,
    iou_threshold: float,
) -> float:
    """Return the F1 of the candidates ``weights`` accepts, matched against truth."""
    accepted: Dict[int, List[Tuple[float, FieldCandidate]]] = {}
    for cand, buckets in scored:
        value = weights.score(buckets)
        if value + EPS < accept_threshold:
            continue
        accepted.setdefault(int(cand.page), []).append((value, cand))

    tp = 0
    fp = 0
    truth_total = sum(len(v) for v in truth_pages.values())
    for page in sorted(set(list(accepted.keys()) + list(truth_pages.keys()))):
        picks = accepted.get(page, [])
        # Strongest first; ties resolved in reading order so matching is deterministic.
        picks.sort(key=lambda pair: (-pair[0], -pair[1].rect.y1, pair[1].rect.x0, pair[1].id))
        available = list(truth_pages.get(page, []))
        used = [False] * len(available)
        for _value, cand in picks:
            best_index = -1
            best_iou = 0.0
            for index, gt in enumerate(available):
                if used[index]:
                    continue
                overlap = cand.rect.iou(gt)
                if overlap > best_iou + EPS:
                    best_iou = overlap
                    best_index = index
            if best_index >= 0 and best_iou + EPS >= iou_threshold:
                used[best_index] = True
                tp += 1
            else:
                fp += 1
    fn = truth_total - tp
    denom = 2 * tp + fp + fn
    if denom <= 0:
        return 0.0
    return (2.0 * tp) / denom


def calibrate_weights(
    candidates: Sequence[FieldCandidate],
    truth: Sequence[Any],
    base_weights: Optional[ScoringWeights] = None,
    *,
    accept_threshold: float = 0.35,
    iou_threshold: float = 0.5,
    step: float = 0.05,
    passes: int = 3,
) -> ScoringWeights:
    """Tune the seven scoring weights by deterministic coordinate ascent on F1.

    Each pass walks the seven weights in contract order and tries ``+step`` and
    ``-step`` on each, renormalizing before every evaluation (only the *ratios* of the
    weights matter, since :meth:`ScoringWeights.score` normalizes).  A move is taken
    only on a **strict** F1 improvement, which is what makes the result both
    deterministic and monotone: the returned weights never score worse than
    ``base_weights`` on the corpus they were fitted to.

    At most ``passes * 7 * 2`` evaluations are performed -- 42 by default -- so the
    routine is bounded regardless of corpus size.  There is no randomness anywhere.

    Args:
        candidates: Scored-in-evidence candidates from the corpus.
        truth: Ground-truth fields; rectangles, candidates or ``(page, rect)`` pairs.
        base_weights: Starting point; the contract defaults when omitted.
        accept_threshold: Score above which a candidate counts as *detected*.
        iou_threshold: IoU above which a candidate matches a truth rectangle.
        step: Coordinate-ascent step size.
        passes: Number of full sweeps over the seven weights.

    Returns:
        A normalized :class:`ScoringWeights` (its seven weights sum to 1.0).
    """
    current = (base_weights or ScoringWeights()).normalized()
    scored = [(c, c.evidence_scores()) for c in candidates or [] if c is not None]
    truth_pages = _truth_by_page(truth)
    if not scored or not truth_pages:
        return current

    best_score = _f1(scored, truth_pages, current, accept_threshold, iou_threshold)
    step = abs(float(step))
    passes = max(0, int(passes))
    for _ in range(passes):
        improved = False
        for name in SCORING_WEIGHT_NAMES:
            base_value = getattr(current, name)
            for delta in (step, -step):
                trial_value = base_value + delta
                if trial_value < 0.0:
                    continue
                trial = ScoringWeights(
                    **{
                        key: (trial_value if key == name else getattr(current, key))
                        for key in SCORING_WEIGHT_NAMES
                    }
                ).normalized()
                value = _f1(scored, truth_pages, trial, accept_threshold, iou_threshold)
                if value > best_score + EPS:
                    best_score = value
                    current = trial
                    improved = True
                    break
        if not improved:
            break
    return current.normalized()
