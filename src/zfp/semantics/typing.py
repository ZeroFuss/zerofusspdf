"""Field type inference: refining a rectangle into ``DATE``, ``CURRENCY``, ``EMAIL``...

Geometry decides the *shape* of a field and settles the types that are visible from the
outside: a 10 pt square is a ``CHECKBOX``, a circle in a run is a ``RADIO``, nine equal
cells are a ``COMB``, a long rule under "Signature" is a ``SIGNATURE``.  Those are never
overturned here -- they were seen, not guessed.

What geometry cannot see is whether a plain rule wants a date, an amount, an e-mail
address or three lines of prose.  This module decides that from four independent
readings:

* the label, resolved through :mod:`zfp.ontology` to a :class:`KeySpec` that declares a
  field type ("E-mail Address" -> ``EMAIL``);
* a printed placeholder in or beside the blank (``MM/DD/YYYY`` -> ``DATE``);
* the rectangle's own height (over :data:`MULTILINE_HEIGHT_RATIO` body lines ->
  ``MULTILINE_TEXT``);
* the glyphs immediately around it (a ``$`` on the left -> ``CURRENCY``, a row of ballot
  boxes -> ``CHOICE``).

Signals that agree combine as ``1 - prod(1 - s_i)``: two independent 0.7 readings make
0.91, which is what "two sources agree" should feel like.  Every reading leaves a reason
code so a report can say *why* a field was called a date.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.geometry import EPS, Rect
from ..core.logging import get_logger
from ..core.types import Evidence, EvidenceKind, FieldCandidate, FieldType, TextSpan
from ..ontology import PatternRule, context_lookup
from ..ontology import get as ontology_get
from ..ontology import lookup as ontology_lookup
from ..ontology import match_placeholder
from .graph import DEFAULT_LINE_HEIGHT, detection_config

__all__ = [
    "infer_field_type",
    "infer_types",
    "TypeSignal",
    "nearby_spans",
    "placeholder_near",
    "body_line_height",
    "GEOMETRIC_TYPES",
    "MULTILINE_HEIGHT_RATIO",
    "CHECKBOX_GLYPHS",
    "CHOICE_PHRASES",
    "SOURCE_AGENT",
    "STRENGTH_LABEL",
    "STRENGTH_HEIGHT",
    "STRENGTH_CURRENCY",
    "STRENGTH_CHOICE",
    "STRENGTH_KEYWORD",
]

LOG = get_logger(__name__)

#: Types that geometry establishes by sight; inference refines but never overrules them.
GEOMETRIC_TYPES = (
    FieldType.CHECKBOX,
    FieldType.RADIO,
    FieldType.COMB,
    FieldType.SIGNATURE,
    FieldType.BUTTON,
    FieldType.LISTBOX,
)
#: A blank taller than this many body lines is a paragraph box.
MULTILINE_HEIGHT_RATIO = 2.2
#: Text drawn as a checkbox rather than stroked as a path.
CHECKBOX_GLYPHS: Tuple[str, ...] = ("☐", "☑", "☒", "□", "■", "❑")
#: Label phrasings that mean "pick one of the options printed next to me".
CHOICE_PHRASES: Tuple[str, ...] = (
    "check one",
    "select one",
    "choose one",
    "mark one",
    "check all",
    "select all",
    "circle one",
)
#: How far beside the rectangle a placeholder or a currency glyph may sit, in line heights.
ADJACENT_LINES = 1.0
#: Written into every :class:`~zfp.core.types.Evidence` this module produces.
SOURCE_AGENT = "semantics.typing"

STRENGTH_LABEL = 0.78
STRENGTH_HEIGHT = 0.70
STRENGTH_CURRENCY = 0.80
STRENGTH_CHOICE = 0.65
STRENGTH_KEYWORD = 0.45
#: Confidence reported for a type geometry already established.
GEOMETRY_CONFIDENCE = 0.95
#: Confidence reported when nothing at all speaks about the type.
DEFAULT_CONFIDENCE = 0.40

_KEYWORD_TYPES: Tuple[Tuple[str, FieldType], ...] = (
    (r"\b(?:date|dob|birthday|expiry|expiration|issued)\b", FieldType.DATE),
    (r"\b(?:e[- ]?mail)\b", FieldType.EMAIL),
    (r"\b(?:phone|telephone|mobile|cell|fax)\b", FieldType.PHONE),
    (r"\b(?:amount|total|salary|income|price|fee|payment|balance|cost|wage)\b",
     FieldType.CURRENCY),
    (r"\b(?:number|quantity|qty|count|age|years|percent|percentage)\b", FieldType.NUMBER),
    (r"\b(?:comments?|remarks?|notes?|description|explain|details?|address\s+2)\b",
     FieldType.MULTILINE_TEXT),
)
_KEYWORD_RULES: Tuple[Tuple[Any, FieldType], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), field_type) for pattern, field_type in _KEYWORD_TYPES
)
#: Deterministic tie-break order when two types score identically.
_TYPE_ORDER: Dict[FieldType, int] = {
    field_type: index for index, field_type in enumerate(FieldType)
}


class TypeSignal:
    """One independent reading of a field's type."""

    __slots__ = ("field_type", "strength", "reason")

    def __init__(self, field_type: FieldType, strength: float, reason: str) -> None:
        self.field_type = field_type
        self.strength = min(1.0, max(0.0, float(strength)))
        self.reason = reason

    def as_tuple(self) -> Tuple[str, float, str]:
        """Return ``(type value, strength, reason)``."""
        return (self.field_type.value, self.strength, self.reason)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "TypeSignal(%s, %.2f, %s)" % (self.field_type.value, self.strength, self.reason)


# --------------------------------------------------------------------------- helpers
def body_line_height(spans: Sequence[TextSpan], page: int) -> float:
    """Median text height on ``page``, falling back to the whole input, then 12 pt."""
    for pool in ([s for s in spans if int(s.page) == int(page)], list(spans)):
        heights = [s.rect.height for s in pool if s is not None and s.rect.height > EPS]
        if heights:
            heights.sort()
            middle = len(heights) // 2
            if len(heights) % 2:
                return heights[middle]
            return 0.5 * (heights[middle - 1] + heights[middle])
    return DEFAULT_LINE_HEIGHT


def _inside(rect: Rect, span: TextSpan) -> bool:
    """True when most of ``span`` sits inside ``rect``."""
    if span.rect.area <= EPS:
        return rect.contains_point(span.rect.center)
    overlap = rect.intersection(span.rect)
    return overlap is not None and overlap.area >= 0.6 * span.rect.area


def _adjacent(rect: Rect, span: TextSpan, reach: float) -> bool:
    """True when ``span`` sits just beside ``rect`` on the same line."""
    if span.rect.vertical_overlap(rect) <= EPS:
        return False
    right_gap = span.rect.x0 - rect.x1
    left_gap = rect.x0 - span.rect.x1
    return (-EPS <= right_gap <= reach) or (-EPS <= left_gap <= reach)


def nearby_spans(
    candidate: FieldCandidate, spans: Sequence[TextSpan], reach: float
) -> Tuple[List[TextSpan], List[TextSpan]]:
    """Split the page's spans into those inside the blank and those just beside it."""
    rect = candidate.rect.normalized()
    inside: List[TextSpan] = []
    beside: List[TextSpan] = []
    for span in spans:
        if span is None or span.is_blank() or int(span.page) != int(candidate.page):
            continue
        if _inside(rect, span):
            inside.append(span)
        elif _adjacent(rect, span, reach):
            beside.append(span)
    inside.sort(key=lambda s: (s.rect.x0, -s.rect.y1, s.text))
    beside.sort(key=lambda s: (s.rect.x0, -s.rect.y1, s.text))
    return (inside, beside)


def placeholder_near(
    candidate: FieldCandidate, spans: Sequence[TextSpan], reach: float
) -> Optional[Tuple[PatternRule, TextSpan]]:
    """Return the strongest printed placeholder in or beside the blank, with its span.

    A placeholder inside the rectangle outranks one beside it: ``MM/DD/YYYY`` printed in
    the box is about that box, while the same string one field over might not be.
    """
    inside, beside = nearby_spans(candidate, spans, reach)
    for pool in (inside, beside):
        best: Optional[Tuple[PatternRule, TextSpan]] = None
        for span in pool:
            rule = match_placeholder(span.text)
            if rule is None:
                continue
            if best is None or rule.confidence > best[0].confidence:
                best = (rule, span)
        if best is not None:
            return best
    return None


def _currency_glyph_left(candidate: FieldCandidate, spans: Sequence[TextSpan], reach: float) -> bool:
    """True when a lone currency sign is printed immediately left of the blank."""
    rect = candidate.rect.normalized()
    for span in spans:
        if span is None or int(span.page) != int(candidate.page):
            continue
        text = (span.text or "").strip()
        if text not in ("$", "€", "£", "¥", "US$", "$ "):
            continue
        if span.rect.vertical_overlap(rect) <= EPS:
            continue
        gap = rect.x0 - span.rect.x1
        if -EPS <= gap <= reach:
            return True
    return False


def _choice_signal(
    candidate: FieldCandidate, spans: Sequence[TextSpan], line_h: float
) -> Optional[TypeSignal]:
    """Detect ``[ ] Yes  [ ] No`` printed beside the blank, or a "select one" label."""
    label = (candidate.visible_label or "").lower()
    for phrase in CHOICE_PHRASES:
        if phrase in label:
            return TypeSignal(FieldType.CHOICE, STRENGTH_CHOICE, "label_choice_phrase")
    rect = candidate.rect.normalized()
    band = rect.inflated(0.0, 1.5 * line_h)
    marks = 0
    for span in spans:
        if span is None or int(span.page) != int(candidate.page):
            continue
        text = (span.text or "").strip()
        if not text or not any(text.startswith(glyph) for glyph in CHECKBOX_GLYPHS):
            continue
        if span.rect.vertical_overlap(band) > EPS:
            marks += 1
    if marks >= 2:
        return TypeSignal(FieldType.CHOICE, STRENGTH_CHOICE, "nearby_options_choice")
    return None


def _label_key(candidate: FieldCandidate) -> Optional[str]:
    """Resolve the candidate's visible label to a canonical key, context included."""
    label = candidate.visible_label
    if not label:
        return None
    if candidate.parent_context:
        found = context_lookup(label, list(candidate.parent_context))
        if found is not None:
            return found
    return ontology_lookup(label)


def _combine(signals: Sequence[TypeSignal]) -> float:
    """Noisy-or over independent strengths: ``1 - prod(1 - s)``."""
    remaining = 1.0
    for signal in signals:
        remaining *= 1.0 - signal.strength
    return 1.0 - remaining


# ------------------------------------------------------------------------ inference
def infer_field_type(
    candidate: FieldCandidate,
    spans: Sequence[TextSpan] = (),
    config: Any = None,
) -> Tuple[FieldType, float, List[str]]:
    """Decide what kind of value a candidate wants.

    Args:
        candidate: The field candidate; it is **not** mutated.
        spans: The page's text spans, used for placeholders, currency glyphs and option
            marks.  May be empty, in which case only the label and the geometry speak.
        config: A :class:`~zfp.core.config.ZfpConfig` or
            :class:`~zfp.core.config.DetectionConfig`; optional.

    Returns:
        ``(field_type, confidence, reason_codes)``.  Reason codes look like
        ``"label_ontology_email"``, ``"pattern_date_mdy_placeholder"``,
        ``"height_multiline"``, ``"currency_glyph_left"``.

    Examples:
        >>> from zfp.core.geometry import Rect
        >>> from zfp.core.types import FieldCandidate
        >>> c = FieldCandidate("f", 0, Rect(0, 0, 120, 12), visible_label="E-mail Address")
        >>> kind, _confidence, reasons = infer_field_type(c)
        >>> kind.value, reasons[0]
        ('email', 'label_ontology_email')
    """
    detection_config(config)  # validates / normalizes the config object
    live = [s for s in spans or () if s is not None and not s.is_blank()]
    if candidate.field_type in GEOMETRIC_TYPES:
        confidence = candidate.confidence.geometry or GEOMETRY_CONFIDENCE
        return (
            candidate.field_type,
            round(min(1.0, max(0.0, confidence)), 6),
            ["geometry_%s" % candidate.field_type.value],
        )

    line_h = body_line_height(live, candidate.page)
    reach = ADJACENT_LINES * line_h
    signals: List[TypeSignal] = []

    key = _label_key(candidate)
    if key is not None:
        spec = ontology_get(key)
        if spec is not None and spec.field_type not in (FieldType.UNKNOWN,):
            signals.append(
                TypeSignal(
                    spec.field_type,
                    STRENGTH_LABEL,
                    "label_ontology_%s" % spec.field_type.value,
                )
            )

    found = placeholder_near(candidate, live, reach)
    if found is not None:
        rule, _span = found
        if rule.field_type is not FieldType.UNKNOWN:
            signals.append(
                TypeSignal(rule.field_type, rule.confidence * 0.9, "pattern_%s" % rule.name)
            )

    if candidate.rect.height > MULTILINE_HEIGHT_RATIO * line_h:
        signals.append(TypeSignal(FieldType.MULTILINE_TEXT, STRENGTH_HEIGHT, "height_multiline"))

    if _currency_glyph_left(candidate, live, reach):
        signals.append(TypeSignal(FieldType.CURRENCY, STRENGTH_CURRENCY, "currency_glyph_left"))

    choice = _choice_signal(candidate, live, line_h)
    if choice is not None:
        signals.append(choice)

    if not any(s.field_type is not FieldType.TEXT for s in signals):
        label = candidate.visible_label or ""
        for pattern, field_type in _KEYWORD_RULES:
            if pattern.search(label):
                signals.append(
                    TypeSignal(
                        field_type, STRENGTH_KEYWORD, "label_keyword_%s" % field_type.value
                    )
                )
                break

    if not signals:
        fallback = (
            candidate.field_type
            if candidate.field_type is not FieldType.UNKNOWN
            else FieldType.TEXT
        )
        return (fallback, DEFAULT_CONFIDENCE, ["default_text"])

    by_type: Dict[FieldType, List[TypeSignal]] = {}
    for signal in signals:
        by_type.setdefault(signal.field_type, []).append(signal)

    scored = sorted(
        ((_combine(group), field_type) for field_type, group in by_type.items()),
        key=lambda item: (-item[0], _TYPE_ORDER.get(item[1], 99)),
    )
    confidence, winner = scored[0]
    reasons = [
        signal.reason
        for signal in sorted(by_type[winner], key=lambda s: (-s.strength, s.reason))
    ]
    reasons.extend(
        "rejected_%s" % signal.reason
        for signal in sorted(signals, key=lambda s: (-s.strength, s.reason))
        if signal.field_type is not winner
    )
    return (winner, round(min(1.0, max(0.0, confidence)), 6), reasons)


def infer_types(
    candidates: Sequence[FieldCandidate],
    spans: Sequence[TextSpan] = (),
    config: Any = None,
) -> List[FieldCandidate]:
    """Apply :func:`infer_field_type` to every candidate, in place.

    Sets :attr:`~zfp.core.types.FieldCandidate.field_type` and
    ``confidence.semantic_type``, records a :class:`~zfp.core.types.Evidence` of kind
    ``PATTERN``, and copies a matched placeholder's format hint (and its character
    budget, for a comb) into the candidate's constraints.

    Returns:
        The same candidate objects, in the order they were given.
    """
    live = [s for s in spans or () if s is not None and not s.is_blank()]
    for candidate in candidates or ():
        if candidate is None:
            continue
        field_type, confidence, reasons = infer_field_type(candidate, live, config)
        candidate.field_type = field_type
        candidate.confidence.semantic_type = confidence
        candidate.evidence = [
            e
            for e in candidate.evidence
            if not (e.kind == EvidenceKind.PATTERN and e.source_agent == SOURCE_AGENT)
        ]
        candidate.add_evidence(
            Evidence(
                kind=EvidenceKind.PATTERN,
                score=confidence,
                detail="%s: %s" % (field_type.value, ",".join(reasons)),
                source_agent=SOURCE_AGENT,
                rect=candidate.rect,
            )
        )
        if field_type is FieldType.MULTILINE_TEXT:
            candidate.constraints.multiline = True
        line_h = body_line_height(live, candidate.page)
        found = placeholder_near(candidate, live, ADJACENT_LINES * line_h)
        if found is not None and candidate.constraints.format_hint is None:
            rule, _span = found
            if rule.format_hint:
                candidate.constraints.format_hint = rule.format_hint
    return list(candidates or ())
