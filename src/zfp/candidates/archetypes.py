"""The eleven form archetypes: geometry in, :class:`FieldCandidate` out.

This is the heart of automatic form detection.  Every detector is a small, independent
class with a ``name`` and a ``detect(ctx)`` method; each one recognises exactly one way
that paper forms ask for input:

============================  ======================================================
:class:`UnderlineFieldDetector`   ``Label: ______``
:class:`BoxFieldDetector`         ``Label [__________]``
:class:`CheckboxDetector`         ``[ ] Yes``
:class:`RadioGroupDetector`       ``Status:  ( ) Single  ( ) Married``
:class:`CombFieldDetector`        ``[ ][ ][ ][ ][ ][ ][ ][ ][ ]``
:class:`DateBoxDetector`          ``[  ] / [  ] / [    ]``
:class:`SignatureLineDetector`    a rule captioned "Signature"
:class:`TableCellDetector`        the empty cells of a ruled grid
:class:`BlankRegionDetector`      borderless whitespace next to a label
:class:`FreeTextAreaDetector`     a large empty block
:class:`ColonRunDetector`         ``Name: ........`` with no vector geometry at all
============================  ======================================================

Rules every detector obeys:

* the id comes from :func:`zfp.core.ids.candidate_id`, so two detectors that agree on a
  rectangle agree on an id;
* the evidence is real -- the right :class:`EvidenceKind` with a real score and the
  rectangle it came from -- and ``confidence.geometry`` reflects how hard that evidence
  is (an exact vector rule is 0.99, an inferred blank region is 0.55-0.75);
* ``canonical_key`` is **never** set here.  That is the semantics stage's job.
  ``visible_label`` *is* set whenever the label was part of the geometry that produced
  the candidate, and ``field_type`` whenever the geometry itself settles it;
* anything overlapping a widget that already exists on the page is skipped.

Deduplication is deliberately **not** done here: overlapping proposals from different
archetypes are evidence, and :mod:`zfp.fusion` owns resolving them.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.config import DetectionConfig
from ..core.geometry import EPS, Rect
from ..core.ids import candidate_id, stable_id
from ..core.logging import get_logger
from ..core.types import (
    Evidence,
    EvidenceKind,
    FieldCandidate,
    FieldType,
    TextSpan,
    VectorPrimitive,
)
from ..ontology import match_placeholder, normalize_label
from .context import (
    CandidateContext,
    build_context,
    span_size,
    vision_call,
)

# Imported at module scope so the detectors can use the extra helpers the vision layer
# offers beyond the contract; every call still goes through the defensive wrappers.
try:  # pragma: no cover - depends on which backend is installed
    from ..vision import primitives as vision_primitives  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    vision_primitives = None  # type: ignore[assignment]

LOG = get_logger(__name__)

__all__ = [
    "CandidateContext",
    "build_context",
    "ArchetypeDetector",
    "UnderlineFieldDetector",
    "BoxFieldDetector",
    "CheckboxDetector",
    "RadioGroupDetector",
    "CombFieldDetector",
    "DateBoxDetector",
    "SignatureLineDetector",
    "TableCellDetector",
    "BlankRegionDetector",
    "FreeTextAreaDetector",
    "ColonRunDetector",
    "DEFAULT_DETECTORS",
    "generate_candidates",
    "clean_label",
    "export_value_for",
    "find_label",
    "label_above",
    "label_below",
    "label_left",
    "label_right",
    "looks_like_date",
    "looks_like_signature",
    "max_chars_for",
    "new_candidate",
    "type_from_label",
]

try:  # Protocol landed in 3.8; the fallback keeps 3.9 type checking happy regardless.
    from typing import Protocol

    class ArchetypeDetector(Protocol):
        """One form archetype: a name and a detection pass over a page."""

        name: str

        def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
            """Return every candidate this archetype recognises on ``ctx``'s page."""
            ...

except ImportError:  # pragma: no cover - Python without typing.Protocol
    ArchetypeDetector = object  # type: ignore[misc,assignment]


# ------------------------------------------------------------------------------ tuning
#: A rule at least this fraction of the page wide is a separator, never a field.
SEPARATOR_PAGE_FRACTION = 0.92
#: Synthesized field height is at least this multiple of the local body font size.
#:
#: ``DetectionConfig.field_height_pt`` is the contract's one number for "the synthesized
#: height above an underline", and it is what ``zfp.synth`` draws its ground truth with,
#: so the font term must never *exceed* it for ordinary body type -- a ratio above 1.0
#: would inflate every field on a 9-12pt page and cost IoU on all of them.  At 1.0 the
#: term does exactly the job it is there for and nothing else: a page set in a face
#: larger than ``field_height_pt`` still gets a field tall enough to hold its own type.
FIELD_HEIGHT_FONT_RATIO = 1.0
#: A box taller than this many body lines is a multi-line entry area.
MULTILINE_HEIGHT_RATIO = 2.2
#: How far above/below a rule a signature caption may sit.
SIGNATURE_LABEL_GAP_PT = 24.0
#: How far under a rule a *caption* may start, as a fraction of the body face.  A
#: caption is set immediately beneath its rule -- a descender's worth of air, no more --
#: and the number has to stay under a full line height or the first ordinary row of the
#: page, printed a line below a running head, reads as one.
CAPTION_BELOW_GAP_RATIO = 0.85
#: A blank region already claimed by harder geometry at this IoU is dropped...
CLAIMED_IOU = 0.30
#: ...as is one this much of whose own area a harder rectangle already covers.
CLAIMED_COVERAGE = 0.35
#: More than this many label-anchored entry blanks inside a void means the void is the
#: gutter between two rows of borderless fields, not one answer box.
FREE_TEXT_MAX_ANCHORED = 1
#: A free-text area is at least this many line heights tall...
FREE_TEXT_MIN_LINES = 3.0
#: ...and at least this fraction of the page wide...
FREE_TEXT_MIN_WIDTH_FRACTION = 0.40
#: ...and no more than this fraction of the page's area, or it is just an empty sheet.
FREE_TEXT_MAX_PAGE_FRACTION = 0.45
#: Words whose presence in a label means "sign here".
SIGNATURE_WORDS = frozenset(
    {"sign", "signs", "signed", "signature", "signatures", "autograph", "esignature"}
)
#: Words whose presence in a label means "a date goes here".
DATE_WORDS = frozenset({"date", "dated", "dob", "birthdate", "birthday"})
#: Normalized label text that maps onto a conventional export value.
_EXPORT_ALIASES: Dict[str, str] = {
    "yes": "Yes",
    "y": "Yes",
    "true": "Yes",
    "checked": "Yes",
    "on": "On",
    "x": "X",
    "no": "No",
    "n": "No",
    "false": "No",
    "unchecked": "Off",
    "off": "Off",
    "na": "NA",
    "n a": "NA",
}

_TRAILING_DECORATION = re.compile(r"[\s:;.…·•*–—_\-]+$")
_LEADING_DECORATION = re.compile(r"^[\s•·*\-–—]+")
_WHITESPACE = re.compile(r"\s+")
_LEADER_RUN = re.compile(r"(_{3,}|\.{4,}|…{2,}|(?:\.\s){3,}\.?|·{4,}|-{6,})")
_NON_WORD = re.compile(r"[^0-9A-Za-z]+")


# ----------------------------------------------------------------------------- helpers
def clean_label(text: Optional[str]) -> str:
    """Strip a printed label down to its human form.

    ``"  First Name: "`` -> ``"First Name"``, ``"Signature ____"`` -> ``"Signature"``.
    Case and internal punctuation are preserved: this is the label a person reads, not
    a normalized ontology key.
    """
    if not text:
        return ""
    stripped = _WHITESPACE.sub(" ", str(text)).strip()
    stripped = _LEADING_DECORATION.sub("", stripped)
    stripped = _TRAILING_DECORATION.sub("", stripped)
    return stripped.strip()


def _tokens(label: str) -> List[str]:
    """Ontology-normalized word tokens of a label."""
    return [t for t in normalize_label(label or "").split() if t]


def looks_like_signature(label: Optional[str]) -> bool:
    """True when ``label`` asks for a signature or initials.

    Matching is per token, so ``"Design Review"`` is not a signature line while
    ``"Signature of Applicant"`` and ``"Initials"`` are.
    """
    for token in _tokens(label or ""):
        if token in SIGNATURE_WORDS or token.startswith("signat") or token.startswith("initial"):
            return True
    return False


def looks_like_date(label: Optional[str]) -> bool:
    """True when ``label`` asks for a date (token match, so ``"Candidate"`` does not)."""
    text = label or ""
    for token in _tokens(text):
        if token in DATE_WORDS:
            return True
    rule = match_placeholder(text)
    return rule is not None and rule.field_type is FieldType.DATE


def type_from_label(label: Optional[str], default: FieldType = FieldType.TEXT) -> FieldType:
    """The field type a label settles on its own.

    Only the two cases the *geometry* stage is entitled to decide are handled: a
    signature caption and a date caption.  Everything else stays ``default`` and is left
    to :mod:`zfp.semantics`.
    """
    if looks_like_signature(label):
        return FieldType.SIGNATURE
    if looks_like_date(label):
        return FieldType.DATE
    return default


def export_value_for(label: Optional[str], fallback: str = "On") -> str:
    """Turn an option label into a PDF export value.

    ``"Yes"`` -> ``"Yes"``, ``"n"`` -> ``"No"``, ``"Married Filing Jointly"`` ->
    ``"Married_Filing_Jointly"``.  Never empty.
    """
    cleaned = clean_label(label)
    if not cleaned:
        return fallback
    normalized = normalize_label(cleaned)
    alias = _EXPORT_ALIASES.get(normalized)
    if alias is not None:
        return alias
    slug = _NON_WORD.sub("_", cleaned).strip("_")
    if not slug:
        return fallback
    return slug[:64]


def _label_confidence(gap: float, max_gap: float) -> float:
    """Map a label-to-field distance onto a 0.4..0.95 confidence."""
    if max_gap <= 0.0:
        return 0.4
    ratio = min(max(gap, 0.0) / max_gap, 1.0)
    return round(max(0.40, 0.95 - 0.55 * ratio), 4)


def _overlap_ok(a: Rect, b: Rect, axis: str, ratio: float = 0.2) -> bool:
    """True when ``a`` and ``b`` share enough extent on one axis to be associated."""
    if axis == "vertical":
        shared = a.vertical_overlap(b)
        span = min(a.height, b.height)
    else:
        shared = a.horizontal_overlap(b)
        span = min(a.width, b.width)
    if shared <= 0.0:
        return False
    if span <= 0.0:
        return True
    return shared >= ratio * span


def label_left(
    ctx: CandidateContext, rect: Rect, max_distance: Optional[float] = None
) -> Optional[Tuple[TextSpan, float]]:
    """Nearest visible span ending to the left of ``rect`` on the same row."""
    limit = ctx.detection.label_max_distance_pt if max_distance is None else float(max_distance)
    probe = Rect(rect.x0 - limit, rect.y0 - 2.0, rect.x0 + 1.0, rect.y1 + 2.0)
    best: Optional[Tuple[TextSpan, float]] = None
    for span in ctx.spans_near(probe, 1.0):
        if span.rect.x1 > rect.x0 + 1.5:
            continue
        gap = rect.x0 - span.rect.x1
        if gap > limit:
            continue
        if not _overlap_ok(span.rect, rect, "vertical"):
            continue
        if best is None or gap < best[1] - EPS:
            best = (span, max(gap, 0.0))
    return best


def label_right(
    ctx: CandidateContext, rect: Rect, max_distance: Optional[float] = None
) -> Optional[Tuple[TextSpan, float]]:
    """Nearest visible span starting to the right of ``rect`` on the same row."""
    limit = ctx.detection.label_max_distance_pt if max_distance is None else float(max_distance)
    probe = Rect(rect.x1 - 1.0, rect.y0 - 2.0, rect.x1 + limit, rect.y1 + 2.0)
    best: Optional[Tuple[TextSpan, float]] = None
    for span in ctx.spans_near(probe, 1.0):
        if span.rect.x0 < rect.x1 - 1.5:
            continue
        gap = span.rect.x0 - rect.x1
        if gap > limit:
            continue
        if not _overlap_ok(span.rect, rect, "vertical"):
            continue
        if best is None or gap < best[1] - EPS:
            best = (span, max(gap, 0.0))
    return best


def label_above(
    ctx: CandidateContext, rect: Rect, max_gap: Optional[float] = None
) -> Optional[Tuple[TextSpan, float]]:
    """Nearest visible span sitting directly above ``rect``."""
    limit = 1.6 * ctx.body_font_size if max_gap is None else float(max_gap)
    probe = Rect(rect.x0 - 2.0, rect.y1 - 1.0, rect.x1 + 2.0, rect.y1 + limit)
    best: Optional[Tuple[TextSpan, float]] = None
    for span in ctx.spans_near(probe, 1.0):
        if span.rect.y0 < rect.y1 - 1.0:
            continue
        gap = span.rect.y0 - rect.y1
        if gap > limit:
            continue
        if not _overlap_ok(span.rect, rect, "horizontal", 0.15):
            continue
        if best is None or gap < best[1] - EPS or (
            abs(gap - best[1]) <= EPS and span.rect.x0 < best[0].rect.x0
        ):
            best = (span, max(gap, 0.0))
    return best


def label_below(
    ctx: CandidateContext, rect: Rect, max_gap: Optional[float] = None
) -> Optional[Tuple[TextSpan, float]]:
    """Nearest visible span sitting directly below ``rect``."""
    limit = 1.6 * ctx.body_font_size if max_gap is None else float(max_gap)
    probe = Rect(rect.x0 - 2.0, rect.y0 - limit, rect.x1 + 2.0, rect.y0 + 1.0)
    best: Optional[Tuple[TextSpan, float]] = None
    for span in ctx.spans_near(probe, 1.0):
        if span.rect.y1 > rect.y0 + 1.0:
            continue
        gap = rect.y0 - span.rect.y1
        if gap > limit:
            continue
        if not _overlap_ok(span.rect, rect, "horizontal", 0.15):
            continue
        if best is None or gap < best[1] - EPS or (
            abs(gap - best[1]) <= EPS and span.rect.x0 < best[0].rect.x0
        ):
            best = (span, max(gap, 0.0))
    return best


#: Default order in which a field looks for its printed label.  Left first because that
#: is where forms put it; then *below*, which is the signature-block idiom (a rule with
#: its caption underneath); then above, for stacked label/entry layouts.
DEFAULT_LABEL_ORDER: Tuple[str, ...] = ("left", "below", "above")


def find_label(
    ctx: CandidateContext,
    rect: Rect,
    order: Sequence[str] = DEFAULT_LABEL_ORDER,
    vertical_gap: Optional[float] = None,
    horizontal_gap: Optional[float] = None,
) -> Tuple[Optional[Tuple[TextSpan, float]], str]:
    """Look for ``rect``'s label in each direction of ``order``, nearest first.

    Returns the ``(span, gap)`` pair and a human description of where it was found, or
    ``(None, "")`` when nothing plausible is adjacent.
    """
    limit = 1.8 * ctx.body_font_size if vertical_gap is None else float(vertical_gap)
    for direction in order:
        if direction == "left":
            found = label_left(ctx, rect, horizontal_gap)
        elif direction == "right":
            found = label_right(ctx, rect, horizontal_gap)
        elif direction == "above":
            found = label_above(ctx, rect, limit)
        elif direction == "below":
            found = label_below(ctx, rect, limit)
        else:  # pragma: no cover - programming error
            continue
        if found is not None and clean_label(found[0].text):
            return found, "label %s" % direction
    return None, ""


def _average_char_width(font_name: str, size: float) -> float:
    """Mean glyph advance for a font at ``size``, used to estimate a character budget."""
    if size <= 0.0:
        return 5.0
    sample = "abcdefghijklmnopqrstuvwxyz "
    try:
        from ..pdfio.fonts import text_width  # local: keeps import cost off the hot path

        width = float(text_width(sample, font_name or "Helvetica", size))
        if width > 0.0:
            return width / len(sample)
    except Exception:  # pragma: no cover - defensive
        pass
    return 0.5 * size


def max_chars_for(width: float, font_size: float, font_name: str = "Helvetica") -> Optional[int]:
    """Estimate how many characters fit in ``width`` points."""
    average = _average_char_width(font_name, font_size)
    if average <= 0.0 or width <= 0.0:
        return None
    return max(1, int(width / average))


def _cached(ctx: CandidateContext, key: str, factory: Callable[[], Any]) -> Any:
    """Memoize a shared derivation on the context for the duration of one run."""
    if key not in ctx.cache:
        ctx.cache[key] = factory()
    return ctx.cache[key]


def new_candidate(
    ctx: CandidateContext,
    rect: Rect,
    kind: str,
    field_type: FieldType = FieldType.UNKNOWN,
    min_width: float = 4.0,
    min_height: float = 4.0,
) -> Optional[FieldCandidate]:
    """Create an empty candidate for ``rect``, or ``None`` when the rectangle is unusable.

    The rectangle is normalized and clipped to the crop box; a candidate that would be
    degenerate, off-page or on top of an existing widget is refused here rather than in
    each of the eleven detectors.
    """
    clipped = ctx.clamp(rect)
    if clipped.width < min_width or clipped.height < min_height:
        return None
    if ctx.blocked_by_widget(clipped):
        return None
    return FieldCandidate(
        id=candidate_id(ctx.page, clipped, kind),
        page=ctx.page,
        rect=clipped,
        field_type=field_type,
    )


def _attach_label(
    candidate: FieldCandidate,
    found: Optional[Tuple[TextSpan, float]],
    max_gap: float,
    detail: str,
) -> str:
    """Attach a found label to ``candidate`` and return its cleaned text."""
    if found is None:
        return ""
    span, gap = found
    text = clean_label(span.text)
    if not text:
        return ""
    candidate.visible_label = text
    candidate.confidence.label_link = _label_confidence(gap, max_gap)
    candidate.add_evidence(
        Evidence(
            kind=EvidenceKind.LABEL_LINK,
            score=candidate.confidence.label_link,
            detail="%s: %r" % (detail, text),
            rect=span.rect,
        )
    )
    return text


def _text_evidence_kind(span: TextSpan) -> EvidenceKind:
    """``OCR_TEXT`` for a recognised span, ``NATIVE_TEXT`` for a parsed one."""
    return EvidenceKind.OCR_TEXT if span.source == "ocr" else EvidenceKind.NATIVE_TEXT


# --------------------------------------------------------------- shared page structure
def field_height(ctx: CandidateContext, near: Rect) -> float:
    """Height of a synthesized entry line near ``near``.

    :attr:`DetectionConfig.field_height_pt` is the convention -- it is the height
    ``zfp.synth`` draws its ground truth with and the height the whole perception layer
    is measured against -- and on any page whose body type fits inside it, it is
    returned unchanged.  It stops being enough only when the page is set in a face
    taller than the configured height, and then the local font size takes over so the
    writing space still matches the printing.  See :data:`FIELD_HEIGHT_FONT_RATIO`.
    """
    det = ctx.detection
    return max(det.field_height_pt, FIELD_HEIGHT_FONT_RATIO * ctx.local_font_size(near))


def rule_field_rect(ctx: CandidateContext, x0: float, x1: float, rule_y: float) -> Rect:
    """The entry rectangle sitting on top of a rule running along ``rule_y``."""
    det = ctx.detection
    y0 = rule_y + det.underline_gap_pt
    return Rect(x0, y0, x1, y0 + field_height(ctx, Rect(x0, y0, x1, y0 + det.field_height_pt)))


def grid_rules(ctx: CandidateContext) -> Tuple[frozenset, frozenset]:
    """Indices of the horizontal/vertical rules that are part of a ruled grid.

    A table's ruling lines must never become underline fields, so they are identified
    once: a horizontal rule crossed by two or more vertical rules, or one that coincides
    with the top or bottom edge of a detected table cell, belongs to the grid.
    """

    def build() -> Tuple[frozenset, frozenset]:
        det = ctx.detection
        tol = max(det.line_merge_tolerance_pt, 2.0)
        horizontal = set()
        vertical = set()
        for index, rule in enumerate(ctx.h_rules):
            y = (rule.rect.y0 + rule.rect.y1) / 2.0
            crossings = 0
            for v in ctx.v_rules:
                vr = v.rect
                x = (vr.x0 + vr.x1) / 2.0
                if x < rule.rect.x0 - tol or x > rule.rect.x1 + tol:
                    continue
                if vr.y0 - tol <= y <= vr.y1 + tol:
                    crossings += 1
            if crossings >= 2:
                horizontal.add(index)
                continue
            for cell in ctx.table_cells:
                if cell.horizontal_overlap(rule.rect) < 0.6 * cell.width:
                    continue
                if abs(cell.y0 - y) <= tol or abs(cell.y1 - y) <= tol:
                    horizontal.add(index)
                    break
        for index, rule in enumerate(ctx.v_rules):
            x = (rule.rect.x0 + rule.rect.x1) / 2.0
            crossings = 0
            for h in ctx.h_rules:
                hr = h.rect
                y = (hr.y0 + hr.y1) / 2.0
                if y < rule.rect.y0 - tol or y > rule.rect.y1 + tol:
                    continue
                if hr.x0 - tol <= x <= hr.x1 + tol:
                    crossings += 1
            if crossings >= 2:
                vertical.add(index)
        return frozenset(horizontal), frozenset(vertical)

    return _cached(ctx, "grid_rules", build)


def rule_on_box_edge(ctx: CandidateContext, rule: VectorPrimitive) -> bool:
    """True when ``rule`` is the top or bottom edge of a detected box."""
    det = ctx.detection
    tol = max(det.line_merge_tolerance_pt, 1.5)
    y = (rule.rect.y0 + rule.rect.y1) / 2.0
    for box in ctx.boxes:
        if box.horizontal_overlap(rule.rect) < 0.8 * min(box.width, rule.rect.width):
            continue
        if abs(box.y0 - y) <= tol or abs(box.y1 - y) <= tol:
            return True
    return False


def spans_on_rule(ctx: CandidateContext, rule: VectorPrimitive) -> List[TextSpan]:
    """Visible spans that sit on ``rule``'s row and cover part of it.

    These are either the labels of a row whose fields all share one long rule -- the
    ``First Name: ____  Last Name: ____`` idiom -- or, when they cover the rule, the
    text of an underlined heading.

    A span qualifies when its box bottom is on the rule's own line (from just below the
    rule up to roughly one cap height above it, so the row above never counts) and it
    covers at least half of its own width of the rule.
    """
    y = rule.rect.y1
    lift = max(0.6 * ctx.body_font_size, 4.0)
    band = Rect(rule.rect.x0, y - 2.5, rule.rect.x1, y + lift)
    found: Dict[Tuple[float, ...], TextSpan] = {}
    for span in ctx.spans_near(band, 1.0):
        if span.rect.y0 < band.y0 or span.rect.y0 > band.y1:
            continue
        overlap = span.rect.horizontal_overlap(rule.rect)
        if span.rect.width > 0.0 and overlap < 0.5 * span.rect.width:
            continue
        if overlap <= 0.0:
            continue
        found[_span_key(span)] = span
    for span in _vision_rule_spans(ctx).get(_rule_key(rule), ()):  # extra vision signal
        found.setdefault(_span_key(span), span)
    return sorted(found.values(), key=lambda s: (s.rect.x0, s.rect.x1, s.text))


def _span_key(span: TextSpan) -> Tuple[float, ...]:
    """Identity of a span for de-duplication across sources."""
    return tuple(round(v, 2) for v in span.rect.as_list()) + (hash(span.text),)


def _rule_key(rule: VectorPrimitive) -> Tuple[float, ...]:
    """Rounded rectangle key identifying a rule."""
    return tuple(round(v, 2) for v in rule.rect.as_list())


def _vision_rule_spans(ctx: CandidateContext) -> Dict[Tuple[float, ...], List[TextSpan]]:
    """Whatever ``zfp.vision.primitives.rule_spans`` reports, keyed by rule rectangle.

    The helper is an extension beyond the contract, so its exact return shape is not
    guaranteed: the documented ``{rule index: spans}`` mapping, a mapping keyed by the
    rule itself, a parallel list of lists and a flat list of spans are all understood,
    and anything else is ignored.
    """

    def build() -> Dict[Tuple[float, ...], List[TextSpan]]:
        rules = list(ctx.h_rules)
        if not rules:
            return {}
        raw = vision_call(
            "rule_spans", rules, list(ctx.text_spans), config=ctx.config, listify=False
        )
        out: Dict[Tuple[float, ...], List[TextSpan]] = {}
        if not raw:
            return out
        items: List[Tuple[Any, Any]]
        if isinstance(raw, dict):
            items = list(raw.items())
        elif len(raw) == len(rules) and all(isinstance(v, (list, tuple)) for v in raw):
            items = list(zip(rules, raw))
        else:
            flat = [s for s in raw if isinstance(s, TextSpan)]
            for rule in rules:
                hits = [s for s in flat if s.rect.horizontal_overlap(rule.rect) > 0.0]
                if hits:
                    out[_rule_key(rule)] = hits
            return out
        for key, spans in items:
            if isinstance(key, int) and 0 <= key < len(rules):
                rule_rect: Any = rules[key].rect
            elif isinstance(key, VectorPrimitive):
                rule_rect = key.rect
            else:
                rule_rect = key
            if not isinstance(rule_rect, Rect):
                continue
            out[tuple(round(v, 2) for v in rule_rect.as_list())] = [
                s for s in spans if isinstance(s, TextSpan)
            ]
        return out

    return _cached(ctx, "vision_rule_spans", build)


def usable_rules(ctx: CandidateContext) -> List[VectorPrimitive]:
    """Horizontal rules that could plausibly carry a field.

    Grid rules, box edges, hairlines shorter than the configured minimum and full-width
    separators are all removed here so that every rule-based detector agrees on the set.
    """

    def build() -> List[VectorPrimitive]:
        det = ctx.detection
        grid_h, _ = grid_rules(ctx)
        out: List[VectorPrimitive] = []
        for index, rule in enumerate(ctx.h_rules):
            if index in grid_h:
                continue
            if rule.rect.width < det.min_line_length_pt:
                continue
            if rule.rect.width > SEPARATOR_PAGE_FRACTION * ctx.page_width:
                continue
            if rule_on_box_edge(ctx, rule):
                continue
            out.append(rule)
        return out

    return _cached(ctx, "usable_rules", build)


def _is_heading_span(ctx: CandidateContext, span: TextSpan) -> bool:
    """True when ``span`` is set as a title or heading rather than as body copy."""
    if "bold" in (span.font_name or "").lower():
        return True
    body = ctx.body_font_size
    return body > 0.0 and span_size(span) >= 1.10 * body


def _is_separator(
    ctx: CandidateContext, rule: VectorPrimitive, sitting: Sequence[TextSpan] = ()
) -> bool:
    """True when a rule spans the whole printed column, i.e. it divides sections.

    Margin to margin under a heading is furniture, not a field.  The test only applies
    when the page actually has a wide printed column *and* the rule is long enough to
    be spanning it -- otherwise a short rule on a nearly empty page would look like it
    covered "everything".  "The printed column" is measured two ways and spanning either
    counts: the bounding box of the ink, which is right when the page prints across its
    whole width, and
    :attr:`~zfp.candidates.context.CandidateContext.text_column`, the band between the
    mirrored margins, which is right when it does not -- a comb sheet whose text all
    clusters on the left still has a right margin, and a rule reaching it is still a
    separator.

    Text printed over the rule does not by itself make it a field.  A document title or
    a running head sits exactly like that -- ``Loan Application`` with a margin-to-margin
    rule 5pt under its baseline is the commonest piece of furniture there is -- and it
    always carries at least one *heading*-set span: bold, or larger than the body face.
    A real entry rule is not drawn back under its own label, so one span of heading type
    over a margin-to-margin rule settles it, even when body-set text (a ``Page 3``
    folio) shares the line.  A rule carrying **only** body-set text is a genuine labelled
    row -- ``First Name: ____   Last Name: ____`` is often drawn as one rule -- and stays
    in play.
    """
    if ctx.content_extent is None:
        # Nothing is printed on this page but the rules themselves.  There is no column
        # for a rule to divide and no heading for it to underline, so the only honest
        # reading of a bare rule is that it is a rule -- this is the scan that OCR could
        # not read, and dropping its geometry would leave the page with nothing at all.
        return False
    if rule.rect.width < 0.6 * ctx.page_width:
        return False
    spanning = False
    for column in (ctx.content_extent, ctx.text_column):
        if column is None or column.width < 0.5 * ctx.page_width:
            continue
        if rule.rect.x0 <= column.x0 + 2.0 and rule.rect.x1 >= column.x1 - 2.0:
            spanning = True
            break
    if not spanning:
        return False
    if sitting and not any(_is_heading_span(ctx, span) for span in sitting):
        return False
    return _caption_below(ctx, rule) is None


def _caption_below(ctx: CandidateContext, rule: VectorPrimitive) -> Optional[TextSpan]:
    """The caption printed immediately under ``rule``, if there is one.

    A rule captioned underneath -- ``______________`` over the word "Initials" -- is a
    field even when it runs the full width of the page.  A section separator never has
    one, and a heading under a rule is not a caption.

    The band is deliberately tighter than a line height (see
    :data:`CAPTION_BELOW_GAP_RATIO`): a caption is welded to its rule, whereas the first
    ordinary row under a running head sits a whole line below it, and reading that row's
    label as a caption is what would turn every continuation page's header rule into a
    field.
    """
    body = ctx.body_font_size
    floor = rule.rect.y0 - CAPTION_BELOW_GAP_RATIO * body
    probe = Rect(rule.rect.x0, floor, rule.rect.x1, rule.rect.y0 - 0.5)
    best: Optional[TextSpan] = None
    for span in ctx.spans_near(probe, 1.0):
        if span.rect.y1 > rule.rect.y0 - 0.5 or span.rect.y1 < floor:
            continue
        if span.rect.horizontal_overlap(rule.rect) < 0.4 * max(span.rect.width, EPS):
            continue
        if "bold" in (span.font_name or "").lower() or span_size(span) >= 1.10 * body:
            continue
        if best is None or span.rect.x0 < best.rect.x0:
            best = span
    return best


def _is_heading_rule(ctx: CandidateContext, rule: VectorPrimitive, sitting: Sequence[TextSpan]) -> bool:
    """True when the text on the rule is a heading the rule merely underlines."""
    if not sitting:
        return False
    width = rule.rect.width
    if width <= 0.0:
        return True
    covered = 0.0
    for span in sitting:
        covered += span.rect.horizontal_overlap(rule.rect)
    if covered >= 0.60 * width:
        return True
    for span in sitting:
        if span.rect.horizontal_overlap(rule.rect) < 0.25 * width:
            continue
        if _is_heading_span(ctx, span):
            return True
    return False


def rule_segments(
    ctx: CandidateContext, rule: VectorPrimitive
) -> List[Tuple[float, float, List[TextSpan]]]:
    """Split a rule into the stretches actually available for input.

    ``First Name: ______  Last Name: ______`` is often drawn as one rule running the
    whole row with the labels printed on top of it.  Each label's x-range is cut out,
    leaving one segment per field; the labels themselves come back with the segments so
    the caller does not have to search for them again.

    Returns an empty list when the rule is a separator or a heading underline.
    """
    det = ctx.detection
    sitting = spans_on_rule(ctx, rule)
    if _is_heading_rule(ctx, rule, sitting):
        return []
    if _is_separator(ctx, rule, sitting):
        return []

    blocks = sorted(
        (max(rule.rect.x0, s.rect.x0), min(rule.rect.x1, s.rect.x1), s)
        for s in sitting
        if min(rule.rect.x1, s.rect.x1) > max(rule.rect.x0, s.rect.x0)
    )
    out: List[Tuple[float, float, List[TextSpan]]] = []
    cursor = rule.rect.x0
    preceding: List[TextSpan] = []
    for start, end, span in blocks:
        if start - cursor >= det.min_line_length_pt:
            out.append((cursor, start, list(preceding)))
            preceding = []
        cursor = max(cursor, end)
        preceding.append(span)
    if rule.rect.x1 - cursor >= det.min_line_length_pt:
        out.append((cursor, rule.rect.x1, list(preceding)))
    return out


def claimed_rects(ctx: CandidateContext) -> List[Rect]:
    """Rectangles that the hard-geometry detectors will propose on this page.

    Used by the whitespace detectors to stay out of the way of stronger evidence,
    without needing to know what the other detectors actually returned.
    """

    def build() -> List[Rect]:
        out: List[Rect] = []
        out.extend(ctx.boxes)
        out.extend(ctx.table_cells)
        out.extend(ctx.checkbox_glyphs)
        out.extend(ctx.circles)
        for run in ctx.comb_runs:
            union = Rect.bounding(run)
            if union is not None:
                out.append(union)
        for rule in usable_rules(ctx):
            for x0, x1, _ in rule_segments(ctx, rule):
                out.append(rule_field_rect(ctx, x0, x1, rule.rect.y1))
        return out

    return _cached(ctx, "claimed_rects", build)


def _served_beside(ctx: CandidateContext, span: TextSpan) -> bool:
    """True when ``span``'s field is printed on its own row, beside it.

    A label with a box or a rule next to it already has its field; the whitespace under
    it is the gutter between two rows of the layout, not a second entry area.
    """
    for other in claimed_rects(ctx):
        if other.contains_rect(span.rect):
            continue
        if other.horizontal_overlap(span.rect) > 0.5 * span.rect.width:
            continue
        if other.vertical_overlap(span.rect) >= 0.4 * max(span.rect.height, EPS):
            return True
    return False


def _on_option_row(ctx: CandidateContext, rect: Rect) -> bool:
    """True when ``rect`` is whitespace on the row of a checkbox or radio run.

    The space between and after a row of options is the gap the options are spread
    across, not somewhere to write.
    """
    for run in option_runs(ctx):
        bbox = run["bbox"]
        if rect.x1 <= bbox.x0 - 2.0:
            continue
        if rect.vertical_overlap(bbox.inflated(0.0, 2.0)) > 0.0:
            return True
    return False


def _inside_text_column(ctx: CandidateContext, rect: Rect, slack: float = 2.0) -> bool:
    """True when ``rect`` stays inside the horizontal band the page prints text in.

    Whitespace analysis works on the crop box, so a gap between two lines comes back
    running edge to edge.  A field never does: it lives inside the printed column, so a
    region that spills past the margins on either side is page background, not an entry
    area.
    """
    column = ctx.text_column
    if column is None or column.width <= 0.0:
        return True
    return rect.x0 >= column.x0 - slack and rect.x1 <= column.x1 + slack


def _is_claimed(ctx: CandidateContext, rect: Rect, threshold: float = CLAIMED_IOU) -> bool:
    """True when harder geometry already covers ``rect``.

    Two tests, because whitespace and vector geometry disagree about extent: a plain
    IoU catches the case where the two rectangles are the same field, and the fraction
    of ``rect`` that is covered catches the slivers a maximal-rectangle sweep leaves
    hanging off the edge of a box.
    """
    area = rect.area
    for other in claimed_rects(ctx):
        if rect.iou(other) > threshold:
            return True
        overlap = rect.intersection(other)
        if overlap is not None and area > 0.0 and overlap.area / area > CLAIMED_COVERAGE:
            return True
    return False


def _swallows_entry_rows(ctx: CandidateContext, region: Rect) -> bool:
    """True when ``region`` covers more than one label-anchored entry blank.

    A maximal empty rectangle is only ever a statement about where the page is free of
    ink, and on a borderless form the empty space beside one row's label runs straight
    into the empty space beside the next one.  The resulting void looks exactly like an
    answer box and is in fact the gutter between two fields, so claiming it costs two
    true positives and adds one false one.

    :attr:`CandidateContext.label_blanks` is the disambiguator: those regions are
    anchored *under a printed label*, so two of them inside one void means two fields
    inside it.  The void the free-text archetype is actually looking for -- a
    ``Comments:`` block -- contains no anchored blank but the one it is itself.
    """
    if not ctx.label_blanks:
        return False
    count = 0
    for blank in ctx.label_blanks:
        if blank.iou(region) > 0.60:
            continue  # this *is* the region, not something inside it
        overlap = region.intersection(blank)
        if overlap is None:
            continue
        if overlap.area > 0.5 * max(blank.area, EPS):
            count += 1
            if count > FREE_TEXT_MAX_ANCHORED:
                return True
    return False


def option_glyphs(ctx: CandidateContext) -> List[Tuple[Rect, str]]:
    """Every small mark that could be a checkbox or a radio button.

    Returns ``(rect, shape)`` pairs where shape is ``"circle"``, ``"box"`` or
    ``"glyph"``; a rectangle reported by more than one source appears once.
    """

    def build() -> List[Tuple[Rect, str]]:
        det = ctx.detection
        seen: Dict[Tuple[float, ...], Tuple[Rect, str]] = {}
        # A comb's cells are the same size as a checkbox but mean something else.
        comb_cells = {
            tuple(round(v, 1) for v in cell.as_list())
            for run in ctx.comb_runs
            for cell in run
        }

        def remember(rect: Rect, shape: str) -> None:
            key = tuple(round(v, 1) for v in rect.as_list())
            if key in comb_cells or key in seen:
                return
            seen[key] = (rect, shape)

        for rect in ctx.circles:
            if _checkbox_sized(rect, det):
                remember(rect, "circle")
        for rect in ctx.checkbox_glyphs:
            remember(rect, "glyph")
        for rect in ctx.boxes:
            if _checkbox_sized(rect, det):
                remember(rect, "box")
        out = list(seen.values())
        out.sort(key=lambda item: (-item[0].y1, item[0].x0))
        return out

    return _cached(ctx, "option_glyphs", build)


def _is_container(ctx: CandidateContext, box: Rect) -> bool:
    """True when ``box`` is the outline around finer geometry rather than a field itself.

    The frame of a ruled table and the border of a comb both come back from
    ``detect_boxes`` looking exactly like a big empty rectangle; they are not entry
    areas, and the cells inside them are.
    """
    contained = 0
    for cell in ctx.table_cells:
        if box.contains_rect(cell.inflated(-0.5)) and cell.area < 0.95 * box.area:
            contained += 1
            if contained >= 2:
                return True
    for run in ctx.comb_runs:
        inside = sum(
            1 for cell in run if box.contains_rect(cell.inflated(-0.5)) and cell.area < box.area
        )
        if inside >= 3:
            return True
    return False


def _checkbox_sized(rect: Rect, det: DetectionConfig) -> bool:
    """True when ``rect`` is small and roughly square."""
    w, h = rect.width, rect.height
    if w < det.checkbox_min_pt or h < det.checkbox_min_pt:
        return False
    if w > det.checkbox_max_pt or h > det.checkbox_max_pt:
        return False
    longest = max(w, h)
    return longest > 0.0 and abs(w - h) <= det.checkbox_aspect_tolerance * longest


def option_runs(ctx: CandidateContext) -> List[Dict[str, Any]]:
    """Group option glyphs into rows/columns that share a stem label.

    A run needs at least two equally sized, evenly spaced marks on one row or in one
    column, plus a stem label to the left of the row or above the column.  Runs of
    circles are radio groups; runs of squares are usually independent checkboxes and are
    reported too, tagged with their shape, so the checkbox detector can tell them apart.
    """

    def build() -> List[Dict[str, Any]]:
        glyphs = option_glyphs(ctx)
        runs: List[Dict[str, Any]] = []
        runs.extend(_row_runs(ctx, glyphs))
        runs.extend(_column_runs(ctx, glyphs, {id(r) for run in runs for r in run["rects"]}))
        return runs

    return _cached(ctx, "option_runs", build)


def _same_size(a: Rect, b: Rect, tol: float = 1.5) -> bool:
    """True when two marks are the same size to within ``tol`` points."""
    return abs(a.width - b.width) <= tol and abs(a.height - b.height) <= tol


def _row_runs(ctx: CandidateContext, glyphs: Sequence[Tuple[Rect, str]]) -> List[Dict[str, Any]]:
    """Runs of marks sharing a row, left to right."""
    rows: List[List[Tuple[Rect, str]]] = []
    for rect, shape in sorted(glyphs, key=lambda item: (-item[0].y1, item[0].x0)):
        placed = False
        for row in rows:
            head = row[0][0]
            tol = max(0.6 * head.height, 2.0)
            if abs(head.center.y - rect.center.y) <= tol and _same_size(head, rect):
                row.append((rect, shape))
                placed = True
                break
        if not placed:
            rows.append([(rect, shape)])
    out: List[Dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda item: item[0].x0)
        for group in _even_runs([r for r, _ in row], horizontal=True):
            if len(group) < 2:
                continue
            shapes = {shape for rect, shape in row if rect in group}
            bbox = Rect.bounding(group)
            if bbox is None:
                continue
            stem = label_left(ctx, Rect(group[0].x0, bbox.y0, group[0].x1, bbox.y1))
            if stem is None:
                stem = label_above(ctx, bbox, 2.2 * ctx.body_font_size)
            if stem is None:
                continue
            out.append(
                {
                    "rects": group,
                    "shape": "circle" if "circle" in shapes else sorted(shapes)[0],
                    "orientation": "row",
                    "stem": stem[0],
                    "stem_gap": stem[1],
                    "bbox": bbox,
                }
            )
    return out


def _column_runs(
    ctx: CandidateContext, glyphs: Sequence[Tuple[Rect, str]], used: Iterable[int]
) -> List[Dict[str, Any]]:
    """Runs of marks sharing a column, top to bottom."""
    taken = set(used)
    columns: List[List[Tuple[Rect, str]]] = []
    for rect, shape in sorted(glyphs, key=lambda item: (item[0].x0, -item[0].y1)):
        if id(rect) in taken:
            continue
        placed = False
        for column in columns:
            head = column[0][0]
            tol = max(0.6 * head.width, 2.0)
            if abs(head.x0 - rect.x0) <= tol and _same_size(head, rect):
                column.append((rect, shape))
                placed = True
                break
        if not placed:
            columns.append([(rect, shape)])
    out: List[Dict[str, Any]] = []
    for column in columns:
        column.sort(key=lambda item: -item[0].y1)
        for group in _even_runs([r for r, _ in column], horizontal=False):
            if len(group) < 2:
                continue
            shapes = {shape for rect, shape in column if rect in group}
            bbox = Rect.bounding(group)
            if bbox is None:
                continue
            stem = label_above(ctx, bbox, 3.0 * ctx.body_font_size)
            if stem is None:
                continue
            out.append(
                {
                    "rects": group,
                    "shape": "circle" if "circle" in shapes else sorted(shapes)[0],
                    "orientation": "column",
                    "stem": stem[0],
                    "stem_gap": stem[1],
                    "bbox": bbox,
                }
            )
    return out


def _stem_of(ctx: CandidateContext, rect: Rect) -> Optional[Tuple[TextSpan, float]]:
    """The stem label of the option run ``rect`` belongs to, if it belongs to one."""
    key = tuple(round(v, 1) for v in rect.as_list())
    for run in option_runs(ctx):
        if len(run["rects"]) < 2:
            continue
        for member in run["rects"]:
            if tuple(round(v, 1) for v in member.as_list()) == key:
                return (run["stem"], run["stem_gap"])
    return None


def _even_runs(rects: Sequence[Rect], horizontal: bool) -> List[List[Rect]]:
    """Split an ordered sequence of marks wherever the spacing stops being regular."""
    if len(rects) < 2:
        return [list(rects)] if rects else []

    def pitch(a: Rect, b: Rect) -> float:
        return (b.x0 - a.x0) if horizontal else (a.y1 - b.y1)

    runs: List[List[Rect]] = []
    run: List[Rect] = [rects[0]]
    expected: Optional[float] = None
    for rect in rects[1:]:
        step = pitch(run[-1], rect)
        limit = max(6.0 * max(rect.width, rect.height), 240.0)
        if step <= 0.0 or step > limit:
            runs.append(run)
            run = [rect]
            expected = None
            continue
        if expected is not None and abs(step - expected) > max(0.35 * expected, 4.0):
            runs.append(run)
            run = [rect]
            expected = None
            continue
        if expected is None and len(run) >= 1:
            expected = step
        run.append(rect)
    runs.append(run)
    return [r for r in runs if r]


# --------------------------------------------------------------------------- detectors
class UnderlineFieldDetector:
    """``Label: ______`` -- the commonest printed-form idiom there is."""

    name = "underline"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit one candidate per usable stretch of every horizontal rule."""
        det = ctx.detection
        out: List[FieldCandidate] = []
        for rule in usable_rules(ctx):
            for x0, x1, preceding in rule_segments(ctx, rule):
                rect = rule_field_rect(ctx, x0, x1, rule.rect.y1)
                candidate = new_candidate(ctx, rect, self.name, FieldType.TEXT)
                if candidate is None:
                    continue
                candidate.confidence.geometry = 0.99 if not preceding else 0.95
                candidate.add_evidence(
                    Evidence(
                        kind=EvidenceKind.VECTOR_LINE,
                        score=candidate.confidence.geometry,
                        detail="horizontal rule %.1f pt wide" % (x1 - x0),
                        rect=rule.rect,
                    )
                )
                found: Optional[Tuple[TextSpan, float]] = None
                source = "label on the rule"
                if preceding:
                    span = preceding[-1]
                    found = (span, max(x0 - span.rect.x1, 0.0))
                if found is None:
                    found, source = find_label(ctx, candidate.rect)
                text = _attach_label(candidate, found, det.label_max_distance_pt, source)
                candidate.field_type = type_from_label(text, FieldType.TEXT)
                size = ctx.local_font_size(candidate.rect)
                candidate.constraints.max_chars_estimate = max_chars_for(
                    candidate.rect.width, size
                )
                out.append(candidate)
        return out


class BoxFieldDetector:
    """``Label [____________]`` -- a stroked rectangle used as the entry area."""

    name = "box"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit one candidate per stroked box that is not furniture."""
        det = ctx.detection
        comb_cells = {
            tuple(round(v, 1) for v in cell.as_list())
            for run in ctx.comb_runs
            for cell in run
        }
        cells = {tuple(round(v, 1) for v in c.as_list()) for c in ctx.table_cells}
        out: List[FieldCandidate] = []
        for box in ctx.boxes:
            key = tuple(round(v, 1) for v in box.as_list())
            if key in cells or key in comb_cells:
                continue
            if _checkbox_sized(box, det):
                continue
            if box.width >= 0.98 * ctx.page_width and box.height >= 0.98 * ctx.page_height:
                continue
            if _is_container(ctx, box):
                continue
            stroke = ctx.stroke_width_for(box)
            inner = box.inflated(-stroke)
            if inner.width <= 0.0 or inner.height <= 0.0:
                inner = box
            multiline = inner.height > MULTILINE_HEIGHT_RATIO * ctx.body_font_size
            ftype = FieldType.MULTILINE_TEXT if multiline else FieldType.TEXT
            candidate = new_candidate(ctx, inner, self.name, ftype)
            if candidate is None:
                continue
            candidate.confidence.geometry = 0.97
            candidate.constraints.multiline = multiline
            candidate.add_evidence(
                Evidence(
                    kind=EvidenceKind.VECTOR_RECT,
                    score=0.97,
                    detail="stroked box %.1f x %.1f pt" % (box.width, box.height),
                    rect=box,
                )
            )
            found, source = find_label(ctx, candidate.rect)
            text = _attach_label(candidate, found, det.label_max_distance_pt, source)
            if not multiline:
                candidate.field_type = type_from_label(text, FieldType.TEXT)
            size = ctx.local_font_size(candidate.rect)
            candidate.constraints.max_chars_estimate = max_chars_for(candidate.rect.width, size)
            out.append(candidate)
        return out


class CheckboxDetector:
    """``[ ] Yes`` -- a small square, circle or box glyph with its option label."""

    name = "checkbox"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit one checkbox candidate per standalone mark."""
        det = ctx.detection
        radio_members = {
            tuple(round(v, 1) for v in rect.as_list())
            for run in option_runs(ctx)
            if run["shape"] == "circle" and len(run["rects"]) >= 2
            for rect in run["rects"]
        }
        out: List[FieldCandidate] = []
        for rect, shape in option_glyphs(ctx):
            key = tuple(round(v, 1) for v in rect.as_list())
            if key in radio_members:
                continue  # a member of a radio group; RadioGroupDetector owns it
            candidate = new_candidate(ctx, rect, self.name, FieldType.CHECKBOX, 2.0, 2.0)
            if candidate is None:
                continue
            kind = {
                "circle": EvidenceKind.VECTOR_CIRCLE,
                "box": EvidenceKind.VECTOR_RECT,
            }.get(shape, EvidenceKind.CHECKBOX_GLYPH)
            candidate.confidence.geometry = 0.96 if shape != "glyph" else 0.90
            candidate.add_evidence(
                Evidence(
                    kind=kind,
                    score=candidate.confidence.geometry,
                    detail="%s %.1f x %.1f pt" % (shape, rect.width, rect.height),
                    rect=rect,
                )
            )
            option, source = find_label(ctx, rect, ("right", "left"))
            option_text = clean_label(option[0].text) if option is not None else ""
            candidate.export_value = export_value_for(option_text)
            stem = _stem_of(ctx, rect)
            if stem is not None:
                # An option in a labelled run belongs to the run's field: the stem names
                # it, the neighbouring word is only its "on" value.
                _attach_label(candidate, stem, det.label_max_distance_pt, "stem label")
                if option_text:
                    candidate.add_evidence(
                        Evidence(
                            kind=_text_evidence_kind(option[0]),
                            score=0.85,
                            detail="option label: %r" % option_text,
                            rect=option[0].rect,
                        )
                    )
            else:
                _attach_label(candidate, option, det.label_max_distance_pt, source)
            out.append(candidate)
        return out


class RadioGroupDetector:
    """``Status:  ( ) Single  ( ) Married`` -- options sharing one stem label."""

    name = "radio_group"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit one candidate per option, all sharing the group's identifier."""
        det = ctx.detection
        out: List[FieldCandidate] = []
        for run in option_runs(ctx):
            if run["shape"] != "circle" or len(run["rects"]) < 2:
                continue
            stem_text = clean_label(run["stem"].text)
            if not stem_text:
                continue
            group_id = stable_id(normalize_label(stem_text), ctx.page, prefix="grp")
            for rect in run["rects"]:
                candidate = new_candidate(ctx, rect, self.name, FieldType.RADIO, 2.0, 2.0)
                if candidate is None:
                    continue
                candidate.group_id = group_id
                candidate.visible_label = stem_text
                candidate.confidence.geometry = 0.96
                candidate.confidence.label_link = _label_confidence(
                    run["stem_gap"], det.label_max_distance_pt
                )
                candidate.add_evidence(
                    Evidence(
                        kind=EvidenceKind.VECTOR_CIRCLE,
                        score=0.96,
                        detail="radio option %.1f pt across" % rect.width,
                        rect=rect,
                    )
                )
                candidate.add_evidence(
                    Evidence(
                        kind=EvidenceKind.LAYOUT,
                        score=0.90,
                        detail="%s run of %d evenly spaced options"
                        % (run["orientation"], len(run["rects"])),
                        rect=run["bbox"],
                    )
                )
                candidate.add_evidence(
                    Evidence(
                        kind=EvidenceKind.LABEL_LINK,
                        score=candidate.confidence.label_link,
                        detail="stem label: %r" % stem_text,
                        rect=run["stem"].rect,
                    )
                )
                option = label_right(ctx, rect, det.label_max_distance_pt)
                if option is None:
                    option = label_below(ctx, rect, 1.6 * ctx.body_font_size)
                option_text = clean_label(option[0].text) if option is not None else ""
                candidate.export_value = export_value_for(
                    option_text, fallback="Option%d" % (run["rects"].index(rect) + 1)
                )
                if option_text:
                    candidate.add_evidence(
                        Evidence(
                            kind=_text_evidence_kind(option[0]),
                            score=0.85,
                            detail="option label: %r" % option_text,
                            rect=option[0].rect,
                        )
                    )
                out.append(candidate)
        return out


class CombFieldDetector:
    """``[ ][ ][ ][ ][ ]`` -- one value spread over a run of equal cells."""

    name = "comb"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit exactly one candidate spanning each comb run."""
        out: List[FieldCandidate] = []
        for run in ctx.comb_runs:
            if len(run) < 3:
                continue
            union = Rect.bounding(run)
            if union is None:
                continue
            stroke = ctx.stroke_width_for(union, default=ctx.median_stroke_width or 0.6)
            inner = union.inflated(-stroke)
            if inner.width <= 0.0 or inner.height <= 0.0:
                inner = union
            candidate = new_candidate(ctx, inner, self.name, FieldType.COMB)
            if candidate is None:
                continue
            candidate.confidence.geometry = 0.97
            candidate.constraints.comb_cells = len(run)
            candidate.constraints.max_chars_estimate = len(run)
            candidate.add_evidence(
                Evidence(
                    kind=EvidenceKind.COMB_CELL,
                    score=0.97,
                    detail="%d equal cells of %.1f pt" % (len(run), run[0].width),
                    rect=union,
                )
            )
            candidate.add_evidence(
                Evidence(
                    kind=EvidenceKind.REPEAT,
                    score=0.90,
                    detail="regular cell pitch",
                    rect=union,
                )
            )
            found, source = find_label(ctx, candidate.rect)
            _attach_label(candidate, found, ctx.detection.label_max_distance_pt, source)
            out.append(candidate)
        return out


class DateBoxDetector:
    """``[  ] / [  ] / [    ]`` and ``____/____/______`` -- split date entry."""

    name = "date_box"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit one DATE candidate per separated group run or dated rule."""
        out: List[FieldCandidate] = []
        out.extend(self._grouped(ctx))
        out.extend(self._placeholders(ctx))
        return out

    # -- three groups divided by "/" or "-" ------------------------------------------
    def _grouped(self, ctx: CandidateContext) -> List[FieldCandidate]:
        det = ctx.detection
        out: List[FieldCandidate] = []
        pool: Dict[Tuple[float, ...], Rect] = {}
        for rect in list(ctx.boxes) + [cell for run in ctx.comb_runs for cell in run]:
            if rect.width > 40.0 or rect.height > 40.0:
                continue
            pool.setdefault(tuple(round(v, 1) for v in rect.as_list()), rect)
        rows: List[List[Rect]] = []
        for rect in sorted(pool.values(), key=lambda r: (-r.y1, r.x0)):
            placed = False
            for row in rows:
                if abs(row[0].y0 - rect.y0) <= 2.0 and abs(row[0].y1 - rect.y1) <= 2.0:
                    row.append(rect)
                    placed = True
                    break
            if not placed:
                rows.append([rect])
        for row in rows:
            row.sort(key=lambda r: r.x0)
            bbox = Rect.bounding(row)
            if bbox is None or len(row) < 3:
                continue
            separators = [
                s
                for s in ctx.spans_near(bbox, 4.0)
                if (s.text or "").strip() in ("/", "-", ".")
                and bbox.x0 <= s.rect.center.x <= bbox.x1
            ]
            if len(separators) < 2:
                continue
            groups = self._split_by(row, sorted(s.rect.center.x for s in separators))
            if len(groups) != 3 or any(not g for g in groups):
                continue
            union = Rect.bounding([r for g in groups for r in g])
            if union is None:
                continue
            candidate = new_candidate(ctx, union, self.name, FieldType.DATE)
            if candidate is None:
                continue
            char = (separators[0].text or "/").strip() or "/"
            candidate.confidence.geometry = 0.94
            candidate.constraints.format_hint = _date_format_hint(
                [len(g) for g in groups], char
            )
            candidate.constraints.comb_cells = sum(len(g) for g in groups)
            candidate.add_evidence(
                Evidence(
                    kind=EvidenceKind.COMB_CELL,
                    score=0.94,
                    detail="three groups of %s separated by %r"
                    % ("/".join(str(len(g)) for g in groups), char),
                    rect=union,
                )
            )
            candidate.add_evidence(
                Evidence(
                    kind=EvidenceKind.PATTERN,
                    score=0.90,
                    detail="date layout %s" % candidate.constraints.format_hint,
                    rect=union,
                )
            )
            found, source = find_label(ctx, candidate.rect)
            _attach_label(candidate, found, det.label_max_distance_pt, source)
            out.append(candidate)
        return out

    @staticmethod
    def _split_by(row: Sequence[Rect], cuts: Sequence[float]) -> List[List[Rect]]:
        """Partition a row of cells at the separator x positions."""
        groups: List[List[Rect]] = [[]]
        index = 0
        for rect in row:
            while index < len(cuts) and rect.x0 > cuts[index]:
                groups.append([])
                index += 1
            groups[-1].append(rect)
        return groups

    # -- a rule whose printed placeholder is a date ----------------------------------
    def _placeholders(self, ctx: CandidateContext) -> List[FieldCandidate]:
        det = ctx.detection
        out: List[FieldCandidate] = []
        for rule in usable_rules(ctx):
            segments = rule_segments(ctx, rule)
            if not segments:
                continue
            for x0, x1, preceding in segments:
                rect = rule_field_rect(ctx, x0, x1, rule.rect.y1)
                hint = self._hint_near(ctx, rect, preceding)
                if hint is None:
                    continue
                candidate = new_candidate(ctx, rect, self.name, FieldType.DATE)
                if candidate is None:
                    continue
                candidate.confidence.geometry = 0.95
                candidate.confidence.semantic_type = float(hint[1])
                candidate.constraints.format_hint = hint[0]
                candidate.add_evidence(
                    Evidence(
                        kind=EvidenceKind.VECTOR_LINE,
                        score=0.95,
                        detail="dated rule %.1f pt wide" % (x1 - x0),
                        rect=rule.rect,
                    )
                )
                candidate.add_evidence(
                    Evidence(
                        kind=EvidenceKind.PATTERN,
                        score=float(hint[1]),
                        detail="date placeholder %s" % hint[0],
                        rect=rect,
                    )
                )
                found: Optional[Tuple[TextSpan, float]] = None
                if preceding:
                    span = preceding[-1]
                    found = (span, max(x0 - span.rect.x1, 0.0))
                if found is None:
                    found = label_left(ctx, candidate.rect)
                _attach_label(candidate, found, det.label_max_distance_pt, "label of date rule")
                out.append(candidate)
        return out

    @staticmethod
    def _hint_near(
        ctx: CandidateContext, rect: Rect, preceding: Sequence[TextSpan]
    ) -> Optional[Tuple[str, float]]:
        """Return ``(format_hint, confidence)`` when a date placeholder sits near ``rect``."""
        probe = Rect(rect.x0 - 6.0, rect.y0 - 2.0 * ctx.body_font_size, rect.x1 + 6.0, rect.y1)
        seen: List[TextSpan] = list(preceding) + ctx.spans_near(probe, 2.0)
        for span in seen:
            rule = match_placeholder(span.text or "")
            if rule is not None and rule.field_type is FieldType.DATE:
                return (rule.format_hint or "MM/DD/YYYY", rule.confidence)
        return None


class SignatureLineDetector:
    """A rule or box captioned "Signature" / "Initials" -- never a date line."""

    name = "signature"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit a SIGNATURE candidate per rule or box whose caption asks for one."""
        out: List[FieldCandidate] = []
        for rule in usable_rules(ctx):
            for x0, x1, preceding in rule_segments(ctx, rule):
                rect = rule_field_rect(ctx, x0, x1, rule.rect.y1)
                found = self._caption(ctx, rect, preceding)
                if found is None:
                    continue
                text = clean_label(found[0].text)
                if looks_like_date(text) or not looks_like_signature(text):
                    continue
                candidate = new_candidate(ctx, rect, self.name, FieldType.SIGNATURE)
                if candidate is None:
                    continue
                candidate.confidence.geometry = 0.98
                candidate.confidence.semantic_type = 0.92
                candidate.add_evidence(
                    Evidence(
                        kind=EvidenceKind.VECTOR_LINE,
                        score=0.98,
                        detail="signature rule %.1f pt wide" % (x1 - x0),
                        rect=rule.rect,
                    )
                )
                _attach_label(
                    candidate,
                    found,
                    ctx.detection.label_max_distance_pt,
                    "signature caption",
                )
                out.append(candidate)
        for box in ctx.boxes:
            if _checkbox_sized(box, ctx.detection):
                continue
            stroke = ctx.stroke_width_for(box)
            inner = box.inflated(-stroke)
            if inner.width <= 0.0 or inner.height <= 0.0:
                inner = box
            found = self._caption(ctx, inner, ())
            if found is None:
                continue
            text = clean_label(found[0].text)
            if looks_like_date(text) or not looks_like_signature(text):
                continue
            candidate = new_candidate(ctx, inner, self.name, FieldType.SIGNATURE)
            if candidate is None:
                continue
            candidate.confidence.geometry = 0.96
            candidate.confidence.semantic_type = 0.92
            candidate.add_evidence(
                Evidence(
                    kind=EvidenceKind.VECTOR_RECT,
                    score=0.96,
                    detail="signature box %.1f x %.1f pt" % (box.width, box.height),
                    rect=box,
                )
            )
            _attach_label(
                candidate, found, ctx.detection.label_max_distance_pt, "signature caption"
            )
            out.append(candidate)
        return out

    @staticmethod
    def _caption(
        ctx: CandidateContext, rect: Rect, preceding: Sequence[TextSpan]
    ) -> Optional[Tuple[TextSpan, float]]:
        """The caption of a signature line: to its left, under it, or over it."""
        if preceding:
            span = preceding[-1]
            if looks_like_signature(span.text):
                return (span, max(rect.x0 - span.rect.x1, 0.0))
        candidates: List[Optional[Tuple[TextSpan, float]]] = [
            label_left(ctx, rect),
            label_below(ctx, rect, SIGNATURE_LABEL_GAP_PT),
            label_above(ctx, rect, SIGNATURE_LABEL_GAP_PT),
        ]
        for found in candidates:
            if found is not None and looks_like_signature(found[0].text):
                return found
        return None


class TableCellDetector:
    """The empty cells of a ruled grid, minus its header row."""

    name = "table_cell"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit one candidate per empty data cell, labelled by its column header."""
        rows = _table_rows(ctx)
        if len(rows) < 2:
            return []
        header = self._header_row(ctx, rows)
        out: List[FieldCandidate] = []
        for row in rows:
            if row is header:
                continue
            row_label = self._cell_text(ctx, row[0])
            for cell in row:
                if self._cell_text(ctx, cell):
                    continue
                stroke = ctx.stroke_width_for(cell)
                inner = cell.inflated(-stroke)
                if inner.width <= 0.0 or inner.height <= 0.0:
                    inner = cell
                candidate = new_candidate(ctx, inner, self.name, FieldType.TEXT)
                if candidate is None:
                    continue
                candidate.confidence.geometry = 0.90
                candidate.add_evidence(
                    Evidence(
                        kind=EvidenceKind.TABLE_CELL,
                        score=0.90,
                        detail="empty grid cell %.1f x %.1f pt" % (cell.width, cell.height),
                        rect=cell,
                    )
                )
                column = self._column_header(ctx, header, cell)
                if column:
                    candidate.visible_label = column
                    candidate.confidence.label_link = 0.80
                    candidate.add_evidence(
                        Evidence(
                            kind=EvidenceKind.LABEL_LINK,
                            score=0.80,
                            detail="column header: %r" % column,
                            rect=cell,
                        )
                    )
                if row_label and row_label != column:
                    candidate.parent_context = [row_label]
                size = ctx.local_font_size(candidate.rect)
                candidate.constraints.max_chars_estimate = max_chars_for(
                    candidate.rect.width, size
                )
                out.append(candidate)
        return out

    @staticmethod
    def _cell_text(ctx: CandidateContext, cell: Rect) -> str:
        """Text printed inside ``cell`` (empty when the cell is blank)."""
        parts: List[str] = []
        for span in ctx.spans_near(cell, 0.0):
            if cell.contains_point(span.rect.center):
                parts.append(span.text.strip())
        return clean_label(" ".join(p for p in parts if p))

    def _header_row(self, ctx: CandidateContext, rows: Sequence[List[Rect]]) -> List[Rect]:
        """The row that captions the table: the topmost one, or the first all-text row."""
        for row in rows:
            if all(self._cell_text(ctx, cell) for cell in row):
                return row
        return list(rows[0])

    def _column_header(
        self, ctx: CandidateContext, header: Sequence[Rect], cell: Rect
    ) -> str:
        """Caption of the header cell sharing ``cell``'s column."""
        best = ""
        best_overlap = 0.0
        for head in header:
            overlap = head.horizontal_overlap(cell)
            if overlap > best_overlap and overlap >= 0.6 * min(head.width, cell.width):
                best_overlap = overlap
                best = self._cell_text(ctx, head)
        return best


def _table_rows(ctx: CandidateContext) -> List[List[Rect]]:
    """Group the detected table cells into rows, top of the page first."""

    def build() -> List[List[Rect]]:
        rows: List[List[Rect]] = []
        for cell in sorted(ctx.table_cells, key=lambda r: (-r.y1, r.x0)):
            placed = False
            for row in rows:
                if abs(row[0].y0 - cell.y0) <= 2.0 and abs(row[0].y1 - cell.y1) <= 2.0:
                    row.append(cell)
                    placed = True
                    break
            if not placed:
                rows.append([cell])
        for row in rows:
            row.sort(key=lambda r: r.x0)
        return rows

    return _cached(ctx, "table_rows", build)


class BlankRegionDetector:
    """Borderless whitespace next to a label -- a form with no ruling at all."""

    name = "blank_region"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit lower-confidence candidates for labelled, unclaimed whitespace.

        Only *entry-sized* whitespace counts: a region taller than a couple of lines is
        either a page-scale void or a free-text area, and
        :class:`FreeTextAreaDetector` owns the latter.
        """
        det = ctx.detection
        tallest = max(MULTILINE_HEIGHT_RATIO * ctx.body_font_size, det.blank_min_height_pt + 8.0)
        out: List[FieldCandidate] = []
        for region in ctx.blank_regions:
            if region.width < det.blank_min_width_pt or region.height < det.blank_min_height_pt:
                continue
            if region.height > tallest:
                continue
            if not _inside_text_column(ctx, region):
                continue
            if _is_claimed(ctx, region):
                continue
            if _on_option_row(ctx, region):
                continue
            found, source = find_label(
                ctx, region, ("left", "above"), 1.6 * ctx.body_font_size
            )
            geometry = 0.65 if source.endswith("left") else 0.55
            if found is None:
                continue
            text = clean_label(found[0].text)
            if not text:
                continue
            if source.endswith("left") and _served_below(ctx, found[0]):
                continue
            if source.endswith("above") and _served_beside(ctx, found[0]):
                continue
            multiline = False
            ftype = FieldType.TEXT
            candidate = new_candidate(ctx, region, self.name, ftype)
            if candidate is None:
                continue
            candidate.confidence.geometry = geometry
            candidate.constraints.multiline = multiline
            candidate.add_evidence(
                Evidence(
                    kind=EvidenceKind.BLANK_REGION,
                    score=geometry,
                    detail="blank %.1f x %.1f pt" % (region.width, region.height),
                    rect=region,
                )
            )
            _attach_label(candidate, found, det.label_max_distance_pt, source)
            if not multiline:
                candidate.field_type = type_from_label(text, FieldType.TEXT)
            size = ctx.local_font_size(candidate.rect)
            candidate.constraints.max_chars_estimate = max_chars_for(candidate.rect.width, size)
            out.append(candidate)
        return out


class FreeTextAreaDetector:
    """A large empty block: comments, explanations, "describe below"."""

    name = "free_text"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit MULTILINE_TEXT candidates for page-scale blank blocks."""
        line_height = 1.2 * ctx.body_font_size
        min_height = FREE_TEXT_MIN_LINES * line_height
        min_width = FREE_TEXT_MIN_WIDTH_FRACTION * ctx.page_width
        page_area = max(ctx.page_rect.area, EPS)
        out: List[FieldCandidate] = []
        for region in ctx.blank_regions:
            if region.height < min_height or region.width < min_width:
                continue
            # A void covering half the sheet is an empty page, not an answer box.
            if region.area > FREE_TEXT_MAX_PAGE_FRACTION * page_area:
                continue
            if not _inside_text_column(ctx, region):
                continue
            if _is_claimed(ctx, region):
                continue
            if _swallows_entry_rows(ctx, region):
                continue
            found = label_above(ctx, region, 2.2 * ctx.body_font_size)
            if found is None or not clean_label(found[0].text):
                continue
            candidate = new_candidate(ctx, region, self.name, FieldType.MULTILINE_TEXT)
            if candidate is None:
                continue
            candidate.confidence.geometry = 0.60
            candidate.constraints.multiline = True
            candidate.add_evidence(
                Evidence(
                    kind=EvidenceKind.BLANK_REGION,
                    score=0.60,
                    detail="free-text area %.1f x %.1f pt" % (region.width, region.height),
                    rect=region,
                )
            )
            candidate.add_evidence(
                Evidence(
                    kind=EvidenceKind.LAYOUT,
                    score=0.70,
                    detail="%.1f line heights tall" % (region.height / max(line_height, EPS)),
                    rect=region,
                )
            )
            _attach_label(
                candidate, found, ctx.detection.label_max_distance_pt, "label above"
            )
            size = ctx.local_font_size(candidate.rect)
            per_line = max_chars_for(candidate.rect.width, size) or 0
            lines = max(1, int(candidate.rect.height / max(line_height, EPS)))
            candidate.constraints.max_chars_estimate = max(per_line * lines, 1)
            out.append(candidate)
        return out


class ColonRunDetector:
    """``Name: ______`` and ``Name .........`` drawn purely as text.

    This is the archetype with no vector geometry at all, which is exactly the shape a
    scanned page arrives in once OCR has produced its words.
    """

    name = "colon_run"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Emit candidates for leader runs and for trailing colons."""
        out: List[FieldCandidate] = []
        for line in ctx.lines:
            for span in line:
                out.extend(self._from_leader(ctx, span))
            out.extend(self._from_colons(ctx, line))
        return out

    # -- "_____" / "....." runs inside the text --------------------------------------
    def _from_leader(self, ctx: CandidateContext, span: TextSpan) -> List[FieldCandidate]:
        det = ctx.detection
        text = span.text or ""
        out: List[FieldCandidate] = []
        for match in _LEADER_RUN.finditer(text):
            x0, x1 = _span_slice_x(span, match.start(), match.end())
            if x1 - x0 < det.min_line_length_pt:
                continue
            base = span.rect.y0
            rect = rule_field_rect(ctx, x0, x1, base)
            if _has_rule_under(ctx, rect):
                continue
            candidate = new_candidate(ctx, rect, self.name, FieldType.TEXT)
            if candidate is None:
                continue
            candidate.confidence.geometry = 0.70
            candidate.add_evidence(
                Evidence(
                    kind=_text_evidence_kind(span),
                    score=0.70,
                    detail="leader run %r" % match.group(0)[:12],
                    rect=Rect(x0, span.rect.y0, x1, span.rect.y1),
                )
            )
            prefix = clean_label(text[: match.start()])
            found: Optional[Tuple[TextSpan, float]] = None
            if prefix:
                found = (
                    TextSpan(
                        text=prefix,
                        rect=Rect(span.rect.x0, span.rect.y0, x0, span.rect.y1),
                        page=span.page,
                        font_name=span.font_name,
                        font_size=span.font_size,
                        source=span.source,
                        confidence=span.confidence,
                    ),
                    0.0,
                )
            if found is None:
                found = label_left(ctx, rect)
            text_label = _attach_label(
                candidate, found, det.label_max_distance_pt, "label before leader run"
            )
            candidate.field_type = type_from_label(text_label, FieldType.TEXT)
            size = ctx.local_font_size(candidate.rect)
            candidate.constraints.max_chars_estimate = max_chars_for(candidate.rect.width, size)
            out.append(candidate)
        return out

    # -- "Label:" with nothing after it ----------------------------------------------
    def _from_colons(self, ctx: CandidateContext, line: Sequence[TextSpan]) -> List[FieldCandidate]:
        det = ctx.detection
        out: List[FieldCandidate] = []
        ordered = sorted(line, key=lambda s: s.rect.x0)
        right_edge = ctx.text_column.x1 if ctx.text_column else ctx.page_rect.x1
        for index, span in enumerate(ordered):
            text = (span.text or "").rstrip()
            if not text.endswith(":"):
                continue
            if _LEADER_RUN.search(span.text or ""):
                continue
            x0 = span.rect.x1 + 0.5 * max(span_size(span), 6.0)
            x1 = ordered[index + 1].rect.x0 - 2.0 if index + 1 < len(ordered) else right_edge
            x1 = min(x1, _first_mark_right_of(ctx, span))
            if x1 - x0 < det.blank_min_width_pt:
                continue
            if _served_below(ctx, span):
                continue
            height = field_height(ctx, span.rect)
            rect = Rect(x0, span.rect.y0, x1, span.rect.y0 + height)
            if _has_rule_under(ctx, rect) or _is_claimed(ctx, rect):
                continue
            candidate = new_candidate(ctx, rect, self.name, FieldType.TEXT)
            if candidate is None:
                continue
            candidate.confidence.geometry = 0.55
            candidate.add_evidence(
                Evidence(
                    kind=_text_evidence_kind(span),
                    score=0.55,
                    detail="label ends with a colon",
                    rect=span.rect,
                )
            )
            label_text = _attach_label(
                candidate, (span, 0.0), det.label_max_distance_pt, "label before colon"
            )
            candidate.field_type = type_from_label(label_text, FieldType.TEXT)
            size = ctx.local_font_size(candidate.rect)
            candidate.constraints.max_chars_estimate = max_chars_for(candidate.rect.width, size)
            out.append(candidate)
        return out


def _span_slice_x(span: TextSpan, start: int, end: int) -> Tuple[float, float]:
    """User-space x range of ``span.text[start:end]``.

    Uses the span's per-glyph rectangles when the parser supplied them, then the font
    metrics, and finally a proportional split of the span's width.
    """
    glyphs = span.glyph_rects or []
    if len(glyphs) >= end > start:
        chunk = Rect.bounding(glyphs[start:end])
        if chunk is not None:
            return (chunk.x0, chunk.x1)
    text = span.text or ""
    size = span_size(span)
    if size > 0.0 and text:
        try:
            from ..pdfio.fonts import text_width

            font = span.font_name or "Helvetica"
            total = float(text_width(text, font, size))
            left = float(text_width(text[:start], font, size))
            width = float(text_width(text[start:end], font, size))
            if width > 0.0 and total > 0.0:
                # Rescale onto the span's own box: character spacing, a horizontal
                # scale or a substituted font all make the metric total disagree with
                # the rectangle the parser measured, and the rectangle is the truth.
                scale = span.rect.width / total if span.rect.width > 0.0 else 1.0
                return (
                    span.rect.x0 + left * scale,
                    span.rect.x0 + (left + width) * scale,
                )
        except Exception:  # pragma: no cover - defensive
            pass
    length = max(len(text), 1)
    unit = span.rect.width / length
    return (span.rect.x0 + unit * start, span.rect.x0 + unit * end)


def _served_below(ctx: CandidateContext, span: TextSpan) -> bool:
    """True when ``span``'s field is printed underneath it rather than beside it.

    A label owns one field.  When the page has put a box, a rule or writing space on the
    line below the label -- the stacked layout, and the borderless idiom -- the space to
    the right of the label is a margin, not an entry area, and the colon at the end of
    the label is punctuation.
    """
    top = span.rect.y0
    width = max(span.rect.width, EPS)
    # A label-anchored region is built flush under its own label, so it must line up.
    for region in ctx.label_blanks:
        if abs(region.y1 - (top - 2.0)) > 4.0:
            continue
        if region.horizontal_overlap(span.rect) >= 0.5 * width:
            return True
    floor = top - 1.5 * ctx.body_font_size
    for other in claimed_rects(ctx):
        if other.y1 > top - 0.5 or other.y1 < floor:
            continue
        if other.horizontal_overlap(span.rect) >= 0.5 * width:
            return True
    return False


def _first_mark_right_of(ctx: CandidateContext, span: TextSpan) -> float:
    """Where the printed marks resume to the right of ``span`` on its own row.

    A comb, a box or a rule beginning just past the colon means the field is already
    drawn; the text detector must not propose a second one over the top of it.
    """
    band = Rect(span.rect.x1, span.rect.y0 - 3.0, ctx.page_rect.x1, span.rect.y1)
    limit = ctx.text_column.x1 if ctx.text_column else ctx.page_rect.x1
    for prim in ctx.primitives:
        rect = prim.rect
        if rect.x1 <= span.rect.x1 + 0.5:
            continue
        if rect.vertical_overlap(band.inflated(0.0, 2.0)) <= 0.0:
            continue
        limit = min(limit, max(rect.x0, span.rect.x1))
    return limit


def _has_rule_under(ctx: CandidateContext, rect: Rect) -> bool:
    """True when a vector rule already underlines ``rect`` (so a rule detector owns it)."""
    band = Rect(rect.x0, rect.y0 - 1.5 * ctx.body_font_size, rect.x1, rect.y1)
    for rule in ctx.h_rules:
        if rule.rect.horizontal_overlap(rect) < 0.5 * min(rule.rect.width, rect.width):
            continue
        y = (rule.rect.y0 + rule.rect.y1) / 2.0
        if band.y0 <= y <= band.y1:
            return True
    return False


def _date_format_hint(sizes: Sequence[int], separator: str) -> str:
    """Format hint for a three-group date, e.g. ``(2, 2, 4)`` -> ``MM/DD/YYYY``."""
    if list(sizes) == [2, 2, 4]:
        return separator.join(("MM", "DD", "YYYY"))
    if list(sizes) == [4, 2, 2]:
        return separator.join(("YYYY", "MM", "DD"))
    if list(sizes) == [2, 2, 2]:
        return separator.join(("MM", "DD", "YY"))
    return separator.join("N" * max(1, int(n)) for n in sizes)


# ------------------------------------------------------------------------ orchestration
#: The eleven detectors, in the order :func:`generate_candidates` runs them.
DEFAULT_DETECTORS: List[Any] = [
    UnderlineFieldDetector(),
    BoxFieldDetector(),
    CheckboxDetector(),
    RadioGroupDetector(),
    CombFieldDetector(),
    DateBoxDetector(),
    SignatureLineDetector(),
    TableCellDetector(),
    BlankRegionDetector(),
    FreeTextAreaDetector(),
    ColonRunDetector(),
]


def generate_candidates(
    ctx: CandidateContext, detectors: Optional[Sequence[Any]] = None
) -> List[FieldCandidate]:
    """Run every archetype detector over one page and return their candidates.

    A detector that raises is logged and skipped: one bad archetype must never cost the
    run the other ten.  Every piece of evidence is tagged with the detector that found
    it, and the result is sorted into reading order -- top to bottom, then left to right
    -- with :attr:`FieldCandidate.order` set to match.

    Overlapping proposals are **kept**.  Deduplication belongs to :mod:`zfp.fusion`,
    which needs to see the agreement between archetypes to score it.
    """
    chosen = list(DEFAULT_DETECTORS if detectors is None else detectors)
    produced: List[FieldCandidate] = []
    for detector in chosen:
        name = str(getattr(detector, "name", type(detector).__name__))
        try:
            found = list(detector.detect(ctx) or [])
        except Exception as exc:
            LOG.warning("archetype detector %r failed on page %d: %s", name, ctx.page, exc)
            continue
        for candidate in found:
            if candidate is None:
                continue
            candidate.evidence = [
                e if e.source_agent else replace(e, source_agent=name) for e in candidate.evidence
            ]
            produced.append(candidate)

    produced.sort(key=lambda c: (c.page, -c.rect.y1, c.rect.x0, c.rect.x1, c.id))
    for index, candidate in enumerate(produced):
        candidate.order = index
    return produced
