"""The label linker: deciding which printed words name which blank.

Geometry says *there is a field here*.  Only the words around it say *what it is*, and
the words are not attached to anything -- a form is a picture, not a data structure.  The
linker scores every nearby span as a possible label and then performs one **global**
assignment, because the naive per-field argmax breaks on the layout forms use most::

    First Name: ______________   Last Name: ______________

Both fields see "Last Name" as a plausible label (it is left of the second field and
right of the first); independent argmax lets one span be claimed twice and leaves a field
unnamed.  Sorting every ``(field, span, score)`` triple and consuming each span once
fixes that, and costs nothing.

Scoring, per relation, distance-decayed and then adjusted:

===================  ======  =========================================================
Relation             Base    Note
===================  ======  =========================================================
left of, same row    1.00    the normal western form layout
above                0.80    column headers, stacked labels
right of             0.45    ...but **1.00** for a checkbox or radio: ``[ ] Yes``
below                0.25    rare, usually a caption
===================  ======  =========================================================

Bonuses reward label-shaped spans (trailing colon, a hit in the ontology, a mutual
nearest-neighbour relationship); penalties push away spans that are really values,
section headings, prose, or text sitting inside another field's rectangle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..core.config import DetectionConfig
from ..core.geometry import EPS, Rect
from ..core.logging import get_logger
from ..core.types import Evidence, EvidenceKind, FieldCandidate, FieldType, TextSpan
from ..ontology import lookup as ontology_lookup
from ..ontology import match_value
from .graph import (
    NODE_LABEL,
    NODE_SECTION,
    VERTICAL_GAP_LINES,
    RelationKind,
    SpatialGraph,
    decay_weight,
    detection_config,
    median_line_height,
    spatial_relation,
)
from .sections import Section, section_for

__all__ = [
    "link_labels",
    "link_stem_labels",
    "score_label",
    "clean_label",
    "looks_like_value",
    "is_option_caption",
    "LabelLink",
    "BASE_SCORES",
    "TOGGLE_SCORES",
    "BONUS_COLON",
    "BONUS_ONTOLOGY",
    "BONUS_MUTUAL",
    "PENALTY_INSIDE_FIELD",
    "PENALTY_VALUE_LIKE",
    "PENALTY_SECTION_HEADING",
    "PENALTY_VERY_LONG",
    "MIN_LABEL_SCORE",
    "LONG_LABEL_CHARS",
    "SOURCE_AGENT",
]

LOG = get_logger(__name__)

#: Relation -> base score for an ordinary (text-ish) field.
BASE_SCORES: Dict[RelationKind, float] = {
    RelationKind.LEFT_OF: 1.00,
    RelationKind.ABOVE: 0.80,
    RelationKind.RIGHT_OF: 0.45,
    RelationKind.BELOW: 0.25,
}
#: Relation -> base score for a checkbox or radio, where the caption sits on the right.
TOGGLE_SCORES: Dict[RelationKind, float] = {
    RelationKind.RIGHT_OF: 1.00,
    RelationKind.ABOVE: 0.80,
    RelationKind.LEFT_OF: 0.55,
    RelationKind.BELOW: 0.25,
}
#: Field types whose caption is printed to the right of the mark.
TOGGLE_TYPES = (FieldType.CHECKBOX, FieldType.RADIO)

BONUS_COLON = 0.25
BONUS_ONTOLOGY = 0.30
BONUS_MUTUAL = 0.15
PENALTY_INSIDE_FIELD = 0.50
PENALTY_VALUE_LIKE = 0.40
PENALTY_SECTION_HEADING = 0.30
PENALTY_VERY_LONG = 0.20
#: A span longer than this reads as prose, not as a label.
LONG_LABEL_CHARS = 60
#: Below this score a span is not worth calling a label.
MIN_LABEL_SCORE = 0.10
#: Written into every :class:`~zfp.core.types.Evidence` this module produces.
SOURCE_AGENT = "semantics.linker"

_DIGITS_RE = re.compile(r"^[\d\s.,/$%()+-]+$")
_WS_RE = re.compile(r"\s+")
_LEADING_DECORATION = re.compile(r"^[\s\-•●▪·*>#]+")
_TRAILING_DECORATION = re.compile(r"[\s:;*_.…]+$")

try:  # pragma: no cover - the fallback only runs on a broken install
    from ..candidates.archetypes import clean_label as _archetype_clean_label
except Exception:  # pragma: no cover
    _archetype_clean_label = None  # type: ignore[assignment]


def clean_label(text: Optional[str]) -> str:
    """Strip a printed label down to the human form: ``"  ZIP: "`` -> ``"ZIP"``.

    Delegates to :func:`zfp.candidates.archetypes.clean_label` so detection and the
    semantic layer always print a label the same way, with an equivalent local fallback
    when that module is unavailable.
    """
    if _archetype_clean_label is not None:
        return _archetype_clean_label(text)
    if not text:  # pragma: no cover - exercised only without zfp.candidates
        return ""
    stripped = _WS_RE.sub(" ", str(text)).strip()
    stripped = _LEADING_DECORATION.sub("", stripped)
    stripped = _TRAILING_DECORATION.sub("", stripped)
    return stripped.strip()


@dataclass
class LabelLink:
    """One scored ``(candidate, span)`` pairing considered by the assignment."""

    candidate_index: int
    span_index: int
    relation: RelationKind
    gap: float
    score: float
    reasons: List[str]

    def sort_key(self) -> Tuple[float, int, int]:
        """Descending score, then a stable tie-break on the two indices."""
        return (-self.score, self.candidate_index, self.span_index)


# --------------------------------------------------------------------------- scoring
def looks_like_value(text: str) -> bool:
    """True when a span reads as a filled-in value rather than as a label.

    ``"10001"`` and ``"12-3456789"`` are values; ``"ZIP:"`` and ``"Tax ID"`` are labels.
    """
    stripped = (text or "").strip()
    if not stripped:
        return False
    if _DIGITS_RE.match(stripped) and any(ch.isdigit() for ch in stripped):
        return True
    return match_value(stripped) is not None


#: Value rules whose "value" is really a checkbox caption.
OPTION_VALUE_RULES = ("yes_no_value", "checkbox_mark_value")


def is_option_caption(text: str) -> bool:
    """True for ``Yes`` / ``No`` / ``N/A`` beside a box: a caption, not a stray value."""
    rule = match_value((text or "").strip())
    return rule is not None and rule.name in OPTION_VALUE_RULES


def score_label(
    span: TextSpan,
    relation: RelationKind,
    gap: float,
    field_type: FieldType,
    max_gap: float,
    vertical_scale: float,
    *,
    inside_other_field: bool = False,
    is_section_heading: bool = False,
    mutual_nearest: bool = False,
) -> Tuple[float, List[str]]:
    """Score one span as the label of one field.

    Args:
        span: The candidate label.
        relation: How the span stands to the field (``LEFT_OF`` = span is on the left).
        gap: Separation in points along the relation's axis.
        field_type: The field's type; checkboxes and radios invert left/right.
        max_gap: Horizontal decay scale, ``DetectionConfig.label_max_distance_pt``.
        vertical_scale: Vertical decay scale, ~2.5 line heights.
        inside_other_field: The span sits inside a different candidate's rectangle.
        is_section_heading: The span is part of a detected section title.
        mutual_nearest: The span is this field's closest label *and* this field is the
            span's closest field.

    Returns:
        ``(score, reasons)`` with the score clamped into ``[0, 1]``.
    """
    table = TOGGLE_SCORES if field_type in TOGGLE_TYPES else BASE_SCORES
    base = table.get(relation)
    if base is None:
        return (0.0, [])
    horizontal = relation in (RelationKind.LEFT_OF, RelationKind.RIGHT_OF)
    scale = max_gap if horizontal else vertical_scale
    score = base * decay_weight(gap, scale)
    reasons = ["%s_%.2f" % (relation.value, base)]

    text = (span.text or "").strip()
    if text.endswith(":"):
        score += BONUS_COLON
        reasons.append("trailing_colon")
    if ontology_lookup(text) is not None:
        score += BONUS_ONTOLOGY
        reasons.append("ontology_hit")
    if mutual_nearest:
        score += BONUS_MUTUAL
        reasons.append("mutual_nearest")

    if inside_other_field:
        score -= PENALTY_INSIDE_FIELD
        reasons.append("inside_other_field")
    if looks_like_value(text) and not (
        field_type in TOGGLE_TYPES and is_option_caption(text)
    ):
        score -= PENALTY_VALUE_LIKE
        reasons.append("value_like")
    if is_section_heading:
        score -= PENALTY_SECTION_HEADING
        reasons.append("section_heading")
    if len(text) > LONG_LABEL_CHARS:
        score -= PENALTY_VERY_LONG
        reasons.append("very_long")

    return (min(1.0, max(0.0, score)), reasons)


# --------------------------------------------------------------------------- indexing
def _span_key(span: TextSpan) -> Tuple[int, Tuple[float, ...], str]:
    """A value key that identifies a span independently of object identity."""
    rect = span.rect.rounded(3)
    return (int(span.page), tuple(rect.as_list()), span.text)


class _Index:
    """Everything the scorer needs to look at, computed once per call."""

    def __init__(
        self,
        candidates: Sequence[FieldCandidate],
        spans: Sequence[TextSpan],
        graph: Optional[SpatialGraph],
        det: DetectionConfig,
        sections: Sequence[Section],
    ) -> None:
        self.candidates = [c for c in candidates or () if c is not None]
        self.spans = [s for s in spans or () if s is not None and not s.is_blank()]
        self.graph = graph
        self.det = det
        self.max_gap = float(det.label_max_distance_pt)
        self.line_h = median_line_height(self.spans)
        self.vertical_gap = max(VERTICAL_GAP_LINES * self.line_h, 1.0)
        self.vertical_scale = max(VERTICAL_GAP_LINES * self.line_h, 10.0)

        self.by_identity: Dict[int, int] = {}
        self.by_key: Dict[Tuple[int, Tuple[float, ...], str], int] = {}
        self.by_page: Dict[int, List[int]] = {}
        for index, span in enumerate(self.spans):
            self.by_identity[id(span)] = index
            self.by_key.setdefault(_span_key(span), index)
            self.by_page.setdefault(int(span.page), []).append(index)

        self.sections = _dedupe_sections(list(sections) + _graph_sections(graph))
        self.heading_spans = _heading_spans(self.spans, self.sections)
        self.candidate_rects: Dict[int, List[Tuple[int, Rect]]] = {}
        for index, candidate in enumerate(self.candidates):
            self.candidate_rects.setdefault(int(candidate.page), []).append(
                (index, candidate.rect.normalized())
            )

    def span_index(self, span: TextSpan) -> Optional[int]:
        """Position of ``span`` in the working list, by identity then by value."""
        found = self.by_identity.get(id(span))
        if found is not None:
            return found
        return self.by_key.get(_span_key(span))

    def inside_other_field(self, span_index: int, candidate_index: int) -> bool:
        """True when this span sits inside a *different* candidate's rectangle."""
        span = self.spans[span_index]
        for index, rect in self.candidate_rects.get(int(span.page), ()):
            if index == candidate_index:
                continue
            if _mostly_inside(rect, span.rect):
                return True
        return False

    def neighbours(
        self, candidate: FieldCandidate, target: Optional[Rect] = None
    ) -> List[Tuple[int, RelationKind, float]]:
        """Return ``(span index, relation, gap)`` for every span near ``candidate``.

        The spatial graph supplies the shortlist when it knows this candidate; the
        relation and the gap are always recomputed from the rectangles so the result
        cannot drift from the caller's configuration.
        """
        rect = (target or candidate.rect).normalized()
        found: List[Tuple[int, RelationKind, float]] = []
        seen: Set[int] = set()
        # A group bounding box is not a graph node, so its neighbours must be scanned.
        shortlist = None if target is not None else self._graph_shortlist(candidate)
        if shortlist is None:
            shortlist = self.by_page.get(int(candidate.page), [])
        for span_index in shortlist:
            if span_index in seen:
                continue
            seen.add(span_index)
            span = self.spans[span_index]
            if int(span.page) != int(candidate.page):
                continue
            relation = spatial_relation(
                span.rect, rect, self.max_gap, self.vertical_gap
            )
            if relation is None:
                continue
            found.append((span_index, relation[0], relation[1]))
        found.sort(key=lambda item: (item[2], item[0]))
        return found

    def _graph_shortlist(self, candidate: FieldCandidate) -> Optional[List[int]]:
        """Span indices the graph reports as adjacent, or ``None`` when it cannot help."""
        graph = self.graph
        if graph is None or not candidate.id or not graph.has_node(candidate.id):
            return None
        out: List[int] = []
        for relation in (
            RelationKind.LEFT_OF,
            RelationKind.RIGHT_OF,
            RelationKind.ABOVE,
            RelationKind.BELOW,
        ):
            for edge in graph.incoming(candidate.id, relation):
                node = graph.node(edge.src)
                if node is None or node.kind != NODE_LABEL:
                    continue
                payload = graph.payload(edge.src)
                index = self.span_index(payload) if isinstance(payload, TextSpan) else None
                if index is None:
                    index = self.by_key.get((node.page, tuple(node.rect.rounded(3).as_list()), node.text))
                if index is not None:
                    out.append(index)
        return out


def _mostly_inside(outer: Rect, inner: Rect) -> bool:
    """True when ``outer`` holds at least 60% of ``inner``."""
    if inner.area <= EPS:
        return outer.contains_point(inner.center)
    overlap = outer.intersection(inner)
    return overlap is not None and overlap.area >= 0.6 * inner.area


def _graph_sections(graph: Optional[SpatialGraph]) -> List[Section]:
    """Recover ``Section`` objects from a graph's section nodes."""
    if graph is None:
        return []
    out: List[Section] = []
    for node in graph.nodes_of(NODE_SECTION):
        payload = graph.payload(node.id)
        if isinstance(payload, Section):
            out.append(payload)
            continue
        out.append(Section(title=node.text, rect=node.rect, level=1, page=node.page))
    return out


def _dedupe_sections(sections: Sequence[Section]) -> List[Section]:
    """Drop duplicate sections, keeping the first of each ``(page, title, rect)``."""
    seen: Set[Tuple[int, str, Tuple[float, ...]]] = set()
    out: List[Section] = []
    for section in sections:
        if section is None or section.rect is None:
            continue
        key = (int(section.page), section.title, tuple(section.rect.rounded(3).as_list()))
        if key in seen:
            continue
        seen.add(key)
        out.append(section)
    return out


def _heading_spans(spans: Sequence[TextSpan], sections: Sequence[Section]) -> Set[int]:
    """Indices of spans that are part of a section title rather than a field label."""
    out: Set[int] = set()
    if not sections:
        return out
    for index, span in enumerate(spans):
        for section in sections:
            title_rect = section.title_rect or section.rect
            if int(section.page) != int(span.page) or title_rect is None:
                continue
            if _mostly_inside(title_rect, span.rect):
                out.add(index)
                break
    return out


# ------------------------------------------------------------------------- evidence
def _clear_link_evidence(candidate: FieldCandidate) -> None:
    """Drop LABEL_LINK evidence this module wrote earlier, so relinking is idempotent."""
    candidate.evidence = [
        e
        for e in candidate.evidence
        if not (e.kind == EvidenceKind.LABEL_LINK and e.source_agent == SOURCE_AGENT)
    ]


def _apply_label(
    candidate: FieldCandidate,
    span: TextSpan,
    score: float,
    relation: RelationKind,
    gap: float,
    detail_prefix: str = "",
) -> None:
    """Write the winning label onto a candidate, with its evidence."""
    text = clean_label(span.text)
    if not text:
        return
    _clear_link_evidence(candidate)
    candidate.visible_label = text
    candidate.confidence.label_link = round(min(1.0, max(0.0, score)), 6)
    detail = "%s%s %r at %.1f pt" % (detail_prefix, relation.value, text, gap)
    candidate.add_evidence(
        Evidence(
            kind=EvidenceKind.LABEL_LINK,
            score=round(min(1.0, max(0.0, score)), 6),
            detail=detail,
            source_agent=SOURCE_AGENT,
            rect=span.rect,
        )
    )


# ----------------------------------------------------------------------------- API
def link_labels(
    candidates: Sequence[FieldCandidate],
    spans: Sequence[TextSpan],
    graph: Optional[SpatialGraph] = None,
    config: Any = None,
    *,
    sections: Sequence[Section] = (),
) -> List[FieldCandidate]:
    """Attach a visible label and a parent context to every candidate.

    Radio and checkbox groups are linked first, through :func:`link_stem_labels`: every
    member of a ``group_id`` shares the stem label printed before the whole group, and
    that stem is then withheld from the general assignment.  The remaining candidates go
    through one greedy global assignment over every scored ``(field, span)`` pair, so no
    span is used twice.

    Args:
        candidates: The candidates to label; they are mutated in place.
        spans: Every text span on the page(s).
        graph: The spatial graph from :func:`zfp.semantics.graph.build_graph`.  Optional:
            without it the neighbour search falls back to a per-page linear scan.
        config: A :class:`~zfp.core.config.ZfpConfig` or
            :class:`~zfp.core.config.DetectionConfig`.
        sections: Extra sections to consider for ``parent_context``; sections already
            present in ``graph`` are picked up automatically.

    Returns:
        The same candidate objects, in the order they were given.
    """
    det = detection_config(config)
    index = _Index(candidates, spans, graph, det, sections)
    if not index.candidates:
        return list(candidates or ())

    consumed: Set[int] = set()
    grouped = {i for i, c in enumerate(index.candidates) if c.group_id}
    if grouped:
        consumed |= _link_stems(index)

    links = _collect_links(index, skip=grouped, consumed=consumed)
    _assign(index, links, consumed)
    _apply_parent_context(index)
    return list(candidates or ())


def link_stem_labels(
    candidates: Sequence[FieldCandidate],
    spans: Sequence[TextSpan],
    graph: Optional[SpatialGraph] = None,
    config: Any = None,
    *,
    sections: Sequence[Section] = (),
) -> List[FieldCandidate]:
    """Give every member of a radio/checkbox group the group's shared stem label.

    ``Marital status:  ( ) Single  ( ) Married  ( ) Divorced`` is one field with three
    options, not three fields.  The stem ("Marital status:") is found relative to the
    *whole group's* bounding box -- left of it, or above it -- never relative to a single
    option, whose neighbouring word is its export value instead.

    Args:
        candidates: Candidates; only those with a ``group_id`` are touched.
        spans: Every text span on the page(s).
        graph: The spatial graph; optional.
        config: A :class:`~zfp.core.config.ZfpConfig` or
            :class:`~zfp.core.config.DetectionConfig`.
        sections: Extra sections for heading detection.

    Returns:
        The same candidate objects, in the order they were given.
    """
    det = detection_config(config)
    index = _Index(candidates, spans, graph, det, sections)
    _link_stems(index)
    return list(candidates or ())


# ------------------------------------------------------------------------ internals
def _collect_links(
    index: _Index, skip: Set[int], consumed: Set[int]
) -> List[LabelLink]:
    """Score every plausible ``(candidate, span)`` pair, mutual-nearest bonus included."""
    raw: List[Tuple[int, int, RelationKind, float]] = []
    for candidate_index, candidate in enumerate(index.candidates):
        if candidate_index in skip:
            continue
        for span_index, relation, gap in index.neighbours(candidate):
            if span_index in consumed:
                continue
            raw.append((candidate_index, span_index, relation, gap))

    nearest_span: Dict[int, Tuple[float, int]] = {}
    nearest_field: Dict[int, Tuple[float, int]] = {}
    for candidate_index, span_index, _relation, gap in raw:
        best = nearest_span.get(candidate_index)
        if best is None or (gap, span_index) < best:
            nearest_span[candidate_index] = (gap, span_index)
        best = nearest_field.get(span_index)
        if best is None or (gap, candidate_index) < best:
            nearest_field[span_index] = (gap, candidate_index)

    links: List[LabelLink] = []
    for candidate_index, span_index, relation, gap in raw:
        candidate = index.candidates[candidate_index]
        mutual = (
            nearest_span.get(candidate_index, (0.0, -1))[1] == span_index
            and nearest_field.get(span_index, (0.0, -1))[1] == candidate_index
        )
        score, reasons = score_label(
            index.spans[span_index],
            relation,
            gap,
            candidate.field_type,
            index.max_gap,
            index.vertical_scale,
            inside_other_field=index.inside_other_field(span_index, candidate_index),
            is_section_heading=span_index in index.heading_spans,
            mutual_nearest=mutual,
        )
        if score < MIN_LABEL_SCORE:
            continue
        links.append(LabelLink(candidate_index, span_index, relation, gap, score, reasons))
    links.sort(key=lambda link: link.sort_key())
    return links


def _assign(index: _Index, links: Sequence[LabelLink], consumed: Set[int]) -> None:
    """Greedy global assignment: best pair first, every span used at most once."""
    taken_candidates: Set[int] = set()
    span_owner: Dict[int, int] = {}
    for link in links:
        if link.candidate_index in taken_candidates:
            continue
        owner = span_owner.get(link.span_index)
        if owner is not None and not _may_share(index, owner, link.candidate_index):
            continue
        if link.span_index in consumed:
            continue
        candidate = index.candidates[link.candidate_index]
        _apply_label(
            candidate,
            index.spans[link.span_index],
            link.score,
            link.relation,
            link.gap,
        )
        taken_candidates.add(link.candidate_index)
        span_owner.setdefault(link.span_index, link.candidate_index)
        LOG.debug(
            "linked %s -> %r (%s, %.3f, %s)",
            candidate.id,
            candidate.visible_label,
            link.relation.value,
            link.score,
            ",".join(link.reasons),
        )


def _may_share(index: _Index, owner_index: int, other_index: int) -> bool:
    """Two candidates may share one span only when they are options of one group."""
    owner = index.candidates[owner_index]
    other = index.candidates[other_index]
    return bool(owner.group_id) and owner.group_id == other.group_id


def _apply_parent_context(index: _Index) -> None:
    """Set ``parent_context`` from the enclosing section chain, outermost first."""
    if not index.sections:
        return
    for candidate in index.candidates:
        chain = section_for(candidate.rect, index.sections, page=int(candidate.page))
        if chain:
            candidate.parent_context = [section.title for section in chain]


def _link_stems(index: _Index) -> Set[int]:
    """Label every group from its stem; return the span indices that were consumed."""
    groups: Dict[str, List[int]] = {}
    for position, candidate in enumerate(index.candidates):
        if candidate.group_id:
            groups.setdefault(str(candidate.group_id), []).append(position)

    consumed: Set[int] = set()
    for group_id in sorted(groups):
        members = groups[group_id]
        rects = [index.candidates[i].rect for i in members]
        bbox = Rect.bounding(rects)
        if bbox is None:
            continue
        anchor = index.candidates[members[0]]
        best: Optional[Tuple[float, int, RelationKind, float]] = None
        for span_index, relation, gap in index.neighbours(anchor, target=bbox):
            if span_index in consumed:
                continue
            span = index.spans[span_index]
            if relation is RelationKind.LEFT_OF and span.rect.x1 > bbox.x0 + EPS:
                continue
            if relation is RelationKind.ABOVE and span.rect.y0 < bbox.y1 - EPS:
                continue
            if relation in (RelationKind.RIGHT_OF, RelationKind.BELOW):
                continue
            score, _reasons = score_label(
                span,
                relation,
                gap,
                FieldType.TEXT,  # the stem names the group, not one option
                index.max_gap,
                index.vertical_scale,
                inside_other_field=index.inside_other_field(span_index, members[0]),
                is_section_heading=span_index in index.heading_spans,
            )
            if score < MIN_LABEL_SCORE:
                continue
            key = (-score, span_index, relation, gap)
            if best is None or key < best:
                best = key
        if best is None:
            continue
        score, span_index, relation, gap = -best[0], best[1], best[2], best[3]
        consumed.add(span_index)
        for position in members:
            _apply_label(
                index.candidates[position],
                index.spans[span_index],
                score,
                relation,
                gap,
                detail_prefix="stem ",
            )
    return consumed
