"""The canonicalization cascade: a printed label becomes a canonical key, or nothing.

Everything downstream -- the vault lookup, the validators, the QA metrics -- speaks
canonical keys, so this is the module that decides what a field *is*.  It is
deliberately, completely deterministic: eight ordered steps, no model, no network, and
the same answer on every machine.  When none of the eight fires the answer is ``None``,
which is an honest "I do not know" that the council can escalate later.

The cascade, best evidence first:

===  ==============================================================  =====
#    Step                                                            Score
===  ==============================================================  =====
1    exact alias hit on the visible label                            0.97
2    section-scoped alias hit (billing vs shipping)                  0.95
2b   the neighbourhood disambiguates an ambiguous label              0.93
3    a printed placeholder in or beside the blank (``##-#######``)   0.90
4    label concatenated with its parent context                      0.88
5    label plus nearby text, through the pattern inferencer          0.82
6    fuzzy alias match above the cutoff                              ratio x 0.85
7    sibling inference inside a resolved namespace                   0.70
8    nothing                                                         0.00
===  ==============================================================  =====

Steps 2 and 2b are *refinements* of step 1 rather than competitors: the exact alias is
computed first, and an explicit section context (2) or, failing that, the surrounding
words (2b) may overrule it with a more specific key.  That is the "nearby terms
influence interpretation" rule -- a bare *Address* printed beside *City* and *State* is
a postal address, the same word beside *E-mail* is not.  Step 7 needs the whole page, so
it lives in :func:`canonicalize_all` together with the conflict pass that stops two
blanks on one page claiming ``person.name.first``.
"""

from __future__ import annotations

import difflib
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..core.geometry import EPS, Rect
from ..core.logging import get_logger
from ..core.types import Evidence, EvidenceKind, FieldCandidate, TextSpan
from ..ontology import children as ontology_children
from ..ontology import context_lookup, fuzzy_lookup, infer_from_context
from ..ontology import get as ontology_get
from ..ontology import lookup as ontology_lookup
from ..ontology import match_placeholder, normalize_label
from .graph import RelationKind, SpatialGraph, detection_config
from .typing import ADJACENT_LINES, body_line_height, nearby_spans

__all__ = [
    "canonicalize",
    "canonicalize_all",
    "disambiguate",
    "neighbourhood_texts",
    "sibling_key",
    "AMBIGUOUS_LABELS",
    "HINT_TOKENS",
    "SCORE_EXACT",
    "SCORE_CONTEXT",
    "SCORE_NEIGHBOUR",
    "SCORE_PLACEHOLDER",
    "SCORE_CONCAT",
    "SCORE_INFERENCE",
    "SCORE_FUZZY_FACTOR",
    "SCORE_SIBLING",
    "FUZZY_CUTOFF",
    "SIBLING_LABEL_CUTOFF",
    "CONFLICT_MARGIN",
    "REPEATABLE_LEAVES",
    "SOURCE_AGENT",
]

LOG = get_logger(__name__)

SCORE_EXACT = 0.97
SCORE_CONTEXT = 0.95
SCORE_NEIGHBOUR = 0.93
SCORE_PLACEHOLDER = 0.90
SCORE_CONCAT = 0.88
SCORE_INFERENCE = 0.82
SCORE_FUZZY_FACTOR = 0.85
SCORE_SIBLING = 0.70
#: A fuzzy alias match below this ratio is noise, not a spelling variant.
FUZZY_CUTOFF = 0.86
#: How close a label must be to a namespace sibling's own label to inherit it.
SIBLING_LABEL_CUTOFF = 0.62
#: Two candidates keep the same key unless one scores at least this much lower.
CONFLICT_MARGIN = 0.08
#: How many neighbouring texts are collected around a candidate.
MAX_NEIGHBOURS = 24
#: Key leaves that legitimately repeat many times on one page.
REPEATABLE_LEAVES = frozenset({"signature", "initials", "date", "page_number"})
#: Written into every :class:`~zfp.core.types.Evidence` this module produces.
SOURCE_AGENT = "semantics.normalizer"

#: Labels whose meaning genuinely depends on what is printed around them, and the
#: readings they can take.  The first entry is the default reading.
_AMBIGUOUS_RAW: Dict[str, Tuple[str, ...]] = {
    "address": ("person.address.street_1", "person.email", "company.website"),
    "number": (
        "document.number",
        "person.phone.mobile",
        "bank.account_number",
        "card.number",
        "insurance.policy_number",
        "person.driver_license.number",
    ),
    "id": ("document.number", "employment.employee_id", "insurance.member_id"),
    "name": ("person.name.full", "company.legal_name", "bank.name"),
    "code": ("person.address.postal_code", "medical.diagnosis_code"),
    "date": ("document.today", "person.date_of_birth", "document.signed_date"),
    "type": ("bank.account_type", "insurance.plan_name"),
    "state": ("person.address.region", "person.driver_license.state"),
    "account": ("bank.account_number", "document.account_number"),
}
#: The same table, filtered to keys that actually exist in the ontology.
AMBIGUOUS_LABELS: Dict[str, Tuple[str, ...]] = {
    label: tuple(key for key in keys if ontology_get(key) is not None)
    for label, keys in _AMBIGUOUS_RAW.items()
}
#: Discriminating words looked for in the neighbourhood, in priority order.
HINT_TOKENS: Tuple[str, ...] = (
    "email", "e mail", "mail", "web", "website", "url",
    "billing", "shipping", "mailing", "home", "work", "business", "employer", "company",
    "street", "city", "state", "province", "zip", "postal", "county", "country",
    "apt", "unit", "suite",
    "phone", "telephone", "mobile", "cell", "fax",
    "bank", "account", "routing", "card", "credit",
    "policy", "insurance", "member", "group",
    "license", "driver", "vehicle", "student", "employee", "patient", "tax",
    "birth", "signature",
)


# --------------------------------------------------------------------------- context
def neighbourhood_texts(
    candidate: FieldCandidate,
    spans: Sequence[TextSpan] = (),
    config: Any = None,
    limit: int = MAX_NEIGHBOURS,
) -> List[str]:
    """Return the printed texts around a candidate, nearest first.

    The band is deliberately wider than the label search: the words that disambiguate
    *Address* are its row neighbours *City* and *State*, not something touching the
    blank.  Ordering is by distance from the candidate's centre with a text tie-break,
    so the result never depends on input order.
    """
    det = detection_config(config)
    rect = candidate.rect.normalized()
    live = [
        s
        for s in spans or ()
        if s is not None and not s.is_blank() and int(s.page) == int(candidate.page)
    ]
    if not live:
        return []
    line_h = body_line_height(live, candidate.page)
    band = rect.inflated(2.0 * det.label_max_distance_pt, 2.5 * line_h)
    centre = rect.center
    scored: List[Tuple[float, str, str]] = []
    for span in live:
        if span.rect.intersection(band) is None:
            continue
        text = (span.text or "").strip()
        if not text:
            continue
        scored.append((round(centre.distance_to(span.rect.center), 4), text, text))
    scored.sort(key=lambda item: (item[0], item[1]))
    out: List[str] = []
    for _distance, text, _raw in scored:
        if text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def disambiguate(label: str, nearby: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Pick the reading of an ambiguous label that the neighbourhood supports.

    Two mechanisms, in order:

    1. **Compound alias.**  A hint word found nearby is glued onto the label and looked
       up: *Address* beside *E-mail* becomes ``"email address"`` -> ``person.email``.
    2. **Namespace vote.**  Each neighbouring text is resolved on its own; whichever
       reading of the label shares a namespace with the most neighbours wins.  *Address*
       beside *City*, *State* and *ZIP* stays a postal address.

    Args:
        label: The field's visible label.
        nearby: Printed texts around the field, nearest first.

    Returns:
        ``(canonical_key, hint)`` or ``None`` when the label is unambiguous or the
        neighbourhood says nothing.
    """
    norm = normalize_label(label)
    readings = AMBIGUOUS_LABELS.get(norm)
    if not readings or not nearby:
        return None

    tokens = _hint_tokens(nearby)

    for token in tokens:
        for phrase in ("%s %s" % (token, norm), "%s %s" % (norm, token)):
            found = ontology_lookup(phrase)
            if found is not None and found in readings:
                return (found, token)
    for token in tokens:
        found = ontology_lookup("%s %s" % (token, norm))
        if found is not None:
            return (found, token)

    votes: Dict[str, int] = {}
    for text in nearby:
        resolved = ontology_lookup(text)
        if resolved is None:
            continue
        for reading in readings:
            if _shares_namespace(reading, resolved):
                votes[reading] = votes.get(reading, 0) + 1
    if not votes:
        return None
    best = sorted(votes.items(), key=lambda item: (-item[1], readings.index(item[0])))[0]
    if best[1] < 1:
        return None
    return (best[0], "namespace_vote")


def _has_token(haystack: str, token: str) -> bool:
    """Whole-word containment over normalized text."""
    return (" %s " % token) in (" %s " % haystack)


def _hint_tokens(nearby: Sequence[str]) -> List[str]:
    """Discriminating words present in ``nearby``, nearest neighbour first.

    Ordering is ``(index of the text the word appears in, position in HINT_TOKENS)`` so
    a hint printed right beside the field outranks one printed across the page, and the
    result never depends on dictionary iteration order.
    """
    ranked: List[Tuple[int, int, str]] = []
    seen: Set[str] = set()
    for position, text in enumerate(nearby):
        normalized = normalize_label(text)
        if not normalized:
            continue
        for index, token in enumerate(HINT_TOKENS):
            if token in seen or not _has_token(normalized, token):
                continue
            seen.add(token)
            ranked.append((position, index, token))
    ranked.sort()
    return [token for _position, _index, token in ranked]


def _shares_namespace(a: str, b: str) -> bool:
    """True when two keys agree on their first two dotted components."""
    return _prefix(a, 2) == _prefix(b, 2)


def _prefix(key: str, depth: int) -> str:
    """The first ``depth`` dotted components of a key."""
    return ".".join(key.split(".")[:depth])


# ------------------------------------------------------------------------- cascade
def canonicalize(
    candidate: FieldCandidate,
    config: Any = None,
    *,
    spans: Sequence[TextSpan] = (),
) -> Tuple[Optional[str], float, List[str]]:
    """Map one candidate onto a canonical key, deterministically.

    Args:
        candidate: The candidate; it is **not** mutated.  Its ``visible_label`` and
            ``parent_context`` carry the evidence, and its ``constraints.format_hint``
            is consulted when no spans are supplied.
        config: A :class:`~zfp.core.config.ZfpConfig` or
            :class:`~zfp.core.config.DetectionConfig`; optional.
        spans: The page's text spans, used for placeholder and neighbourhood evidence.

    Returns:
        ``(canonical_key, confidence, reason_codes)``.  ``canonical_key`` is ``None``
        with confidence ``0.0`` and reason ``["unresolved"]`` when nothing fires.

    Examples:
        >>> from zfp.core.geometry import Rect
        >>> from zfp.core.types import FieldCandidate
        >>> c = FieldCandidate("f", 0, Rect(0, 0, 80, 12), visible_label="ZIP / Postal")
        >>> canonicalize(c)[0]
        'person.address.postal_code'
    """
    label = (candidate.visible_label or "").strip()
    parents = [str(p) for p in (candidate.parent_context or []) if str(p).strip()]
    live = [s for s in spans or () if s is not None and not s.is_blank()]

    base = ontology_lookup(label) if label else None
    scoped = context_lookup(label, parents) if (label and parents) else None
    nearby = neighbourhood_texts(candidate, live, config) if live else []

    # 2. an explicit section context beats the plain alias when it is more specific
    if scoped is not None and scoped != base:
        return (
            scoped,
            SCORE_CONTEXT,
            ["context_alias", "context:%s" % normalize_label(parents[-1])],
        )

    # 2b. failing that, the surrounding words may overrule an ambiguous label
    if label:
        flip = disambiguate(label, nearby)
        if flip is not None and flip[0] != base:
            return (
                flip[0],
                SCORE_NEIGHBOUR,
                ["neighbour_disambiguation", "hint:%s" % flip[1]],
            )

    # 1. exact alias
    if base is not None:
        return (base, SCORE_EXACT, ["exact_alias"])

    # 3. a printed placeholder in or beside the blank
    rule = _placeholder_rule(candidate, live, config)
    if rule is not None and rule.canonical_hint:
        return (
            rule.canonical_hint,
            SCORE_PLACEHOLDER,
            ["placeholder_pattern", "pattern:%s" % rule.name],
        )

    # 4. label glued to its parent context
    if label and parents:
        for parent in reversed(parents):
            for phrase in ("%s %s" % (parent, label), "%s %s" % (label, parent)):
                found = ontology_lookup(phrase)
                if found is not None:
                    return (
                        found,
                        SCORE_CONCAT,
                        ["label_context_concat", "context:%s" % normalize_label(parent)],
                    )

    # 5. label plus neighbourhood, through the pattern inferencer
    if label or nearby:
        inferred = infer_from_context(label, nearby)
        if inferred is not None and inferred.canonical_hint:
            return (
                inferred.canonical_hint,
                SCORE_INFERENCE,
                ["context_inference", "pattern:%s" % inferred.name],
            )

    # 6. fuzzy alias
    if label:
        matches = fuzzy_lookup(label, cutoff=FUZZY_CUTOFF)
        if matches:
            key, ratio = matches[0]
            return (
                key,
                round(ratio * SCORE_FUZZY_FACTOR, 6),
                ["fuzzy_alias", "ratio:%.2f" % ratio],
            )

    return (None, 0.0, ["unresolved"])


def _placeholder_rule(
    candidate: FieldCandidate, spans: Sequence[TextSpan], config: Any
) -> Optional[Any]:
    """The strongest placeholder rule speaking about this blank, or ``None``.

    Looks at the text printed inside or immediately beside the rectangle first, then
    falls back to a format hint the detection layer already recorded, so the cascade
    still works when the caller has no spans to give.
    """
    if spans:
        line_h = body_line_height(spans, candidate.page)
        inside, beside = nearby_spans(candidate, spans, ADJACENT_LINES * line_h)
        for pool in (inside, beside):
            best = None
            for span in pool:
                rule = match_placeholder(span.text)
                if rule is None or not rule.canonical_hint:
                    continue
                if best is None or rule.confidence > best.confidence:
                    best = rule
            if best is not None:
                return best
    hint = candidate.constraints.format_hint
    if hint:
        rule = match_placeholder(hint)
        if rule is not None and rule.canonical_hint:
            return rule
        from ..ontology import PATTERNS

        for known in sorted(PATTERNS, key=lambda r: (-r.confidence, r.name)):
            if known.format_hint == hint and known.canonical_hint:
                return known
    return None


# --------------------------------------------------------------------- page cascade
def sibling_key(
    candidate: FieldCandidate,
    neighbours: Sequence[FieldCandidate],
) -> Optional[Tuple[str, str]]:
    """Infer a key from resolved row/column siblings sharing one namespace.

    A row reading ``[Street] [City] [____] [ZIP]`` where three of four blanks resolved
    into ``person.address.*`` says the fourth is the missing ``person.address.*`` child
    whose label the blank's own label most resembles -- which is how "St." next to
    "City" and "ZIP" becomes ``person.address.region``.

    Returns:
        ``(canonical_key, namespace)`` or ``None``.
    """
    label = (candidate.visible_label or "").strip()
    if not label:
        return None
    resolved = [n.canonical_key for n in neighbours if n.canonical_key]
    if len(resolved) < 2:
        return None

    counts: Dict[str, int] = {}
    for key in resolved:
        parts = key.split(".")
        for depth in range(2, len(parts)):
            counts[_prefix(key, depth)] = counts.get(_prefix(key, depth), 0) + 1
    usable = [(count, prefix.count("."), prefix) for prefix, count in counts.items() if count >= 2]
    if not usable:
        return None
    usable.sort(key=lambda item: (-item[0], -item[1], item[2]))
    namespace = usable[0][2]

    taken = {key for key in resolved if key.startswith(namespace + ".")}
    norm = normalize_label(label)
    best: Optional[Tuple[float, str]] = None
    for spec in ontology_children(namespace):
        if spec.key in taken:
            continue
        forms = [normalize_label(spec.label), normalize_label(spec.leaf.replace("_", " "))]
        forms.extend(normalize_label(alias) for alias in spec.aliases)
        ratio = 0.0
        for form in forms:
            if not form:
                continue
            ratio = max(ratio, difflib.SequenceMatcher(None, norm, form).ratio())
        if ratio < SIBLING_LABEL_CUTOFF:
            continue
        if best is None or (ratio, spec.key) > (best[0], best[1]):
            best = (ratio, spec.key)
    if best is None:
        return None
    return (best[1], namespace)


def canonicalize_all(
    candidates: Sequence[FieldCandidate],
    graph: Optional[SpatialGraph] = None,
    config: Any = None,
    *,
    spans: Sequence[TextSpan] = (),
) -> List[FieldCandidate]:
    """Run the cascade over a whole page, then reconcile the results.

    Three passes:

    1. :func:`canonicalize` per candidate (steps 0-6);
    2. sibling inference for what is left, using ``SAME_ROW`` / ``SAME_COLUMN`` peers
       from the graph (step 7);
    3. conflict resolution -- two candidates on one page claiming the same
       non-repeatable key, where one is clearly weaker, demote the weaker one to
       ``None``.  Genuine repeats (the same label appearing again, which is the normal
       case on a multi-page form) are never demoted.

    Each resolved candidate gets its ``canonical_key``, a ``semantic_type`` confidence
    no lower than the cascade's score, and one ``PATTERN`` evidence carrying the reason
    codes.

    Returns:
        The same candidate objects, in the order they were given.
    """
    live = [c for c in candidates or () if c is not None]
    if not live:
        return list(candidates or ())
    page_spans = [s for s in spans or () if s is not None and not s.is_blank()]

    results: Dict[int, Tuple[Optional[str], float, List[str]]] = {}
    for index, candidate in enumerate(live):
        results[index] = canonicalize(candidate, config, spans=page_spans)

    by_id = {candidate.id: index for index, candidate in enumerate(live)}
    for index, candidate in enumerate(live):
        key, score, reasons = results[index]
        if key is not None:
            continue
        peers = _peers(candidate, live, by_id, graph, results)
        found = sibling_key(candidate, peers)
        if found is None:
            continue
        results[index] = (
            found[0],
            SCORE_SIBLING,
            ["sibling_namespace", "namespace:%s" % found[1]],
        )

    _resolve_conflicts(live, results)

    for index, candidate in enumerate(live):
        key, score, reasons = results[index]
        candidate.canonical_key = key
        candidate.evidence = [
            e
            for e in candidate.evidence
            if not (e.kind == EvidenceKind.PATTERN and e.source_agent == SOURCE_AGENT)
        ]
        if key is None:
            continue
        candidate.confidence.semantic_type = max(
            candidate.confidence.semantic_type, round(score, 6)
        )
        candidate.add_evidence(
            Evidence(
                kind=EvidenceKind.PATTERN,
                score=round(score, 6),
                detail="%s (%s)" % (key, ",".join(reasons)),
                source_agent=SOURCE_AGENT,
                rect=candidate.rect,
            )
        )
    return list(candidates or ())


def _peers(
    candidate: FieldCandidate,
    live: Sequence[FieldCandidate],
    by_id: Dict[str, int],
    graph: Optional[SpatialGraph],
    results: Dict[int, Tuple[Optional[str], float, List[str]]],
) -> List[FieldCandidate]:
    """Row and column peers of a candidate, carrying the keys resolved so far."""
    found: List[FieldCandidate] = []
    seen: Set[str] = set()
    if graph is not None and candidate.id and graph.has_node(candidate.id):
        for relation in (RelationKind.SAME_ROW, RelationKind.SAME_COLUMN):
            for edge in graph.neighbors(candidate.id, relation):
                index = by_id.get(edge.dst)
                if index is None or edge.dst in seen:
                    continue
                seen.add(edge.dst)
                found.append(_with_key(live[index], results[index][0]))
    if not found:
        rect = candidate.rect.normalized()
        for index, other in enumerate(live):
            if other is candidate or int(other.page) != int(candidate.page):
                continue
            if _same_band(rect, other.rect.normalized()):
                found.append(_with_key(other, results[index][0]))
    return found


def _with_key(candidate: FieldCandidate, key: Optional[str]) -> FieldCandidate:
    """A shallow view of a candidate carrying the key resolved this pass."""
    if key is None or candidate.canonical_key == key:
        return candidate
    clone = FieldCandidate(
        id=candidate.id,
        page=candidate.page,
        rect=candidate.rect,
        field_type=candidate.field_type,
        visible_label=candidate.visible_label,
        canonical_key=key,
    )
    return clone


def _same_band(a: Rect, b: Rect) -> bool:
    """True when two rects share a row or a left edge (the graph-free fallback)."""
    if a.vertical_overlap(b) > 0.5 * min(a.height, b.height) + EPS:
        return True
    return abs(a.x0 - b.x0) <= 8.0


def _repeatable(key: str) -> bool:
    """True when one page may legitimately carry this key several times."""
    return key.rsplit(".", 1)[-1] in REPEATABLE_LEAVES


def _resolve_conflicts(
    live: Sequence[FieldCandidate],
    results: Dict[int, Tuple[Optional[str], float, List[str]]],
) -> None:
    """Demote the clearly weaker of two candidates claiming one non-repeatable key."""
    from .repeats import find_repeated_fields

    groups = find_repeated_fields(live)
    repeat_of: Dict[str, int] = {}
    for number, group in enumerate(groups):
        for member in group:
            repeat_of[member.id] = number

    buckets: Dict[Tuple[int, str], List[int]] = {}
    for index, candidate in enumerate(live):
        key = results[index][0]
        if key is None or _repeatable(key):
            continue
        buckets.setdefault((int(candidate.page), key), []).append(index)

    for (page, key), members in sorted(buckets.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda i: (-results[i][1], live[i].id))
        best = members[0]
        for other in members[1:]:
            same_repeat = (
                live[best].id in repeat_of
                and repeat_of.get(live[best].id) == repeat_of.get(live[other].id)
            )
            if same_repeat:
                continue
            if results[best][1] - results[other][1] < CONFLICT_MARGIN:
                continue
            reasons = list(results[other][2]) + [
                "conflict_demoted",
                "conflicts_with:%s" % live[best].id,
            ]
            LOG.debug(
                "demoting %s on page %d: %s already claimed by %s",
                live[other].id,
                page,
                key,
                live[best].id,
            )
            results[other] = (None, 0.0, reasons)
