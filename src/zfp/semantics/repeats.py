"""Repeated fields: the same question asked twice is free supervision.

Long forms ask for the same thing over and over -- "Applicant Name" heads every one of
five pages, "Date" sits under every signature, an account number is repeated on the
remittance stub.  That repetition is not noise, it is evidence:

* a field whose label appears three times is far less likely to be a detection artefact;
* a value resolved once can be propagated to every sibling, so one vault hit fills five
  blanks;
* if two members of a group end up with *different* values, something upstream is wrong
  and the run should say so rather than writing a contradictory document.

Grouping is by canonical key when the normalizer resolved one, and by normalized visible
label otherwise, so the mechanism works before *and* after canonicalization.  Only groups
of two or more count -- a single field is not a repeat.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.logging import get_logger
from ..core.types import Evidence, EvidenceKind, FieldCandidate, FilledValue
from ..ontology import normalize_label

__all__ = [
    "find_repeated_fields",
    "propagate",
    "check_consistency",
    "repeat_evidence",
    "group_key",
    "as_dict",
    "REPEAT_BASE_SCORE",
    "REPEAT_STEP",
    "REPEAT_MAX_SCORE",
    "SOURCE_AGENT",
]

LOG = get_logger(__name__)

#: Score of a two-member repeat group.
REPEAT_BASE_SCORE = 0.60
#: Added per extra member.
REPEAT_STEP = 0.10
#: Ceiling, so a very repetitive form cannot claim certainty from repetition alone.
REPEAT_MAX_SCORE = 0.95
#: Written into every :class:`~zfp.core.types.Evidence` this module produces.
SOURCE_AGENT = "semantics.repeats"
#: Reason code added to every value this module copies onto a sibling.
REASON_PROPAGATED = "repeat_propagated"


def group_key(candidate: FieldCandidate) -> Optional[Tuple[str, str]]:
    """Return the grouping key of one candidate, or ``None`` when it has none.

    ``("key", "person.name.full")`` once the normalizer has spoken, otherwise
    ``("label", "applicant name")``.  The two spaces never mix: a canonicalized field is
    never grouped with an uncanonicalized one on label evidence alone.
    """
    if candidate is None:
        return None
    if candidate.canonical_key:
        return ("key", str(candidate.canonical_key))
    label = normalize_label(candidate.visible_label or "")
    if not label:
        return None
    return ("label", label)


def _candidate_sort_key(candidate: FieldCandidate) -> Tuple[int, float, float, str]:
    """Reading order across pages: page, then top to bottom, then left to right."""
    rect = candidate.rect.normalized()
    return (int(candidate.page), -rect.y1, rect.x0, str(candidate.id))


def find_repeated_fields(
    candidates: Sequence[FieldCandidate],
) -> List[List[FieldCandidate]]:
    """Group candidates that ask for the same thing.

    Args:
        candidates: The page's or the document's candidates.

    Returns:
        One list per repeat group, each holding **two or more** candidates in reading
        order.  Groups themselves are ordered by kind (``key`` before ``label``) and
        then by the key text, so the output is stable.

    Examples:
        >>> from zfp.core.geometry import Rect
        >>> from zfp.core.types import FieldCandidate
        >>> a = FieldCandidate("a", 0, Rect(0, 0, 10, 10), visible_label="Applicant Name")
        >>> b = FieldCandidate("b", 1, Rect(0, 0, 10, 10), visible_label="Applicant name:")
        >>> len(find_repeated_fields([a, b])[0])
        2
    """
    buckets: Dict[Tuple[str, str], List[FieldCandidate]] = {}
    for candidate in candidates or ():
        key = group_key(candidate)
        if key is None:
            continue
        buckets.setdefault(key, []).append(candidate)

    out: List[List[FieldCandidate]] = []
    for key in sorted(buckets):
        members = buckets[key]
        if len(members) < 2:
            continue
        out.append(sorted(members, key=_candidate_sort_key))
    return out


def repeat_evidence(candidates: Sequence[FieldCandidate]) -> List[FieldCandidate]:
    """Add a ``REPEAT`` evidence to every member of every repeat group.

    The score grows with the size of the group -- three occurrences of one label is
    stronger structural evidence than two -- and is capped at
    :data:`REPEAT_MAX_SCORE`.  Re-running this replaces the evidence it wrote before, so
    a pipeline may call it after every stage.

    Returns:
        The same candidate objects, in the order they were given.
    """
    groups = find_repeated_fields(candidates)
    for candidate in candidates or ():
        if candidate is None:
            continue
        candidate.evidence = [
            e
            for e in candidate.evidence
            if not (e.kind == EvidenceKind.REPEAT and e.source_agent == SOURCE_AGENT)
        ]
    for group in groups:
        score = min(
            REPEAT_MAX_SCORE, REPEAT_BASE_SCORE + REPEAT_STEP * (len(group) - 2)
        )
        first = group[0]
        name = first.canonical_key or first.visible_label or "field"
        pages = sorted({int(member.page) for member in group})
        detail = "%d occurrences of %r on page(s) %s" % (
            len(group),
            name,
            ",".join(str(p) for p in pages),
        )
        for member in group:
            member.add_evidence(
                Evidence(
                    kind=EvidenceKind.REPEAT,
                    score=round(score, 6),
                    detail=detail,
                    source_agent=SOURCE_AGENT,
                    rect=member.rect,
                )
            )
    return list(candidates or ())


# ---------------------------------------------------------------------- propagation
def _value_index(values: Sequence[FilledValue]) -> Dict[str, FilledValue]:
    """Index resolved values by field name, first occurrence winning."""
    out: Dict[str, FilledValue] = {}
    for value in values or ():
        if value is None:
            continue
        out.setdefault(str(value.field_name), value)
    return out


def _values_for(
    group: Sequence[FieldCandidate], index: Dict[str, FilledValue]
) -> List[Tuple[FieldCandidate, FilledValue]]:
    """Pair each member with the value recorded for it, when there is one."""
    pairs: List[Tuple[FieldCandidate, FilledValue]] = []
    for member in group:
        for name in (member.id, member.canonical_key, member.visible_label):
            if not name:
                continue
            found = index.get(str(name))
            if found is not None:
                pairs.append((member, found))
                break
    return pairs


def _best(pairs: Sequence[Tuple[FieldCandidate, FilledValue]]) -> Optional[FilledValue]:
    """The value a group should agree on: filled first, then highest confidence."""
    usable = [(m, v) for m, v in pairs if v.value is not None and str(v.value) != ""]
    if not usable:
        return None
    ranked = sorted(
        usable,
        key=lambda item: (
            0 if item[1].status == "filled" else 1,
            -float(item[1].confidence or 0.0),
            str(item[1].field_name),
        ),
    )
    return ranked[0][1]


def propagate(
    groups: Sequence[Sequence[FieldCandidate]],
    values: Sequence[FilledValue],
) -> List[FilledValue]:
    """Give every member of a repeat group the group's agreed value.

    One vault hit fills every occurrence: the strongest value in a group (a ``filled``
    one first, then the highest confidence) is copied onto every member that has no
    value or a weaker one, and every member of the group ends up carrying the group's
    maximum confidence.

    Args:
        groups: Repeat groups, normally from :func:`find_repeated_fields`.
        values: Values resolved so far.  Matched to a member by ``field_name`` against
            the member's id, canonical key or visible label.

    Returns:
        The expanded list: every input value (updated in place where the group
        disagreed) plus one new :class:`~zfp.core.types.FilledValue` per member that had
        none, ordered by ``field_name``.  Values belonging to no group are passed
        through untouched.
    """
    index = _value_index(values)
    out: Dict[str, FilledValue] = dict(index)

    for group in groups or ():
        members = [m for m in group if m is not None]
        if len(members) < 2:
            continue
        pairs = _values_for(members, index)
        winner = _best(pairs)
        if winner is None:
            continue
        confidence = max(
            [float(v.confidence or 0.0) for _m, v in pairs] + [float(winner.confidence or 0.0)]
        )
        provenance = dict(winner.provenance or {})
        for member in members:
            existing: Optional[FilledValue] = None
            for name in (member.id, member.canonical_key, member.visible_label):
                if name and str(name) in out:
                    existing = out[str(name)]
                    break
            if existing is None:
                copied = dict(provenance)
                copied["propagated_from"] = winner.field_name
                out[str(member.id)] = FilledValue(
                    field_name=str(member.id),
                    canonical_key=member.canonical_key or winner.canonical_key,
                    value=winner.value,
                    confidence=round(confidence, 6),
                    provenance=copied,
                    status=winner.status,
                    reason_codes=list(winner.reason_codes) + [REASON_PROPAGATED],
                )
                continue
            changed = existing.value != winner.value
            existing.value = winner.value
            existing.confidence = round(max(float(existing.confidence or 0.0), confidence), 6)
            if existing.canonical_key is None:
                existing.canonical_key = member.canonical_key or winner.canonical_key
            if changed:
                existing.provenance = dict(existing.provenance or {})
                existing.provenance["propagated_from"] = winner.field_name
                existing.status = winner.status
            if REASON_PROPAGATED not in existing.reason_codes:
                existing.reason_codes.append(REASON_PROPAGATED)

    return [out[name] for name in sorted(out)]


def check_consistency(
    groups: Sequence[Sequence[FieldCandidate]],
    values: Sequence[FilledValue],
) -> List[str]:
    """Report every repeat group whose members carry different values.

    A form that says one thing on page 1 and another on page 4 is a defect, and it is
    always worth surfacing before the document is written.

    Args:
        groups: Repeat groups, normally from :func:`find_repeated_fields`.
        values: The values to check.

    Returns:
        One human-readable line per disagreeing group, sorted; empty when every group
        agrees (a group where only one member has a value agrees trivially).
    """
    index = _value_index(values)
    problems: List[str] = []
    for group in groups or ():
        members = [m for m in group if m is not None]
        if len(members) < 2:
            continue
        pairs = _values_for(members, index)
        distinct: List[str] = []
        for _member, value in pairs:
            if value.value is None:
                continue
            text = str(value.value)
            if text not in distinct:
                distinct.append(text)
        if len(distinct) < 2:
            continue
        first = members[0]
        name = first.canonical_key or first.visible_label or first.id
        problems.append(
            "%s: %d occurrences disagree (%s)"
            % (name, len(members), ", ".join(repr(v) for v in sorted(distinct)))
        )
    return sorted(problems)


def as_dict(groups: Sequence[Sequence[FieldCandidate]]) -> List[Dict[str, Any]]:
    """Return a JSON-ready summary of repeat groups, for reports and QA."""
    out: List[Dict[str, Any]] = []
    for group in groups or ():
        if not group:
            continue
        first = group[0]
        out.append(
            {
                "key": first.canonical_key,
                "label": first.visible_label,
                "count": len(group),
                "members": [member.id for member in group],
                "pages": sorted({int(member.page) for member in group}),
            }
        )
    return out
