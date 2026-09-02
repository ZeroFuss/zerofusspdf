"""Question builders and the JSON-Schema subset validator.

A council question is *closed*.  The answer space is an enum built from the ontology --
the fuzzy alias hits for the visible label plus the sibling namespace of the candidate's
resolved neighbours -- with a single reserved escape hatch, ``"unknown"``.  A member can
therefore only ever return a key ZFP already knows how to normalize, validate and fill,
or admit it does not know.  There is no open string anywhere in the protocol.

No question ever asks where a rectangle is.  Geometry is settled by the deterministic
detectors and the fusion stage before the council convenes; the council is asked what a
field *means*, never where it is.

The schemas are plain JSON Schema, restricted to the subset :func:`validate_answer`
implements: ``type``, ``enum``, ``const``, ``required``, ``properties``,
``additionalProperties``, ``items``, ``minimum``, ``maximum``, ``minLength``,
``maxLength``, ``minItems`` and ``maxItems``.  The validator is 60 deterministic lines;
importing ``jsonschema`` would violate the zero-dependency rule for no benefit.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.config import PrivacyConfig
from ..core.types import FieldCandidate, FieldType
from ..ontology import CANONICAL_KEYS, children, context_lookup, fuzzy_lookup, lookup
from .base import UNKNOWN, Question
from .redaction import redact_context

__all__ = [
    "MAX_OPTIONS",
    "FUZZY_CUTOFF",
    "question_context",
    "derive_options",
    "canonical_key_schema",
    "field_type_schema",
    "choice_set_schema",
    "ambiguity_schema",
    "canonical_key_question",
    "field_type_question",
    "choice_set_question",
    "ambiguity_question",
    "validate_answer",
]

#: How many canonical keys a closed enum may offer.  A larger set is not a better
#: question: it dilutes every member's confidence and makes the verdict less legible.
MAX_OPTIONS = 12

#: Alias-similarity floor for options that reach the ballot.  Lower than
#: :func:`zfp.ontology.fuzzy_lookup`'s own default because the council only convenes
#: once the confident paths have already failed.
FUZZY_CUTOFF = 0.60

_CONFIDENCE_PROPERTY: Dict[str, Any] = {"type": "number", "minimum": 0, "maximum": 1}
_REASON_CODES_PROPERTY: Dict[str, Any] = {"type": "array", "items": {"type": "string"}}


# ----------------------------------------------------------------------------------
# Context
# ----------------------------------------------------------------------------------
def question_context(
    candidate: FieldCandidate,
    extra: Optional[Mapping[str, Any]] = None,
    *,
    privacy: Optional[PrivacyConfig] = None,
) -> Dict[str, Any]:
    """Assemble and redact the context describing one unresolved candidate.

    What goes in: the visible label, the section context, the document title supplied by
    the caller, the detected field type, the *shape* of the blank (width, height, comb
    cell count, character budget) and the evidence kinds that produced it.

    What never goes in: the page, its text, its image, or any value.  Whatever the
    caller passes in ``extra`` is filtered by :func:`~zfp.council.redaction.redact_context`
    exactly like everything else, so a careless caller cannot widen the egress surface.
    """
    rect = candidate.rect
    shape: Dict[str, Any] = {
        "width": round(rect.width, 1),
        "height": round(rect.height, 1),
    }
    constraints = candidate.constraints
    if constraints.comb_cells:
        shape["comb_cells"] = int(constraints.comb_cells)
    if constraints.max_chars_estimate:
        shape["max_chars"] = int(constraints.max_chars_estimate)
    if constraints.multiline:
        shape["multiline"] = True

    field_type = candidate.field_type
    ctx: Dict[str, Any] = {
        "label": candidate.visible_label or "",
        "section": [str(s) for s in candidate.parent_context],
        "field_type": getattr(field_type, "value", str(field_type)),
        "shape": shape,
        "page": int(candidate.page),
        "candidate": candidate.id,
        "sources": sorted(str(s) for s in candidate.sources),
    }
    if constraints.format_hint:
        ctx["format_hint"] = str(constraints.format_hint)
    if constraints.choices:
        ctx["declared_choices"] = [str(c) for c in constraints.choices]
    if candidate.canonical_key:
        ctx["provisional_key"] = str(candidate.canonical_key)
    if extra:
        for key, value in extra.items():
            ctx[str(key)] = value
    return redact_context(ctx, privacy)


def _neighbour_keys(context: Optional[Mapping[str, Any]]) -> List[str]:
    """Return the resolved canonical keys of neighbouring candidates, deduplicated."""
    if not context:
        return []
    out: List[str] = []
    for source in ("neighbour_keys", "sibling_keys", "resolved_neighbours"):
        raw = context.get(source)
        if not raw:
            continue
        items: Iterable[Any] = raw if isinstance(raw, (list, tuple)) else [raw]
        for item in items:
            key = str(item).strip()
            if key and key in CANONICAL_KEYS and key not in out:
                out.append(key)
    return out


def _parent_of(key: str) -> str:
    """Return the namespace prefix of a canonical key (``a.b.c`` -> ``a.b``)."""
    return key.rsplit(".", 1)[0] if "." in key else key


def derive_options(
    candidate: FieldCandidate,
    context: Optional[Mapping[str, Any]] = None,
    *,
    limit: int = MAX_OPTIONS,
) -> List[str]:
    """Build the closed option set for a candidate.

    Sources, strongest first:

    1. the candidate's own provisional key, if the deterministic stages produced one;
    2. the exact and context-scoped alias hits for the visible label;
    3. :func:`zfp.ontology.fuzzy_lookup` over the whole alias space, scored by ratio;
    4. the sibling namespace of any resolved neighbours -- the reason a blank between
       "City" and "State" gets ``person.address.postal_code`` on the ballot even when
       its own label is illegible.

    Sorted by ``(-score, key)`` and capped at ``limit``, so the ballot is identical
    across runs.  ``"unknown"`` is never an option here; the schema adds it.
    """
    scores: Dict[str, float] = {}

    def offer(key: str, score: float) -> None:
        if not key or key not in CANONICAL_KEYS:
            return
        if score > scores.get(key, -1.0):
            scores[key] = score

    label = (candidate.visible_label or "").strip()
    parents = list(candidate.parent_context)

    if candidate.canonical_key:
        offer(str(candidate.canonical_key), 1.0)
    if label:
        scoped = context_lookup(label, parents)
        if scoped:
            offer(scoped, 0.99)
        plain = lookup(label)
        if plain:
            offer(plain, 0.98)
        for key, ratio in fuzzy_lookup(label, cutoff=FUZZY_CUTOFF):
            offer(key, min(0.97, float(ratio)))

    for neighbour in _neighbour_keys(context):
        parent = _parent_of(neighbour)
        for spec in children(parent):
            offer(spec.key, 0.55 if spec.key != neighbour else 0.40)

    if candidate.canonical_key:
        for spec in children(_parent_of(str(candidate.canonical_key))):
            offer(spec.key, 0.50)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [key for key, _ in ranked[: max(1, int(limit))]]


# ----------------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------------
def _closed_enum(options: Sequence[str]) -> List[str]:
    """Return the option list with ``"unknown"`` appended exactly once."""
    seen: List[str] = []
    for option in options:
        text = str(option).strip()
        if text and text != UNKNOWN and text not in seen:
            seen.append(text)
    seen.append(UNKNOWN)
    return seen


def canonical_key_schema(options: Sequence[str]) -> Dict[str, Any]:
    """Schema for "which canonical key does this field ask for?"."""
    return {
        "type": "object",
        "required": ["canonical_key", "confidence"],
        "additionalProperties": False,
        "properties": {
            "canonical_key": {"enum": _closed_enum(options)},
            "confidence": dict(_CONFIDENCE_PROPERTY),
            "reason_codes": dict(_REASON_CODES_PROPERTY),
        },
    }


def field_type_schema() -> Dict[str, Any]:
    """Schema for "what kind of control is this?" over the :class:`FieldType` values."""
    return {
        "type": "object",
        "required": ["field_type", "confidence"],
        "additionalProperties": False,
        "properties": {
            "field_type": {"enum": _closed_enum([t.value for t in FieldType])},
            "confidence": dict(_CONFIDENCE_PROPERTY),
            "reason_codes": dict(_REASON_CODES_PROPERTY),
        },
    }


def choice_set_schema(nearby_options: Sequence[str]) -> Dict[str, Any]:
    """Schema for "which of these nearby captions are this field's choices?"."""
    return {
        "type": "object",
        "required": ["choices", "confidence"],
        "additionalProperties": False,
        "properties": {
            "choices": {
                "type": "array",
                "items": {"enum": _closed_enum(nearby_options)},
                "maxItems": max(1, len(_closed_enum(nearby_options))),
            },
            "confidence": dict(_CONFIDENCE_PROPERTY),
            "reason_codes": dict(_REASON_CODES_PROPERTY),
        },
    }


def ambiguity_schema(competing: Sequence[str]) -> Dict[str, Any]:
    """Schema for "which of these competing keys wins?"."""
    return {
        "type": "object",
        "required": ["winner", "confidence"],
        "additionalProperties": False,
        "properties": {
            "winner": {"enum": _closed_enum(competing)},
            "confidence": dict(_CONFIDENCE_PROPERTY),
            "reason_codes": dict(_REASON_CODES_PROPERTY),
        },
    }


# ----------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------
def _describe(ctx: Mapping[str, Any]) -> str:
    """Render the redacted context as one deterministic clause list."""
    parts: List[str] = []
    label = ctx.get("label")
    parts.append('label "%s"' % label if label else "no visible label")
    section = ctx.get("section")
    if section:
        joined = " / ".join(str(s) for s in section) if isinstance(section, list) else str(section)
        if joined:
            parts.append('section "%s"' % joined)
    title = ctx.get("document_title")
    if title:
        parts.append('document "%s"' % title)
    shape = ctx.get("shape")
    if isinstance(shape, Mapping):
        width = shape.get("width")
        height = shape.get("height")
        if width is not None and height is not None:
            parts.append("blank %sx%s pt" % (width, height))
        if shape.get("comb_cells"):
            parts.append("%s comb cells" % shape["comb_cells"])
    placeholder = ctx.get("placeholder")
    if placeholder:
        parts.append('placeholder "%s"' % placeholder)
    hint = ctx.get("format_hint")
    if hint:
        parts.append('format hint "%s"' % hint)
    ftype = ctx.get("field_type")
    if ftype:
        parts.append("detected control type %s" % ftype)
    return "; ".join(parts)


def canonical_key_question(
    candidate: FieldCandidate,
    options: Sequence[str] = (),
    context: Optional[Mapping[str, Any]] = None,
    *,
    privacy: Optional[PrivacyConfig] = None,
) -> Question:
    """Ask which canonical key an unresolved candidate is asking for.

    Args:
        candidate: The candidate the deterministic stages could not settle.
        options: The closed key set to vote over.  When empty it is derived with
            :func:`derive_options`.
        context: Extra structural context (document title, placeholder text, resolved
            neighbour keys).  Redacted before it reaches the question.
        privacy: Egress policy applied to the context.

    Returns:
        A :class:`Question` whose ``canonical_key`` enum is closed over ``options``
        plus ``"unknown"``.
    """
    ctx = question_context(candidate, context, privacy=privacy)
    ballot = list(options) or derive_options(candidate, ctx)
    ballot = [key for key in ballot if key in CANONICAL_KEYS]
    ctx["options"] = list(ballot)
    prompt = (
        "Which canonical key does this form field ask for? %s. "
        'Answer with exactly one key from the enum, or "unknown".' % _describe(ctx)
    )
    return Question.build(
        "canonical_key", prompt, canonical_key_schema(ballot), ctx, ballot
    )


def field_type_question(
    candidate: FieldCandidate,
    context: Optional[Mapping[str, Any]] = None,
    *,
    privacy: Optional[PrivacyConfig] = None,
) -> Question:
    """Ask what kind of control a candidate should become.

    The geometry is already settled; this asks only for the semantic control type
    (a date blank versus a plain text blank, a checkbox versus a radio member).
    """
    ctx = question_context(candidate, context, privacy=privacy)
    options = [t.value for t in FieldType if t is not FieldType.UNKNOWN]
    ctx["options"] = list(options)
    prompt = (
        "What kind of form control is this blank? %s. "
        'Answer with exactly one type from the enum, or "unknown".' % _describe(ctx)
    )
    return Question.build("field_type", prompt, field_type_schema(), ctx, options)


def choice_set_question(
    candidate: FieldCandidate,
    nearby_options: Sequence[str] = (),
    context: Optional[Mapping[str, Any]] = None,
    *,
    privacy: Optional[PrivacyConfig] = None,
) -> Question:
    """Ask which nearby captions form this field's choice list.

    ``nearby_options`` are the caption texts the perception stage found beside the
    control.  The answer is a subset of them -- never a new string.
    """
    ctx = question_context(candidate, context, privacy=privacy)
    captions: List[str] = []
    for option in nearby_options:
        text = str(option).strip()
        if text and text not in captions:
            captions.append(text)
    if not captions:
        captions = [str(c) for c in candidate.constraints.choices]
    ctx["nearby_options"] = list(captions)
    prompt = (
        "Which of the nearby captions are the selectable choices of this control? %s. "
        "Answer with a subset of the enum; use an empty array when none of them are."
        % _describe(ctx)
    )
    return Question.build(
        "choice_set", prompt, choice_set_schema(captions), ctx, captions
    )


def ambiguity_question(
    candidate: FieldCandidate,
    competing: Sequence[str] = (),
    context: Optional[Mapping[str, Any]] = None,
    *,
    privacy: Optional[PrivacyConfig] = None,
) -> Question:
    """Ask which of several competing canonical keys wins.

    Used when two deterministic stages resolved the same blank to different keys --
    ``billing.address.city`` versus ``shipping.address.city``, say -- and the tie has to
    be broken by meaning rather than by geometry.
    """
    ctx = question_context(candidate, context, privacy=privacy)
    keys: List[str] = []
    for key in competing:
        text = str(key).strip()
        if text and text not in keys:
            keys.append(text)
    ctx["competing"] = list(keys)
    labels = {k: CANONICAL_KEYS[k].label for k in keys if k in CANONICAL_KEYS}
    if labels:
        ctx["competing_labels"] = [
            "%s = %s" % (k, labels[k]) for k in sorted(labels)
        ]
    prompt = (
        "Two deterministic stages disagree about this field. Which competing canonical "
        'key is correct? %s. Answer with exactly one key from the enum, or "unknown".'
        % _describe(ctx)
    )
    return Question.build("ambiguity", prompt, ambiguity_schema(keys), ctx, keys)


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------
_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, Mapping),
    "array": lambda v: isinstance(v, (list, tuple)),
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _same_scalar(a: Any, b: Any) -> bool:
    """Equality that refuses Python's ``True == 1`` conflation."""
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def _check_type(value: Any, declared: Any) -> bool:
    """Check a ``type`` keyword, which may be a name or a list of names."""
    names = declared if isinstance(declared, (list, tuple)) else [declared]
    for name in names:
        check = _TYPE_CHECKS.get(str(name))
        if check is not None and check(value):
            return True
    return False


def _validate(value: Any, schema: Any) -> bool:
    """Validate ``value`` against one schema node of the supported subset."""
    if not isinstance(schema, Mapping):
        return False
    if not schema:
        return True

    if "type" in schema and not _check_type(value, schema["type"]):
        return False
    if "const" in schema and not _same_scalar(value, schema["const"]):
        return False
    if "enum" in schema:
        allowed = schema["enum"]
        if not isinstance(allowed, (list, tuple)):
            return False
        if not any(_same_scalar(value, option) for option in allowed):
            return False

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False

    if isinstance(value, Mapping):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        for name in schema.get("required", ()) or ():
            if str(name) not in value:
                return False
        additional = schema.get("additionalProperties", True)
        for name, item in value.items():
            sub = properties.get(str(name))
            if sub is None:
                if additional is False:
                    return False
                if isinstance(additional, Mapping) and not _validate(item, additional):
                    return False
                continue
            if not _validate(item, sub):
                return False

    if isinstance(value, (list, tuple)):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return False
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False
        items = schema.get("items")
        if isinstance(items, Mapping):
            for item in value:
                if not _validate(item, items):
                    return False

    return True


def validate_answer(answer: Any, schema: Mapping[str, Any]) -> bool:
    """Return True when ``answer`` satisfies ``schema``.

    A deterministic validator for the JSON-Schema subset the council uses: ``type``,
    ``enum``, ``const``, ``required``, ``properties``, ``additionalProperties``,
    ``items``, ``minimum``/``maximum``, ``minLength``/``maxLength`` and
    ``minItems``/``maxItems``.  Unknown keywords are ignored rather than rejected, so a
    schema carrying documentation fields (``title``, ``description``) still validates.

    Booleans are never accepted where a number is required: JSON Schema separates the
    two even though Python does not.
    """
    if answer is None:
        return False
    return _validate(answer, schema)
