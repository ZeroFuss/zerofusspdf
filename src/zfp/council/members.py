"""The council members.

Four members are **always available** and are pure Python -- no model, no network, no
optional dependency:

``RulesMember``
    The ontology speaking for itself: exact and context-scoped alias hits plus the
    deterministic placeholder patterns.  A printed ``##-#######`` beside "Tax ID" is not
    an opinion, it is a rule, and this member votes it at near-certainty.
``HeuristicMember``
    Layout and section reasoning.  Prefers an option whose namespace matches the section
    it sits under and the namespace its resolved neighbours already occupy, and
    penalizes an option whose declared field type contradicts the geometry the detectors
    already settled.
``OntologyFuzzyMember``
    ``difflib`` over the alias space of the options on the ballot; the confidence *is*
    the similarity ratio.  This is the member that survives OCR damage.
``SiblingConsensusMember``
    Namespace completion.  Neighbours resolved to ``person.address.city`` and
    ``person.address.region`` make ``person.address.postal_code`` the obvious reading of
    the blank between them, whatever its label says.

``OpenRouterMember`` is the fifth, optional member.  It is off unless an API key is
present **and** ``PrivacyConfig.allow_external_inference`` is true; calling it with
egress disabled raises :class:`~zfp.core.errors.PolicyError` rather than quietly doing
nothing.  Its reply is validated against the question's schema, and any failure comes
back as a zero-confidence vote carrying an error -- never as an exception.

Every member returns a :class:`~zfp.council.base.Vote` even when it fails internally.
A broken member costs the council one opinion; it never costs it the verdict.
"""

from __future__ import annotations

import difflib
import json
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..core.config import PrivacyConfig
from ..core.errors import PolicyError
from ..core.optional import optional_import
from ..core.types import FieldType
from ..ontology import (
    CANONICAL_KEYS,
    aliases_for,
    context_lookup,
    fuzzy_lookup,
    infer_from_context,
    lookup,
    match_placeholder,
    match_value,
    normalize_label,
)
from . import openrouter as _openrouter
from .base import UNKNOWN, BaseMember, Question, Vote
from .questions import validate_answer
from .redaction import assert_no_secrets

__all__ = [
    "SECTION_NAMESPACES",
    "LOCAL_CLASSIFIER_MODULE",
    "RulesMember",
    "HeuristicMember",
    "OntologyFuzzyMember",
    "SiblingConsensusMember",
    "LocalModelMember",
    "OpenRouterMember",
    "local_members",
]

#: Section vocabulary -> ontology namespace.  Only namespaces that actually exist in
#: :data:`zfp.ontology.CANONICAL_KEYS` appear on the right-hand side.
SECTION_NAMESPACES: Dict[str, str] = {
    "employer": "company",
    "employers": "company",
    "company": "company",
    "business": "company",
    "corporate": "company",
    "organization": "company",
    "organisation": "company",
    "firm": "company",
    "vendor": "company",
    "supplier": "company",
    "billing": "billing",
    "bill": "billing",
    "invoice": "billing",
    "shipping": "shipping",
    "ship": "shipping",
    "delivery": "shipping",
    "mailing": "mailing",
    "applicant": "person",
    "personal": "person",
    "individual": "person",
    "patient": "person",
    "borrower": "person",
    "tenant": "person",
    "guardian": "person",
    "parent": "person",
    "spouse": "person",
    "bank": "bank",
    "banking": "bank",
    "deposit": "bank",
    "card": "card",
    "payment": "card",
    "credit": "card",
    "vehicle": "vehicle",
    "auto": "vehicle",
    "automobile": "vehicle",
    "insurance": "insurance",
    "policy": "insurance",
    "coverage": "insurance",
    "medical": "medical",
    "health": "medical",
    "physician": "medical",
    "clinical": "medical",
    "employment": "employment",
    "job": "employment",
    "work": "employment",
    "occupation": "employment",
    "education": "education",
    "school": "education",
    "academic": "education",
    "military": "military",
    "service": "military",
    "property": "property",
    "premises": "property",
    "lease": "lease",
    "rental": "lease",
    "tenancy": "lease",
    "tax": "tax",
    "taxes": "tax",
    "irs": "tax",
    "legal": "legal",
    "court": "legal",
    "travel": "travel",
    "passport": "travel",
    "trip": "travel",
    "consent": "consent",
    "authorization": "consent",
    "signature": "consent",
    "credentials": "credentials",
    "login": "credentials",
}


# ----------------------------------------------------------------------------------
# Shared context helpers
# ----------------------------------------------------------------------------------
def _text(ctx: Mapping[str, Any], key: str) -> str:
    """Return a context entry as a stripped string."""
    value = ctx.get(key)
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value).strip()
    return str(value).strip()


def _list(ctx: Mapping[str, Any], key: str) -> List[str]:
    """Return a context entry as a list of stripped strings."""
    value = ctx.get(key)
    if not value:
        return []
    items: Iterable[Any] = value if isinstance(value, (list, tuple)) else [value]
    return [str(v).strip() for v in items if str(v).strip()]


def _fragments(ctx: Mapping[str, Any]) -> List[str]:
    """Every text fragment near the blank, longest first and deduplicated."""
    out: List[str] = []
    for key in ("placeholder", "format_hint", "nearby", "nearby_text", "inline_text"):
        for item in _list(ctx, key):
            if item not in out:
                out.append(item)
    out.sort(key=lambda s: (-len(s), s))
    return out


def _neighbours(ctx: Mapping[str, Any]) -> List[str]:
    """Resolved canonical keys of the candidate's neighbours."""
    out: List[str] = []
    for key in ("neighbour_keys", "sibling_keys", "resolved_neighbours"):
        for item in _list(ctx, key):
            if item in CANONICAL_KEYS and item not in out:
                out.append(item)
    return out


def _parent_of(key: str) -> str:
    """Namespace prefix of a canonical key (``a.b.c`` -> ``a.b``)."""
    return key.rsplit(".", 1)[0] if "." in key else key


def _detected_type(ctx: Mapping[str, Any]) -> Optional[FieldType]:
    """The control type the deterministic detectors already settled, if any."""
    raw = _text(ctx, "field_type")
    if not raw:
        return None
    try:
        detected = FieldType(raw)
    except ValueError:
        return None
    return None if detected is FieldType.UNKNOWN else detected


def _section_namespaces(ctx: Mapping[str, Any]) -> List[str]:
    """Ontology namespaces implied by the section / document context."""
    words: List[str] = []
    for key in ("section", "document_title", "parent_context"):
        words.extend(normalize_label(_text(ctx, key)).split())
    out: List[str] = []
    for word in words:
        namespace = SECTION_NAMESPACES.get(word)
        if namespace and namespace not in out:
            out.append(namespace)
    return out


def _round2_leader(q: Question) -> str:
    """The answer that led the first round, when this is a second-round question."""
    if int(q.context.get("round", 1) or 1) < 2:
        return ""
    return _text(q.context, "leading_answer")


def _options(q: Question) -> List[str]:
    """The closed ballot, without ``"unknown"``."""
    if q.options:
        return [o for o in q.options if o != UNKNOWN]
    return [o for o in q.enum() if o != UNKNOWN]


def _key_vote(
    member: BaseMember, q: Question, key: str, confidence: float, codes: Sequence[str]
) -> Vote:
    """Vote ``key`` when the ballot allows it, otherwise report the ballot miss."""
    if not key:
        return member._unknown(q, "no_signal")
    if not q.allows(key):
        return member._unknown(q, "resolved_key_not_on_ballot")
    return member._answer(q, key, confidence, codes)


def _type_vote(
    member: BaseMember, q: Question, ftype: Any, confidence: float, codes: Sequence[str]
) -> Vote:
    """Vote a field type, normalizing enum/str and honouring the ballot."""
    value = getattr(ftype, "value", ftype)
    if not value:
        return member._unknown(q, "no_signal")
    return _key_vote(member, q, str(value), confidence, codes)


# ----------------------------------------------------------------------------------
# RulesMember
# ----------------------------------------------------------------------------------
class RulesMember(BaseMember):
    """The ontology's own answer: alias hits and deterministic placeholder patterns.

    The placeholder patterns are the strong signal.  A redacted context still carries
    the blank's *shape* -- ``"12-3456789"`` reaches the council as ``"##-#######"`` --
    and that shape matches exactly one rule, so this member can vote an EIN at 0.97
    without any value ever leaving the machine.
    """

    name = "rules"

    def _resolve(self, q: Question) -> Tuple[Optional[str], float, List[str], Optional[Any]]:
        """Return ``(key, confidence, reason_codes, pattern_rule)`` for a question."""
        ctx = q.context
        label = _text(ctx, "label")
        section = _list(ctx, "section")
        fragments = _fragments(ctx)
        codes: List[str] = []

        by_label: Optional[str] = None
        if label:
            by_label = context_lookup(label, section) or lookup(label)
            if by_label:
                codes.append("alias_exact")

        # A rule that came from ink printed on the page is a much stronger signal than
        # one merely declared for the key the label resolved to, so the two are tracked
        # separately and reported with different reason codes.
        printed = None
        for fragment in fragments:
            hit = match_placeholder(fragment)
            if hit is not None:
                printed = hit
                break
        if printed is None and fragments:
            inferred = infer_from_context(label, fragments)
            if inferred is not None and any(
                match_placeholder(f) is not None or match_value(f) is not None
                for f in fragments
            ):
                printed = inferred
        declared = infer_from_context(label, ()) if (printed is None and label) else None

        if printed is not None and printed.canonical_hint:
            codes.append("placeholder_pattern:%s" % printed.name)
            if by_label and by_label == printed.canonical_hint:
                codes.append("pattern_label_agree")
                return printed.canonical_hint, max(printed.confidence, 0.95), codes, printed
            if by_label is None:
                return printed.canonical_hint, printed.confidence, codes, printed
            codes.append("pattern_label_conflict")
            if printed.confidence >= 0.90:
                return printed.canonical_hint, printed.confidence * 0.85, codes, printed
            return by_label, 0.70, codes, printed

        if by_label:
            provisional = _text(ctx, "provisional_key")
            confidence = 0.92 if provisional and provisional == by_label else 0.88
            if declared is not None and declared.canonical_hint == by_label:
                codes.append("key_declared_format:%s" % declared.name)
                confidence = min(0.94, confidence + 0.02)
            return by_label, confidence, codes, declared or printed
        return None, 0.0, codes, declared or printed

    def _vote(self, q: Question) -> Vote:
        key, confidence, codes, rule = self._resolve(q)

        if q.kind in ("canonical_key", "ambiguity"):
            if key is None:
                return self._unknown(q, "no_alias_or_pattern")
            return _key_vote(self, q, key, confidence, codes)

        if q.kind == "field_type":
            if rule is not None:
                return _type_vote(
                    self, q, rule.field_type, rule.confidence, codes + ["pattern_type"]
                )
            if key is not None and key in CANONICAL_KEYS:
                return _type_vote(
                    self,
                    q,
                    CANONICAL_KEYS[key].field_type,
                    min(confidence, 0.90),
                    codes + ["key_declares_type"],
                )
            return self._unknown(q, "no_alias_or_pattern")

        if q.kind == "choice_set":
            captions = [c for c in _options(q) if len(c) <= 48]
            declared = [c for c in _list(q.context, "declared_choices") if c in captions]
            if declared:
                return self._answer(q, declared, 0.90, ["declared_choices"])
            detected = _detected_type(q.context)
            if captions and detected in (
                FieldType.CHOICE,
                FieldType.RADIO,
                FieldType.CHECKBOX,
                FieldType.LISTBOX,
            ):
                return self._answer(q, captions, 0.72, ["nearby_captions", "control_is_selectable"])
            return self._unknown(q, "no_choice_evidence")

        return self._unknown(q, "unsupported_question_kind")


# ----------------------------------------------------------------------------------
# HeuristicMember
# ----------------------------------------------------------------------------------
class HeuristicMember(BaseMember):
    """Layout and section reasoning over the closed ballot.

    Scores every option on the ballot: a namespace matching the section it sits under,
    a namespace matching the one its neighbours already occupy, a parent-context
    qualifier that the section corroborates -- and a hard penalty when the option's
    declared field type contradicts the control type the geometry already settled, since
    ``person.signature`` is not a plausible reading of a 12-point text blank.
    """

    name = "heuristic"

    #: Below this a heuristic score is noise and the member abstains.
    floor = 0.50

    def _score(self, q: Question) -> List[Tuple[float, str, List[str]]]:
        """Score the ballot, best first."""
        ctx = q.context
        section_ns = _section_namespaces(ctx)
        section_tokens = set(normalize_label(_text(ctx, "section")).split())
        neighbours = _neighbours(ctx)
        neighbour_ns: List[str] = [n.split(".", 1)[0] for n in neighbours]
        neighbour_parents = [_parent_of(n) for n in neighbours]
        detected = _detected_type(ctx)
        provisional = _text(ctx, "provisional_key")
        leader = _round2_leader(q)

        scored: List[Tuple[float, str, List[str]]] = []
        for option in _options(q):
            spec = CANONICAL_KEYS.get(option)
            if spec is None:
                continue
            score = 0.45
            codes: List[str] = []
            namespace = spec.namespace
            if namespace in section_ns:
                score += 0.25
                codes.append("section_namespace")
            if namespace in neighbour_ns:
                score += 0.15
                codes.append("neighbour_namespace")
            if _parent_of(option) in neighbour_parents:
                score += 0.10
                codes.append("neighbour_parent")
            if spec.parents and section_tokens.intersection(
                {p.lower() for p in spec.parents}
            ):
                score += 0.10
                codes.append("parent_context_match")
            if spec.parents and not section_tokens.intersection(
                {p.lower() for p in spec.parents}
            ):
                score -= 0.10
                codes.append("parent_context_unsupported")
            if detected is not None:
                if spec.field_type is detected:
                    score += 0.10
                    codes.append("field_type_agrees")
                elif detected is not FieldType.TEXT:
                    score -= 0.30
                    codes.append("field_type_disagrees")
            if provisional and option == provisional:
                score += 0.05
                codes.append("provisional_key")
            if leader and option == leader and score > 0.0:
                score += 0.05
                codes.append("round2_leader")
            scored.append((round(min(1.0, max(0.0, score)), 4), option, sorted(codes)))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return scored

    def _vote(self, q: Question) -> Vote:
        if q.kind == "choice_set":
            return self._unknown(q, "no_layout_signal_for_choices")

        scored = self._score(q)
        if not scored:
            return self._unknown(q, "empty_ballot")
        best_score, best_option, codes = scored[0]
        if best_score < self.floor:
            return self._unknown(q, "below_heuristic_floor")
        if len(scored) > 1 and abs(scored[1][0] - best_score) < 0.02:
            best_score = round(best_score * 0.90, 4)
            codes = sorted(codes + ["near_tie"])

        if q.kind in ("canonical_key", "ambiguity"):
            return _key_vote(self, q, best_option, best_score, codes)
        if q.kind == "field_type":
            spec = CANONICAL_KEYS.get(best_option)
            if spec is None:
                return self._unknown(q, "no_layout_signal")
            return _type_vote(self, q, spec.field_type, best_score * 0.9, codes)
        return self._unknown(q, "unsupported_question_kind")


# ----------------------------------------------------------------------------------
# OntologyFuzzyMember
# ----------------------------------------------------------------------------------
class OntologyFuzzyMember(BaseMember):
    """``difflib`` over the alias space; the confidence is the similarity ratio.

    Two passes, both deterministic: the global :func:`zfp.ontology.fuzzy_lookup` (which
    may propose a key that is not on the ballot -- reported, never voted) and a direct
    ratio against every alias of every option, which is what lets a damaged
    "Frst Narne" still land on ``person.name.first``.
    """

    name = "ontology"

    #: Ratios below this are coincidence, not similarity.
    floor = 0.55

    def _best(self, label: str, options: Sequence[str]) -> Tuple[float, str, List[str]]:
        """Return ``(ratio, option, reason_codes)`` for the closest option."""
        norm = normalize_label(label)
        if not norm:
            return 0.0, "", []
        global_hits = dict(fuzzy_lookup(label, cutoff=self.floor))
        best_ratio = 0.0
        best_option = ""
        for option in options:
            spec = CANONICAL_KEYS.get(option)
            if spec is None:
                continue
            forms = set(aliases_for(option))
            forms.add(normalize_label(spec.label))
            forms.add(spec.leaf.replace("_", " "))
            ratio = 0.0
            for form in sorted(forms):
                if not form:
                    continue
                candidate = difflib.SequenceMatcher(None, norm, form).ratio()
                if candidate > ratio:
                    ratio = candidate
            if option in global_hits and global_hits[option] > ratio:
                ratio = global_hits[option]
            if ratio > best_ratio or (ratio == best_ratio and option < best_option):
                best_ratio, best_option = ratio, option

        codes = ["alias_similarity"]
        if global_hits and best_option not in global_hits:
            codes.append("global_hit_off_ballot")
        return round(best_ratio, 4), best_option, codes

    def _vote(self, q: Question) -> Vote:
        if q.kind == "choice_set":
            return self._unknown(q, "no_alias_signal_for_choices")

        label = _text(q.context, "label")
        if not label:
            return self._unknown(q, "no_label")
        options = _options(q)
        if q.kind == "field_type":
            hits = fuzzy_lookup(label, cutoff=self.floor)
            if not hits:
                return self._unknown(q, "no_alias_similarity")
            key, ratio = hits[0]
            spec = CANONICAL_KEYS.get(key)
            if spec is None:
                return self._unknown(q, "no_alias_similarity")
            return _type_vote(
                self, q, spec.field_type, ratio, ["alias_similarity", "key_declares_type"]
            )

        ratio, option, codes = self._best(label, options)
        if not option or ratio < self.floor:
            return self._unknown(q, "no_alias_similarity")
        leader = _round2_leader(q)
        if leader and option == leader:
            ratio = round(min(1.0, ratio + 0.05), 4)
            codes = sorted(codes + ["round2_leader"])
        return _key_vote(self, q, option, ratio, codes)


# ----------------------------------------------------------------------------------
# SiblingConsensusMember
# ----------------------------------------------------------------------------------
class SiblingConsensusMember(BaseMember):
    """Namespace completion from the candidate's resolved neighbours.

    Neighbours at ``person.address.city`` and ``person.address.region`` make the blank
    between them ``person.address.postal_code``: the form is completing an address, and
    the missing member of that namespace is the answer even when the blank's own label
    is missing or illegible.
    """

    name = "sibling"

    def _vote(self, q: Question) -> Vote:
        if q.kind == "choice_set":
            return self._unknown(q, "no_sibling_signal_for_choices")

        neighbours = _neighbours(q.context)
        if not neighbours:
            return self._unknown(q, "no_resolved_neighbours")

        counts: Dict[str, int] = {}
        for neighbour in neighbours:
            parent = _parent_of(neighbour)
            counts[parent] = counts.get(parent, 0) + 1
        parent, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        share = count / float(len(neighbours))
        taken = set(neighbours)
        leader = _round2_leader(q)

        scored: List[Tuple[float, str, List[str]]] = []
        for option in _options(q):
            if _parent_of(option) != parent:
                continue
            codes = ["namespace_completion:%s" % parent]
            score = 0.40 + 0.45 * share
            if option in taken:
                score -= 0.25
                codes.append("namespace_slot_taken")
            if count < 2:
                score = min(score, 0.55)
                codes.append("single_neighbour")
            if leader and option == leader:
                score += 0.05
                codes.append("round2_leader")
            scored.append((round(min(1.0, max(0.0, score)), 4), option, sorted(codes)))

        if not scored:
            return self._unknown(q, "no_sibling_on_ballot")
        scored.sort(key=lambda item: (-item[0], item[1]))
        score, option, codes = scored[0]

        if q.kind == "field_type":
            spec = CANONICAL_KEYS.get(option)
            if spec is None:
                return self._unknown(q, "no_sibling_on_ballot")
            return _type_vote(self, q, spec.field_type, score * 0.8, codes)
        return _key_vote(self, q, option, score, codes)


# ----------------------------------------------------------------------------------
# LocalModelMember
# ----------------------------------------------------------------------------------
#: Optional module a deployment can install to supply an on-device classifier.  It must
#: expose ``classify(question: dict) -> dict | None`` returning an answer object shaped
#: like the question's schema.  Absent by default, and absence is not an error.
LOCAL_CLASSIFIER_MODULE = "zfp_local_classifier"


class LocalModelMember(BaseMember):
    """Hook for an on-device classifier.  Disabled unless one is supplied.

    A local model is not an egress concern -- nothing leaves the machine -- but it is
    also not something ZFP ships, so this member is a socket for one: pass any callable
    taking the question as a plain dict and returning an answer object (or ``None``), or
    install a module named :data:`LOCAL_CLASSIFIER_MODULE` exposing ``classify``.  With
    neither, :meth:`available` is false and the council simply never asks.

    The reply is held to exactly the same standard as the remote member's: validated
    against the question schema, refused when it names a key outside the ballot, and
    never allowed to raise.
    """

    name = "local_model"

    def __init__(
        self,
        classifier: Optional[Callable[[Dict[str, Any]], Optional[Mapping[str, Any]]]] = None,
        *,
        module_name: str = LOCAL_CLASSIFIER_MODULE,
        name: Optional[str] = None,
    ) -> None:
        self.module_name = str(module_name)
        self.classifier = classifier if classifier is not None else self._discover()
        if name:
            self.name = str(name)

    def _discover(self) -> Optional[Callable[[Dict[str, Any]], Optional[Mapping[str, Any]]]]:
        """Look for an optional on-device classifier; degrade quietly when absent."""
        found = optional_import(self.module_name, attr="classify")
        if not found:
            return None
        return found.module  # type: ignore[return-value]

    def available(self) -> bool:
        """True only when a classifier was supplied or discovered."""
        return callable(self.classifier)

    def _vote(self, q: Question) -> Vote:
        if not callable(self.classifier):
            return self._unknown(q, "no_local_classifier")
        reply = self.classifier(q.as_dict())
        if reply is None:
            return self._unknown(q, "classifier_declined")
        answer = dict(reply)
        if not validate_answer(answer, q.schema):
            return Vote.failed(self.name, "classifier reply does not satisfy the question schema")
        value = answer.get(q.answer_field)
        if isinstance(value, str) and value != UNKNOWN and not q.allows(value):
            return Vote.failed(self.name, "classifier answered %r, which is off the ballot" % value)
        if value is None or (isinstance(value, str) and value == UNKNOWN):
            return self._unknown(q, "classifier_answered_unknown")
        codes = [str(c) for c in (answer.get("reason_codes") or [])]
        return self._answer(q, value, answer.get("confidence", 0.0), codes + ["local_classifier"])


# ----------------------------------------------------------------------------------
# OpenRouterMember
# ----------------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are one member of a form-understanding council. You are given the redacted, "
    "structural context of a single blank on a form -- never the document and never any "
    "value. Answer the question with JSON matching the provided schema and nothing "
    "else. Choose one entry from the enum; if the evidence does not support any entry, "
    'answer "unknown". Never invent a key. Never describe where the field is.'
)


class OpenRouterMember(BaseMember):
    """Optional remote member, off by default.

    ``available()`` is false unless an API key is present *and*
    ``PrivacyConfig.allow_external_inference`` is true, so a default configuration never
    reaches the network.  :meth:`vote` raises :class:`~zfp.core.errors.PolicyError` when
    called anyway -- a refusal is loud, not silent.

    Every other failure (transport, malformed JSON, an answer that misses the schema, a
    key outside the ballot) comes back as a zero-confidence vote with ``error`` set, so
    one flaky provider cannot take the council down.
    """

    name = "openrouter"
    remote = True

    def __init__(
        self,
        model: str = _openrouter.DEFAULT_MODEL,
        *,
        api_key: Optional[str] = None,
        privacy: Optional[PrivacyConfig] = None,
        base_url: str = _openrouter.DEFAULT_BASE_URL,
        timeout: float = _openrouter.DEFAULT_TIMEOUT,
        name: Optional[str] = None,
    ) -> None:
        self.model = str(model)
        self.privacy = privacy if privacy is not None else PrivacyConfig()
        self.base_url = str(base_url)
        self.timeout = float(timeout)
        self.api_key = api_key if api_key is not None else os.environ.get("OPENROUTER_API_KEY", "")
        if name:
            self.name = str(name)

    def available(self) -> bool:
        """True only when a key exists and the privacy policy permits egress."""
        return bool(self.api_key) and bool(self.privacy.allow_external_inference)

    def messages_for(self, q: Question) -> List[Dict[str, str]]:
        """Build the two-message payload for a question.

        The user message is the redacted context serialized with sorted keys, so the
        same question produces byte-identical bytes on the wire across runs.
        """
        payload = {
            "question": q.prompt,
            "kind": q.kind,
            "context": q.context,
            "options": list(q.options) + [UNKNOWN],
        }
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, sort_keys=True, ensure_ascii=False)},
        ]

    def _vote(self, q: Question) -> Vote:
        if not self.privacy.allow_external_inference:
            raise PolicyError(
                "OpenRouterMember refuses to vote: PrivacyConfig.allow_external_inference "
                "is False"
            )
        if not self.api_key:
            return Vote.failed(self.name, "no OPENROUTER_API_KEY available")

        # Last gate before a socket exists.  A secret here is a privacy incident, not a
        # bad answer, so it raises instead of degrading.
        assert_no_secrets(q.context, self.privacy)

        started = _openrouter.monotonic_ms()
        try:
            reply = _openrouter.chat_json(
                self.model,
                self.messages_for(q),
                q.schema,
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                zdr=self.privacy.require_zero_data_retention,
                allow_providers=list(self.privacy.provider_allowlist) or None,
                privacy=self.privacy,
            )
        except PolicyError:
            raise
        except Exception as exc:  # noqa: BLE001 - a remote failure is a vote, not a crash
            vote = Vote.failed(self.name, "%s: %s" % (type(exc).__name__, exc))
            vote.latency_ms = _openrouter.monotonic_ms() - started
            return vote

        latency = _openrouter.monotonic_ms() - started
        if not validate_answer(reply, q.schema):
            vote = Vote.failed(self.name, "reply does not satisfy the question schema")
            vote.latency_ms = latency
            return vote

        value = reply.get(q.answer_field)
        confidence = reply.get("confidence", 0.0)
        codes = [str(c) for c in (reply.get("reason_codes") or [])]
        signature = value if isinstance(value, str) else UNKNOWN
        if isinstance(value, str) and value != UNKNOWN and not q.allows(value):
            vote = Vote.failed(self.name, "answer %r is outside the ballot" % value)
            vote.latency_ms = latency
            return vote
        if signature == UNKNOWN and not isinstance(value, (list, tuple)):
            vote = self._unknown(q, "model_answered_unknown")
            vote.latency_ms = latency
            return vote

        vote = self._answer(q, value, confidence, codes + ["remote_model:%s" % self.model])
        vote.latency_ms = latency
        return vote


def local_members() -> List[BaseMember]:
    """Return the four always-available local members, in deterministic name order."""
    members: List[BaseMember] = [
        HeuristicMember(),
        OntologyFuzzyMember(),
        RulesMember(),
        SiblingConsensusMember(),
    ]
    members.sort(key=lambda m: m.name)
    return members
