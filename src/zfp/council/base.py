"""Core protocol types of the semantic council.

The council is the only stage of ZFP that may consult an opinion instead of a rule, so
its protocol is deliberately narrow:

* a :class:`Question` is one closed question with a JSON Schema the answer must satisfy;
* a :class:`Vote` is one member's structured answer plus a confidence;
* a :class:`Verdict` is the analyst's reading of the votes -- consensus, dissent,
  contradictions and blind spots.

There is **no free prose anywhere in the protocol**.  A member never returns a sentence,
never returns a rectangle, and never returns a key outside the question's closed option
set.  Everything a member is allowed to say is expressible as JSON matching
``Question.schema``.

Determinism: :attr:`Question.id` is a :func:`~zfp.core.ids.stable_id` over
``(kind, prompt, sorted context)``, so the same question asked in two different runs --
or two different processes -- carries the same identifier.  Ties are broken on
``(-confidence, member_name)`` everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - typing only, Protocol exists on every supported runtime
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover - Python < 3.8
    Protocol = object  # type: ignore[assignment]

    def runtime_checkable(cls):  # type: ignore[misc]
        return cls


from ..core.errors import CouncilError
from ..core.ids import stable_id

__all__ = [
    "UNKNOWN",
    "QUESTION_KINDS",
    "ANSWER_FIELDS",
    "answer_field",
    "answer_signature",
    "Question",
    "Vote",
    "Verdict",
    "CouncilMember",
    "BaseMember",
]

#: The single reserved answer every schema admits: "I do not know".  A member that
#: cannot answer says this rather than guessing, and the analyst never lets the unknown
#: group win a vote.
UNKNOWN = "unknown"

#: The four questions the council is allowed to be asked.
QUESTION_KINDS: Tuple[str, ...] = (
    "canonical_key",
    "field_type",
    "choice_set",
    "ambiguity",
)

#: The name of the decisive property in each kind's answer schema.
ANSWER_FIELDS: Dict[str, str] = {
    "canonical_key": "canonical_key",
    "field_type": "field_type",
    "choice_set": "choices",
    "ambiguity": "winner",
}


def answer_field(kind: str) -> str:
    """Return the decisive schema property for a question ``kind``.

    Unknown kinds fall back to ``"answer"`` so a future question type still groups
    sensibly instead of collapsing every vote onto one bucket.
    """
    return ANSWER_FIELDS.get(kind, "answer")


def answer_signature(answer: Optional[Mapping[str, Any]], kind: str) -> str:
    """Collapse an answer into the string the analyst groups votes by.

    Only the decisive property participates: ``confidence`` and ``reason_codes`` are
    commentary, not the answer.  A list answer (``choice_set``) is normalized to its
    sorted, de-duplicated members joined by ``"|"`` so that two members proposing the
    same choices in a different order agree.  Anything empty, missing or explicitly
    unknown collapses onto :data:`UNKNOWN`.
    """
    if not answer:
        return UNKNOWN
    value = answer.get(answer_field(kind))
    if value is None:
        return UNKNOWN
    if isinstance(value, (list, tuple)):
        parts = sorted({str(v).strip() for v in value if str(v).strip()})
        parts = [p for p in parts if p != UNKNOWN]
        return "|".join(parts) if parts else UNKNOWN
    text = str(value).strip()
    return text if text else UNKNOWN


@dataclass(frozen=True)
class Question:
    """One closed question put to the council.

    Attributes:
        id: Deterministic identifier over ``(kind, prompt, sorted context)``.
        kind: One of :data:`QUESTION_KINDS`.
        prompt: The human-readable question.  Never contains a document value; it is
            built from the already-redacted context.
        schema: JSON Schema (the subset understood by
            :func:`zfp.council.questions.validate_answer`) every answer must satisfy.
        context: Redacted structural context -- label shape, section, geometry facts.
            Never the document, never a page image, never a value.
        options: The closed option set the answer's enum was built from, without the
            trailing ``"unknown"``.

    ``schema`` and ``context`` are excluded from ``__hash__`` (dicts are unhashable)
    but still participate in ``__eq__``; since ``id`` is derived from both, hashing on
    the remaining fields stays consistent with equality.
    """

    id: str
    kind: str
    prompt: str
    schema: Dict[str, Any] = field(hash=False)
    context: Dict[str, Any] = field(hash=False)
    options: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.kind or not str(self.kind).strip():
            raise CouncilError("Question.kind must be a non-empty string")
        if not isinstance(self.schema, dict):
            raise CouncilError("Question.schema must be a dict")
        if not isinstance(self.context, dict):
            raise CouncilError("Question.context must be a dict")
        if not isinstance(self.options, tuple):
            object.__setattr__(self, "options", tuple(str(o) for o in self.options))

    # -- construction ------------------------------------------------------------
    @staticmethod
    def compute_id(kind: str, prompt: str, context: Mapping[str, Any]) -> str:
        """Return the stable identifier of a question.

        ``canonical_repr`` sorts mappings by key, so a context assembled in a different
        insertion order yields the same id.
        """
        return stable_id(str(kind), str(prompt), dict(context or {}), prefix="q")

    @staticmethod
    def build(
        kind: str,
        prompt: str,
        schema: Mapping[str, Any],
        context: Mapping[str, Any],
        options: Sequence[str] = (),
    ) -> "Question":
        """Build a question, deriving :attr:`id` from its content."""
        ctx = dict(context or {})
        return Question(
            id=Question.compute_id(kind, prompt, ctx),
            kind=str(kind),
            prompt=str(prompt),
            schema=dict(schema or {}),
            context=ctx,
            options=tuple(str(o) for o in options),
        )

    def derived(self, **context_updates: Any) -> "Question":
        """Return a copy of this question with extra context and a new id.

        Used by the council for a second deliberation round: the disagreement summary
        is *context*, so the derived question is a genuinely different question with a
        genuinely different identifier, and the run stays reproducible.
        """
        ctx = dict(self.context)
        ctx.update(context_updates)
        return Question(
            id=Question.compute_id(self.kind, self.prompt, ctx),
            kind=self.kind,
            prompt=self.prompt,
            schema=dict(self.schema),
            context=ctx,
            options=tuple(self.options),
        )

    # -- accessors ---------------------------------------------------------------
    @property
    def answer_field(self) -> str:
        """The decisive property name in this question's schema."""
        return answer_field(self.kind)

    def enum(self) -> List[str]:
        """Return the closed enum the answer's decisive property admits."""
        props = self.schema.get("properties") or {}
        spec = props.get(self.answer_field) or {}
        if "enum" in spec:
            return [str(v) for v in spec["enum"]]
        items = spec.get("items") or {}
        if isinstance(items, dict) and "enum" in items:
            return [str(v) for v in items["enum"]]
        return list(self.options) + [UNKNOWN]

    def allows(self, value: str) -> bool:
        """True when ``value`` is inside this question's closed enum."""
        return str(value) in self.enum()

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready mapping."""
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "schema": self.schema,
            "context": self.context,
            "options": list(self.options),
        }


@dataclass
class Vote:
    """One member's structured answer to one question.

    Attributes:
        member: The member's stable name.  Also the final deterministic tie-break.
        answer: JSON object matching the question schema, or ``{}`` on failure.
        confidence: 0..1.  Doubles as the vote's *weight* in the analyst's consensus.
        reason_codes: Short machine-readable codes ("placeholder_pattern",
            "section_namespace") -- never prose.
        latency_ms: Measured only for the remote member; local members leave it at
            ``0.0`` so a local council is bit-for-bit reproducible.
        error: Set when the member failed internally.  A member never raises for its
            own failure; it returns a zero-confidence vote carrying the message.
    """

    member: str
    answer: Dict[str, Any]
    confidence: float
    reason_codes: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None

    def __post_init__(self) -> None:
        self.member = str(self.member)
        self.answer = dict(self.answer or {})
        self.confidence = _clamp01(self.confidence)
        self.reason_codes = [str(c) for c in (self.reason_codes or [])]

    @classmethod
    def unknown(
        cls, member: str, *, reason: str = "no_signal", confidence: float = 0.0
    ) -> "Vote":
        """Return an explicit "I do not know" vote."""
        return cls(
            member=member,
            answer={"confidence": _clamp01(confidence)},
            confidence=_clamp01(confidence),
            reason_codes=[reason],
        )

    @classmethod
    def failed(cls, member: str, message: str) -> "Vote":
        """Return a zero-confidence vote recording an internal failure."""
        return cls(
            member=member,
            answer={},
            confidence=0.0,
            reason_codes=["member_error"],
            error=str(message),
        )

    def signature(self, kind: str) -> str:
        """Return the grouping signature of this vote for a question ``kind``."""
        return answer_signature(self.answer, kind)

    def is_unknown(self, kind: str) -> bool:
        """True when this vote does not decide the question."""
        return self.signature(kind) == UNKNOWN

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready mapping."""
        return {
            "member": self.member,
            "answer": dict(self.answer),
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


@dataclass
class Verdict:
    """The analyst's reading of a round of votes.

    Attributes:
        question_id: The :attr:`Question.id` this verdict answers.
        answer: The winning answer object, or ``None``.  ``None`` is a real outcome:
            below ``CouncilConfig.escalate_below_confidence`` the council declines to
            guess rather than inventing a key.
        confidence: The winning group's mean confidence scaled by :attr:`consensus`.
        consensus: Winning group weight / total weight, weights being confidences.
        votes: Every vote cast, sorted ``(-confidence, member)``.
        dissent: Every vote outside the winning group.
        blind_spots: What *no* member addressed.
        contradictions: Pairs of high-confidence votes that disagree.
        escalated: True when a second round ran or the answer was withheld.
        agreed_by: Names of the members backing the winning answer.
    """

    question_id: str
    answer: Optional[Dict[str, Any]]
    confidence: float
    consensus: float
    votes: List[Vote]
    dissent: List[Vote]
    blind_spots: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    escalated: bool = False
    agreed_by: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)
        self.consensus = _clamp01(self.consensus)

    @property
    def decided(self) -> bool:
        """True when the council produced an answer."""
        return self.answer is not None

    def value(self, kind: str) -> Optional[Any]:
        """Return the decisive value of the answer, or ``None`` when undecided."""
        if self.answer is None:
            return None
        return self.answer.get(answer_field(kind))

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready mapping (used by traces and the QA dashboard)."""
        return {
            "question_id": self.question_id,
            "answer": None if self.answer is None else dict(self.answer),
            "confidence": self.confidence,
            "consensus": self.consensus,
            "votes": [v.as_dict() for v in self.votes],
            "dissent": [v.as_dict() for v in self.dissent],
            "blind_spots": list(self.blind_spots),
            "contradictions": list(self.contradictions),
            "escalated": self.escalated,
            "agreed_by": list(self.agreed_by),
        }


@runtime_checkable
class CouncilMember(Protocol):
    """The one interface a council member must satisfy."""

    name: str

    def available(self) -> bool:
        """True when this member can vote right now (deps, credentials, policy)."""
        ...

    def vote(self, q: Question) -> Vote:
        """Return a structured vote.  Must not raise for its own internal failure."""
        ...


class BaseMember:
    """Convenience base implementing the "never raise for your own failure" rule.

    Subclasses implement :meth:`_vote`.  Anything that escapes it is captured into a
    zero-confidence :class:`Vote` carrying the error, except a
    :class:`~zfp.core.errors.PolicyError`, which is a deliberate refusal by the privacy
    layer and must reach the caller.
    """

    #: Stable member name; also the last deterministic tie-break.
    name: str = "member"
    #: True for members that leave the machine.  Used to report a local-only blind spot.
    remote: bool = False

    def available(self) -> bool:
        """True by default: the local members are always available."""
        return True

    def vote(self, q: Question) -> Vote:
        """Run :meth:`_vote`, converting an internal failure into a failed vote."""
        from ..core.errors import PolicyError  # local import: keeps the module graph flat

        try:
            vote = self._vote(q)
        except PolicyError:
            raise
        except Exception as exc:  # noqa: BLE001 - a member must never break the council
            return Vote.failed(self.name, "%s: %s" % (type(exc).__name__, exc))
        if vote is None:
            return Vote.unknown(self.name, reason="no_answer")
        return vote

    def _vote(self, q: Question) -> Optional[Vote]:
        """Produce this member's vote.  Overridden by every concrete member."""
        raise CouncilError("%s does not implement _vote" % type(self).__name__)

    # -- helpers shared by the concrete members ----------------------------------
    def _answer(
        self, q: Question, value: Any, confidence: float, reason_codes: Iterable[str]
    ) -> Vote:
        """Build a schema-shaped vote for ``value`` with ``confidence``."""
        conf = _clamp01(confidence)
        answer: Dict[str, Any] = {
            q.answer_field: value,
            "confidence": conf,
        }
        codes = sorted({str(c) for c in reason_codes})
        if codes:
            answer["reason_codes"] = codes
        return Vote(
            member=self.name, answer=answer, confidence=conf, reason_codes=codes
        )

    def _unknown(self, q: Question, reason: str = "no_signal") -> Vote:
        """Build an explicit unknown vote for ``q``."""
        answer: Dict[str, Any] = {q.answer_field: UNKNOWN, "confidence": 0.0}
        if q.kind == "choice_set":
            answer[q.answer_field] = []
        answer["reason_codes"] = [reason]
        return Vote(member=self.name, answer=answer, confidence=0.0, reason_codes=[reason])


def _clamp01(value: Any) -> float:
    """Clamp ``value`` into ``[0, 1]``; non-numeric input becomes ``0.0``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number
