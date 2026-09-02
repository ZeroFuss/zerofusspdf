"""The semantic council: the only stage of ZFP that may hold an opinion.

Everything upstream of this package is deterministic: the interpreter reads the content
stream, the detectors propose rectangles, fusion snaps them to the ink, the ontology
resolves the labels it recognizes.  A council convenes **only** for what is left --
a field whose *meaning* those stages could not settle.

Three properties define the protocol:

**Closed questions.**  A member votes on one :class:`~zfp.council.base.Question` with a
strict JSON schema whose answer space is an enum drawn from the ontology, plus
``"unknown"``.  There is no free prose anywhere, in either direction.

**No geometry.**  No model is ever asked where a rectangle is.  Geometry is evidence,
not opinion; the council is asked what a field *means*.

**Local by default.**  Four members -- ontology rules, layout heuristics, fuzzy alias
matching and sibling-namespace completion -- are pure Python and always available.  The
remote member refuses to run unless ``PrivacyConfig.allow_external_inference`` is
explicitly enabled, and even then it sends only redacted structural context under
``max_context_chars``, over zero-data-retention routing, to a provider on the allow-list.

Typical use::

    from zfp.core.config import ZfpConfig
    from zfp.council import build_default_council, canonical_key_question

    council = build_default_council(ZfpConfig.default())
    question = canonical_key_question(candidate, options, {"document_title": title})
    verdict = council.deliberate(question)
    if verdict.answer is not None:
        candidate.canonical_key = verdict.answer["canonical_key"]

``verdict.answer`` is ``None`` whenever the council could not clear
``CouncilConfig.escalate_below_confidence``.  That is a real answer -- "data
unavailable" -- and it is always preferable to an invented one.
"""

from __future__ import annotations

from .base import (
    ANSWER_FIELDS,
    QUESTION_KINDS,
    UNKNOWN,
    BaseMember,
    CouncilMember,
    Question,
    Verdict,
    Vote,
    answer_field,
    answer_signature,
)
from .council import Council, build_default_council
from .members import (
    HeuristicMember,
    LocalModelMember,
    OntologyFuzzyMember,
    OpenRouterMember,
    RulesMember,
    SiblingConsensusMember,
    local_members,
)
from .questions import (
    ambiguity_question,
    ambiguity_schema,
    canonical_key_question,
    canonical_key_schema,
    choice_set_question,
    choice_set_schema,
    derive_options,
    field_type_question,
    field_type_schema,
    question_context,
    validate_answer,
)
from .redaction import assert_no_secrets, redact_context, redact_text

__all__ = [
    # protocol
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
    # council
    "Council",
    "build_default_council",
    # members
    "RulesMember",
    "HeuristicMember",
    "OntologyFuzzyMember",
    "SiblingConsensusMember",
    "LocalModelMember",
    "OpenRouterMember",
    "local_members",
    # questions
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
    # privacy
    "redact_context",
    "redact_text",
    "assert_no_secrets",
]
