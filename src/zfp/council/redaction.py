"""Egress redaction -- the privacy boundary of the council.

``docs/PRIVACY.md`` states what may leave the machine when external inference is
explicitly enabled: the visible label, the section/parent context, the document title,
the *shape* of the blank, and the canonical keys under consideration.  Never the
document, never a page image, never a page's full text, never a value.

This module is what enforces that.  :func:`redact_context` is applied to **every**
question context -- including the ones only local members will ever see -- so that a
context which later escalates cannot carry something the local path was sloppy about.

Three rules, in order:

1. **Drop by key.**  Any context key naming a ``secret``-class ontology key (``ssn``,
   ``ein``, ``card_number``, ``cvv``, ``password``, ...) is removed outright, as is any
   key that names a stored *value* rather than a structural fact.
2. **Mask by shape.**  With ``PrivacyConfig.redact_values_in_prompts`` set, every digit
   becomes ``#``.  ``"12-3456789"`` becomes ``"##-#######"``: the placeholder's *shape*
   survives -- which is exactly what makes it recognizable as an EIN blank -- while its
   content does not.  Email-shaped strings additionally lose their letters.
3. **Truncate.**  Every string is truncated against one shared
   ``max_context_chars`` budget, spent in sorted-key order so the result is identical
   across runs.

:func:`assert_no_secrets` is the last gate before a socket is opened.  It raises
:class:`~zfp.core.errors.PolicyError` when a secret-class value survived anyway; the QA
dashboard counts those as *unapproved PII egress*, and the count must be zero.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.config import PrivacyConfig
from ..core.errors import PolicyError
from ..ontology import CANONICAL_KEYS, lookup

__all__ = [
    "SECRET_VALUE_PATTERNS",
    "VALUE_KEY_TOKENS",
    "ID_KEY_NAMES",
    "redact_text",
    "redact_context",
    "assert_no_secrets",
    "context_char_count",
    "secret_key_for",
]

#: Context keys whose *content* is a stored value rather than a structural fact.
#: These never leave the machine, whatever they contain.
VALUE_KEY_TOKENS: Tuple[str, ...] = (
    "value",
    "values",
    "filled",
    "filled_value",
    "filled_values",
    "current_value",
    "default_value",
    "answer_value",
    "vault",
    "vault_entry",
    "profile",
    "secret",
    "secrets",
    "credential",
    "credentials",
    "page_text",
    "full_text",
    "document_text",
    "page_image",
    "image",
    "raw",
)

#: Keys holding an opaque correlation identifier.  A ``stable_id`` is hex, so it trips
#: digit heuristics constantly; it is exempt from digit masking and from the digit-run
#: consistency check, but never from the shaped secret detectors below.
ID_KEY_NAMES: Tuple[str, ...] = ("id", "candidate", "question", "group", "group_id")

_DIGIT_RE = re.compile(r"\d")
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMAILISH_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z#]{2,}$")
_LETTER_RE = re.compile(r"[A-Za-z]")
_LONG_DIGIT_RUN_RE = re.compile(r"\d{5,}")

#: Unanchored detectors for values that must never appear in an outbound prompt.
#: Shape-specific on purpose: a false negative is a privacy incident, a false positive is
#: only a refusal, so these lean strict but never fire on a masked (``#``) shape.
SECRET_VALUE_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("ein", re.compile(r"\b\d{2}-\d{7}\b")),
    ("itin", re.compile(r"\b9\d{2}-\d{2}-\d{4}\b")),
    ("card_number", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{3,4}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")),
    ("bare_digit_run", re.compile(r"\b\d{9,}\b")),
)


def _norm_key(name: Any) -> str:
    """Return a context key in a comparable form."""
    return str(name).strip().lower()


def secret_key_for(name: Any) -> Optional[str]:
    """Return the ``secret``-class canonical key a context key names, if any.

    ``"person.ssn"`` hits :data:`~zfp.ontology.CANONICAL_KEYS` directly; ``"ssn"``,
    ``"tax_id"`` and ``"card_number"`` resolve through the alias index.  Returns
    ``None`` for a key that names nothing secret.
    """
    raw = str(name).strip()
    spec = CANONICAL_KEYS.get(raw)
    if spec is None:
        hit = lookup(raw)
        spec = CANONICAL_KEYS.get(hit) if hit else None
    if spec is not None and spec.sensitivity == "secret":
        return spec.key
    return None


def _is_value_key(name: Any) -> bool:
    """True when a context key names stored content rather than structure."""
    norm = _norm_key(name).replace("-", "_").replace(" ", "_")
    if norm in VALUE_KEY_TOKENS:
        return True
    return any(norm.endswith("_" + token) for token in VALUE_KEY_TOKENS)


def _is_id_key(name: Any) -> bool:
    """True when a context key holds an opaque identifier."""
    norm = _norm_key(name).replace("-", "_").replace(" ", "_")
    return norm in ID_KEY_NAMES or norm.endswith("_id")


def redact_text(s: str, privacy: PrivacyConfig) -> str:
    """Redact a single string for outbound use.

    Collapses whitespace, strips control characters, replaces every digit with ``#``
    when ``privacy.redact_values_in_prompts`` is set, blanks the letters of an
    email-shaped token, and truncates to ``privacy.max_context_chars``.
    """
    text = _CTRL_RE.sub(" ", str(s))
    text = _WS_RE.sub(" ", text).strip()
    if privacy.redact_values_in_prompts:
        emailish = _EMAILISH_RE.match(text) is not None
        text = _DIGIT_RE.sub("#", text)
        if emailish:
            text = _LETTER_RE.sub("x", text)
    limit = max(0, int(privacy.max_context_chars))
    if len(text) > limit:
        text = text[:limit]
    return text


class _Budget:
    """A shared character allowance spent in a deterministic walk order."""

    def __init__(self, total: int) -> None:
        self.remaining = max(0, int(total))

    def take(self, text: str) -> str:
        """Return as much of ``text`` as the remaining allowance permits."""
        if self.remaining <= 0:
            return ""
        if len(text) <= self.remaining:
            self.remaining -= len(text)
            return text
        kept = text[: self.remaining]
        self.remaining = 0
        return kept


def _redact_value(
    value: Any, privacy: PrivacyConfig, budget: _Budget, *, opaque: bool
) -> Any:
    """Redact one context value, recursing into mappings and sequences."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        # Small integers are structure (comb cells, page index, character budgets).
        # A long one is a value wearing a number's clothes.
        if privacy.redact_values_in_prompts and not opaque and abs(value) >= 10**6:
            return "#" * len(str(abs(value)))
        return value
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, str):
        if opaque:
            text = _WS_RE.sub(" ", _CTRL_RE.sub(" ", value)).strip()
        else:
            text = redact_text(value, privacy)
        return budget.take(text)
    if isinstance(value, Mapping):
        return _redact_mapping(value, privacy, budget)
    if isinstance(value, (list, tuple, set, frozenset)):
        items: Sequence[Any]
        if isinstance(value, (set, frozenset)):
            items = sorted(str(v) for v in value)
        else:
            items = list(value)
        out: List[Any] = []
        for item in items:
            out.append(_redact_value(item, privacy, budget, opaque=opaque))
        return out
    return budget.take(redact_text(str(value), privacy))


def _redact_mapping(
    ctx: Mapping[str, Any], privacy: PrivacyConfig, budget: _Budget
) -> Dict[str, Any]:
    """Redact a mapping, dropping secret and value-bearing keys."""
    out: Dict[str, Any] = {}
    for raw_key in sorted(ctx.keys(), key=lambda k: str(k)):
        key = str(raw_key)
        if secret_key_for(key) is not None:
            continue
        if _is_value_key(key):
            continue
        out[key] = _redact_value(
            ctx[raw_key], privacy, budget, opaque=_is_id_key(key)
        )
    return out


def redact_context(
    ctx: Mapping[str, Any], privacy: Optional[PrivacyConfig] = None
) -> Dict[str, Any]:
    """Return a redacted copy of a council question context.

    Args:
        ctx: The raw context assembled by a question builder.
        privacy: Egress policy.  Defaults to a fresh :class:`PrivacyConfig`, which is
            the local-first default (no egress, values redacted).

    Returns:
        A new dict with secret-class and value-bearing keys removed, strings masked
        and the total string budget capped at ``privacy.max_context_chars``.  Keys are
        emitted in sorted order, so two runs produce byte-identical JSON.
    """
    policy = privacy if privacy is not None else PrivacyConfig()
    if not isinstance(ctx, Mapping):
        raise PolicyError("redact_context expects a mapping, got %r" % type(ctx).__name__)
    budget = _Budget(policy.max_context_chars)
    return _redact_mapping(ctx, policy, budget)


def _walk(ctx: Any, path: str = "") -> List[Tuple[str, Any]]:
    """Flatten a context into ``(path, scalar)`` pairs in deterministic order."""
    found: List[Tuple[str, Any]] = []
    if isinstance(ctx, Mapping):
        for key in sorted(ctx.keys(), key=lambda k: str(k)):
            child = "%s.%s" % (path, key) if path else str(key)
            found.extend(_walk(ctx[key], child))
    elif isinstance(ctx, (list, tuple, set, frozenset)):
        items = sorted(ctx, key=str) if isinstance(ctx, (set, frozenset)) else list(ctx)
        for index, item in enumerate(items):
            found.extend(_walk(item, "%s[%d]" % (path, index)))
    else:
        found.append((path, ctx))
    return found


def context_char_count(ctx: Mapping[str, Any]) -> int:
    """Return the total number of characters held in a context's strings."""
    return sum(len(v) for _, v in _walk(ctx) if isinstance(v, str))


def _leaf_name(path: str) -> str:
    """Return the last mapping key of a walk path (``"shape.width[0]"`` -> ``"width"``)."""
    tail = path.rsplit(".", 1)[-1]
    return tail.split("[", 1)[0]


def assert_no_secrets(
    ctx: Mapping[str, Any], privacy: Optional[PrivacyConfig] = None
) -> None:
    """Raise :class:`~zfp.core.errors.PolicyError` if a secret survived redaction.

    This is the check the QA dashboard's *unapproved PII egress* counter is built on.
    It runs unconditionally -- a secret-class value must never leave the machine under
    any configuration -- and it is the last thing that happens before
    :func:`zfp.council.openrouter.chat_json` opens a socket.

    Args:
        ctx: The context about to be sent.
        privacy: Egress policy.  When ``redact_values_in_prompts`` is set, a surviving
            run of five or more digits is treated as proof the redactor was bypassed.

    Raises:
        PolicyError: On a secret-class key, a shaped secret value (SSN, EIN, card
            number, IBAN, long bare digit run), or an unmasked digit run.
    """
    policy = privacy if privacy is not None else PrivacyConfig()
    if not isinstance(ctx, Mapping):
        raise PolicyError("assert_no_secrets expects a mapping, got %r" % type(ctx).__name__)

    for path, value in _walk(ctx):
        name = _leaf_name(path)
        secret = secret_key_for(name)
        if secret is not None:
            raise PolicyError(
                "secret-class key %r (%s) present in council context at %r"
                % (name, secret, path or "<root>")
            )
        if value is None or isinstance(value, bool):
            continue
        text = value if isinstance(value, str) else str(value)
        if not text:
            continue
        opaque = _is_id_key(name)
        for label, pattern in SECRET_VALUE_PATTERNS:
            if opaque and label == "bare_digit_run":
                continue
            if pattern.search(text):
                raise PolicyError(
                    "unapproved PII egress: %s-shaped value at %r"
                    % (label, path or "<root>")
                )
        if policy.redact_values_in_prompts and not opaque:
            if _LONG_DIGIT_RUN_RE.search(text):
                raise PolicyError(
                    "unredacted digit run at %r: the redactor was bypassed"
                    % (path or "<root>")
                )
