"""Placeholder and value pattern rules.

Deterministic rules beat models whenever they fire, and printed forms are full of them.
The blank itself usually declares its own format: a line printed ``MM/DD/YYYY`` is a date,
``___-__-____`` is a US SSN, ``(___) ___-____`` is a US phone number, ``##-#######`` is an
EIN and ``$ ______.__`` is currency.  Actual *values* read back out of a filled form carry
the same information: ``12-3456789`` is an EIN, ``jane@example.com`` is an email address.

This module holds both halves.  Every :class:`PatternRule` declares whether it matches a
printed placeholder, a real value, or both, and carries a confidence the fusion stage
folds into :class:`~zfp.core.types.Confidence`.  Every regex is compiled exactly once,
when this module is imported.

Three entry points:

``match_placeholder(text)``
    Classify the literal placeholder printed inside a blank.
``match_value(value)``
    Classify a concrete value.
``infer_from_context(label, nearby)``
    Combine a label hit from :mod:`zfp.ontology.aliases` with placeholder text found near
    the candidate.  When the two agree the confidence is boosted, which is exactly the
    "``Tax ID`` + nearby ``##-#######`` -> ``company.tax_id.ein`` @ .995" cascade the
    design calls for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.types import FieldType
from .aliases import lookup as alias_lookup
from .aliases import normalize_label

__all__ = [
    "PatternRule",
    "PATTERNS",
    "PATTERNS_BY_NAME",
    "MATCH_MODES",
    "US_STATES",
    "match_placeholder",
    "match_value",
    "match_all",
    "infer_from_context",
    "rules_for_key",
]

#: Legal values for :attr:`PatternRule.matches`.
MATCH_MODES: Tuple[str, ...] = ("placeholder", "value", "both")

#: USPS two-letter codes, used by the state-abbreviation value rule.
US_STATES: Tuple[str, ...] = (
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL",
    "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE",
    "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "PR", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "VI", "WA", "WV", "WI", "WY", "AS", "GU", "MP",
    "AE", "AA", "AP",
)
_STATE_ALT = "|".join(sorted(US_STATES))

_WS_RE = re.compile(r"\s+")
_TRIM_RE = re.compile(r"^(?:[\s•*]+|-\s+)|[\s:;,]+$")
_TRAILING_PERIOD_RE = re.compile(r"(?<=[A-Za-z0-9])\.$")


@dataclass(frozen=True)
class PatternRule:
    """One deterministic format rule.

    Attributes:
        name: Stable rule identifier, used in reason codes and QA reports.
        regex: The pattern source.  It is compiled once during ``__post_init__``.
        field_type: The field type implied by a match.
        format_hint: Human format string, e.g. ``"NN-NNNNNNN"``.
        canonical_hint: Canonical key the match suggests, when the format is specific
            enough to imply one (``##-#######`` implies ``company.tax_id.ein``).
        confidence: 0..1 strength of the implication on its own, without label support.
        matches: ``"placeholder"``, ``"value"`` or ``"both"`` -- which side of the form
            lifecycle this rule applies to.
        description: One-line human explanation used in reports.
    """

    name: str
    regex: str
    field_type: FieldType = FieldType.TEXT
    format_hint: Optional[str] = None
    canonical_hint: Optional[str] = None
    confidence: float = 0.90
    matches: str = "both"
    description: str = ""
    compiled: re.Pattern = field(default=None, init=False, repr=False, compare=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.matches not in MATCH_MODES:
            raise ValueError(
                f"PatternRule {self.name!r} has invalid matches={self.matches!r}"
            )
        object.__setattr__(self, "compiled", re.compile(self.regex))

    def applies_to(self, mode: str) -> bool:
        """True when this rule participates in ``"placeholder"`` or ``"value"`` matching."""
        return self.matches == "both" or self.matches == mode

    def full_match(self, text: str) -> bool:
        """True when ``text`` matches this rule end to end."""
        return self.compiled.fullmatch(text) is not None

    def search(self, text: str) -> bool:
        """True when this rule occurs anywhere inside ``text``."""
        return self.compiled.search(text) is not None

    def as_dict(self) -> Dict[str, object]:
        """Return a plain JSON-serializable mapping of this rule."""
        return {
            "name": self.name,
            "regex": self.regex,
            "field_type": self.field_type.value,
            "format_hint": self.format_hint,
            "canonical_hint": self.canonical_hint,
            "confidence": self.confidence,
            "matches": self.matches,
            "description": self.description,
        }


_R = PatternRule
_T = FieldType.TEXT
_D = FieldType.DATE
_N = FieldType.NUMBER
_C = FieldType.CURRENCY
_E = FieldType.EMAIL
_P = FieldType.PHONE
_B = FieldType.CHECKBOX
_G = FieldType.SIGNATURE
_K = FieldType.COMB

#: Every deterministic format rule, in declaration order.
PATTERNS: List[PatternRule] = [
    # ---- printed date placeholders ---------------------------------------------------
    _R("date_mdy_placeholder", r"(?i)^m{2}\s*[/\-.]\s*d{2}\s*[/\-.]\s*y{2,4}$", _D,
       "MM/DD/YYYY", "document.today", 0.96, "placeholder",
       "US month/day/year placeholder printed in the blank"),
    _R("date_dmy_placeholder", r"(?i)^d{2}\s*[/\-.]\s*m{2}\s*[/\-.]\s*y{2,4}$", _D,
       "DD/MM/YYYY", "document.today", 0.94, "placeholder",
       "International day/month/year placeholder"),
    _R("date_iso_placeholder", r"(?i)^y{4}\s*[-/]\s*m{2}\s*[-/]\s*d{2}$", _D,
       "YYYY-MM-DD", "document.today", 0.95, "placeholder",
       "ISO 8601 placeholder"),
    _R("date_underscore_placeholder", r"^_{1,4}\s*[/\-]\s*_{1,4}\s*[/\-]\s*_{2,6}$", _D,
       "MM/DD/YYYY", "document.today", 0.88, "placeholder",
       "Three underscore runs separated by slashes: a date entry line"),
    _R("date_box_placeholder", r"(?i)^(?:mm|dd|yyyy|yy)$", _D,
       "MM/DD/YYYY", None, 0.55, "placeholder",
       "A single date component printed under one comb box"),
    _R("month_day_year_placeholder", r"(?i)^month\s*,?\s*day\s*,?\s*year$", _D,
       "Month D, YYYY", "document.today", 0.90, "placeholder",
       "Long-form 'Month Day, Year' placeholder"),
    _R("ordinal_date_placeholder",
       r"(?i)^this\s+_{2,}\s+day\s+of\s+_{2,}\s*,?\s*(?:20\s*_{1,2}|_{2,})?$", _D,
       "Month D, YYYY", "document.signed_date", 0.92, "placeholder",
       "Legal execution line: 'this ___ day of ______, 20__'"),
    _R("mmyy_expiry_placeholder", r"(?i)^m{2}\s*/\s*y{2}$", _T,
       "MM/YY", "card.expiration", 0.95, "placeholder",
       "Payment-card expiry placeholder"),

    # ---- identifier placeholders -----------------------------------------------------
    _R("ssn_placeholder", r"^_{3}\s*-\s*_{2}\s*-\s*_{4}$", _T,
       "NNN-NN-NNNN", "person.ssn", 0.97, "placeholder",
       "US SSN underscore placeholder ___-__-____"),
    _R("ssn_hash_placeholder", r"^#{3}\s*-\s*#{2}\s*-\s*#{4}$", _T,
       "NNN-NN-NNNN", "person.ssn", 0.96, "placeholder",
       "US SSN hash placeholder ###-##-####"),
    _R("ein_placeholder", r"^#{2}\s*-\s*#{7}$", _T,
       "NN-NNNNNNN", "company.tax_id.ein", 0.97, "placeholder",
       "EIN hash placeholder ##-#######"),
    _R("ein_underscore_placeholder", r"^_{2}\s*-\s*_{7}$", _T,
       "NN-NNNNNNN", "company.tax_id.ein", 0.90, "placeholder",
       "EIN underscore placeholder __-_______"),
    _R("vin_placeholder", r"(?i)^[#x]{17}$", _T,
       "17 alphanumeric", "vehicle.vin", 0.62, "placeholder",
       "Seventeen numeric/alpha cells: a VIN comb"),
    _R("iban_placeholder", r"(?i)^[a-z]{2}\s?\d{2}\s?(?:[_#x]{4}\s?){2,8}$", _T,
       "CCNN AAAA AAAA", "bank.iban", 0.85, "placeholder",
       "IBAN country/check-digit prefix followed by masked groups"),
    _R("routing_placeholder", r"^#{9}$", _T,
       "NNNNNNNNN", "bank.routing_number", 0.58, "placeholder",
       "Nine numeric cells: an ABA routing number"),
    _R("account_number_placeholder", r"^#{10,17}$", _T,
       "NNNNNNNNNNNN", "bank.account_number", 0.52, "placeholder",
       "Ten to seventeen numeric cells: a bank account number"),
    _R("card_placeholder", r"^(?:_{4}[\s\-]){3}_{4}$", _T,
       "NNNN NNNN NNNN NNNN", "card.number", 0.95, "placeholder",
       "Credit card 4x4 underscore groups"),
    _R("card_hash_placeholder", r"^(?:#{4}[\s\-]){3}#{4}$", _T,
       "NNNN NNNN NNNN NNNN", "card.number", 0.95, "placeholder",
       "Credit card 4x4 hash groups"),
    _R("cvv_placeholder", r"(?i)^(?:cvv2?|cvc|cid)\s*[:\-]?\s*[_#]{3,4}$", _T,
       "NNN", "card.cvv", 0.92, "placeholder",
       "Card security code box labelled inline"),

    # ---- address / contact placeholders ----------------------------------------------
    _R("zip4_placeholder", r"^_{5}\s*-\s*_{4}$", _T,
       "NNNNN-NNNN", "person.address.postal_code", 0.95, "placeholder",
       "ZIP+4 underscore placeholder _____-____"),
    _R("zip4_hash_placeholder", r"^#{5}\s*-\s*#{4}$", _T,
       "NNNNN-NNNN", "person.address.postal_code", 0.94, "placeholder",
       "ZIP+4 hash placeholder #####-####"),
    _R("zip5_placeholder", r"^#{5}$", _T,
       "NNNNN", "person.address.postal_code", 0.50, "placeholder",
       "Five numeric cells: a five digit ZIP comb"),
    _R("postal_ca_placeholder", r"(?i)^a\s?1\s?a\s?\s?1\s?a\s?1$", _T,
       "A1A 1A1", "person.address.postal_code", 0.90, "placeholder",
       "Canadian postal code placeholder A1A 1A1"),
    _R("state_two_letter_placeholder", r"(?i)^(?:xx|_{2})$", _T,
       "XX", "person.address.region", 0.52, "placeholder",
       "A two-cell box: a state abbreviation"),
    _R("phone_paren_placeholder", r"^\(\s*[_#]{3}\s*\)\s*[_#]{3}\s*-\s*[_#]{4}$", _P,
       "(NNN) NNN-NNNN", "person.phone.mobile", 0.97, "placeholder",
       "US phone placeholder (___) ___-____"),
    _R("phone_dash_placeholder", r"^[_#]{3}\s*-\s*[_#]{3}\s*-\s*[_#]{4}$", _P,
       "NNN-NNN-NNNN", "person.phone.mobile", 0.95, "placeholder",
       "US phone placeholder ___-___-____"),
    _R("phone_ext_placeholder", r"(?i)^(?:ext\.?|extension|x)\s*[_#]{2,5}$", _T,
       "NNNN", "person.phone.extension", 0.90, "placeholder",
       "Phone extension box"),
    _R("email_placeholder", r"(?i)^[_#x]{2,}\s*@\s*[_#x]{2,}\s*\.\s*[_#xa-z]{2,}$", _E,
       "user@example.com", "person.email", 0.94, "placeholder",
       "Masked email placeholder containing an @"),
    _R("url_placeholder", r"(?i)^(?:https?://|www\s*\.)[_#x.\s]*$", _T,
       "https://example.com", "company.website", 0.80, "placeholder",
       "Masked web address placeholder"),

    # ---- amount / numeric placeholders -----------------------------------------------
    _R("currency_placeholder", r"^\$\s*[_#]{2,}\s*\.\s*[_#]{2}$", _C,
       "$N,NNN.NN", "document.amount", 0.95, "placeholder",
       "Currency placeholder $ ______.__"),
    _R("currency_blank_placeholder", r"^\$\s*[_#]{3,}$", _C,
       "$N,NNN.NN", "document.amount", 0.88, "placeholder",
       "Dollar sign followed by a blank run"),
    _R("percent_placeholder", r"^[_#]{1,4}(?:\s*\.\s*[_#]{1,2})?\s*%$", _N,
       "N%", None, 0.90, "placeholder",
       "Percentage placeholder ending in %"),
    _R("time_placeholder",
       r"(?i)^h{1,2}\s*:\s*m{2}(?:\s*:\s*s{2})?(?:\s*(?:am\s*/\s*pm|a\s*/\s*p|am|pm))?$", _T,
       "HH:MM", None, 0.93, "placeholder",
       "Clock time placeholder HH:MM with optional meridiem"),
    _R("numeric_box_placeholder", r"^#{2,}$", _N,
       "N", None, 0.45, "placeholder",
       "Generic run of numeric cells"),

    # ---- structural placeholders -----------------------------------------------------
    _R("comb_placeholder", r"^(?:\[\s?\]|[☐□])(?:\s*(?:\[\s?\]|[☐□])){2,}$", _K,
       None, None, 0.90, "placeholder",
       "Three or more equal cells in a row: a comb field"),
    _R("checkbox_placeholder", r"^(?:\[\s*\]|\(\s*\)|[☐□○◯])$", _B,
       None, None, 0.92, "placeholder",
       "A single empty box or circle: a checkbox"),
    _R("initials_pair_placeholder", r"^[_#]{2}\s*/\s*[_#]{2}$", _T,
       "XX", "person.initials", 0.85, "placeholder",
       "Paired two-cell boxes: an initials block"),
    _R("signature_placeholder",
       r"(?i)^(?:x\s*_{3,}|_{6,}\s*\(?\s*signature\s*\)?|signature\s*_{3,}|sign\s+here\s*_*)$",
       _G, None, "person.signature", 0.90, "placeholder",
       "Signature rule: an X or a long line annotated 'signature'"),
    _R("underscore_blank_placeholder", r"^_{4,}$", _T,
       None, None, 0.40, "placeholder",
       "A bare write-on-the-line blank with no format information"),
    _R("dot_leader_placeholder", r"^[.…]{4,}$", _T,
       None, None, 0.40, "placeholder",
       "Leader dots running to a value column"),

    # ---- value classification --------------------------------------------------------
    _R("ssn_value", r"^\d{3}-\d{2}-\d{4}$", _T,
       "NNN-NN-NNNN", "person.ssn", 0.95, "value",
       "A written SSN"),
    _R("ein_value", r"^\d{2}-\d{7}$", _T,
       "NN-NNNNNNN", "company.tax_id.ein", 0.93, "value",
       "A written EIN"),
    _R("zip9_value", r"^\d{5}-\d{4}$", _T,
       "NNNNN-NNNN", "person.address.postal_code", 0.93, "value",
       "A written ZIP+4"),
    _R("zip5_value", r"^\d{5}$", _T,
       "NNNNN", "person.address.postal_code", 0.70, "value",
       "A written five digit ZIP"),
    _R("phone_us_value",
       r"^(?:\+?1[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{4}$", _P,
       "(NNN) NNN-NNNN", "person.phone.mobile", 0.90, "value",
       "A written North American phone number"),
    _R("email_value", r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$", _E,
       "user@example.com", "person.email", 0.96, "value",
       "A written email address"),
    _R("url_value", r"(?i)^(?:https?://|www\.)[\w\-.]+\.[a-z]{2,}(?:/\S*)?$", _T,
       "https://example.com", "company.website", 0.90, "value",
       "A written web address"),
    _R("currency_value", r"^-?\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$", _C,
       "$N,NNN.NN", "document.amount", 0.92, "value",
       "A written currency amount with a dollar sign"),
    _R("decimal_amount_value", r"^-?\d{1,3}(?:,\d{3})*\.\d{2}$", _C,
       "N,NNN.NN", None, 0.70, "value",
       "A two-decimal amount with no currency symbol"),
    _R("date_iso_value", r"^(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$", _D,
       "YYYY-MM-DD", "document.today", 0.95, "value",
       "A written ISO date"),
    _R("date_dmy_value", r"^(?:1[3-9]|2\d|3[01])/(?:0?[1-9]|1[0-2])/(?:\d{2}|\d{4})$", _D,
       "DD/MM/YYYY", "document.today", 0.80, "value",
       "A written date whose first component exceeds 12, so day-first"),
    _R("date_mdy_value",
       r"^(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:\d{2}|\d{4})$", _D,
       "MM/DD/YYYY", "document.today", 0.88, "value",
       "A written US date"),
    _R("mmyy_value", r"^(?:0[1-9]|1[0-2])\s*/\s*\d{2}$", _T,
       "MM/YY", "card.expiration", 0.85, "value",
       "A written card expiry"),
    _R("card_value", r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$", _T,
       "NNNN NNNN NNNN NNNN", "card.number", 0.88, "value",
       "A written sixteen digit payment card number"),
    _R("iban_value", r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$", _T,
       "CCNN AAAA AAAA", "bank.iban", 0.90, "value",
       "A written IBAN"),
    _R("vin_value", r"^[A-HJ-NPR-Z0-9]{17}$", _T,
       "17 alphanumeric", "vehicle.vin", 0.90, "value",
       "A written VIN: seventeen alphanumerics excluding I, O and Q"),
    _R("time_value", r"(?i)^(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\s*(?:[ap]\.?m\.?)?$", _T,
       "HH:MM", None, 0.85, "value",
       "A written clock time"),
    _R("percent_value", r"^\d{1,3}(?:\.\d+)?\s?%$", _N,
       "N%", None, 0.90, "value",
       "A written percentage"),
    _R("state_abbrev_value", r"^(?:" + _STATE_ALT + r")$", _T,
       "XX", "person.address.region", 0.72, "value",
       "A USPS two-letter state or territory code"),
    _R("routing_value", r"^\d{9}$", _T,
       "NNNNNNNNN", "bank.routing_number", 0.55, "value",
       "Nine bare digits: most likely an ABA routing number"),
    _R("npi_value", r"^\d{10}$", _T,
       "NNNNNNNNNN", "medical.npi", 0.42, "value",
       "Ten bare digits: possibly an NPI"),
    _R("account_number_value", r"^\d{8,17}$", _T,
       "NNNNNNNNNNNN", "bank.account_number", 0.50, "value",
       "A long bare digit run: probably an account number"),
    _R("year_value", r"^(?:18|19|20)\d{2}$", _T,
       "YYYY", None, 0.60, "value",
       "A four digit year"),
    _R("cvv_value", r"^\d{3,4}$", _T,
       "NNN", "card.cvv", 0.35, "value",
       "Three or four bare digits: possibly a card security code"),
    _R("yes_no_value", r"(?i)^(?:yes|no|y|n|true|false)$", _B,
       None, "misc.yes_no", 0.85, "value",
       "An affirmative or negative answer"),
    _R("checkbox_mark_value", r"^[xX✓✔☑☒]$", _B,
       None, None, 0.80, "value",
       "An X or tick mark written into a checkbox"),
]

#: Rule lookup by stable name.
PATTERNS_BY_NAME: Dict[str, PatternRule] = {rule.name: rule for rule in PATTERNS}
if len(PATTERNS_BY_NAME) != len(PATTERNS):
    raise ValueError("duplicate PatternRule name in PATTERNS")

_PLACEHOLDER_RULES: Tuple[PatternRule, ...] = tuple(
    sorted(
        (r for r in PATTERNS if r.applies_to("placeholder")),
        key=lambda r: (-r.confidence, r.name),
    )
)
_VALUE_RULES: Tuple[PatternRule, ...] = tuple(
    sorted(
        (r for r in PATTERNS if r.applies_to("value")),
        key=lambda r: (-r.confidence, r.name),
    )
)


def _clean(text: str) -> str:
    """Trim decoration and collapse internal whitespace without touching the glyphs."""
    if not text:
        return ""
    cleaned = _WS_RE.sub(" ", _TRIM_RE.sub("", text)).strip()
    return _TRAILING_PERIOD_RE.sub("", cleaned)


def match_placeholder(text: str) -> Optional[PatternRule]:
    """Classify the literal placeholder printed inside a blank.

    ``match_placeholder("##-#######")`` returns the EIN rule; ``"MM/DD/YYYY"`` returns
    the US date rule.  The whole (whitespace-collapsed) string must match end to end.
    Ties are broken by descending confidence then rule name, so the result is stable.
    """
    cleaned = _clean(text)
    if not cleaned:
        return None
    for rule in _PLACEHOLDER_RULES:
        if rule.full_match(cleaned):
            return rule
    return None


def match_value(value: str) -> Optional[PatternRule]:
    """Classify a concrete value, e.g. ``"12-3456789"`` -> the EIN rule."""
    cleaned = _clean(value)
    if not cleaned:
        return None
    for rule in _VALUE_RULES:
        if rule.full_match(cleaned):
            return rule
    return None


def match_all(text: str, mode: str = "placeholder") -> List[PatternRule]:
    """Return every rule of ``mode`` that matches ``text``, best first.

    Useful for the council's rules member, which wants to see the alternatives rather
    than only the winner.
    """
    if mode not in ("placeholder", "value"):
        raise ValueError(f"mode must be 'placeholder' or 'value', got {mode!r}")
    cleaned = _clean(text)
    if not cleaned:
        return []
    pool = _PLACEHOLDER_RULES if mode == "placeholder" else _VALUE_RULES
    return [rule for rule in pool if rule.full_match(cleaned)]


def rules_for_key(key: str) -> List[PatternRule]:
    """Return every rule whose ``canonical_hint`` is ``key``, best first."""
    return sorted(
        (r for r in PATTERNS if r.canonical_hint == key),
        key=lambda r: (-r.confidence, r.name),
    )


def _nearby_fragments(nearby: Sequence[str] | str | None) -> List[str]:
    """Flatten ``nearby`` into candidate placeholder strings, longest first."""
    if not nearby:
        return []
    items: Iterable[str] = [nearby] if isinstance(nearby, str) else nearby
    out: List[str] = []
    for item in items:
        text = _clean(str(item))
        if not text:
            continue
        if text not in out:
            out.append(text)
        for token in text.split(" "):
            token = token.strip()
            if token and token not in out:
                out.append(token)
    out.sort(key=lambda s: (-len(s), s))
    return out


def _boost(rule: PatternRule) -> PatternRule:
    """Return ``rule`` with its confidence pulled halfway to certainty.

    Applied when an independent label hit corroborates the placeholder, which is the
    two-source agreement the resolver treats as near-certain.
    """
    return replace(rule, confidence=round(rule.confidence + (1.0 - rule.confidence) * 0.5, 4))


def infer_from_context(
    label: str, nearby: Sequence[str] | str = ()
) -> Optional[PatternRule]:
    """Combine a label hit with placeholder text found near a candidate.

    ``infer_from_context("Tax ID", "##-#######")`` resolves the label to
    ``company.tax_id.ein`` through the alias index, finds the EIN placeholder rule in the
    nearby text, sees that the two agree and returns the rule with a boosted confidence.

    When only the nearby text matches, the best nearby rule is returned unchanged.  When
    only the label resolves, the strongest rule declared for that canonical key is
    returned, again unchanged.  Returns ``None`` when neither side says anything.
    """
    key = alias_lookup(label) if label else None
    fragments = _nearby_fragments(nearby)

    found: List[PatternRule] = []
    for fragment in fragments:
        rule = match_placeholder(fragment)
        if rule is not None and rule not in found:
            found.append(rule)
    if not found and fragments:
        for fragment in fragments:
            rule = match_value(fragment)
            if rule is not None and rule not in found:
                found.append(rule)
    found.sort(key=lambda r: (-r.confidence, r.name))

    if found:
        if key is not None:
            for rule in found:
                if rule.canonical_hint == key:
                    return _boost(rule)
            for rule in found:
                if rule.canonical_hint is not None and rule.canonical_hint.split(".")[0] == key.split(".")[0]:
                    return rule
        return found[0]

    if key is not None:
        by_key = rules_for_key(key)
        if by_key:
            return by_key[0]
        normalized = normalize_label(label)
        if normalized:
            direct = match_placeholder(normalized)
            if direct is not None:
                return direct
    return None
