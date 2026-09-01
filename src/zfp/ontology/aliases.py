"""Label normalization and the alias -> canonical-key index.

Real forms never agree on wording.  One document says "ZIP", the next "Postal Code",
"Mail ZIP", "Zipcode" or "ZIP / Postal"; one says "Given Name", another "Forename" or
"First Name".  This module collapses all of that into a single normalized string space
and resolves it to a canonical key from :mod:`zfp.ontology.keys`.

Three layers, in increasing tolerance:

``lookup``
    Exact hit on :func:`normalize_label`, then a short list of deterministic rewrites
    (drop the leading role word, drop trailing filler, singularize the last word).
``context_lookup``
    Same, but a surrounding section context ("Billing", "Ship To") steers the answer
    toward the parent-scoped duplicate keys.
``fuzzy_lookup``
    ``difflib`` similarity over the whole alias key space, for OCR noise and typos.

:data:`ALIAS_INDEX` is built at import time from every :class:`~zfp.ontology.keys.KeySpec`
alias plus a generated expansion pass driven by the small rule tables below.  Insertion is
first-wins in key declaration order, so generic labels ("name", "address", "zip") belong to
the ``person`` namespace and specialized namespaces only claim what nobody took.
"""

from __future__ import annotations

import difflib
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .keys import CANONICAL_KEYS

__all__ = [
    "ALIAS_INDEX",
    "normalize_label",
    "lookup",
    "fuzzy_lookup",
    "context_lookup",
    "alias_count",
    "aliases_for",
    "ROLE_PREFIXES",
    "TRAILING_FILLERS",
    "PARENT_SYNONYMS",
]

# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------

#: Unicode dash-ish characters that must not survive normalization as distinct glyphs.
_DASHES = "‐‑‒–—―⁃−－­"
_DASH_TABLE = {ord(ch): "-" for ch in _DASHES}
_SPACE_TABLE = {
    0x00A0: " ", 0x2002: " ", 0x2003: " ", 0x2009: " ", 0x200A: " ",
    0x202F: " ", 0x205F: " ", 0x3000: " ", 0x200B: "",
}

_POSSESSIVE_RE = re.compile(r"['\u2019]s(?![a-z0-9])")
_PLURAL_POSSESSIVE_RE = re.compile(r"s['\u2019](?![a-z0-9])")
_PAREN_RE = re.compile(r"[\(\[\{][^\)\]\}]*[\)\]\}]")
_ENUM_RE = re.compile(r"^\s*(?:\d{1,3}|[a-z]|[ivxlcdm]{1,4})\s*[.)\]]\s+", re.IGNORECASE)
_BULLET_RE = re.compile(r"^[\s\-•●▪·*>#]+")
_TRAILING_MARK_RE = re.compile(r"[\s:;*.,…]+$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")


def normalize_label(s: str) -> str:
    """Fold a printed form label into the canonical alias string space.

    The transform is: lowercase; unify unicode dashes and spaces; delete parenthesised
    hints such as ``(required)`` or ``(if any)``; drop a leading enumeration such as
    ``1.`` or ``a)`` and any bullet glyph; strip trailing ``:``/``*``/ellipsis; map ``&``
    to ``and``; replace every remaining non-alphanumeric run with a single space; collapse
    whitespace and strip.

    ``"  2) E-Mail Address (required):  "`` becomes ``"e mail address"``.
    """
    if not s:
        return ""
    text = s.translate(_SPACE_TABLE).translate(_DASH_TABLE).lower()
    text = _PLURAL_POSSESSIVE_RE.sub("s", _POSSESSIVE_RE.sub("", text))
    text = _PAREN_RE.sub(" ", text)
    text = _BULLET_RE.sub("", text)
    text = _ENUM_RE.sub("", text)
    text = _TRAILING_MARK_RE.sub("", text)
    text = text.replace("&", " and ")
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------------------
# Rewrite vocabulary used by lookup()
# --------------------------------------------------------------------------------------

#: Leading qualifiers that identify *whose* value a field holds rather than *what* it is.
ROLE_PREFIXES: Tuple[str, ...] = (
    "please print", "please type", "print", "applicant", "co applicant", "employee",
    "borrower", "co borrower", "patient", "customer", "client", "member", "student",
    "tenant", "buyer", "seller", "owner", "guarantor", "individual", "primary",
    "your", "my", "the", "enter", "provide",
)

#: Trailing words that decorate a label without changing its meaning.
TRAILING_FILLERS: Tuple[str, ...] = (
    "here", "below", "above", "if any", "if applicable", "optional", "required",
    "please", "print", "only", "in full", "as applicable", "field", "info",
)

#: Section words that select a parent-scoped duplicate key.
PARENT_SYNONYMS: Dict[str, str] = {
    "billing": "billing", "bill": "billing", "billed": "billing", "invoice": "billing",
    "payment": "billing", "shipping": "shipping", "ship": "shipping",
    "shipped": "shipping", "delivery": "shipping", "deliver": "shipping",
    "mailing": "mailing", "mail": "mailing", "correspondence": "mailing",
    "postal": "mailing",
}

#: Phrase rewrites used to grow the index.  Each pair is applied on word boundaries to
#: every base alias, one substitution at a time, and the result is inserted only when the
#: normalized string is still unclaimed.
_PHRASE_VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("number", "no"), ("number", "num"), ("number", "nbr"), ("no", "number"),
    ("num", "number"), ("street", "st"), ("st", "street"), ("avenue", "ave"),
    ("apartment", "apt"), ("apt", "apartment"), ("telephone", "phone"),
    ("telephone", "tel"), ("phone", "telephone"), ("phone", "tel"),
    ("mobile", "cell"), ("cell", "mobile"), ("email", "e mail"), ("e mail", "email"),
    ("identification", "id"), ("id", "identification"), ("address", "addr"),
    ("addr", "address"), ("amount", "amt"), ("amt", "amount"), ("quantity", "qty"),
    ("department", "dept"), ("dept", "department"), ("company", "business"),
    ("company", "organization"), ("business", "company"), ("organization", "company"),
    ("date of birth", "dob"), ("date of birth", "birth date"), ("dob", "date of birth"),
    ("social security number", "ssn"), ("ssn", "social security number"),
    ("zip", "postal"), ("postal", "zip"), ("zip code", "zipcode"),
    ("signature", "sign"), ("first name", "given name"), ("last name", "surname"),
    ("state", "province"), ("city", "town"), ("employer identification number", "ein"),
    ("federal tax id", "fed id"), ("routing number", "aba"), ("account", "acct"),
    ("acct", "account"), ("expiration", "exp"), ("exp", "expiration"),
    ("percent", "pct"), ("year", "yr"), ("month", "mo"),
)

#: Keys that get the full role-prefix expansion ("applicant first name", "borrower ssn").
_PREFIXED_KEYS: Tuple[str, ...] = (
    "person.name.full", "person.name.first", "person.name.middle",
    "person.name.middle_initial", "person.name.last", "person.name.suffix",
    "person.name.maiden", "person.address.street_1", "person.address.street_2",
    "person.address.unit", "person.address.city", "person.address.county",
    "person.address.region", "person.address.postal_code", "person.address.country",
    "person.address.full", "person.phone.mobile", "person.phone.home",
    "person.phone.work", "person.phone.fax", "person.email", "person.date_of_birth",
    "person.ssn", "person.signature", "person.initials", "person.job_title",
    "person.age", "person.employer", "person.occupation", "document.today",
    "document.signed_date", "employment.employee_id",
)

#: Prefixes applied to :data:`_PREFIXED_KEYS`.
_GENERATED_PREFIXES: Tuple[str, ...] = (
    "applicant", "co applicant", "employee", "borrower", "patient", "customer",
    "member", "student", "tenant", "buyer", "seller", "owner", "guarantor",
    "your", "primary", "please print",
)

#: Curated additions that no rule would produce but real forms print constantly.
_EXTRA_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("postal", "person.address.postal_code"),
    ("zip / postal", "person.address.postal_code"),
    ("zip/postal code", "person.address.postal_code"),
    ("postal / zip", "person.address.postal_code"),
    ("mailing zip", "person.address.postal_code"),
    ("home zip", "person.address.postal_code"),
    ("state / province", "person.address.region"),
    ("state/prov", "person.address.region"),
    ("country / region", "person.address.country"),
    ("street address (no p.o. box)", "person.address.street_1"),
    ("no. and street", "person.address.street_1"),
    ("e-mail", "person.email"),
    ("email addr", "person.email"),
    ("tel.", "person.phone.mobile"),
    ("tel no", "person.phone.mobile"),
    ("ph", "person.phone.mobile"),
    ("ph no", "person.phone.mobile"),
    ("phone (day)", "person.phone.work"),
    ("mobile #", "person.phone.mobile"),
    ("d.o.b", "person.date_of_birth"),
    ("date of birth (mm/dd/yyyy)", "person.date_of_birth"),
    ("soc. sec. #", "person.ssn"),
    ("ssn/tin", "person.ssn"),
    ("last 4 of ssn", "person.ssn"),
    ("sign here", "person.signature"),
    ("signature of applicant", "person.signature"),
    ("x (sign here)", "person.signature"),
    ("date signed", "document.signed_date"),
    ("today's date", "document.today"),
    ("date (mm/dd/yyyy)", "document.today"),
    ("title / position", "person.job_title"),
    ("position / title", "person.job_title"),
    ("company / organization", "company.legal_name"),
    ("business/company name", "company.legal_name"),
    ("fed. id #", "company.tax_id.ein"),
    ("federal employer identification number", "company.tax_id.ein"),
    ("ein / tax id", "company.tax_id.ein"),
    ("acct #", "bank.account_number"),
    ("acct. no.", "bank.account_number"),
    ("routing / aba", "bank.routing_number"),
    ("aba / routing number", "bank.routing_number"),
    ("bank acct #", "bank.account_number"),
    ("amount due", "document.amount"),
    ("amount ($)", "document.amount"),
    ("total ($)", "document.total"),
    ("po #", "document.purchase_order"),
    ("inv #", "document.invoice_number"),
    ("ref. #", "document.reference_number"),
    ("case #", "document.case_number"),
    ("policy #", "document.policy_number"),
    ("cc #", "card.number"),
    ("card #", "card.number"),
    ("exp. date", "card.expiration"),
    ("mm/yy", "card.expiration"),
    ("cvv/cvc", "card.cvv"),
    ("emp. id", "employment.employee_id"),
    ("dl #", "person.driver_license.number"),
    ("driver's license", "person.driver_license.number"),
    ("driver's license no.", "person.driver_license.number"),
    ("driver's license state", "person.driver_license.state"),
    ("operator's license", "person.driver_license.number"),
    ("vin #", "vehicle.vin"),
    ("npi #", "medical.npi"),
    ("group #", "insurance.group_number"),
    ("member #", "insurance.member_id"),
    ("i have read and agree", "consent.terms_accepted"),
    ("check here if you agree", "consent.agree"),
    ("please explain", "document.notes"),
    ("if other, please specify", "misc.other"),
)


# --------------------------------------------------------------------------------------
# Index construction
# --------------------------------------------------------------------------------------
_WORD_CACHE: Dict[str, re.Pattern] = {}


def _word_re(phrase: str) -> re.Pattern:
    rx = _WORD_CACHE.get(phrase)
    if rx is None:
        rx = re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])")
        _WORD_CACHE[phrase] = rx
    return rx


def _build_index() -> Dict[str, str]:
    index: Dict[str, str] = {}

    def add(raw: str, key: str) -> None:
        norm = normalize_label(raw)
        if norm and norm not in index:
            index[norm] = key

    # Pass 1 -- declared aliases, then the human label, in key declaration order.
    for spec in CANONICAL_KEYS.values():
        for alias in spec.aliases:
            add(alias, spec.key)
    for spec in CANONICAL_KEYS.values():
        add(spec.label, spec.key)

    # Pass 2 -- curated additions the rules cannot reach.
    for raw, key in _EXTRA_ALIASES:
        add(raw, key)

    # Pass 3 -- role prefixes over the person-shaped keys.
    for key in _PREFIXED_KEYS:
        spec = CANONICAL_KEYS[key]
        stems: List[str] = []
        for alias in spec.aliases[:2]:
            stems.append(normalize_label(alias))
        stems.append(normalize_label(spec.label))
        for prefix in _GENERATED_PREFIXES:
            for stem in stems:
                if stem and not stem.startswith(prefix + " "):
                    add(prefix + " " + stem, key)

    # Pass 4 -- parent-context prefixes for the scoped duplicate keys.
    inverse_parents: Dict[str, List[str]] = {}
    for word, canon in PARENT_SYNONYMS.items():
        inverse_parents.setdefault(canon, []).append(word)
    for spec in CANONICAL_KEYS.values():
        for parent in spec.parents:
            for word in sorted(inverse_parents.get(parent, [])):
                tail = spec.scoped_suffix().split(".", 1)[-1].replace("_", " ")
                add(word + " " + tail, spec.key)
                add(word + " to " + tail, spec.key)

    # Pass 5 -- phrase rewrites over everything gathered so far.
    snapshot = list(index.items())
    for norm, key in snapshot:
        for source, target in _PHRASE_VARIANTS:
            rx = _word_re(source)
            if rx.search(norm):
                add(rx.sub(target, norm), key)

    return index


#: Normalized alias -> canonical key.  Built once at import time.
ALIAS_INDEX: Dict[str, str] = _build_index()

#: Deterministically ordered alias keys, used by :func:`fuzzy_lookup`.
_ALIAS_KEYS: List[str] = sorted(ALIAS_INDEX)


def _build_context_remap() -> Dict[Tuple[str, str], str]:
    """Map ``(parent, base_key) -> scoped_key`` by matching namespace-stripped suffixes."""
    by_suffix: Dict[str, List[str]] = {}
    for spec in CANONICAL_KEYS.values():
        if not spec.parents:
            by_suffix.setdefault(spec.scoped_suffix(), []).append(spec.key)
    remap: Dict[Tuple[str, str], str] = {}
    for spec in CANONICAL_KEYS.values():
        if not spec.parents:
            continue
        for base in by_suffix.get(spec.scoped_suffix(), ()):
            for parent in spec.parents:
                remap.setdefault((parent, base), spec.key)
    return remap


_CONTEXT_REMAP: Dict[Tuple[str, str], str] = _build_context_remap()


def alias_count() -> int:
    """Return the number of normalized aliases in :data:`ALIAS_INDEX`."""
    return len(ALIAS_INDEX)


def aliases_for(key: str) -> List[str]:
    """Return every normalized alias that resolves to ``key``, sorted."""
    return sorted(alias for alias, target in ALIAS_INDEX.items() if target == key)


# --------------------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------------------

def _strip_leading_roles(norm: str) -> str:
    changed = True
    text = norm
    while changed and text:
        changed = False
        for prefix in ROLE_PREFIXES:
            for form in (prefix, prefix + "s"):
                if text.startswith(form + " ") and len(text) > len(form) + 1:
                    text = text[len(form) + 1 :]
                    changed = True
                    break
            if changed:
                break
    return text


def _strip_trailing_fillers(norm: str) -> str:
    changed = True
    text = norm
    while changed and text:
        changed = False
        for filler in TRAILING_FILLERS:
            if text.endswith(" " + filler) and len(text) > len(filler) + 1:
                text = text[: -(len(filler) + 1)]
                changed = True
                break
    return text


def _singularize(norm: str) -> str:
    if not norm:
        return norm
    head, _, last = norm.rpartition(" ")
    word = last or norm
    if len(word) > 3 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    else:
        return norm
    return (head + " " + word).strip() if head else word


def _rewrites(norm: str) -> List[str]:
    """Deterministic rewrite ladder tried after an exact miss."""
    lead = _strip_leading_roles(norm)
    tail = _strip_trailing_fillers(norm)
    both = _strip_trailing_fillers(lead)
    forms: List[str] = []
    for candidate in (tail, lead, both):
        for variant in (candidate, _singularize(candidate)):
            if variant and variant != norm and variant not in forms:
                forms.append(variant)
    singular = _singularize(norm)
    if singular != norm and singular not in forms:
        forms.insert(0, singular)
    return forms


def lookup(label: str) -> Optional[str]:
    """Resolve a printed label to a canonical key, or ``None``.

    Tries the exact normalized label first, then a short deterministic rewrite ladder:
    singularize the trailing word, drop a leading role qualifier
    ("Applicant", "Borrower", "Your", "Please print"), drop trailing filler
    ("here", "below", "optional"), and the combination of the two.
    """
    norm = normalize_label(label)
    if not norm:
        return None
    hit = ALIAS_INDEX.get(norm)
    if hit is not None:
        return hit
    for variant in _rewrites(norm):
        hit = ALIAS_INDEX.get(variant)
        if hit is not None:
            return hit
    return None


def fuzzy_lookup(label: str, cutoff: float = 0.82) -> List[Tuple[str, float]]:
    """Return ``(canonical_key, ratio)`` pairs for near-miss labels.

    Uses :func:`difflib.get_close_matches` over the whole normalized alias space, which
    makes it tolerant of OCR damage ("Frst Narne") and of spellings nobody anticipated.
    Results are deduplicated per canonical key keeping the best ratio, and sorted by
    ``(-ratio, key)`` so the output is stable across runs.
    """
    norm = normalize_label(label)
    if not norm:
        return []
    candidates = difflib.get_close_matches(norm, _ALIAS_KEYS, n=12, cutoff=cutoff)
    best: Dict[str, float] = {}
    for alias in candidates:
        ratio = difflib.SequenceMatcher(None, norm, alias).ratio()
        key = ALIAS_INDEX[alias]
        if ratio > best.get(key, -1.0):
            best[key] = ratio
    return sorted(best.items(), key=lambda item: (-item[1], item[0]))


def _parent_tokens(parent_context: Sequence[str] | str | None) -> List[str]:
    """Normalize a section context into canonical parent tokens, order preserved."""
    if not parent_context:
        return []
    items: Iterable[str]
    items = [parent_context] if isinstance(parent_context, str) else parent_context
    out: List[str] = []
    for item in items:
        for token in normalize_label(str(item)).split():
            canon = PARENT_SYNONYMS.get(token)
            if canon is not None and canon not in out:
                out.append(canon)
    return out


def context_lookup(
    label: str, parent_context: Sequence[str] | str = ()
) -> Optional[str]:
    """Resolve a label while honouring the surrounding section context.

    A bare "Address" under a "Billing Information" heading is
    ``billing.address.street_1``, while the same label under "Ship To" is
    ``shipping.address.street_1``.  Resolution order is: parent-qualified alias
    ("billing address"), then the plain :func:`lookup` result remapped onto its
    parent-scoped twin, then the plain result unchanged.
    """
    norm = normalize_label(label)
    if not norm:
        return None
    parents = _parent_tokens(parent_context)

    for parent in parents:
        for word, canon in sorted(PARENT_SYNONYMS.items()):
            if canon != parent:
                continue
            for joiner in (" ", " to "):
                hit = ALIAS_INDEX.get(normalize_label(word + joiner + norm))
                if hit is not None and parent in CANONICAL_KEYS[hit].parents:
                    return hit

    base = lookup(label)
    if base is None:
        return None
    for parent in parents:
        scoped = _CONTEXT_REMAP.get((parent, base))
        if scoped is not None:
            return scoped
    return base
