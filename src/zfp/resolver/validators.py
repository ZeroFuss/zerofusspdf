"""Value validators: does a resolved, normalized value actually satisfy this field?

Every function returns a :class:`ValidationOutcome` rather than raising -- validation
failure is data the caller acts on (mark the value ``invalid``, keep the reason), not an
exception to unwind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Callable, Dict, Optional

from .normalizers import parse_date

_DIGITS = re.compile(r"\D")


@dataclass
class ValidationOutcome:
    ok: bool
    message: str = ""
    normalized: Optional[str] = None


def _d(value: str) -> str:
    return _DIGITS.sub("", value or "")


def luhn(value: str, spec: object = None) -> ValidationOutcome:
    d = _d(value)
    if not d.isdigit() or len(d) < 12:
        return ValidationOutcome(False, "not enough digits for a card number")
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return ValidationOutcome(total % 10 == 0, "" if total % 10 == 0 else "fails Luhn check")


_SSN_INVALID_AREA = {"000", "666"}


def ssn(value: str, spec: object = None) -> ValidationOutcome:
    d = _d(value)
    if len(d) != 9:
        return ValidationOutcome(False, "SSN must have 9 digits")
    area, group, serial = d[0:3], d[3:5], d[5:9]
    if area in _SSN_INVALID_AREA or area.startswith("9") or group == "00" or serial == "0000":
        return ValidationOutcome(False, "not a valid SSN area/group/serial")
    return ValidationOutcome(True, normalized="%s-%s-%s" % (area, group, serial))


_EIN_VALID_PREFIXES = {
    "01", "02", "03", "04", "05", "06", "10", "11", "12", "13", "14", "15", "16",
    "20", "21", "22", "23", "24", "25", "26", "27", "30", "31", "32", "33", "34",
    "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47",
    "48", "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61",
    "62", "63", "64", "65", "66", "67", "68", "71", "72", "73", "74", "75", "76",
    "77", "80", "81", "82", "83", "84", "85", "86", "87", "88", "90", "91", "92",
    "93", "94", "95", "98", "99",
}


def ein(value: str, spec: object = None) -> ValidationOutcome:
    d = _d(value)
    if len(d) != 9:
        return ValidationOutcome(False, "EIN must have 9 digits")
    if d[0:2] not in _EIN_VALID_PREFIXES:
        return ValidationOutcome(False, "not a recognized EIN prefix")
    return ValidationOutcome(True, normalized="%s-%s" % (d[0:2], d[2:9]))


_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$")


def email(value: str, spec: object = None) -> ValidationOutcome:
    s = (value or "").strip()
    if not _EMAIL_RE.match(s):
        return ValidationOutcome(False, "not a valid email address")
    return ValidationOutcome(True, normalized=s.lower())


def phone_us(value: str, spec: object = None) -> ValidationOutcome:
    d = _d(value)
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    if len(d) != 10:
        return ValidationOutcome(False, "US phone numbers have 10 digits")
    if d[0] in "01":
        return ValidationOutcome(False, "area code cannot start with 0 or 1")
    return ValidationOutcome(True, normalized="(%s) %s-%s" % (d[0:3], d[3:6], d[6:10]))


def zip_us(value: str, spec: object = None) -> ValidationOutcome:
    d = _d(value)
    if len(d) not in (5, 9):
        return ValidationOutcome(False, "US ZIP codes have 5 or 9 digits")
    return ValidationOutcome(True)


def date_(value: str, spec: object = None, *, not_in_future: bool = False,
         not_in_past: bool = False) -> ValidationOutcome:
    parsed = parse_date(value) if not isinstance(value, date) else value
    if parsed is None:
        return ValidationOutcome(False, "not a recognizable calendar date")
    today = date.today() if hasattr(date, "today") else None
    if not_in_future and today is not None and parsed > today:
        return ValidationOutcome(False, "date is in the future")
    if not_in_past and today is not None and parsed < today:
        return ValidationOutcome(False, "date is in the past")
    return ValidationOutcome(True, normalized=parsed.isoformat())


_IBAN_COUNTRY_LEN = {
    "GB": 22, "DE": 22, "FR": 27, "ES": 24, "IT": 27, "NL": 18, "BE": 16, "CH": 21,
    "AT": 20, "IE": 22, "PT": 25, "LU": 20, "SE": 24, "NO": 15, "DK": 18, "FI": 18,
    "PL": 28,
}


def iban(value: str, spec: object = None) -> ValidationOutcome:
    s = re.sub(r"\s+", "", (value or "")).upper()
    if len(s) < 4 or not s[:2].isalpha() or not s[2:4].isdigit():
        return ValidationOutcome(False, "not a valid IBAN shape")
    expected_len = _IBAN_COUNTRY_LEN.get(s[:2])
    if expected_len is not None and len(s) != expected_len:
        return ValidationOutcome(False, "wrong length for %s IBAN" % s[:2])
    rearranged = s[4:] + s[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    if int(numeric) % 97 != 1:
        return ValidationOutcome(False, "fails IBAN mod-97 checksum")
    return ValidationOutcome(True, normalized=s)


def routing_aba(value: str, spec: object = None) -> ValidationOutcome:
    d = _d(value)
    if len(d) != 9:
        return ValidationOutcome(False, "ABA routing numbers have 9 digits")
    weights = (3, 7, 1, 3, 7, 1, 3, 7, 1)
    total = sum(int(c) * w for c, w in zip(d, weights))
    if total % 10 != 0:
        return ValidationOutcome(False, "fails ABA routing checksum")
    return ValidationOutcome(True, normalized=d)


_VIN_TRANSLIT = {**{c: v for c, v in zip("ABCDEFGH", range(1, 9))},
                 **{c: v for c, v in zip("JKLMNPRSTUVWXYZ",
                                         [1, 2, 3, 4, 5, 7, 9, 2, 3, 4, 5, 6, 7, 8, 9])}}
_VIN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def vin(value: str, spec: object = None) -> ValidationOutcome:
    s = re.sub(r"[\s-]", "", (value or "")).upper()
    if len(s) != 17 or any(c in "IOQ" for c in s):
        return ValidationOutcome(False, "VIN must be 17 characters, no I/O/Q")
    total = 0
    for ch, w in zip(s, _VIN_WEIGHTS):
        v = int(ch) if ch.isdigit() else _VIN_TRANSLIT.get(ch)
        if v is None:
            return ValidationOutcome(False, "invalid VIN character %r" % ch)
        total += v * w
    remainder = total % 11
    check = "X" if remainder == 10 else str(remainder)
    if s[8] != check:
        return ValidationOutcome(False, "fails VIN check digit")
    return ValidationOutcome(True, normalized=s)


def nonempty(value: str, spec: object = None) -> ValidationOutcome:
    ok = bool((value or "").strip())
    return ValidationOutcome(ok, "" if ok else "value is empty")


def max_length(value: str, spec: object = None) -> ValidationOutcome:
    limit = getattr(spec, "max_length", None)
    if limit is None:
        return ValidationOutcome(True)
    ok = len(value or "") <= int(limit)
    return ValidationOutcome(ok, "" if ok else "exceeds max length %d" % limit)


def regex(value: str, spec: object = None) -> ValidationOutcome:
    pattern = getattr(spec, "pattern", None)
    if not pattern:
        return ValidationOutcome(True)
    ok = re.fullmatch(pattern, value or "") is not None
    return ValidationOutcome(ok, "" if ok else "does not match required pattern")


def choice(value: str, spec: object = None) -> ValidationOutcome:
    choices = getattr(spec, "choices", None)
    if not choices:
        return ValidationOutcome(True)
    ok = value in choices
    return ValidationOutcome(ok, "" if ok else "not one of the allowed choices")


REGISTRY: Dict[str, Callable[..., ValidationOutcome]] = {
    "luhn": luhn, "ssn": ssn, "ein": ein, "email": email, "phone_us": phone_us,
    "zip_us": zip_us, "date": date_, "iban": iban, "routing_aba": routing_aba,
    "vin": vin, "nonempty": nonempty, "max_length": max_length, "regex": regex,
    "choice": choice,
}


def validate(value: str, spec: object) -> ValidationOutcome:
    """Dispatch on ``spec.validator``; unset means "nothing to check"."""
    name = getattr(spec, "validator", None)
    fn = REGISTRY.get(name) if name else None
    if fn is None:
        return ValidationOutcome(True)
    return fn(value, spec)


def validate_against_constraints(value: str, constraints: object) -> ValidationOutcome:
    """Check a value against a :class:`~zfp.core.types.FieldConstraints`."""
    max_chars = getattr(constraints, "max_chars_estimate", None)
    if max_chars is not None and len(value or "") > int(max_chars) * 1.15:
        return ValidationOutcome(False, "value likely too long for the field's rectangle")
    comb_cells = getattr(constraints, "comb_cells", None)
    if comb_cells is not None and len(value or "") > int(comb_cells):
        return ValidationOutcome(False, "value has more characters than comb cells")
    pattern = getattr(constraints, "pattern", None)
    if pattern and re.fullmatch(pattern, value or "") is None:
        return ValidationOutcome(False, "does not match the field's expected pattern")
    choices = getattr(constraints, "choices", None)
    if choices and value not in choices:
        return ValidationOutcome(False, "not one of the field's declared choices")
    return ValidationOutcome(True)


__all__ = ["ValidationOutcome", "REGISTRY", "validate", "validate_against_constraints"] + \
    list(REGISTRY.keys())
