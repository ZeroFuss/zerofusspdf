"""Value normalizers: format a resolved value the way a specific field expects it.

Every function has the signature ``(value: str, spec: KeySpec) -> str``. Unknown or
malformed input degrades to a best-effort cleanup rather than raising -- normalization is
never the last line of defense; :mod:`zfp.resolver.validators` is.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Callable, Dict, Optional

_DIGITS = re.compile(r"\d+")


def _digits_only(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def upper(value: str, spec: object = None) -> str:
    return (value or "").strip().upper()


def lower(value: str, spec: object = None) -> str:
    return (value or "").strip().lower()


def title(value: str, spec: object = None) -> str:
    return (value or "").strip().title()


def strip_ws(value: str, spec: object = None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def digits_only(value: str, spec: object = None) -> str:
    return _digits_only(value)


def phone_us(value: str, spec: object = None) -> str:
    d = _digits_only(value)
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    if len(d) != 10:
        return strip_ws(value)
    return "(%s) %s-%s" % (d[0:3], d[3:6], d[6:10])


def phone_e164(value: str, spec: object = None) -> str:
    d = _digits_only(value)
    if len(d) == 10:
        d = "1" + d
    if not d:
        return ""
    return "+" + d


def ssn(value: str, spec: object = None) -> str:
    d = _digits_only(value)
    if len(d) != 9:
        return strip_ws(value)
    return "%s-%s-%s" % (d[0:3], d[3:5], d[5:9])


def ein(value: str, spec: object = None) -> str:
    d = _digits_only(value)
    if len(d) != 9:
        return strip_ws(value)
    return "%s-%s" % (d[0:2], d[2:9])


def zip5(value: str, spec: object = None) -> str:
    d = _digits_only(value)
    return d[:5].zfill(5) if d else strip_ws(value)


def zip9(value: str, spec: object = None) -> str:
    d = _digits_only(value)
    if len(d) < 9:
        return zip5(value)
    return "%s-%s" % (d[0:5], d[5:9])


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

_ORDINAL_SUFFIX = re.compile(r"(\d+)(st|nd|rd|th)\b", re.IGNORECASE)


def parse_date(value: str) -> Optional[date]:
    """Parse a date from a wide but unambiguous range of forms, US-first for slashes.

    Handles ``YYYY-MM-DD`` (ISO first, since it is unambiguous), ``M/D/YYYY`` /
    ``M/D/YY`` (US month-first, the documented convention when a bare slash form is
    given), and ``Month D, YYYY`` / ``D Month YYYY`` (with or without an ordinal suffix
    and a comma). Returns ``None`` rather than guessing when the text does not parse.
    """
    if not value:
        return None
    s = value.strip()
    s = _ORDINAL_SUFFIX.sub(r"\1", s)

    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", s)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        try:
            return date(y, mo, d)
        except ValueError:
            return None

    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if m:
        mo, d, y = (int(x) for x in m.groups())
        if y < 100:
            y += 2000 if y < 70 else 1900
        try:
            return date(y, mo, d)
        except ValueError:
            return None

    m = re.match(r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$", s)
    if m:
        name, d, y = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        mo = _MONTHS.get(name)
        if mo:
            try:
                return date(y, mo, d)
            except ValueError:
                return None

    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{4})$", s)
    if m:
        d, name, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        mo = _MONTHS.get(name)
        if mo:
            try:
                return date(y, mo, d)
            except ValueError:
                return None

    for fmt in ("%Y/%m/%d", "%m-%d-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _date_normalizer(fmt: str) -> Callable[[str, object], str]:
    def _fn(value: str, spec: object = None) -> str:
        parsed = parse_date(value)
        if parsed is None:
            return strip_ws(value)
        return parsed.strftime(fmt)
    return _fn


date_mdy = _date_normalizer("%m/%d/%Y")
date_ymd = _date_normalizer("%Y-%m-%d")
date_dmy = _date_normalizer("%d/%m/%Y")


def currency(value: str, spec: object = None) -> str:
    s = (value or "").strip()
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    s = s.lstrip("-").strip("()").lstrip("$").replace(",", "").strip()
    try:
        amount = float(s)
    except ValueError:
        return strip_ws(value)
    formatted = "{:,.2f}".format(abs(amount))
    return ("-" if neg else "") + formatted


_STATE_ABBREV = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY", "district of columbia": "DC",
    "alberta": "AB", "british columbia": "BC", "manitoba": "MB", "new brunswick": "NB",
    "newfoundland and labrador": "NL", "nova scotia": "NS", "ontario": "ON",
    "prince edward island": "PE", "quebec": "QC", "saskatchewan": "SK",
}
_VALID_ABBREVS = set(_STATE_ABBREV.values())


def state_abbrev(value: str, spec: object = None) -> str:
    s = (value or "").strip()
    up = s.upper()
    if up in _VALID_ABBREVS:
        return up
    return _STATE_ABBREV.get(s.lower(), s)


_COUNTRY_ISO2 = {
    "united states": "US", "united states of america": "US", "usa": "US", "u.s.a.": "US",
    "u.s.": "US", "canada": "CA", "mexico": "MX", "united kingdom": "GB", "uk": "GB",
    "great britain": "GB", "england": "GB", "france": "FR", "germany": "DE", "italy": "IT",
    "spain": "ES", "portugal": "PT", "netherlands": "NL", "belgium": "BE", "switzerland": "CH",
    "austria": "AT", "sweden": "SE", "norway": "NO", "denmark": "DK", "finland": "FI",
    "ireland": "IE", "poland": "PL", "russia": "RU", "china": "CN", "japan": "JP",
    "south korea": "KR", "korea": "KR", "india": "IN", "australia": "AU",
    "new zealand": "NZ", "brazil": "BR", "argentina": "AR", "chile": "CL",
    "colombia": "CO", "peru": "PE", "south africa": "ZA", "egypt": "EG", "nigeria": "NG",
    "kenya": "KE", "israel": "IL", "turkey": "TR", "saudi arabia": "SA",
    "united arab emirates": "AE", "uae": "AE", "singapore": "SG", "malaysia": "MY",
    "thailand": "TH", "vietnam": "VN", "philippines": "PH", "indonesia": "ID",
    "pakistan": "PK", "bangladesh": "BD", "greece": "GR", "czech republic": "CZ",
    "czechia": "CZ", "hungary": "HU", "romania": "RO", "ukraine": "UA", "iceland": "IS",
    "luxembourg": "LU", "taiwan": "TW", "hong kong": "HK", "scotland": "GB", "wales": "GB",
}


def country_iso2(value: str, spec: object = None) -> str:
    s = (value or "").strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return _COUNTRY_ISO2.get(s.lower(), s)


_NAME_PARTICLES = {"van", "von", "de", "der", "den", "la", "le", "di", "da", "del", "dos"}


def name_case(value: str, spec: object = None) -> str:
    s = strip_ws(value)
    if not s:
        return s
    words = s.split(" ")
    out = []
    for i, word in enumerate(words):
        lower_word = word.lower()
        if i > 0 and lower_word in _NAME_PARTICLES:
            out.append(lower_word)
            continue
        out.append(_cap_hyphen_apostrophe(word))
    return " ".join(out)


def _cap_hyphen_apostrophe(word: str) -> str:
    def cap_piece(piece: str) -> str:
        if not piece:
            return piece
        if piece.lower().startswith("mc") and len(piece) > 2:
            return "Mc" + piece[2].upper() + piece[3:].lower()
        if piece.lower().startswith("mac") and len(piece) > 3:
            return "Mac" + piece[3].upper() + piece[4:].lower()
        return piece[0].upper() + piece[1:].lower()

    parts = re.split(r"([\-'])", word)
    return "".join(cap_piece(p) if p not in ("-", "'") else p for p in parts)


def credit_card(value: str, spec: object = None) -> str:
    d = _digits_only(value)
    if not d:
        return strip_ws(value)
    groups = [d[i:i + 4] for i in range(0, len(d), 4)]
    return " ".join(groups)


def iban(value: str, spec: object = None) -> str:
    s = re.sub(r"\s+", "", (value or "")).upper()
    if not s:
        return s
    return " ".join(s[i:i + 4] for i in range(0, len(s), 4))


def boolean_yes_no(value: str, spec: object = None) -> str:
    s = (value or "").strip().lower()
    if s in ("1", "true", "yes", "y", "on", "checked"):
        return "Yes"
    if s in ("0", "false", "no", "n", "off", "unchecked", ""):
        return "No"
    return strip_ws(value)


def vin(value: str, spec: object = None) -> str:
    s = re.sub(r"[\s-]", "", (value or "")).upper()
    return s


REGISTRY: Dict[str, Callable[[str, object], str]] = {
    "upper": upper, "lower": lower, "title": title, "digits_only": digits_only,
    "phone_us": phone_us, "phone_e164": phone_e164, "ssn": ssn, "ein": ein,
    "zip5": zip5, "zip9": zip9, "date_mdy": date_mdy, "date_ymd": date_ymd,
    "date_dmy": date_dmy, "currency": currency, "state_abbrev": state_abbrev,
    "country_iso2": country_iso2, "email": lower, "strip_ws": strip_ws,
    "name_case": name_case, "credit_card": credit_card, "iban": iban,
    "boolean_yes_no": boolean_yes_no, "vin": vin,
}


def normalize(value: str, spec: object) -> str:
    """Dispatch on ``spec.normalizer`` with a safe strip-whitespace fallback."""
    name = getattr(spec, "normalizer", None)
    fn = REGISTRY.get(name) if name else None
    if fn is None:
        return strip_ws(value)
    return fn(value, spec)


__all__ = ["REGISTRY", "normalize", "parse_date"] + list(REGISTRY.keys())
