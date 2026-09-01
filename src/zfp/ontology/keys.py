"""The canonical form-key namespace.

This module is the semantic backbone of ZFP.  Every field ZFP detects is eventually
mapped onto exactly one *canonical key* -- a dotted, lowercase path such as
``person.address.postal_code``.  Once a document field carries a canonical key it no
longer matters whether the printed label said "ZIP", "Postal Code", "Mail ZIP",
"Zipcode" or "ZIP / Postal": the resolver, the vault, the validators and the QA metrics
all speak the canonical namespace.

A :class:`KeySpec` is the complete declaration of one key.  It carries the expected
:class:`~zfp.core.types.FieldType`, a human label, the literal aliases seen on real
forms, an optional value regex, a format hint, a sensitivity band, a length budget,
the name of the normalizer / validator that should process the value, and the *parent
contexts* which discriminate otherwise identical keys (``billing.address.city`` vs
``shipping.address.city``).

The module holds only data plus three accessors; alias expansion lives in
:mod:`zfp.ontology.aliases` and placeholder inference in :mod:`zfp.ontology.patterns`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.errors import SemanticError
from ..core.types import FieldType

__all__ = [
    "KeySpec",
    "CANONICAL_KEYS",
    "SENSITIVITIES",
    "get",
    "all_keys",
    "children",
    "namespaces",
    "keys_with_parent",
    "key_count",
]

SENSITIVITIES: Tuple[str, ...] = ("normal", "pii", "secret")


@dataclass(frozen=True)
class KeySpec:
    """Declaration of a single canonical key.

    Attributes:
        key: Dotted lowercase canonical path, e.g. ``person.name.first``.
        field_type: The PDF-level field type this key normally materializes as.
        label: Human readable label used in reports and tooltips.
        aliases: Literal label spellings observed on real forms.  These are normalized
            and folded into :data:`zfp.ontology.aliases.ALIAS_INDEX`.
        pattern: Optional anchored regex a *value* for this key should satisfy.
        format_hint: Human/format-level hint, e.g. ``"MM/DD/YYYY"`` or ``"NN-NNNNNNN"``.
        sensitivity: ``"normal"``, ``"pii"`` or ``"secret"``.  ``"secret"`` values are
            never placed into model prompts and never logged.
        max_length: Soft maximum character budget used for comb sizing and validation.
        normalizer: Name of a callable in ``zfp.resolver.normalizers.REGISTRY``.
        validator: Name of a callable in ``zfp.resolver.validators.REGISTRY``.
        parents: Context discriminators.  A key with ``parents=("billing",)`` is only
            preferred when the surrounding section context mentions billing.
    """

    key: str
    field_type: FieldType
    label: str
    aliases: Tuple[str, ...] = ()
    pattern: Optional[str] = None
    format_hint: Optional[str] = None
    sensitivity: str = "normal"
    max_length: Optional[int] = None
    normalizer: Optional[str] = None
    validator: Optional[str] = None
    parents: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.sensitivity not in SENSITIVITIES:
            raise SemanticError(
                f"KeySpec {self.key!r} has invalid sensitivity {self.sensitivity!r}"
            )

    @property
    def namespace(self) -> str:
        """The first path component, e.g. ``"person"``."""
        return self.key.split(".", 1)[0]

    @property
    def leaf(self) -> str:
        """The final path component, e.g. ``"postal_code"``."""
        return self.key.rsplit(".", 1)[-1]

    @property
    def path(self) -> Tuple[str, ...]:
        """The dotted key split into components."""
        return tuple(self.key.split("."))

    @property
    def is_sensitive(self) -> bool:
        """True when the value must never appear in an outbound prompt verbatim."""
        return self.sensitivity in ("pii", "secret")

    def scoped_suffix(self) -> str:
        """The key with its leading namespace component removed.

        ``billing.address.city`` and ``person.address.city`` share the suffix
        ``address.city``; this is how a parent-scoped key is matched to the base key it
        specializes.
        """
        parts = self.key.split(".", 1)
        return parts[1] if len(parts) == 2 else parts[0]

    def as_dict(self) -> Dict[str, Any]:
        """Return a plain JSON-serializable mapping of this spec."""
        return {
            "key": self.key,
            "field_type": self.field_type.value,
            "label": self.label,
            "aliases": list(self.aliases),
            "pattern": self.pattern,
            "format_hint": self.format_hint,
            "sensitivity": self.sensitivity,
            "max_length": self.max_length,
            "normalizer": self.normalizer,
            "validator": self.validator,
            "parents": list(self.parents),
        }


# --------------------------------------------------------------------------------------
# Table shorthands.  The declaration table below is very wide; single-letter aliases for
# the field types and shared regexes keep one key on one readable line.
# --------------------------------------------------------------------------------------
_T = FieldType.TEXT
_M = FieldType.MULTILINE_TEXT
_D = FieldType.DATE
_N = FieldType.NUMBER
_C = FieldType.CURRENCY
_E = FieldType.EMAIL
_P = FieldType.PHONE
_B = FieldType.CHECKBOX
_H = FieldType.CHOICE
_G = FieldType.SIGNATURE
_K = FieldType.COMB

RE_SSN = r"^\d{3}-?\d{2}-?\d{4}$"
RE_ITIN = r"^9\d{2}-?\d{2}-?\d{4}$"
RE_EIN = r"^\d{2}-?\d{7}$"
RE_ZIP = r"^\d{5}(?:-\d{4})?$"
RE_ZIP4 = r"^\d{4}$"
RE_EMAIL = r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"
RE_PHONE = r"^\+?1?[\s.\-]?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}$"
RE_EXT = r"^\d{1,6}$"
RE_STATE = r"^[A-Za-z]{2}$"
RE_DATE = r"^\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}$"
RE_CARD = r"^\d{4}[ \-]?\d{4}[ \-]?\d{4}[ \-]?\d{1,7}$"
RE_ROUTING = r"^\d{9}$"
RE_ACCOUNT = r"^[0-9\-]{4,20}$"
RE_SWIFT = r"^[A-Za-z]{4}[A-Za-z]{2}[A-Za-z0-9]{2}(?:[A-Za-z0-9]{3})?$"
RE_IBAN = r"^[A-Za-z]{2}\d{2}[A-Za-z0-9]{10,30}$"
RE_VIN = r"^[A-HJ-NPR-Za-hj-npr-z0-9]{17}$"
RE_MONEY = r"^-?\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?$"
RE_YEAR = r"^(?:18|19|20)\d{2}$"
RE_MONTH_NUM = r"^(?:0?[1-9]|1[0-2])$"
RE_NPI = r"^\d{10}$"
RE_CVV = r"^\d{3,4}$"
RE_MMYY = r"^(?:0[1-9]|1[0-2])\s*[/\-]\s*\d{2,4}$"
RE_URL = r"^(?:https?://)?(?:[\w\-]+\.)+[A-Za-z]{2,}(?:/\S*)?$"
RE_PERCENT = r"^\d{1,3}(?:\.\d+)?\s?%?$"
RE_NAICS = r"^\d{2,6}$"
RE_DUNS = r"^\d{9}$"
RE_GPA = r"^[0-4](?:\.\d{1,2})?$"
RE_YESNO = r"^(?:y|n|yes|no|true|false|x|1|0)$"


def _spec(
    key: str,
    field_type: FieldType,
    label: str,
    *aliases: str,
    pattern: Optional[str] = None,
    fmt: Optional[str] = None,
    sens: str = "normal",
    maxlen: Optional[int] = None,
    norm: Optional[str] = None,
    valid: Optional[str] = None,
    parents: Tuple[str, ...] = (),
) -> KeySpec:
    """Compact constructor used by the declaration table below."""
    return KeySpec(
        key=key,
        field_type=field_type,
        label=label,
        aliases=tuple(aliases),
        pattern=pattern,
        format_hint=fmt,
        sensitivity=sens,
        max_length=maxlen,
        normalizer=norm,
        validator=valid,
        parents=tuple(parents),
    )


def _address_block(
    prefix: str,
    human: str,
    *,
    parents: Tuple[str, ...] = (),
    sens: str = "pii",
    extras: bool = False,
) -> List[KeySpec]:
    """Build the six-to-eight key postal address block shared by every namespace.

    ``prefix`` is the dotted parent (``"billing.address"``), ``human`` the label stem
    ("Billing").  ``extras`` adds the recipient name/phone lines that billing and
    shipping blocks carry on commerce forms.
    """
    low = human.lower()
    out = [
        _spec(
            prefix + ".street_1", _T, human + " Street Address",
            low + " address", low + " street address", low + " address line 1",
            low + " street", low + " address 1",
            sens=sens, maxlen=80, norm="strip_ws", valid="nonempty", parents=parents,
        ),
        _spec(
            prefix + ".street_2", _T, human + " Address Line 2",
            low + " address line 2", low + " address 2", low + " apt", low + " suite",
            sens=sens, maxlen=80, norm="strip_ws", parents=parents,
        ),
        _spec(
            prefix + ".city", _T, human + " City",
            low + " city", low + " town", low + " city or town",
            sens=sens, maxlen=50, norm="name_case", valid="nonempty", parents=parents,
        ),
        _spec(
            prefix + ".region", _T, human + " State",
            low + " state", low + " province", low + " state or province", low + " region",
            pattern=RE_STATE, fmt="XX", sens=sens, maxlen=2,
            norm="state_abbrev", parents=parents,
        ),
        _spec(
            prefix + ".postal_code", _T, human + " ZIP Code",
            low + " zip", low + " zip code", low + " postal code", low + " zipcode",
            low + " post code",
            pattern=RE_ZIP, fmt="NNNNN", sens=sens, maxlen=10,
            norm="zip5", valid="zip_us", parents=parents,
        ),
        _spec(
            prefix + ".country", _T, human + " Country",
            low + " country", low + " nation", low + " country or region",
            sens=sens, maxlen=56, norm="country_iso2", parents=parents,
        ),
    ]
    if extras:
        out.append(
            _spec(
                prefix + ".name", _T, human + " Name",
                low + " name", low + " contact name", low + " recipient",
                low + " attention", low + " attn",
                sens="pii", maxlen=80, norm="name_case", valid="nonempty", parents=parents,
            )
        )
        out.append(
            _spec(
                prefix + ".phone", _P, human + " Phone",
                low + " phone", low + " phone number", low + " telephone",
                pattern=RE_PHONE, fmt="(NNN) NNN-NNNN", sens="pii", maxlen=20,
                norm="phone_us", valid="phone_us", parents=parents,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Declaration table.
#
# Order is significant: earlier specs claim generic aliases ("name", "address", "zip")
# ahead of later, more specialized ones.  ``zfp.ontology.aliases`` folds the table in
# declaration order using first-wins semantics.
# --------------------------------------------------------------------------------------
_SPECS: List[KeySpec] = []

# ---- person.name ---------------------------------------------------------------------
_SPECS += [
    _spec("person.name.full", _T, "Full Name",
          "name", "full name", "your name", "legal name", "full legal name",
          "name of applicant", "applicant name", "print name", "printed name",
          "name (first, middle, last)", "complete name", "individual name",
          sens="pii", maxlen=80, norm="name_case", valid="nonempty"),
    _spec("person.name.first", _T, "First Name",
          "first name", "given name", "forename", "first", "fname", "christian name",
          "first (given) name", "name first", "given names",
          sens="pii", maxlen=40, norm="name_case", valid="nonempty"),
    _spec("person.name.middle", _T, "Middle Name",
          "middle name", "middle", "mname", "second name", "middle names",
          sens="pii", maxlen=40, norm="name_case"),
    _spec("person.name.middle_initial", _T, "Middle Initial",
          "middle initial", "mi", "m i", "middle init", "initial",
          fmt="X", sens="pii", maxlen=1, norm="upper"),
    _spec("person.name.last", _T, "Last Name",
          "last name", "surname", "family name", "last", "lname", "sur name",
          "last (family) name", "name last", "second surname",
          sens="pii", maxlen=40, norm="name_case", valid="nonempty"),
    _spec("person.name.prefix", _H, "Name Prefix",
          "prefix", "title", "salutation", "name prefix", "mr mrs ms", "honorific",
          "courtesy title", "mr ms mrs dr",
          sens="normal", maxlen=10, norm="title"),
    _spec("person.name.suffix", _T, "Name Suffix",
          "suffix", "name suffix", "jr sr iii", "generational suffix", "sfx",
          sens="normal", maxlen=10, norm="upper"),
    _spec("person.name.maiden", _T, "Maiden Name",
          "maiden name", "birth name", "name at birth", "former name", "previous name",
          "other names used", "prior name",
          sens="pii", maxlen=40, norm="name_case"),
    _spec("person.name.preferred", _T, "Preferred Name",
          "preferred name", "nickname", "goes by", "known as", "preferred first name",
          "display name",
          sens="pii", maxlen=40, norm="name_case"),
    _spec("person.name.initials", _T, "Name Initials",
          "name initials", "initials of name", "your initials", "first and last initials",
          fmt="XX", sens="pii", maxlen=5, norm="upper"),
]

# ---- person.address ------------------------------------------------------------------
_SPECS += [
    _spec("person.address.street_1", _T, "Street Address",
          "address", "street address", "address line 1", "street", "home address",
          "residence address", "residential address", "street name and number",
          "address 1", "addr", "number and street", "mailing address",
          "present address", "current address", "street address line 1",
          sens="pii", maxlen=80, norm="strip_ws", valid="nonempty"),
    _spec("person.address.street_2", _T, "Address Line 2",
          "address line 2", "address 2", "street address line 2", "line 2",
          "apartment suite unit", "additional address", "c o", "care of",
          sens="pii", maxlen=80, norm="strip_ws"),
    _spec("person.address.unit", _T, "Apartment / Unit",
          "apt", "apartment", "unit", "suite", "apt no", "unit number", "suite number",
          "apartment number", "apt unit", "room", "floor", "building",
          sens="pii", maxlen=20, norm="strip_ws"),
    _spec("person.address.city", _T, "City",
          "city", "town", "city or town", "city town", "municipality", "locality",
          "city name", "town city",
          sens="pii", maxlen=50, norm="name_case", valid="nonempty"),
    _spec("person.address.county", _T, "County",
          "county", "parish", "county parish", "borough", "district",
          sens="pii", maxlen=50, norm="name_case"),
    _spec("person.address.region", _T, "State / Province",
          "state", "province", "state province", "state or province", "region",
          "st", "state prov", "province territory", "state territory", "county state",
          pattern=RE_STATE, fmt="XX", sens="pii", maxlen=2, norm="state_abbrev"),
    _spec("person.address.postal_code", _T, "ZIP / Postal Code",
          "zip", "zip code", "postal code", "zipcode", "zip postal", "mail zip",
          "postcode", "post code", "zip postal code", "postal zip", "zip cd",
          "zip or postal code", "postal code zip", "zip 5",
          pattern=RE_ZIP, fmt="NNNNN", sens="pii", maxlen=10,
          norm="zip5", valid="zip_us"),
    _spec("person.address.postal_code_ext", _T, "ZIP+4 Extension",
          "zip 4", "zip plus 4", "zip extension", "plus four", "zip4",
          "last four of zip",
          pattern=RE_ZIP4, fmt="NNNN", sens="pii", maxlen=4,
          norm="digits_only"),
    _spec("person.address.country", _T, "Country",
          "country", "nation", "country or region", "country region", "country of residence",
          sens="pii", maxlen=56, norm="country_iso2"),
    _spec("person.address.full", _M, "Full Address",
          "full address", "complete address", "address city state zip",
          "street city state zip", "full mailing address", "address in full",
          sens="pii", maxlen=200, norm="strip_ws"),
]

# ---- person.phone --------------------------------------------------------------------
_SPECS += [
    _spec("person.phone.mobile", _P, "Mobile Phone",
          "phone", "telephone", "phone number", "tel", "mobile", "cell", "cell phone",
          "mobile phone", "mobile number", "cell number", "cellular", "mobile no",
          "phone no", "telephone number", "contact number", "primary phone",
          "day phone", "daytime phone",
          pattern=RE_PHONE, fmt="(NNN) NNN-NNNN", sens="pii", maxlen=20,
          norm="phone_us", valid="phone_us"),
    _spec("person.phone.home", _P, "Home Phone",
          "home phone", "home telephone", "home number", "residence phone",
          "evening phone", "night phone", "land line", "landline",
          pattern=RE_PHONE, fmt="(NNN) NNN-NNNN", sens="pii", maxlen=20,
          norm="phone_us", valid="phone_us"),
    _spec("person.phone.work", _P, "Work Phone",
          "work phone", "business phone", "office phone", "work telephone",
          "work number", "office number", "business telephone",
          pattern=RE_PHONE, fmt="(NNN) NNN-NNNN", sens="pii", maxlen=20,
          norm="phone_us", valid="phone_us"),
    _spec("person.phone.fax", _P, "Fax",
          "fax", "fax number", "facsimile", "fax no", "telefax",
          pattern=RE_PHONE, fmt="(NNN) NNN-NNNN", sens="pii", maxlen=20,
          norm="phone_us"),
    _spec("person.phone.extension", _T, "Phone Extension",
          "ext", "extension", "phone extension", "ext no", "x",
          pattern=RE_EXT, fmt="NNNN", sens="normal", maxlen=6, norm="digits_only"),
]

# ---- person identity -----------------------------------------------------------------
_SPECS += [
    _spec("person.email", _E, "Email Address",
          "email", "e-mail", "email address", "e-mail address", "your email",
          "email id", "electronic mail", "e mail", "contact email", "primary email",
          "personal email",
          pattern=RE_EMAIL, fmt="user@example.com", sens="pii", maxlen=80,
          norm="email", valid="email"),
    _spec("person.email_alt", _E, "Alternate Email",
          "alternate email", "secondary email", "other email", "email 2",
          "alternative email address", "backup email",
          pattern=RE_EMAIL, fmt="user@example.com", sens="pii", maxlen=80,
          norm="email", valid="email"),
    _spec("person.date_of_birth", _D, "Date of Birth",
          "dob", "d.o.b.", "date of birth", "birth date", "birthdate", "birthday",
          "born on", "date born", "dt of birth",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="pii", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("person.place_of_birth", _T, "Place of Birth",
          "place of birth", "birth place", "birthplace", "city of birth",
          "country of birth", "pob",
          sens="pii", maxlen=60, norm="name_case"),
    _spec("person.gender", _H, "Gender",
          "gender", "sex", "gender identity", "male female", "m f",
          sens="pii", maxlen=20, norm="title"),
    _spec("person.marital_status", _H, "Marital Status",
          "marital status", "married single", "civil status", "relationship status",
          sens="pii", maxlen=20, norm="title"),
    _spec("person.nationality", _T, "Nationality",
          "nationality", "national origin", "country of nationality",
          sens="pii", maxlen=40, norm="name_case"),
    _spec("person.citizenship", _T, "Citizenship",
          "citizenship", "citizen of", "country of citizenship", "citizenship status",
          sens="pii", maxlen=40, norm="name_case"),
    _spec("person.ssn", _T, "Social Security Number",
          "ssn", "social security number", "social security no", "soc sec no",
          "s.s.n.", "social security", "ss number", "socsec", "ssn tin",
          "social security number ssn",
          pattern=RE_SSN, fmt="NNN-NN-NNNN", sens="secret", maxlen=11,
          norm="ssn", valid="ssn"),
    _spec("person.itin", _T, "ITIN",
          "itin", "individual taxpayer identification number", "taxpayer id number",
          "tin", "individual tax id",
          pattern=RE_ITIN, fmt="9NN-NN-NNNN", sens="secret", maxlen=11,
          norm="ssn"),
    _spec("person.age", _N, "Age",
          "age", "age in years", "current age", "years of age",
          fmt="NN", sens="pii", maxlen=3, norm="digits_only"),
    _spec("person.height", _T, "Height",
          "height", "ht", "height ft in", "height cm",
          sens="pii", maxlen=12, norm="strip_ws"),
    _spec("person.weight", _T, "Weight",
          "weight", "wt", "weight lbs", "weight kg",
          sens="pii", maxlen=12, norm="strip_ws"),
    _spec("person.occupation", _T, "Occupation",
          "occupation", "profession", "trade", "line of work",
          sens="normal", maxlen=60, norm="title"),
    _spec("person.employer", _T, "Employer",
          "employer", "employer name", "name of employer", "current employer",
          "place of employment", "employed by",
          sens="normal", maxlen=80, norm="strip_ws"),
    _spec("person.job_title", _T, "Job Title",
          "job title", "position", "title position", "role", "your title",
          "position held", "position title",
          sens="normal", maxlen=60, norm="title"),
    _spec("person.signature", _G, "Signature",
          "signature", "sign", "sign here", "your signature", "signature of applicant",
          "applicant signature", "sign name", "signature x", "signed",
          sens="pii", maxlen=80),
    _spec("person.initials", _T, "Initials",
          "initials", "initial here", "please initial", "init", "initials here",
          fmt="XX", sens="pii", maxlen=5, norm="upper"),
    _spec("person.photo", _T, "Photograph",
          "photo", "photograph", "picture", "attach photo", "passport photo",
          sens="pii", maxlen=200),
]

# ---- person credentials / documents --------------------------------------------------
_SPECS += [
    _spec("person.driver_license.number", _T, "Driver License Number",
          "drivers license", "driver license number", "dl no", "dl number",
          "drivers license number", "license number", "driver s license",
          "operator license number", "dln",
          sens="pii", maxlen=25, norm="upper"),
    _spec("person.driver_license.state", _T, "Driver License State",
          "license state", "dl state", "state of issue", "issuing state",
          "drivers license state",
          pattern=RE_STATE, fmt="XX", sens="pii", maxlen=2, norm="state_abbrev"),
    _spec("person.driver_license.expiration", _D, "Driver License Expiration",
          "license expiration", "dl expiration", "license exp date",
          "drivers license expiration date", "license expires",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="pii", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("person.passport.number", _T, "Passport Number",
          "passport", "passport number", "passport no", "travel document number",
          sens="pii", maxlen=20, norm="upper"),
    _spec("person.passport.country", _T, "Passport Country",
          "passport country", "country of issuance", "passport issuing country",
          "issuing country",
          sens="pii", maxlen=56, norm="country_iso2"),
    _spec("person.passport.expiration", _D, "Passport Expiration",
          "passport expiration", "passport expiry", "passport exp date",
          "passport expiration date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="pii", maxlen=10,
          norm="date_mdy", valid="date"),
]

# ---- person emergency contact / family ------------------------------------------------
_SPECS += [
    _spec("person.emergency_contact.name", _T, "Emergency Contact Name",
          "emergency contact", "emergency contact name", "in case of emergency contact",
          "ice contact", "notify in case of emergency", "emergency contact person",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("person.emergency_contact.relationship", _T, "Emergency Contact Relationship",
          "emergency contact relationship", "relationship to you",
          "relationship to applicant", "emergency relationship",
          sens="pii", maxlen=40, norm="title"),
    _spec("person.emergency_contact.phone", _P, "Emergency Contact Phone",
          "emergency phone", "emergency contact phone", "emergency contact number",
          "emergency telephone",
          pattern=RE_PHONE, fmt="(NNN) NNN-NNNN", sens="pii", maxlen=20,
          norm="phone_us", valid="phone_us"),
    _spec("person.spouse.name", _T, "Spouse Name",
          "spouse name", "spouse", "husband wife name", "name of spouse",
          "partner name",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("person.spouse.date_of_birth", _D, "Spouse Date of Birth",
          "spouse date of birth", "spouse dob", "spouse birth date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="pii", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("person.spouse.ssn", _T, "Spouse SSN",
          "spouse ssn", "spouse social security number", "spouse s social security number",
          pattern=RE_SSN, fmt="NNN-NN-NNNN", sens="secret", maxlen=11,
          norm="ssn", valid="ssn"),
    _spec("person.dependent.name", _T, "Dependent Name",
          "dependent name", "dependent", "child name", "name of dependent",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("person.dependent.relationship", _T, "Dependent Relationship",
          "dependent relationship", "relationship to dependent",
          sens="pii", maxlen=40, norm="title"),
    _spec("person.dependent.date_of_birth", _D, "Dependent Date of Birth",
          "dependent date of birth", "dependent dob", "child date of birth",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="pii", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("person.dependent.ssn", _T, "Dependent SSN",
          "dependent ssn", "dependent social security number", "child ssn",
          pattern=RE_SSN, fmt="NNN-NN-NNNN", sens="secret", maxlen=11,
          norm="ssn", valid="ssn"),
]

# ---- company -------------------------------------------------------------------------
_SPECS += [
    _spec("company.legal_name", _T, "Legal Entity Name",
          "company name", "business name", "company", "legal entity name",
          "name of company", "name of business", "organization name", "firm name",
          "corporation name", "entity name", "employer legal name", "business legal name",
          "company legal name",
          sens="normal", maxlen=100, norm="strip_ws", valid="nonempty"),
    _spec("company.dba_name", _T, "DBA / Trade Name",
          "dba", "d.b.a.", "doing business as", "trade name", "dba name",
          "fictitious name", "assumed name", "operating name",
          sens="normal", maxlen=100, norm="strip_ws"),
    _spec("company.entity_type", _H, "Entity Type",
          "entity type", "business type", "type of entity", "legal structure",
          "business structure", "form of organization", "type of business",
          sens="normal", maxlen=40, norm="title"),
    _spec("company.formation_state", _T, "State of Formation",
          "state of formation", "state of incorporation", "formation state",
          "incorporated in", "state organized", "domicile state",
          pattern=RE_STATE, fmt="XX", sens="normal", maxlen=2, norm="state_abbrev"),
    _spec("company.formation_date", _D, "Date of Formation",
          "date of formation", "date of incorporation", "formation date",
          "incorporation date", "date established", "date business started",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("company.website", _T, "Website",
          "website", "web site", "url", "web address", "company website",
          "homepage", "www",
          pattern=RE_URL, fmt="https://example.com", sens="normal", maxlen=120,
          norm="lower"),
    _spec("company.industry", _T, "Industry",
          "industry", "sector", "line of business", "nature of business",
          "type of industry",
          sens="normal", maxlen=60, norm="title"),
    _spec("company.naics", _T, "NAICS Code",
          "naics", "naics code", "north american industry classification",
          pattern=RE_NAICS, fmt="NNNNNN", sens="normal", maxlen=6, norm="digits_only"),
    _spec("company.sic", _T, "SIC Code",
          "sic", "sic code", "standard industrial classification",
          pattern=RE_NAICS, fmt="NNNN", sens="normal", maxlen=4, norm="digits_only"),
    _spec("company.duns", _T, "DUNS Number",
          "duns", "duns number", "d u n s", "dun and bradstreet number",
          pattern=RE_DUNS, fmt="NNNNNNNNN", sens="normal", maxlen=9, norm="digits_only"),
    _spec("company.tax_id.ein", _T, "Employer Identification Number",
          "ein", "federal ein", "employer identification number", "federal tax id",
          "fed id", "fein", "federal id number", "federal employer id",
          "employer id number", "tax id", "tax id number", "taxpayer identification number",
          "federal tax identification number", "irs ein",
          pattern=RE_EIN, fmt="NN-NNNNNNN", sens="secret", maxlen=10,
          norm="ein", valid="ein"),
    _spec("company.tax_id.vat", _T, "VAT Number",
          "vat", "vat number", "vat id", "vat registration number", "value added tax number",
          sens="normal", maxlen=20, norm="upper"),
    _spec("company.tax_id.gst", _T, "GST Number",
          "gst", "gst number", "gst hst number", "goods and services tax number",
          sens="normal", maxlen=20, norm="upper"),
    _spec("company.tax_id.state_tax_id", _T, "State Tax ID",
          "state tax id", "state id number", "state tax identification number",
          "state employer id", "state registration number",
          sens="normal", maxlen=20, norm="upper"),
]
_SPECS += _address_block("company.address", "Company", sens="normal")
_SPECS += [
    _spec("company.contact.name", _T, "Company Contact Name",
          "contact name", "contact person", "primary contact", "authorized representative",
          "point of contact", "representative name", "contact",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("company.contact.title", _T, "Company Contact Title",
          "contact title", "title of contact", "representative title",
          "officer title", "title position",
          sens="normal", maxlen=60, norm="title"),
    _spec("company.contact.email", _E, "Company Contact Email",
          "contact email", "business email", "company email", "work email",
          "office email",
          pattern=RE_EMAIL, fmt="user@example.com", sens="pii", maxlen=80,
          norm="email", valid="email"),
    _spec("company.contact.phone", _P, "Company Contact Phone",
          "contact phone", "business phone number", "company phone", "office telephone",
          "contact telephone",
          pattern=RE_PHONE, fmt="(NNN) NNN-NNNN", sens="pii", maxlen=20,
          norm="phone_us", valid="phone_us"),
]

# ---- bank / card ---------------------------------------------------------------------
_SPECS += [
    _spec("bank.name", _T, "Bank Name",
          "bank", "bank name", "name of bank", "financial institution",
          "institution name", "depository name",
          sens="normal", maxlen=80, norm="strip_ws"),
    _spec("bank.routing_number", _T, "Routing Number",
          "routing number", "aba", "aba number", "routing aba", "aba routing number",
          "routing transit number", "rtn", "bank routing number", "transit number",
          pattern=RE_ROUTING, fmt="NNNNNNNNN", sens="pii", maxlen=9,
          norm="digits_only", valid="routing_aba"),
    _spec("bank.account_number", _T, "Bank Account Number",
          "account number", "acct number", "acct", "bank account number",
          "account no", "checking account number", "savings account number",
          "deposit account number", "account num",
          pattern=RE_ACCOUNT, fmt="NNNNNNNNNNNN", sens="secret", maxlen=20,
          norm="digits_only"),
    _spec("bank.account_type", _H, "Account Type",
          "account type", "type of account", "checking or savings", "checking savings",
          sens="normal", maxlen=20, norm="title"),
    _spec("bank.swift", _T, "SWIFT / BIC",
          "swift", "swift code", "bic", "swift bic", "bic code", "swift bic code",
          pattern=RE_SWIFT, fmt="AAAABBCCDDD", sens="normal", maxlen=11, norm="upper"),
    _spec("bank.iban", _T, "IBAN",
          "iban", "iban number", "international bank account number",
          pattern=RE_IBAN, fmt="CCNN AAAA ...", sens="secret", maxlen=34,
          norm="iban", valid="iban"),
    _spec("bank.branch", _T, "Bank Branch",
          "branch", "branch name", "bank branch", "branch location", "branch address",
          sens="normal", maxlen=80, norm="strip_ws"),
    _spec("bank.account_holder", _T, "Account Holder",
          "account holder", "account holder name", "name on account",
          "name of account holder", "depositor name",
          sens="pii", maxlen=80, norm="name_case"),
]
_SPECS += [
    _spec("card.number", _T, "Card Number",
          "card number", "credit card number", "cc number", "card no",
          "debit card number", "payment card number", "card 16 digits",
          pattern=RE_CARD, fmt="NNNN NNNN NNNN NNNN", sens="secret", maxlen=19,
          norm="credit_card", valid="luhn"),
    _spec("card.brand", _H, "Card Brand",
          "card brand", "card type", "type of card", "visa mastercard",
          "credit card type",
          sens="normal", maxlen=20, norm="title"),
    _spec("card.expiration_month", _T, "Card Expiration Month",
          "exp month", "expiration month", "card expiration month", "month exp",
          pattern=RE_MONTH_NUM, fmt="MM", sens="normal", maxlen=2, norm="digits_only"),
    _spec("card.expiration_year", _T, "Card Expiration Year",
          "exp year", "expiration year", "card expiration year", "year exp",
          pattern=RE_YEAR, fmt="YYYY", sens="normal", maxlen=4, norm="digits_only"),
    _spec("card.expiration", _T, "Card Expiration",
          "exp date", "expiration date mm yy", "card expiration", "valid thru",
          "good thru", "exp",
          pattern=RE_MMYY, fmt="MM/YY", sens="normal", maxlen=7, norm="strip_ws"),
    _spec("card.cvv", _T, "Card Security Code",
          "cvv", "cvc", "cid", "security code", "card security code", "cvv2",
          "card verification value", "card code",
          pattern=RE_CVV, fmt="NNN", sens="secret", maxlen=4, norm="digits_only"),
    _spec("card.holder_name", _T, "Cardholder Name",
          "cardholder name", "name on card", "card holder", "cardholder",
          "name as it appears on card",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("card.billing_zip", _T, "Card Billing ZIP",
          "billing zip code for card", "card billing zip", "card zip",
          pattern=RE_ZIP, fmt="NNNNN", sens="pii", maxlen=10,
          norm="zip5", valid="zip_us"),
]

# ---- parent-context discriminated address blocks --------------------------------------
_SPECS += _address_block("billing.address", "Billing", parents=("billing",), extras=True)
_SPECS += _address_block("shipping.address", "Shipping", parents=("shipping",), extras=True)
_SPECS += _address_block("mailing.address", "Mailing", parents=("mailing",))

# ---- credentials ---------------------------------------------------------------------
_SPECS += [
    _spec("credentials.username", _T, "Username",
          "username", "user name", "user id", "login", "login id", "account id",
          sens="pii", maxlen=64, norm="lower"),
    _spec("credentials.password", _T, "Password",
          "password", "pass word", "passcode", "new password", "confirm password",
          sens="secret", maxlen=128),
    _spec("credentials.pin", _T, "PIN",
          "pin", "pin number", "personal identification number", "pin code",
          fmt="NNNN", sens="secret", maxlen=8, norm="digits_only"),
    _spec("credentials.security_question", _T, "Security Question",
          "security question", "challenge question", "secret question",
          sens="normal", maxlen=120, norm="strip_ws"),
    _spec("credentials.security_answer", _T, "Security Answer",
          "security answer", "answer to security question", "secret answer",
          sens="secret", maxlen=120),
]

# ---- document ------------------------------------------------------------------------
_SPECS += [
    _spec("document.title", _T, "Document Title",
          "document title", "form title", "title of document", "subject",
          "name of form",
          sens="normal", maxlen=120, norm="strip_ws"),
    _spec("document.number", _T, "Document Number",
          "document number", "form number", "document no", "form no", "doc number",
          sens="normal", maxlen=40, norm="upper"),
    _spec("document.type", _H, "Document Type",
          "document type", "form type", "type of document", "type of form",
          sens="normal", maxlen=40, norm="title"),
    _spec("document.effective_date", _D, "Effective Date",
          "effective date", "date effective", "commencement date", "start date",
          "date of effect", "effective as of",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("document.expiration_date", _D, "Expiration Date",
          "expiration date", "expiry date", "date of expiration", "expires on",
          "termination date", "valid until", "end date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("document.signed_date", _D, "Date Signed",
          "date signed", "signature date", "date of signature", "signed on",
          "dated", "date of execution", "execution date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("document.today", _D, "Today's Date",
          "date", "today's date", "todays date", "current date", "date completed",
          "date of application", "application date", "date submitted", "submission date",
          "date prepared", "date filled",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("document.reference_number", _T, "Reference Number",
          "reference number", "ref no", "reference", "our reference", "your reference",
          "ref number", "tracking number", "confirmation number",
          sens="normal", maxlen=40, norm="upper"),
    _spec("document.case_number", _T, "Case Number",
          "case number", "case no", "matter number", "file number", "docket number",
          "claim number",
          sens="normal", maxlen=40, norm="upper"),
    _spec("document.policy_number", _T, "Policy Number",
          "policy number", "policy no", "contract number", "certificate number",
          sens="normal", maxlen=40, norm="upper"),
    _spec("document.account_number", _T, "Customer Account Number",
          "customer account number", "member account number", "your account number",
          "account reference", "customer number", "client number",
          sens="pii", maxlen=40, norm="upper"),
    _spec("document.invoice_number", _T, "Invoice Number",
          "invoice number", "invoice no", "invoice", "bill number", "statement number",
          sens="normal", maxlen=40, norm="upper"),
    _spec("document.purchase_order", _T, "Purchase Order",
          "purchase order", "po number", "p o number", "po no", "purchase order number",
          sens="normal", maxlen=40, norm="upper"),
    _spec("document.amount", _C, "Amount",
          "amount", "amt", "amount due", "sum", "value", "amount requested",
          "amount paid", "payment amount",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
    _spec("document.subtotal", _C, "Subtotal",
          "subtotal", "sub total", "net amount", "amount before tax",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
    _spec("document.tax", _C, "Tax",
          "tax", "sales tax", "tax amount", "vat amount", "gst amount",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
    _spec("document.total", _C, "Total",
          "total", "grand total", "total amount", "total due", "balance due",
          "amount enclosed", "total payment",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
    _spec("document.currency", _H, "Currency",
          "currency", "currency code", "in currency", "denominated in",
          sens="normal", maxlen=3, norm="upper"),
    _spec("document.description", _M, "Description",
          "description", "describe", "details", "particulars", "description of goods",
          "nature of request",
          sens="normal", maxlen=500, norm="strip_ws"),
    _spec("document.notes", _M, "Notes",
          "notes", "remarks", "additional information", "additional notes",
          "special instructions", "explain", "explanation",
          sens="normal", maxlen=1000, norm="strip_ws"),
    _spec("document.page_count", _N, "Page Count",
          "page count", "number of pages", "total pages", "pages",
          fmt="N", sens="normal", maxlen=4, norm="digits_only"),
    _spec("document.status", _H, "Status",
          "status", "current status", "application status",
          sens="normal", maxlen=30, norm="title"),
    _spec("document.version", _T, "Version",
          "version", "revision", "rev", "version number",
          sens="normal", maxlen=20, norm="upper"),
    _spec("document.department", _T, "Department",
          "department", "dept", "division", "business unit",
          sens="normal", maxlen=60, norm="title"),
    _spec("document.location", _T, "Location",
          "location", "site", "place", "facility", "office location",
          sens="normal", maxlen=80, norm="strip_ws"),
    _spec("document.received_date", _D, "Date Received",
          "date received", "received date", "received on", "date of receipt",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("document.due_date", _D, "Due Date",
          "due date", "date due", "payment due date", "deadline", "due by",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("document.quantity", _N, "Quantity",
          "quantity", "qty", "number of units", "units", "no of items",
          fmt="N", sens="normal", maxlen=10, norm="digits_only"),
    _spec("document.unit_price", _C, "Unit Price",
          "unit price", "price per unit", "rate", "price each", "unit cost",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
    _spec("document.discount", _C, "Discount",
          "discount", "discount amount", "less discount", "rebate",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
    _spec("document.payment_method", _H, "Payment Method",
          "payment method", "method of payment", "pay by", "form of payment",
          sens="normal", maxlen=30, norm="title"),
    _spec("document.terms", _M, "Terms",
          "terms", "payment terms", "terms and conditions", "terms of payment",
          sens="normal", maxlen=500, norm="strip_ws"),
]

# ---- document.signer -----------------------------------------------------------------
_SPECS += [
    _spec("document.signer.name", _T, "Signer Name",
          "signer name", "name of signer", "name of authorized signer",
          "print name of signer", "signed by", "name (print)", "authorized signer name",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("document.signer.title", _T, "Signer Title",
          "title of signer", "signer title", "official title", "title of officer",
          "authorized signer title",
          sens="normal", maxlen=60, norm="title"),
    _spec("document.signer.signature", _G, "Signer Signature",
          "authorized signature", "signature of authorized representative",
          "signature of signer", "signature of officer", "signature of owner",
          "signature of borrower", "signature of employee",
          sens="pii", maxlen=80),
    _spec("document.signer.initials", _T, "Signer Initials",
          "signer initials", "initials of signer", "initial each page",
          fmt="XX", sens="pii", maxlen=5, norm="upper"),
    _spec("document.signer.date", _D, "Signer Date",
          "signer date", "date of signer", "date signature", "date (signature)",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("document.signer.capacity", _T, "Signing Capacity",
          "capacity", "signing capacity", "acting as", "in the capacity of",
          sens="normal", maxlen=60, norm="title"),
    _spec("document.signer.witness_name", _T, "Witness Name",
          "witness name", "witness", "name of witness", "print witness name",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("document.signer.witness_signature", _G, "Witness Signature",
          "witness signature", "signature of witness",
          sens="pii", maxlen=80),
    _spec("document.signer.notary_name", _T, "Notary Name",
          "notary name", "notary public", "name of notary", "notary",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("document.signer.notary_commission_expiration", _D, "Notary Commission Expiration",
          "commission expires", "my commission expires", "notary commission expiration",
          "commission expiration date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
]

# ---- vehicle -------------------------------------------------------------------------
_SPECS += [
    _spec("vehicle.vin", _T, "VIN",
          "vin", "vehicle identification number", "vin number", "chassis number",
          "serial number of vehicle",
          pattern=RE_VIN, fmt="17 alphanumeric", sens="normal", maxlen=17,
          norm="upper", valid="vin"),
    _spec("vehicle.make", _T, "Vehicle Make",
          "make", "vehicle make", "manufacturer", "car make",
          sens="normal", maxlen=40, norm="title"),
    _spec("vehicle.model", _T, "Vehicle Model",
          "model", "vehicle model", "car model",
          sens="normal", maxlen=40, norm="title"),
    _spec("vehicle.year", _T, "Vehicle Year",
          "year", "vehicle year", "model year", "year of vehicle",
          pattern=RE_YEAR, fmt="YYYY", sens="normal", maxlen=4, norm="digits_only"),
    _spec("vehicle.plate", _T, "License Plate",
          "license plate", "plate number", "tag number", "registration number",
          "plate", "license plate number",
          sens="normal", maxlen=12, norm="upper"),
    _spec("vehicle.state", _T, "Vehicle Registration State",
          "plate state", "vehicle state", "state of registration", "registration state",
          pattern=RE_STATE, fmt="XX", sens="normal", maxlen=2, norm="state_abbrev"),
    _spec("vehicle.color", _T, "Vehicle Color",
          "color", "vehicle color", "colour", "car color",
          sens="normal", maxlen=20, norm="title"),
    _spec("vehicle.mileage", _N, "Vehicle Mileage",
          "mileage", "odometer", "odometer reading", "miles", "km reading",
          fmt="N", sens="normal", maxlen=8, norm="digits_only"),
]

# ---- insurance -----------------------------------------------------------------------
_SPECS += [
    _spec("insurance.carrier", _T, "Insurance Carrier",
          "insurance carrier", "carrier", "insurance company", "insurer",
          "name of insurance company", "underwriter",
          sens="normal", maxlen=80, norm="strip_ws"),
    _spec("insurance.policy_number", _T, "Insurance Policy Number",
          "insurance policy number", "insurance id", "policy id",
          sens="normal", maxlen=40, norm="upper"),
    _spec("insurance.group_number", _T, "Group Number",
          "group number", "group no", "group id", "plan group number",
          sens="normal", maxlen=40, norm="upper"),
    _spec("insurance.member_id", _T, "Member ID",
          "member id", "member number", "subscriber id", "insurance member id",
          "id number on card",
          sens="pii", maxlen=40, norm="upper"),
    _spec("insurance.effective_date", _D, "Coverage Effective Date",
          "coverage effective date", "insurance effective date", "coverage start date",
          "policy effective date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("insurance.expiration_date", _D, "Coverage Expiration Date",
          "coverage expiration date", "insurance expiration date", "coverage end date",
          "policy expiration date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("insurance.plan_name", _T, "Plan Name",
          "plan name", "insurance plan", "plan", "coverage plan", "plan type",
          sens="normal", maxlen=60, norm="strip_ws"),
    _spec("insurance.subscriber_name", _T, "Subscriber Name",
          "subscriber name", "policy holder", "policyholder", "name of subscriber",
          "insured name", "primary insured",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("insurance.subscriber_relationship", _H, "Relationship to Subscriber",
          "relationship to subscriber", "relationship to insured",
          "patient relationship to subscriber",
          sens="pii", maxlen=30, norm="title"),
]

# ---- medical -------------------------------------------------------------------------
_SPECS += [
    _spec("medical.provider", _T, "Healthcare Provider",
          "provider", "provider name", "healthcare provider", "clinic name",
          "hospital name", "facility name", "practice name",
          sens="normal", maxlen=80, norm="strip_ws"),
    _spec("medical.npi", _T, "NPI",
          "npi", "npi number", "national provider identifier", "provider npi",
          pattern=RE_NPI, fmt="NNNNNNNNNN", sens="normal", maxlen=10,
          norm="digits_only"),
    _spec("medical.diagnosis_code", _T, "Diagnosis Code",
          "diagnosis code", "icd code", "icd 10", "icd 10 code", "dx code",
          sens="pii", maxlen=10, norm="upper"),
    _spec("medical.procedure_code", _T, "Procedure Code",
          "procedure code", "cpt code", "cpt", "hcpcs code", "px code",
          sens="normal", maxlen=10, norm="upper"),
    _spec("medical.allergies", _M, "Allergies",
          "allergies", "known allergies", "drug allergies", "list allergies",
          "allergy list",
          sens="pii", maxlen=500, norm="strip_ws"),
    _spec("medical.medications", _M, "Medications",
          "medications", "current medications", "list medications", "prescriptions",
          "meds",
          sens="pii", maxlen=500, norm="strip_ws"),
    _spec("medical.conditions", _M, "Medical Conditions",
          "medical conditions", "conditions", "medical history", "existing conditions",
          "pre existing conditions", "health conditions",
          sens="pii", maxlen=500, norm="strip_ws"),
    _spec("medical.physician_name", _T, "Physician Name",
          "physician name", "doctor name", "primary care physician", "pcp",
          "attending physician", "name of physician", "referring physician",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("medical.pharmacy", _T, "Pharmacy",
          "pharmacy", "pharmacy name", "preferred pharmacy", "drug store",
          sens="normal", maxlen=80, norm="strip_ws"),
    _spec("medical.blood_type", _H, "Blood Type",
          "blood type", "blood group", "abo type",
          sens="pii", maxlen=3, norm="upper"),
]

# ---- employment ----------------------------------------------------------------------
_SPECS += [
    _spec("employment.employer_name", _T, "Employer Name",
          "employer company name", "name of current employer", "employer business name",
          "company you work for", "present employer",
          sens="normal", maxlen=80, norm="strip_ws"),
    _spec("employment.job_title", _T, "Employment Job Title",
          "employment title", "job title position", "occupation title",
          "current job title", "your position title",
          sens="normal", maxlen=60, norm="title"),
    _spec("employment.start_date", _D, "Employment Start Date",
          "hire date", "date of hire", "employment start date", "date started",
          "start of employment", "employed since", "date employed",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("employment.end_date", _D, "Employment End Date",
          "employment end date", "date of separation", "last day worked",
          "termination of employment", "date ended", "separation date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("employment.supervisor", _T, "Supervisor",
          "supervisor", "supervisor name", "manager name", "reports to",
          "immediate supervisor", "name of supervisor",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("employment.salary", _C, "Salary",
          "salary", "annual salary", "gross salary", "wage", "hourly rate",
          "compensation", "monthly salary", "rate of pay",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="pii", maxlen=20, norm="currency"),
    _spec("employment.pay_frequency", _H, "Pay Frequency",
          "pay frequency", "pay period", "paid how often", "frequency of pay",
          "payroll frequency",
          sens="normal", maxlen=20, norm="title"),
    _spec("employment.department", _T, "Employment Department",
          "employee department", "work department", "assigned department",
          sens="normal", maxlen=60, norm="title"),
    _spec("employment.employee_id", _T, "Employee ID",
          "employee id", "employee number", "emp id", "staff id", "payroll id",
          "employee id number", "badge number",
          sens="pii", maxlen=20, norm="upper"),
    _spec("employment.work_address", _T, "Work Address",
          "work address", "business address", "office address", "employer address",
          "address of employer", "work location address",
          sens="normal", maxlen=120, norm="strip_ws"),
    _spec("employment.hours_per_week", _N, "Hours Per Week",
          "hours per week", "weekly hours", "hrs per week", "hours worked per week",
          "average hours",
          fmt="NN", sens="normal", maxlen=4, norm="digits_only"),
]

# ---- education -----------------------------------------------------------------------
_SPECS += [
    _spec("education.school_name", _T, "School Name",
          "school", "school name", "name of school", "university", "college",
          "institution attended", "name of institution", "high school",
          sens="normal", maxlen=100, norm="strip_ws"),
    _spec("education.degree", _H, "Degree",
          "degree", "degree earned", "degree obtained", "diploma", "qualification",
          "highest degree", "level of education",
          sens="normal", maxlen=60, norm="title"),
    _spec("education.field_of_study", _T, "Field of Study",
          "field of study", "major", "course of study", "area of study", "concentration",
          "discipline",
          sens="normal", maxlen=60, norm="title"),
    _spec("education.graduation_date", _D, "Graduation Date",
          "graduation date", "date of graduation", "date graduated", "year graduated",
          "completion date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("education.gpa", _N, "GPA",
          "gpa", "grade point average", "cumulative gpa", "g p a",
          pattern=RE_GPA, fmt="N.NN", sens="normal", maxlen=5, norm="strip_ws"),
    _spec("education.student_id", _T, "Student ID",
          "student id", "student number", "student id number", "enrollment number",
          "matriculation number",
          sens="pii", maxlen=20, norm="upper"),
]

# ---- property ------------------------------------------------------------------------
_SPECS += [
    _spec("property.address", _T, "Property Address",
          "property address", "subject property address", "address of property",
          "premises address", "property location",
          sens="normal", maxlen=120, norm="strip_ws"),
    _spec("property.parcel_number", _T, "Parcel Number",
          "parcel number", "apn", "assessor parcel number", "tax parcel id",
          "parcel id", "folio number",
          sens="normal", maxlen=40, norm="upper"),
    _spec("property.legal_description", _M, "Legal Description",
          "legal description", "legal description of property", "lot and block",
          "metes and bounds",
          sens="normal", maxlen=1000, norm="strip_ws"),
    _spec("property.purchase_price", _C, "Purchase Price",
          "purchase price", "sale price", "price of property", "contract price",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
    _spec("property.purchase_date", _D, "Purchase Date",
          "purchase date", "date of purchase", "date acquired", "closing date",
          "date of sale",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("property.square_feet", _N, "Square Feet",
          "square feet", "sq ft", "square footage", "living area", "floor area",
          fmt="N", sens="normal", maxlen=10, norm="digits_only"),
]

# ---- tax -----------------------------------------------------------------------------
_SPECS += [
    _spec("tax.filing_status", _H, "Filing Status",
          "filing status", "tax filing status", "single married filing jointly",
          sens="pii", maxlen=40, norm="title"),
    _spec("tax.year", _T, "Tax Year",
          "tax year", "year ending", "for the year", "calendar year",
          pattern=RE_YEAR, fmt="YYYY", sens="normal", maxlen=4, norm="digits_only"),
    _spec("tax.exemptions", _N, "Exemptions",
          "exemptions", "number of exemptions", "total exemptions",
          fmt="N", sens="normal", maxlen=3, norm="digits_only"),
    _spec("tax.withholding_allowances", _N, "Withholding Allowances",
          "allowances", "withholding allowances", "number of allowances",
          "total number of allowances",
          fmt="N", sens="normal", maxlen=3, norm="digits_only"),
    _spec("tax.gross_income", _C, "Gross Income",
          "gross income", "total income", "gross wages", "gross pay",
          "annual gross income",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="pii", maxlen=20, norm="currency"),
    _spec("tax.adjusted_gross_income", _C, "Adjusted Gross Income",
          "adjusted gross income", "agi", "a g i",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="pii", maxlen=20, norm="currency"),
    _spec("tax.refund_amount", _C, "Refund Amount",
          "refund", "refund amount", "amount to be refunded", "overpayment",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
    _spec("tax.amount_owed", _C, "Amount Owed",
          "amount owed", "amount you owe", "tax due", "balance owed",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
]

# ---- legal ---------------------------------------------------------------------------
_SPECS += [
    _spec("legal.court_name", _T, "Court Name",
          "court", "court name", "name of court", "in the court of", "judicial district",
          sens="normal", maxlen=100, norm="strip_ws"),
    _spec("legal.jurisdiction", _T, "Jurisdiction",
          "jurisdiction", "venue", "governing jurisdiction", "state of jurisdiction",
          sens="normal", maxlen=60, norm="title"),
    _spec("legal.plaintiff", _T, "Plaintiff",
          "plaintiff", "petitioner", "claimant", "plaintiff name",
          sens="pii", maxlen=100, norm="name_case"),
    _spec("legal.defendant", _T, "Defendant",
          "defendant", "respondent", "defendant name",
          sens="pii", maxlen=100, norm="name_case"),
    _spec("legal.attorney_name", _T, "Attorney Name",
          "attorney", "attorney name", "counsel", "attorney for", "lawyer name",
          "name of attorney",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("legal.bar_number", _T, "Bar Number",
          "bar number", "state bar no", "attorney bar number", "bar no",
          sens="normal", maxlen=20, norm="upper"),
    _spec("legal.filing_date", _D, "Filing Date",
          "filing date", "date filed", "date of filing",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("legal.case_type", _H, "Case Type",
          "case type", "type of case", "nature of suit", "cause of action",
          sens="normal", maxlen=60, norm="title"),
]

# ---- lease ---------------------------------------------------------------------------
_SPECS += [
    _spec("lease.start_date", _D, "Lease Start Date",
          "lease start date", "lease commencement date", "tenancy start date",
          "lease begins",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("lease.end_date", _D, "Lease End Date",
          "lease end date", "lease expiration date", "tenancy end date", "lease ends",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("lease.monthly_rent", _C, "Monthly Rent",
          "monthly rent", "rent", "rent amount", "monthly rental", "base rent",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
    _spec("lease.security_deposit", _C, "Security Deposit",
          "security deposit", "deposit", "deposit amount", "damage deposit",
          pattern=RE_MONEY, fmt="$N,NNN.NN", sens="normal", maxlen=20, norm="currency"),
    _spec("lease.property_address", _T, "Leased Premises Address",
          "leased premises", "rental address", "premises", "leased property address",
          "rental property address",
          sens="normal", maxlen=120, norm="strip_ws"),
    _spec("lease.landlord_name", _T, "Landlord Name",
          "landlord", "landlord name", "lessor", "name of landlord", "owner name",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("lease.tenant_name", _T, "Tenant Name",
          "tenant", "tenant name", "lessee", "name of tenant", "renter name",
          sens="pii", maxlen=80, norm="name_case"),
    _spec("lease.term_months", _N, "Lease Term (Months)",
          "lease term", "term in months", "length of lease", "term months",
          fmt="N", sens="normal", maxlen=4, norm="digits_only"),
]

# ---- military ------------------------------------------------------------------------
_SPECS += [
    _spec("military.branch", _H, "Military Branch",
          "branch of service", "military branch", "service branch", "armed forces branch",
          sens="normal", maxlen=40, norm="title"),
    _spec("military.rank", _T, "Military Rank",
          "rank", "military rank", "grade", "pay grade",
          sens="normal", maxlen=30, norm="title"),
    _spec("military.service_number", _T, "Service Number",
          "service number", "military id number", "dod id", "military service number",
          sens="pii", maxlen=20, norm="upper"),
    _spec("military.discharge_date", _D, "Discharge Date",
          "discharge date", "date of discharge", "date of separation from service",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("military.status", _H, "Military Status",
          "military status", "veteran status", "active duty status", "service status",
          sens="normal", maxlen=30, norm="title"),
]

# ---- travel --------------------------------------------------------------------------
_SPECS += [
    _spec("travel.departure_date", _D, "Departure Date",
          "departure date", "date of departure", "depart date", "outbound date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("travel.return_date", _D, "Return Date",
          "return date", "date of return", "inbound date", "arrival date",
          pattern=RE_DATE, fmt="MM/DD/YYYY", sens="normal", maxlen=10,
          norm="date_mdy", valid="date"),
    _spec("travel.destination", _T, "Destination",
          "destination", "travelling to", "traveling to", "destination city",
          "place of destination",
          sens="normal", maxlen=80, norm="title"),
    _spec("travel.origin", _T, "Origin",
          "origin", "departing from", "point of origin", "origin city",
          sens="normal", maxlen=80, norm="title"),
    _spec("travel.flight_number", _T, "Flight Number",
          "flight number", "flight no", "flight",
          sens="normal", maxlen=10, norm="upper"),
    _spec("travel.confirmation_number", _T, "Booking Confirmation",
          "booking reference", "record locator", "pnr", "booking confirmation number",
          sens="normal", maxlen=20, norm="upper"),
]

# ---- consent -------------------------------------------------------------------------
_SPECS += [
    _spec("consent.agree", _B, "I Agree",
          "i agree", "agree", "yes i agree", "i consent", "consent", "check to agree",
          pattern=RE_YESNO, sens="normal", maxlen=3, norm="boolean_yes_no"),
    _spec("consent.opt_in", _B, "Opt In",
          "opt in", "subscribe", "yes send me", "sign me up", "please contact me",
          "add me to the mailing list",
          pattern=RE_YESNO, sens="normal", maxlen=3, norm="boolean_yes_no"),
    _spec("consent.opt_out", _B, "Opt Out",
          "opt out", "unsubscribe", "do not contact me", "no thank you",
          "do not share my information",
          pattern=RE_YESNO, sens="normal", maxlen=3, norm="boolean_yes_no"),
    _spec("consent.acknowledge", _B, "Acknowledgement",
          "i acknowledge", "acknowledge", "acknowledgement", "i have read and understand",
          "i understand",
          pattern=RE_YESNO, sens="normal", maxlen=3, norm="boolean_yes_no"),
    _spec("consent.certify", _B, "Certification",
          "i certify", "certify", "certification", "under penalty of perjury",
          "i declare",
          pattern=RE_YESNO, sens="normal", maxlen=3, norm="boolean_yes_no"),
    _spec("consent.terms_accepted", _B, "Terms Accepted",
          "i accept the terms", "accept terms", "terms accepted",
          "i agree to the terms and conditions", "agree to terms",
          pattern=RE_YESNO, sens="normal", maxlen=3, norm="boolean_yes_no"),
    _spec("consent.privacy_accepted", _B, "Privacy Policy Accepted",
          "i accept the privacy policy", "privacy policy accepted",
          "i agree to the privacy policy", "privacy notice acknowledged",
          pattern=RE_YESNO, sens="normal", maxlen=3, norm="boolean_yes_no"),
]

# ---- misc ----------------------------------------------------------------------------
_SPECS += [
    _spec("misc.yes_no", _B, "Yes / No",
          "yes no", "y n", "yes or no", "check one", "check if applicable",
          pattern=RE_YESNO, sens="normal", maxlen=3, norm="boolean_yes_no"),
    _spec("misc.other", _T, "Other",
          "other", "other please specify", "if other specify", "other specify",
          sens="normal", maxlen=100, norm="strip_ws"),
    _spec("misc.comments", _M, "Comments",
          "comments", "comment", "feedback", "additional comments", "your comments",
          sens="normal", maxlen=1000, norm="strip_ws"),
    _spec("misc.reason", _M, "Reason",
          "reason", "reason for request", "purpose", "reason for", "why",
          "purpose of request",
          sens="normal", maxlen=500, norm="strip_ws"),
    _spec("misc.relationship", _T, "Relationship",
          "relationship", "relation", "relationship to", "your relationship",
          sens="pii", maxlen=40, norm="title"),
    _spec("misc.how_did_you_hear", _H, "How Did You Hear About Us",
          "how did you hear about us", "how did you hear", "referral source",
          "referred by", "where did you hear about us",
          sens="normal", maxlen=60, norm="strip_ws"),
    _spec("misc.preferred_contact_method", _H, "Preferred Contact Method",
          "preferred contact method", "best way to contact you", "preferred method of contact",
          "how should we contact you", "contact preference",
          sens="normal", maxlen=30, norm="title"),
]


# --------------------------------------------------------------------------------------
# Registry construction and validation.
# --------------------------------------------------------------------------------------
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")


def _build_registry(specs: List[KeySpec]) -> Dict[str, KeySpec]:
    registry: Dict[str, KeySpec] = {}
    for spec in specs:
        if not _KEY_RE.match(spec.key):
            raise SemanticError(
                f"canonical key {spec.key!r} is not a dotted lowercase path"
            )
        if spec.key in registry:
            raise SemanticError(f"duplicate canonical key {spec.key!r}")
        for name in (spec.normalizer, spec.validator):
            if name is not None and not name.isidentifier():
                raise SemanticError(
                    f"key {spec.key!r} references non-identifier callable {name!r}"
                )
        if spec.pattern is not None:
            re.compile(spec.pattern)
        registry[spec.key] = spec
    return registry


#: The canonical key namespace, in declaration order.  Declaration order is the alias
#: precedence order used by :mod:`zfp.ontology.aliases`.
CANONICAL_KEYS: Dict[str, KeySpec] = _build_registry(_SPECS)


def get(key: str) -> Optional[KeySpec]:
    """Return the :class:`KeySpec` for ``key``, or ``None`` when it is not canonical."""
    return CANONICAL_KEYS.get(key)


def all_keys() -> List[KeySpec]:
    """Return every :class:`KeySpec`, sorted by canonical key for determinism."""
    return sorted(CANONICAL_KEYS.values(), key=lambda s: s.key)


def children(prefix: str) -> List[KeySpec]:
    """Return every spec whose key lives under ``prefix``.

    ``children("person.name")`` returns the ten name keys.  ``children("person")``
    returns the whole person namespace, including nested paths.  The prefix itself is
    never included.
    """
    stem = prefix.rstrip(".") + "."
    return sorted(
        (s for s in CANONICAL_KEYS.values() if s.key.startswith(stem)),
        key=lambda s: s.key,
    )


def namespaces() -> List[str]:
    """Return the sorted list of top-level namespaces (``person``, ``company``, ...)."""
    return sorted({s.namespace for s in CANONICAL_KEYS.values()})


def keys_with_parent(parent: str) -> List[KeySpec]:
    """Return every spec whose ``parents`` tuple contains ``parent`` (case-insensitive)."""
    low = parent.strip().lower()
    return sorted(
        (s for s in CANONICAL_KEYS.values() if low in tuple(p.lower() for p in s.parents)),
        key=lambda s: s.key,
    )


def key_count() -> int:
    """Return the number of declared canonical keys."""
    return len(CANONICAL_KEYS)
