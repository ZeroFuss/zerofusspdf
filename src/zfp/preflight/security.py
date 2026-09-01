"""Security triage: what the document's own rules allow ZFP to do.

Two questions are answered here, and both of them gate everything downstream:

* **May we write at all?**  Encryption carries a permission word (``/P``) whose bits
  say whether printing, modification, extraction, annotation and form filling are
  granted.  :func:`inspect` decodes it exactly, including the two details that trip up
  most implementations: ``/P`` is a *signed* 32-bit integer (so it is almost always
  negative) and the bit numbers in the specification are *1-based*.
* **May we write *here*?**  A signature freezes the bytes it covers.  A certification
  signature additionally declares, through its ``/DocMDP`` transform, which later
  changes are permitted: ``/P 1`` none at all, ``/P 2`` form filling and signing,
  ``/P 3`` those plus annotations.  :func:`inspect_signatures` reports them and
  :func:`can_add_form_fields` turns them into a single allow/refuse decision.

:func:`can_add_form_fields` is the gate the orchestrator's ``SecurityGateAgent`` calls.
It is deliberately conservative: it refuses unless it can *positively* establish that
the edit is permitted, and it always says why.  It never silently allows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.logging import get_logger
from ..pdfio.document import Document
from ..pdfio.objects import PdfArray, PdfDict, PdfName, PdfNull, PdfRef, PdfStream, PdfString

__all__ = [
    "SecurityState",
    "SignatureState",
    "inspect",
    "inspect_security",
    "inspect_signatures",
    "can_add_form_fields",
    "has_permission",
    "PERM_PRINT",
    "PERM_MODIFY",
    "PERM_EXTRACT",
    "PERM_ANNOTATE",
    "PERM_FILL_FORMS",
    "PERM_EXTRACT_ACCESSIBILITY",
    "PERM_ASSEMBLE",
    "PERM_PRINT_HIGH",
    "DOCMDP_NO_CHANGES",
    "DOCMDP_FORM_FILL",
    "DOCMDP_FORM_FILL_AND_ANNOTATE",
]

_log = get_logger(__name__)

# -- standard security handler permission bits (PDF 32000-1 table 22, 1-based) ----------
PERM_PRINT = 3
PERM_MODIFY = 4
PERM_EXTRACT = 5
PERM_ANNOTATE = 6
PERM_FILL_FORMS = 9
PERM_EXTRACT_ACCESSIBILITY = 10
PERM_ASSEMBLE = 11
PERM_PRINT_HIGH = 12

#: ``/DocMDP`` ``/P`` levels (PDF 32000-1 table 254).
DOCMDP_NO_CHANGES = 1
DOCMDP_FORM_FILL = 2
DOCMDP_FORM_FILL_AND_ANNOTATE = 3

#: How deep the AcroForm field tree is walked while hunting for signatures.
_MAX_DEPTH = 64
#: ``/P`` when the file grants everything.
_ALL_PERMISSIONS = -1


def _signed32(value: int) -> int:
    """Reinterpret ``value`` as a signed 32-bit integer.

    Producers write ``/P`` either way round -- ``-3904`` and ``4294963392`` are the same
    word -- and the bit tests only agree if both are normalized first.
    """
    value = int(value) & 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def has_permission(permissions: int, bit: int) -> bool:
    """True when 1-based permission ``bit`` is granted by the ``/P`` word.

    Args:
        permissions: The ``/P`` value, signed or unsigned.
        bit: The specification's bit number, counting from 1 at the least significant.
    """
    if bit < 1 or bit > 32:
        return False
    return bool((int(permissions) >> (bit - 1)) & 1)


# ---------------------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------------------
@dataclass
class SecurityState:
    """The decoded encryption and permission state of one document."""

    encrypted: bool = False
    authenticated: bool = True
    permissions: int = _ALL_PERMISSIONS
    can_print: bool = True
    can_modify: bool = True
    can_extract: bool = True
    can_annotate: bool = True
    can_fill_forms: bool = True
    can_assemble: bool = True
    revision: int = 0
    method: str = "none"
    # -- beyond the six headline rights, the two bits the spec breaks out separately --
    can_extract_accessibility: bool = True
    can_print_high_quality: bool = True
    version: int = 0
    is_owner: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "encrypted": self.encrypted,
            "authenticated": self.authenticated,
            "permissions": self.permissions,
            "can_print": self.can_print,
            "can_modify": self.can_modify,
            "can_extract": self.can_extract,
            "can_annotate": self.can_annotate,
            "can_fill_forms": self.can_fill_forms,
            "can_assemble": self.can_assemble,
            "can_extract_accessibility": self.can_extract_accessibility,
            "can_print_high_quality": self.can_print_high_quality,
            "revision": self.revision,
            "version": self.version,
            "method": self.method,
            "is_owner": self.is_owner,
        }

    def describe(self) -> str:
        """One-line human summary for the preflight block and QA findings."""
        if not self.encrypted:
            return "unencrypted"
        granted = [
            label
            for label, ok in (
                ("print", self.can_print),
                ("modify", self.can_modify),
                ("extract", self.can_extract),
                ("annotate", self.can_annotate),
                ("fill", self.can_fill_forms),
                ("assemble", self.can_assemble),
            )
            if ok
        ]
        return "encrypted %s R%d, %s, allows: %s" % (
            self.method,
            self.revision,
            "authenticated" if self.authenticated else "locked",
            ", ".join(granted) if granted else "nothing",
        )


def _encrypt_dict(doc: Document) -> Optional[PdfDict]:
    """Return the resolved ``/Encrypt`` dictionary, or ``None`` when unencrypted."""
    enc = getattr(doc.file, "encrypt_dict", None)
    if isinstance(enc, PdfStream):
        enc = enc.dict
    if isinstance(enc, PdfDict):
        return enc
    trailer = getattr(doc.file, "trailer", None)
    raw = trailer.get("Encrypt") if isinstance(trailer, dict) else None
    if raw is None or isinstance(raw, PdfNull):
        return None
    value = doc.resolve(raw)
    if isinstance(value, PdfStream):
        value = value.dict
    return value if isinstance(value, PdfDict) else None


def _method_name(doc: Document, enc: PdfDict, handler: Any) -> str:
    """Name the encryption algorithm, e.g. ``"AES-128"`` or ``"RC4-40"``."""
    version = int(enc.get_int("V", 0, doc) or 0)
    bits = int(enc.get_int("Length", 40, doc) or 40)
    if version >= 5:
        return "AES-256"
    if version == 4:
        cfm = None
        filters = doc.resolve(enc.get("CF"))
        stream_filter = enc.get_name("StmF", "Identity", doc)
        if isinstance(filters, PdfDict) and stream_filter:
            spec = doc.resolve(filters.get(stream_filter))
            if isinstance(spec, PdfStream):
                spec = spec.dict
            if isinstance(spec, PdfDict):
                cfm = spec.get_name("CFM", None, doc)
                bits = int(spec.get_int("Length", bits, doc) or bits)
        if bits <= 40:  # /Length inside a crypt filter is in bytes for some producers
            bits *= 8
        if cfm == "AESV2":
            return "AES-128"
        if cfm == "AESV3":
            return "AES-256"
        if cfm in (None, "Identity"):
            return "Identity"
        return "RC4-%d" % bits
    if version == 2:
        return "RC4-%d" % bits
    if version in (0, 1):
        return "RC4-40"
    handler_method = getattr(handler, "stream_method", None)
    return str(handler_method or "unknown")


def inspect(doc: Document) -> SecurityState:
    """Decode a document's encryption and permission state.

    An unencrypted document reports every permission granted and ``permissions == -1``,
    which is the ``/P`` word meaning "all bits set".

    For an encrypted document the bits are read straight out of ``/P``:
    bit 3 print, 4 modify, 5 extract, 6 annotate (and fill, at revision 2),
    9 fill form fields, 10 extract for accessibility, 11 assemble, 12 high-quality
    print.  Revision 2 files only define bits 3-6, so the revision-3 refinements fall
    back to their revision-2 parents rather than reading a bit the producer never set.

    A document that failed to authenticate reports every capability as ``False``: its
    permission word cannot be trusted (it is verified as part of the key derivation),
    and its contents cannot be read anyway.

    Args:
        doc: The open document.

    Returns:
        A :class:`SecurityState`.  Never raises.
    """
    enc = _encrypt_dict(doc)
    if enc is None:
        return SecurityState()

    handler = getattr(doc.file, "security", None)
    authenticated = bool(getattr(doc.file, "is_authenticated", False))
    is_owner = bool(getattr(handler, "is_owner", False))

    raw_permissions = enc.get_int("P", _ALL_PERMISSIONS, doc)
    permissions = _signed32(
        raw_permissions if isinstance(raw_permissions, int) else _ALL_PERMISSIONS
    )
    revision = int(enc.get_int("R", 0, doc) or 0)
    version = int(enc.get_int("V", 0, doc) or 0)

    state = SecurityState(
        encrypted=True,
        authenticated=authenticated,
        permissions=permissions,
        revision=revision,
        version=version,
        method=_method_name(doc, enc, handler),
        is_owner=is_owner,
    )

    if not authenticated:
        state.can_print = False
        state.can_modify = False
        state.can_extract = False
        state.can_annotate = False
        state.can_fill_forms = False
        state.can_assemble = False
        state.can_extract_accessibility = False
        state.can_print_high_quality = False
        return state

    if is_owner:
        # The owner password unlocks everything regardless of the permission word.
        return state

    modify = has_permission(permissions, PERM_MODIFY)
    annotate = has_permission(permissions, PERM_ANNOTATE)
    printing = has_permission(permissions, PERM_PRINT)
    state.can_print = printing
    state.can_modify = modify
    # Bit 5 only: bit 10 grants extraction *for accessibility*, which is a narrower
    # right, so folding it in here would let ZFP copy text the owner forbade copying.
    state.can_extract = has_permission(permissions, PERM_EXTRACT)
    state.can_annotate = annotate
    if revision >= 3:
        state.can_fill_forms = (
            has_permission(permissions, PERM_FILL_FORMS) or annotate or modify
        )
        state.can_assemble = has_permission(permissions, PERM_ASSEMBLE) or modify
        state.can_extract_accessibility = (
            has_permission(permissions, PERM_EXTRACT_ACCESSIBILITY) or state.can_extract
        )
        state.can_print_high_quality = (
            has_permission(permissions, PERM_PRINT_HIGH) and printing
        )
    else:
        # Revision 2 defines bits 3-6 only: filling is part of "annotate", assembling
        # is part of "modify", and the two refinements do not exist at all.
        state.can_fill_forms = annotate or modify
        state.can_assemble = modify
        state.can_extract_accessibility = state.can_extract
        state.can_print_high_quality = printing
    return state


#: Alias with a less stdlib-shadowing name, for callers that ``from ... import *``.
inspect_security = inspect


# ---------------------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------------------
@dataclass
class SignatureState:
    """One signature found in the document."""

    field_name: str = ""
    name: str = ""
    date: str = ""
    sub_filter: str = ""
    filter_name: str = ""
    has_docmdp: bool = False
    docmdp_permission: Optional[int] = None

    @property
    def certification(self) -> bool:
        """True when this is a *certification* signature (it carries ``/DocMDP``)."""
        return self.has_docmdp

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "field_name": self.field_name,
            "name": self.name,
            "date": self.date,
            "sub_filter": self.sub_filter,
            "filter_name": self.filter_name,
            "has_docmdp": self.has_docmdp,
            "docmdp_permission": self.docmdp_permission,
        }

    def describe(self) -> str:
        """One-line human summary."""
        who = self.name or "<unnamed>"
        where = self.field_name or "<unnamed field>"
        if self.has_docmdp:
            level = self.docmdp_permission
            lock = "DocMDP /P=%s" % ("?" if level is None else level)
        else:
            lock = "approval signature"
        return "%s signed by %s (%s, %s)" % (where, who, self.sub_filter or "?", lock)


def _as_list(value: Any) -> List[Any]:
    """Return ``value`` as a list; ``None``/null yields ``[]``, scalars are wrapped."""
    if value is None or isinstance(value, PdfNull):
        return []
    if isinstance(value, (PdfArray, list, tuple)):
        return list(value)
    return [value]


def _text(doc: Document, value: Any) -> str:
    """Render a PDF text-ish value as a plain string."""
    value = doc.resolve(value)
    if isinstance(value, PdfString):
        return value.text()
    if isinstance(value, PdfName):
        return value.value
    if isinstance(value, (bytes, bytearray)):
        return PdfString(bytes(value)).text()
    if isinstance(value, str):
        return value
    return ""


def _docmdp_from_signature(doc: Document, signature: PdfDict) -> Tuple[bool, Optional[int]]:
    """Read a signature dictionary's ``/DocMDP`` transform.

    Returns:
        ``(present, permission)``.  ``permission`` is the ``/TransformParams /P`` level,
        defaulting to :data:`DOCMDP_FORM_FILL` when the transform exists but omits it,
        which is the specification's own default.
    """
    for entry in _as_list(doc.resolve(signature.get("Reference"))):
        reference = doc.resolve(entry)
        if not isinstance(reference, PdfDict):
            continue
        if reference.get_name("TransformMethod", None, doc) != "DocMDP":
            continue
        params = doc.resolve(reference.get("TransformParams"))
        level: Optional[int] = DOCMDP_FORM_FILL
        if isinstance(params, PdfDict):
            declared = params.get_int("P", None, doc)
            if isinstance(declared, int):
                level = int(declared)
        return (True, level)
    return (False, None)


def _perms_docmdp(doc: Document) -> Tuple[Optional[PdfRef], Optional[PdfDict]]:
    """Return the catalog's ``/Perms /DocMDP`` signature as ``(reference, dictionary)``."""
    perms = doc.resolve(doc.catalog.get("Perms"))
    if not isinstance(perms, PdfDict):
        return (None, None)
    raw = perms.get("DocMDP")
    if raw is None or isinstance(raw, PdfNull):
        return (None, None)
    value = doc.resolve(raw)
    return (raw if isinstance(raw, PdfRef) else None, value if isinstance(value, PdfDict) else None)


def _signature_state(
    doc: Document,
    field_name: str,
    signature: PdfDict,
    perms_ref: Optional[PdfRef],
    value_ref: Optional[PdfRef],
    perms_dict: Optional[PdfDict],
) -> SignatureState:
    """Build a :class:`SignatureState` from a signature dictionary."""
    has_docmdp, level = _docmdp_from_signature(doc, signature)
    if not has_docmdp and perms_dict is not None:
        # The catalog certifies this signature even though its own /Reference array is
        # missing or unreadable: treat the certification as present at the spec default.
        same = (
            perms_ref is not None
            and value_ref is not None
            and perms_ref.num == value_ref.num
        ) or perms_dict is signature
        if same:
            has_docmdp, level = True, DOCMDP_FORM_FILL
    return SignatureState(
        field_name=field_name,
        name=_text(doc, signature.get("Name")),
        date=_text(doc, signature.get("M")),
        sub_filter=signature.get_name("SubFilter", "", doc) or "",
        filter_name=signature.get_name("Filter", "", doc) or "",
        has_docmdp=has_docmdp,
        docmdp_permission=level,
    )


def _walk_signature_fields(
    doc: Document,
    node: Any,
    parent_name: str,
    inherited_type: Optional[str],
    depth: int,
    visited: set,
    out: List[Tuple[str, PdfDict, Optional[PdfRef]]],
) -> None:
    """Collect ``(qualified name, signature dict, /V reference)`` for signed /Sig fields."""
    if depth > _MAX_DEPTH:
        return
    ref = node if isinstance(node, PdfRef) else None
    if ref is not None:
        if ref.num in visited:
            return
        visited.add(ref.num)
    field = doc.resolve(node)
    if not isinstance(field, PdfDict):
        return

    field_type = field.get_name("FT", None, doc) or inherited_type
    partial = _text(doc, field.get("T"))
    name = "%s.%s" % (parent_name, partial) if parent_name and partial else (partial or parent_name)

    kids = _as_list(doc.resolve(field.get("Kids")))
    if kids:
        for kid in kids:
            _walk_signature_fields(doc, kid, name, field_type, depth + 1, visited, out)

    if field_type != "Sig":
        return
    raw_value = field.get("V")
    value = doc.resolve(raw_value)
    if isinstance(value, PdfDict):
        out.append((name, value, raw_value if isinstance(raw_value, PdfRef) else None))


def inspect_signatures(doc: Document) -> List[SignatureState]:
    """Report every signature in the document, in field-tree order.

    Both places a signature hides are searched: the AcroForm field tree (walking
    ``/Kids`` so a signature nested under a parent field is still found, and inheriting
    ``/FT`` the way the specification requires) and the page annotations, which is where
    a signature widget lives when the producer forgot to list it in ``/AcroForm
    /Fields``.  Duplicates are collapsed by field name and signature identity.

    Args:
        doc: The open document.

    Returns:
        One :class:`SignatureState` per signed ``/FT /Sig`` field.  Never raises.
    """
    perms_ref, perms_dict = _perms_docmdp(doc)
    found: List[Tuple[str, PdfDict, Optional[PdfRef]]] = []
    visited: set = set()

    try:
        acroform = doc.acroform()
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("%s: /AcroForm unreadable: %s", doc.document_id, exc)
        acroform = None
    if isinstance(acroform, PdfDict):
        for entry in _as_list(doc.resolve(acroform.get("Fields"))):
            _walk_signature_fields(doc, entry, "", None, 0, visited, found)

    try:
        pages = doc.pages
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("%s: pages unreadable: %s", doc.document_id, exc)
        pages = []
    for page in pages:
        for annot in page.annotations():
            if annot.get_name("Subtype", None, doc) != "Widget":
                continue
            if annot.get_name("FT", None, doc) != "Sig":
                continue
            raw_value = annot.get("V")
            value = doc.resolve(raw_value)
            if not isinstance(value, PdfDict):
                continue
            name = _text(doc, annot.get("T"))
            found.append((name, value, raw_value if isinstance(raw_value, PdfRef) else None))

    states: List[SignatureState] = []
    seen: set = set()
    for name, signature, value_ref in found:
        key = (name, value_ref.num if value_ref is not None else id(signature))
        if key in seen:
            continue
        seen.add(key)
        states.append(
            _signature_state(doc, name, signature, perms_ref, value_ref, perms_dict)
        )
    return states


# ---------------------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------------------
def can_add_form_fields(doc: Document) -> Tuple[bool, str]:
    """Decide whether ZFP may add AcroForm fields to this document, and say why.

    This is the gate ``SecurityGateAgent`` calls before anything is written.  The rules,
    in order:

    1. **Encrypted and not authenticated** -> refused.  Without the key nothing can even
       be read back, let alone written.
    2. **Encrypted, authenticated, but neither "modify" nor "fill form fields"
       granted** -> refused.  The permission word is the document owner's stated intent
       and ZFP honours it.
    3. **Signed** -> allowed *only* as an incremental update, and *only* when a
       ``/DocMDP`` transform declares ``/P >= 2`` (form filling and signing are
       permitted changes).  ``/P 1`` and a signature with no readable ``/DocMDP``
       declaration are both refused: without a permitted-changes declaration there is no
       evidence the edit is legal, and a broken signature is worse than a missing field.
    4. Otherwise -> allowed.

    An encrypted *and* signed document must clear both gates; the encryption gate is
    evaluated first because it is the one that makes writing impossible rather than
    merely impermissible.

    Args:
        doc: The open document.

    Returns:
        ``(allowed, reason)``.  ``reason`` is always populated -- on refusal it names
        the rule that fired, and on approval it names the constraint the writer must
        respect (incremental-only, for encrypted and signed documents).
    """
    try:
        state = inspect(doc)
    except Exception as exc:  # pragma: no cover - inspect is already defensive
        return (False, "security state could not be read: %s" % (exc,))

    if state.encrypted and not state.authenticated:
        return (
            False,
            "encrypted (%s) and no password authenticated: the document cannot be "
            "modified" % state.method,
        )
    if state.encrypted and not (state.can_fill_forms or state.can_modify):
        return (
            False,
            "encrypted (%s, /P=%d): permissions grant neither modification nor form "
            "filling" % (state.method, state.permissions),
        )

    try:
        signatures = inspect_signatures(doc)
    except Exception as exc:  # pragma: no cover - inspect_signatures is defensive
        return (False, "signature state could not be read: %s" % (exc,))
    try:
        signed = bool(signatures) or doc.is_signed()
    except Exception as exc:  # pragma: no cover - defensive
        return (False, "signature state could not be read: %s" % (exc,))

    prefix = ""
    if state.encrypted:
        prefix = "encrypted (%s, /P=%d) but form filling is permitted; " % (
            state.method,
            state.permissions,
        )

    if not signed:
        if prefix:
            return (True, prefix + "unsigned: form fields may be added")
        return (True, "unencrypted and unsigned: form fields may be added")

    if not signatures:
        return (
            False,
            prefix
            + "signed (the catalog declares /Perms /DocMDP) but no signature dictionary "
            "could be read, so no permitted-changes level is known: refusing to write",
        )

    certified = [
        s for s in signatures if s.has_docmdp and s.docmdp_permission is not None
    ]
    if not certified:
        return (
            False,
            prefix
            + "signed by %d signature(s) with no /DocMDP transform: the document "
            "declares no permitted changes, so adding fields would invalidate it"
            % len(signatures),
        )

    level = min(int(s.docmdp_permission or 0) for s in certified)
    if level < DOCMDP_FORM_FILL:
        return (
            False,
            prefix
            + "signed with /DocMDP /P=%d (no changes permitted): refusing to add form "
            "fields" % level,
        )
    return (
        True,
        prefix
        + "signed with /DocMDP /P=%d (form fill and signing are permitted changes): "
        "form fields may be added as an incremental update only, leaving the signed "
        "byte range intact" % level,
    )
