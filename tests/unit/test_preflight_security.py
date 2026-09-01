"""Unit tests for :mod:`zfp.preflight.security`."""

from __future__ import annotations

import hashlib
import json
import unittest

from zfp.core.serde import dumps
from zfp.core.types import DocumentClass
from zfp.pdfio.crypt import PASSWORD_PAD, pad_password, rc4
from zfp.pdfio.document import Document
from zfp.pdfio.objects import PdfArray, PdfDict, PdfName, PdfRef, PdfString
from zfp.pdfio.writer import build_document
from zfp.preflight.classifier import profile_document, route
from zfp.preflight.security import (
    DOCMDP_FORM_FILL,
    PERM_EXTRACT,
    PERM_EXTRACT_ACCESSIBILITY,
    PERM_FILL_FORMS,
    PERM_MODIFY,
    PERM_PRINT,
    SecurityState,
    SignatureState,
    can_add_form_fields,
    has_permission,
    inspect,
    inspect_signatures,
)

DOC_ID = b"0123456789abcdef"


# --------------------------------------------------------------------------------------
# Encryption fixtures -- the algorithms written out independently of the module
# --------------------------------------------------------------------------------------
def _md5(data: bytes) -> bytes:
    return hashlib.md5(data).digest()


def _owner_entry(revision: int, key_bytes: int) -> bytes:
    """Algorithm 3 with empty owner and user passwords."""
    digest = _md5(PASSWORD_PAD)
    if revision >= 3:
        for _ in range(50):
            digest = _md5(digest)
    key = digest[: 5 if revision == 2 else key_bytes]
    value = pad_password(b"")
    if revision == 2:
        return rc4(key, value)
    value = rc4(key, value)
    for index in range(1, 20):
        value = rc4(bytes(byte ^ index for byte in key), value)
    return value


def _file_key(owner_entry: bytes, permissions: int, revision: int, key_bytes: int) -> bytes:
    """Algorithm 2 with an empty user password."""
    material = PASSWORD_PAD + owner_entry
    material += (permissions & 0xFFFFFFFF).to_bytes(4, "little") + DOC_ID
    digest = _md5(material)
    if revision >= 3:
        for _ in range(50):
            digest = _md5(digest[:key_bytes])
    return digest[:key_bytes]


def _user_entry(key: bytes, revision: int) -> bytes:
    """Algorithms 4 and 5."""
    if revision == 2:
        return rc4(key, PASSWORD_PAD)
    value = rc4(key, _md5(PASSWORD_PAD + DOC_ID))
    for index in range(1, 20):
        value = rc4(bytes(byte ^ index for byte in key), value)
    return value + b"\x00" * 16


def encrypt_dict(
    permissions: int,
    *,
    revision: int = 2,
    version: int = 1,
    key_bytes: int = 5,
    valid: bool = True,
    aes: bool = False,
) -> PdfDict:
    """A ``/Encrypt`` dictionary that authenticates with the empty user password.

    With ``valid=False`` the ``/U`` entry is deliberately wrong, which is what a
    document protected by a real user password looks like to a reader that does not
    have it.
    """
    owner = _owner_entry(revision, key_bytes)
    key = _file_key(owner, permissions, revision, key_bytes)
    user = _user_entry(key, revision) if valid else b"\x00" * 32
    enc = PdfDict(
        {
            "Filter": PdfName("Standard"),
            "V": version,
            "R": revision,
            "O": PdfString(owner),
            "U": PdfString(user),
            "P": permissions,
            "Length": key_bytes * 8,
        }
    )
    if aes:
        enc["CF"] = PdfDict(
            {"StdCF": PdfDict({"CFM": PdfName("AESV2"), "Length": key_bytes})}
        )
        enc["StmF"] = PdfName("StdCF")
        enc["StrF"] = PdfName("StdCF")
    return enc


def encrypted_pdf(enc: PdfDict) -> Document:
    """A one-page document carrying ``enc`` as its ``/Encrypt`` dictionary."""
    objects = {
        1: PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2)}),
        2: PdfDict({"Type": PdfName("Pages"), "Kids": PdfArray([PdfRef(3)]), "Count": 1}),
        3: PdfDict(
            {
                "Type": PdfName("Page"),
                "Parent": PdfRef(2),
                "MediaBox": PdfArray([0, 0, 612, 792]),
            }
        ),
        4: enc,
    }
    data = build_document(
        objects,
        PdfRef(1),
        extra_trailer={
            "Encrypt": PdfRef(4),
            "ID": PdfArray([PdfString(DOC_ID), PdfString(DOC_ID)]),
        },
    )
    return Document.open(data)


# --------------------------------------------------------------------------------------
# Signature fixtures
# --------------------------------------------------------------------------------------
def signed_pdf(
    *,
    docmdp: "int | None" = None,
    perms_entry: bool = False,
    nested: bool = False,
    widget_only: bool = False,
    reference_missing: bool = False,
) -> Document:
    """A one-page document with one signed ``/FT /Sig`` field.

    Args:
        docmdp: ``/DocMDP`` ``/P`` level to declare, or ``None`` for an approval
            signature with no transform at all.
        perms_entry: Also point the catalog's ``/Perms /DocMDP`` at the signature.
        nested: Put the signature field under a parent field, so ``/FT`` has to be
            inherited and the name has to be qualified.
        widget_only: Attach the signature to the page's ``/Annots`` without listing it
            in ``/AcroForm /Fields``.
        reference_missing: Declare ``/Perms /DocMDP`` but give the signature no
            ``/Reference`` array.
    """
    signature = PdfDict(
        {
            "Type": PdfName("Sig"),
            "Filter": PdfName("Adobe.PPKLite"),
            "SubFilter": PdfName("adbe.pkcs7.detached"),
            "Name": PdfString.from_text("Ada Lovelace"),
            "M": PdfString.from_text("D:20240117120000Z"),
            "ByteRange": PdfArray([0, 100, 200, 300]),
        }
    )
    if docmdp is not None and not reference_missing:
        signature["Reference"] = PdfArray(
            [
                PdfDict(
                    {
                        "Type": PdfName("SigRef"),
                        "TransformMethod": PdfName("DocMDP"),
                        "TransformParams": PdfDict(
                            {"Type": PdfName("TransformParams"), "P": docmdp, "V": PdfName("1.2")}
                        ),
                    }
                )
            ]
        )

    field = PdfDict(
        {
            "Type": PdfName("Annot"),
            "Subtype": PdfName("Widget"),
            "FT": PdfName("Sig"),
            "T": PdfString.from_text("signature_1"),
            "Rect": PdfArray([72, 72, 300, 140]),
            "V": PdfRef(7),
            "P": PdfRef(3),
        }
    )
    catalog = PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2), "AcroForm": PdfRef(5)})
    page = PdfDict(
        {
            "Type": PdfName("Page"),
            "Parent": PdfRef(2),
            "MediaBox": PdfArray([0, 0, 612, 792]),
            "Annots": PdfArray([PdfRef(6)]),
        }
    )
    acroform = PdfDict({"Fields": PdfArray([PdfRef(6)]), "SigFlags": 3})
    objects = {
        1: catalog,
        2: PdfDict({"Type": PdfName("Pages"), "Kids": PdfArray([PdfRef(3)]), "Count": 1}),
        3: page,
        5: acroform,
        6: field,
        7: signature,
    }
    if nested:
        # The widget kid keeps its own /T (so the name has to be qualified) but not
        # its /FT, which now has to be inherited from the parent field.
        del field["FT"]
        objects[8] = PdfDict(
            {
                "FT": PdfName("Sig"),
                "T": PdfString.from_text("approvals"),
                "Kids": PdfArray([PdfRef(6)]),
            }
        )
        field["Parent"] = PdfRef(8)
        acroform["Fields"] = PdfArray([PdfRef(8)])
    if widget_only:
        acroform["Fields"] = PdfArray()
    if perms_entry or reference_missing:
        catalog["Perms"] = PdfDict({"DocMDP": PdfRef(7)})
    return Document.open(build_document(objects, PdfRef(1)))


# --------------------------------------------------------------------------------------
# Permission decoding
# --------------------------------------------------------------------------------------
class PermissionBitTests(unittest.TestCase):
    def test_bits_are_one_indexed(self):
        self.assertTrue(has_permission(0b100, PERM_PRINT))
        self.assertFalse(has_permission(0b011, PERM_PRINT))
        self.assertFalse(has_permission(-1, 0))
        self.assertFalse(has_permission(-1, 33))
        self.assertTrue(has_permission(-1, PERM_FILL_FORMS))

    def test_unencrypted_document_grants_everything(self):
        state = inspect(Document.from_pages_blank(1))
        self.assertFalse(state.encrypted)
        self.assertTrue(state.authenticated)
        self.assertEqual(state.permissions, -1)
        self.assertTrue(state.can_print)
        self.assertTrue(state.can_modify)
        self.assertTrue(state.can_extract)
        self.assertTrue(state.can_annotate)
        self.assertTrue(state.can_fill_forms)
        self.assertTrue(state.can_assemble)
        self.assertEqual(state.method, "none")
        self.assertEqual(state.describe(), "unencrypted")

    def test_known_restrictive_permission_word(self):
        # -3904 == 0xFFFFF0C0: every optional bit cleared, reserved bits set.
        doc = encrypted_pdf(encrypt_dict(-3904, revision=3, version=2, key_bytes=16))
        state = inspect(doc)
        self.assertTrue(state.encrypted)
        self.assertTrue(state.authenticated)
        self.assertEqual(state.permissions, -3904)
        self.assertEqual(state.revision, 3)
        self.assertEqual(state.method, "RC4-128")
        self.assertFalse(state.can_print)
        self.assertFalse(state.can_modify)
        self.assertFalse(state.can_extract)
        self.assertFalse(state.can_annotate)
        self.assertFalse(state.can_fill_forms)
        self.assertFalse(state.can_assemble)
        self.assertFalse(state.can_extract_accessibility)
        self.assertFalse(state.can_print_high_quality)
        self.assertIn("allows: nothing", state.describe())

    def test_unsigned_permission_word_is_normalised(self):
        doc = encrypted_pdf(encrypt_dict(0xFFFFF0C0, revision=3, version=2, key_bytes=16))
        self.assertEqual(inspect(doc).permissions, -3904)

    def test_fill_forms_only(self):
        permissions = -3904 | (1 << (PERM_FILL_FORMS - 1))
        doc = encrypted_pdf(encrypt_dict(permissions, revision=3, version=2, key_bytes=16))
        state = inspect(doc)
        self.assertTrue(state.can_fill_forms)
        self.assertFalse(state.can_modify)
        self.assertFalse(state.can_print)

    def test_modify_implies_fill_and_assemble(self):
        permissions = -3904 | (1 << (PERM_MODIFY - 1))
        doc = encrypted_pdf(encrypt_dict(permissions, revision=3, version=2, key_bytes=16))
        state = inspect(doc)
        self.assertTrue(state.can_modify)
        self.assertTrue(state.can_fill_forms)
        self.assertTrue(state.can_assemble)

    def test_revision_two_derives_fill_from_the_annotate_bit(self):
        permissions = -3904 | (1 << 5)  # bit 6, annotate/fill
        doc = encrypted_pdf(encrypt_dict(permissions))
        state = inspect(doc)
        self.assertEqual(state.revision, 2)
        self.assertEqual(state.method, "RC4-40")
        self.assertTrue(state.can_annotate)
        self.assertTrue(state.can_fill_forms)
        self.assertFalse(state.can_assemble)

    def test_accessibility_extraction_is_not_plain_extraction(self):
        permissions = -3904 | (1 << (PERM_EXTRACT_ACCESSIBILITY - 1))
        doc = encrypted_pdf(encrypt_dict(permissions, revision=3, version=2, key_bytes=16))
        state = inspect(doc)
        self.assertFalse(state.can_extract)
        self.assertTrue(state.can_extract_accessibility)

    def test_plain_extraction_implies_accessible_extraction(self):
        permissions = -3904 | (1 << (PERM_EXTRACT - 1))
        doc = encrypted_pdf(encrypt_dict(permissions, revision=3, version=2, key_bytes=16))
        state = inspect(doc)
        self.assertTrue(state.can_extract)
        self.assertTrue(state.can_extract_accessibility)

    def test_locked_document_reports_nothing_granted(self):
        doc = encrypted_pdf(encrypt_dict(-1, valid=False))
        state = inspect(doc)
        self.assertTrue(state.encrypted)
        self.assertFalse(state.authenticated)
        self.assertFalse(state.can_modify)
        self.assertFalse(state.can_fill_forms)
        self.assertIn("locked", state.describe())

    def test_aes_method_is_named(self):
        doc = encrypted_pdf(
            encrypt_dict(-1, revision=4, version=4, key_bytes=16, aes=True)
        )
        self.assertEqual(inspect(doc).method, "AES-128")

    def test_state_serialises(self):
        payload = SecurityState().as_dict()
        json.loads(dumps(payload))
        self.assertEqual(payload["permissions"], -1)


# --------------------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------------------
class SignatureTests(unittest.TestCase):
    def test_unsigned_document_has_no_signatures(self):
        self.assertEqual(inspect_signatures(Document.from_pages_blank(1)), [])

    def test_signature_fields_are_reported(self):
        signatures = inspect_signatures(signed_pdf())
        self.assertEqual(len(signatures), 1)
        signature = signatures[0]
        self.assertEqual(signature.field_name, "signature_1")
        self.assertEqual(signature.name, "Ada Lovelace")
        self.assertEqual(signature.date, "D:20240117120000Z")
        self.assertEqual(signature.sub_filter, "adbe.pkcs7.detached")
        self.assertEqual(signature.filter_name, "Adobe.PPKLite")
        self.assertFalse(signature.has_docmdp)
        self.assertIsNone(signature.docmdp_permission)
        self.assertFalse(signature.certification)

    def test_docmdp_level_is_read(self):
        signature = inspect_signatures(signed_pdf(docmdp=2))[0]
        self.assertTrue(signature.has_docmdp)
        self.assertTrue(signature.certification)
        self.assertEqual(signature.docmdp_permission, 2)
        self.assertIn("DocMDP /P=2", signature.describe())

    def test_nested_signature_field_inherits_type_and_name(self):
        signatures = inspect_signatures(signed_pdf(docmdp=3, nested=True))
        self.assertEqual(len(signatures), 1)
        self.assertEqual(signatures[0].field_name, "approvals.signature_1")
        self.assertEqual(signatures[0].docmdp_permission, 3)

    def test_signature_widget_missing_from_fields_is_still_found(self):
        signatures = inspect_signatures(signed_pdf(widget_only=True))
        self.assertEqual(len(signatures), 1)
        self.assertEqual(signatures[0].field_name, "signature_1")

    def test_field_and_widget_are_not_double_counted(self):
        self.assertEqual(len(inspect_signatures(signed_pdf(docmdp=2))), 1)

    def test_perms_docmdp_without_a_reference_array_defaults_to_form_fill(self):
        signature = inspect_signatures(signed_pdf(reference_missing=True))[0]
        self.assertTrue(signature.has_docmdp)
        self.assertEqual(signature.docmdp_permission, DOCMDP_FORM_FILL)

    def test_signature_state_serialises(self):
        payload = SignatureState(field_name="s", name="A").as_dict()
        json.loads(dumps(payload))
        self.assertEqual(payload["field_name"], "s")


# --------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------
class CanAddFormFieldsTests(unittest.TestCase):
    def test_plain_document_is_allowed(self):
        allowed, reason = can_add_form_fields(Document.from_pages_blank(1))
        self.assertTrue(allowed)
        self.assertIn("unencrypted and unsigned", reason)

    def test_locked_document_is_refused(self):
        allowed, reason = can_add_form_fields(encrypted_pdf(encrypt_dict(-1, valid=False)))
        self.assertFalse(allowed)
        self.assertIn("no password", reason)

    def test_encrypted_without_fill_or_modify_is_refused(self):
        doc = encrypted_pdf(encrypt_dict(-3904, revision=3, version=2, key_bytes=16))
        allowed, reason = can_add_form_fields(doc)
        self.assertFalse(allowed)
        self.assertIn("neither modification nor form filling", reason)
        self.assertIn("/P=-3904", reason)

    def test_encrypted_with_fill_permission_is_allowed(self):
        permissions = -3904 | (1 << (PERM_FILL_FORMS - 1))
        doc = encrypted_pdf(encrypt_dict(permissions, revision=3, version=2, key_bytes=16))
        allowed, reason = can_add_form_fields(doc)
        self.assertTrue(allowed)
        self.assertIn("encrypted", reason)
        self.assertIn("form fields may be added", reason)

    def test_signed_without_docmdp_is_refused(self):
        doc = signed_pdf()
        self.assertTrue(doc.is_signed())
        allowed, reason = can_add_form_fields(doc)
        self.assertFalse(allowed)
        self.assertIn("no /DocMDP transform", reason)

    def test_signed_with_docmdp_one_is_refused(self):
        allowed, reason = can_add_form_fields(signed_pdf(docmdp=1))
        self.assertFalse(allowed)
        self.assertIn("/DocMDP /P=1", reason)
        self.assertIn("no changes permitted", reason)

    def test_signed_with_docmdp_two_is_allowed_incrementally(self):
        allowed, reason = can_add_form_fields(signed_pdf(docmdp=2, perms_entry=True))
        self.assertTrue(allowed)
        self.assertIn("/DocMDP /P=2", reason)
        self.assertIn("incremental update only", reason)

    def test_signed_with_docmdp_three_is_allowed(self):
        allowed, reason = can_add_form_fields(signed_pdf(docmdp=3))
        self.assertTrue(allowed)
        self.assertIn("/DocMDP /P=3", reason)

    def test_the_strictest_signature_wins(self):
        doc = signed_pdf(docmdp=3)
        # A second, stricter certification signature on the same document.
        acroform = doc.ensure_acroform()
        strict = doc.writer.add_object(
            PdfDict(
                {
                    "Type": PdfName("Sig"),
                    "SubFilter": PdfName("adbe.pkcs7.detached"),
                    "Reference": PdfArray(
                        [
                            PdfDict(
                                {
                                    "TransformMethod": PdfName("DocMDP"),
                                    "TransformParams": PdfDict({"P": 1}),
                                }
                            )
                        ]
                    ),
                }
            )
        )
        field = doc.writer.add_object(
            PdfDict(
                {
                    "FT": PdfName("Sig"),
                    "T": PdfString.from_text("signature_2"),
                    "V": strict,
                }
            )
        )
        acroform["Fields"].append(field)
        self.assertEqual(len(inspect_signatures(doc)), 2)
        allowed, reason = can_add_form_fields(doc)
        self.assertFalse(allowed)
        self.assertIn("/P=1", reason)

    def test_certification_without_a_readable_signature_is_refused(self):
        objects = {
            1: PdfDict(
                {
                    "Type": PdfName("Catalog"),
                    "Pages": PdfRef(2),
                    "Perms": PdfDict({"DocMDP": PdfRef(9)}),
                }
            ),
            2: PdfDict({"Type": PdfName("Pages"), "Kids": PdfArray([PdfRef(3)]), "Count": 1}),
            3: PdfDict(
                {
                    "Type": PdfName("Page"),
                    "Parent": PdfRef(2),
                    "MediaBox": PdfArray([0, 0, 612, 792]),
                }
            ),
        }
        doc = Document.open(build_document(objects, PdfRef(1)))
        self.assertTrue(doc.is_signed())
        self.assertEqual(inspect_signatures(doc), [])
        allowed, reason = can_add_form_fields(doc)
        self.assertFalse(allowed)
        self.assertIn("no signature dictionary", reason)

    def test_a_refusal_always_carries_a_reason(self):
        for doc in (
            encrypted_pdf(encrypt_dict(-1, valid=False)),
            encrypted_pdf(encrypt_dict(-3904, revision=3, version=2, key_bytes=16)),
            signed_pdf(),
            signed_pdf(docmdp=1),
        ):
            allowed, reason = can_add_form_fields(doc)
            self.assertFalse(allowed)
            self.assertTrue(reason.strip())


# --------------------------------------------------------------------------------------
# The gate and the router must never disagree about the same file
# --------------------------------------------------------------------------------------
class RoutingConsistencyTests(unittest.TestCase):
    def test_locked_document_routes_as_encrypted(self):
        doc = encrypted_pdf(encrypt_dict(-1, valid=False))
        profile = profile_document(doc)
        self.assertTrue(profile.encrypted)
        self.assertFalse(profile.can_modify)
        self.assertIs(profile.doc_class, DocumentClass.ENCRYPTED)
        self.assertIs(route(profile), DocumentClass.ENCRYPTED)
        self.assertIn(
            "encrypted: structure-only access (no password accepted)", profile.warnings
        )
        self.assertFalse(can_add_form_fields(doc)[0])

    def test_deny_all_permissions_route_as_encrypted(self):
        doc = encrypted_pdf(encrypt_dict(-3904, revision=3, version=2, key_bytes=16))
        profile = profile_document(doc)
        self.assertFalse(profile.can_modify)
        self.assertIs(profile.doc_class, DocumentClass.ENCRYPTED)
        self.assertFalse(can_add_form_fields(doc)[0])
        self.assertTrue(
            any("permissions deny modification" in w for w in profile.warnings),
            profile.warnings,
        )

    def test_form_fill_permission_is_not_routed_as_encrypted(self):
        permissions = -3904 | (1 << (PERM_FILL_FORMS - 1))
        doc = encrypted_pdf(encrypt_dict(permissions, revision=3, version=2, key_bytes=16))
        profile = profile_document(doc)
        self.assertTrue(profile.encrypted)
        self.assertTrue(profile.can_modify)
        self.assertIsNot(profile.doc_class, DocumentClass.ENCRYPTED)
        self.assertTrue(can_add_form_fields(doc)[0])
        self.assertTrue(
            any("incremental-only edits permitted" in w for w in profile.warnings),
            profile.warnings,
        )

    def test_signed_document_routes_as_signed(self):
        doc = signed_pdf(docmdp=2)
        profile = profile_document(doc)
        self.assertTrue(profile.signed)
        self.assertIs(profile.doc_class, DocumentClass.SIGNED)
        self.assertIn("signed: incremental-only edits permitted", profile.warnings)
        allowed, reason = can_add_form_fields(doc)
        self.assertTrue(allowed)
        self.assertIn("incremental", reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
