"""Unit tests for :mod:`zfp.pdfio.crypt` and the encryption path through ``PdfFile``.

The block ciphers themselves live in :mod:`zfp.vault.cipher`.  Tests that genuinely
need them skip when that module is absent; the published RC4 vectors always run,
falling back to a local implementation so this file is never silently vacuous.
"""

from __future__ import annotations

import binascii
import hashlib
import time
import unittest

from zfp.core.errors import EncryptedDocumentError
from zfp.pdfio.crypt import (
    PASSWORD_PAD,
    StandardSecurityHandler,
    compute_legacy_key,
    compute_owner_rc4_key,
    compute_user_entry,
    hash_r6,
    pad_password,
)
from zfp.pdfio.objects import PdfDict, PdfName, PdfString
from zfp.pdfio.parser import PdfFile

try:  # the cipher primitives are written by a sibling module and may land later
    from zfp.vault import cipher as CIPHER
except Exception:  # noqa: BLE001 - absence is the case under test
    CIPHER = None


def _available(*names: str) -> bool:
    return CIPHER is not None and all(callable(getattr(CIPHER, n, None)) for n in names)


HAVE_RC4 = _available("rc4")
HAVE_AES = _available("rc4", "aes_ecb_encrypt", "aes_cbc_decrypt")


def _local_rc4(key: bytes, data: bytes) -> bytes:
    """A minimal RC4 used only when :mod:`zfp.vault.cipher` is not importable yet."""
    state = list(range(256))
    j = 0
    for i in range(256):
        j = (j + state[i] + key[i % len(key)]) & 0xFF
        state[i], state[j] = state[j], state[i]
    i = j = 0
    out = bytearray()
    for byte in data:
        i = (i + 1) & 0xFF
        j = (j + state[i]) & 0xFF
        state[i], state[j] = state[j], state[i]
        out.append(byte ^ state[(state[i] + state[j]) & 0xFF])
    return bytes(out)


def rc4(key: bytes, data: bytes) -> bytes:
    """RC4 from the real cipher module when present, else the local fallback."""
    if HAVE_RC4:
        return bytes(CIPHER.rc4(bytes(key), bytes(data)))
    return _local_rc4(bytes(key), bytes(data))


def aes_cbc_encrypt(key: bytes, iv: bytes, data: bytes, pad: bool = True) -> bytes:
    """CBC encryption built on ``aes_ecb_encrypt``; fixture helper only."""
    if pad:
        fill = 16 - len(data) % 16
        data = bytes(data) + bytes([fill]) * fill
    previous = bytes(iv)
    out = bytearray()
    for start in range(0, len(data), 16):
        block = bytes(a ^ b for a, b in zip(data[start : start + 16], previous))
        previous = bytes(CIPHER.aes_ecb_encrypt(bytes(key), block))
        out += previous
    return bytes(out)


def r6_hash_is_affordable() -> bool:
    """Probe whether the revision-6 hash loop fits the suite's time budget.

    Algorithm 2.B runs at least 64 rounds of AES over 2 KiB, so a naive pure-python
    block cipher makes it a multi-second operation.  This measures the real one instead
    of guessing.
    """
    if not HAVE_AES:
        return False
    start = time.perf_counter()
    for _ in range(32):
        CIPHER.aes_ecb_encrypt(bytes(16), bytes(16))
    elapsed = time.perf_counter() - start
    if elapsed <= 0:
        return True
    per_block = elapsed / 32
    # Two hashes of >= 64 rounds x 128 blocks each, kept under two seconds.
    return per_block * 64 * 128 * 2 < 2.0


HAVE_FAST_R6 = r6_hash_is_affordable()

MD5 = lambda data: hashlib.md5(data).digest()  # noqa: E731 - a local alias, not an API
DOC_ID = b"0123456789abcdef"


# --------------------------------------------------------------------------------------
# Fixture construction (the encryption side of the algorithms)
# --------------------------------------------------------------------------------------


def make_owner_entry(owner_password: bytes, user_password: bytes, revision: int, key_bytes: int) -> bytes:
    """Algorithm 3: build a ``/O`` entry."""
    digest = MD5(pad_password(owner_password))
    if revision >= 3:
        for _ in range(50):
            digest = MD5(digest)
    key = digest[: 5 if revision == 2 else key_bytes]
    value = pad_password(user_password)
    if revision == 2:
        return rc4(key, value)
    value = rc4(key, value)
    for index in range(1, 20):
        value = rc4(bytes(byte ^ index for byte in key), value)
    return value


def make_file_key(
    user_password: bytes,
    owner_entry: bytes,
    permissions: int,
    revision: int,
    key_bytes: int,
    doc_id: bytes = DOC_ID,
    encrypt_metadata: bool = True,
) -> bytes:
    """Algorithm 2, written straight from the spec, independent of the module."""
    material = pad_password(user_password) + owner_entry
    material += (permissions & 0xFFFFFFFF).to_bytes(4, "little") + doc_id
    if revision >= 4 and not encrypt_metadata:
        material += b"\xff\xff\xff\xff"
    digest = MD5(material)
    if revision >= 3:
        for _ in range(50):
            digest = MD5(digest[:key_bytes])
    return digest[:key_bytes]


def make_user_entry(key: bytes, revision: int, doc_id: bytes = DOC_ID) -> bytes:
    """Algorithms 4 and 5: build a ``/U`` entry."""
    if revision == 2:
        return rc4(key, PASSWORD_PAD)
    value = rc4(key, MD5(PASSWORD_PAD + doc_id))
    for index in range(1, 20):
        value = rc4(bytes(byte ^ index for byte in key), value)
    return value + b"\x00" * 16


def make_object_key(key: bytes, num: int, gen: int, aes: bool = False) -> bytes:
    """Algorithm 1: the per-object key."""
    material = key + (num & 0xFFFFFF).to_bytes(3, "little") + (gen & 0xFFFF).to_bytes(2, "little")
    if aes:
        material += b"sAlT"
    return MD5(material)[: min(len(key) + 5, 16)]


def legacy_encrypt_dict(revision: int, version: int, key_bytes: int, permissions: int = -1,
                        owner_password: bytes = b"", user_password: bytes = b"",
                        aes: bool = False) -> tuple[PdfDict, bytes]:
    """Build a complete ``/Encrypt`` dictionary plus the file key it implies."""
    owner_entry = make_owner_entry(owner_password, user_password, revision, key_bytes)
    key = make_file_key(user_password, owner_entry, permissions, revision, key_bytes)
    user_entry = make_user_entry(key, revision)
    enc = PdfDict(
        {
            "Filter": PdfName("Standard"),
            "V": version,
            "R": revision,
            "O": PdfString(owner_entry),
            "U": PdfString(user_entry),
            "P": permissions,
            "Length": key_bytes * 8,
        }
    )
    if aes:
        enc["CF"] = PdfDict({"StdCF": PdfDict({"CFM": PdfName("AESV2"), "Length": key_bytes})})
        enc["StmF"] = PdfName("StdCF")
        enc["StrF"] = PdfName("StdCF")
    return enc, key


def encrypted_pdf(content: bytes, title: bytes) -> tuple[bytes, bytes]:
    """A complete RC4-40 encrypted one-page PDF with an empty user password."""
    owner_entry = make_owner_entry(b"", b"", 2, 5)
    key = make_file_key(b"", owner_entry, -1, 2, 5)
    user_entry = make_user_entry(key, 2)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
        None,  # filled in below
        b"<< /Filter /Standard /V 1 /R 2 /Length 40 /P -1 /O <%s> /U <%s> >>"
        % (
            binascii.hexlify(owner_entry).upper(),
            binascii.hexlify(user_entry).upper(),
        ),
        b"<< /Title <%s> >>"
        % binascii.hexlify(rc4(make_object_key(key, 6, 0), title)).upper(),
    ]
    body = rc4(make_object_key(key, 4, 0), content)
    objects[3] = b"<< /Length %d >>\nstream\n" % len(body) + body + b"\nendstream"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for number, obj in enumerate(objects, start=1):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + obj + b"\nendobj\n"
    table = len(out)
    count = len(objects) + 1
    out += b"xref\n0 %d\n0000000000 65535 f \n" % count
    for number in range(1, count):
        out += b"%010d 00000 n \n" % offsets[number]
    hex_id = binascii.hexlify(DOC_ID).upper()
    out += (
        b"trailer\n<< /Size %d /Root 1 0 R /Info 6 0 R /Encrypt 5 0 R "
        b"/ID [<%s> <%s>] >>\nstartxref\n%d\n%%%%EOF\n" % (count, hex_id, hex_id, table)
    )
    return bytes(out), user_entry


# --------------------------------------------------------------------------------------
# Published cipher vectors
# --------------------------------------------------------------------------------------


class Rc4VectorTests(unittest.TestCase):
    """The classic RC4 test vectors; these run whether or not the cipher module exists."""

    def test_key_plaintext_vector(self):
        self.assertEqual(
            binascii.hexlify(rc4(b"Key", b"Plaintext")).upper(),
            b"BBF316E8D940AF0AD3",
        )

    def test_wiki_pedia_vector(self):
        self.assertEqual(
            binascii.hexlify(rc4(b"Wiki", b"pedia")).upper(), b"1021BF0420"
        )

    def test_secret_attack_vector(self):
        self.assertEqual(
            binascii.hexlify(rc4(b"Secret", b"Attack at dawn")).upper(),
            b"45A01F645FC35B383552544B9BF5",
        )

    def test_rc4_is_involutive(self):
        self.assertEqual(rc4(b"k3y", rc4(b"k3y", b"round trip")), b"round trip")


# --------------------------------------------------------------------------------------
# Handler configuration (no cipher required)
# --------------------------------------------------------------------------------------


class HandlerConfigTests(unittest.TestCase):
    def test_pad_password(self):
        self.assertEqual(pad_password(b""), PASSWORD_PAD)
        self.assertEqual(pad_password(b"ab")[:2], b"ab")
        self.assertEqual(len(pad_password(b"x" * 100)), 32)
        self.assertEqual(pad_password(b"ab")[2:], PASSWORD_PAD[:30])

    def test_unsupported_security_handler(self):
        with self.assertRaises(EncryptedDocumentError):
            StandardSecurityHandler(PdfDict({"Filter": PdfName("Ubiquitous")}))

    def test_unsupported_crypt_filter_method(self):
        enc = PdfDict(
            {
                "Filter": PdfName("Standard"),
                "V": 4,
                "R": 4,
                "CF": PdfDict({"C": PdfDict({"CFM": PdfName("Bogus")})}),
                "StmF": PdfName("C"),
                "StrF": PdfName("C"),
            }
        )
        with self.assertRaises(EncryptedDocumentError):
            StandardSecurityHandler(enc)

    def test_identity_filters_are_the_v4_default(self):
        enc = PdfDict({"Filter": PdfName("Standard"), "V": 4, "R": 4, "Length": 128})
        handler = StandardSecurityHandler(enc)
        self.assertEqual(handler.stream_method, "Identity")
        self.assertEqual(handler.string_method, "Identity")

    def test_crypt_filter_length_in_bytes_is_promoted_to_bits(self):
        enc = PdfDict(
            {
                "Filter": PdfName("Standard"),
                "V": 4,
                "R": 4,
                "Length": 40,
                "CF": PdfDict({"S": PdfDict({"CFM": PdfName("AESV2"), "Length": 16})}),
                "StmF": PdfName("S"),
                "StrF": PdfName("S"),
            }
        )
        handler = StandardSecurityHandler(enc)
        self.assertEqual(handler.key_bytes, 16)
        self.assertEqual(handler.stream_method, "AESV2")

    def test_v1_forces_forty_bit_keys(self):
        enc = PdfDict({"Filter": PdfName("Standard"), "V": 1, "R": 2, "Length": 128})
        self.assertEqual(StandardSecurityHandler(enc).key_bytes, 5)

    def test_decrypt_without_authentication_raises(self):
        enc = PdfDict({"Filter": PdfName("Standard"), "V": 1, "R": 2, "Length": 40})
        handler = StandardSecurityHandler(enc)
        self.assertFalse(handler.authenticated)
        with self.assertRaises(EncryptedDocumentError):
            handler.decrypt(b"ciphertext", 1, 0, True)
        self.assertEqual(handler.decrypt(b"", 1, 0, True), b"")

    def test_permission_bits(self):
        enc = PdfDict({"Filter": PdfName("Standard"), "V": 2, "R": 3, "P": -1})
        handler = StandardSecurityHandler(enc)
        self.assertTrue(handler.permission(4))
        self.assertTrue(handler.permission(9))
        self.assertFalse(handler.permission(0))
        self.assertFalse(handler.permission(33))
        enc = PdfDict({"Filter": PdfName("Standard"), "V": 2, "R": 3, "P": 0})
        handler = StandardSecurityHandler(enc)
        self.assertFalse(handler.permission(4))
        self.assertFalse(handler.can_modify)

    def test_encrypt_metadata_default(self):
        enc = PdfDict({"Filter": PdfName("Standard"), "V": 4, "R": 4})
        self.assertTrue(StandardSecurityHandler(enc).encrypt_metadata)
        enc["EncryptMetadata"] = False
        self.assertFalse(StandardSecurityHandler(enc).encrypt_metadata)

    def test_r5_hash_is_plain_sha256(self):
        self.assertEqual(
            hash_r6(b"pw", b"01234567", b"", 5),
            hashlib.sha256(b"pw" + b"01234567").digest(),
        )
        self.assertEqual(
            hash_r6(b"pw", b"01234567", b"extra", 5),
            hashlib.sha256(b"pw" + b"01234567" + b"extra").digest(),
        )


# --------------------------------------------------------------------------------------
# Revisions 2-4
# --------------------------------------------------------------------------------------


@unittest.skipUnless(HAVE_RC4, "zfp.vault.cipher.rc4 is not available yet")
class LegacyHandlerTests(unittest.TestCase):
    def test_r2_rc4_40_empty_user_password(self):
        enc, key = legacy_encrypt_dict(2, 1, 5)
        handler = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "")
        self.assertTrue(handler.authenticated)
        self.assertFalse(handler.is_owner)
        self.assertEqual(handler.key, key)
        self.assertEqual(handler.key_bytes, 5)

    def test_r2_decrypts_a_stream(self):
        enc, key = legacy_encrypt_dict(2, 1, 5)
        handler = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "")
        plaintext = b"BT /F1 12 Tf (hello) Tj ET"
        ciphertext = rc4(make_object_key(key, 3, 0), plaintext)
        self.assertEqual(handler.decrypt(ciphertext, 3, 0, False), plaintext)
        self.assertEqual(handler.decrypt(ciphertext, 3, 0, True), plaintext)

    def test_object_key_depends_on_object_number(self):
        enc, key = legacy_encrypt_dict(2, 1, 5)
        handler = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "")
        self.assertEqual(handler.object_key(3, 0, False), make_object_key(key, 3, 0))
        self.assertNotEqual(handler.object_key(3, 0, False), handler.object_key(4, 0, False))
        self.assertNotEqual(handler.object_key(3, 0, False), handler.object_key(3, 1, False))
        self.assertNotEqual(handler.object_key(3, 0, False), handler.object_key(3, 0, True))

    def test_r3_rc4_128_empty_user_password(self):
        enc, key = legacy_encrypt_dict(3, 2, 16, permissions=-3904, owner_password=b"owner")
        handler = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "")
        self.assertTrue(handler.authenticated)
        self.assertEqual(handler.key, key)
        self.assertEqual(handler.key_bytes, 16)

    def test_r3_owner_password_recovers_the_user_key(self):
        enc, key = legacy_encrypt_dict(
            3, 2, 16, permissions=-3904, owner_password=b"owner", user_password=b"letmein"
        )
        empty = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "")
        self.assertFalse(empty.authenticated)
        user = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "letmein")
        self.assertTrue(user.authenticated)
        self.assertFalse(user.is_owner)
        self.assertEqual(user.key, key)
        owner = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "owner")
        self.assertTrue(owner.authenticated)
        self.assertTrue(owner.is_owner)
        self.assertEqual(owner.key, key)

    def test_wrong_password_is_rejected(self):
        enc, _ = legacy_encrypt_dict(
            3, 2, 16, owner_password=b"owner", user_password=b"letmein"
        )
        handler = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "nope")
        self.assertFalse(handler.authenticated)
        with self.assertRaises(EncryptedDocumentError):
            handler.decrypt(b"anything", 1, 0, False)

    def test_authenticate_can_be_retried(self):
        enc, key = legacy_encrypt_dict(
            3, 2, 16, owner_password=b"owner", user_password=b"letmein"
        )
        handler = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "")
        self.assertFalse(handler.authenticated)
        self.assertTrue(handler.authenticate("letmein"))
        self.assertEqual(handler.key, key)

    def test_helper_functions_agree_with_the_fixtures(self):
        owner_entry = make_owner_entry(b"o", b"u", 3, 16)
        key = make_file_key(b"u", owner_entry, -1, 3, 16)
        self.assertEqual(
            compute_legacy_key(b"u", owner_entry, -1, DOC_ID, 3, 16, True), key
        )
        self.assertEqual(compute_user_entry(key, DOC_ID, 3), make_user_entry(key, 3)[:16])
        self.assertEqual(len(compute_owner_rc4_key(b"o", 3, 16)), 16)
        self.assertEqual(len(compute_owner_rc4_key(b"o", 2, 16)), 5)

    def test_describe_mentions_the_scheme(self):
        enc, _ = legacy_encrypt_dict(3, 2, 16)
        handler = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "")
        text = handler.describe()
        self.assertIn("R3", text)
        self.assertIn("128-bit", text)
        self.assertIn("StandardSecurityHandler", repr(handler))


@unittest.skipUnless(HAVE_AES, "zfp.vault.cipher AES primitives are not available yet")
class AesV2HandlerTests(unittest.TestCase):
    def test_aesv2_round_trip(self):
        enc, key = legacy_encrypt_dict(4, 4, 16, aes=True)
        handler = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "")
        self.assertTrue(handler.authenticated)
        self.assertEqual(handler.stream_method, "AESV2")
        plaintext = b"the quick brown fox jumps over the lazy dog"
        iv = bytes(range(16))
        ciphertext = iv + aes_cbc_encrypt(make_object_key(key, 7, 0, True), iv, plaintext)
        self.assertEqual(handler.decrypt(ciphertext, 7, 0, False), plaintext)

    def test_identity_string_filter_is_honoured(self):
        enc, _ = legacy_encrypt_dict(4, 4, 16, aes=True)
        enc["StrF"] = PdfName("Identity")
        handler = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "")
        self.assertEqual(handler.string_method, "Identity")
        self.assertEqual(handler.decrypt(b"as written", 7, 0, True), b"as written")

    def test_aes_payload_shorter_than_an_iv_is_empty(self):
        enc, _ = legacy_encrypt_dict(4, 4, 16, aes=True)
        handler = StandardSecurityHandler.from_encrypt_dict(enc, DOC_ID, "")
        self.assertEqual(handler.decrypt(b"short", 7, 0, False), b"")


# --------------------------------------------------------------------------------------
# Revisions 5 and 6 (AES-256)
# --------------------------------------------------------------------------------------


FILE_KEY_256 = bytes((index * 7 + 3) & 0xFF for index in range(32))
VALIDATION_SALT = b"VSALT!!!"
KEY_SALT = b"KSALT!!!"
OWNER_VALIDATION_SALT = b"OVSALT!!"
OWNER_KEY_SALT = b"OKSALT!!"


def modern_encrypt_dict(revision: int, owner_password: bytes = b"ownerpw") -> PdfDict:
    """Build an AES-256 ``/Encrypt`` dictionary whose empty user password works."""
    user_entry = hash_r6(b"", VALIDATION_SALT, b"", revision) + VALIDATION_SALT + KEY_SALT
    user_key_entry = aes_cbc_encrypt(
        hash_r6(b"", KEY_SALT, b"", revision), b"\x00" * 16, FILE_KEY_256, pad=False
    )
    owner_entry = (
        hash_r6(owner_password, OWNER_VALIDATION_SALT, user_entry[:48], revision)
        + OWNER_VALIDATION_SALT
        + OWNER_KEY_SALT
    )
    owner_key_entry = aes_cbc_encrypt(
        hash_r6(owner_password, OWNER_KEY_SALT, user_entry[:48], revision),
        b"\x00" * 16,
        FILE_KEY_256,
        pad=False,
    )
    return PdfDict(
        {
            "Filter": PdfName("Standard"),
            "V": 5,
            "R": revision,
            "Length": 256,
            "P": -1028,
            "U": PdfString(user_entry),
            "UE": PdfString(user_key_entry),
            "O": PdfString(owner_entry),
            "OE": PdfString(owner_key_entry),
            "CF": PdfDict({"StdCF": PdfDict({"CFM": PdfName("AESV3"), "Length": 32})}),
            "StmF": PdfName("StdCF"),
            "StrF": PdfName("StdCF"),
        }
    )


@unittest.skipUnless(HAVE_AES, "zfp.vault.cipher AES primitives are not available yet")
class AesV3RevisionFiveTests(unittest.TestCase):
    def setUp(self):
        self.enc = modern_encrypt_dict(5)

    def test_empty_user_password_unwraps_the_file_key(self):
        handler = StandardSecurityHandler.from_encrypt_dict(self.enc, b"", "")
        self.assertTrue(handler.authenticated)
        self.assertFalse(handler.is_owner)
        self.assertEqual(handler.key, FILE_KEY_256)
        self.assertEqual(handler.key_bytes, 32)

    def test_owner_password_unwraps_the_same_key(self):
        handler = StandardSecurityHandler.from_encrypt_dict(self.enc, b"", "ownerpw")
        self.assertTrue(handler.authenticated)
        self.assertTrue(handler.is_owner)
        self.assertEqual(handler.key, FILE_KEY_256)

    def test_wrong_password_is_rejected(self):
        handler = StandardSecurityHandler.from_encrypt_dict(self.enc, b"", "wrong")
        self.assertFalse(handler.authenticated)

    def test_aesv3_uses_the_file_key_directly(self):
        handler = StandardSecurityHandler.from_encrypt_dict(self.enc, b"", "")
        plaintext = b"AES-256 encrypted content stream"
        iv = bytes(range(16, 32))
        ciphertext = iv + aes_cbc_encrypt(FILE_KEY_256, iv, plaintext)
        # The object and generation numbers are irrelevant for AESV3.
        self.assertEqual(handler.decrypt(ciphertext, 3, 0, False), plaintext)
        self.assertEqual(handler.decrypt(ciphertext, 99, 7, True), plaintext)

    def test_unicode_password_is_utf8_encoded(self):
        user_entry = hash_r6("pä".encode(), VALIDATION_SALT, b"", 5) + VALIDATION_SALT + KEY_SALT
        user_key_entry = aes_cbc_encrypt(
            hash_r6("pä".encode(), KEY_SALT, b"", 5), b"\x00" * 16, FILE_KEY_256, pad=False
        )
        enc = PdfDict(self.enc)
        enc["U"] = PdfString(user_entry)
        enc["UE"] = PdfString(user_key_entry)
        enc["O"] = PdfString(b"")
        handler = StandardSecurityHandler.from_encrypt_dict(enc, b"", "pä")
        self.assertTrue(handler.authenticated)
        self.assertEqual(handler.key, FILE_KEY_256)


@unittest.skipUnless(
    HAVE_FAST_R6,
    "the revision-6 hardened hash is too slow with the available AES implementation",
)
class AesV3RevisionSixTests(unittest.TestCase):
    def test_hardened_hash_authenticates_and_unwraps(self):
        enc = modern_encrypt_dict(6)
        handler = StandardSecurityHandler.from_encrypt_dict(enc, b"", "")
        self.assertTrue(handler.authenticated)
        self.assertEqual(handler.key, FILE_KEY_256)

    def test_hardened_hash_differs_from_the_plain_sha256_form(self):
        self.assertNotEqual(
            hash_r6(b"", VALIDATION_SALT, b"", 6),
            hash_r6(b"", VALIDATION_SALT, b"", 5),
        )
        self.assertEqual(len(hash_r6(b"", VALIDATION_SALT, b"", 6)), 32)


# --------------------------------------------------------------------------------------
# The encryption path through PdfFile
# --------------------------------------------------------------------------------------


CONTENT = b"BT /F1 12 Tf 72 720 Td (Secret) Tj ET"
TITLE = b"Confidential Report"


@unittest.skipUnless(HAVE_RC4, "zfp.vault.cipher.rc4 is not available yet")
class EncryptedDocumentTests(unittest.TestCase):
    def test_empty_user_password_is_tried_automatically(self):
        data, _ = encrypted_pdf(CONTENT, TITLE)
        pdf = PdfFile.load(data)
        self.assertTrue(pdf.is_encrypted)
        self.assertTrue(pdf.is_authenticated)
        self.assertFalse(pdf.rebuilt)
        self.assertEqual(pdf.warnings, [])

    def test_streams_and_strings_are_decrypted_transparently(self):
        data, _ = encrypted_pdf(CONTENT, TITLE)
        pdf = PdfFile.load(data)
        page = pdf.page_dicts()[0]
        self.assertEqual(pdf.resolve(page["Contents"]).decoded(pdf), CONTENT)
        info = pdf.resolve(pdf.trailer["Info"])
        self.assertEqual(info["Title"].text(), TITLE.decode("ascii"))

    def test_the_encrypt_dictionary_itself_is_never_decrypted(self):
        data, user_entry = encrypted_pdf(CONTENT, TITLE)
        pdf = PdfFile.load(data)
        enc = pdf.resolve(pdf.trailer["Encrypt"])
        self.assertEqual(enc["U"].raw, user_entry)

    def test_structure_only_mode_when_the_password_is_unknown(self):
        data, user_entry = encrypted_pdf(CONTENT, TITLE)
        broken = data.replace(
            b"/U <%s>" % binascii.hexlify(user_entry).upper(),
            b"/U <%s>" % binascii.hexlify(bytes(32)).upper(),
        )
        pdf = PdfFile.load(broken)
        self.assertTrue(pdf.is_encrypted)
        self.assertFalse(pdf.is_authenticated)
        self.assertTrue(any("password" in warning for warning in pdf.warnings))
        # Structure stays walkable ...
        self.assertEqual(pdf.object_numbers(), [1, 2, 3, 4, 5, 6])
        page = pdf.page_dicts()[0]
        self.assertEqual(page.get_name("Type"), "Page")
        # ... but content does not come out.
        with self.assertRaises(EncryptedDocumentError):
            pdf.resolve(page["Contents"]).decoded(pdf)
        with self.assertRaises(EncryptedDocumentError):
            pdf.resolve(pdf.trailer["Info"])["Title"].text()

    def test_unencrypted_documents_report_authenticated(self):
        data = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /MediaBox [0 0 1 1] >>\nendobj\n"
            b"trailer\n<< /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
        )
        pdf = PdfFile.load(data)
        self.assertFalse(pdf.is_encrypted)
        self.assertTrue(pdf.is_authenticated)
        self.assertTrue(pdf.authenticate("anything"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
