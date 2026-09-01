"""The PDF standard security handler (``/Filter /Standard``).

Covers every revision that appears in practice:

===  ===  ==================================================================
R    V    Scheme
===  ===  ==================================================================
2    1    RC4, 40-bit key, MD5 key derivation
3    2    RC4, 40-128 bit key, 50-round MD5 key derivation
4    4    RC4 or AES-128 (``/CFM /V2`` or ``/CFM /AESV2``) via crypt filters
5    5    AES-256, plain SHA-256 password hash (Adobe extension level 3)
6    5    AES-256, hardened SHA-256/384/512 hash loop (ISO 32000-2)
===  ===  ==================================================================

The block ciphers themselves live in :mod:`zfp.vault.cipher` and are imported lazily
inside the functions that need them, so importing this module -- and therefore
:mod:`zfp.pdfio.parser` -- never depends on the cipher module being present.  When it
really is missing, an :class:`~zfp.core.errors.EncryptedDocumentError` explains what is
required rather than an ``ImportError`` leaking out.

Nothing here is a bypass: an owner password is accepted because it is a *credential*,
and permissions are reported, never silently overridden.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Tuple

from ..core.errors import EncryptedDocumentError
from ..core.logging import get_logger
from .objects import PdfDict, PdfName, PdfStream, PdfString

__all__ = [
    "PASSWORD_PAD",
    "StandardSecurityHandler",
]

logger = get_logger(__name__)

#: The 32-byte padding string from PDF 32000-1 Algorithm 2.
PASSWORD_PAD = bytes(
    (
        0x28, 0xBF, 0x4E, 0x5E, 0x4E, 0x75, 0x8A, 0x41,
        0x64, 0x00, 0x4E, 0x56, 0xFF, 0xFA, 0x01, 0x08,
        0x2E, 0x2E, 0x00, 0xB6, 0xD0, 0x68, 0x3E, 0x80,
        0x2F, 0x0C, 0xA9, 0xFE, 0x64, 0x53, 0x69, 0x7A,
    )
)

_ZERO_IV = b"\x00" * 16
_AES_SALT = b"sAlT"


# --------------------------------------------------------------------------------------
# Lazy access to the block ciphers
# --------------------------------------------------------------------------------------


def _cipher_module() -> Any:
    """Import :mod:`zfp.vault.cipher`, translating failure into a domain error."""
    try:
        from ..vault import cipher as module  # noqa: PLC0415 - deliberately lazy
    except Exception as exc:  # noqa: BLE001 - ImportError and anything it raises
        raise EncryptedDocumentError(
            "PDF decryption requires zfp.vault.cipher (rc4, aes_cbc_decrypt, "
            f"aes_ecb_encrypt), which could not be imported: {exc}"
        ) from exc
    return module


def _cipher_function(name: str) -> Callable[..., Any]:
    module = _cipher_module()
    function = getattr(module, name, None)
    if not callable(function):
        raise EncryptedDocumentError(
            f"PDF decryption requires zfp.vault.cipher.{name}, which is not available"
        )
    return function


def rc4(key: bytes, data: bytes) -> bytes:
    """RC4 keystream cipher, delegated to :mod:`zfp.vault.cipher`."""
    return bytes(_cipher_function("rc4")(bytes(key), bytes(data)))


def aes_cbc_decrypt(key: bytes, data: bytes) -> bytes:
    """AES-CBC decrypt where ``data`` starts with the IV and ends with PKCS#7 padding."""
    if len(data) <= 16:
        return b""
    return bytes(_cipher_function("aes_cbc_decrypt")(bytes(key), bytes(data)))


def aes_cbc_encrypt_raw(key: bytes, iv: bytes, data: bytes) -> bytes:
    """CBC mode over :func:`zfp.vault.cipher.aes_ecb_encrypt`, no padding added.

    Used only by the revision-6 hardened hash, whose input is always a multiple of the
    block size by construction (64 repetitions of the same chunk).
    """
    encrypt_block = _cipher_function("aes_ecb_encrypt")
    key = bytes(key)
    previous = bytes(iv)
    out = bytearray()
    for start in range(0, len(data) - len(data) % 16, 16):
        block = data[start : start + 16]
        mixed = bytes(a ^ b for a, b in zip(block, previous))
        previous = bytes(encrypt_block(key, mixed))
        out += previous
    return bytes(out)


def aes_cbc_decrypt_raw(key: bytes, iv: bytes, data: bytes) -> bytes:
    """AES-CBC decrypt with an explicit IV and **no** padding removal.

    Revision 5/6 wraps the 32-byte file key in ``/UE`` / ``/OE`` as exactly two AES
    blocks with a zero IV and no padding, so the padded helper cannot be used directly.
    A raw single-block decryptor is used when :mod:`zfp.vault.cipher` exposes one;
    otherwise the padded helper is called and any padding it stripped -- which by
    definition was real plaintext here -- is restored.
    """
    module = _cipher_module()
    block_decrypt = getattr(module, "aes_ecb_decrypt", None)
    if not callable(block_decrypt):
        block_decrypt = getattr(module, "aes_decrypt_block", None)
    key = bytes(key)
    data = bytes(data)
    if callable(block_decrypt):
        previous = bytes(iv)
        out = bytearray()
        for start in range(0, len(data) - len(data) % 16, 16):
            block = data[start : start + 16]
            plain = bytes(block_decrypt(key, block))
            out += bytes(a ^ b for a, b in zip(plain, previous))
            previous = block
        return bytes(out)
    result = bytes(_cipher_function("aes_cbc_decrypt")(key, bytes(iv) + data))
    missing = len(data) - len(result)
    if 1 <= missing <= 16:
        result += bytes([missing]) * missing
    return result[: len(data)]


# --------------------------------------------------------------------------------------
# Password and key derivation helpers
# --------------------------------------------------------------------------------------


def _as_bytes(value: Any) -> bytes:
    """Coerce a PDF string / name / raw value to bytes."""
    if value is None:
        return b""
    if isinstance(value, PdfString):
        return value.raw
    if isinstance(value, PdfName):
        return value.value.encode("latin-1", "replace")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, str):
        return value.encode("latin-1", "replace")
    return b""


def _legacy_password_bytes(password: Any) -> bytes:
    """Encode a password for revisions 2-4 (PDFDocEncoding, i.e. Latin-1 in practice)."""
    if isinstance(password, (bytes, bytearray)):
        return bytes(password)
    return str(password or "").encode("latin-1", "replace")


def _modern_password_bytes(password: Any) -> bytes:
    """Encode a password for revisions 5-6: UTF-8, truncated to 127 bytes."""
    if isinstance(password, (bytes, bytearray)):
        return bytes(password)[:127]
    return str(password or "").encode("utf-8", "replace")[:127]


def pad_password(password: bytes) -> bytes:
    """Algorithm 2 step (a): pad or truncate to exactly 32 bytes."""
    return (bytes(password) + PASSWORD_PAD)[:32]


def compute_legacy_key(
    padded_password: bytes,
    owner_entry: bytes,
    permissions: int,
    doc_id: bytes,
    revision: int,
    key_bytes: int,
    encrypt_metadata: bool,
) -> bytes:
    """Algorithm 2: derive the file encryption key for revisions 2-4."""
    digest_input = bytearray(pad_password(padded_password))
    digest_input += owner_entry[:32]
    digest_input += (int(permissions) & 0xFFFFFFFF).to_bytes(4, "little")
    digest_input += doc_id
    if revision >= 4 and not encrypt_metadata:
        digest_input += b"\xff\xff\xff\xff"
    digest = hashlib.md5(bytes(digest_input)).digest()
    if revision >= 3:
        for _ in range(50):
            digest = hashlib.md5(digest[:key_bytes]).digest()
    return digest[:key_bytes]


def compute_user_entry(key: bytes, doc_id: bytes, revision: int) -> bytes:
    """Algorithms 4 and 5: the ``/U`` value implied by a candidate file key."""
    if revision == 2:
        return rc4(key, PASSWORD_PAD)
    digest = hashlib.md5(PASSWORD_PAD + doc_id).digest()
    value = rc4(key, digest)
    for round_index in range(1, 20):
        value = rc4(bytes(byte ^ round_index for byte in key), value)
    return value


def compute_owner_rc4_key(password: bytes, revision: int, key_bytes: int) -> bytes:
    """Algorithm 3 steps (a)-(d): the RC4 key derived from the owner password."""
    digest = hashlib.md5(pad_password(password)).digest()
    if revision >= 3:
        for _ in range(50):
            digest = hashlib.md5(digest).digest()
    return digest[: 5 if revision == 2 else key_bytes]


def recover_user_password(owner_entry: bytes, rc4_key: bytes, revision: int) -> bytes:
    """Algorithm 7: undo ``/O`` to get the padded user password."""
    if revision == 2:
        return rc4(rc4_key, owner_entry)
    value = bytes(owner_entry)
    for round_index in range(19, -1, -1):
        value = rc4(bytes(byte ^ round_index for byte in rc4_key), value)
    return value


def hash_r6(password: bytes, salt: bytes, extra: bytes, revision: int) -> bytes:
    """ISO 32000-2 Algorithm 2.B (and its revision-5 SHA-256-only predecessor)."""
    digest = hashlib.sha256(password + salt + extra).digest()
    if revision <= 5:
        return digest
    rounds = 0
    while True:
        block = (password + digest + extra) * 64
        encrypted = aes_cbc_encrypt_raw(digest[:16], digest[16:32], block)
        selector = sum(encrypted[:16]) % 3
        if selector == 0:
            digest = hashlib.sha256(encrypted).digest()
        elif selector == 1:
            digest = hashlib.sha384(encrypted).digest()
        else:
            digest = hashlib.sha512(encrypted).digest()
        rounds += 1
        if rounds >= 64 and encrypted[-1] <= rounds - 32:
            break
    return digest[:32]


# --------------------------------------------------------------------------------------
# The handler
# --------------------------------------------------------------------------------------


class StandardSecurityHandler:
    """Decrypts strings and streams for a document using ``/Filter /Standard``.

    Build one with :meth:`from_encrypt_dict`; it authenticates immediately with the
    supplied password (the empty user password by default) and records the outcome in
    :attr:`authenticated`.  :meth:`decrypt` refuses to run until authentication has
    succeeded.
    """

    def __init__(self, enc: PdfDict, doc_id: bytes = b"") -> None:
        self.enc: PdfDict = enc if isinstance(enc, PdfDict) else PdfDict(enc or {})
        self.doc_id: bytes = _as_bytes(doc_id)

        filter_name = self.enc.get_name("Filter", "Standard")
        if filter_name not in (None, "Standard"):
            raise EncryptedDocumentError(
                f"unsupported security handler /{filter_name} "
                "(only /Standard is implemented)"
            )

        self.version: int = int(self.enc.get_int("V", 0) or 0)
        self.revision: int = int(self.enc.get_int("R", 2 if self.version < 2 else 3) or 2)
        permissions = self.enc.get_int("P", -1)
        self.permissions: int = int(-1 if permissions is None else permissions)
        self.owner_entry: bytes = _as_bytes(self.enc.get("O"))
        self.user_entry: bytes = _as_bytes(self.enc.get("U"))
        self.owner_key_entry: bytes = _as_bytes(self.enc.get("OE"))
        self.user_key_entry: bytes = _as_bytes(self.enc.get("UE"))
        self.perms_entry: bytes = _as_bytes(self.enc.get("Perms"))
        self.encrypt_metadata: bool = bool(self.enc.get_bool("EncryptMetadata", True))

        declared_bits = int(self.enc.get_int("Length", 40) or 40)
        self.stream_method, stream_bits = self._crypt_filter("StmF", declared_bits)
        self.string_method, string_bits = self._crypt_filter("StrF", declared_bits)
        if self.revision >= 5:
            self.key_bytes = 32
        else:
            bits = declared_bits
            if self.version >= 4:
                bits = stream_bits or string_bits or declared_bits
            elif self.version <= 1:
                bits = 40
            self.key_bytes = max(5, min(16, int(bits) // 8))

        self.key: bytes = b""
        self.authenticated: bool = False
        self.is_owner: bool = False

    # -- construction -----------------------------------------------------------------
    @staticmethod
    def from_encrypt_dict(
        enc: PdfDict, doc_id: Any = b"", password: str = ""
    ) -> StandardSecurityHandler:
        """Build a handler from an ``/Encrypt`` dictionary and the file's first ``/ID``.

        Authentication with ``password`` is attempted immediately; check
        :attr:`authenticated` for the result rather than expecting an exception.
        """
        if isinstance(enc, PdfStream):
            enc = enc.dict
        if not isinstance(enc, PdfDict):
            raise EncryptedDocumentError("/Encrypt is not a dictionary")
        handler = StandardSecurityHandler(enc, _as_bytes(doc_id))
        handler.authenticate(password)
        return handler

    def _crypt_filter(self, which: str, default_bits: int) -> Tuple[str, int]:
        """Resolve ``/StmF`` or ``/StrF`` to ``(method, key bits)``."""
        if self.version < 4:
            return ("V2", default_bits if self.version >= 2 else 40)
        name = self.enc.get_name(which, "Identity")
        if name in (None, "Identity"):
            return ("Identity", default_bits)
        filters = self.enc.get_dict("CF")
        spec = filters.get(name) if isinstance(filters, PdfDict) else None
        if isinstance(spec, PdfStream):
            spec = spec.dict
        if not isinstance(spec, PdfDict):
            return ("Identity", default_bits)
        method = spec.get_name("CFM", "None") or "None"
        bits = int(spec.get_int("Length", 0) or 0)
        if 0 < bits <= 40:
            bits *= 8  # /Length inside a crypt filter is documented in bytes
        if bits <= 0:
            bits = default_bits
        if method in ("None", "Identity"):
            return ("Identity", bits)
        if method not in ("V2", "AESV2", "AESV3"):
            raise EncryptedDocumentError(f"unsupported crypt filter method /{method}")
        return (method, bits)

    # -- authentication ---------------------------------------------------------------
    def authenticate(self, password: str = "") -> bool:
        """Try ``password`` as the user password, then as the owner password.

        Sets :attr:`key`, :attr:`authenticated` and :attr:`is_owner`, and returns
        whether the document can now be decrypted.
        """
        self.authenticated = False
        self.is_owner = False
        self.key = b""
        if self.revision >= 5:
            ok = self._authenticate_modern(password)
        else:
            ok = self._authenticate_legacy(password)
        self.authenticated = ok
        return ok

    def _authenticate_legacy(self, password: str) -> bool:
        raw = _legacy_password_bytes(password)
        key = compute_legacy_key(
            raw,
            self.owner_entry,
            self.permissions,
            self.doc_id,
            self.revision,
            self.key_bytes,
            self.encrypt_metadata,
        )
        if self._user_entry_matches(key):
            self.key = key
            return True
        rc4_key = compute_owner_rc4_key(raw, self.revision, self.key_bytes)
        recovered = recover_user_password(self.owner_entry, rc4_key, self.revision)
        key = compute_legacy_key(
            recovered,
            self.owner_entry,
            self.permissions,
            self.doc_id,
            self.revision,
            self.key_bytes,
            self.encrypt_metadata,
        )
        if self._user_entry_matches(key):
            self.key = key
            self.is_owner = True
            return True
        return False

    def _user_entry_matches(self, key: bytes) -> bool:
        if not self.user_entry:
            return False
        computed = compute_user_entry(key, self.doc_id, self.revision)
        if self.revision == 2:
            return computed[:32] == self.user_entry[:32]
        return computed[:16] == self.user_entry[:16]

    def _authenticate_modern(self, password: str) -> bool:
        raw = _modern_password_bytes(password)
        user = self.user_entry
        owner = self.owner_entry
        if len(user) >= 48:
            validation_salt = user[32:40]
            key_salt = user[40:48]
            if hash_r6(raw, validation_salt, b"", self.revision) == user[:32]:
                intermediate = hash_r6(raw, key_salt, b"", self.revision)
                self.key = aes_cbc_decrypt_raw(
                    intermediate, _ZERO_IV, self.user_key_entry
                )[:32]
                return len(self.key) == 32
        if len(owner) >= 48 and len(user) >= 48:
            validation_salt = owner[32:40]
            key_salt = owner[40:48]
            if hash_r6(raw, validation_salt, user[:48], self.revision) == owner[:32]:
                intermediate = hash_r6(raw, key_salt, user[:48], self.revision)
                self.key = aes_cbc_decrypt_raw(
                    intermediate, _ZERO_IV, self.owner_key_entry
                )[:32]
                self.is_owner = True
                return len(self.key) == 32
        return False

    # -- decryption -------------------------------------------------------------------
    def object_key(self, num: int, gen: int, aes: bool) -> bytes:
        """Algorithm 1: the per-object key for revisions 2-4."""
        material = bytearray(self.key)
        material += (int(num) & 0xFFFFFF).to_bytes(3, "little")
        material += (int(gen) & 0xFFFF).to_bytes(2, "little")
        if aes:
            material += _AES_SALT
        digest = hashlib.md5(bytes(material)).digest()
        return digest[: min(len(self.key) + 5, 16)]

    def decrypt(self, data: bytes, num: int, gen: int, is_string: bool) -> bytes:
        """Decrypt one string or stream belonging to object ``num`` generation ``gen``.

        Raises :class:`~zfp.core.errors.EncryptedDocumentError` when no valid password
        has been established, which is what keeps a locked document readable only as
        structure.
        """
        if not data:
            return b""
        if not self.authenticated:
            raise EncryptedDocumentError(
                "cannot decrypt: no valid password for this document"
            )
        method = self.string_method if is_string else self.stream_method
        if method == "Identity":
            return bytes(data)
        if method == "AESV3":
            return aes_cbc_decrypt(self.key, data)
        if method == "AESV2":
            return aes_cbc_decrypt(self.object_key(num, gen, True), data)
        return rc4(self.object_key(num, gen, False), data)

    # -- reporting --------------------------------------------------------------------
    def permission(self, bit: int) -> bool:
        """True when permission ``bit`` (1-based, as numbered in the spec) is granted."""
        if bit < 1 or bit > 32:
            return False
        return bool((self.permissions >> (bit - 1)) & 1)

    @property
    def can_modify(self) -> bool:
        """Permission bit 4 -- modify the document."""
        return self.is_owner or self.permission(4)

    @property
    def can_fill_forms(self) -> bool:
        """Permission bit 9 -- fill in form fields."""
        return self.is_owner or self.permission(9) or self.permission(4)

    def describe(self) -> str:
        """One-line human summary used in warnings and QA findings."""
        owner = " (owner)" if self.is_owner else ""
        return (
            f"Standard V{self.version} R{self.revision} "
            f"{self.stream_method}/{self.string_method} "
            f"{self.key_bytes * 8}-bit{owner}"
        )

    def __repr__(self) -> str:
        return (
            f"StandardSecurityHandler({self.describe()}, "
            f"authenticated={self.authenticated})"
        )
