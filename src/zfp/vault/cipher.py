"""Pure-Python cryptographic primitives.

This module is the single source of block/stream cipher code for ZFP.  It serves two
consumers:

* :mod:`zfp.vault.store` -- the encrypted profile vault, which uses
  ChaCha20-Poly1305 (RFC 8439) with a scrypt-derived key.
* :mod:`zfp.pdfio.crypt` -- PDF standard security, which needs RC4 and AES
  (ECB single blocks for the revision 6 hardened hash, CBC for /UE, /OE and for
  AESV2/AESV3 string and stream decryption).

Everything here is standard library only: no ``cryptography``, no ``pycryptodome``.
The AES tables are *generated* at import time from the GF(2**8) multiplicative
inverse and the AES affine transform -- nothing is hand-transcribed -- and the
resulting T-tables make the cipher fast enough to process hundreds of kilobytes
per second on CPython.

All lookup tables are immutable module-level tuples and the key-schedule cache is a
:func:`functools.lru_cache`, so every function in this module is thread safe.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import struct
from functools import lru_cache
from typing import List, Tuple

from ..core.errors import VaultError

__all__ = [
    "AES_BLOCK_SIZE",
    "CHACHA20_KEY_SIZE",
    "CHACHA20_NONCE_SIZE",
    "POLY1305_TAG_SIZE",
    "PBKDF2_ROUNDS",
    "rc4",
    "aes_ecb_encrypt",
    "aes_ecb_decrypt",
    "aes_cbc_encrypt",
    "aes_cbc_decrypt",
    "aes_cbc_decrypt_no_padding",
    "aes_ctr",
    "chacha20",
    "poly1305_mac",
    "chacha20_poly1305_encrypt",
    "chacha20_poly1305_decrypt",
    "derive_key",
    "kdf_name",
    "random_bytes",
    "constant_time_eq",
]


AES_BLOCK_SIZE = 16
CHACHA20_KEY_SIZE = 32
CHACHA20_NONCE_SIZE = 12
POLY1305_TAG_SIZE = 16
PBKDF2_ROUNDS = 200_000

_MASK32 = 0xFFFFFFFF
_AES_KEY_SIZES = (16, 24, 32)


# ======================================================================================
# AES -- table generation
# ======================================================================================


def _build_gf_tables() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Build the GF(2**8) exp/log tables for the AES field (modulus 0x11B, generator 3)."""
    exp: List[int] = [0] * 512
    log: List[int] = [0] * 256
    x = 1
    for i in range(255):
        exp[i] = x
        log[x] = i
        doubled = (x << 1) ^ (0x11B if x & 0x80 else 0)
        x = (doubled ^ x) & 0xFF  # x * 3 == xtime(x) ^ x
    for i in range(255, 512):
        exp[i] = exp[i - 255]
    return tuple(exp), tuple(log)


_GF_EXP, _GF_LOG = _build_gf_tables()


def _gf_mul(a: int, b: int) -> int:
    """Multiply two elements of GF(2**8) using the log tables."""
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[_GF_LOG[a] + _GF_LOG[b]]


def _build_sboxes() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Generate the AES S-box from the field inverse plus the affine transform."""
    sbox: List[int] = [0] * 256
    for value in range(256):
        inverse = 0 if value == 0 else _GF_EXP[255 - _GF_LOG[value]]
        result = inverse
        rotated = inverse
        for _ in range(4):
            rotated = ((rotated << 1) | (rotated >> 7)) & 0xFF
            result ^= rotated
        sbox[value] = result ^ 0x63
    inv_sbox: List[int] = [0] * 256
    for value, substituted in enumerate(sbox):
        inv_sbox[substituted] = value
    return tuple(sbox), tuple(inv_sbox)


_SBOX, _INV_SBOX = _build_sboxes()

# Single-factor multiplication tables used by the MixColumns family.
_MUL2 = tuple(_gf_mul(v, 2) for v in range(256))
_MUL3 = tuple(_gf_mul(v, 3) for v in range(256))
_MUL9 = tuple(_gf_mul(v, 9) for v in range(256))
_MUL11 = tuple(_gf_mul(v, 11) for v in range(256))
_MUL13 = tuple(_gf_mul(v, 13) for v in range(256))
_MUL14 = tuple(_gf_mul(v, 14) for v in range(256))


def _rotr32(word: int, bits: int) -> int:
    return ((word >> bits) | (word << (32 - bits))) & _MASK32


def _build_t_tables() -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]:
    """Precompute the combined SubBytes+ShiftRows+MixColumns tables (encrypt & decrypt)."""
    te0 = []
    td0 = []
    for value in range(256):
        s = _SBOX[value]
        te0.append((_MUL2[s] << 24) | (s << 16) | (s << 8) | _MUL3[s])
        v = _INV_SBOX[value]
        td0.append((_MUL14[v] << 24) | (_MUL9[v] << 16) | (_MUL13[v] << 8) | _MUL11[v])
    encrypt = tuple(tuple(_rotr32(w, 8 * i) for w in te0) for i in range(4))
    decrypt = tuple(tuple(_rotr32(w, 8 * i) for w in td0) for i in range(4))
    return encrypt, decrypt


_TE, _TD = _build_t_tables()
_TE0, _TE1, _TE2, _TE3 = _TE
_TD0, _TD1, _TD2, _TD3 = _TD


# ======================================================================================
# AES -- key schedule
# ======================================================================================


def _sub_word(word: int) -> int:
    sbox = _SBOX
    return (
        (sbox[word >> 24] << 24)
        | (sbox[(word >> 16) & 0xFF] << 16)
        | (sbox[(word >> 8) & 0xFF] << 8)
        | sbox[word & 0xFF]
    )


def _inv_mix_column(word: int) -> int:
    """Apply InvMixColumns to one packed 32-bit column (used for the decrypt schedule)."""
    a0 = word >> 24
    a1 = (word >> 16) & 0xFF
    a2 = (word >> 8) & 0xFF
    a3 = word & 0xFF
    b0 = _MUL14[a0] ^ _MUL11[a1] ^ _MUL13[a2] ^ _MUL9[a3]
    b1 = _MUL9[a0] ^ _MUL14[a1] ^ _MUL11[a2] ^ _MUL13[a3]
    b2 = _MUL13[a0] ^ _MUL9[a1] ^ _MUL14[a2] ^ _MUL11[a3]
    b3 = _MUL11[a0] ^ _MUL13[a1] ^ _MUL9[a2] ^ _MUL14[a3]
    return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3


@lru_cache(maxsize=64)
def _key_schedule(key: bytes) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    """Expand ``key`` into ``(encrypt_schedule, decrypt_schedule, rounds)``.

    The decryption schedule is the one required by the *equivalent inverse cipher*:
    the round-key groups are reversed and InvMixColumns is applied to every group
    except the first and the last.
    """
    nk = len(key) // 4
    rounds = nk + 6
    total = 4 * (rounds + 1)
    words: List[int] = list(struct.unpack(f">{nk}I", key))
    rcon = 1
    for index in range(nk, total):
        temp = words[index - 1]
        if index % nk == 0:
            temp = _sub_word(((temp << 8) | (temp >> 24)) & _MASK32) ^ (rcon << 24)
            rcon = ((rcon << 1) ^ 0x1B) & 0xFF if rcon & 0x80 else rcon << 1
        elif nk > 6 and index % nk == 4:
            temp = _sub_word(temp)
        words.append(words[index - nk] ^ temp)

    decrypt: List[int] = []
    for round_index in range(rounds, -1, -1):
        decrypt.extend(words[4 * round_index : 4 * round_index + 4])
    for index in range(4, 4 * rounds):
        decrypt[index] = _inv_mix_column(decrypt[index])
    return tuple(words), tuple(decrypt), rounds


def _check_aes_key(key: bytes) -> bytes:
    key = bytes(key)
    if len(key) not in _AES_KEY_SIZES:
        raise VaultError(f"AES key must be 16, 24 or 32 bytes, got {len(key)}")
    return key


# ======================================================================================
# AES -- core block transforms
# ======================================================================================


def _encrypt_words(
    rk: Tuple[int, ...], rounds: int, s0: int, s1: int, s2: int, s3: int
) -> Tuple[int, int, int, int]:
    t0_table, t1_table, t2_table, t3_table = _TE0, _TE1, _TE2, _TE3
    s0 ^= rk[0]
    s1 ^= rk[1]
    s2 ^= rk[2]
    s3 ^= rk[3]
    offset = 4
    for _ in range(rounds - 1):
        t0 = (
            t0_table[s0 >> 24]
            ^ t1_table[(s1 >> 16) & 0xFF]
            ^ t2_table[(s2 >> 8) & 0xFF]
            ^ t3_table[s3 & 0xFF]
            ^ rk[offset]
        )
        t1 = (
            t0_table[s1 >> 24]
            ^ t1_table[(s2 >> 16) & 0xFF]
            ^ t2_table[(s3 >> 8) & 0xFF]
            ^ t3_table[s0 & 0xFF]
            ^ rk[offset + 1]
        )
        t2 = (
            t0_table[s2 >> 24]
            ^ t1_table[(s3 >> 16) & 0xFF]
            ^ t2_table[(s0 >> 8) & 0xFF]
            ^ t3_table[s1 & 0xFF]
            ^ rk[offset + 2]
        )
        t3 = (
            t0_table[s3 >> 24]
            ^ t1_table[(s0 >> 16) & 0xFF]
            ^ t2_table[(s1 >> 8) & 0xFF]
            ^ t3_table[s2 & 0xFF]
            ^ rk[offset + 3]
        )
        s0, s1, s2, s3 = t0, t1, t2, t3
        offset += 4

    sbox = _SBOX
    r0 = (
        (sbox[s0 >> 24] << 24)
        | (sbox[(s1 >> 16) & 0xFF] << 16)
        | (sbox[(s2 >> 8) & 0xFF] << 8)
        | sbox[s3 & 0xFF]
    ) ^ rk[offset]
    r1 = (
        (sbox[s1 >> 24] << 24)
        | (sbox[(s2 >> 16) & 0xFF] << 16)
        | (sbox[(s3 >> 8) & 0xFF] << 8)
        | sbox[s0 & 0xFF]
    ) ^ rk[offset + 1]
    r2 = (
        (sbox[s2 >> 24] << 24)
        | (sbox[(s3 >> 16) & 0xFF] << 16)
        | (sbox[(s0 >> 8) & 0xFF] << 8)
        | sbox[s1 & 0xFF]
    ) ^ rk[offset + 2]
    r3 = (
        (sbox[s3 >> 24] << 24)
        | (sbox[(s0 >> 16) & 0xFF] << 16)
        | (sbox[(s1 >> 8) & 0xFF] << 8)
        | sbox[s2 & 0xFF]
    ) ^ rk[offset + 3]
    return r0, r1, r2, r3


def _decrypt_words(
    rk: Tuple[int, ...], rounds: int, s0: int, s1: int, s2: int, s3: int
) -> Tuple[int, int, int, int]:
    t0_table, t1_table, t2_table, t3_table = _TD0, _TD1, _TD2, _TD3
    s0 ^= rk[0]
    s1 ^= rk[1]
    s2 ^= rk[2]
    s3 ^= rk[3]
    offset = 4
    for _ in range(rounds - 1):
        t0 = (
            t0_table[s0 >> 24]
            ^ t1_table[(s3 >> 16) & 0xFF]
            ^ t2_table[(s2 >> 8) & 0xFF]
            ^ t3_table[s1 & 0xFF]
            ^ rk[offset]
        )
        t1 = (
            t0_table[s1 >> 24]
            ^ t1_table[(s0 >> 16) & 0xFF]
            ^ t2_table[(s3 >> 8) & 0xFF]
            ^ t3_table[s2 & 0xFF]
            ^ rk[offset + 1]
        )
        t2 = (
            t0_table[s2 >> 24]
            ^ t1_table[(s1 >> 16) & 0xFF]
            ^ t2_table[(s0 >> 8) & 0xFF]
            ^ t3_table[s3 & 0xFF]
            ^ rk[offset + 2]
        )
        t3 = (
            t0_table[s3 >> 24]
            ^ t1_table[(s2 >> 16) & 0xFF]
            ^ t2_table[(s1 >> 8) & 0xFF]
            ^ t3_table[s0 & 0xFF]
            ^ rk[offset + 3]
        )
        s0, s1, s2, s3 = t0, t1, t2, t3
        offset += 4

    inv = _INV_SBOX
    r0 = (
        (inv[s0 >> 24] << 24)
        | (inv[(s3 >> 16) & 0xFF] << 16)
        | (inv[(s2 >> 8) & 0xFF] << 8)
        | inv[s1 & 0xFF]
    ) ^ rk[offset]
    r1 = (
        (inv[s1 >> 24] << 24)
        | (inv[(s0 >> 16) & 0xFF] << 16)
        | (inv[(s3 >> 8) & 0xFF] << 8)
        | inv[s2 & 0xFF]
    ) ^ rk[offset + 1]
    r2 = (
        (inv[s2 >> 24] << 24)
        | (inv[(s1 >> 16) & 0xFF] << 16)
        | (inv[(s0 >> 8) & 0xFF] << 8)
        | inv[s3 & 0xFF]
    ) ^ rk[offset + 2]
    r3 = (
        (inv[s3 >> 24] << 24)
        | (inv[(s2 >> 16) & 0xFF] << 16)
        | (inv[(s1 >> 8) & 0xFF] << 8)
        | inv[s0 & 0xFF]
    ) ^ rk[offset + 3]
    return r0, r1, r2, r3


# ======================================================================================
# AES -- public API
# ======================================================================================


def aes_ecb_encrypt(key: bytes, block: bytes) -> bytes:
    """Encrypt exactly one 16-byte block with AES-128/192/256.

    Args:
        key: 16, 24 or 32 raw key bytes.
        block: exactly 16 plaintext bytes.

    Returns:
        The 16-byte ciphertext block.

    Raises:
        VaultError: if the key or block length is wrong.
    """
    key = _check_aes_key(key)
    block = bytes(block)
    if len(block) != AES_BLOCK_SIZE:
        raise VaultError(f"AES block must be exactly 16 bytes, got {len(block)}")
    schedule, _, rounds = _key_schedule(key)
    return struct.pack(">4I", *_encrypt_words(schedule, rounds, *struct.unpack(">4I", block)))


def aes_ecb_decrypt(key: bytes, block: bytes) -> bytes:
    """Decrypt exactly one 16-byte block with AES-128/192/256.

    Args:
        key: 16, 24 or 32 raw key bytes.
        block: exactly 16 ciphertext bytes.

    Returns:
        The 16-byte plaintext block.

    Raises:
        VaultError: if the key or block length is wrong.
    """
    key = _check_aes_key(key)
    block = bytes(block)
    if len(block) != AES_BLOCK_SIZE:
        raise VaultError(f"AES block must be exactly 16 bytes, got {len(block)}")
    _, schedule, rounds = _key_schedule(key)
    return struct.pack(">4I", *_decrypt_words(schedule, rounds, *struct.unpack(">4I", block)))


def _ecb_encrypt_buffer(key: bytes, data: bytes) -> bytes:
    """Encrypt a whole block-aligned buffer in ECB mode (internal helper)."""
    schedule, _, rounds = _key_schedule(key)
    out = bytearray(len(data))
    pack_into = struct.pack_into
    unpack_from = struct.unpack_from
    encrypt = _encrypt_words
    for offset in range(0, len(data), 16):
        s0, s1, s2, s3 = unpack_from(">4I", data, offset)
        pack_into(">4I", out, offset, *encrypt(schedule, rounds, s0, s1, s2, s3))
    return bytes(out)


def _ecb_decrypt_buffer(key: bytes, data: bytes) -> bytes:
    """Decrypt a whole block-aligned buffer in ECB mode (internal helper)."""
    _, schedule, rounds = _key_schedule(key)
    out = bytearray(len(data))
    pack_into = struct.pack_into
    unpack_from = struct.unpack_from
    decrypt = _decrypt_words
    for offset in range(0, len(data), 16):
        s0, s1, s2, s3 = unpack_from(">4I", data, offset)
        pack_into(">4I", out, offset, *decrypt(schedule, rounds, s0, s1, s2, s3))
    return bytes(out)


def _xor_bytes(left: bytes, right: bytes) -> bytes:
    """XOR two equal-length byte strings via a single big-integer operation."""
    size = len(left)
    if size == 0:
        return b""
    return (
        int.from_bytes(left, "big") ^ int.from_bytes(right, "big")
    ).to_bytes(size, "big")


def _pkcs7_pad(data: bytes) -> bytes:
    pad = AES_BLOCK_SIZE - (len(data) % AES_BLOCK_SIZE)
    return data + bytes([pad]) * pad


def _pkcs7_unpad(data: bytes) -> bytes:
    """Strip PKCS#7 padding, returning ``data`` untouched when the padding is invalid."""
    if not data:
        return data
    pad = data[-1]
    if 1 <= pad <= AES_BLOCK_SIZE and len(data) >= pad:
        if data[-pad:] == bytes([pad]) * pad:
            return data[: len(data) - pad]
    return data


def aes_cbc_encrypt(key: bytes, plaintext: bytes, iv: bytes | None = None) -> bytes:
    """Encrypt with AES-CBC and PKCS#7 padding.

    Args:
        key: 16, 24 or 32 raw key bytes.
        plaintext: any length, padded internally.
        iv: 16-byte initialisation vector; a fresh random one is generated when omitted.

    Returns:
        ``iv || ciphertext`` -- the IV is always prefixed so the result is self-contained.

    Raises:
        VaultError: if the key or IV length is wrong.
    """
    key = _check_aes_key(key)
    if iv is None:
        iv = random_bytes(AES_BLOCK_SIZE)
    iv = bytes(iv)
    if len(iv) != AES_BLOCK_SIZE:
        raise VaultError(f"AES-CBC IV must be exactly 16 bytes, got {len(iv)}")
    data = _pkcs7_pad(bytes(plaintext))
    schedule, _, rounds = _key_schedule(key)
    out = bytearray(len(data))
    previous = struct.unpack(">4I", iv)
    encrypt = _encrypt_words
    for offset in range(0, len(data), 16):
        b0, b1, b2, b3 = struct.unpack_from(">4I", data, offset)
        previous = encrypt(
            schedule,
            rounds,
            b0 ^ previous[0],
            b1 ^ previous[1],
            b2 ^ previous[2],
            b3 ^ previous[3],
        )
        struct.pack_into(">4I", out, offset, *previous)
    return iv + bytes(out)


def aes_cbc_decrypt_no_padding(key: bytes, data: bytes) -> bytes:
    """Decrypt ``iv || ciphertext`` with AES-CBC without touching the padding.

    Args:
        key: 16, 24 or 32 raw key bytes.
        data: the IV followed by a whole number of cipher blocks.

    Returns:
        The raw plaintext, padding bytes included.

    Raises:
        VaultError: if the key is invalid or the body is not block aligned.
    """
    key = _check_aes_key(key)
    data = bytes(data)
    if len(data) < AES_BLOCK_SIZE:
        raise VaultError("AES-CBC input must contain at least a 16-byte IV")
    iv, body = data[:AES_BLOCK_SIZE], data[AES_BLOCK_SIZE:]
    if not body:
        return b""
    if len(body) % AES_BLOCK_SIZE:
        raise VaultError(
            f"AES-CBC ciphertext must be a multiple of 16 bytes, got {len(body)}"
        )
    # CBC decryption is parallel over blocks: decrypt every block, then XOR the
    # whole buffer against the ciphertext shifted by one block.
    decrypted = _ecb_decrypt_buffer(key, body)
    return _xor_bytes(decrypted, iv + body[: len(body) - AES_BLOCK_SIZE])


def aes_cbc_decrypt(key: bytes, data: bytes) -> bytes:
    """Decrypt ``iv || ciphertext`` with AES-CBC and strip PKCS#7 padding.

    The padding check is deliberately tolerant: PDF producers are known to emit
    strings whose final block is not padded at all, so when the trailing bytes do
    not form valid PKCS#7 the raw plaintext is returned instead of raising.

    Args:
        key: 16, 24 or 32 raw key bytes.
        data: the IV followed by a whole number of cipher blocks.

    Returns:
        The plaintext with padding removed when the padding was well formed.

    Raises:
        VaultError: if the key is invalid or the body is not block aligned.
    """
    return _pkcs7_unpad(aes_cbc_decrypt_no_padding(key, data))


def aes_ctr(key: bytes, nonce: bytes, data: bytes, initial_counter: int = 0) -> bytes:
    """Encrypt or decrypt with AES in counter mode.

    The counter block is ``nonce || counter`` where the counter occupies the
    remaining ``16 - len(nonce)`` bytes big-endian and increments once per block.
    A 16-byte nonce is taken as the full initial counter block, which is then
    incremented as a single 128-bit big-endian integer (the NIST SP 800-38A layout).

    Args:
        key: 16, 24 or 32 raw key bytes.
        nonce: 0..16 bytes.
        data: plaintext or ciphertext; CTR is its own inverse.
        initial_counter: the counter value for the first block.

    Returns:
        A buffer the same length as ``data``.

    Raises:
        VaultError: if the key or nonce length is wrong.
    """
    key = _check_aes_key(key)
    nonce = bytes(nonce)
    if len(nonce) > AES_BLOCK_SIZE:
        raise VaultError(f"AES-CTR nonce must be at most 16 bytes, got {len(nonce)}")
    data = bytes(data)
    if not data:
        return b""
    blocks = (len(data) + 15) // 16
    counter_bytes = AES_BLOCK_SIZE - len(nonce)
    if counter_bytes == 0:
        # The whole nonce is the initial counter block: increment it as one
        # 128-bit big-endian integer (NIST SP 800-38A layout).
        base = int.from_bytes(nonce, "big")
        stream = b"".join(
            ((base + initial_counter + i) % (1 << 128)).to_bytes(16, "big")
            for i in range(blocks)
        )
    else:
        modulus = 1 << (8 * counter_bytes)
        prefix = int.from_bytes(nonce, "big") << (8 * counter_bytes)
        stream = b"".join(
            (prefix | ((initial_counter + i) % modulus)).to_bytes(16, "big")
            for i in range(blocks)
        )
    keystream = _ecb_encrypt_buffer(key, stream)[: len(data)]
    return _xor_bytes(data, keystream)


# ======================================================================================
# RC4
# ======================================================================================


def rc4(key: bytes, data: bytes) -> bytes:
    """Apply the RC4 stream cipher (encryption and decryption are identical).

    Args:
        key: 1..256 key bytes.
        data: the buffer to transform.

    Returns:
        A buffer the same length as ``data``.

    Raises:
        VaultError: if the key is empty or longer than 256 bytes.
    """
    key = bytes(key)
    if not key:
        raise VaultError("RC4 key must not be empty")
    if len(key) > 256:
        raise VaultError(f"RC4 key must be at most 256 bytes, got {len(key)}")
    data = bytes(data)
    if not data:
        return b""

    state = list(range(256))
    key_length = len(key)
    j = 0
    for i in range(256):
        j = (j + state[i] + key[i % key_length]) & 0xFF
        state[i], state[j] = state[j], state[i]

    out = bytearray(len(data))
    i = 0
    j = 0
    for index, byte in enumerate(data):
        i = (i + 1) & 0xFF
        j = (j + state[i]) & 0xFF
        state[i], state[j] = state[j], state[i]
        out[index] = byte ^ state[(state[i] + state[j]) & 0xFF]
    return bytes(out)


# ======================================================================================
# ChaCha20 (RFC 8439)
# ======================================================================================

_CHACHA_CONSTANTS = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)


def _chacha20_block(key_words: Tuple[int, ...], nonce_words: Tuple[int, ...], counter: int) -> bytes:
    """Produce the 64-byte ChaCha20 keystream block for ``counter``."""
    k0, k1, k2, k3, k4, k5, k6, k7 = key_words
    n0, n1, n2 = nonce_words
    c0, c1, c2, c3 = _CHACHA_CONSTANTS
    counter &= _MASK32

    x0, x1, x2, x3 = c0, c1, c2, c3
    x4, x5, x6, x7 = k0, k1, k2, k3
    x8, x9, x10, x11 = k4, k5, k6, k7
    x12, x13, x14, x15 = counter, n0, n1, n2

    mask = _MASK32
    for _ in range(10):
        # Column round.
        x0 = (x0 + x4) & mask
        x12 ^= x0
        x12 = ((x12 << 16) | (x12 >> 16)) & mask
        x8 = (x8 + x12) & mask
        x4 ^= x8
        x4 = ((x4 << 12) | (x4 >> 20)) & mask
        x0 = (x0 + x4) & mask
        x12 ^= x0
        x12 = ((x12 << 8) | (x12 >> 24)) & mask
        x8 = (x8 + x12) & mask
        x4 ^= x8
        x4 = ((x4 << 7) | (x4 >> 25)) & mask

        x1 = (x1 + x5) & mask
        x13 ^= x1
        x13 = ((x13 << 16) | (x13 >> 16)) & mask
        x9 = (x9 + x13) & mask
        x5 ^= x9
        x5 = ((x5 << 12) | (x5 >> 20)) & mask
        x1 = (x1 + x5) & mask
        x13 ^= x1
        x13 = ((x13 << 8) | (x13 >> 24)) & mask
        x9 = (x9 + x13) & mask
        x5 ^= x9
        x5 = ((x5 << 7) | (x5 >> 25)) & mask

        x2 = (x2 + x6) & mask
        x14 ^= x2
        x14 = ((x14 << 16) | (x14 >> 16)) & mask
        x10 = (x10 + x14) & mask
        x6 ^= x10
        x6 = ((x6 << 12) | (x6 >> 20)) & mask
        x2 = (x2 + x6) & mask
        x14 ^= x2
        x14 = ((x14 << 8) | (x14 >> 24)) & mask
        x10 = (x10 + x14) & mask
        x6 ^= x10
        x6 = ((x6 << 7) | (x6 >> 25)) & mask

        x3 = (x3 + x7) & mask
        x15 ^= x3
        x15 = ((x15 << 16) | (x15 >> 16)) & mask
        x11 = (x11 + x15) & mask
        x7 ^= x11
        x7 = ((x7 << 12) | (x7 >> 20)) & mask
        x3 = (x3 + x7) & mask
        x15 ^= x3
        x15 = ((x15 << 8) | (x15 >> 24)) & mask
        x11 = (x11 + x15) & mask
        x7 ^= x11
        x7 = ((x7 << 7) | (x7 >> 25)) & mask

        # Diagonal round.
        x0 = (x0 + x5) & mask
        x15 ^= x0
        x15 = ((x15 << 16) | (x15 >> 16)) & mask
        x10 = (x10 + x15) & mask
        x5 ^= x10
        x5 = ((x5 << 12) | (x5 >> 20)) & mask
        x0 = (x0 + x5) & mask
        x15 ^= x0
        x15 = ((x15 << 8) | (x15 >> 24)) & mask
        x10 = (x10 + x15) & mask
        x5 ^= x10
        x5 = ((x5 << 7) | (x5 >> 25)) & mask

        x1 = (x1 + x6) & mask
        x12 ^= x1
        x12 = ((x12 << 16) | (x12 >> 16)) & mask
        x11 = (x11 + x12) & mask
        x6 ^= x11
        x6 = ((x6 << 12) | (x6 >> 20)) & mask
        x1 = (x1 + x6) & mask
        x12 ^= x1
        x12 = ((x12 << 8) | (x12 >> 24)) & mask
        x11 = (x11 + x12) & mask
        x6 ^= x11
        x6 = ((x6 << 7) | (x6 >> 25)) & mask

        x2 = (x2 + x7) & mask
        x13 ^= x2
        x13 = ((x13 << 16) | (x13 >> 16)) & mask
        x8 = (x8 + x13) & mask
        x7 ^= x8
        x7 = ((x7 << 12) | (x7 >> 20)) & mask
        x2 = (x2 + x7) & mask
        x13 ^= x2
        x13 = ((x13 << 8) | (x13 >> 24)) & mask
        x8 = (x8 + x13) & mask
        x7 ^= x8
        x7 = ((x7 << 7) | (x7 >> 25)) & mask

        x3 = (x3 + x4) & mask
        x14 ^= x3
        x14 = ((x14 << 16) | (x14 >> 16)) & mask
        x9 = (x9 + x14) & mask
        x4 ^= x9
        x4 = ((x4 << 12) | (x4 >> 20)) & mask
        x3 = (x3 + x4) & mask
        x14 ^= x3
        x14 = ((x14 << 8) | (x14 >> 24)) & mask
        x9 = (x9 + x14) & mask
        x4 ^= x9
        x4 = ((x4 << 7) | (x4 >> 25)) & mask

    return struct.pack(
        "<16I",
        (x0 + c0) & mask,
        (x1 + c1) & mask,
        (x2 + c2) & mask,
        (x3 + c3) & mask,
        (x4 + k0) & mask,
        (x5 + k1) & mask,
        (x6 + k2) & mask,
        (x7 + k3) & mask,
        (x8 + k4) & mask,
        (x9 + k5) & mask,
        (x10 + k6) & mask,
        (x11 + k7) & mask,
        (x12 + counter) & mask,
        (x13 + n0) & mask,
        (x14 + n1) & mask,
        (x15 + n2) & mask,
    )


def _check_chacha_params(key: bytes, nonce: bytes) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    key = bytes(key)
    nonce = bytes(nonce)
    if len(key) != CHACHA20_KEY_SIZE:
        raise VaultError(f"ChaCha20 key must be 32 bytes, got {len(key)}")
    if len(nonce) != CHACHA20_NONCE_SIZE:
        raise VaultError(f"ChaCha20 nonce must be 12 bytes, got {len(nonce)}")
    return struct.unpack("<8I", key), struct.unpack("<3I", nonce)


def chacha20(key: bytes, nonce: bytes, data: bytes, counter: int = 1) -> bytes:
    """Encrypt or decrypt with ChaCha20 (RFC 8439); the cipher is its own inverse.

    Args:
        key: 32 raw key bytes.
        nonce: 12-byte (96-bit) nonce.
        data: plaintext or ciphertext.
        counter: 32-bit block counter for the first block (RFC 8439 AEAD uses 1).

    Returns:
        A buffer the same length as ``data``.

    Raises:
        VaultError: if the key or nonce length is wrong.
    """
    key_words, nonce_words = _check_chacha_params(key, nonce)
    data = bytes(data)
    size = len(data)
    if size == 0:
        return b""
    blocks = (size + 63) // 64
    keystream = b"".join(
        _chacha20_block(key_words, nonce_words, (counter + i) & _MASK32) for i in range(blocks)
    )
    return _xor_bytes(data, keystream[:size])


# ======================================================================================
# Poly1305 (RFC 8439)
# ======================================================================================

_POLY1305_P = (1 << 130) - 5
_POLY1305_CLAMP = 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF


def poly1305_mac(key: bytes, msg: bytes) -> bytes:
    """Compute the 16-byte Poly1305 one-time authenticator (RFC 8439 section 2.5).

    Args:
        key: a 32-byte one-time key; the first half is clamped into ``r``, the
            second half is the additive term ``s``.
        msg: the message to authenticate.

    Returns:
        The 16-byte tag.

    Raises:
        VaultError: if the key is not exactly 32 bytes.
    """
    key = bytes(key)
    if len(key) != 32:
        raise VaultError(f"Poly1305 key must be 32 bytes, got {len(key)}")
    msg = bytes(msg)
    r = int.from_bytes(key[:16], "little") & _POLY1305_CLAMP
    s = int.from_bytes(key[16:], "little")
    prime = _POLY1305_P
    accumulator = 0
    for offset in range(0, len(msg), 16):
        chunk = msg[offset : offset + 16]
        accumulator = (
            accumulator + int.from_bytes(chunk, "little") + (1 << (8 * len(chunk)))
        ) % prime
        accumulator = (accumulator * r) % prime
    return ((accumulator + s) & ((1 << 128) - 1)).to_bytes(16, "little")


# ======================================================================================
# ChaCha20-Poly1305 AEAD (RFC 8439 section 2.8)
# ======================================================================================


def _pad16(data: bytes) -> bytes:
    remainder = len(data) % 16
    return b"" if remainder == 0 else b"\x00" * (16 - remainder)


def _aead_tag(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
    one_time_key = chacha20(key, nonce, b"\x00" * 32, counter=0)
    mac_data = b"".join(
        (
            aad,
            _pad16(aad),
            ciphertext,
            _pad16(ciphertext),
            len(aad).to_bytes(8, "little"),
            len(ciphertext).to_bytes(8, "little"),
        )
    )
    return poly1305_mac(one_time_key, mac_data)


def chacha20_poly1305_encrypt(
    key: bytes, nonce: bytes, plaintext: bytes, aad: bytes = b""
) -> bytes:
    """Seal ``plaintext`` with ChaCha20-Poly1305 (RFC 8439).

    Args:
        key: 32 raw key bytes.
        nonce: 12-byte nonce; never reuse one with the same key.
        plaintext: the data to protect.
        aad: additional authenticated data, covered by the tag but not encrypted.

    Returns:
        ``ciphertext || tag`` where the tag is 16 bytes.

    Raises:
        VaultError: if the key or nonce length is wrong.
    """
    key = bytes(key)
    nonce = bytes(nonce)
    aad = bytes(aad)
    ciphertext = chacha20(key, nonce, bytes(plaintext), counter=1)
    return ciphertext + _aead_tag(key, nonce, ciphertext, aad)


def chacha20_poly1305_decrypt(
    key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes = b""
) -> bytes:
    """Open a ChaCha20-Poly1305 sealed message (RFC 8439).

    Args:
        key: 32 raw key bytes.
        nonce: the 12-byte nonce used to seal.
        ciphertext: ``ciphertext || tag`` as produced by
            :func:`chacha20_poly1305_encrypt`.
        aad: the same additional authenticated data used when sealing.

    Returns:
        The recovered plaintext.

    Raises:
        VaultError: if the input is truncated, the key/nonce are malformed, or the
            authentication tag does not match (tampering or a wrong key).
    """
    key = bytes(key)
    nonce = bytes(nonce)
    aad = bytes(aad)
    ciphertext = bytes(ciphertext)
    if len(ciphertext) < POLY1305_TAG_SIZE:
        raise VaultError("ChaCha20-Poly1305 message is too short to contain a tag")
    body = ciphertext[: len(ciphertext) - POLY1305_TAG_SIZE]
    tag = ciphertext[len(ciphertext) - POLY1305_TAG_SIZE :]
    expected = _aead_tag(key, nonce, body, aad)
    if not constant_time_eq(expected, tag):
        raise VaultError("ChaCha20-Poly1305 authentication failed: tag mismatch")
    return chacha20(key, nonce, body, counter=1)


# ======================================================================================
# Key derivation, randomness, comparison
# ======================================================================================


def _scrypt_usable() -> bool:
    """Probe ``hashlib.scrypt``; some builds expose it without OpenSSL support."""
    function = getattr(hashlib, "scrypt", None)
    if function is None:
        return False
    try:
        function(b"probe", salt=b"probe", n=2, r=1, p=1, dklen=1, maxmem=1 << 20)
    except Exception:  # pragma: no cover - only on crippled OpenSSL builds
        return False
    return True


#: Immutable marker naming the KDF this platform actually uses.
_KDF_NAME = "scrypt" if _scrypt_usable() else "pbkdf2_hmac_sha256"


def kdf_name() -> str:
    """Return the name of the key derivation function in use on this platform.

    Either ``"scrypt"`` (preferred) or ``"pbkdf2_hmac_sha256"`` when the platform's
    hashlib cannot do scrypt.  Vault files record this so they can be reopened with
    the same derivation.
    """
    return _KDF_NAME


def derive_key(
    password: str | bytes,
    salt: bytes,
    *,
    n: int = 2 ** 14,
    r: int = 8,
    p: int = 1,
    dklen: int = 32,
) -> bytes:
    """Derive a symmetric key from a password.

    Uses :func:`hashlib.scrypt` with the given cost parameters.  When the platform
    lacks scrypt the function falls back to PBKDF2-HMAC-SHA256 with
    :data:`PBKDF2_ROUNDS` iterations; :func:`kdf_name` reports which one ran.

    Args:
        password: the secret, encoded as UTF-8 when given as ``str``.
        salt: a per-vault random salt (8 bytes or more is recommended).
        n: scrypt CPU/memory cost, a power of two greater than 1.
        r: scrypt block size factor.
        p: scrypt parallelisation factor.
        dklen: length of the derived key in bytes.

    Returns:
        Exactly ``dklen`` bytes of key material.

    Raises:
        VaultError: if the parameters are out of range or derivation fails.
    """
    if isinstance(password, str):
        password_bytes = password.encode("utf-8")
    else:
        password_bytes = bytes(password)
    salt = bytes(salt)
    if not salt:
        raise VaultError("derive_key requires a non-empty salt")
    if dklen <= 0:
        raise VaultError(f"derive_key dklen must be positive, got {dklen}")
    if n < 2 or (n & (n - 1)) != 0:
        raise VaultError(f"derive_key n must be a power of two greater than 1, got {n!r}")
    if r < 1 or p < 1:
        raise VaultError("derive_key r and p must both be >= 1")

    if _KDF_NAME == "scrypt":
        # scrypt needs roughly 128 * r * n bytes; give OpenSSL explicit headroom.
        maxmem = 128 * r * (n + p + 2) + (1 << 20)
        try:
            return hashlib.scrypt(
                password_bytes, salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=maxmem
            )
        except (ValueError, MemoryError) as exc:
            raise VaultError(f"scrypt key derivation failed: {exc}") from exc
    return hashlib.pbkdf2_hmac("sha256", password_bytes, salt, PBKDF2_ROUNDS, dklen)


def random_bytes(n: int) -> bytes:
    """Return ``n`` cryptographically strong random bytes.

    Raises:
        VaultError: if ``n`` is negative.
    """
    if n < 0:
        raise VaultError(f"random_bytes needs a non-negative length, got {n}")
    if n == 0:
        return b""
    return secrets.token_bytes(n)


def constant_time_eq(a: bytes, b: bytes) -> bool:
    """Compare two byte strings without leaking their contents through timing."""
    try:
        return hmac.compare_digest(bytes(a), bytes(b))
    except (TypeError, ValueError):
        return False
