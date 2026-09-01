"""Unit tests for :mod:`zfp.vault.cipher`.

Every algorithm is pinned to a published test vector:

* RC4    -- the classic Wikipedia/`arcfour` vectors.
* AES    -- FIPS-197 appendices B and C, plus NIST SP 800-38A F.5.1 for CTR.
* ChaCha20, Poly1305 and the AEAD -- RFC 8439 sections 2.3.2, 2.4.2, 2.5.2 and 2.8.2.
"""

from __future__ import annotations

import random
import unittest
from concurrent.futures import ThreadPoolExecutor

from zfp.core.errors import VaultError, ZfpError
from zfp.vault import cipher


def _hex(value: str) -> bytes:
    return bytes.fromhex(value.replace(" ", "").replace("\n", ""))


# RFC 8439 uses this sentence throughout sections 2.4.2 and 2.8.2.
SUNSCREEN = (
    b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip "
    b"for the future, sunscreen would be it."
)


class RC4TestCase(unittest.TestCase):
    def test_published_vectors(self) -> None:
        self.assertEqual(cipher.rc4(b"Key", b"Plaintext").hex(), "bbf316e8d940af0ad3")
        self.assertEqual(cipher.rc4(b"Wiki", b"pedia").hex(), "1021bf0420")

    def test_is_its_own_inverse(self) -> None:
        key = b"Secret"
        data = bytes(range(256)) * 3
        self.assertEqual(cipher.rc4(key, cipher.rc4(key, data)), data)

    def test_empty_data(self) -> None:
        self.assertEqual(cipher.rc4(b"Key", b""), b"")

    def test_empty_key_rejected(self) -> None:
        with self.assertRaises(VaultError):
            cipher.rc4(b"", b"data")

    def test_oversized_key_rejected(self) -> None:
        with self.assertRaises(VaultError):
            cipher.rc4(b"k" * 257, b"data")


class AesBlockTestCase(unittest.TestCase):
    """FIPS-197 appendix B (AES-128) and appendix C (AES-192 / AES-256)."""

    PLAINTEXT = _hex("00112233445566778899aabbccddeeff")
    VECTORS = (
        (_hex("000102030405060708090a0b0c0d0e0f"), "69c4e0d86a7b0430d8cdb78070b4c55a"),
        (
            _hex("000102030405060708090a0b0c0d0e0f1011121314151617"),
            "dda97ca4864cdfe06eaf70a0ec0d7191",
        ),
        (
            _hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"),
            "8ea2b7ca516745bfeafc49904b496089",
        ),
    )

    def test_fips197_encrypt(self) -> None:
        for key, expected in self.VECTORS:
            with self.subTest(bits=len(key) * 8):
                self.assertEqual(cipher.aes_ecb_encrypt(key, self.PLAINTEXT).hex(), expected)

    def test_fips197_decrypt(self) -> None:
        for key, expected in self.VECTORS:
            with self.subTest(bits=len(key) * 8):
                self.assertEqual(cipher.aes_ecb_decrypt(key, _hex(expected)), self.PLAINTEXT)

    def test_round_trip_random_blocks(self) -> None:
        rng = random.Random(20240501)
        for key_size in (16, 24, 32):
            key = bytes(rng.randrange(256) for _ in range(key_size))
            for _ in range(8):
                block = bytes(rng.randrange(256) for _ in range(16))
                encrypted = cipher.aes_ecb_encrypt(key, block)
                self.assertNotEqual(encrypted, block)
                self.assertEqual(cipher.aes_ecb_decrypt(key, encrypted), block)

    def test_interleaved_key_sizes_do_not_confuse_the_cache(self) -> None:
        # The key schedule is memoised; walk the sizes repeatedly to prove the
        # cache key includes the whole key and not just its identity.
        for _ in range(3):
            for key, expected in self.VECTORS:
                self.assertEqual(cipher.aes_ecb_encrypt(key, self.PLAINTEXT).hex(), expected)

    def test_bad_key_size(self) -> None:
        for size in (0, 8, 15, 17, 31, 33):
            with self.subTest(size=size):
                with self.assertRaises(VaultError):
                    cipher.aes_ecb_encrypt(bytes(size), self.PLAINTEXT)
                with self.assertRaises(VaultError):
                    cipher.aes_ecb_decrypt(bytes(size), self.PLAINTEXT)

    def test_bad_block_size(self) -> None:
        with self.assertRaises(VaultError):
            cipher.aes_ecb_encrypt(bytes(16), b"short")
        with self.assertRaises(VaultError):
            cipher.aes_ecb_decrypt(bytes(16), bytes(17))

    def test_errors_are_zfp_errors(self) -> None:
        with self.assertRaises(ZfpError):
            cipher.aes_ecb_encrypt(b"too-short", self.PLAINTEXT)


class AesCbcTestCase(unittest.TestCase):
    def test_round_trip_every_length_0_to_33(self) -> None:
        iv = _hex("000102030405060708090a0b0c0d0e0f")
        for key_size in (16, 24, 32):
            key = bytes(range(key_size))
            for length in range(34):
                plaintext = bytes((index * 7 + 3) & 0xFF for index in range(length))
                blob = cipher.aes_cbc_encrypt(key, plaintext, iv)
                self.assertEqual(blob[:16], iv)
                # PKCS#7 always appends at least one byte and pads to a block.
                self.assertEqual(len(blob) - 16, (length // 16 + 1) * 16)
                with self.subTest(key_size=key_size, length=length):
                    self.assertEqual(cipher.aes_cbc_decrypt(key, blob), plaintext)

    def test_random_iv_when_omitted(self) -> None:
        key = bytes(range(16))
        first = cipher.aes_cbc_encrypt(key, b"same message")
        second = cipher.aes_cbc_encrypt(key, b"same message")
        self.assertNotEqual(first[:16], second[:16])
        self.assertNotEqual(first, second)
        self.assertEqual(cipher.aes_cbc_decrypt(key, first), b"same message")
        self.assertEqual(cipher.aes_cbc_decrypt(key, second), b"same message")

    def test_chains_blocks_against_the_ecb_primitive(self) -> None:
        key = bytes(range(32))
        iv = bytes(range(16))
        plaintext = bytes(range(32))
        blob = cipher.aes_cbc_encrypt(key, plaintext, iv)
        first = cipher.aes_ecb_encrypt(key, bytes(a ^ b for a, b in zip(plaintext[:16], iv)))
        self.assertEqual(blob[16:32], first)
        second = cipher.aes_ecb_encrypt(
            key, bytes(a ^ b for a, b in zip(plaintext[16:32], first))
        )
        self.assertEqual(blob[32:48], second)

    def test_no_padding_variant_keeps_the_padding(self) -> None:
        key = bytes(range(16))
        iv = bytes(range(16))
        blob = cipher.aes_cbc_encrypt(key, b"abc", iv)
        self.assertEqual(cipher.aes_cbc_decrypt_no_padding(key, blob), b"abc" + bytes([13]) * 13)

    def test_tolerates_unpadded_ciphertext(self) -> None:
        # Some PDF producers emit AES strings whose final block was never padded.
        key = bytes(range(16))
        iv = bytes(range(16))
        plaintext = b"A" * 16  # trailing byte 0x41 is not a legal PKCS#7 length
        raw = cipher.aes_ecb_encrypt(key, bytes(a ^ b for a, b in zip(plaintext, iv)))
        self.assertEqual(cipher.aes_cbc_decrypt(key, iv + raw), plaintext)

    def test_misaligned_ciphertext_rejected(self) -> None:
        with self.assertRaises(VaultError):
            cipher.aes_cbc_decrypt(bytes(16), bytes(16) + bytes(20))

    def test_truncated_input_rejected(self) -> None:
        with self.assertRaises(VaultError):
            cipher.aes_cbc_decrypt(bytes(16), b"tooshort")

    def test_iv_only_decrypts_to_empty(self) -> None:
        self.assertEqual(cipher.aes_cbc_decrypt(bytes(16), bytes(16)), b"")

    def test_bad_iv_length(self) -> None:
        with self.assertRaises(VaultError):
            cipher.aes_cbc_encrypt(bytes(16), b"data", b"short-iv")


class AesCtrTestCase(unittest.TestCase):
    def test_nist_sp800_38a_vector(self) -> None:
        key = _hex("2b7e151628aed2a6abf7158809cf4f3c")
        counter_block = _hex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff")
        plaintext = _hex(
            "6bc1bee22e409f96e93d7e117393172a"
            "ae2d8a571e03ac9c9eb76fac45af8e51"
            "30c81c46a35ce411e5fbc1191a0a52ef"
            "f69f2445df4f9b17ad2b417be66c3710"
        )
        expected = (
            "874d6191b620e3261bef6864990db6ce"
            "9806f66b7970fdff8617187bb9fffdff"
            "5ae4df3edbd5d35e5b4f09020db03eab"
            "1e031dda2fbe03d1792170a0f3009cee"
        )
        self.assertEqual(cipher.aes_ctr(key, counter_block, plaintext).hex(), expected)

    def test_is_its_own_inverse(self) -> None:
        key = bytes(range(24))
        nonce = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        data = bytes(range(256)) * 2 + b"tail"
        self.assertEqual(cipher.aes_ctr(key, nonce, cipher.aes_ctr(key, nonce, data)), data)

    def test_initial_counter_offsets_the_keystream(self) -> None:
        key = bytes(range(16))
        nonce = bytes(8)
        data = bytes(64)
        whole = cipher.aes_ctr(key, nonce, data, 0)
        tail = cipher.aes_ctr(key, nonce, data[32:], 2)
        self.assertEqual(whole[32:], tail)

    def test_empty_input(self) -> None:
        self.assertEqual(cipher.aes_ctr(bytes(16), bytes(8), b""), b"")

    def test_oversized_nonce_rejected(self) -> None:
        with self.assertRaises(VaultError):
            cipher.aes_ctr(bytes(16), bytes(17), b"data")


class ChaCha20TestCase(unittest.TestCase):
    KEY = bytes(range(32))

    def test_rfc8439_block_function(self) -> None:
        """Section 2.3.2: keystream for counter 1 with the 0x09 nonce."""
        keystream = cipher.chacha20(
            self.KEY, _hex("000000090000004a00000000"), bytes(64), counter=1
        )
        self.assertEqual(
            keystream.hex(),
            "10f1e7e4d13b5915500fdd1fa32071c4"
            "c7d1f4c733c068030422aa9ac3d46c4e"
            "d2826446079faa0914c2d705d98b02a2"
            "b5129cd1de164eb9cbd083e8a2503c4e",
        )

    def test_rfc8439_encryption_vector(self) -> None:
        """Section 2.4.2: the sunscreen paragraph."""
        ciphertext = cipher.chacha20(
            self.KEY, _hex("000000000000004a00000000"), SUNSCREEN, counter=1
        )
        self.assertEqual(
            ciphertext.hex(),
            "6e2e359a2568f98041ba0728dd0d6981"
            "e97e7aec1d4360c20a27afccfd9fae0b"
            "f91b65c5524733ab8f593dabcd62b357"
            "1639d624e65152ab8f530c359f0861d8"
            "07ca0dbf500d6a6156a38e088a22b65e"
            "52bc514d16ccf806818ce91ab7793736"
            "5af90bbf74a35be6b40b8eedf2785e42"
            "874d",
        )

    def test_decrypts_by_re_encrypting(self) -> None:
        nonce = _hex("000000000000004a00000000")
        ciphertext = cipher.chacha20(self.KEY, nonce, SUNSCREEN)
        self.assertEqual(cipher.chacha20(self.KEY, nonce, ciphertext), SUNSCREEN)

    def test_counter_zero_differs_from_counter_one(self) -> None:
        nonce = bytes(12)
        self.assertNotEqual(
            cipher.chacha20(self.KEY, nonce, bytes(64), counter=0),
            cipher.chacha20(self.KEY, nonce, bytes(64), counter=1),
        )

    def test_block_boundaries(self) -> None:
        nonce = bytes(12)
        stream = cipher.chacha20(self.KEY, nonce, bytes(200))
        for length in (0, 1, 63, 64, 65, 128, 199, 200):
            with self.subTest(length=length):
                self.assertEqual(cipher.chacha20(self.KEY, nonce, bytes(length)), stream[:length])

    def test_bad_key_and_nonce(self) -> None:
        with self.assertRaises(VaultError):
            cipher.chacha20(bytes(16), bytes(12), b"data")
        with self.assertRaises(VaultError):
            cipher.chacha20(self.KEY, bytes(8), b"data")


class Poly1305TestCase(unittest.TestCase):
    def test_rfc8439_vector(self) -> None:
        key = _hex("85d6be7857556d337f4452fe42d506a8" "0103808afb0db2fd4abff6af4149f51b")
        tag = cipher.poly1305_mac(key, b"Cryptographic Forum Research Group")
        self.assertEqual(tag.hex(), "a8061dc1305136c6c22b8baf0c0127a9")

    def test_tag_length_and_determinism(self) -> None:
        key = bytes(range(32))
        first = cipher.poly1305_mac(key, b"message")
        self.assertEqual(len(first), 16)
        self.assertEqual(first, cipher.poly1305_mac(key, b"message"))
        self.assertNotEqual(first, cipher.poly1305_mac(key, b"messagf"))

    def test_empty_message(self) -> None:
        key = bytes(range(32))
        self.assertEqual(len(cipher.poly1305_mac(key, b"")), 16)

    def test_bad_key_length(self) -> None:
        with self.assertRaises(VaultError):
            cipher.poly1305_mac(bytes(16), b"message")


class ChaCha20Poly1305TestCase(unittest.TestCase):
    KEY = bytes(range(0x80, 0xA0))
    NONCE = _hex("070000004041424344454647")
    AAD = _hex("50515253c0c1c2c3c4c5c6c7")
    CIPHERTEXT = (
        "d31a8d34648e60db7b86afbc53ef7ec2"
        "a4aded51296e08fea9e2b5a736ee62d6"
        "3dbea45e8ca9671282fafb69da92728b"
        "1a71de0a9e060b2905d6a5b67ecd3b36"
        "92ddbd7f2d778b8c9803aee328091b58"
        "fab324e4fad675945585808b4831d7bc"
        "3ff4def08e4b7a9de576d26586cec64b"
        "6116"
    )
    TAG = "1ae10b594f09e26a7e902ecbd0600691"

    def test_rfc8439_seal_vector(self) -> None:
        sealed = cipher.chacha20_poly1305_encrypt(self.KEY, self.NONCE, SUNSCREEN, self.AAD)
        self.assertEqual(sealed[:-16].hex(), self.CIPHERTEXT)
        self.assertEqual(sealed[-16:].hex(), self.TAG)

    def test_rfc8439_open_vector(self) -> None:
        sealed = _hex(self.CIPHERTEXT) + _hex(self.TAG)
        self.assertEqual(
            cipher.chacha20_poly1305_decrypt(self.KEY, self.NONCE, sealed, self.AAD), SUNSCREEN
        )

    def test_round_trip(self) -> None:
        key = cipher.random_bytes(32)
        nonce = cipher.random_bytes(12)
        for payload in (b"", b"x", b"y" * 63, b"z" * 64, b"w" * 65, bytes(range(256))):
            with self.subTest(size=len(payload)):
                sealed = cipher.chacha20_poly1305_encrypt(key, nonce, payload, b"header")
                self.assertEqual(len(sealed), len(payload) + 16)
                self.assertEqual(
                    cipher.chacha20_poly1305_decrypt(key, nonce, sealed, b"header"), payload
                )

    def test_tampered_ciphertext_is_rejected(self) -> None:
        key = bytes(range(32))
        nonce = bytes(range(12))
        sealed = bytearray(cipher.chacha20_poly1305_encrypt(key, nonce, b"top secret value"))
        sealed[3] ^= 0x01
        with self.assertRaises(VaultError):
            cipher.chacha20_poly1305_decrypt(key, nonce, bytes(sealed))

    def test_tampered_tag_is_rejected(self) -> None:
        key = bytes(range(32))
        nonce = bytes(range(12))
        sealed = bytearray(cipher.chacha20_poly1305_encrypt(key, nonce, b"top secret value"))
        sealed[-1] ^= 0x80
        with self.assertRaises(VaultError):
            cipher.chacha20_poly1305_decrypt(key, nonce, bytes(sealed))

    def test_tampered_aad_is_rejected(self) -> None:
        key = bytes(range(32))
        nonce = bytes(range(12))
        sealed = cipher.chacha20_poly1305_encrypt(key, nonce, b"payload", b"aad-v1")
        with self.assertRaises(VaultError):
            cipher.chacha20_poly1305_decrypt(key, nonce, sealed, b"aad-v2")

    def test_wrong_key_is_rejected(self) -> None:
        nonce = bytes(range(12))
        sealed = cipher.chacha20_poly1305_encrypt(bytes(32), nonce, b"payload")
        with self.assertRaises(VaultError):
            cipher.chacha20_poly1305_decrypt(bytes([1]) + bytes(31), nonce, sealed)

    def test_truncated_message_is_rejected(self) -> None:
        with self.assertRaises(VaultError):
            cipher.chacha20_poly1305_decrypt(bytes(32), bytes(12), b"short")


class KeyDerivationTestCase(unittest.TestCase):
    def test_kdf_name(self) -> None:
        self.assertIn(cipher.kdf_name(), ("scrypt", "pbkdf2_hmac_sha256"))

    def test_deterministic(self) -> None:
        first = cipher.derive_key("correct horse", b"salt-1234", n=1024, r=8, p=1)
        second = cipher.derive_key("correct horse", b"salt-1234", n=1024, r=8, p=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)

    def test_str_and_bytes_passwords_agree(self) -> None:
        self.assertEqual(
            cipher.derive_key("pass-wörd", b"salty", n=1024),
            cipher.derive_key("pass-wörd".encode(), b"salty", n=1024),
        )

    def test_salt_and_password_change_the_key(self) -> None:
        base = cipher.derive_key("pw", b"salt-a", n=1024)
        self.assertNotEqual(base, cipher.derive_key("pw", b"salt-b", n=1024))
        self.assertNotEqual(base, cipher.derive_key("pw2", b"salt-a", n=1024))

    def test_dklen_is_honoured(self) -> None:
        for dklen in (16, 24, 32, 64):
            with self.subTest(dklen=dklen):
                self.assertEqual(len(cipher.derive_key("pw", b"salt", n=1024, dklen=dklen)), dklen)

    def test_default_cost_parameters_work(self) -> None:
        self.assertEqual(len(cipher.derive_key("pw", b"salt-default")), 32)

    def test_invalid_parameters(self) -> None:
        with self.assertRaises(VaultError):
            cipher.derive_key("pw", b"")
        with self.assertRaises(VaultError):
            cipher.derive_key("pw", b"salt", n=1000)  # not a power of two
        with self.assertRaises(VaultError):
            cipher.derive_key("pw", b"salt", n=1)
        with self.assertRaises(VaultError):
            cipher.derive_key("pw", b"salt", n=1024, dklen=0)
        with self.assertRaises(VaultError):
            cipher.derive_key("pw", b"salt", n=1024, r=0)


class UtilityTestCase(unittest.TestCase):
    def test_random_bytes(self) -> None:
        self.assertEqual(len(cipher.random_bytes(24)), 24)
        self.assertEqual(cipher.random_bytes(0), b"")
        self.assertNotEqual(cipher.random_bytes(16), cipher.random_bytes(16))
        with self.assertRaises(VaultError):
            cipher.random_bytes(-1)

    def test_constant_time_eq(self) -> None:
        self.assertTrue(cipher.constant_time_eq(b"abc", b"abc"))
        self.assertFalse(cipher.constant_time_eq(b"abc", b"abd"))
        self.assertFalse(cipher.constant_time_eq(b"abc", b"abcd"))
        self.assertTrue(cipher.constant_time_eq(b"", b""))
        self.assertFalse(cipher.constant_time_eq(b"abc", "abc"))


class ConcurrencyAndVolumeTestCase(unittest.TestCase):
    def test_key_schedule_cache_is_thread_safe(self) -> None:
        plaintext = _hex("00112233445566778899aabbccddeeff")
        jobs = [bytes(range(size)) for size in (16, 24, 32)] * 8
        expected = [cipher.aes_ecb_encrypt(key, plaintext) for key in jobs]
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda key: cipher.aes_ecb_encrypt(key, plaintext), jobs))
        self.assertEqual(results, expected)

    def test_bulk_buffers_round_trip(self) -> None:
        payload = bytes((index * 31 + 7) & 0xFF for index in range(64 * 1024))
        aes_key = bytes(range(32))
        self.assertEqual(
            cipher.aes_ctr(aes_key, bytes(8), cipher.aes_ctr(aes_key, bytes(8), payload)),
            payload,
        )
        blob = cipher.aes_cbc_encrypt(aes_key, payload, bytes(16))
        self.assertEqual(cipher.aes_cbc_decrypt(aes_key, blob), payload)

        chacha_key = bytes(range(32))
        sealed = cipher.chacha20_poly1305_encrypt(chacha_key, bytes(12), payload, b"bulk")
        self.assertEqual(
            cipher.chacha20_poly1305_decrypt(chacha_key, bytes(12), sealed, b"bulk"), payload
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
