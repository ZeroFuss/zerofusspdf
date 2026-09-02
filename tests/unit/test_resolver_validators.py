"""Unit tests for :mod:`zfp.resolver.validators`."""

from __future__ import annotations

import unittest

from zfp.resolver import validators as V


class LuhnTests(unittest.TestCase):
    def test_accepts_known_valid_and_rejects_tampered(self):
        self.assertTrue(V.luhn("4111111111111111").ok)
        self.assertFalse(V.luhn("4111111111111112").ok)


class SsnTests(unittest.TestCase):
    def test_rejects_invalid_areas(self):
        self.assertFalse(V.ssn("000-12-3456").ok)
        self.assertFalse(V.ssn("666-12-3456").ok)
        self.assertFalse(V.ssn("900-12-3456").ok)

    def test_accepts_plausible_ssn(self):
        self.assertTrue(V.ssn("123-45-6789").ok)


class RoutingAbaTests(unittest.TestCase):
    def test_accepts_known_valid_routing_number(self):
        self.assertTrue(V.routing_aba("021000021").ok)

    def test_rejects_bad_checksum(self):
        self.assertFalse(V.routing_aba("021000022").ok)


class VinTests(unittest.TestCase):
    def test_accepts_known_good_vin(self):
        self.assertTrue(V.vin("1HGCM82633A004352").ok)

    def test_rejects_letters_i_o_q(self):
        self.assertFalse(V.vin("1HGCM82633A00435I").ok)


class IbanTests(unittest.TestCase):
    def test_accepts_known_valid_iban(self):
        self.assertTrue(V.iban("GB82WEST12345698765432").ok)

    def test_rejects_bad_checksum(self):
        self.assertFalse(V.iban("GB82WEST12345698765433").ok)


class EmailPhoneZipTests(unittest.TestCase):
    def test_email(self):
        self.assertTrue(V.email("a@b.com").ok)
        self.assertFalse(V.email("not-an-email").ok)

    def test_phone_us_rejects_bad_area_code(self):
        self.assertFalse(V.phone_us("0125550134").ok)
        self.assertTrue(V.phone_us("2125550134").ok)

    def test_zip_us(self):
        self.assertTrue(V.zip_us("10001").ok)
        self.assertTrue(V.zip_us("100011234").ok)
        self.assertFalse(V.zip_us("1001").ok)


class ConstraintValidationTests(unittest.TestCase):
    def test_value_too_long_for_max_chars_estimate(self):
        from zfp.core.types import FieldConstraints
        c = FieldConstraints(max_chars_estimate=5)
        self.assertFalse(V.validate_against_constraints("way too long a value", c).ok)
        self.assertTrue(V.validate_against_constraints("short", c).ok)

    def test_comb_cells_limit(self):
        from zfp.core.types import FieldConstraints
        c = FieldConstraints(comb_cells=4)
        self.assertFalse(V.validate_against_constraints("12345", c).ok)
        self.assertTrue(V.validate_against_constraints("1234", c).ok)


if __name__ == "__main__":
    unittest.main()
