"""Unit tests for :mod:`zfp.resolver.normalizers`."""

from __future__ import annotations

import unittest

from zfp.resolver import normalizers as N


class IdempotenceTests(unittest.TestCase):
    def test_every_normalizer_is_callable_and_idempotent(self):
        for name, fn in N.REGISTRY.items():
            once = fn("test value 123", None)
            twice = fn(once, None)
            self.assertEqual(once, twice, name)


class SpecificFormatTests(unittest.TestCase):
    def test_ssn(self):
        self.assertEqual(N.ssn("123456789"), "123-45-6789")

    def test_ein(self):
        self.assertEqual(N.ein("123456789"), "12-3456789")

    def test_phone_us(self):
        self.assertEqual(N.phone_us("2125550134"), "(212) 555-0134")
        self.assertEqual(N.phone_us("12125550134"), "(212) 555-0134")

    def test_phone_e164(self):
        self.assertEqual(N.phone_e164("2125550134"), "+12125550134")

    def test_zip9(self):
        self.assertEqual(N.zip9("100011234"), "10001-1234")

    def test_currency(self):
        self.assertEqual(N.currency("1234.5"), "1,234.50")
        self.assertEqual(N.currency("$1,234.56"), "1,234.56")

    def test_state_abbrev(self):
        self.assertEqual(N.state_abbrev("Illinois"), "IL")
        self.assertEqual(N.state_abbrev("il"), "IL")
        self.assertEqual(N.state_abbrev("IL"), "IL")

    def test_country_iso2(self):
        self.assertEqual(N.country_iso2("United States"), "US")
        self.assertEqual(N.country_iso2("us"), "US")

    def test_name_case_handles_particles_and_prefixes(self):
        self.assertEqual(N.name_case("mcdonald"), "McDonald")
        self.assertEqual(N.name_case("o'brien"), "O'Brien")
        # The leading word of a standalone name field is capitalized even when it is a
        # particle (a name field starting "van der Berg" reads as a typo); a particle
        # mid-name stays lowercase, matching common convention.
        self.assertEqual(N.name_case("van der berg"), "Van der Berg")
        self.assertEqual(N.name_case("jan van der berg"), "Jan van der Berg")

    def test_credit_card_grouped_in_fours(self):
        self.assertEqual(N.credit_card("4111111111111111"), "4111 1111 1111 1111")

    def test_iban_spaced_in_fours(self):
        self.assertEqual(N.iban("GB82WEST12345698765432"), "GB82 WEST 1234 5698 7654 32")

    def test_boolean_yes_no(self):
        self.assertEqual(N.boolean_yes_no("true"), "Yes")
        self.assertEqual(N.boolean_yes_no("0"), "No")


class ParseDateTests(unittest.TestCase):
    def test_all_four_documented_shapes(self):
        cases = [
            ("2026-08-27", (2026, 8, 27)),
            ("8/27/2026", (2026, 8, 27)),
            ("Aug 27, 2026", (2026, 8, 27)),
            ("27 August 2026", (2026, 8, 27)),
        ]
        for text, (y, m, d) in cases:
            parsed = N.parse_date(text)
            self.assertIsNotNone(parsed, text)
            self.assertEqual((parsed.year, parsed.month, parsed.day), (y, m, d), text)

    def test_unparseable_returns_none(self):
        self.assertIsNone(N.parse_date("not a date"))


if __name__ == "__main__":
    unittest.main()
