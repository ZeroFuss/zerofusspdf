"""Placeholder and value pattern rules."""

from __future__ import annotations

import re
import unittest

from zfp.core.types import FieldType
from zfp.ontology import patterns as P
from zfp.ontology.keys import CANONICAL_KEYS


class TestRuleTableIntegrity(unittest.TestCase):
    def test_at_least_30_rules(self) -> None:
        self.assertGreaterEqual(len(P.PATTERNS), 30)

    def test_every_regex_compiles(self) -> None:
        for rule in P.PATTERNS:
            with self.subTest(rule=rule.name):
                re.compile(rule.regex)
                self.assertIsNotNone(rule.compiled)

    def test_regexes_are_compiled_once_at_import(self) -> None:
        rule = P.PATTERNS_BY_NAME["ein_placeholder"]
        self.assertIs(rule.compiled, P.PATTERNS_BY_NAME["ein_placeholder"].compiled)

    def test_regexes_are_anchored(self) -> None:
        for rule in P.PATTERNS:
            with self.subTest(rule=rule.name):
                self.assertTrue(rule.regex.endswith("$"), rule.regex)

    def test_names_are_unique_identifiers(self) -> None:
        names = [rule.name for rule in P.PATTERNS]
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            self.assertTrue(name.isidentifier(), name)

    def test_confidences_are_bounded(self) -> None:
        for rule in P.PATTERNS:
            self.assertGreater(rule.confidence, 0.0, rule.name)
            self.assertLessEqual(rule.confidence, 1.0, rule.name)

    def test_field_types_are_enum_members(self) -> None:
        for rule in P.PATTERNS:
            self.assertIsInstance(rule.field_type, FieldType)

    def test_match_modes_are_legal(self) -> None:
        for rule in P.PATTERNS:
            self.assertIn(rule.matches, P.MATCH_MODES, rule.name)

    def test_canonical_hints_are_declared_keys(self) -> None:
        for rule in P.PATTERNS:
            if rule.canonical_hint is not None:
                self.assertIn(rule.canonical_hint, CANONICAL_KEYS, rule.name)

    def test_descriptions_present(self) -> None:
        for rule in P.PATTERNS:
            self.assertTrue(rule.description.strip(), rule.name)

    def test_both_sides_covered(self) -> None:
        modes = {rule.matches for rule in P.PATTERNS}
        self.assertIn("placeholder", modes)
        self.assertIn("value", modes)

    def test_invalid_match_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            P.PatternRule("bad", r"^x$", matches="sometimes")

    def test_as_dict_is_jsonable(self) -> None:
        import json

        payload = P.PATTERNS_BY_NAME["ein_placeholder"].as_dict()
        json.dumps(payload)
        self.assertEqual(payload["canonical_hint"], "company.tax_id.ein")


class TestMatchPlaceholder(unittest.TestCase):
    def test_ein_placeholder_from_the_research_table(self) -> None:
        rule = P.match_placeholder("##-#######")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.name, "ein_placeholder")
        self.assertEqual(rule.canonical_hint, "company.tax_id.ein")
        self.assertEqual(rule.format_hint, "NN-NNNNNNN")

    def test_research_placeholder_table(self) -> None:
        cases = {
            "MM/DD/YYYY": ("date_mdy_placeholder", FieldType.DATE),
            "___-__-____": ("ssn_placeholder", FieldType.TEXT),
            "(___) ___-____": ("phone_paren_placeholder", FieldType.PHONE),
            "_____-____": ("zip4_placeholder", FieldType.TEXT),
            "##-#######": ("ein_placeholder", FieldType.TEXT),
            "$ ______.__": ("currency_placeholder", FieldType.CURRENCY),
        }
        for text, (name, field_type) in cases.items():
            with self.subTest(text=text):
                rule = P.match_placeholder(text)
                self.assertIsNotNone(rule, text)
                self.assertEqual(rule.name, name)
                self.assertEqual(rule.field_type, field_type)

    def test_extended_placeholder_table(self) -> None:
        cases = {
            "DD/MM/YYYY": "date_dmy_placeholder",
            "YYYY-MM-DD": "date_iso_placeholder",
            "__/__/____": "date_underscore_placeholder",
            "Month Day, Year": "month_day_year_placeholder",
            "MM/YY": "mmyy_expiry_placeholder",
            "___-___-____": "phone_dash_placeholder",
            "ext. ____": "phone_ext_placeholder",
            "____ ____ ____ ____": "card_placeholder",
            "#### #### #### ####": "card_hash_placeholder",
            "HH:MM": "time_placeholder",
            "HH:MM AM/PM": "time_placeholder",
            "__.__%": "percent_placeholder",
            "[ ]": "checkbox_placeholder",
            "[ ] [ ] [ ] [ ]": "comb_placeholder",
            "X _________": "signature_placeholder",
            "_________________": "underscore_blank_placeholder",
            "..........": "dot_leader_placeholder",
            "#########": "routing_placeholder",
            "XX": "state_two_letter_placeholder",
            "A1A 1A1": "postal_ca_placeholder",
            "____@_____.___": "email_placeholder",
            "CVV ___": "cvv_placeholder",
            "__/__": "initials_pair_placeholder",
        }
        for text, name in cases.items():
            with self.subTest(text=text):
                rule = P.match_placeholder(text)
                self.assertIsNotNone(rule, text)
                self.assertEqual(rule.name, name, text)

    def test_ordinal_date_line(self) -> None:
        rule = P.match_placeholder("this ____ day of ________, 20__")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.name, "ordinal_date_placeholder")

    def test_case_insensitive_where_appropriate(self) -> None:
        self.assertEqual(
            P.match_placeholder("mm/dd/yyyy").name, "date_mdy_placeholder"
        )

    def test_surrounding_whitespace_and_colon_are_tolerated(self) -> None:
        self.assertEqual(P.match_placeholder("  ##-#######  :").name, "ein_placeholder")

    def test_no_match_returns_none(self) -> None:
        self.assertIsNone(P.match_placeholder("Employee Handbook"))
        self.assertIsNone(P.match_placeholder(""))

    def test_deterministic(self) -> None:
        first = [P.match_placeholder(t) for t in ("##-#######", "MM/DD/YYYY", "[ ]")]
        second = [P.match_placeholder(t) for t in ("##-#######", "MM/DD/YYYY", "[ ]")]
        self.assertEqual([r.name for r in first], [r.name for r in second])


class TestMatchValue(unittest.TestCase):
    def test_ein_value_agrees_with_the_ein_placeholder(self) -> None:
        placeholder = P.match_placeholder("##-#######")
        value = P.match_value("12-3456789")
        self.assertIsNotNone(value)
        self.assertEqual(value.name, "ein_value")
        self.assertEqual(value.canonical_hint, placeholder.canonical_hint)
        self.assertEqual(value.format_hint, placeholder.format_hint)
        self.assertEqual(value.field_type, placeholder.field_type)

    def test_value_table(self) -> None:
        cases = {
            "123-45-6789": "ssn_value",
            "12-3456789": "ein_value",
            "10001": "zip5_value",
            "10001-1234": "zip9_value",
            "(212) 555-1234": "phone_us_value",
            "212-555-1234": "phone_us_value",
            "+1 212 555 1234": "phone_us_value",
            "jane@example.com": "email_value",
            "https://example.com/x": "url_value",
            "$1,234.56": "currency_value",
            "2026-09-01": "date_iso_value",
            "09/01/2026": "date_mdy_value",
            "25/12/2026": "date_dmy_value",
            "4111 1111 1111 1111": "card_value",
            "GB82WEST12345698765432": "iban_value",
            "1HGCM82633A004352": "vin_value",
            "12:30 PM": "time_value",
            "15%": "percent_value",
            "NY": "state_abbrev_value",
            "yes": "yes_no_value",
            "X": "checkbox_mark_value",
            "12/28": "mmyy_value",
        }
        for value, name in cases.items():
            with self.subTest(value=value):
                rule = P.match_value(value)
                self.assertIsNotNone(rule, value)
                self.assertEqual(rule.name, name, value)

    def test_vin_rejects_i_o_and_q(self) -> None:
        self.assertIsNone(P.match_value("1HGCM82633A0043IO"))

    def test_no_match_returns_none(self) -> None:
        self.assertIsNone(P.match_value("Employee Handbook"))
        self.assertIsNone(P.match_value(""))

    def test_placeholder_rules_do_not_leak_into_value_matching(self) -> None:
        rule = P.match_value("##-#######")
        self.assertIsNone(rule)


class TestMatchAll(unittest.TestCase):
    def test_returns_best_first(self) -> None:
        matches = P.match_all("123456789", mode="value")
        self.assertTrue(matches)
        self.assertEqual(
            [m.name for m in matches],
            [m.name for m in sorted(matches, key=lambda r: (-r.confidence, r.name))],
        )
        self.assertIn("routing_value", [m.name for m in matches])
        self.assertIn("account_number_value", [m.name for m in matches])

    def test_rejects_bad_mode(self) -> None:
        with self.assertRaises(ValueError):
            P.match_all("x", mode="nonsense")

    def test_empty_text(self) -> None:
        self.assertEqual(P.match_all(""), [])


class TestInferFromContext(unittest.TestCase):
    def test_label_and_placeholder_agree_and_boost(self) -> None:
        rule = P.infer_from_context("Tax ID", "##-#######")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.name, "ein_placeholder")
        self.assertEqual(rule.canonical_hint, "company.tax_id.ein")
        self.assertGreater(
            rule.confidence, P.PATTERNS_BY_NAME["ein_placeholder"].confidence
        )
        self.assertLessEqual(rule.confidence, 1.0)

    def test_placeholder_only(self) -> None:
        rule = P.infer_from_context("Line 7b", "(___) ___-____")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.name, "phone_paren_placeholder")

    def test_label_only_falls_back_to_the_key_rule(self) -> None:
        rule = P.infer_from_context("Social Security Number", ())
        self.assertIsNotNone(rule)
        self.assertEqual(rule.canonical_hint, "person.ssn")

    def test_nearby_may_be_a_sequence(self) -> None:
        rule = P.infer_from_context("Employer ID", ["some noise", "##-#######"])
        self.assertEqual(rule.name, "ein_placeholder")

    def test_nearby_sentence_is_tokenized(self) -> None:
        rule = P.infer_from_context("Tax ID", "Enter EIN ##-####### exactly")
        self.assertEqual(rule.name, "ein_placeholder")

    def test_value_in_nearby_text_is_used_as_a_fallback(self) -> None:
        rule = P.infer_from_context("Federal Tax ID", "12-3456789")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.canonical_hint, "company.tax_id.ein")

    def test_nothing_known_returns_none(self) -> None:
        self.assertIsNone(P.infer_from_context("qzxwv frobnicator", "lorem ipsum"))

    def test_empty_inputs(self) -> None:
        self.assertIsNone(P.infer_from_context("", ()))

    def test_deterministic(self) -> None:
        a = P.infer_from_context("Tax ID", "##-#######")
        b = P.infer_from_context("Tax ID", "##-#######")
        self.assertEqual(a, b)


class TestRulesForKey(unittest.TestCase):
    def test_sorted_best_first(self) -> None:
        rules = P.rules_for_key("company.tax_id.ein")
        self.assertTrue(rules)
        self.assertEqual(
            rules, sorted(rules, key=lambda r: (-r.confidence, r.name))
        )
        for rule in rules:
            self.assertEqual(rule.canonical_hint, "company.tax_id.ein")

    def test_unknown_key(self) -> None:
        self.assertEqual(P.rules_for_key("person.name.nonexistent"), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
