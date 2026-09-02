"""Egress redaction: the privacy boundary described in docs/PRIVACY.md.

Two rules are load-bearing and are asserted here directly:

* a placeholder's *shape* survives redaction while its content does not, which is what
  lets the ``RulesMember`` recognize an EIN blank without any value ever leaving the
  machine;
* a ``secret``-class value never reaches a prompt under any configuration, and
  :func:`~zfp.council.redaction.assert_no_secrets` is the check the QA dashboard's
  "unapproved PII egress" counter is built on.
"""

from __future__ import annotations

import unittest

from zfp.core.config import PrivacyConfig
from zfp.core.errors import PolicyError
from zfp.council import redaction as R


class TestRedactText(unittest.TestCase):
    def setUp(self) -> None:
        self.privacy = PrivacyConfig()

    def test_digits_become_hashes_of_the_same_length(self) -> None:
        self.assertEqual(R.redact_text("12-3456789", self.privacy), "##-#######")
        self.assertEqual(R.redact_text("(415) 555-0134", self.privacy), "(###) ###-####")

    def test_shape_survives_so_a_pattern_still_matches(self) -> None:
        from zfp.ontology import match_placeholder

        rule = match_placeholder(R.redact_text("12-3456789", self.privacy))
        self.assertIsNotNone(rule)
        self.assertEqual(rule.canonical_hint, "company.tax_id.ein")

    def test_letters_are_kept_so_a_label_stays_readable(self) -> None:
        self.assertEqual(R.redact_text("Employer Identification Number", self.privacy), "Employer Identification Number")

    def test_emails_lose_their_letters_too(self) -> None:
        self.assertEqual(R.redact_text("Jane.Doe@example.com", self.privacy), "xxxx.xxx@xxxxxxx.xxx")

    def test_whitespace_and_control_characters_are_collapsed(self) -> None:
        self.assertEqual(R.redact_text("  First\t\tName\n ", self.privacy), "First Name")
        self.assertEqual(R.redact_text("a\x00b", self.privacy), "a b")

    def test_truncated_to_the_budget(self) -> None:
        tight = PrivacyConfig(max_context_chars=5)
        self.assertEqual(R.redact_text("abcdefghij", tight), "abcde")

    def test_redaction_can_be_disabled(self) -> None:
        loose = PrivacyConfig(redact_values_in_prompts=False)
        self.assertEqual(R.redact_text("12-3456789", loose), "12-3456789")


class TestRedactContext(unittest.TestCase):
    def setUp(self) -> None:
        self.privacy = PrivacyConfig()

    def test_masks_a_value_shape_and_drops_a_secret_class_key(self) -> None:
        out = R.redact_context(
            {
                "label": "Tax ID",
                "placeholder": "12-3456789",
                "ssn": "123-45-6789",
                "person.ssn": "123-45-6789",
                "card_number": "4111 1111 1111 1111",
            },
            self.privacy,
        )
        self.assertEqual(out["placeholder"], "##-#######")
        self.assertEqual(out["label"], "Tax ID")
        self.assertNotIn("ssn", out)
        self.assertNotIn("person.ssn", out)
        self.assertNotIn("card_number", out)

    def test_drops_value_bearing_keys(self) -> None:
        out = R.redact_context(
            {
                "label": "First Name",
                "value": "Jane",
                "filled_value": "Jane",
                "page_text": "the entire page",
                "vault": {"person.name.first": "Jane"},
                "declared_choices": ["Yes", "No"],
            },
            self.privacy,
        )
        self.assertEqual(sorted(out), ["declared_choices", "label"])

    def test_recurses_into_nested_structures(self) -> None:
        out = R.redact_context(
            {"shape": {"comb_cells": 9, "sample": "12-3456789"}, "nearby": ["ZIP 94107"]},
            self.privacy,
        )
        self.assertEqual(out["shape"]["sample"], "##-#######")
        self.assertEqual(out["shape"]["comb_cells"], 9)
        self.assertEqual(out["nearby"], ["ZIP #####"])

    def test_secret_keys_are_dropped_from_nested_mappings(self) -> None:
        out = R.redact_context({"outer": {"ssn": "123-45-6789", "label": "SSN"}}, self.privacy)
        self.assertEqual(out["outer"], {"label": "SSN"})

    def test_keys_are_emitted_in_sorted_order(self) -> None:
        out = R.redact_context({"z": "1", "a": "2", "m": "3"}, self.privacy)
        self.assertEqual(list(out), ["a", "m", "z"])

    def test_total_string_budget_is_enforced(self) -> None:
        privacy = PrivacyConfig(max_context_chars=12)
        out = R.redact_context({"a": "x" * 10, "b": "y" * 10, "c": "z" * 10}, privacy)
        self.assertEqual(R.context_char_count(out), 12)
        self.assertEqual(out["a"], "x" * 10)
        self.assertEqual(out["b"], "yy")
        self.assertEqual(out["c"], "")

    def test_long_integers_are_masked_but_structure_is_not(self) -> None:
        out = R.redact_context({"comb_cells": 9, "page": 3, "long_number": 123456789}, self.privacy)
        self.assertEqual(out["comb_cells"], 9)
        self.assertEqual(out["page"], 3)
        self.assertEqual(out["long_number"], "#########")

    def test_a_key_naming_a_secret_account_is_dropped_outright(self) -> None:
        out = R.redact_context({"account": 123456789, "label": "Account"}, self.privacy)
        self.assertEqual(out, {"label": "Account"})

    def test_identifiers_pass_through_unmasked(self) -> None:
        out = R.redact_context({"candidate": "fc_3f9a12", "group_id": "g_77"}, self.privacy)
        self.assertEqual(out["candidate"], "fc_3f9a12")
        self.assertEqual(out["group_id"], "g_77")

    def test_is_deterministic(self) -> None:
        ctx = {"label": "Tax ID", "nearby": ["12-3456789", "Employer"], "shape": {"width": 180.0}}
        self.assertEqual(R.redact_context(ctx, self.privacy), R.redact_context(ctx, self.privacy))

    def test_defaults_to_the_local_first_policy(self) -> None:
        self.assertEqual(R.redact_context({"placeholder": "12-3456789"})["placeholder"], "##-#######")

    def test_rejects_a_non_mapping(self) -> None:
        with self.assertRaises(PolicyError):
            R.redact_context(["not", "a", "mapping"], self.privacy)  # type: ignore[arg-type]

    def test_secret_key_detection_uses_the_ontology(self) -> None:
        self.assertEqual(R.secret_key_for("ssn"), "person.ssn")
        self.assertEqual(R.secret_key_for("person.ssn"), "person.ssn")
        self.assertEqual(R.secret_key_for("tax_id"), "company.tax_id.ein")
        self.assertIsNone(R.secret_key_for("label"))
        self.assertIsNone(R.secret_key_for("city"))


class TestAssertNoSecrets(unittest.TestCase):
    def setUp(self) -> None:
        self.privacy = PrivacyConfig()

    def test_raises_on_a_leaked_ssn_value(self) -> None:
        with self.assertRaises(PolicyError) as caught:
            R.assert_no_secrets({"note": "applicant ssn is 123-45-6789"}, self.privacy)
        self.assertIn("unapproved PII egress", str(caught.exception))

    def test_raises_on_a_leaked_ssn_even_with_redaction_disabled(self) -> None:
        loose = PrivacyConfig(redact_values_in_prompts=False)
        with self.assertRaises(PolicyError):
            R.assert_no_secrets({"note": "123-45-6789"}, loose)

    def test_raises_on_an_ein_a_card_number_and_an_iban(self) -> None:
        for value in ("12-3456789", "4111 1111 1111 1111", "GB33BUKB20201555555555"):
            with self.assertRaises(PolicyError):
                R.assert_no_secrets({"note": value}, self.privacy)

    def test_raises_on_a_secret_class_key(self) -> None:
        with self.assertRaises(PolicyError) as caught:
            R.assert_no_secrets({"ssn": "redacted"}, self.privacy)
        self.assertIn("secret-class key", str(caught.exception))

    def test_raises_on_an_unmasked_digit_run(self) -> None:
        with self.assertRaises(PolicyError) as caught:
            R.assert_no_secrets({"note": "94107-1234"}, self.privacy)
        self.assertIn("redactor was bypassed", str(caught.exception))

    def test_raises_on_a_secret_hiding_in_a_nested_list(self) -> None:
        with self.assertRaises(PolicyError):
            R.assert_no_secrets({"nearby": ["ok", {"deep": "123-45-6789"}]}, self.privacy)

    def test_accepts_a_redacted_context(self) -> None:
        ctx = R.redact_context(
            {
                "label": "Tax ID",
                "placeholder": "12-3456789",
                "section": ["Employer"],
                "shape": {"width": 180.0, "height": 12.0, "comb_cells": 9},
                "candidate": "fc_9a41bb20",
                "ssn": "123-45-6789",
            },
            self.privacy,
        )
        R.assert_no_secrets(ctx, self.privacy)

    def test_accepts_shapes_and_placeholders(self) -> None:
        R.assert_no_secrets({"placeholder": "##-#######", "shape": {"width": 180.0}}, self.privacy)

    def test_rejects_a_non_mapping(self) -> None:
        with self.assertRaises(PolicyError):
            R.assert_no_secrets("123-45-6789", self.privacy)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
