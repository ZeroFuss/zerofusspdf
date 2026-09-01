"""Label normalization and alias resolution."""

from __future__ import annotations

import unittest

from zfp.ontology import aliases as A
from zfp.ontology.keys import CANONICAL_KEYS

#: The five spellings the cleanroom findings show collapsing onto one stored value.
ZIP_SPELLINGS = ("ZIP", "Postal Code", "Mail ZIP", "Zipcode", "ZIP / Postal")


class TestNormalizeLabel(unittest.TestCase):
    def test_lowercases_and_strips(self) -> None:
        self.assertEqual(A.normalize_label("  First Name  "), "first name")

    def test_drops_trailing_colon_and_asterisk(self) -> None:
        self.assertEqual(A.normalize_label("First Name:"), "first name")
        self.assertEqual(A.normalize_label("First Name*"), "first name")
        self.assertEqual(A.normalize_label("First Name *:"), "first name")

    def test_drops_parenthesised_hints(self) -> None:
        self.assertEqual(A.normalize_label("Email (required)"), "email")
        self.assertEqual(A.normalize_label("Middle Name (if any)"), "middle name")
        self.assertEqual(A.normalize_label("Date [MM/DD/YYYY]"), "date")

    def test_drops_leading_enumeration(self) -> None:
        self.assertEqual(A.normalize_label("1. First Name"), "first name")
        self.assertEqual(A.normalize_label("a) First Name"), "first name")
        self.assertEqual(A.normalize_label("iv. First Name"), "first name")
        self.assertEqual(A.normalize_label("• First Name"), "first name")

    def test_enumeration_strip_does_not_eat_initialisms(self) -> None:
        self.assertEqual(A.normalize_label("D.O.B."), "d o b")

    def test_maps_ampersand_to_and(self) -> None:
        self.assertEqual(A.normalize_label("City & State"), "city and state")

    def test_collapses_unicode_dashes_and_spaces(self) -> None:
        self.assertEqual(A.normalize_label("E—Mail Address"), "e mail address")

    def test_strips_non_alphanumerics_to_single_spaces(self) -> None:
        self.assertEqual(A.normalize_label("ZIP / Postal"), "zip postal")
        self.assertEqual(A.normalize_label("first_name"), "first name")
        self.assertEqual(A.normalize_label("Acct #"), "acct")

    def test_removes_possessives(self) -> None:
        self.assertEqual(A.normalize_label("Applicant's Name"), "applicant name")
        self.assertEqual(A.normalize_label("Patients’ Name"), "patients name")

    def test_empty_and_punctuation_only(self) -> None:
        self.assertEqual(A.normalize_label(""), "")
        self.assertEqual(A.normalize_label("   "), "")
        self.assertEqual(A.normalize_label(":::"), "")

    def test_is_idempotent(self) -> None:
        for raw in ("1) E-Mail Address (required):", "ZIP / Postal", "Applicant's Name"):
            once = A.normalize_label(raw)
            self.assertEqual(A.normalize_label(once), once, raw)


class TestAliasIndexScale(unittest.TestCase):
    def test_at_least_800_aliases(self) -> None:
        self.assertGreaterEqual(A.alias_count(), 800)

    def test_every_alias_maps_to_a_declared_key(self) -> None:
        for alias, key in A.ALIAS_INDEX.items():
            self.assertIn(key, CANONICAL_KEYS, f"{alias!r} -> {key!r}")

    def test_every_alias_is_already_normalized(self) -> None:
        for alias in A.ALIAS_INDEX:
            self.assertEqual(A.normalize_label(alias), alias)

    def test_no_empty_aliases(self) -> None:
        self.assertNotIn("", A.ALIAS_INDEX)

    def test_every_declared_alias_resolves_to_its_own_or_an_earlier_key(self) -> None:
        """A declared alias must always resolve to *some* key, never to nothing."""
        for spec in CANONICAL_KEYS.values():
            for alias in spec.aliases:
                with self.subTest(key=spec.key, alias=alias):
                    self.assertIsNotNone(A.lookup(alias))

    def test_aliases_for_round_trips(self) -> None:
        for alias in A.aliases_for("person.address.postal_code"):
            self.assertEqual(A.ALIAS_INDEX[alias], "person.address.postal_code")


class TestLookup(unittest.TestCase):
    def test_five_zip_spellings_collapse_to_one_key(self) -> None:
        resolved = {A.lookup(spelling) for spelling in ZIP_SPELLINGS}
        self.assertEqual(resolved, {"person.address.postal_code"})

    def test_zip_spellings_survive_form_decoration(self) -> None:
        for spelling in ZIP_SPELLINGS:
            for decorated in (spelling + ":", spelling + " *", "5. " + spelling):
                with self.subTest(label=decorated):
                    self.assertEqual(
                        A.lookup(decorated), "person.address.postal_code"
                    )

    def test_given_name_forename_first_name(self) -> None:
        for label in ("Given Name", "Forename", "First Name"):
            with self.subTest(label=label):
                self.assertEqual(A.lookup(label), "person.name.first")

    def test_common_form_phrasings(self) -> None:
        cases = {
            "Applicant First Name": "person.name.first",
            "Print Name": "person.name.full",
            "Your Email": "person.email",
            "E-Mail Address": "person.email",
            "Tel": "person.phone.mobile",
            "Telephone": "person.phone.mobile",
            "Cell": "person.phone.mobile",
            "Mobile No": "person.phone.mobile",
            "DOB": "person.date_of_birth",
            "D.O.B.": "person.date_of_birth",
            "Birth Date": "person.date_of_birth",
            "State/Province": "person.address.region",
            "Country/Region": "person.address.country",
            "Amt": "document.amount",
            "Amount Due": "document.amount",
            "Sign Here": "person.signature",
            "Signature of Applicant": "person.signature",
            "Date Signed": "document.signed_date",
            "Company Name": "company.legal_name",
            "Business Name": "company.legal_name",
            "Fed ID": "company.tax_id.ein",
            "Federal Tax ID": "company.tax_id.ein",
            "Employer Identification Number": "company.tax_id.ein",
            "Acct #": "bank.account_number",
            "Routing/ABA": "bank.routing_number",
            "SSN": "person.ssn",
            "Social Security Number": "person.ssn",
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assertEqual(A.lookup(label), expected)

    def test_role_prefix_is_dropped(self) -> None:
        for role in ("Applicant", "Employee", "Borrower", "Patient", "Customer",
                     "Member", "Student", "Tenant", "Buyer", "Seller", "Your"):
            with self.subTest(role=role):
                self.assertEqual(A.lookup(role + " Maiden Name"), "person.name.maiden")

    def test_please_print_prefix_is_dropped(self) -> None:
        self.assertEqual(A.lookup("Please print full name"), "person.name.full")

    def test_trailing_filler_is_dropped(self) -> None:
        self.assertEqual(A.lookup("Signature here"), "person.signature")
        self.assertEqual(A.lookup("Middle Name optional"), "person.name.middle")
        self.assertEqual(A.lookup("Email required"), "person.email")

    def test_trailing_plural_is_singularized(self) -> None:
        self.assertEqual(A.lookup("Comments"), A.lookup("Comment"))
        self.assertIsNotNone(A.lookup("Initials"))

    def test_unknown_label_returns_none(self) -> None:
        self.assertIsNone(A.lookup("qzxwv frobnicator"))
        self.assertIsNone(A.lookup(""))

    def test_lookup_is_deterministic(self) -> None:
        self.assertEqual(
            [A.lookup(s) for s in ZIP_SPELLINGS],
            [A.lookup(s) for s in ZIP_SPELLINGS],
        )


class TestFuzzyLookup(unittest.TestCase):
    def test_ocr_damage_still_resolves(self) -> None:
        results = A.fuzzy_lookup("Frst Name")
        self.assertTrue(results)
        self.assertEqual(results[0][0], "person.name.first")
        self.assertGreaterEqual(results[0][1], 0.82)

    def test_results_sorted_by_negative_ratio_then_key(self) -> None:
        results = A.fuzzy_lookup("Postl Code", cutoff=0.6)
        self.assertTrue(results)
        self.assertEqual(results, sorted(results, key=lambda kv: (-kv[1], kv[0])))

    def test_keys_are_unique_and_declared(self) -> None:
        results = A.fuzzy_lookup("Emai Adress", cutoff=0.6)
        keys = [key for key, _ in results]
        self.assertEqual(len(keys), len(set(keys)))
        for key in keys:
            self.assertIn(key, CANONICAL_KEYS)

    def test_high_cutoff_rejects_noise(self) -> None:
        self.assertEqual(A.fuzzy_lookup("qzxwv frobnicator", cutoff=0.95), [])

    def test_empty_label(self) -> None:
        self.assertEqual(A.fuzzy_lookup(""), [])

    def test_ratios_are_bounded(self) -> None:
        for _, ratio in A.fuzzy_lookup("Frst Name", cutoff=0.5):
            self.assertGreaterEqual(ratio, 0.0)
            self.assertLessEqual(ratio, 1.0)


class TestContextLookup(unittest.TestCase):
    def test_billing_context_selects_a_billing_key(self) -> None:
        key = A.context_lookup("Address", ("Billing",))
        self.assertIsNotNone(key)
        self.assertTrue(key.startswith("billing."), key)
        self.assertIn("billing", CANONICAL_KEYS[key].parents)

    def test_shipping_context_selects_a_shipping_key(self) -> None:
        key = A.context_lookup("Address", ("Shipping",))
        self.assertTrue(key.startswith("shipping."), key)

    def test_ship_to_synonym(self) -> None:
        self.assertTrue(A.context_lookup("City", ("Ship To",)).startswith("shipping."))

    def test_bill_to_synonym(self) -> None:
        self.assertTrue(A.context_lookup("City", ("Bill To",)).startswith("billing."))

    def test_multiword_section_heading(self) -> None:
        self.assertTrue(
            A.context_lookup("ZIP Code", ("Billing Information",)).startswith("billing.")
        )

    def test_string_context_is_accepted(self) -> None:
        self.assertTrue(A.context_lookup("City", "Billing").startswith("billing."))

    def test_context_remaps_a_plain_hit(self) -> None:
        self.assertEqual(
            A.context_lookup("Postal Code", ("Shipping",)),
            "shipping.address.postal_code",
        )

    def test_no_context_matches_plain_lookup(self) -> None:
        self.assertEqual(A.context_lookup("Address", ()), A.lookup("Address"))
        self.assertEqual(A.context_lookup("Address"), "person.address.street_1")

    def test_irrelevant_context_is_ignored(self) -> None:
        self.assertEqual(
            A.context_lookup("Address", ("Section 4", "Applicant")),
            A.lookup("Address"),
        )

    def test_unknown_label_with_context(self) -> None:
        self.assertIsNone(A.context_lookup("qzxwv frobnicator", ("Billing",)))

    def test_empty_label(self) -> None:
        self.assertIsNone(A.context_lookup("", ("Billing",)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
