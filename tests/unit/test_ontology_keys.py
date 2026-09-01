"""Structural guarantees for the canonical key namespace."""

from __future__ import annotations

import re
import unittest

from zfp.core.types import FieldType
from zfp.ontology import keys as K
from zfp.ontology.keys import CANONICAL_KEYS, KeySpec

KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")

# Names the resolver registries are contractually required to provide.
KNOWN_NORMALIZERS = frozenset(
    """upper lower title digits_only phone_us phone_e164 ssn ein zip5 zip9 date_mdy
    date_ymd date_dmy currency state_abbrev country_iso2 email strip_ws name_case
    credit_card iban boolean_yes_no""".split()
)
KNOWN_VALIDATORS = frozenset(
    """luhn ssn ein email phone_us zip_us date iban routing_aba vin nonempty max_length
    regex choice""".split()
)


class TestKeyNamespaceScale(unittest.TestCase):
    def test_at_least_240_keys(self) -> None:
        self.assertGreaterEqual(K.key_count(), 240)
        self.assertEqual(K.key_count(), len(CANONICAL_KEYS))

    def test_required_namespaces_present(self) -> None:
        for namespace in (
            "person", "company", "bank", "card", "billing", "shipping", "document",
            "vehicle", "insurance", "medical", "employment", "education", "property",
            "consent", "misc",
        ):
            with self.subTest(namespace=namespace):
                self.assertTrue(K.children(namespace), namespace)

    def test_research_document_keys_exist(self) -> None:
        """Every key named in the cleanroom findings must be declared."""
        for key in (
            "person.name.first", "person.name.middle", "person.name.last",
            "person.name.full", "person.address.street_1", "person.address.street_2",
            "person.address.city", "person.address.region", "person.address.postal_code",
            "person.address.country", "person.email", "person.phone.mobile",
            "person.date_of_birth", "company.legal_name", "company.dba_name",
            "company.tax_id.ein", "bank.routing_number", "bank.account_number",
            "bank.account_type", "document.effective_date", "document.signer.title",
            "document.signer.signature", "person.initials",
        ):
            with self.subTest(key=key):
                self.assertIsNotNone(K.get(key), key)


class TestKeySpecIntegrity(unittest.TestCase):
    def test_keys_are_unique_dotted_and_lowercase(self) -> None:
        seen = set()
        for key, spec in CANONICAL_KEYS.items():
            self.assertEqual(key, spec.key)
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertEqual(key, key.lower())
            self.assertRegex(key, KEY_RE)
            self.assertIn(".", key)

    def test_field_types_are_enum_members(self) -> None:
        for spec in K.all_keys():
            self.assertIsInstance(spec.field_type, FieldType)

    def test_labels_are_non_empty(self) -> None:
        for spec in K.all_keys():
            self.assertTrue(spec.label.strip(), spec.key)

    def test_sensitivity_band_is_legal(self) -> None:
        for spec in K.all_keys():
            self.assertIn(spec.sensitivity, K.SENSITIVITIES, spec.key)

    def test_secret_keys_are_marked(self) -> None:
        for key in (
            "person.ssn", "person.itin", "card.number", "card.cvv",
            "bank.account_number", "credentials.password", "credentials.pin",
            "company.tax_id.ein",
        ):
            with self.subTest(key=key):
                self.assertEqual(CANONICAL_KEYS[key].sensitivity, "secret")

    def test_pii_keys_are_marked(self) -> None:
        for key in (
            "person.name.first", "person.name.last", "person.address.street_1",
            "person.address.postal_code", "person.date_of_birth", "person.phone.mobile",
            "person.email",
        ):
            with self.subTest(key=key):
                self.assertEqual(CANONICAL_KEYS[key].sensitivity, "pii")
                self.assertTrue(CANONICAL_KEYS[key].is_sensitive)

    def test_normalizer_and_validator_names_are_identifiers(self) -> None:
        for spec in K.all_keys():
            for name in (spec.normalizer, spec.validator):
                if name is None:
                    continue
                with self.subTest(key=spec.key, name=name):
                    self.assertTrue(name.isidentifier(), name)
                    self.assertEqual(name, name.lower())

    def test_referenced_callables_are_in_the_contract_registries(self) -> None:
        for spec in K.all_keys():
            if spec.normalizer is not None:
                self.assertIn(spec.normalizer, KNOWN_NORMALIZERS, spec.key)
            if spec.validator is not None:
                self.assertIn(spec.validator, KNOWN_VALIDATORS, spec.key)

    def test_patterns_compile_and_are_anchored(self) -> None:
        for spec in K.all_keys():
            if spec.pattern is None:
                continue
            with self.subTest(key=spec.key):
                re.compile(spec.pattern)
                self.assertTrue(spec.pattern.startswith("^"))
                self.assertTrue(spec.pattern.endswith("$"))

    def test_max_length_is_positive_when_present(self) -> None:
        for spec in K.all_keys():
            if spec.max_length is not None:
                self.assertGreater(spec.max_length, 0, spec.key)

    def test_aliases_are_unique_within_a_spec(self) -> None:
        for spec in K.all_keys():
            self.assertEqual(len(spec.aliases), len(set(spec.aliases)), spec.key)

    def test_spec_is_frozen(self) -> None:
        spec = CANONICAL_KEYS["person.name.first"]
        with self.assertRaises(AttributeError):
            spec.key = "nope"  # type: ignore[misc]

    def test_invalid_sensitivity_rejected(self) -> None:
        from zfp.core.errors import SemanticError

        with self.assertRaises(SemanticError):
            KeySpec(key="x.y", field_type=FieldType.TEXT, label="X", sensitivity="hush")


class TestParentDiscrimination(unittest.TestCase):
    def test_billing_and_shipping_duplicate_the_address_block(self) -> None:
        billing = {s.scoped_suffix() for s in K.keys_with_parent("billing")}
        shipping = {s.scoped_suffix() for s in K.keys_with_parent("shipping")}
        self.assertEqual(billing, shipping)
        self.assertIn("address.postal_code", billing)

    def test_parent_scoped_keys_declare_their_parent(self) -> None:
        for spec in K.keys_with_parent("billing"):
            self.assertIn("billing", spec.parents)
            self.assertTrue(spec.key.startswith("billing."))

    def test_base_person_address_has_no_parents(self) -> None:
        self.assertEqual(CANONICAL_KEYS["person.address.city"].parents, ())


class TestAccessors(unittest.TestCase):
    def test_get_returns_none_for_unknown(self) -> None:
        self.assertIsNone(K.get("person.name.nonexistent"))

    def test_all_keys_is_sorted_and_complete(self) -> None:
        specs = K.all_keys()
        self.assertEqual(len(specs), len(CANONICAL_KEYS))
        self.assertEqual([s.key for s in specs], sorted(CANONICAL_KEYS))

    def test_children_excludes_the_prefix_itself(self) -> None:
        kids = K.children("person.name")
        self.assertGreaterEqual(len(kids), 10)
        self.assertTrue(all(s.key.startswith("person.name.") for s in kids))
        self.assertNotIn("person.name", [s.key for s in kids])

    def test_children_is_recursive(self) -> None:
        person = {s.key for s in K.children("person")}
        self.assertIn("person.name.first", person)
        self.assertIn("person.driver_license.number", person)

    def test_children_tolerates_trailing_dot(self) -> None:
        self.assertEqual(
            [s.key for s in K.children("person.name.")],
            [s.key for s in K.children("person.name")],
        )

    def test_namespaces_are_sorted_and_unique(self) -> None:
        names = K.namespaces()
        self.assertEqual(names, sorted(set(names)))
        self.assertIn("person", names)

    def test_spec_helpers(self) -> None:
        spec = CANONICAL_KEYS["billing.address.postal_code"]
        self.assertEqual(spec.namespace, "billing")
        self.assertEqual(spec.leaf, "postal_code")
        self.assertEqual(spec.scoped_suffix(), "address.postal_code")
        self.assertEqual(spec.path, ("billing", "address", "postal_code"))

    def test_as_dict_is_jsonable(self) -> None:
        import json

        payload = CANONICAL_KEYS["person.ssn"].as_dict()
        json.dumps(payload)
        self.assertEqual(payload["key"], "person.ssn")
        self.assertEqual(payload["field_type"], "text")
        self.assertEqual(payload["sensitivity"], "secret")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
