"""Unit tests for :mod:`zfp.vault.store`."""

from __future__ import annotations

import os
import tempfile
import unittest

from zfp.core.errors import VaultError
from zfp.vault.store import ProfileVault


class BasicStoreTests(unittest.TestCase):
    def test_put_get_roundtrip(self):
        v = ProfileVault()
        v.put("person.name.first", "Jane", source="manual", confidence=0.9)
        entry = v.get("person.name.first")
        self.assertEqual(entry.value, "Jane")
        self.assertEqual(entry.source, "manual")

    def test_keys_sorted(self):
        v = ProfileVault()
        v.put("b.key", "1")
        v.put("a.key", "2")
        self.assertEqual(v.keys(), ["a.key", "b.key"])


class PersistenceTests(unittest.TestCase):
    def test_unencrypted_roundtrip(self):
        v = ProfileVault(profile_id="p1")
        v.put("person.name.first", "Jane")
        path = tempfile.mktemp()
        try:
            v.save(path)
            v2 = ProfileVault.load(path)
            self.assertEqual(v2.get("person.name.first").value, "Jane")
            self.assertEqual(v2.profile_id, "p1")
        finally:
            os.remove(path)

    def test_encrypted_roundtrip(self):
        v = ProfileVault()
        v.put("person.name.first", "Jane")
        path = tempfile.mktemp()
        try:
            v.save(path, password="s3cret!")
            v2 = ProfileVault.load(path, password="s3cret!")
            self.assertEqual(v2.get("person.name.first").value, "Jane")
        finally:
            os.remove(path)

    def test_wrong_password_raises_vault_error(self):
        v = ProfileVault()
        v.put("k", "v")
        path = tempfile.mktemp()
        try:
            v.save(path, password="right")
            with self.assertRaises(VaultError):
                ProfileVault.load(path, password="wrong")
        finally:
            os.remove(path)

    def test_tampered_file_raises_vault_error(self):
        v = ProfileVault()
        v.put("k", "v")
        path = tempfile.mktemp()
        try:
            v.save(path, password="right")
            with open(path, "r+b") as fh:
                data = bytearray(fh.read())
                data[-1] ^= 0xFF
                fh.seek(0)
                fh.write(data)
            with self.assertRaises(VaultError):
                ProfileVault.load(path, password="right")
        finally:
            os.remove(path)


class DerivationTests(unittest.TestCase):
    def test_full_name_synthesized_from_first_and_last(self):
        v = ProfileVault()
        v.put("person.name.first", "Jane", confidence=1.0)
        v.put("person.name.last", "Public", confidence=1.0)
        entry = v.resolve("person.name.full")
        self.assertEqual(entry.value, "Jane Public")
        self.assertEqual(entry.source, "derived")
        self.assertIn("person.name.first", entry.provenance["derived_from"])
        self.assertIn("person.name.last", entry.provenance["derived_from"])

    def test_missing_component_yields_no_derivation(self):
        v = ProfileVault()
        v.put("person.name.first", "Jane")
        self.assertIsNone(v.resolve("person.name.full"))


class ParentContextTests(unittest.TestCase):
    def test_billing_context_prefers_billing_scoped_entry(self):
        v = ProfileVault()
        v.put("person.address.street_1", "123 Main St")
        v.put("billing.address.street_1", "1 Corp Way")
        result = v.resolve("person.address.street_1", parent_context=("Billing",))
        self.assertEqual(result.value, "1 Corp Way")

    def test_no_billing_entry_falls_back_to_person(self):
        v = ProfileVault()
        v.put("person.address.street_1", "123 Main St")
        result = v.resolve("person.address.street_1", parent_context=("Billing",))
        self.assertEqual(result.value, "123 Main St")


class LabelLearningTests(unittest.TestCase):
    def test_observe_label_and_suggest(self):
        v = ProfileVault()
        v.put("person.address.postal_code", "10001")
        v.observe_label("person.address.postal_code", "ZIP / Postal")
        self.assertEqual(v.suggest_key_for_label("ZIP / Postal"), "person.address.postal_code")

    def test_observe_label_on_unknown_key_is_a_noop(self):
        v = ProfileVault()
        v.observe_label("nonexistent.key", "Foo")  # must not raise
        self.assertIsNone(v.suggest_key_for_label("Foo"))


class RedactSecretsTests(unittest.TestCase):
    def test_secret_values_are_masked_normal_values_are_not(self):
        v = ProfileVault()
        v.put("person.ssn", "123456789", sensitivity="secret")
        v.put("person.name.first", "Jane", sensitivity="normal")
        redacted = v.redact_secrets()
        self.assertNotEqual(redacted.get("person.ssn").value, "123456789")
        self.assertEqual(redacted.get("person.name.first").value, "Jane")


class MergeTests(unittest.TestCase):
    def test_higher_confidence_wins_by_default(self):
        a = ProfileVault()
        a.put("k", "low", confidence=0.5)
        b = ProfileVault()
        b.put("k", "high", confidence=0.9)
        merged = a.merge(b)
        self.assertEqual(merged.get("k").value, "high")


class FromFormResultsTests(unittest.TestCase):
    def test_learns_only_filled_values(self):
        from zfp.core.types import FilledValue

        class FakeReport:
            values = [
                FilledValue(field_name="a", canonical_key="k1", value="v1",
                           confidence=0.95, status="filled"),
                FilledValue(field_name="b", canonical_key="k2", value=None,
                           confidence=0.0, status="unavailable"),
            ]

        v = ProfileVault.from_form_results(FakeReport())
        self.assertEqual(v.get("k1").value, "v1")
        self.assertIsNone(v.get("k2"))


if __name__ == "__main__":
    unittest.main()
