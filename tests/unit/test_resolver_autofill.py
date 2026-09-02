"""Unit tests for :mod:`zfp.resolver.autofill`."""

from __future__ import annotations

import unittest

from zfp.core.config import ZfpConfig
from zfp.core.geometry import Rect
from zfp.core.types import FieldSpec, FieldType, FormSchema
from zfp.resolver.autofill import AutofillResolver
from zfp.resolver.policy import FillPolicy, SigningPolicy
from zfp.vault.store import ProfileVault


def _spec(field_type=FieldType.TEXT, canonical_key=None, **kw):
    return FieldSpec(name=kw.pop("name", "f"), field_type=field_type, page=0,
                     rect=Rect(0, 0, 100, 14), canonical_key=canonical_key, **kw)


class BasicResolutionTests(unittest.TestCase):
    def test_resolves_from_vault_and_normalizes(self):
        vault = ProfileVault()
        vault.put("person.phone.mobile", "2125550134", confidence=1.0)
        resolver = AutofillResolver(vault, ZfpConfig.default())
        result = resolver.resolve_field(_spec(canonical_key="person.phone.mobile"))
        self.assertEqual(result.status, "filled")
        self.assertEqual(result.value, "(212) 555-0134")

    def test_no_vault_entry_is_unavailable_never_invented(self):
        vault = ProfileVault()
        resolver = AutofillResolver(vault, ZfpConfig.default())
        result = resolver.resolve_field(_spec(canonical_key="person.name.first"))
        self.assertEqual(result.status, "unavailable")
        self.assertIsNone(result.value)


class ConservativeModeTests(unittest.TestCase):
    def test_low_confidence_value_is_withheld_not_written(self):
        vault = ProfileVault()
        vault.put("person.name.first", "Jane", confidence=0.5)  # below 0.90 default
        resolver = AutofillResolver(vault, ZfpConfig.default())
        result = resolver.resolve_field(_spec(canonical_key="person.name.first"))
        self.assertEqual(result.status, "low_confidence")
        self.assertIsNone(result.value)
        self.assertNotIn("Jane", str(result.provenance))

    def test_high_confidence_value_is_filled(self):
        vault = ProfileVault()
        vault.put("person.name.first", "Jane", confidence=0.99)
        resolver = AutofillResolver(vault, ZfpConfig.default())
        result = resolver.resolve_field(_spec(canonical_key="person.name.first"))
        self.assertEqual(result.status, "filled")
        self.assertEqual(result.value, "Jane")


class ValidationTests(unittest.TestCase):
    def test_invalid_value_is_not_written(self):

        vault = ProfileVault()
        vault.put("person.email", "not-an-email", confidence=1.0)
        resolver = AutofillResolver(vault, ZfpConfig.default())
        result = resolver.resolve_field(_spec(canonical_key="person.email"))
        self.assertEqual(result.status, "invalid")


class SignatureFieldTests(unittest.TestCase):
    def test_blocked_without_signing_policy(self):
        vault = ProfileVault()
        resolver = AutofillResolver(vault, ZfpConfig.default())
        result = resolver.resolve_field(_spec(field_type=FieldType.SIGNATURE))
        self.assertEqual(result.status, "policy_blocked")

    def test_allowed_signing_policy_falls_through_to_normal_resolution(self):
        vault = ProfileVault()
        policy = SigningPolicy(allow_autosign=True, authorized_signers=("jane",))
        resolver = AutofillResolver(vault, ZfpConfig.default(), signing_policy=policy)
        result = resolver.resolve_field(_spec(field_type=FieldType.SIGNATURE))
        self.assertNotEqual(result.status, "policy_blocked")


class SecretFieldTests(unittest.TestCase):
    def test_secret_class_key_blocked_by_default(self):
        vault = ProfileVault()
        vault.put("person.ssn", "123456789", confidence=1.0, sensitivity="secret")
        resolver = AutofillResolver(vault, ZfpConfig.default())
        result = resolver.resolve_field(_spec(canonical_key="person.ssn"))
        self.assertEqual(result.status, "policy_blocked")

    def test_secret_class_key_allowed_with_explicit_policy(self):
        vault = ProfileVault()
        vault.put("person.ssn", "123456789", confidence=1.0, sensitivity="secret")
        resolver = AutofillResolver(vault, ZfpConfig.default(),
                                    fill_policy=FillPolicy(allow_secret_fields=True))
        result = resolver.resolve_field(_spec(canonical_key="person.ssn"))
        self.assertEqual(result.status, "filled")


class RepeatPropagationTests(unittest.TestCase):
    def test_three_instances_of_one_key_all_receive_the_same_value(self):
        from zfp.core.types import FieldCandidate

        vault = ProfileVault()
        vault.put("person.name.first", "Jane", confidence=1.0)
        specs = [
            _spec(name="f1", canonical_key="person.name.first"),
            _spec(name="f2", canonical_key="person.name.first"),
            _spec(name="f3", canonical_key="person.name.first"),
        ]
        candidates = [
            FieldCandidate(id=s.name, page=0, rect=s.rect, field_type=s.field_type,
                          canonical_key=s.canonical_key) for s in specs
        ]
        schema = FormSchema(document_id="d", fields=specs, source_candidates=candidates)
        resolver = AutofillResolver(vault, ZfpConfig.default())
        report = resolver.resolve_schema(schema, candidates)
        values = {v.field_name: v.value for v in report.values if v.field_name in
                 ("f1", "f2", "f3")}
        self.assertEqual(set(values.values()), {"Jane"})


class SummaryTests(unittest.TestCase):
    def test_summary_by_status_counts_correctly(self):
        vault = ProfileVault()
        vault.put("person.name.first", "Jane", confidence=1.0)
        specs = [
            _spec(name="a", canonical_key="person.name.first"),
            _spec(name="b", canonical_key="person.name.last"),
        ]
        schema = FormSchema(document_id="d", fields=specs)
        resolver = AutofillResolver(vault, ZfpConfig.default())
        report = resolver.resolve_schema(schema)
        summary = resolver.summary_by_status(report)
        self.assertEqual(summary.get("filled"), 1)
        self.assertEqual(summary.get("unavailable"), 1)


if __name__ == "__main__":
    unittest.main()
