"""Policy gates: what ZFP is allowed to fill, and under what authorization.

Two independent boundaries live here. A :class:`SigningPolicy` gates applying a legally
meaningful signature -- ZFP identifies signature fields and prepopulates authorized
identity metadata around them, but never signs without one. A :class:`FillPolicy` gates
placing secret-sensitivity values (SSNs, card numbers) into any field at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..core.types import FieldSpec, FieldType


@dataclass
class SigningPolicy:
    """Authorization to apply a signature value, as opposed to merely creating the field."""

    allow_autosign: bool = False
    authorized_signers: Tuple[str, ...] = ()
    require_explicit_consent: bool = True


@dataclass
class FillPolicy:
    allow_secret_fields: bool = False
    allow_derived: bool = True
    blocked_keys: Tuple[str, ...] = ()


def check_signature_fill(spec: FieldSpec, policy: Optional[SigningPolicy]
                         ) -> Tuple[bool, str]:
    """Whether a signature field may receive an actual signature value."""
    if spec.field_type is not FieldType.SIGNATURE:
        return True, "not a signature field"
    if policy is None:
        return False, "no SigningPolicy provided; signature fields are never auto-signed"
    if not policy.allow_autosign:
        return False, "SigningPolicy.allow_autosign is False"
    if policy.require_explicit_consent and not policy.authorized_signers:
        return False, "SigningPolicy requires explicit consent but names no authorized signer"
    return True, "authorized by SigningPolicy"


def check_fill_allowed(spec: FieldSpec, key: Optional[str], vault_entry: Optional[object],
                       policy: Optional[FillPolicy] = None,
                       *, signing_policy: Optional[SigningPolicy] = None) -> Tuple[bool, str]:
    """The single policy gate :class:`~zfp.resolver.autofill.AutofillResolver` calls
    before writing any value."""
    policy = policy or FillPolicy()

    if spec.field_type is FieldType.SIGNATURE:
        return check_signature_fill(spec, signing_policy)

    if key and key in policy.blocked_keys:
        return False, "canonical key %r is explicitly blocked" % key

    sensitivity = getattr(vault_entry, "sensitivity", None)
    if sensitivity == "secret" and not policy.allow_secret_fields:
        return False, "value is secret-sensitivity; FillPolicy.allow_secret_fields is False"

    source = getattr(vault_entry, "source", None)
    if source == "derived" and not policy.allow_derived:
        return False, "value is a derived/synthesized value; FillPolicy.allow_derived is False"

    return True, "allowed"


__all__ = ["SigningPolicy", "FillPolicy", "check_signature_fill", "check_fill_allowed"]
