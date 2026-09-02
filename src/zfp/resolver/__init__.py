"""Normalization, validation, policy and autofill resolution.

Turns a vault value into the exact string a specific field needs, refuses to place a
value it cannot validate, and never invents one it does not have.
"""

from __future__ import annotations

from .autofill import AutofillResolver
from .normalizers import normalize
from .policy import FillPolicy, SigningPolicy, check_fill_allowed, check_signature_fill
from .validators import ValidationOutcome, validate, validate_against_constraints

__all__ = [
    "AutofillResolver", "normalize", "validate", "validate_against_constraints",
    "ValidationOutcome", "SigningPolicy", "FillPolicy", "check_signature_fill",
    "check_fill_allowed",
]
