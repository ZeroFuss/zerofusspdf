"""Exception hierarchy for ZFP.

Every error raised anywhere inside :mod:`zfp` MUST be a subclass of :class:`ZfpError`
so callers can guard the whole engine with a single ``except``.
"""

from __future__ import annotations

__all__ = [
    "ZfpError",
    "PdfParseError",
    "PdfWriteError",
    "EncryptedDocumentError",
    "UnsupportedFeatureError",
    "DetectionError",
    "SemanticError",
    "VaultError",
    "ValidationError",
    "PolicyError",
    "AgentError",
    "CouncilError",
    "QAError",
]


class ZfpError(Exception):
    """Base class for every error raised by ZFP."""


class PdfParseError(ZfpError):
    """The input bytes could not be understood as a PDF."""


class PdfWriteError(ZfpError):
    """A PDF could not be serialized."""


class EncryptedDocumentError(ZfpError):
    """The document is encrypted and could not be decrypted with the given password."""


class UnsupportedFeatureError(ZfpError):
    """An optional capability (dependency, backend, binary) is unavailable."""


class DetectionError(ZfpError):
    """Field detection failed irrecoverably."""


class SemanticError(ZfpError):
    """Semantic interpretation (labels, ontology, typing) failed."""


class VaultError(ZfpError):
    """The profile vault could not be read, written, or decrypted."""


class ValidationError(ZfpError):
    """A value or structure failed validation."""


class PolicyError(ZfpError):
    """A privacy, signing, or egress policy refused the operation."""


class AgentError(ZfpError):
    """An agent failed while executing a task."""


class CouncilError(ZfpError):
    """The ambiguity council could not produce a verdict."""


class QAError(ZfpError):
    """Post-write verification failed to run."""
