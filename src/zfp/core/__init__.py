"""``zfp.core`` — the shared foundation every other ZFP module imports.

Geometry, configuration, the type vocabulary, errors, logging, deterministic ids,
JSON serialization and optional-dependency discovery. Nothing in this package imports
a third-party library or another ZFP subpackage, so it is always safe to import.
"""

from __future__ import annotations

from .config import (
    SCORING_WEIGHT_NAMES,
    AutofillConfig,
    CouncilConfig,
    DetectionConfig,
    OcrConfig,
    OrchestratorConfig,
    PrivacyConfig,
    ScoringWeights,
    ZfpConfig,
)
from .errors import (
    AgentError,
    CouncilError,
    DetectionError,
    EncryptedDocumentError,
    PdfParseError,
    PdfWriteError,
    PolicyError,
    QAError,
    SemanticError,
    UnsupportedFeatureError,
    ValidationError,
    VaultError,
    ZfpError,
)
from .geometry import EPS, Matrix, PageGeometry, Point, Rect
from .ids import candidate_id, canonical_repr, stable_id
from .logging import LogContext, configure, get_logger
from .optional import OptionalModule, capability_report, have, optional_import
from .serde import dumps, loads, register_decoder, to_jsonable
from .types import (
    CONFIDENCE_WEIGHTS,
    EVIDENCE_BUCKETS,
    SCORE_KEYS,
    Confidence,
    DocumentClass,
    DocumentProfile,
    Evidence,
    EvidenceKind,
    FieldCandidate,
    FieldConstraints,
    FieldSpec,
    FieldType,
    FilledValue,
    FillReport,
    FormSchema,
    PageMode,
    PageProfile,
    RasterWord,
    TextSpan,
    VectorPrimitive,
)
from .units import (
    MM_PER_INCH,
    PT_PER_INCH,
    dpi_to_scale,
    inch_to_pt,
    mm_to_pt,
    pt_to_inch,
    pt_to_mm,
    pt_to_px,
    px_to_pt,
    scale_to_dpi,
)

__all__ = [
    # errors
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
    # geometry
    "EPS",
    "Point",
    "Rect",
    "Matrix",
    "PageGeometry",
    # units
    "PT_PER_INCH",
    "MM_PER_INCH",
    "pt_to_px",
    "px_to_pt",
    "mm_to_pt",
    "pt_to_mm",
    "inch_to_pt",
    "pt_to_inch",
    "dpi_to_scale",
    "scale_to_dpi",
    # optional deps
    "OptionalModule",
    "optional_import",
    "have",
    "capability_report",
    # logging
    "get_logger",
    "configure",
    "LogContext",
    # config
    "DetectionConfig",
    "ScoringWeights",
    "OcrConfig",
    "PrivacyConfig",
    "CouncilConfig",
    "AutofillConfig",
    "OrchestratorConfig",
    "ZfpConfig",
    "SCORING_WEIGHT_NAMES",
    # types
    "FieldType",
    "PageMode",
    "DocumentClass",
    "EvidenceKind",
    "Evidence",
    "Confidence",
    "FieldConstraints",
    "TextSpan",
    "VectorPrimitive",
    "RasterWord",
    "FieldCandidate",
    "PageProfile",
    "DocumentProfile",
    "FieldSpec",
    "FormSchema",
    "FilledValue",
    "FillReport",
    "SCORE_KEYS",
    "EVIDENCE_BUCKETS",
    "CONFIDENCE_WEIGHTS",
    # ids
    "stable_id",
    "candidate_id",
    "canonical_repr",
    # serde
    "to_jsonable",
    "dumps",
    "loads",
    "register_decoder",
]
