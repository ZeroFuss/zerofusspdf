"""Configuration objects for the whole engine.

Every knob the pipeline exposes lives here as a plain dataclass so a run can be
described, serialized, diffed and reproduced from a single JSON document.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from .errors import ValidationError
from .logging import get_logger

__all__ = [
    "DetectionConfig",
    "ScoringWeights",
    "OcrConfig",
    "PrivacyConfig",
    "CouncilConfig",
    "AutofillConfig",
    "OrchestratorConfig",
    "ZfpConfig",
    "SCORING_WEIGHT_NAMES",
]

_log = get_logger(__name__)

#: The seven evidence buckets, in contract order.
SCORING_WEIGHT_NAMES = (
    "geometric_evidence",
    "blank_region_evidence",
    "nearby_label_evidence",
    "layout_consistency",
    "repeated_pattern_evidence",
    "semantic_type_confidence",
    "model_consensus",
)


@dataclass
class DetectionConfig:
    """Geometric thresholds for the deterministic detectors (points unless stated)."""

    min_line_length_pt: float = 24.0
    max_line_thickness_pt: float = 3.0
    line_merge_tolerance_pt: float = 1.5
    field_height_pt: float = 12.0
    underline_gap_pt: float = 2.0
    label_max_distance_pt: float = 120.0
    checkbox_min_pt: float = 5.0
    checkbox_max_pt: float = 22.0
    checkbox_aspect_tolerance: float = 0.35
    blank_min_width_pt: float = 40.0
    blank_min_height_pt: float = 9.0
    comb_cell_tolerance_pt: float = 2.0
    min_candidate_confidence: float = 0.35
    dedup_iou_threshold: float = 0.55


@dataclass
class ScoringWeights:
    """Weights of the seven independent evidence buckets used to score a candidate."""

    geometric_evidence: float = 0.30
    blank_region_evidence: float = 0.20
    nearby_label_evidence: float = 0.15
    layout_consistency: float = 0.10
    repeated_pattern_evidence: float = 0.10
    semantic_type_confidence: float = 0.10
    model_consensus: float = 0.05

    def as_tuple(self) -> tuple:
        """Return the seven weights in contract order."""
        return tuple(getattr(self, name) for name in SCORING_WEIGHT_NAMES)

    def total(self) -> float:
        """Sum of the seven weights."""
        return float(sum(self.as_tuple()))

    def normalized(self) -> ScoringWeights:
        """Return a copy whose weights sum to exactly 1.0.

        Negative weights are clamped to zero. When every weight is zero the buckets are
        weighted uniformly (``1/7`` each) so scoring never divides by zero.
        """
        values = [max(0.0, float(v)) for v in self.as_tuple()]
        total = sum(values)
        if total <= 0.0:
            uniform = 1.0 / len(SCORING_WEIGHT_NAMES)
            values = [uniform] * len(SCORING_WEIGHT_NAMES)
        else:
            values = [v / total for v in values]
        return ScoringWeights(**dict(zip(SCORING_WEIGHT_NAMES, values)))

    def score(self, evidence: Mapping[str, float]) -> float:
        """Return ``sum(weight_i * evidence[name_i])`` using the normalized weights.

        Missing buckets contribute ``0.0``.
        """
        weights = self.normalized()
        total = 0.0
        for name in SCORING_WEIGHT_NAMES:
            try:
                value = float(evidence.get(name, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            total += getattr(weights, name) * value
        return total


@dataclass
class OcrConfig:
    """OCR cascade settings. OCR never runs on a page that already has native text."""

    enabled: bool = True
    dpi: int = 300
    engines: List[str] = field(default_factory=lambda: ["tesseract", "paddle"])
    min_word_confidence: float = 0.55
    escalate_below: float = 0.70
    languages: List[str] = field(default_factory=lambda: ["eng"])


@dataclass
class PrivacyConfig:
    """Egress policy. External inference is opt-in and off by default."""

    allow_external_inference: bool = False
    require_zero_data_retention: bool = True
    provider_allowlist: List[str] = field(default_factory=list)
    max_context_chars: int = 2000
    redact_values_in_prompts: bool = True


@dataclass
class CouncilConfig:
    """Ambiguity-council behaviour."""

    enabled: bool = True
    quorum: int = 3
    agreement_threshold: float = 0.66
    escalate_below_confidence: float = 0.80
    max_rounds: int = 2
    providers: List[str] = field(default_factory=lambda: ["rules", "heuristic", "ontology"])


@dataclass
class AutofillConfig:
    """Value-resolution policy. ``conservative`` never emits a low-confidence value."""

    mode: str = "conservative"
    min_fill_confidence: float = 0.90
    min_completion_confidence: float = 0.55
    propagate_repeats: bool = True
    require_validation: bool = True


@dataclass
class OrchestratorConfig:
    """Agent-mesh execution policy."""

    max_workers: int = 8
    page_shard_size: int = 4
    stage_timeout_s: float = 300.0
    fail_fast: bool = False
    deterministic: bool = True


_SECTIONS = {
    "detection": DetectionConfig,
    "scoring": ScoringWeights,
    "ocr": OcrConfig,
    "privacy": PrivacyConfig,
    "council": CouncilConfig,
    "autofill": AutofillConfig,
    "orchestrator": OrchestratorConfig,
}


def _section_from_dict(cls: type, data: Any, section: str) -> Any:
    """Build one config section, ignoring (but logging) unknown keys."""
    if data is None:
        return cls()
    if isinstance(data, cls):
        return data
    if not isinstance(data, Mapping):
        raise ValidationError("config section '%s' must be a mapping, got %r" % (section, type(data)))
    known = {f.name for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key in known:
            kwargs[key] = value
        else:
            _log.warning("ignoring unknown config key %s.%s", section, key)
    return cls(**kwargs)


@dataclass
class ZfpConfig:
    """The complete configuration of one ZFP run."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    scoring: ScoringWeights = field(default_factory=ScoringWeights)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    council: CouncilConfig = field(default_factory=CouncilConfig)
    autofill: AutofillConfig = field(default_factory=AutofillConfig)
    orchestrator: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    seed: int = 0

    @staticmethod
    def default() -> ZfpConfig:
        """Return the stock configuration."""
        return ZfpConfig()

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> ZfpConfig:
        """Build a configuration from a (possibly partial) mapping.

        Unknown keys are ignored with a warning so newer config files stay loadable.
        """
        if d is None:
            return ZfpConfig()
        if not isinstance(d, Mapping):
            raise ValidationError("ZfpConfig.from_dict expects a mapping, got %r" % (type(d),))
        kwargs: Dict[str, Any] = {}
        for section, cls in _SECTIONS.items():
            kwargs[section] = _section_from_dict(cls, d.get(section), section)
        seed = d.get("seed", 0)
        try:
            kwargs["seed"] = int(seed)
        except (TypeError, ValueError) as exc:
            raise ValidationError("config seed must be an integer, got %r" % (seed,)) from exc
        for key in d:
            if key not in _SECTIONS and key != "seed":
                _log.warning("ignoring unknown config key %s", key)
        return ZfpConfig(**kwargs)

    @staticmethod
    def from_file(path: str | os.PathLike) -> ZfpConfig:
        """Load a JSON configuration file.

        Raises:
            ValidationError: when the file is missing or is not valid JSON.
        """
        try:
            with open(os.fspath(path), encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError as exc:
            raise ValidationError("cannot read config file %s: %s" % (path, exc)) from exc
        except ValueError as exc:
            raise ValidationError("config file %s is not valid JSON: %s" % (path, exc)) from exc
        return ZfpConfig.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain nested dictionary suitable for ``json.dumps``."""
        out: Dict[str, Any] = {}
        for section in _SECTIONS:
            out[section] = dataclasses.asdict(getattr(self, section))
        out["seed"] = self.seed
        return out

    def as_dict(self) -> Dict[str, Any]:
        """Alias of :meth:`to_dict` for the common serialization protocol."""
        return self.to_dict()
