"""Form-archetype detection: geometry in, :class:`~zfp.core.types.FieldCandidate` out.

:mod:`zfp.candidates.context` derives everything the detectors share about one page --
merged rules, boxes, circles, checkbox glyphs, table cells, comb runs, blank regions,
line-grouped text and a spatial index -- exactly once.  :mod:`zfp.candidates.archetypes`
holds the eleven detectors that turn those structures into candidates.

Typical use::

    ctx = build_context(page_index, geometry, spans, primitives, words, config)
    candidates = generate_candidates(ctx)
"""

from __future__ import annotations

from .archetypes import (
    DEFAULT_DETECTORS,
    ArchetypeDetector,
    BlankRegionDetector,
    BoxFieldDetector,
    CheckboxDetector,
    ColonRunDetector,
    CombFieldDetector,
    DateBoxDetector,
    FreeTextAreaDetector,
    RadioGroupDetector,
    SignatureLineDetector,
    TableCellDetector,
    UnderlineFieldDetector,
    clean_label,
    export_value_for,
    generate_candidates,
)
from .context import CandidateContext, build_context, detection_config, vision_call

__all__ = [
    "CandidateContext",
    "build_context",
    "detection_config",
    "vision_call",
    "ArchetypeDetector",
    "UnderlineFieldDetector",
    "BoxFieldDetector",
    "CheckboxDetector",
    "RadioGroupDetector",
    "CombFieldDetector",
    "DateBoxDetector",
    "SignatureLineDetector",
    "TableCellDetector",
    "BlankRegionDetector",
    "FreeTextAreaDetector",
    "ColonRunDetector",
    "DEFAULT_DETECTORS",
    "generate_candidates",
    "clean_label",
    "export_value_for",
]
