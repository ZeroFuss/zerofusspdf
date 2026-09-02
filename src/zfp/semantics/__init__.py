"""The semantic layer: turning geometry into meaning.

Detection ends with a pile of rectangles that are *probably* fields.  This package
decides what each one means, using nothing but the page itself and the deterministic
:mod:`zfp.ontology`:

:mod:`zfp.semantics.graph`
    The spatial graph -- ``label --left-of--> field``, ``section --contains--> label``,
    ``radio --member-of--> group`` -- built once and consulted by everything else.
:mod:`zfp.semantics.sections`
    Section headings from typography alone, and the enclosing chain a field inherits as
    its ``parent_context``.
:mod:`zfp.semantics.linker`
    Which printed words name which blank, decided by one global assignment so a label
    is never claimed twice.
:mod:`zfp.semantics.typing`
    ``TEXT`` refined into ``DATE`` / ``CURRENCY`` / ``EMAIL`` / ``MULTILINE_TEXT`` from
    the label, a printed placeholder, the rectangle's height and the glyphs beside it.
:mod:`zfp.semantics.normalizer`
    The eight-step canonicalization cascade, plus the page-level conflict pass.
:mod:`zfp.semantics.repeats`
    The same question asked twice: grouping, value propagation and consistency checks.

The whole package is deterministic and offline.  Nothing here calls a model; when the
cascade cannot decide it returns ``None`` and leaves the escalation to
:mod:`zfp.council`.

A typical page runs::

    sections = detect_sections(spans, geometry)
    graph = build_graph(spans, candidates, sections, geometry)
    link_labels(candidates, spans, graph, config)
    infer_types(candidates, spans, config)
    canonicalize_all(candidates, graph, config, spans=spans)
    repeat_evidence(candidates)
"""

from __future__ import annotations

from .graph import (
    GRID_CELL_PT,
    MAX_EDGES_PER_RELATION,
    NODE_FIELD,
    NODE_LABEL,
    NODE_SECTION,
    Edge,
    Node,
    RelationKind,
    SpatialGraph,
    build_graph,
    decay_weight,
    spatial_relation,
)
from .linker import (
    BASE_SCORES,
    TOGGLE_SCORES,
    LabelLink,
    clean_label,
    link_labels,
    link_stem_labels,
    looks_like_value,
    score_label,
)
from .normalizer import (
    AMBIGUOUS_LABELS,
    CONFLICT_MARGIN,
    FUZZY_CUTOFF,
    canonicalize,
    canonicalize_all,
    disambiguate,
    neighbourhood_texts,
    sibling_key,
)
from .repeats import (
    check_consistency,
    find_repeated_fields,
    group_key,
    propagate,
    repeat_evidence,
)
from .sections import (
    SECTION_SCORE_THRESHOLD,
    Section,
    detect_sections,
    heading_score,
    section_for,
)
from .typing import (
    GEOMETRIC_TYPES,
    MULTILINE_HEIGHT_RATIO,
    TypeSignal,
    infer_field_type,
    infer_types,
    placeholder_near,
)

__all__ = [
    # graph
    "RelationKind",
    "Node",
    "Edge",
    "SpatialGraph",
    "build_graph",
    "spatial_relation",
    "decay_weight",
    "NODE_LABEL",
    "NODE_FIELD",
    "NODE_SECTION",
    "GRID_CELL_PT",
    "MAX_EDGES_PER_RELATION",
    # sections
    "Section",
    "detect_sections",
    "section_for",
    "heading_score",
    "SECTION_SCORE_THRESHOLD",
    # linker
    "link_labels",
    "link_stem_labels",
    "score_label",
    "looks_like_value",
    "clean_label",
    "LabelLink",
    "BASE_SCORES",
    "TOGGLE_SCORES",
    # typing
    "infer_field_type",
    "infer_types",
    "placeholder_near",
    "TypeSignal",
    "GEOMETRIC_TYPES",
    "MULTILINE_HEIGHT_RATIO",
    # normalizer
    "canonicalize",
    "canonicalize_all",
    "disambiguate",
    "neighbourhood_texts",
    "sibling_key",
    "AMBIGUOUS_LABELS",
    "FUZZY_CUTOFF",
    "CONFLICT_MARGIN",
    # repeats
    "find_repeated_fields",
    "propagate",
    "check_consistency",
    "repeat_evidence",
    "group_key",
]
