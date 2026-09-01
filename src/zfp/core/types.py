"""The shared vocabulary of the engine.

Every stage of the pipeline speaks in these types: perception produces
:class:`TextSpan`, :class:`VectorPrimitive` and :class:`RasterWord`; detection produces
:class:`FieldCandidate`; the writer consumes :class:`FieldSpec` inside a
:class:`FormSchema`; autofill reports :class:`FilledValue` inside a :class:`FillReport`.
All rectangles are PDF user space (y-up, points).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .geometry import PageGeometry, Point, Rect
from .serde import register_decoder, to_jsonable

__all__ = [
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
]


class FieldType(str, Enum):
    """The semantic type of a form field."""

    TEXT = "text"
    MULTILINE_TEXT = "multiline_text"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    CHOICE = "choice"
    LISTBOX = "listbox"
    SIGNATURE = "signature"
    DATE = "date"
    NUMBER = "number"
    CURRENCY = "currency"
    EMAIL = "email"
    PHONE = "phone"
    COMB = "comb"
    BUTTON = "button"
    UNKNOWN = "unknown"

    @property
    def pdf_kind(self) -> str:
        """Return the PDF ``/FT`` value: ``"Tx"``, ``"Btn"``, ``"Ch"`` or ``"Sig"``."""
        return _PDF_KIND[self]


_PDF_KIND: Dict[FieldType, str] = {}


class PageMode(str, Enum):
    """What a single page actually contains."""

    NATIVE_DOCUMENT = "native_document"
    FLAT_NATIVE_FORM = "flat_native_form"
    SCANNED_FORM = "scanned_form"
    SCANNED_DOCUMENT = "scanned_document"
    HYBRID = "hybrid"
    INTERACTIVE_FORM = "interactive_form"
    EMPTY = "empty"


class DocumentClass(str, Enum):
    """The routing decision for a whole document."""

    EXISTING_ACROFORM = "existing_acroform"
    FLAT_NATIVE_FORM = "flat_native_form"
    SCANNED_FORM = "scanned_form"
    HYBRID = "hybrid"
    XFA = "xfa"
    ENCRYPTED = "encrypted"
    SIGNED = "signed"
    NON_FORM = "non_form"


class EvidenceKind(str, Enum):
    """Where a piece of evidence for a candidate came from."""

    VECTOR_LINE = "vector_line"
    VECTOR_RECT = "vector_rect"
    VECTOR_CIRCLE = "vector_circle"
    NATIVE_TEXT = "native_text"
    OCR_TEXT = "ocr_text"
    BLANK_REGION = "blank_region"
    LABEL_LINK = "label_link"
    PATTERN = "pattern"
    LAYOUT = "layout"
    REPEAT = "repeat"
    MODEL = "model"
    EXISTING_WIDGET = "existing_widget"
    TABLE_CELL = "table_cell"
    COMB_CELL = "comb_cell"
    CHECKBOX_GLYPH = "checkbox_glyph"


_PDF_KIND.update(
    {
        FieldType.TEXT: "Tx",
        FieldType.MULTILINE_TEXT: "Tx",
        FieldType.DATE: "Tx",
        FieldType.NUMBER: "Tx",
        FieldType.CURRENCY: "Tx",
        FieldType.EMAIL: "Tx",
        FieldType.PHONE: "Tx",
        FieldType.COMB: "Tx",
        FieldType.CHECKBOX: "Btn",
        FieldType.RADIO: "Btn",
        FieldType.BUTTON: "Btn",
        FieldType.CHOICE: "Ch",
        FieldType.LISTBOX: "Ch",
        FieldType.SIGNATURE: "Sig",
        FieldType.UNKNOWN: "Tx",
    }
)

#: The seven scoring buckets, mirroring ``zfp.core.config.ScoringWeights`` field names.
SCORE_KEYS: Tuple[str, ...] = (
    "geometric_evidence",
    "blank_region_evidence",
    "nearby_label_evidence",
    "layout_consistency",
    "repeated_pattern_evidence",
    "semantic_type_confidence",
    "model_consensus",
)

#: Which scoring bucket each :class:`EvidenceKind` contributes to.
EVIDENCE_BUCKETS: Dict[EvidenceKind, str] = {
    EvidenceKind.VECTOR_LINE: "geometric_evidence",
    EvidenceKind.VECTOR_RECT: "geometric_evidence",
    EvidenceKind.VECTOR_CIRCLE: "geometric_evidence",
    EvidenceKind.EXISTING_WIDGET: "geometric_evidence",
    EvidenceKind.CHECKBOX_GLYPH: "geometric_evidence",
    EvidenceKind.TABLE_CELL: "geometric_evidence",
    EvidenceKind.COMB_CELL: "geometric_evidence",
    EvidenceKind.BLANK_REGION: "blank_region_evidence",
    EvidenceKind.LABEL_LINK: "nearby_label_evidence",
    EvidenceKind.LAYOUT: "layout_consistency",
    EvidenceKind.REPEAT: "repeated_pattern_evidence",
    EvidenceKind.PATTERN: "semantic_type_confidence",
    EvidenceKind.NATIVE_TEXT: "semantic_type_confidence",
    EvidenceKind.OCR_TEXT: "semantic_type_confidence",
    EvidenceKind.MODEL: "model_consensus",
}

#: Relative importance of the four confidence axes in :meth:`Confidence.overall`.
CONFIDENCE_WEIGHTS: Dict[str, float] = {
    "geometry": 0.35,
    "label_link": 0.25,
    "semantic_type": 0.25,
    "autofill_value": 0.15,
}


# --------------------------------------------------------------------------- helpers
def _as_rect(value: Any) -> Optional[Rect]:
    """Coerce a serialized rectangle back into a :class:`Rect`."""
    if value is None:
        return None
    if isinstance(value, Rect):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return Rect.from_list(value)
    raise TypeError("cannot interpret %r as a Rect" % (value,))


def _as_enum(enum_cls: Any, value: Any, default: Any) -> Any:
    """Coerce a serialized enum value, falling back to ``default``."""
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        return default


def _clamp01(value: float) -> float:
    """Clamp a score into ``[0, 1]``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


_PUNCT_TO_SPACE = str.maketrans({c: " " for c in "/\\|_–—−-"})
_PUNCT_DROP = re.compile(r"[^0-9a-z\s]")
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- evidence
@dataclass(frozen=True)
class Evidence:
    """One independent observation supporting a candidate."""

    kind: EvidenceKind
    score: float
    detail: str = ""
    source_agent: str = ""
    rect: Optional[Rect] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "kind": self.kind.value if isinstance(self.kind, EvidenceKind) else str(self.kind),
            "score": self.score,
            "detail": self.detail,
            "source_agent": self.source_agent,
            "rect": self.rect.as_list() if self.rect is not None else None,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Evidence:
        """Rebuild an :class:`Evidence` from :meth:`as_dict` output."""
        return Evidence(
            kind=_as_enum(EvidenceKind, d.get("kind"), EvidenceKind.LAYOUT),
            score=float(d.get("score", 0.0) or 0.0),
            detail=str(d.get("detail", "") or ""),
            source_agent=str(d.get("source_agent", "") or ""),
            rect=_as_rect(d.get("rect")),
        )


@dataclass
class Confidence:
    """Four independent confidence axes, never collapsed into one opaque number."""

    geometry: float = 0.0
    label_link: float = 0.0
    semantic_type: float = 0.0
    autofill_value: float = 0.0

    def overall(self) -> float:
        """Return the weighted geometric mean of the **non-zero** axes.

        ``overall = exp( sum(w_i * ln(c_i)) / sum(w_i) )`` taken over the axes with
        ``c_i > 0`` and weights ``geometry=0.35, label_link=0.25, semantic_type=0.25,
        autofill_value=0.15`` (see :data:`CONFIDENCE_WEIGHTS`).

        An axis of exactly ``0.0`` means *not measured* (it is the field default) and is
        excluded, so an unmeasured axis cannot silently annihilate the whole score the
        way a plain geometric mean would. A geometric mean is used instead of an
        arithmetic one because it punishes a single weak axis much harder: a candidate
        with perfect geometry but a poor label link is not a good candidate.
        Returns ``0.0`` when no axis has been measured.
        """
        num = 0.0
        den = 0.0
        for name, weight in CONFIDENCE_WEIGHTS.items():
            value = _clamp01(getattr(self, name))
            if value <= 0.0:
                continue
            num += weight * math.log(value)
            den += weight
        if den <= 0.0:
            return 0.0
        return math.exp(num / den)

    def as_dict(self) -> Dict[str, float]:
        """Return the four axes as plain floats."""
        return {
            "geometry": self.geometry,
            "label_link": self.label_link,
            "semantic_type": self.semantic_type,
            "autofill_value": self.autofill_value,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> Confidence:
        """Rebuild a :class:`Confidence` from :meth:`as_dict` output."""
        return Confidence(
            geometry=float(d.get("geometry", 0.0) or 0.0),
            label_link=float(d.get("label_link", 0.0) or 0.0),
            semantic_type=float(d.get("semantic_type", 0.0) or 0.0),
            autofill_value=float(d.get("autofill_value", 0.0) or 0.0),
        )


@dataclass
class FieldConstraints:
    """Everything the writer needs to constrain a field's input."""

    max_chars_estimate: Optional[int] = None
    required: bool = False
    multiline: bool = False
    comb_cells: Optional[int] = None
    choices: List[str] = field(default_factory=list)
    pattern: Optional[str] = None
    format_hint: Optional[str] = None
    read_only: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "max_chars_estimate": self.max_chars_estimate,
            "required": self.required,
            "multiline": self.multiline,
            "comb_cells": self.comb_cells,
            "choices": list(self.choices),
            "pattern": self.pattern,
            "format_hint": self.format_hint,
            "read_only": self.read_only,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> FieldConstraints:
        """Rebuild a :class:`FieldConstraints` from :meth:`as_dict` output."""
        return FieldConstraints(
            max_chars_estimate=d.get("max_chars_estimate"),
            required=bool(d.get("required", False)),
            multiline=bool(d.get("multiline", False)),
            comb_cells=d.get("comb_cells"),
            choices=list(d.get("choices") or []),
            pattern=d.get("pattern"),
            format_hint=d.get("format_hint"),
            read_only=bool(d.get("read_only", False)),
        )


# --------------------------------------------------------------------------- perception
@dataclass
class TextSpan:
    """A run of text with a user-space bounding box, from the parser or from OCR."""

    text: str
    rect: Rect
    page: int
    font_name: str = ""
    font_size: float = 0.0
    source: str = "native"
    confidence: float = 1.0
    glyph_rects: List[Rect] = field(default_factory=list)
    baseline: Optional[float] = None

    def is_blank(self) -> bool:
        """True when the span carries no visible characters."""
        return not self.text or not self.text.strip()

    def normalized_text(self) -> str:
        """Lowercase, punctuation-stripped, whitespace-collapsed form of the text.

        Separator punctuation (``/ \\ | _ -`` and dashes) becomes a space; all other
        punctuation is dropped outright, so ``"E-Mail:"`` -> ``"e mail"`` and
        ``"Applicant's Name*"`` -> ``"applicants name"``.
        """
        lowered = (self.text or "").lower().translate(_PUNCT_TO_SPACE)
        dropped = _PUNCT_DROP.sub("", lowered)
        return _WS.sub(" ", dropped).strip()

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "text": self.text,
            "rect": self.rect.as_list(),
            "page": self.page,
            "font_name": self.font_name,
            "font_size": self.font_size,
            "source": self.source,
            "confidence": self.confidence,
            "glyph_rects": [r.as_list() for r in self.glyph_rects],
            "baseline": self.baseline,
        }


@dataclass
class VectorPrimitive:
    """A stroked or filled path element extracted from a content stream."""

    kind: str
    rect: Rect
    page: int
    stroke_width: float = 0.0
    filled: bool = False
    stroked: bool = True
    points: List[Point] = field(default_factory=list)

    #: A primitive counts as a rule when one side is this many times the other.
    ORIENTATION_RATIO = 3.0

    def orientation(self) -> str:
        """Return ``"horizontal"``, ``"vertical"`` or ``"other"``.

        A primitive is horizontal when its width is at least
        :data:`ORIENTATION_RATIO` times its height (and vice versa); anything squarer
        is ``"other"``.
        """
        w, h = self.rect.width, self.rect.height
        if w <= 0.0 and h <= 0.0:
            return "other"
        if h <= 0.0 or w >= h * self.ORIENTATION_RATIO:
            return "horizontal"
        if w <= 0.0 or h >= w * self.ORIENTATION_RATIO:
            return "vertical"
        return "other"

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "kind": self.kind,
            "rect": self.rect.as_list(),
            "page": self.page,
            "stroke_width": self.stroke_width,
            "filled": self.filled,
            "stroked": self.stroked,
            "points": [list(p.as_tuple()) for p in self.points],
            "orientation": self.orientation(),
        }


@dataclass
class RasterWord:
    """One OCR word, already converted from pixels into PDF user space."""

    text: str
    rect: Rect
    confidence: float
    page: int
    line_id: int = -1
    block_id: int = -1
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "text": self.text,
            "rect": self.rect.as_list(),
            "confidence": self.confidence,
            "page": self.page,
            "line_id": self.line_id,
            "block_id": self.block_id,
            "alternatives": [[a, float(s)] for a, s in self.alternatives],
        }


# --------------------------------------------------------------------------- detection
@dataclass
class FieldCandidate:
    """A proposed interactive field, before it becomes a writer :class:`FieldSpec`."""

    id: str
    page: int
    rect: Rect
    field_type: FieldType = FieldType.UNKNOWN
    sources: List[str] = field(default_factory=list)
    visible_label: Optional[str] = None
    canonical_key: Optional[str] = None
    parent_context: List[str] = field(default_factory=list)
    confidence: Confidence = field(default_factory=Confidence)
    constraints: FieldConstraints = field(default_factory=FieldConstraints)
    evidence: List[Evidence] = field(default_factory=list)
    group_id: Optional[str] = None
    export_value: Optional[str] = None
    order: int = 0

    def add_evidence(self, e: Evidence) -> None:
        """Append evidence and record its kind in :attr:`sources` (first occurrence wins)."""
        self.evidence.append(e)
        kind = e.kind.value if isinstance(e.kind, EvidenceKind) else str(e.kind)
        if kind not in self.sources:
            self.sources.append(kind)

    def evidence_scores(self) -> Dict[str, float]:
        """Collapse accumulated evidence into the seven :data:`SCORE_KEYS` buckets.

        Each bucket takes the **maximum** score of the evidence mapped onto it (see
        :data:`EVIDENCE_BUCKETS`); buckets with no evidence stay at ``0.0``. The result
        is exactly the mapping ``ScoringWeights.score`` consumes.
        """
        scores = {key: 0.0 for key in SCORE_KEYS}
        for item in self.evidence:
            bucket = EVIDENCE_BUCKETS.get(item.kind)
            if bucket is None:
                continue
            value = _clamp01(item.score)
            if value > scores[bucket]:
                scores[bucket] = value
        return scores

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "id": self.id,
            "page": self.page,
            "rect": self.rect.as_list(),
            "field_type": self.field_type.value,
            "sources": list(self.sources),
            "visible_label": self.visible_label,
            "canonical_key": self.canonical_key,
            "parent_context": list(self.parent_context),
            "confidence": self.confidence.as_dict(),
            "constraints": self.constraints.as_dict(),
            "evidence": [e.as_dict() for e in self.evidence],
            "group_id": self.group_id,
            "export_value": self.export_value,
            "order": self.order,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> FieldCandidate:
        """Rebuild a :class:`FieldCandidate` from :meth:`as_dict` output."""
        return FieldCandidate(
            id=str(d.get("id", "")),
            page=int(d.get("page", 0)),
            rect=_as_rect(d.get("rect")) or Rect(0.0, 0.0, 0.0, 0.0),
            field_type=_as_enum(FieldType, d.get("field_type"), FieldType.UNKNOWN),
            sources=list(d.get("sources") or []),
            visible_label=d.get("visible_label"),
            canonical_key=d.get("canonical_key"),
            parent_context=list(d.get("parent_context") or []),
            confidence=Confidence.from_dict(d.get("confidence") or {}),
            constraints=FieldConstraints.from_dict(d.get("constraints") or {}),
            evidence=[Evidence.from_dict(e) for e in (d.get("evidence") or [])],
            group_id=d.get("group_id"),
            export_value=d.get("export_value"),
            order=int(d.get("order", 0)),
        )


# --------------------------------------------------------------------------- profiles
@dataclass
class PageProfile:
    """What preflight learned about one page."""

    index: int
    geometry: PageGeometry
    mode: PageMode
    has_native_text: bool = False
    has_raster: bool = False
    has_vector: bool = False
    has_widgets: bool = False
    char_count: int = 0
    image_area_ratio: float = 0.0
    vector_op_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "index": self.index,
            "geometry": to_jsonable(self.geometry),
            "mode": self.mode.value,
            "has_native_text": self.has_native_text,
            "has_raster": self.has_raster,
            "has_vector": self.has_vector,
            "has_widgets": self.has_widgets,
            "char_count": self.char_count,
            "image_area_ratio": self.image_area_ratio,
            "vector_op_count": self.vector_op_count,
        }


@dataclass
class DocumentProfile:
    """What preflight learned about a whole document."""

    document_id: str
    page_count: int
    pages: List[PageProfile]
    encrypted: bool = False
    can_modify: bool = True
    signed: bool = False
    acroform: bool = False
    xfa: bool = False
    dynamic_xfa: bool = False
    tagged: bool = False
    producer: str = ""
    version: str = ""
    doc_class: DocumentClass = DocumentClass.NON_FORM
    warnings: List[str] = field(default_factory=list)

    @property
    def native_text_pages(self) -> List[int]:
        """Indices of pages carrying extractable native text."""
        return [p.index for p in self.pages if p.has_native_text]

    @property
    def raster_pages(self) -> List[int]:
        """Indices of pages carrying raster imagery."""
        return [p.index for p in self.pages if p.has_raster]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "document_id": self.document_id,
            "page_count": self.page_count,
            "pages": [p.as_dict() for p in self.pages],
            "encrypted": self.encrypted,
            "can_modify": self.can_modify,
            "signed": self.signed,
            "acroform": self.acroform,
            "xfa": self.xfa,
            "dynamic_xfa": self.dynamic_xfa,
            "tagged": self.tagged,
            "producer": self.producer,
            "version": self.version,
            "doc_class": self.doc_class.value,
            "warnings": list(self.warnings),
            "native_text_pages": self.native_text_pages,
            "raster_pages": self.raster_pages,
        }


# --------------------------------------------------------------------------- output
@dataclass
class FieldSpec:
    """Writer input. One entry == one AcroForm field (may own several widgets)."""

    name: str
    field_type: FieldType
    page: int
    rect: Rect
    canonical_key: Optional[str] = None
    value: Optional[str] = None
    default_value: Optional[str] = None
    tooltip: Optional[str] = None
    required: bool = False
    read_only: bool = False
    max_length: Optional[int] = None
    multiline: bool = False
    comb_cells: Optional[int] = None
    choices: List[str] = field(default_factory=list)
    export_value: Optional[str] = None
    font_name: str = "Helv"
    font_size: float = 0.0
    text_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    border_color: Optional[Tuple[float, float, float]] = None
    background_color: Optional[Tuple[float, float, float]] = None
    border_width: float = 0.0
    alignment: int = 0
    group: Optional[str] = None
    tab_order: int = 0
    extra_widgets: List[Tuple[int, Rect]] = field(default_factory=list)

    @property
    def pdf_kind(self) -> str:
        """The PDF ``/FT`` value for this field."""
        return self.field_type.pdf_kind

    def widgets(self) -> List[Tuple[int, Rect]]:
        """Return every ``(page, rect)`` widget owned by this field, primary first."""
        return [(self.page, self.rect)] + [(int(p), r) for p, r in self.extra_widgets]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "name": self.name,
            "field_type": self.field_type.value,
            "page": self.page,
            "rect": self.rect.as_list(),
            "canonical_key": self.canonical_key,
            "value": self.value,
            "default_value": self.default_value,
            "tooltip": self.tooltip,
            "required": self.required,
            "read_only": self.read_only,
            "max_length": self.max_length,
            "multiline": self.multiline,
            "comb_cells": self.comb_cells,
            "choices": list(self.choices),
            "export_value": self.export_value,
            "font_name": self.font_name,
            "font_size": self.font_size,
            "text_color": list(self.text_color),
            "border_color": list(self.border_color) if self.border_color else None,
            "background_color": list(self.background_color) if self.background_color else None,
            "border_width": self.border_width,
            "alignment": self.alignment,
            "group": self.group,
            "tab_order": self.tab_order,
            "extra_widgets": [[int(p), r.as_list()] for p, r in self.extra_widgets],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> FieldSpec:
        """Rebuild a :class:`FieldSpec` from :meth:`as_dict` output."""

        def _color(value: Any) -> Optional[Tuple[float, float, float]]:
            if value is None:
                return None
            seq = [float(v) for v in value]
            return (seq[0], seq[1], seq[2])

        return FieldSpec(
            name=str(d.get("name", "")),
            field_type=_as_enum(FieldType, d.get("field_type"), FieldType.UNKNOWN),
            page=int(d.get("page", 0)),
            rect=_as_rect(d.get("rect")) or Rect(0.0, 0.0, 0.0, 0.0),
            canonical_key=d.get("canonical_key"),
            value=d.get("value"),
            default_value=d.get("default_value"),
            tooltip=d.get("tooltip"),
            required=bool(d.get("required", False)),
            read_only=bool(d.get("read_only", False)),
            max_length=d.get("max_length"),
            multiline=bool(d.get("multiline", False)),
            comb_cells=d.get("comb_cells"),
            choices=list(d.get("choices") or []),
            export_value=d.get("export_value"),
            font_name=str(d.get("font_name", "Helv")),
            font_size=float(d.get("font_size", 0.0) or 0.0),
            text_color=_color(d.get("text_color")) or (0.0, 0.0, 0.0),
            border_color=_color(d.get("border_color")),
            background_color=_color(d.get("background_color")),
            border_width=float(d.get("border_width", 0.0) or 0.0),
            alignment=int(d.get("alignment", 0) or 0),
            group=d.get("group"),
            tab_order=int(d.get("tab_order", 0) or 0),
            extra_widgets=[
                (int(p), _as_rect(r) or Rect(0.0, 0.0, 0.0, 0.0))
                for p, r in (d.get("extra_widgets") or [])
            ],
        )


@dataclass
class FormSchema:
    """The complete set of fields to write into a document."""

    document_id: str
    fields: List[FieldSpec] = field(default_factory=list)
    source_candidates: List[FieldCandidate] = field(default_factory=list)

    def by_name(self, name: str) -> Optional[FieldSpec]:
        """Return the field with this fully qualified name, or ``None``."""
        for spec in self.fields:
            if spec.name == name:
                return spec
        return None

    def by_page(self, page: int) -> List[FieldSpec]:
        """Return every field whose primary or extra widget lives on ``page``."""
        out: List[FieldSpec] = []
        for spec in self.fields:
            if spec.page == page or any(int(p) == page for p, _ in spec.extra_widgets):
                out.append(spec)
        return out

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "document_id": self.document_id,
            "fields": [f.as_dict() for f in self.fields],
            "source_candidates": [c.as_dict() for c in self.source_candidates],
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> FormSchema:
        """Rebuild a :class:`FormSchema` from :meth:`as_dict` output."""
        return FormSchema(
            document_id=str(d.get("document_id", "")),
            fields=[FieldSpec.from_dict(f) for f in (d.get("fields") or [])],
            source_candidates=[
                FieldCandidate.from_dict(c) for c in (d.get("source_candidates") or [])
            ],
        )


@dataclass
class FilledValue:
    """The outcome of resolving one field against the vault."""

    field_name: str
    canonical_key: Optional[str]
    value: Optional[str]
    confidence: float
    provenance: Dict[str, Any] = field(default_factory=dict)
    status: str = "filled"
    reason_codes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "field_name": self.field_name,
            "canonical_key": self.canonical_key,
            "value": self.value,
            "confidence": self.confidence,
            "provenance": to_jsonable(self.provenance),
            "status": self.status,
            "reason_codes": list(self.reason_codes),
        }


@dataclass
class FillReport:
    """The result of an autofill pass over a whole schema."""

    document_id: str
    values: List[FilledValue] = field(default_factory=list)
    filled_count: int = 0
    unresolved_count: int = 0

    def recount(self) -> FillReport:
        """Recompute :attr:`filled_count` / :attr:`unresolved_count` from :attr:`values`."""
        self.filled_count = sum(1 for v in self.values if v.status == "filled")
        self.unresolved_count = len(self.values) - self.filled_count
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "document_id": self.document_id,
            "values": [v.as_dict() for v in self.values],
            "filled_count": self.filled_count,
            "unresolved_count": self.unresolved_count,
        }


register_decoder(FormSchema, lambda d: FormSchema.from_dict(d))
register_decoder(FieldSpec, lambda d: FieldSpec.from_dict(d))
register_decoder(FieldCandidate, lambda d: FieldCandidate.from_dict(d))
