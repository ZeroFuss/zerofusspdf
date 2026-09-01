# ZFP Interface Contract (normative)

This document is **normative**. Every module in `src/zfp/` MUST conform to the signatures
below. Parallel implementers own disjoint file sets and integrate against this contract only.

Rules that hold everywhere:

1. Python **>= 3.9**. Every module starts with `from __future__ import annotations`.
2. **Zero mandatory third-party dependencies.** The whole core path (parse -> detect ->
   write -> fill -> verify) MUST work on a bare CPython stdlib. Third-party libraries
   (`pikepdf`, `pypdf`, `opencv`, `pytesseract`, `paddleocr`, `numpy`, `Pillow`,
   `requests`, `fastapi`) are **optional adapters** discovered at runtime through
   `zfp.core.optional.optional_import`. A missing optional dep MUST degrade, never crash.
3. **Determinism.** No `random` without a seeded `random.Random`. No wall-clock in any
   value that lands in a PDF or a comparison. Parallel results are always re-sorted into a
   stable order before returning.
4. **Coordinates.** All rectangles crossing a module boundary are PDF user space, y-up,
   origin at the *page* origin, `x0<=x1`, `y0<=y1`, floats in points. Raster/pixel space
   never escapes `zfp.raster`/`zfp.ocr`/`zfp.vision`; those modules convert via
   `PageGeometry` before returning.
5. **Errors.** Raise only subclasses of `zfp.core.errors.ZfpError`.
6. **Logging.** `zfp.core.logging.get_logger(__name__)`. Never `print` outside `zfp.cli`.
7. **Typing.** Public functions are fully annotated. Dataclasses use `@dataclass(slots=False)`
   for 3.9 compatibility; frozen where stated.
8. **Tests.** Every module ships `tests/unit/test_<module>.py` using plain `unittest`
   (pytest runs it fine, but no pytest-only features). No network in tests. No sleeps.

---

## 1. `zfp.core`

### `zfp/core/errors.py`
```python
class ZfpError(Exception): ...
class PdfParseError(ZfpError): ...
class PdfWriteError(ZfpError): ...
class EncryptedDocumentError(ZfpError): ...
class UnsupportedFeatureError(ZfpError): ...
class DetectionError(ZfpError): ...
class SemanticError(ZfpError): ...
class VaultError(ZfpError): ...
class ValidationError(ZfpError): ...
class PolicyError(ZfpError): ...          # privacy / signing / egress refusal
class AgentError(ZfpError): ...
class CouncilError(ZfpError): ...
class QAError(ZfpError): ...
```

### `zfp/core/geometry.py`
```python
EPS: float = 1e-6

@dataclass(frozen=True)
class Point:
    x: float; y: float
    def translated(self, dx: float, dy: float) -> Point: ...
    def distance_to(self, other: Point) -> float: ...
    def as_tuple(self) -> tuple[float, float]: ...

@dataclass(frozen=True)
class Rect:
    x0: float; y0: float; x1: float; y1: float
    @staticmethod
    def from_points(a: Point, b: Point) -> Rect: ...
    @staticmethod
    def from_list(v: Sequence[float]) -> Rect: ...          # normalizes
    @staticmethod
    def bounding(rects: Iterable[Rect]) -> Optional[Rect]: ...
    @property
    def width(self) -> float: ...
    @property
    def height(self) -> float: ...
    @property
    def area(self) -> float: ...
    @property
    def center(self) -> Point: ...
    def normalized(self) -> Rect: ...                        # x0<=x1, y0<=y1
    def inflated(self, dx: float, dy: float = None) -> Rect: ...
    def translated(self, dx: float, dy: float) -> Rect: ...
    def scaled(self, sx: float, sy: float = None) -> Rect: ...
    def union(self, other: Rect) -> Rect: ...
    def intersection(self, other: Rect) -> Optional[Rect]: ...
    def iou(self, other: Rect) -> float: ...
    def contains_point(self, p: Point) -> bool: ...
    def contains_rect(self, other: Rect) -> bool: ...
    def intersects(self, other: Rect) -> bool: ...
    def horizontal_overlap(self, other: Rect) -> float: ...  # overlap length in points
    def vertical_overlap(self, other: Rect) -> float: ...
    def as_list(self) -> list[float]: ...                    # [x0,y0,x1,y1]
    def rounded(self, ndigits: int = 3) -> Rect: ...

@dataclass(frozen=True)
class Matrix:
    a: float=1; b: float=0; c: float=0; d: float=1; e: float=0; f: float=0
    @staticmethod
    def identity() -> Matrix: ...
    @staticmethod
    def translation(tx: float, ty: float) -> Matrix: ...
    @staticmethod
    def scaling(sx: float, sy: float) -> Matrix: ...
    @staticmethod
    def rotation(degrees: float) -> Matrix: ...
    def concat(self, other: Matrix) -> Matrix: ...   # self THEN other  (self x other)
    def apply(self, p: Point) -> Point: ...
    def apply_xy(self, x: float, y: float) -> tuple[float, float]: ...
    def transform_rect(self, r: Rect) -> Rect: ...   # axis-aligned bbox of 4 corners
    def inverted(self) -> Matrix: ...
    def as_tuple(self) -> tuple[float,...]: ...

@dataclass(frozen=True)
class PageGeometry:
    """Everything needed to move between raster pixels and PDF user space."""
    index: int
    media_box: Rect
    crop_box: Rect
    rotation: int = 0                    # normalized to 0/90/180/270
    @property
    def width(self) -> float: ...        # crop_box width
    @property
    def height(self) -> float: ...
    @property
    def display_size(self) -> tuple[float, float]: ...   # swapped when rotation is 90/270
    def render_matrix(self, scale: float) -> Matrix: ...     # user space -> pixel space
    def pixel_to_user(self, px: float, py: float, scale: float) -> Point: ...
    def user_to_pixel(self, x: float, y: float, scale: float) -> tuple[float, float]: ...
    def pixel_rect_to_user(self, rect: Rect, scale: float) -> Rect: ...
    def clamp(self, r: Rect) -> Rect: ...                 # clip to crop_box
```
`render_matrix` maps user space to a top-left-origin pixel raster of size
`(display_w*scale, display_h*scale)`, applying `rotation`. Round trip
`pixel_rect_to_user(user_to_pixel(...))` must be exact to `1e-6` for all four rotations.

### `zfp/core/units.py`
`PT_PER_INCH=72.0`, `pt_to_px(pt, dpi)`, `px_to_pt(px, dpi)`, `mm_to_pt`, `pt_to_mm`,
`dpi_to_scale(dpi)` (= dpi/72).

### `zfp/core/optional.py`
```python
@dataclass(frozen=True)
class OptionalModule:
    name: str; module: Optional[Any]; version: Optional[str]; error: Optional[str]
    def __bool__(self) -> bool: ...
    def require(self, feature: str) -> Any: ...   # raises UnsupportedFeatureError

def optional_import(name: str, *, attr: str | None = None) -> OptionalModule: ...   # cached
def have(name: str) -> bool: ...
def capability_report() -> dict[str, dict[str, Any]]: ...    # for CLI `zfp doctor`
```

### `zfp/core/logging.py`
`get_logger(name)`, `configure(level="INFO", json=False, stream=None)`, `LogContext`
contextmanager adding `document_id`/`page`/`agent` fields.

### `zfp/core/config.py`
```python
@dataclass
class DetectionConfig:
    min_line_length_pt: float = 24.0
    max_line_thickness_pt: float = 3.0
    line_merge_tolerance_pt: float = 1.5
    field_height_pt: float = 12.0          # synthesized height above an underline
    underline_gap_pt: float = 2.0          # baseline offset above the rule
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
    geometric_evidence: float = 0.30
    blank_region_evidence: float = 0.20
    nearby_label_evidence: float = 0.15
    layout_consistency: float = 0.10
    repeated_pattern_evidence: float = 0.10
    semantic_type_confidence: float = 0.10
    model_consensus: float = 0.05
    def normalized(self) -> ScoringWeights: ...
    def score(self, evidence: Mapping[str, float]) -> float: ...

@dataclass
class OcrConfig:
    enabled: bool = True
    dpi: int = 300
    engines: list[str] = ["tesseract", "paddle"]
    min_word_confidence: float = 0.55
    escalate_below: float = 0.70
    languages: list[str] = ["eng"]

@dataclass
class PrivacyConfig:
    allow_external_inference: bool = False
    require_zero_data_retention: bool = True
    provider_allowlist: list[str] = []
    max_context_chars: int = 2000
    redact_values_in_prompts: bool = True

@dataclass
class CouncilConfig:
    enabled: bool = True
    quorum: int = 3
    agreement_threshold: float = 0.66
    escalate_below_confidence: float = 0.80
    max_rounds: int = 2
    providers: list[str] = ["rules", "heuristic", "ontology"]

@dataclass
class AutofillConfig:
    mode: str = "conservative"        # "conservative" | "completion" | "off"
    min_fill_confidence: float = 0.90
    min_completion_confidence: float = 0.55
    propagate_repeats: bool = True
    require_validation: bool = True

@dataclass
class OrchestratorConfig:
    max_workers: int = 8
    page_shard_size: int = 4
    stage_timeout_s: float = 300.0
    fail_fast: bool = False
    deterministic: bool = True

@dataclass
class ZfpConfig:
    detection: DetectionConfig
    scoring: ScoringWeights
    ocr: OcrConfig
    privacy: PrivacyConfig
    council: CouncilConfig
    autofill: AutofillConfig
    orchestrator: OrchestratorConfig
    seed: int = 0
    @staticmethod
    def default() -> ZfpConfig: ...
    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> ZfpConfig: ...
    @staticmethod
    def from_file(path: str | os.PathLike) -> ZfpConfig: ...    # JSON
    def to_dict(self) -> dict[str, Any]: ...
```
Mutable defaults are produced with `field(default_factory=...)`.

### `zfp/core/types.py`
```python
class FieldType(str, Enum):
    TEXT="text"; MULTILINE_TEXT="multiline_text"; CHECKBOX="checkbox"; RADIO="radio"
    CHOICE="choice"; LISTBOX="listbox"; SIGNATURE="signature"; DATE="date"
    NUMBER="number"; CURRENCY="currency"; EMAIL="email"; PHONE="phone"
    COMB="comb"; BUTTON="button"; UNKNOWN="unknown"
    @property
    def pdf_kind(self) -> str: ...     # "Tx" | "Btn" | "Ch" | "Sig"

class PageMode(str, Enum):
    NATIVE_DOCUMENT="native_document"; FLAT_NATIVE_FORM="flat_native_form"
    SCANNED_FORM="scanned_form"; SCANNED_DOCUMENT="scanned_document"
    HYBRID="hybrid"; INTERACTIVE_FORM="interactive_form"; EMPTY="empty"

class DocumentClass(str, Enum):
    EXISTING_ACROFORM="existing_acroform"; FLAT_NATIVE_FORM="flat_native_form"
    SCANNED_FORM="scanned_form"; HYBRID="hybrid"; XFA="xfa"
    ENCRYPTED="encrypted"; SIGNED="signed"; NON_FORM="non_form"

class EvidenceKind(str, Enum):
    VECTOR_LINE="vector_line"; VECTOR_RECT="vector_rect"; VECTOR_CIRCLE="vector_circle"
    NATIVE_TEXT="native_text"; OCR_TEXT="ocr_text"; BLANK_REGION="blank_region"
    LABEL_LINK="label_link"; PATTERN="pattern"; LAYOUT="layout"; REPEAT="repeat"
    MODEL="model"; EXISTING_WIDGET="existing_widget"; TABLE_CELL="table_cell"
    COMB_CELL="comb_cell"; CHECKBOX_GLYPH="checkbox_glyph"

@dataclass(frozen=True)
class Evidence:
    kind: EvidenceKind
    score: float                       # 0..1
    detail: str = ""
    source_agent: str = ""
    rect: Optional[Rect] = None

@dataclass
class Confidence:
    geometry: float = 0.0
    label_link: float = 0.0
    semantic_type: float = 0.0
    autofill_value: float = 0.0
    def overall(self) -> float: ...    # geometric-ish mean, documented in code
    def as_dict(self) -> dict[str, float]: ...

@dataclass
class FieldConstraints:
    max_chars_estimate: Optional[int] = None
    required: bool = False
    multiline: bool = False
    comb_cells: Optional[int] = None
    choices: list[str] = []
    pattern: Optional[str] = None       # regex
    format_hint: Optional[str] = None   # "MM/DD/YYYY", "NN-NNNNNNN", ...
    read_only: bool = False
    def as_dict(self) -> dict[str, Any]: ...

@dataclass
class TextSpan:
    text: str
    rect: Rect
    page: int
    font_name: str = ""
    font_size: float = 0.0
    source: str = "native"              # "native" | "ocr"
    confidence: float = 1.0
    glyph_rects: list[Rect] = []
    baseline: Optional[float] = None
    def is_blank(self) -> bool: ...
    def normalized_text(self) -> str: ...   # lowercase, collapse ws, strip punctuation

@dataclass
class VectorPrimitive:
    kind: str                           # "line" | "rect" | "circle" | "path"
    rect: Rect
    page: int
    stroke_width: float = 0.0
    filled: bool = False
    stroked: bool = True
    points: list[Point] = []
    def orientation(self) -> str: ...   # "horizontal" | "vertical" | "other"

@dataclass
class RasterWord:
    text: str
    rect: Rect                          # already converted to USER SPACE
    confidence: float
    page: int
    line_id: int = -1
    block_id: int = -1
    alternatives: list[tuple[str, float]] = []

@dataclass
class FieldCandidate:
    id: str
    page: int
    rect: Rect
    field_type: FieldType = FieldType.UNKNOWN
    sources: list[str] = []             # e.g. ["vector_line","native_text"]
    visible_label: Optional[str] = None
    canonical_key: Optional[str] = None
    parent_context: list[str] = []
    confidence: Confidence = Confidence()
    constraints: FieldConstraints = FieldConstraints()
    evidence: list[Evidence] = []
    group_id: Optional[str] = None      # radio group / comb group
    export_value: Optional[str] = None  # radio/checkbox "on" state
    order: int = 0                      # reading order index
    def add_evidence(self, e: Evidence) -> None: ...
    def evidence_scores(self) -> dict[str, float]: ...   # keyed by ScoringWeights fields
    def as_dict(self) -> dict[str, Any]: ...

@dataclass
class PageProfile:
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
    def as_dict(self) -> dict[str, Any]: ...

@dataclass
class DocumentProfile:
    document_id: str
    page_count: int
    pages: list[PageProfile]
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
    warnings: list[str] = []
    @property
    def native_text_pages(self) -> list[int]: ...
    @property
    def raster_pages(self) -> list[int]: ...
    def as_dict(self) -> dict[str, Any]: ...

@dataclass
class FieldSpec:
    """Writer input. One entry == one AcroForm field (may own several widgets)."""
    name: str                           # fully qualified PDF field name
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
    choices: list[str] = []
    export_value: Optional[str] = None
    font_name: str = "Helv"
    font_size: float = 0.0              # 0 == auto-fit
    text_color: tuple[float,float,float] = (0.0,0.0,0.0)
    border_color: Optional[tuple[float,float,float]] = None
    background_color: Optional[tuple[float,float,float]] = None
    border_width: float = 0.0
    alignment: int = 0                  # 0 left 1 center 2 right
    group: Optional[str] = None         # radio group name
    tab_order: int = 0
    extra_widgets: list[tuple[int, Rect]] = []   # (page, rect) for multi-widget fields
    def as_dict(self) -> dict[str, Any]: ...

@dataclass
class FormSchema:
    document_id: str
    fields: list[FieldSpec] = []
    source_candidates: list[FieldCandidate] = []
    def by_name(self, name: str) -> Optional[FieldSpec]: ...
    def by_page(self, page: int) -> list[FieldSpec]: ...
    def as_dict(self) -> dict[str, Any]: ...
    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> FormSchema: ...

@dataclass
class FilledValue:
    field_name: str
    canonical_key: Optional[str]
    value: Optional[str]
    confidence: float
    provenance: dict[str, Any] = {}
    status: str = "filled"      # "filled"|"unavailable"|"low_confidence"|"invalid"|"policy_blocked"
    reason_codes: list[str] = []

@dataclass
class FillReport:
    document_id: str
    values: list[FilledValue] = []
    filled_count: int = 0
    unresolved_count: int = 0
    def as_dict(self) -> dict[str, Any]: ...
```
Every `list`/`dict` default above is a `field(default_factory=...)`.

### `zfp/core/ids.py`
`stable_id(*parts: Any, prefix: str = "", length: int = 12) -> str` — blake2b hex of the
joined repr; deterministic across runs and processes. `candidate_id(page, rect, kind)`.

### `zfp/core/serde.py`
`to_jsonable(obj)` (dataclass/Enum/Rect aware), `dumps(obj, indent=2)`,
`loads(s)`; `register_decoder(cls, fn)`.

---

## 2. `zfp.pdfio` — dependency-free PDF object layer

### `zfp/pdfio/objects.py`
```python
class PdfObject: ...                      # marker base
class PdfNull(PdfObject): NULL = PdfNull()
@dataclass(frozen=True) class PdfName(PdfObject):
    value: str                            # WITHOUT leading '/'
    def __str__(self) -> str: ...         # '/Name'
@dataclass(frozen=True) class PdfRef(PdfObject):
    num: int; gen: int = 0
class PdfString(PdfObject):
    def __init__(self, raw: bytes, hexform: bool = False): ...
    raw: bytes; hexform: bool
    def text(self) -> str: ...            # PDFDocEncoding / UTF-16BE BOM aware
    @staticmethod
    def from_text(s: str) -> PdfString: ...
class PdfArray(PdfObject, list): ...
class PdfDict(PdfObject, dict):           # keys are plain str WITHOUT '/'
    def get_name(self, key, default=None) -> Optional[str]: ...
class PdfStream(PdfObject):
    def __init__(self, d: PdfDict, raw: bytes): ...
    dict: PdfDict; raw: bytes
    def decoded(self, resolver: Resolver | None = None) -> bytes: ...   # cached
```
Numbers/booleans/None are plain `int`/`float`/`bool`/`PdfNull.NULL`.

### `zfp/pdfio/lexer.py`
`class Token(NamedTuple): kind: str; value: Any; pos: int` with kinds
`num,name,string,hexstring,dict_open,dict_close,array_open,array_close,keyword,eof`.
`class Lexer: def __init__(self, data: bytes, pos: int = 0); def next_token(self) -> Token; def peek(self) -> Token`.

### `zfp/pdfio/parser.py`
```python
class Resolver(Protocol):
    def resolve(self, obj: Any) -> Any: ...

class PdfFile:
    data: bytes
    trailer: PdfDict
    version: str
    @staticmethod
    def load(data: bytes) -> PdfFile: ...              # raises PdfParseError
    @staticmethod
    def open(path) -> PdfFile: ...
    def resolve(self, obj: Any) -> Any: ...            # follows PdfRef chains
    def get_object(self, num: int, gen: int = 0) -> Any: ...
    def object_numbers(self) -> list[int]: ...
    @property
    def catalog(self) -> PdfDict: ...
    @property
    def is_encrypted(self) -> bool: ...
    def rebuild_xref(self) -> None: ...                # brute-force scan fallback
```
Must handle: classic xref tables, cross-reference **streams** (`/Type/XRef`, `/W`,
`/Index`, predictors), object streams (`/Type/ObjStm`), `/Prev` chains, hybrid
`/XRefStm`, and a brute-force `rebuild_xref()` when the xref is broken. Linearized files
parse normally. Encryption: detect `/Encrypt`, expose `is_encrypted`; decryption itself is
`zfp/pdfio/crypt.py` (RC4 + AES-V2/V4/V5 with empty-user-password support; if the
optional crypto backend is missing, RC4 and AES-128/256 are implemented in
`zfp/vault/cipher.py` primitives — reuse them, do not vendor a second AES).

### `zfp/pdfio/filters.py`
`decode(data: bytes, filters: list[str], parms: list[PdfDict|None]) -> bytes` supporting
`FlateDecode` (+ PNG/TIFF predictors), `LZWDecode`, `ASCIIHexDecode`, `ASCII85Decode`,
`RunLengthDecode`; image filters (`DCTDecode`, `JPXDecode`, `CCITTFaxDecode`,
`JBIG2Decode`) are returned untouched and flagged via `is_image_filter(name)`.
`encode_flate(data) -> bytes`.

### `zfp/pdfio/writer.py`
```python
class PdfWriter:
    def __init__(self, pdf: PdfFile): ...
    def allocate(self) -> int: ...                     # next free object number
    def set_object(self, num: int, obj: Any) -> None: ...
    def add_object(self, obj: Any) -> PdfRef: ...
    def update_trailer(self, key: str, value: Any) -> None: ...
    def write_incremental(self) -> bytes: ...          # original bytes + appended revision
    def write_full(self) -> bytes: ...                 # full rewrite, classic xref
```
Incremental update MUST keep the original byte prefix intact (this is how ZFP proves it
did not disturb the visual substrate) and MUST carry `/Prev` plus a matching `/ID`.

### `zfp/pdfio/document.py`
```python
class Page:
    index: int
    dict: PdfDict
    geometry: PageGeometry
    def inherited(self, key: str) -> Any: ...
    def content_bytes(self) -> bytes: ...              # concatenated, decoded
    def resources(self) -> PdfDict: ...
    def annotations(self) -> list[PdfDict]: ...
    def add_annotation(self, ref: PdfRef) -> None: ...

class Document:
    @staticmethod
    def open(source: str | os.PathLike | bytes, password: str | None = None) -> Document: ...
    file: PdfFile
    writer: PdfWriter
    document_id: str                                   # stable_id of the bytes
    @property
    def page_count(self) -> int: ...
    @property
    def pages(self) -> list[Page]: ...
    def page(self, i: int) -> Page: ...
    def acroform(self) -> Optional[PdfDict]: ...
    def ensure_acroform(self) -> PdfDict: ...
    def has_xfa(self) -> bool: ...
    def is_signed(self) -> bool: ...
    def existing_fields(self) -> list[FieldSpec]: ...  # read back real widgets
    def save(self, path, incremental: bool = True) -> None: ...
    def to_bytes(self, incremental: bool = True) -> bytes: ...
```

### `zfp/pdfio/fonts.py`
```python
STANDARD_14: dict[str, str]                            # alias -> base font name
def widths_for(base_font: str) -> dict[int, int]: ...  # codepoint -> width/1000
def text_width(text: str, base_font: str, size: float) -> float: ...
def font_ascent(base_font: str) -> float
def font_descent(base_font: str) -> float
def fit_font_size(text: str, base_font: str, rect: Rect, *, max_size=12.0,
                  min_size=4.0, padding=2.0) -> float: ...
def wrap_text(text: str, base_font: str, size: float, width: float) -> list[str]: ...
def escape_pdf_text(s: str) -> bytes: ...
def ensure_standard_font(doc: Document, base_font: str = "Helvetica") -> tuple[str, PdfRef]
```
Widths for Helvetica/Times/Courier/Symbol/ZapfDingbats families live in
`zfp/pdfio/_afm_data.py` as compact literal tables (WinAnsi codepoints 32..255).

---

## 3. Perception

### `zfp/preflight/classifier.py`
```python
def profile_document(doc: Document, config: ZfpConfig | None = None) -> DocumentProfile: ...
def classify_page(doc: Document, index: int, config) -> PageProfile: ...
def route(profile: DocumentProfile) -> DocumentClass: ...
```

### `zfp/native/content.py`
```python
@dataclass
class ContentState: ctm: Matrix; text_matrix: Matrix; line_matrix: Matrix; font: str
    font_size: float; char_spacing: float; word_spacing: float; horizontal_scale: float
    leading: float; rise: float; render_mode: int; stroke_width: float

class ContentStreamInterpreter:
    def __init__(self, page: Page, resolver: Resolver, config: ZfpConfig | None = None): ...
    def run(self) -> ContentResult: ...

@dataclass
class ContentResult:
    spans: list[TextSpan]
    primitives: list[VectorPrimitive]
    images: list[Rect]
    op_count: int
```
Handles `q Q cm BT ET Tf Td TD Tm T* TJ Tj ' " Tc Tw Tz TL Ts Tr` and path ops
`m l c v y re h S s f F f* B B* b b* n W W* g rg k cs sc scn G RG K gs w Do BI/ID/EI`.
Unknown operators are skipped safely. Form XObjects are recursed (depth-limited to 8).

### `zfp/native/text.py`
`group_spans_into_lines(spans) -> list[list[TextSpan]]`,
`reading_order(spans, page_geometry) -> list[TextSpan]`,
`merge_adjacent_spans(spans, gap_ratio=0.3) -> list[TextSpan]`,
`detect_columns(spans, page_geometry) -> list[Rect]`.

### `zfp/vision/primitives.py`  (pure python, no numpy)
```python
def normalize_primitives(prims: list[VectorPrimitive], config) -> list[VectorPrimitive]
def horizontal_rules(prims, config) -> list[VectorPrimitive]
def vertical_rules(prims, config) -> list[VectorPrimitive]
def merge_collinear(rules, config) -> list[VectorPrimitive]
def detect_boxes(prims, config) -> list[Rect]           # from 4 rules or filled rects
def detect_checkbox_glyphs(prims, spans, config) -> list[Rect]
def detect_circles(prims, config) -> list[Rect]
def detect_table_cells(h_rules, v_rules, config) -> list[Rect]
def blank_regions(spans, prims, geometry, config) -> list[Rect]
```

### `zfp/vision/raster_shapes.py` (optional numpy/opencv adapter)
`detect_shapes_from_image(image, page_geometry, scale, config) -> RasterShapes`
with graceful `UnsupportedFeatureError` when neither backend exists.
`RasterShapes` = dataclass(h_rules, v_rules, boxes, circles, blanks) — all in USER space.

### `zfp/raster/render.py`
```python
@dataclass(frozen=True)
class RenderedPage: page: int; width: int; height: int; scale: float
    gray: bytes                 # row-major 8bpp, len == width*height
    backend: str
def render_page(doc: Document, index: int, dpi: int = 300) -> RenderedPage: ...
def available_backends() -> list[str]: ...     # "pymupdf","pdftoppm","pypdfium2","none"
def embedded_page_images(doc, index) -> list[tuple[Rect, bytes, str]]   # always available
```
When no renderer exists, `render_page` raises `UnsupportedFeatureError` and callers MUST
fall back to `embedded_page_images` (the common scan case: one full-page DCT/CCITT image).

### `zfp/raster/preprocess.py`
`estimate_skew(page) -> float`, `deskew(page, angle) -> RenderedPage`,
`denoise(page) -> RenderedPage`, `binarize(page, method="sauvola") -> RenderedPage`,
`normalize_contrast(page) -> RenderedPage`, `detect_orientation(page) -> int`,
`preprocess(page, config) -> tuple[RenderedPage, PreprocessReport]`. Pure-python
implementations required; numpy path used when available.

### `zfp/ocr/engine.py`
```python
class OcrEngine(Protocol):
    name: str
    def available(self) -> bool: ...
    def recognize(self, page: RenderedPage, geometry: PageGeometry, config: OcrConfig) -> list[RasterWord]: ...

class TesseractEngine(OcrEngine): ...          # pytesseract or `tesseract` binary TSV
class PaddleEngine(OcrEngine): ...
class NullEngine(OcrEngine): ...               # always available, returns []

def ocr_cascade(page, geometry, config) -> OcrResult
@dataclass
class OcrResult: words: list[RasterWord]; engine: str; mean_confidence: float
    escalated: bool; suspects: list[RasterWord]
def words_to_spans(words: list[RasterWord]) -> list[TextSpan]
```
Rule from the spec: **never OCR a page that already has native text.** `ocr_cascade`
must assert that and return empty with `engine="skipped_native_text"`.

### `zfp/candidates/archetypes.py`
```python
@dataclass
class CandidateContext:
    page: int; geometry: PageGeometry; spans: list[TextSpan]
    primitives: list[VectorPrimitive]; words: list[RasterWord]; config: ZfpConfig

class ArchetypeDetector(Protocol):
    name: str
    def detect(self, ctx: CandidateContext) -> list[FieldCandidate]: ...

# concrete detectors (each a class in this module):
UnderlineFieldDetector      # LABEL + horizontal rule
BoxFieldDetector            # LABEL + rectangle
CheckboxDetector            # small square/circle + label
RadioGroupDetector          # >=2 circles/squares on a row/column sharing a stem label
CombFieldDetector           # repeated equal cells  [ ][ ][ ]
DateBoxDetector             # __/__/____ patterns
SignatureLineDetector       # rule + "signature"/"initials"/"date"
TableCellDetector           # grid intersection cells that are empty
BlankRegionDetector         # borderless whitespace beneath/right-of a label
FreeTextAreaDetector        # large empty block
ColonRunDetector            # "Name: ................" leader dots

DEFAULT_DETECTORS: list[ArchetypeDetector]
def generate_candidates(ctx: CandidateContext, detectors=None) -> list[FieldCandidate]
```

### `zfp/fusion/geometry_fusion.py`
```python
def fuse(candidates: list[FieldCandidate], config) -> list[FieldCandidate]
def snap_to_primitive(rect: Rect, prims: list[VectorPrimitive], tol: float) -> Rect
def deduplicate(cands, iou_threshold) -> list[FieldCandidate]
def suppress_overlaps(cands, config) -> list[FieldCandidate]
def calibrate_rect(rect: Rect, field_type: FieldType, config) -> Rect
def score_candidates(cands, weights: ScoringWeights) -> list[FieldCandidate]
```

---

## 4. Semantics

### `zfp/ontology/keys.py`
`CANONICAL_KEYS: dict[str, KeySpec]` covering at minimum every key named in the research
document plus a full person/company/bank/document/vehicle/insurance/medical/employment
namespace (target: >= 220 keys).
```python
@dataclass(frozen=True)
class KeySpec:
    key: str
    field_type: FieldType
    label: str
    aliases: tuple[str, ...] = ()
    pattern: Optional[str] = None
    format_hint: Optional[str] = None
    sensitivity: str = "normal"        # "normal"|"pii"|"secret"
    max_length: Optional[int] = None
    normalizer: Optional[str] = None   # name in zfp.resolver.normalizers.REGISTRY
    validator: Optional[str] = None    # name in zfp.resolver.validators.REGISTRY
    parents: tuple[str, ...] = ()      # e.g. ("billing","shipping") context discriminators
def get(key) -> Optional[KeySpec]
def all_keys() -> list[KeySpec]
def children(prefix: str) -> list[KeySpec]
```

### `zfp/ontology/aliases.py`
`ALIAS_INDEX: dict[str, str]` normalized-alias -> canonical key (>= 700 entries).
`normalize_label(s) -> str`, `lookup(label) -> Optional[str]`,
`fuzzy_lookup(label, cutoff=0.82) -> list[tuple[str, float]]` (stdlib `difflib`).

### `zfp/ontology/patterns.py`
`PATTERNS: list[PatternRule]` with `PatternRule(name, regex, field_type, format_hint,
canonical_hint, confidence)` covering the spec's placeholder table (SSN, phone, ZIP+4,
EIN, currency, MM/DD/YYYY, credit card, IBAN, VIN, email, URL, time, percentage, ...).
`match_placeholder(text) -> Optional[PatternRule]`, `infer_from_context(label, nearby) -> Optional[PatternRule]`.

### `zfp/semantics/graph.py`
```python
class RelationKind(str, Enum): LEFT_OF; RIGHT_OF; ABOVE; BELOW; SAME_ROW; SAME_COLUMN
    CONTAINS; MEMBER_OF; PRECEDES
@dataclass class Node: id: str; kind: str; rect: Rect; page: int; text: str = ""
@dataclass class Edge: src: str; dst: str; kind: RelationKind; weight: float
class SpatialGraph:
    def add_node(n) / add_edge(e) / neighbors(id, kind=None) / node(id)
    def to_dict(self) -> dict
def build_graph(spans, candidates, sections, geometry) -> SpatialGraph
```

### `zfp/semantics/linker.py`
`link_labels(candidates, spans, graph, config) -> list[FieldCandidate]` — assigns
`visible_label`, `parent_context`, `confidence.label_link`. Scoring prefers left-of on the
same baseline, then above, then enclosing section header; distance-decayed.

### `zfp/semantics/sections.py`
`detect_sections(spans, geometry) -> list[Section]` where
`Section(title, rect, level, page)`; heuristics: font size, boldness in name, all-caps,
numbering, leading whitespace.

### `zfp/semantics/typing.py`
`infer_field_type(candidate, spans, config) -> tuple[FieldType, float, list[str]]`
(type, confidence, reason codes). Uses geometry + label + pattern + choices nearby.

### `zfp/semantics/normalizer.py`
`canonicalize(candidate, config) -> tuple[Optional[str], float, list[str]]` — the
deterministic mapping cascade: exact -> alias -> pattern -> section-scoped alias ->
fuzzy -> None. Never calls a model.

### `zfp/semantics/repeats.py`
`find_repeated_fields(candidates) -> list[list[FieldCandidate]]`,
`propagate(groups, values) -> list[FilledValue]`, `check_consistency(...) -> list[str]`.

---

## 5. Council / AI escalation

### `zfp/council/base.py`
```python
@dataclass(frozen=True)
class Question:
    id: str
    kind: str                       # "canonical_key" | "field_type" | "choice_set" | "ambiguity"
    prompt: str
    schema: dict[str, Any]          # JSON Schema the answer must satisfy
    context: dict[str, Any]         # already redacted
    options: list[str] = ()

@dataclass
class Vote:
    member: str
    answer: dict[str, Any]
    confidence: float
    reason_codes: list[str] = []
    latency_ms: float = 0.0
    error: Optional[str] = None

@dataclass
class Verdict:
    question_id: str
    answer: Optional[dict[str, Any]]
    confidence: float
    consensus: float                # fraction agreeing with the winning answer
    votes: list[Vote]
    dissent: list[Vote]
    blind_spots: list[str] = []
    contradictions: list[str] = []
    escalated: bool = False
    def as_dict(self) -> dict[str, Any]: ...

class CouncilMember(Protocol):
    name: str
    def available(self) -> bool: ...
    def vote(self, q: Question) -> Vote: ...
```

### `zfp/council/members.py`
`RulesMember` (ontology + patterns, always available),
`HeuristicMember` (layout/section reasoning, always available),
`OntologyFuzzyMember` (difflib over aliases, always available),
`OpenRouterMember(model, api_key, ...)` (optional; strict JSON-schema structured output,
provider allow-list, ZDR header, refuses to run when `PrivacyConfig` forbids egress),
`LocalModelMember` (hook for a local classifier; disabled by default).

### `zfp/council/council.py`
```python
class Council:
    def __init__(self, members: list[CouncilMember], config: CouncilConfig,
                 privacy: PrivacyConfig, logger=None): ...
    def deliberate(self, q: Question) -> Verdict: ...
    def deliberate_many(self, qs: list[Question], max_workers: int = 4) -> list[Verdict]: ...
    def analyst(self, votes: list[Vote], q: Question) -> Verdict: ...   # consensus/contradiction/blind-spot pass
def build_default_council(config: ZfpConfig) -> Council: ...
```
Deterministic tie-breaks: sort by `(-confidence, member_name)`.

### `zfp/council/openrouter.py`
Thin client over `urllib.request` (no `requests` dependency): `chat_json(model, messages,
schema, *, api_key, base_url, timeout, zdr=True, allow_providers=None) -> dict`.
Never called when `PrivacyConfig.allow_external_inference` is False; raises `PolicyError`.

### `zfp/council/redaction.py`
`redact_context(ctx, privacy) -> dict` — strips values, truncates to
`max_context_chars`, replaces digits with `#` when `redact_values_in_prompts`.

---

## 6. Vault, resolution, validation

### `zfp/vault/cipher.py` (pure python, no deps)
`chacha20_poly1305_encrypt(key, nonce, plaintext, aad) -> bytes`,
`chacha20_poly1305_decrypt(key, nonce, ciphertext, aad) -> bytes`,
`derive_key(password, salt, *, n=2**14, r=8, p=1, dklen=32) -> bytes` (hashlib.scrypt),
`random_bytes(n)`, plus `aes_cbc_decrypt`, `aes_ecb_encrypt`, `rc4` used by
`zfp/pdfio/crypt.py`. Must pass RFC 8439 / FIPS-197 published test vectors in unit tests.

### `zfp/vault/store.py`
```python
@dataclass
class VaultEntry:
    key: str; value: str
    source: str = "manual"          # verified_profile|prior_form|crm|import|manual|inferred
    verified_at: Optional[str] = None
    confidence: float = 1.0
    sensitivity: str = "normal"
    labels_seen: list[str] = []
    def as_dict(self) -> dict[str, Any]: ...

class ProfileVault:
    def __init__(self, entries=None, *, profile_id="default"): ...
    def put(self, key, value, **provenance) -> VaultEntry
    def get(self, key) -> Optional[VaultEntry]
    def resolve(self, key, parent_context=()) -> Optional[VaultEntry]   # honors billing/shipping parents
    def observe_label(self, key, label) -> None      # learning loop
    def keys(self) -> list[str]
    def as_dict(self) -> dict
    @staticmethod
    def from_dict(d) -> ProfileVault
    def save(self, path, password: str | None = None) -> None    # encrypted when password given
    @staticmethod
    def load(path, password: str | None = None) -> ProfileVault
```
Encrypted file format: magic `ZFPV1\n` + JSON header (kdf params, nonce) + ciphertext.

### `zfp/resolver/normalizers.py`
`REGISTRY: dict[str, Callable[[str, KeySpec], str]]` with at least
`upper, lower, title, digits_only, phone_us, phone_e164, ssn, ein, zip5, zip9, date_mdy,
date_ymd, date_dmy, currency, state_abbrev, country_iso2, email, strip_ws, name_case,
credit_card, iban, boolean_yes_no`.

### `zfp/resolver/validators.py`
`REGISTRY: dict[str, Callable[[str, KeySpec], ValidationOutcome]]`;
`ValidationOutcome(ok: bool, message: str = "", normalized: Optional[str] = None)`;
includes `luhn`, `ssn`, `ein`, `email`, `phone_us`, `zip_us`, `date`, `iban`, `routing_aba`,
`vin`, `nonempty`, `max_length`, `regex`, `choice`.

### `zfp/resolver/autofill.py`
```python
class AutofillResolver:
    def __init__(self, vault: ProfileVault, config: ZfpConfig, council: Council | None = None): ...
    def resolve_field(self, spec: FieldSpec, candidate: FieldCandidate | None = None) -> FilledValue
    def resolve_schema(self, schema: FormSchema, candidates=None) -> FillReport
```
Order: exact canonical key -> parent-context-scoped key -> alias -> repeats propagation ->
council escalation (only if `council` present and confidence below threshold) -> unavailable.
Mode `conservative` never emits a value below `min_fill_confidence`; it marks
`status="unavailable"` instead. **Never invents values.** Signature fields are never
auto-filled with a signature; they are marked `policy_blocked` unless a signing policy
explicitly authorizes it (`zfp/resolver/policy.py:SigningPolicy`).

### `zfp/resolver/policy.py`
`SigningPolicy(allow_autosign=False, authorized_signers=(), require_explicit_consent=True)`,
`check_signature_fill(spec, policy) -> tuple[bool, str]`.

---

## 7. Output

### `zfp/acroform/writer.py`
```python
class AcroFormWriter:
    def __init__(self, doc: Document, config: ZfpConfig | None = None): ...
    def write(self, schema: FormSchema) -> WriteReport
    def write_field(self, spec: FieldSpec) -> PdfRef
    def set_values(self, values: Mapping[str, str]) -> None
    def flatten(self, field_names: Iterable[str] | None = None) -> None

@dataclass
class WriteReport:
    fields_written: int; widgets_written: int; pages_touched: list[int]
    warnings: list[str]; field_refs: dict[str, str]
```
Requirements: creates `/AcroForm` with `/Fields`, `/DA`, `/DR` (Helv + ZaDb), `/NeedAppearances`
false plus **generated appearance streams**, correct `/FT`, `/Ff` flags (Multiline 1<<12,
Required 1<<1, ReadOnly 1<<0, Comb 1<<24, Radio 1<<15, NoToggleToOff 1<<14, Pushbutton 1<<16,
Combo 1<<17, Edit 1<<18, Sort 1<<19, MultiSelect 1<<21), `/MaxLen`, `/Opt`, `/TU`, `/T`, `/V`,
`/DV`, `/Rect`, `/P`, `/F` = 4 (Print), radio kids sharing a parent with `/AS` states, and
appends widgets to each page's `/Annots`. Writes **incrementally** by default.

### `zfp/acroform/reader.py`
`read_fields(doc) -> list[FieldSpec]`, `read_values(doc) -> dict[str,str]`,
`field_tree(doc) -> dict`, `export_fdf(doc) -> bytes`, `export_json(doc) -> dict`,
`import_json(doc, data) -> None`, `export_xml`/`export_csv`.

### `zfp/appearance/streams.py`
`text_appearance(spec, value, resources) -> bytes`,
`checkbox_appearance(spec, on: bool) -> bytes`,
`radio_appearance(spec, on: bool) -> bytes`,
`choice_appearance(spec, value) -> bytes`,
`signature_placeholder_appearance(spec) -> bytes`,
`build_xobject(doc, spec, content: bytes, resources: PdfDict) -> PdfRef`.
Auto font size (`font_size == 0`) uses `fonts.fit_font_size`; multiline uses `wrap_text`;
comb fields distribute glyphs across `comb_cells` cells.

### `zfp/qa/verify.py`
```python
@dataclass
class QAFinding: severity: str; code: str; message: str; page: Optional[int] = None
    field: Optional[str] = None
@dataclass
class QAReport:
    findings: list[QAFinding]; passed: bool; metrics: dict[str, Any]
    def as_dict(self) -> dict: ...
def verify_document(original: bytes, produced: bytes, schema: FormSchema,
                    config: ZfpConfig) -> QAReport
def check_integrity(data: bytes) -> list[QAFinding]        # reparse, xref, page tree
def check_prefix_preserved(original: bytes, produced: bytes) -> list[QAFinding]
def check_fields_roundtrip(produced: bytes, schema: FormSchema) -> list[QAFinding]
def check_no_overlap(schema: FormSchema) -> list[QAFinding]
def check_in_page_bounds(doc, schema) -> list[QAFinding]
```

### `zfp/qa/renderdiff.py`
`page_hash(rendered) -> str`, `diff_pages(a, b, mask: list[Rect]) -> DiffResult`
(`changed_pixels`, `ratio`, `bbox`), `visual_preservation_report(...)`. Degrades to a
structural (content-stream) diff when no renderer exists.

### `zfp/qa/metrics.py`
Implements the spec's quality dashboard: `field_geometry_metrics(pred, truth)` (IoU,
center error, per-edge error), `recall_precision`, `type_macro_f1`,
`label_association_rate`, `canonical_accuracy`, `ocr_cer_wer`, `autofill_exact_match`,
`repeat_consistency`, and `MetricsDashboard.render_text()/as_dict()`.

---

## 8. Agent mesh  (`zfp.agents`) — the deployment layer

### `zfp/agents/base.py`
```python
class AgentRole(str, Enum):
    ORCHESTRATOR="orchestrator"; FACILITATOR="facilitator"; COUNCIL="council"
    SPECIALIST="specialist"; SUBAGENT="subagent"; VERIFIER="verifier"; SCRIBE="scribe"

@dataclass
class AgentTask:
    id: str; kind: str; payload: dict[str, Any]
    page: Optional[int] = None; parent_id: Optional[str] = None
    priority: int = 0; deadline_s: Optional[float] = None
    def child(self, kind: str, payload: dict, **kw) -> AgentTask: ...

@dataclass
class AgentResult:
    agent: str; task_id: str; ok: bool
    value: Any = None; error: Optional[str] = None
    confidence: float = 1.0; duration_ms: float = 0.0
    evidence: list[Evidence] = []
    telemetry: dict[str, Any] = {}
    subresults: list[AgentResult] = []
    @staticmethod
    def failure(agent, task_id, error) -> AgentResult: ...

class Blackboard:
    """Thread-safe shared state. Namespaced keys 'ns/key'."""
    def put(self, key, value) -> None
    def get(self, key, default=None) -> Any
    def append(self, key, value) -> None
    def extend(self, key, values) -> None
    def merge(self, key, mapping) -> None
    def keys(self) -> list[str]
    def snapshot(self) -> dict[str, Any]

@dataclass
class AgentContext:
    document: Any                       # zfp.pdfio.document.Document (Any to avoid import cycle)
    config: ZfpConfig
    blackboard: Blackboard
    council: Optional[Any] = None
    vault: Optional[Any] = None
    logger: Any = None
    trace: Optional[Trace] = None
    def child(self, **overrides) -> AgentContext: ...

class Agent(ABC):
    name: str
    role: AgentRole = AgentRole.SPECIALIST
    capabilities: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()      # blackboard keys that must exist
    produces: tuple[str, ...] = ()      # blackboard keys written
    def can_handle(self, task: AgentTask) -> bool: ...
    @abstractmethod
    def run(self, task: AgentTask, ctx: AgentContext) -> AgentResult: ...
    def __call__(self, task, ctx) -> AgentResult: ...   # wraps run() with timing + error capture
```

### `zfp/agents/trace.py`
`TraceEvent(t_index, agent, role, task_id, kind, status, duration_ms, detail)`,
`Trace.record(...)`, `Trace.as_dict()`, `Trace.render_tree() -> str` (ASCII deployment
tree the CLI prints), `Trace.timeline() -> list[TraceEvent]`. `t_index` is a monotonic
counter, **not** a clock, so traces are reproducible.

### `zfp/agents/scheduler.py`
```python
@dataclass
class Stage:
    name: str
    agents: list[Agent]
    mode: str = "parallel"              # "parallel" | "sequential"
    fanout: str = "document"            # "document" | "page" | "shard" | "candidate"
    optional: bool = False
    gate: Optional[Callable[[AgentContext], bool]] = None

class Scheduler:
    def __init__(self, config: OrchestratorConfig, trace: Trace | None = None): ...
    def run_stage(self, stage: Stage, tasks: list[AgentTask], ctx: AgentContext) -> list[AgentResult]
    def map(self, agent: Agent, tasks: list[AgentTask], ctx: AgentContext) -> list[AgentResult]
```
Parallelism uses `concurrent.futures.ThreadPoolExecutor(max_workers)`. Results are
returned sorted by `(task.page or -1, task.id, agent.name)` for determinism.

### `zfp/agents/facilitator.py`
```python
@dataclass
class Proposal:
    agent: str; value: Any; confidence: float
    evidence: list[Evidence] = []; reason_codes: list[str] = []

@dataclass
class Reconciliation:
    value: Any; confidence: float; contributors: list[str]
    conflicts: list[str] = []; escalate: bool = False

class Facilitator(ABC):
    name: str
    role = AgentRole.FACILITATOR
    @abstractmethod
    def reconcile(self, proposals: list[Proposal], ctx: AgentContext) -> Reconciliation: ...

class GeometryFacilitator(Facilitator): ...    # fuses vector/OCR/CV rectangles; snaps
class SemanticFacilitator(Facilitator): ...    # merges label/pattern/ontology proposals
class ValueFacilitator(Facilitator): ...       # vault vs repeats vs council
class ConflictLog: ...                          # records every disagreement for QA
```

### `zfp/agents/specialists.py`
Concrete `Agent` subclasses, each thin over the corresponding library module:
`PreflightAgent, NativeTextAgent, VectorGeometryAgent, RasterRenderAgent,
ScanPreprocessAgent, OcrAgent, ShapeDetectionAgent, BlankRegionAgent,
CandidateGeneratorAgent, GeometryFusionAgent, SectionAgent, LabelLinkAgent,
PatternRuleAgent, TypeInferenceAgent, CanonicalizeAgent, RepeatConsistencyAgent,
AmbiguityTriageAgent, VaultResolveAgent, NormalizeValidateAgent, SchemaBuilderAgent,
AcroFormWriteAgent, AppearanceAgent, IntegrityAgent, RenderDiffAgent, MetricsAgent,
ReportAgent, ExistingFormAgent, XfaCompatAgent, SecurityGateAgent`.
Each declares `requires`/`produces` blackboard keys from `zfp/agents/keys.py`.

### `zfp/agents/keys.py`
String constants for every blackboard key (`BB_PROFILE`, `BB_SPANS`, `BB_PRIMITIVES`,
`BB_WORDS`, `BB_CANDIDATES`, `BB_SECTIONS`, `BB_GRAPH`, `BB_SCHEMA`, `BB_FILL_REPORT`,
`BB_WRITE_REPORT`, `BB_QA_REPORT`, `BB_CONFLICTS`, `BB_VERDICTS`, ...). Page-scoped keys
use `page_key(base, page)`.

### `zfp/agents/subagents.py`
`PageShardSubAgent(parent: Agent, pages: list[int])` — the page-sharding mechanism that
removes the 25-page limit; `RegionSubAgent` for cropped re-examination of a low-confidence
region; `spawn_page_subagents(agent, profile, shard_size) -> list[AgentTask]`.

### `zfp/agents/orchestrator.py`
```python
@dataclass
class RunReport:
    document_id: str
    profile: DocumentProfile
    schema: FormSchema
    fill_report: Optional[FillReport]
    qa_report: Optional[QAReport]
    trace: Trace
    stages: list[str]
    warnings: list[str]
    output_path: Optional[str] = None
    def as_dict(self) -> dict[str, Any]: ...
    def summary(self) -> str: ...

class Orchestrator:
    def __init__(self, config: ZfpConfig, *, council=None, vault=None,
                 scheduler: Scheduler | None = None): ...
    def stages(self) -> list[Stage]: ...          # the default deployment plan
    def deploy(self, doc: Document, *, fill: bool = True, write: bool = True) -> RunReport
    def deploy_path(self, path, out_path=None, **kw) -> RunReport

def build_default_orchestrator(config: ZfpConfig | None = None, **kw) -> Orchestrator
```
Default stage plan (names are stable, the CLI prints them):
`security-gate, preflight, route, native-extract, raster-ocr, shape-detect,
candidate-generate, geometry-fuse, section-detect, label-link, type-infer,
canonicalize, ambiguity-council, repeat-reconcile, schema-build, vault-resolve,
validate, acroform-write, appearance, integrity-verify, render-diff, metrics, report`.

---

## 9. Pipeline / services / CLI / API

### `zfp/pipeline/run.py`
`process(path, *, out=None, config=None, vault=None, fill=True) -> RunReport`,
`detect_only(path, config=None) -> FormSchema`,
`fill_existing(path, data: Mapping[str,str], out) -> FillReport`,
`batch(paths, **kw) -> list[RunReport]`.

### `zfp/services/*.py`
`pages.py` (merge, split, rotate, reorder, insert, delete, replace, extract),
`watermark.py`, `compress.py`, `redact.py`, `extract.py` (structured layout JSON),
`convert.py` (text/HTML/markdown export; optional adapters for docx/xlsx),
`protect.py` (encrypt/decrypt with owner-password gate — never a bypass),
`compare.py` (structural + text diff), `accessibility.py` (tag/reading-order inference).
Each exposes plain functions taking/returning `Document` or bytes.

### `zfp/cli/main.py`
`zfp` command with subcommands:
`doctor` (capability report), `preflight FILE`, `detect FILE [--json]`,
`build FILE -o OUT` (detect + write AcroForm), `fill FILE -o OUT [--vault V]`,
`auto FILE -o OUT` (full deploy), `agents` (print the deployment tree),
`vault (init|set|get|list|import|export)`, `schema FILE`, `verify ORIG OUT`,
`synth OUT [--kind ...]`, `metrics PRED TRUTH`, `services ...`.
Uses `argparse`; exit code 0 ok, 1 findings, 2 error.

### `zfp/api/app.py`
Optional FastAPI app factory `create_app(config)`; import-guarded so the package imports
fine without FastAPI installed.

---

## 10. Synthetic corpus (`zfp.synth`)

### `zfp/synth/generator.py`
```python
@dataclass
class GroundTruthField:
    name: str; canonical_key: str; field_type: FieldType; page: int; rect: Rect
    label: str; expected_value: str = ""
@dataclass
class SyntheticForm:
    pdf_bytes: bytes; fields: list[GroundTruthField]; seed: int; kind: str
    def save(self, path) -> None
    def truth_dict(self) -> dict

@dataclass
class SynthOptions:
    kind: str = "underline"        # underline|boxed|checkbox|table|comb|mixed|borderless|multipage
    pages: int = 1
    seed: int = 0
    font: str = "Helvetica"
    font_size: float = 10.0
    line_width: float = 0.6
    include_sections: bool = True
    rotation: int = 0
    locale: str = "en_US"

def generate(options: SynthOptions) -> SyntheticForm
def generate_corpus(n: int, *, seed=0, kinds=None) -> list[SyntheticForm]
```
`generate` must emit a **real, parseable PDF** built through `zfp.pdfio.writer` with
base-14 fonts — no third-party writer. Ground-truth rectangles are exact by construction.

---

## 11. Test policy

- `tests/unit/…` per module, `unittest.TestCase`.
- `tests/integration/test_end_to_end.py` MUST prove the spec's critical vertical slice:
  synth form -> preflight -> detect -> AcroForm write -> reopen -> read fields back ->
  autofill from a vault -> verify.
- `tests/fixtures/factory.py` builds fixtures via `zfp.synth` (no binary blobs in git).
- Target: recall >= 0.90 and mean IoU >= 0.80 on the synthetic `underline`, `boxed`,
  `checkbox` and `comb` corpora; assert these in `tests/integration/test_quality_gates.py`.
