"""Document triage: decide *which perception system a page actually needs*.

This is the cheapest layer in ZFP and the most consequential one.  Most "AI PDF"
tools rasterize every page on sight and then ask a model to look at the picture,
which throws away the glyph positions, the vector rules and the widget rectangles
the file already contained -- information no OCR pass can ever recover exactly.
Triage exists so that never happens: a page is rendered only after this module has
established that there is nothing else left to read.

The classifier therefore has a hard constraint: **it must not render, and it must not
run the full content-stream interpreter** (``zfp.native.content`` owns that, and it is
an order of magnitude more expensive).  Everything here is derived from

* the page's ``/Annots`` array -- widgets are a fact, not an inference;
* a single linear token scan of the decoded content stream;
* the ``/XObject`` sub-dictionary of the page resources;
* the document catalog (``/AcroForm``, ``/XFA``, ``/MarkInfo``, ``/Perms``).

Two numbers are therefore approximations rather than measurements, and both are
documented as such where they are computed: :func:`_estimate_image_area_ratio`
(no CTM is tracked, so image coverage is inferred from placement matrices and pixel
dimensions) and the "form cue" rule count (raw user-space operands, not transformed
ones).  Everything else -- widget presence, character counts, operator counts, the
presence of raster imagery -- is exact.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.config import DetectionConfig, ZfpConfig
from ..core.geometry import PageGeometry
from ..core.logging import get_logger
from ..core.serde import to_jsonable
from ..core.types import DocumentClass, DocumentProfile, PageMode, PageProfile
from ..pdfio.document import US_LETTER, Document, Page
from ..pdfio.lexer import DELIMITERS, WHITESPACE, Lexer
from ..pdfio.objects import PdfDict, PdfRef, PdfStream
from .security import inspect as inspect_security

__all__ = [
    "ContentScan",
    "scan_content",
    "classify_page",
    "profile_document",
    "route",
    "describe",
    "profile_to_dict",
    "FORM_MARKER_PATTERNS",
    "MIN_NATIVE_TEXT_CHARS",
    "MIN_VECTOR_OPS",
    "RASTER_DOMINANT_RATIO",
    "HYBRID_MIN_RASTER_RATIO",
    "VECTOR_POOR_MAX_OPS",
    "MIN_FORM_RULES",
    "MIN_FORM_MARKERS",
    "MIN_FORM_CHECKBOXES",
]

_log = get_logger(__name__)

# -- thresholds -------------------------------------------------------------------------
#: A page needs this many characters before its text layer counts as usable.
MIN_NATIVE_TEXT_CHARS = 8
#: Path operators before a page is considered to carry a vector layer at all.
MIN_VECTOR_OPS = 4
#: Image coverage above which a page is "raster dominant" -- a scan, not an illustration.
RASTER_DOMINANT_RATIO = 0.6
#: Image coverage above which raster + native text is a hybrid rather than a native page.
HYBRID_MIN_RASTER_RATIO = 0.25
#: A scan's own content stream is tiny (a clip rectangle and a placement at most).
VECTOR_POOR_MAX_OPS = 12
#: Horizontal rules that make a native page look like a form.
MIN_FORM_RULES = 3
#: Distinct marker words that make a native page look like a form.
MIN_FORM_MARKERS = 2
#: Checkbox-sized squares that make a page look like a form on their own.
MIN_FORM_CHECKBOXES = 3
#: A placement matrix covering at least this fraction of the page is a full-page image.
FULL_PAGE_SCALE_RATIO = 0.85
#: Plausible scan resolutions used to turn image pixels into points.
MIN_IMAGE_DPI = 72.0
MAX_IMAGE_DPI = 600.0
#: Safety valve: content streams larger than this are scanned only up to the limit.
MAX_SCAN_BYTES = 8 << 20
#: Text kept for the marker-word test (the character *count* is never truncated).
MAX_SCAN_TEXT_CHARS = 40000
#: How deep to walk nested form XObjects when inventorying images.
_MAX_XOBJECT_DEPTH = 3

# -- operator vocabulary ----------------------------------------------------------------
#: Text-showing operators whose operands are the page's characters.
TEXT_SHOW_OPS = frozenset({"Tj", "TJ", "'", '"'})
#: Path-painting operators (contract section 3).
PATH_PAINT_OPS = frozenset({"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"})
#: Path-construction operators (contract section 3).
PATH_CONSTRUCT_OPS = frozenset({"re", "m", "l", "c", "v", "y"})
#: Everything counted by :attr:`PageProfile.vector_op_count`.
VECTOR_OPS = PATH_PAINT_OPS | PATH_CONSTRUCT_OPS
#: Painting operators that actually stroke the current path.
_STROKE_OPS = frozenset({"S", "s", "B", "B*", "b", "b*"})

#: Words that betray a form even when the page has no rules at all.  Matched
#: case-insensitively on word boundaries; a marker counts once however often it occurs.
FORM_MARKER_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("name", r"\bnames?\b"),
        ("address", r"\baddress(es)?\b"),
        ("date", r"\bdates?\b"),
        ("signature", r"\bsignatures?\b"),
        ("phone", r"\bphone\b"),
        ("email", r"\be-?mail\b"),
        ("city", r"\bcity\b"),
        ("state", r"\bstate\b"),
        ("zip", r"\bzip\b"),
        ("title", r"\btitle\b"),
        ("please print", r"\bplease\s+print\b"),
        ("check one", r"\bcheck\s+one\b"),
        ("yes", r"\byes\b"),
        ("no", r"\bno\b"),
    )
)

_INLINE_W_RE = re.compile(rb"/(?:W|Width)\s+(\d+)")
_INLINE_H_RE = re.compile(rb"/(?:H|Height)\s+(\d+)")
_NON_WS_RE = re.compile(r"\S")
_BOUNDARY = WHITESPACE | DELIMITERS


# ---------------------------------------------------------------------------------------
# The cheap content-stream scan
# ---------------------------------------------------------------------------------------
@dataclass
class ContentScan:
    """What one linear pass over a decoded content stream found.

    This is deliberately *not* an interpretation of the stream: no graphics state, no
    CTM, no text matrix, no font decoding.  It is the smallest amount of structure that
    lets triage answer "is there anything here but pixels?".
    """

    char_count: int = 0
    text: str = ""
    has_visible_text: bool = False
    vector_op_count: int = 0
    horizontal_rules: int = 0
    checkbox_glyphs: int = 0
    inline_images: List[Tuple[float, float]] = dataclass_field(default_factory=list)
    inline_image_count: int = 0
    placements: List[Tuple[float, float]] = dataclass_field(default_factory=list)
    do_names: List[str] = dataclass_field(default_factory=list)
    operator_count: int = 0
    truncated: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary (used by ``zfp preflight --debug``)."""
        return {
            "char_count": self.char_count,
            "has_visible_text": self.has_visible_text,
            "vector_op_count": self.vector_op_count,
            "horizontal_rules": self.horizontal_rules,
            "checkbox_glyphs": self.checkbox_glyphs,
            "inline_image_count": self.inline_image_count,
            "placements": [list(p) for p in self.placements],
            "do_names": list(self.do_names),
            "operator_count": self.operator_count,
            "truncated": self.truncated,
        }


def _number(value: Any) -> Optional[float]:
    """Return ``value`` as a float when it is a real PDF number, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _numbers(operands: Sequence[Any], count: int) -> Optional[List[float]]:
    """Return the last ``count`` operands as floats, or ``None`` if they are not numbers."""
    if len(operands) < count:
        return None
    out: List[float] = []
    for item in operands[-count:]:
        value = _number(item)
        if value is None:
            return None
        out.append(value)
    return out


def _decode_string(raw: Any) -> str:
    """Decode a content-stream string operand to text, best effort.

    Content-stream strings are encoded in whatever the *font* says, and resolving that
    requires the interpreter's font machinery.  Triage only needs enough text to spot
    marker words, so this decodes UTF-16 when the byte-order mark is present and falls
    back to Latin-1 -- which is exact for the WinAnsi/Standard encodings that carry
    virtually every English form label.
    """
    if not isinstance(raw, (bytes, bytearray)):
        return ""
    data = bytes(raw)
    if data[:2] in (b"\xfe\xff", b"\xff\xfe"):
        try:
            return data.decode("utf-16")
        except (UnicodeDecodeError, ValueError):  # pragma: no cover - defensive
            pass
    return data.decode("latin-1", "replace")


def _skip_inline_image(data: bytes, pos: int) -> Tuple[int, Optional[Tuple[float, float]]]:
    """Skip a ``BI ... ID <binary> EI`` region, returning ``(new position, size)``.

    Inline image data is raw bytes that happen to sit inside the content stream; lexing
    it would produce garbage tokens and could easily invent thousands of phantom
    operators.  The region is therefore skipped textually.  ``size`` is the image's
    ``/W``x``/H`` in pixels when the abbreviated dictionary declares them.

    Args:
        data: The whole decoded content stream.
        pos: Offset just past the ``BI`` operator.

    Returns:
        The offset just past the closing ``EI`` (or the end of the data when the region
        is unterminated), and the pixel size when it could be read.
    """
    n = len(data)
    id_pos = -1
    search = pos
    while search < n:
        found = data.find(b"ID", search)
        if found < 0:
            break
        before_ok = found == 0 or data[found - 1] in _BOUNDARY
        after = found + 2
        after_ok = after >= n or data[after] in _BOUNDARY
        if before_ok and after_ok:
            id_pos = found
            break
        search = found + 2
    if id_pos < 0:
        return (n, None)

    header = data[pos:id_pos]
    size: Optional[Tuple[float, float]] = None
    width_match = _INLINE_W_RE.search(header)
    height_match = _INLINE_H_RE.search(header)
    if width_match and height_match:
        size = (float(width_match.group(1)), float(height_match.group(1)))

    # Exactly one white-space byte separates ID from the binary payload.
    start = min(id_pos + 3, n)
    search = start
    while search < n:
        found = data.find(b"EI", search)
        if found < 0:
            return (n, size)
        before_ok = found > 0 and data[found - 1] in WHITESPACE
        after = found + 2
        after_ok = after >= n or data[after] in _BOUNDARY
        if before_ok and after_ok:
            return (after, size)
        search = found + 2
    return (n, size)


def scan_content(
    data: bytes,
    detection: Optional[DetectionConfig] = None,
    *,
    max_bytes: int = MAX_SCAN_BYTES,
) -> ContentScan:
    """Scan a decoded content stream once, cheaply, and report what it contains.

    The scan is a token walk, not an interpretation.  It tracks exactly three things
    beyond simple counting: the current point (so ``m``/``l`` runs can be measured), the
    current path's pending horizontal segments (so only *stroked* lines count as rules),
    and ``BI``/``EI`` regions (so binary inline-image data is never mistaken for
    operators).

    Args:
        data: The page's decoded, concatenated content stream.
        detection: Thresholds for what counts as a thin horizontal rule.
        max_bytes: Safety valve for pathological streams; the excess is ignored and
            :attr:`ContentScan.truncated` is set.

    Returns:
        A :class:`ContentScan`.  Never raises: malformed bytes simply yield fewer facts.
    """
    cfg = detection or DetectionConfig()
    scan = ContentScan()
    if not data:
        return scan
    if len(data) > max_bytes:
        data = data[:max_bytes]
        scan.truncated = True

    max_thickness = float(cfg.max_line_thickness_pt)
    min_length = float(cfg.min_line_length_pt)
    tolerance = float(cfg.line_merge_tolerance_pt)
    box_min = float(cfg.checkbox_min_pt)
    box_max = float(cfg.checkbox_max_pt)
    box_aspect = float(cfg.checkbox_aspect_tolerance)

    lexer = Lexer(data)
    operands: List[Any] = []
    array_stack: List[List[Any]] = []
    text_parts: List[str] = []
    text_chars = 0
    current: Optional[Tuple[float, float]] = None
    pending_rules = 0

    def add_text(raw: Any) -> None:
        nonlocal text_chars
        piece = _decode_string(raw)
        if not piece:
            return
        scan.char_count += len(piece)
        if not scan.has_visible_text and _NON_WS_RE.search(piece):
            scan.has_visible_text = True
        if text_chars < MAX_SCAN_TEXT_CHARS:
            text_parts.append(piece)
            text_chars += len(piece)

    def end_show() -> None:
        """Separate consecutive show operators so their words cannot fuse.

        ``(Yes) Tj (No) Tj`` are two runs on the page, not the word ``YesNo``; the
        elements of a single ``TJ`` array are *not* separated, because those really are
        one word split for kerning.  The separator is not counted in ``char_count``,
        which stays a faithful sum of the string payloads.
        """
        if text_parts and not text_parts[-1].endswith(" "):
            text_parts.append(" ")

    while True:
        token = lexer.next_token()
        kind = token.kind
        if kind == "eof":
            break

        if kind in ("num", "string", "hexstring", "name"):
            if array_stack:
                array_stack[-1].append(token.value)
            else:
                operands.append(token.value)
                if len(operands) > 8:
                    del operands[:-8]
            continue
        if kind == "array_open":
            array_stack.append([])
            continue
        if kind == "array_close":
            if array_stack:
                closed = array_stack.pop()
                if array_stack:
                    array_stack[-1].append(closed)
                else:
                    operands.append(closed)
            continue
        if kind in ("dict_open", "dict_close"):
            continue
        if kind != "keyword":  # pragma: no cover - the lexer emits nothing else
            continue

        op = token.value
        scan.operator_count += 1

        if op == "BI":
            new_pos, size = _skip_inline_image(data, lexer.pos)
            lexer.pos = new_pos
            scan.inline_image_count += 1
            if size is not None:
                scan.inline_images.append(size)
            operands.clear()
            array_stack.clear()
            continue

        if op in TEXT_SHOW_OPS:
            if op == "TJ":
                items = operands[-1] if operands and isinstance(operands[-1], list) else []
                for item in items:
                    add_text(item)
            else:
                if operands:
                    add_text(operands[-1])
            end_show()
        elif op in VECTOR_OPS:
            scan.vector_op_count += 1
            if op == "re":
                box = _numbers(operands, 4)
                if box is not None:
                    _x, _y, width, height = box
                    width, height = abs(width), abs(height)
                    if height <= max_thickness and width >= min_length:
                        scan.horizontal_rules += 1
                    elif (
                        box_min <= width <= box_max
                        and box_min <= height <= box_max
                        and abs(width - height) <= box_aspect * max(width, height)
                    ):
                        scan.checkbox_glyphs += 1
                current = None
            elif op == "m":
                point = _numbers(operands, 2)
                current = (point[0], point[1]) if point else None
            elif op == "l":
                point = _numbers(operands, 2)
                if point is not None:
                    if current is not None:
                        dx = point[0] - current[0]
                        dy = point[1] - current[1]
                        if abs(dy) <= tolerance and abs(dx) >= min_length:
                            pending_rules += 1
                    current = (point[0], point[1])
                else:
                    current = None
            elif op in ("c", "v", "y"):
                point = _numbers(operands, 2)
                current = (point[0], point[1]) if point else None
            if op in PATH_PAINT_OPS:
                if op in _STROKE_OPS:
                    scan.horizontal_rules += pending_rules
                pending_rules = 0
                current = None
        elif op == "n":
            # A path discarded without being painted draws nothing, so its segments are
            # not rules.  ``h`` (close subpath) is deliberately not handled: the path is
            # still live and may yet be stroked.
            pending_rules = 0
            current = None
        elif op == "cm":
            matrix = _numbers(operands, 6)
            if matrix is not None:
                width = math.hypot(matrix[0], matrix[1])
                height = math.hypot(matrix[2], matrix[3])
                scan.placements.append((width, height))
        elif op == "Do":
            if operands and isinstance(operands[-1], str):
                scan.do_names.append(operands[-1])

        operands.clear()
        array_stack.clear()

    # Stripped because the trailing show-operator separator is an artefact of the
    # scan, not page content; ``char_count`` is unaffected by either.
    scan.text = "".join(text_parts).strip()
    return scan


# ---------------------------------------------------------------------------------------
# Resource inspection
# ---------------------------------------------------------------------------------------
def _image_xobjects(doc: Document, page: Page) -> Dict[str, Tuple[float, float]]:
    """Inventory the image XObjects reachable from a page's resources.

    Nested form XObjects are walked (depth-limited, cycle-guarded) because scanners do
    sometimes wrap the page image in one; their images are keyed by ``"<form>/<name>"``
    so a caller can still tell a directly placed image from a nested one.

    Returns:
        Resource name -> ``(pixel width, pixel height)``.  A stream that declares no
        usable ``/Width``/``/Height`` reports ``(0.0, 0.0)`` -- it still proves the page
        carries raster imagery, it just cannot contribute to the area estimate.
    """
    found: Dict[str, Tuple[float, float]] = {}
    visited: set = set()

    def walk(resources: Any, prefix: str, depth: int) -> None:
        if depth > _MAX_XOBJECT_DEPTH or not isinstance(resources, PdfDict):
            return
        xobjects = doc.resolve(resources.get("XObject"))
        if not isinstance(xobjects, PdfDict):
            return
        for name in sorted(xobjects.keys()):
            raw = xobjects.get(name)
            if isinstance(raw, PdfRef):
                if raw.num in visited:
                    continue
                visited.add(raw.num)
            stream = doc.resolve(raw)
            if not isinstance(stream, PdfStream):
                continue
            subtype = stream.dict.get_name("Subtype", None, doc)
            key = "%s%s" % (prefix, name)
            if subtype == "Image":
                width = stream.dict.get_int("Width", 0, doc) or 0
                height = stream.dict.get_int("Height", 0, doc) or 0
                found[key] = (float(width), float(height))
            elif subtype == "Form":
                walk(doc.resolve(stream.dict.get("Resources")), key + "/", depth + 1)

    walk(page.resources(), "", 0)
    return found


def _has_widget_annotation(doc: Document, page: Page) -> bool:
    """True when the page's ``/Annots`` array carries a ``/Subtype /Widget``."""
    for annot in page.annotations():
        if annot.get_name("Subtype", None, doc) == "Widget":
            return True
    return False


def _estimate_image_area_ratio(
    images: Dict[str, Tuple[float, float]],
    scan: ContentScan,
    geometry: PageGeometry,
) -> float:
    """Estimate the fraction of the page covered by raster imagery.

    **This is an approximation and is meant to be.**  The exact answer needs the CTM in
    force at every ``Do``, which needs the full content-stream interpreter -- too
    expensive for a triage pass that runs before anything else.  Two cheap signals are
    used instead:

    1. *Placement matrices.*  Every ``cm`` operand set is reduced to the width and height
       of the unit square it maps.  When the page places exactly one image and some
       ``cm`` scales the unit square to at least
       :data:`FULL_PAGE_SCALE_RATIO` of both page dimensions, the page is treated as
       fully covered (ratio ``1.0``).  This is the overwhelmingly common scan layout
       (``q W H 0 0 0 0 cm /Im0 Do Q``) and it is right for it.
    2. *Pixel dimensions at a plausible scan resolution.*  Otherwise each image's
       ``/Width``x``/Height`` is converted to points at the DPI its own pixel count
       implies, clamped to :data:`MIN_IMAGE_DPI`..:data:`MAX_IMAGE_DPI`, and multiplied
       by the number of times the resource is actually invoked with ``Do``.

    Both branches deliberately over- rather than under-estimate: a page wrongly called
    raster-dominant merely gets an OCR pass it did not need, while a missed scan would
    silently produce an empty form.  The result is clamped to ``[0, 1]``.
    """
    page_width = float(geometry.width)
    page_height = float(geometry.height)
    if page_width <= 0.0 or page_height <= 0.0:  # pragma: no cover - geometry guards this
        return 0.0
    page_area = page_width * page_height

    sizes: List[Tuple[float, float, int]] = []
    for name, (px_w, px_h) in sorted(images.items()):
        placements = sum(1 for used in scan.do_names if used == name) or 1
        sizes.append((px_w, px_h, placements))
    for px_w, px_h in scan.inline_images:
        sizes.append((px_w, px_h, 1))

    if not sizes:
        return 0.0

    if len(sizes) == 1 and sizes[0][2] == 1:
        for width, height in scan.placements:
            if (
                width >= FULL_PAGE_SCALE_RATIO * page_width
                and height >= FULL_PAGE_SCALE_RATIO * page_height
            ):
                return 1.0

    total = 0.0
    for px_w, px_h, placements in sizes:
        if px_w <= 0.0 or px_h <= 0.0:
            continue
        implied = max(px_w * 72.0 / page_width, px_h * 72.0 / page_height)
        dpi = min(max(implied, MIN_IMAGE_DPI), MAX_IMAGE_DPI)
        placed_w = px_w * 72.0 / dpi
        placed_h = px_h * 72.0 / dpi
        total += (placed_w * placed_h) * placements
    return max(0.0, min(1.0, total / page_area))


def _marker_hits(text: str) -> List[str]:
    """Return the distinct form marker words present in ``text``, in table order."""
    if not text:
        return []
    return [label for label, pattern in FORM_MARKER_PATTERNS if pattern.search(text)]


def _has_form_cues(scan: ContentScan) -> bool:
    """True when a native page *looks* like a form.

    Three independent cues, any one of which is enough:

    * at least :data:`MIN_FORM_RULES` thin horizontal rules -- ``re`` rectangles whose
      height is under ``max_line_thickness_pt`` and whose width clears
      ``min_line_length_pt``, plus stroked ``m``/``l`` segments of the same shape.  The
      operands are raw, i.e. not run through the CTM, which is accurate for the
      unscaled coordinate systems form producers overwhelmingly use;
    * at least :data:`MIN_FORM_CHECKBOXES` checkbox-sized near-squares (a check-box
      form has no rules at all, so the rule cue alone would miss it entirely);
    * at least :data:`MIN_FORM_MARKERS` distinct marker words from
      :data:`FORM_MARKER_PATTERNS`.
    """
    if scan.horizontal_rules >= MIN_FORM_RULES:
        return True
    if scan.checkbox_glyphs >= MIN_FORM_CHECKBOXES:
        return True
    return len(_marker_hits(scan.text)) >= MIN_FORM_MARKERS


def _page_mode(
    *,
    has_widgets: bool,
    has_native_text: bool,
    has_vector: bool,
    has_raster: bool,
    vector_op_count: int,
    image_area_ratio: float,
    form_cues: bool,
) -> PageMode:
    """Apply the triage decision table.  Order is significant and is the contract's.

    1. widgets win outright -- the page already *is* a form, nothing may be re-detected;
    2. nothing at all -> ``EMPTY``;
    3. raster-dominant, no text, no meaningful vector layer -> ``SCANNED_FORM`` (the
       page is one image, so whether it is a form can only be settled after OCR; the
       form path is the one that keeps every option open);
    4. raster-dominant with no form cues -> ``SCANNED_DOCUMENT`` (the classic OCR'd
       scan: a full-page image plus an invisible text layer);
    5. raster plus native text above :data:`HYBRID_MIN_RASTER_RATIO` -> ``HYBRID``;
    6. native text with form cues -> ``FLAT_NATIVE_FORM``;
    7. otherwise ``NATIVE_DOCUMENT`` -- except that a page with no native text at all
       cannot be a native document, so a remaining raster page falls back to the scan
       modes.
    """
    if has_widgets:
        return PageMode.INTERACTIVE_FORM
    if not has_native_text and not has_vector and not has_raster:
        return PageMode.EMPTY

    raster_dominant = has_raster and image_area_ratio > RASTER_DOMINANT_RATIO
    vector_poor = vector_op_count <= VECTOR_POOR_MAX_OPS

    if raster_dominant and not has_native_text and vector_poor:
        return PageMode.SCANNED_FORM
    if raster_dominant and not form_cues:
        return PageMode.SCANNED_DOCUMENT
    if has_raster and has_native_text and image_area_ratio > HYBRID_MIN_RASTER_RATIO:
        return PageMode.HYBRID
    if has_native_text and form_cues:
        return PageMode.FLAT_NATIVE_FORM
    if not has_native_text and has_raster:
        # Raster-dominant, vector-rich, no text: still a scan, and the cues say which.
        return PageMode.SCANNED_FORM if form_cues else PageMode.SCANNED_DOCUMENT
    return PageMode.NATIVE_DOCUMENT


def classify_page(
    doc: Document, index: int, config: Optional[ZfpConfig] = None
) -> PageProfile:
    """Classify one page without rendering it.

    Args:
        doc: The open document.
        index: Zero-based page index.
        config: Run configuration; ``None`` uses :meth:`ZfpConfig.default`.

    Returns:
        A fully populated :class:`~zfp.core.types.PageProfile`.

    Raises:
        ValidationError: ``index`` is out of range.
    """
    cfg = config or ZfpConfig.default()
    page = doc.page(index)
    geometry = page.geometry

    has_widgets = _has_widget_annotation(doc, page)
    scan = scan_content(page.content_bytes(), cfg.detection)
    images = _image_xobjects(doc, page)

    has_raster = bool(images) or scan.inline_image_count > 0
    image_area_ratio = (
        _estimate_image_area_ratio(images, scan, geometry) if has_raster else 0.0
    )
    has_native_text = scan.char_count >= MIN_NATIVE_TEXT_CHARS and scan.has_visible_text
    has_vector = scan.vector_op_count >= MIN_VECTOR_OPS

    mode = _page_mode(
        has_widgets=has_widgets,
        has_native_text=has_native_text,
        has_vector=has_vector,
        has_raster=has_raster,
        vector_op_count=scan.vector_op_count,
        image_area_ratio=image_area_ratio,
        form_cues=_has_form_cues(scan),
    )

    return PageProfile(
        index=int(index),
        geometry=geometry,
        mode=mode,
        has_native_text=has_native_text,
        has_raster=has_raster,
        has_vector=has_vector,
        has_widgets=has_widgets,
        char_count=scan.char_count,
        image_area_ratio=round(image_area_ratio, 6),
        vector_op_count=scan.vector_op_count,
    )


# ---------------------------------------------------------------------------------------
# Document level
# ---------------------------------------------------------------------------------------
def _empty_profile(doc: Document, index: int) -> PageProfile:
    """The profile used for a page that could not be read at all."""
    try:
        geometry = doc.page(index).geometry
    except Exception:  # pragma: no cover - the page dictionary itself is broken
        geometry = PageGeometry(index=index, media_box=US_LETTER, crop_box=US_LETTER)
    return PageProfile(index=index, geometry=geometry, mode=PageMode.EMPTY)


def _producer(doc: Document) -> str:
    """Read ``/Info /Producer``, falling back to ``/Creator``; ``""`` when absent."""
    trailer = getattr(doc.file, "trailer", None)
    info = doc.resolve(trailer.get("Info")) if isinstance(trailer, dict) else None
    if not isinstance(info, PdfDict):
        return ""
    for key in ("Producer", "Creator"):
        try:
            text = info.get_text(key, None, doc)
        except Exception:  # pragma: no cover - locked strings in an encrypted file
            text = None
        if text:
            return text
    return ""


def _is_tagged(doc: Document) -> bool:
    """True when the catalog declares ``/MarkInfo << /Marked true >>``."""
    mark_info = doc.resolve(doc.catalog.get("MarkInfo"))
    if not isinstance(mark_info, PdfDict):
        return False
    return bool(mark_info.get_bool("Marked", False, doc))


def _acroform_field_count(doc: Document) -> int:
    """Number of terminal AcroForm fields, or ``0`` when the tree cannot be read."""
    try:
        return len(doc.existing_fields())
    except Exception as exc:  # pragma: no cover - a broken field tree
        _log.debug("%s: field tree unreadable: %s", doc.document_id, exc)
        return 0


def profile_document(
    doc: Document, config: Optional[ZfpConfig] = None
) -> DocumentProfile:
    """Profile a whole document: every page, plus the document-level facts.

    This function never raises because of page content.  A page that cannot be read is
    reported as :attr:`PageMode.EMPTY` with a ``page N unparseable: ...`` warning, so a
    single corrupt object can never take down a run.

    Args:
        doc: The open document.
        config: Run configuration; ``None`` uses :meth:`ZfpConfig.default`.

    Returns:
        A :class:`~zfp.core.types.DocumentProfile` whose ``doc_class`` is the routing
        decision (see :func:`route`).
    """
    cfg = config or ZfpConfig.default()
    warnings: List[str] = []

    state = inspect_security(doc)
    encrypted = state.encrypted
    # ``can_modify`` answers the only question routing actually asks: *may ZFP write to
    # this document at all?*  That is the union of the "modify" and "fill form fields"
    # permissions, which is exactly the test
    # :func:`zfp.preflight.security.can_add_form_fields` applies -- so the route and the
    # security gate can never disagree about the same file.
    can_modify = True
    if encrypted:
        can_modify = bool(state.can_modify or state.can_fill_forms)
        if not state.authenticated:
            warnings.append("encrypted: structure-only access (no password accepted)")
        elif not can_modify:
            warnings.append(
                "encrypted (%s, /P=%d): permissions deny modification"
                % (state.method, state.permissions)
            )
        else:
            warnings.append(
                "encrypted (%s): incremental-only edits permitted" % state.method
            )

    for message in list(getattr(doc.file, "warnings", []) or []):
        text = "parser: %s" % (message,)
        if text not in warnings:
            warnings.append(text)

    try:
        acroform = doc.acroform() is not None
    except Exception as exc:  # pragma: no cover - defensive
        acroform = False
        warnings.append("acroform unreadable: %s" % (exc,))
    try:
        xfa = doc.has_xfa()
        dynamic_xfa = doc.xfa_is_dynamic() if xfa else False
    except Exception as exc:  # pragma: no cover - defensive
        xfa = dynamic_xfa = False
        warnings.append("xfa unreadable: %s" % (exc,))
    try:
        signed = doc.is_signed()
    except Exception as exc:  # pragma: no cover - defensive
        signed = False
        warnings.append("signature state unreadable: %s" % (exc,))

    if dynamic_xfa:
        warnings.append("dynamic XFA: routed to compatibility layer")
    elif xfa:
        warnings.append("static XFA: treated as its AcroForm shadow")
    if signed:
        warnings.append("signed: incremental-only edits permitted")

    try:
        page_count = doc.page_count
    except Exception as exc:  # pragma: no cover - a broken page tree
        page_count = 0
        warnings.append("page tree unreadable: %s" % (exc,))
    if page_count == 0:
        warnings.append("no pages could be parsed")

    pages: List[PageProfile] = []
    for index in range(page_count):
        try:
            pages.append(classify_page(doc, index, cfg))
        except Exception as exc:
            warnings.append("page %d unparseable: %s" % (index, exc))
            _log.warning("%s: page %d unparseable: %s", doc.document_id, index, exc)
            pages.append(_empty_profile(doc, index))

    profile = DocumentProfile(
        document_id=doc.document_id,
        page_count=page_count,
        pages=pages,
        encrypted=encrypted,
        can_modify=can_modify,
        signed=signed,
        acroform=acroform,
        xfa=xfa,
        dynamic_xfa=dynamic_xfa,
        tagged=_is_tagged(doc),
        producer=_producer(doc),
        version=str(getattr(doc.file, "version", "") or ""),
        warnings=warnings,
    )
    field_count = _acroform_field_count(doc) if acroform else 0
    profile.doc_class = _route(profile, field_count)
    if acroform and field_count > 0 and not any(p.has_widgets for p in pages):
        profile.warnings.append(
            "AcroForm declares %d field(s) but no page carries a widget annotation"
            % field_count
        )
    return profile


def _route(profile: DocumentProfile, field_count: Optional[int]) -> DocumentClass:
    """The routing table.  ``field_count`` is ``None`` when it is not known."""
    # 1. Security first: what may be done outranks what the pages contain.
    if profile.encrypted and not profile.can_modify:
        return DocumentClass.ENCRYPTED
    # 2. A signature restricts every later decision, so it outranks form flavour.
    if profile.signed:
        return DocumentClass.SIGNED
    # 3. Dynamic XFA is a different rendering model; only the compat layer may touch it.
    if profile.dynamic_xfa:
        return DocumentClass.XFA
    # 4. An existing interactive form is ground truth -- never re-detect it.  Static XFA
    #    with an AcroForm routes here too: its AcroForm shadow is a real form.
    has_widget_page = any(p.has_widgets for p in profile.pages)
    if has_widget_page:
        return DocumentClass.EXISTING_ACROFORM
    # An /AcroForm with no widget on any page only counts when it is known to hold
    # fields (or is the shadow of a static XFA packet).  An empty AcroForm dictionary,
    # which plenty of producers leave behind, must not send a document down the
    # read-the-existing-form path where it would find nothing at all.
    if profile.acroform and (profile.xfa or (field_count or 0) > 0):
        return DocumentClass.EXISTING_ACROFORM
    # 5. Otherwise the pages decide.
    return _majority_class(profile)


def _majority_class(profile: DocumentProfile) -> DocumentClass:
    """Fold the per-page modes into one document class.

    * no form-ish page at all -> ``NON_FORM``;
    * hybrid pages at least as common as either pure kind -> ``HYBRID``;
    * native form pages mixed with raster pages -> ``HYBRID`` (both perception systems
      are genuinely required);
    * otherwise the more common of ``SCANNED_FORM`` / ``FLAT_NATIVE_FORM``, ties going
      to ``SCANNED_FORM`` because an unnecessary OCR pass is cheaper than a missed one.
    """
    counts: Dict[PageMode, int] = {}
    for page in profile.pages:
        counts[page.mode] = counts.get(page.mode, 0) + 1
    scanned_form = counts.get(PageMode.SCANNED_FORM, 0)
    flat = counts.get(PageMode.FLAT_NATIVE_FORM, 0)
    hybrid = counts.get(PageMode.HYBRID, 0)
    scanned_doc = counts.get(PageMode.SCANNED_DOCUMENT, 0)

    if scanned_form + flat + hybrid == 0:
        return DocumentClass.NON_FORM
    if hybrid and hybrid >= max(scanned_form, flat):
        return DocumentClass.HYBRID
    if flat and (scanned_form or scanned_doc or hybrid):
        return DocumentClass.HYBRID
    if scanned_form >= flat:
        return DocumentClass.SCANNED_FORM
    return DocumentClass.FLAT_NATIVE_FORM


def route(profile: DocumentProfile) -> DocumentClass:
    """Decide which pipeline a document belongs in.

    Precedence, highest first -- and the order matters, because several of these are
    true at once on real files:

    1. ``ENCRYPTED``  -- encrypted *and* not modifiable.  Nothing may be written.
    2. ``SIGNED``     -- a signature exists; only an incremental revision is legal.
    3. ``XFA``        -- *dynamic* XFA.  Static XFA is not routed here: it behaves like
       a plain AcroForm and goes to ``EXISTING_ACROFORM``.
    4. ``EXISTING_ACROFORM`` -- any page carries a widget, or an AcroForm exists with
       fields.  Existing fields are read, never re-detected.
    5. page-mode majority -- ``SCANNED_FORM`` / ``FLAT_NATIVE_FORM`` / ``HYBRID``.
    6. ``NON_FORM``   -- nothing form-like anywhere.

    Note:
        Called with a profile alone, step 4 can only see widget annotations and the XFA
        flag, since a :class:`DocumentProfile` carries no field count.
        :func:`profile_document` knows the real count and folds it in, so
        ``profile.doc_class`` is the authoritative answer for the rare document whose
        fields exist but whose widgets are not reachable from any page -- ``route()``
        alone classifies that one by page mode instead.
    """
    return _route(profile, None)


# ---------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------
def profile_to_dict(profile: DocumentProfile) -> Dict[str, Any]:
    """Return the profile as JSON-ready primitives (``zfp.core.serde``)."""
    return to_jsonable(profile)


def _page_line(page: PageProfile) -> str:
    """One aligned line of the per-page table."""
    geometry = page.geometry
    size = "%gx%g" % (round(geometry.width, 2), round(geometry.height, 2))
    flags = []
    if page.has_widgets:
        flags.append("widgets")
    if page.has_native_text:
        flags.append("text")
    if page.has_vector:
        flags.append("vector")
    if page.has_raster:
        flags.append("raster")
    if geometry.rotation:
        flags.append("rot%d" % geometry.rotation)
    return "    %4d  %-17s %-12s chars=%-7d vec=%-6d img=%.2f  %s" % (
        page.index,
        page.mode.value,
        size,
        page.char_count,
        page.vector_op_count,
        page.image_area_ratio,
        " ".join(flags) if flags else "-",
    )


def describe(profile: DocumentProfile) -> str:
    """Render the human-readable preflight block the CLI prints.

    Deterministic and free of wall-clock values, so two runs over the same bytes
    produce byte-identical output and the block can be diffed in QA.
    """
    security = "encrypted" if profile.encrypted else "unencrypted"
    security += ", modifiable" if profile.can_modify else ", read-only"
    security += ", signed" if profile.signed else ", unsigned"

    form = "acroform=%s xfa=%s tagged=%s" % (
        "yes" if profile.acroform else "no",
        ("dynamic" if profile.dynamic_xfa else "static") if profile.xfa else "no",
        "yes" if profile.tagged else "no",
    )

    lines: List[str] = [
        "preflight  %s" % profile.document_id,
        "  class      %s" % profile.doc_class.value,
        "  pages      %d" % profile.page_count,
        "  version    PDF %s" % (profile.version or "?"),
        "  security   %s" % security,
        "  form       %s" % form,
    ]
    if profile.producer:
        lines.append("  producer   %s" % profile.producer)
    if profile.pages:
        lines.append("  page modes")
        lines.extend(_page_line(page) for page in profile.pages)
    lines.append("  native text pages  %s" % (profile.native_text_pages or "[]"))
    lines.append("  raster pages       %s" % (profile.raster_pages or "[]"))
    if profile.warnings:
        lines.append("  warnings")
        lines.extend("    ! %s" % warning for warning in profile.warnings)
    return "\n".join(lines)
