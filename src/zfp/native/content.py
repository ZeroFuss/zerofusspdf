"""The content-stream interpreter: a born-digital page turned into geometry.

Everything ZFP knows about a native page comes from here.  :class:`ContentStreamInterpreter`
walks a page's decoded content stream and produces

* :class:`~zfp.core.types.TextSpan` objects with a real user-space box **per glyph**,
  the baseline y, the resolved base font and the effective (transform-scaled) font size;
* :class:`~zfp.core.types.VectorPrimitive` objects for every painted subpath, classified
  as ``line`` / ``rect`` / ``circle`` / ``path`` and carrying the subpath's own points;
* the transformed bounding boxes of image XObjects and inline images;
* the boxes of *white* fills, which is how the blank-region detector tells "paper" from
  "nothing was ever drawn here".

Three behaviours are load-bearing and easy to get wrong, so they are stated explicitly:

**Invisible text is kept, but marked.**  Render modes 3 (invisible) and 7 (clip-only) are
exactly how a scanner hides an OCR layer behind a page image.  Those spans are emitted
with ``confidence == 0.0``; :attr:`ContentResult.visible_spans` filters them out.  A
detector that treats them as visible ink will hallucinate fields all over a scan.

**A clip-only path is not ink.**  ``W n`` establishes a clipping region and paints
nothing, so it produces no primitive.  ``W`` followed by a real painting operator does
paint, and does.

**White fills are still primitives.**  A white filled rectangle is emitted with
``kind="rect"``, ``filled=True`` and ``stroked=False`` (the fill-only convention), and its
box is additionally listed in :attr:`ContentResult.white_fills`.  Downstream code that
wants "regions that look blank" reads ``white_fills``; code that wants "regions that were
painted" reads ``primitives``.

The interpreter never raises on bad content.  A per-operator failure bumps
:attr:`ContentResult.errors` and execution continues with the next operator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ..core.config import ZfpConfig
from ..core.geometry import Matrix, Point, Rect
from ..core.logging import get_logger
from ..core.types import TextSpan, VectorPrimitive
from ..pdfio.lexer import Lexer
from ..pdfio.objects import PdfDict, PdfName, PdfNull, PdfStream, resolve_with
from .encoding import FontProgram, load_font

__all__ = [
    "WHITE_LUMINANCE",
    "MAX_FORM_DEPTH",
    "ContentState",
    "ContentResult",
    "ContentStreamInterpreter",
    "analyze_page",
]

_log = get_logger(__name__)

#: A fill at or above this luminance counts as white (paper, not ink).
WHITE_LUMINANCE = 0.95
#: How deep ``Do`` may recurse into Form XObjects.
MAX_FORM_DEPTH = 8
#: Segments used when flattening a cubic Bezier for bounding and classification.
_BEZIER_STEPS = 12
#: Upper bound on the points kept in :attr:`VectorPrimitive.points`.
_MAX_POINTS = 64
#: A bbox this close to square counts as "near square" for circle detection.
_SQUARE_RATIO = 1.35
#: Operand stack guard: content streams in the wild carry junk before an operator.
_MAX_OPERANDS = 96

_PAINT_OPS = frozenset(("S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n"))
_FILL_OPS = frozenset(("f", "F", "f*", "B", "B*", "b", "b*"))
_STROKE_OPS = frozenset(("S", "s", "B", "B*", "b", "b*"))
_CLOSE_OPS = frozenset(("s", "b", "b*"))
#: Text render modes that put no marks on the page.
INVISIBLE_RENDER_MODES = frozenset((3, 7))


# --------------------------------------------------------------------------------------
# Graphics state
# --------------------------------------------------------------------------------------

@dataclass
class ContentState:
    """The graphics and text state at one point in a content stream.

    The contract's fields come first; the colour fields after them are what lets the
    interpreter answer "was that fill white?", which the blank-region detector needs and
    which cannot be recovered later.

    ``text_matrix`` and ``line_matrix`` live here for convenience but are *not* part of
    the graphics state proper: ``BT`` resets them and ``Q`` deliberately leaves them
    alone, exactly as the PDF specification requires.
    """

    ctm: Matrix = field(default_factory=Matrix.identity)
    text_matrix: Matrix = field(default_factory=Matrix.identity)
    line_matrix: Matrix = field(default_factory=Matrix.identity)
    font: str = ""
    font_size: float = 0.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    horizontal_scale: float = 100.0
    leading: float = 0.0
    rise: float = 0.0
    render_mode: int = 0
    stroke_width: float = 1.0
    # -- colour (extension) -----------------------------------------------------------
    fill_luminance: float = 0.0
    stroke_luminance: float = 0.0
    fill_is_pattern: bool = False
    stroke_is_pattern: bool = False
    fill_space: str = "gray"
    stroke_space: str = "gray"

    def copy(self) -> ContentState:
        """Return an independent copy (every field is an immutable value)."""
        return replace(self)

    @property
    def horizontal_factor(self) -> float:
        """``Tz`` as a plain multiplier (100 -> 1.0)."""
        return self.horizontal_scale / 100.0

    def fill_is_white(self) -> bool:
        """True when the current non-stroking colour is paper-white."""
        return not self.fill_is_pattern and self.fill_luminance >= WHITE_LUMINANCE

    def scaled_line_width(self) -> float:
        """Line width in user space: ``w`` scaled by the CTM's isotropic factor."""
        det = abs(self.ctm.determinant())
        return float(self.stroke_width) * (math.sqrt(det) if det > 0.0 else 1.0)


# --------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------

@dataclass
class ContentResult:
    """Everything one page's content stream produced.

    Attributes:
        spans: Text spans in stream order, invisible ones included.
        primitives: Painted subpaths in stream order.
        images: Transformed boxes of image XObjects and inline images.
        op_count: How many operators were executed, recursion included.
        errors: How many operators failed and were skipped.
        white_fills: Boxes of every fill painted in a near-white colour.
    """

    spans: List[TextSpan] = field(default_factory=list)
    primitives: List[VectorPrimitive] = field(default_factory=list)
    images: List[Rect] = field(default_factory=list)
    op_count: int = 0
    errors: int = 0
    white_fills: List[Rect] = field(default_factory=list)

    @property
    def visible_spans(self) -> List[TextSpan]:
        """The spans that actually put marks on the page (render mode not 3 or 7)."""
        return [span for span in self.spans if span.confidence > 0.0]

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready summary (spans and primitives fully expanded)."""
        return {
            "spans": [span.as_dict() for span in self.spans],
            "primitives": [prim.as_dict() for prim in self.primitives],
            "images": [rect.as_list() for rect in self.images],
            "op_count": self.op_count,
            "errors": self.errors,
            "white_fills": [rect.as_list() for rect in self.white_fills],
        }


# --------------------------------------------------------------------------------------
# Path construction
# --------------------------------------------------------------------------------------

class _SubPath:
    """One subpath, already in user space.

    Points are transformed by the CTM in force when each point was constructed, which is
    what the specification requires and what makes a ``cm`` between ``m`` and ``l``
    behave correctly.
    """

    __slots__ = ("points", "closed", "from_rect", "beziers", "segments")

    def __init__(self, start: Tuple[float, float]) -> None:
        self.points: List[Tuple[float, float]] = [start]
        self.closed: bool = False
        self.from_rect: bool = False
        self.beziers: int = 0
        self.segments: int = 0

    @property
    def current(self) -> Tuple[float, float]:
        return self.points[-1]

    def line_to(self, point: Tuple[float, float]) -> None:
        self.points.append(point)
        self.segments += 1

    def bbox(self) -> Rect:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return Rect(min(xs), min(ys), max(xs), max(ys))

    def distinct_count(self) -> int:
        """Number of points ignoring an exact repeat of the previous one."""
        count = 0
        previous: Optional[Tuple[float, float]] = None
        for point in self.points:
            if previous is None or point != previous:
                count += 1
            previous = point
        return count

    def is_axis_aligned_rect(self) -> bool:
        """True when the points trace an axis-aligned rectangle (4 or 5 vertices)."""
        if self.from_rect:
            return True
        pts = self.points
        if len(pts) == 5 and _close(pts[0][0], pts[4][0]) and _close(pts[0][1], pts[4][1]):
            pts = pts[:4]
        if len(pts) != 4 or self.beziers:
            return False
        for index in range(4):
            ax, ay = pts[index]
            bx, by = pts[(index + 1) % 4]
            if not (_close(ax, bx) or _close(ay, by)):
                return False
        return True


class _PathBuilder:
    """Accumulates subpaths until a painting operator consumes them."""

    __slots__ = ("subpaths", "start")

    def __init__(self) -> None:
        self.subpaths: List[_SubPath] = []
        self.start: Optional[Tuple[float, float]] = None

    def move_to(self, point: Tuple[float, float]) -> None:
        self.subpaths.append(_SubPath(point))
        self.start = point

    def ensure(self, point: Tuple[float, float]) -> _SubPath:
        """Return the open subpath, starting one at ``point`` when there is none."""
        if not self.subpaths:
            self.move_to(point)
        return self.subpaths[-1]

    def close(self) -> None:
        if not self.subpaths:
            return
        sub = self.subpaths[-1]
        sub.closed = True
        if len(sub.points) > 1 and sub.points[0] != sub.points[-1]:
            sub.points.append(sub.points[0])

    def clear(self) -> None:
        self.subpaths = []
        self.start = None


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    """Float comparison used for rectangle-shape tests."""
    return abs(a - b) <= tol


def _flatten_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Sample a cubic Bezier, excluding its start point.

    Flattening rather than using the control hull matters: the control points of a
    circle approximation sit well outside the circle, and a checkbox-sized error there
    is the difference between "circle" and "not a checkbox".
    """
    out: List[Tuple[float, float]] = []
    for step in range(1, _BEZIER_STEPS + 1):
        t = step / float(_BEZIER_STEPS)
        u = 1.0 - t
        a = u * u * u
        b = 3.0 * u * u * t
        c = 3.0 * u * t * t
        d = t * t * t
        out.append(
            (
                a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Colour
# --------------------------------------------------------------------------------------

_SPACE_ALIASES = {
    "DeviceGray": "gray",
    "CalGray": "gray",
    "G": "gray",
    "DeviceRGB": "rgb",
    "CalRGB": "rgb",
    "RGB": "rgb",
    "DeviceCMYK": "cmyk",
    "CMYK": "cmyk",
    "Pattern": "pattern",
    "Indexed": "indexed",
    "I": "indexed",
    "Separation": "subtractive",
    "DeviceN": "subtractive",
    "Lab": "other",
}


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _luminance(space: str, values: Sequence[float]) -> Optional[float]:
    """Perceived luminance of a colour, or ``None`` when it cannot be known.

    ``subtractive`` covers ``/Separation`` and ``/DeviceN``, where a tint of 0 means "no
    ink" -- that is, white paper -- and 1 means full ink.
    """
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if space == "subtractive":
        if not numbers:
            return None
        return _clamp01(1.0 - max(numbers))
    if space == "indexed" or space == "pattern":
        return None
    if space == "gray" and len(numbers) >= 1:
        return _clamp01(numbers[0])
    if space == "rgb" and len(numbers) >= 3:
        r, g, b = (_clamp01(numbers[0]), _clamp01(numbers[1]), _clamp01(numbers[2]))
        return 0.299 * r + 0.587 * g + 0.114 * b
    if space == "cmyk" and len(numbers) >= 4:
        c, m, y, k = (_clamp01(n) for n in numbers[:4])
        r = (1.0 - c) * (1.0 - k)
        g = (1.0 - m) * (1.0 - k)
        b = (1.0 - y) * (1.0 - k)
        return 0.299 * r + 0.587 * g + 0.114 * b
    # Unknown space: fall back to the operand count, which is what viewers do.
    if len(numbers) == 1:
        return _clamp01(numbers[0])
    if len(numbers) == 3:
        return 0.299 * _clamp01(numbers[0]) + 0.587 * _clamp01(numbers[1]) + 0.114 * _clamp01(numbers[2])
    if len(numbers) == 4:
        return _luminance("cmyk", numbers)
    return None


# --------------------------------------------------------------------------------------
# The interpreter
# --------------------------------------------------------------------------------------

class ContentStreamInterpreter:
    """Execute one page's content stream and collect its geometry.

    Args:
        page: The :class:`~zfp.pdfio.document.Page` to interpret.
        resolver: Anything with ``.resolve`` (the owning :class:`Document` is the usual
            choice), or ``None`` to take the page's own document.
        config: Detection thresholds; ``ZfpConfig.default()`` when omitted.  Only
            ``config.detection.max_line_thickness_pt`` is consulted, as the cut-off
            below which a painted subpath is a rule rather than a box.

    Examples:
        >>> from zfp.pdfio.document import Document
        >>> from zfp.pdfio.objects import PdfDict, PdfStream
        >>> doc = Document.from_pages_blank(1)
        >>> page = doc.page(0)
        >>> doc.writer.set_object(
        ...     page.dict["Contents"].num,
        ...     PdfStream(PdfDict({}), b"BT /F1 12 Tf 100 700 Td (Hi) Tj ET"),
        ... )
        >>> result = ContentStreamInterpreter(page, doc).run()
        >>> result.spans[0].text, result.spans[0].baseline
        ('Hi', 700.0)
    """

    #: Recursion limit for ``Do`` on Form XObjects.
    MAX_FORM_DEPTH = MAX_FORM_DEPTH

    def __init__(self, page: Any, resolver: Any = None, config: Optional[ZfpConfig] = None) -> None:
        self.page = page
        self.resolver = resolver if resolver is not None else getattr(page, "document", None)
        self.config = config if config is not None else ZfpConfig.default()
        self.page_index = int(getattr(page, "index", 0))
        self._font_cache: Dict[int, FontProgram] = {}
        self._font_keepalive: List[Any] = []
        self._thin = float(self.config.detection.max_line_thickness_pt)

    # -- entry point ------------------------------------------------------------------
    def run(self) -> ContentResult:
        """Interpret the page and return everything it drew.

        Returns:
            A populated :class:`ContentResult`.  Never raises: a page whose content is
            unreadable comes back empty with ``errors >= 1``.
        """
        result = ContentResult()
        try:
            data = self.page.content_bytes()
        except Exception:
            _log.debug("page %d: content could not be decoded", self.page_index)
            result.errors += 1
            return result
        try:
            resources = self.page.resources()
        except Exception:  # pragma: no cover - resources() is already lenient
            resources = PdfDict()
            result.errors += 1
        state = ContentState()
        self._execute(data, resources, state, 0, set(), result)
        return result

    # -- resolution helpers -----------------------------------------------------------
    def _resolve(self, value: Any) -> Any:
        """Follow references; PDF ``null`` becomes ``None``."""
        out = resolve_with(value, self.resolver)
        if out is None or isinstance(out, PdfNull):
            return None
        return out

    def _resource(self, resources: Any, category: str, name: str) -> Any:
        """Look up ``/<category> /<name>`` in a resource dictionary."""
        holder = self._resolve(resources)
        if not isinstance(holder, dict):
            return None
        group = self._resolve(holder.get(category))
        if not isinstance(group, dict):
            return None
        return self._resolve(group.get(name))

    def _font_for(self, resources: Any, name: str) -> FontProgram:
        """Return the cached :class:`FontProgram` for a ``Tf`` resource name."""
        font_dict = self._resource(resources, "Font", name)
        key = id(font_dict) if font_dict is not None else 0
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        program = load_font(font_dict, self.resolver)
        self._font_cache[key] = program
        if font_dict is not None:
            # Keep a strong reference so the id() key can never be recycled mid-run.
            self._font_keepalive.append(font_dict)
        return program

    # -- operand parsing --------------------------------------------------------------
    def _read_array(self, lexer: Lexer) -> List[Any]:
        """Collect operands up to the matching ``]`` (the ``[`` is already consumed)."""
        items: List[Any] = []
        for _ in range(100000):
            token = lexer.next_token()
            if token.kind in ("eof", "array_close"):
                return items
            if token.kind == "name":
                items.append(PdfName(token.value))
            elif token.kind == "array_open":
                items.append(self._read_array(lexer))
            elif token.kind == "dict_open":
                items.append(self._read_dict(lexer))
            elif token.kind == "dict_close":
                continue
            else:
                items.append(token.value)
        return items  # pragma: no cover - only a pathological stream reaches this

    def _read_dict(self, lexer: Lexer) -> PdfDict:
        """Collect a ``<< ... >>`` operand (the ``<<`` is already consumed)."""
        out = PdfDict()
        key: Optional[str] = None
        for _ in range(100000):
            token = lexer.next_token()
            if token.kind in ("eof", "dict_close"):
                return out
            if key is None:
                if token.kind == "name":
                    key = token.value
                continue
            if token.kind == "name":
                out[key] = PdfName(token.value)
            elif token.kind == "array_open":
                out[key] = self._read_array(lexer)
            elif token.kind == "dict_open":
                out[key] = self._read_dict(lexer)
            elif token.kind == "keyword":
                out[key] = {"true": True, "false": False}.get(token.value, token.value)
            else:
                out[key] = token.value
            key = None
        return out  # pragma: no cover - only a pathological stream reaches this

    # -- the token loop ---------------------------------------------------------------
    def _execute(
        self,
        data: bytes,
        resources: Any,
        state: ContentState,
        depth: int,
        visited: Set[int],
        result: ContentResult,
    ) -> None:
        """Run one content stream (a page's, or a Form XObject's) to completion."""
        if not data:
            return
        lexer = Lexer(data)
        operands: List[Any] = []
        stack: List[ContentState] = []
        path = _PathBuilder()
        guard = 0
        limit = len(data) * 4 + 4096
        while guard < limit:
            guard += 1
            token = lexer.next_token()
            kind = token.kind
            if kind == "eof":
                break
            if kind == "num" or kind == "string" or kind == "hexstring":
                operands.append(token.value)
            elif kind == "name":
                operands.append(PdfName(token.value))
            elif kind == "array_open":
                operands.append(self._read_array(lexer))
            elif kind == "dict_open":
                operands.append(self._read_dict(lexer))
            elif kind in ("array_close", "dict_close"):
                continue
            elif kind == "keyword":
                word = token.value
                if word == "true":
                    operands.append(True)
                    continue
                if word == "false":
                    operands.append(False)
                    continue
                if word == "null":
                    operands.append(None)
                    continue
                result.op_count += 1
                try:
                    if word == "BI":
                        self._inline_image(lexer, state, result)
                    elif word in ("W", "W*"):
                        # Clipping paths narrow the visible area only; the path itself is
                        # painted (or not) by the operator that follows, so nothing to do.
                        pass
                    elif word in _PAINT_OPS:
                        try:
                            self._paint(word, path, state, result)
                        finally:
                            # A painting operator always ends the path, even when the
                            # primitive it produced could not be built.
                            path.clear()
                    else:
                        self._operator(
                            word, operands, state, stack, path, resources, depth, visited, result
                        )
                except Exception:
                    result.errors += 1
                    _log.debug("page %d: operator %r failed", self.page_index, word)
                operands = []
                continue
            if len(operands) > _MAX_OPERANDS:
                del operands[: len(operands) - 32]

    # -- operator dispatch ------------------------------------------------------------
    def _operator(
        self,
        op: str,
        operands: List[Any],
        state: ContentState,
        stack: List[ContentState],
        path: _PathBuilder,
        resources: Any,
        depth: int,
        visited: Set[int],
        result: ContentResult,
    ) -> None:
        """Apply one non-painting, non-inline-image operator."""
        # -- graphics state ------------------------------------------------------------
        if op == "q":
            stack.append(state.copy())
            return
        if op == "Q":
            if stack:
                saved = stack.pop()
                # Q restores the graphics state; the text matrices are not part of it.
                text_matrix, line_matrix = state.text_matrix, state.line_matrix
                _assign(state, saved)
                state.text_matrix = text_matrix
                state.line_matrix = line_matrix
            return
        if op == "cm":
            values = _numbers(operands, 6)
            if values is not None:
                state.ctm = Matrix(*values).concat(state.ctm)
            return
        if op == "w":
            values = _numbers(operands, 1)
            if values is not None:
                state.stroke_width = values[0]
            return
        if op == "gs":
            self._ext_gstate(operands, state, resources)
            return

        # -- colour --------------------------------------------------------------------
        if op in ("g", "G", "rg", "RG", "k", "K", "cs", "CS", "sc", "scn", "SC", "SCN"):
            self._colour(op, operands, state, resources)
            return

        # -- path construction ---------------------------------------------------------
        if op == "m":
            values = _numbers(operands, 2)
            if values is not None:
                path.move_to(state.ctm.apply_xy(values[0], values[1]))
            return
        if op == "l":
            values = _numbers(operands, 2)
            if values is not None:
                point = state.ctm.apply_xy(values[0], values[1])
                path.ensure(point).line_to(point)
            return
        if op in ("c", "v", "y"):
            self._curve(op, operands, path, state)
            return
        if op == "h":
            path.close()
            return
        if op == "re":
            values = _numbers(operands, 4)
            if values is not None:
                self._rectangle(values, path, state)
            return

        # -- text ----------------------------------------------------------------------
        if op == "BT":
            state.text_matrix = Matrix.identity()
            state.line_matrix = Matrix.identity()
            return
        if op == "ET":
            return
        if op == "Tf":
            self._select_font(operands, state)
            return
        if op == "Td":
            values = _numbers(operands, 2)
            if values is not None:
                self._next_line(state, values[0], values[1])
            return
        if op == "TD":
            values = _numbers(operands, 2)
            if values is not None:
                state.leading = -values[1]
                self._next_line(state, values[0], values[1])
            return
        if op == "Tm":
            values = _numbers(operands, 6)
            if values is not None:
                state.text_matrix = Matrix(*values)
                state.line_matrix = state.text_matrix
            return
        if op == "T*":
            self._next_line(state, 0.0, -state.leading)
            return
        if op == "TL":
            values = _numbers(operands, 1)
            if values is not None:
                state.leading = values[0]
            return
        if op == "Tc":
            values = _numbers(operands, 1)
            if values is not None:
                state.char_spacing = values[0]
            return
        if op == "Tw":
            values = _numbers(operands, 1)
            if values is not None:
                state.word_spacing = values[0]
            return
        if op == "Tz":
            values = _numbers(operands, 1)
            if values is not None:
                state.horizontal_scale = values[0]
            return
        if op == "Ts":
            values = _numbers(operands, 1)
            if values is not None:
                state.rise = values[0]
            return
        if op == "Tr":
            values = _numbers(operands, 1)
            if values is not None:
                state.render_mode = int(values[0])
            return
        if op == "Tj":
            raw = _last_bytes(operands)
            if raw is not None:
                self._show(raw, state, resources, result)
            return
        if op == "TJ":
            items = operands[-1] if operands and isinstance(operands[-1], list) else None
            if items is not None:
                self._show_array(items, state, resources, result)
            return
        if op == "'":
            raw = _last_bytes(operands)
            self._next_line(state, 0.0, -state.leading)
            if raw is not None:
                self._show(raw, state, resources, result)
            return
        if op == '"':
            raw = _last_bytes(operands)
            values = _numbers(operands[:-1], 2) if len(operands) >= 3 else None
            if values is not None:
                state.word_spacing = values[0]
                state.char_spacing = values[1]
            self._next_line(state, 0.0, -state.leading)
            if raw is not None:
                self._show(raw, state, resources, result)
            return

        # -- XObjects ------------------------------------------------------------------
        if op == "Do":
            name = _last_name(operands)
            if name is not None:
                self._do_xobject(name, state, resources, depth, visited, result)
            return

        # Everything else (BMC/BDC/EMC, MP/DP, sh, d, i, j, J, M, ri, d0, d1, BX, EX and
        # any operator a producer invented) has no geometric effect worth modelling.

    # -- graphics state helpers -------------------------------------------------------
    def _ext_gstate(self, operands: List[Any], state: ContentState, resources: Any) -> None:
        """Apply the parts of an ``/ExtGState`` that change geometry."""
        name = _last_name(operands)
        if name is None:
            return
        gs = self._resource(resources, "ExtGState", name)
        if not isinstance(gs, dict):
            return
        width = self._resolve(gs.get("LW"))
        if isinstance(width, (int, float)) and not isinstance(width, bool):
            state.stroke_width = float(width)
        font = self._resolve(gs.get("Font"))
        if isinstance(font, (list, tuple)) and len(font) >= 2:
            size = self._resolve(font[1])
            if isinstance(size, (int, float)) and not isinstance(size, bool):
                state.font_size = float(size)

    def _colour(self, op: str, operands: List[Any], state: ContentState, resources: Any) -> None:
        """Track enough colour to answer 'is this fill white?'."""
        stroking = op in ("G", "RG", "K", "CS", "SC", "SCN")
        if op in ("g", "G"):
            space, values = "gray", _tail_numbers(operands, 1)
        elif op in ("rg", "RG"):
            space, values = "rgb", _tail_numbers(operands, 3)
        elif op in ("k", "K"):
            space, values = "cmyk", _tail_numbers(operands, 4)
        elif op in ("cs", "CS"):
            name = _last_name(operands)
            space = self._colour_space(name, resources)
            if stroking:
                state.stroke_space = space
                state.stroke_is_pattern = space == "pattern"
                state.stroke_luminance = 0.0
            else:
                state.fill_space = space
                state.fill_is_pattern = space == "pattern"
                state.fill_luminance = 0.0
            return
        else:  # sc / scn / SC / SCN
            space = state.stroke_space if stroking else state.fill_space
            pattern = bool(operands) and isinstance(operands[-1], PdfName)
            if pattern:
                if stroking:
                    state.stroke_is_pattern = True
                    state.stroke_luminance = 0.0
                else:
                    state.fill_is_pattern = True
                    state.fill_luminance = 0.0
                return
            values = [v for v in operands if isinstance(v, (int, float)) and not isinstance(v, bool)]

        if values is None:
            return
        level = _luminance(space, values)
        if stroking:
            state.stroke_space = space
            state.stroke_is_pattern = False
            state.stroke_luminance = 0.0 if level is None else level
        else:
            state.fill_space = space
            state.fill_is_pattern = False
            state.fill_luminance = 0.0 if level is None else level

    def _colour_space(self, name: Optional[str], resources: Any) -> str:
        """Resolve a ``cs``/``CS`` operand onto one of the internal space kinds."""
        if not name:
            return "other"
        direct = _SPACE_ALIASES.get(name)
        if direct is not None:
            return direct
        entry = self._resource(resources, "ColorSpace", name)
        return self._space_of(entry, 0)

    def _space_of(self, entry: Any, depth: int) -> str:
        """Classify a resolved colour space object."""
        if depth > 4:
            return "other"
        entry = self._resolve(entry)
        if isinstance(entry, PdfName):
            return _SPACE_ALIASES.get(entry.value, "other")
        if isinstance(entry, (list, tuple)) and entry:
            head = self._resolve(entry[0])
            head_name = head.value if isinstance(head, PdfName) else ""
            if head_name == "ICCBased" and len(entry) > 1:
                stream = self._resolve(entry[1])
                count = None
                if isinstance(stream, PdfStream):
                    count = self._resolve(stream.dict.get("N"))
                if count == 1:
                    return "gray"
                if count == 4:
                    return "cmyk"
                return "rgb"
            if head_name in ("Separation", "DeviceN"):
                return "subtractive"
            if head_name in ("Indexed", "I"):
                return "indexed"
            if head_name == "Pattern":
                return "pattern"
            return _SPACE_ALIASES.get(head_name, "other")
        return "other"

    # -- path helpers -----------------------------------------------------------------
    def _curve(self, op: str, operands: List[Any], path: _PathBuilder, state: ContentState) -> None:
        """Append a cubic Bezier (``c``, ``v`` or ``y``) to the current subpath."""
        needed = 6 if op == "c" else 4
        values = _numbers(operands, needed)
        if values is None:
            return
        points = [state.ctm.apply_xy(values[i], values[i + 1]) for i in range(0, needed, 2)]
        start = path.subpaths[-1].current if path.subpaths else points[0]
        sub = path.ensure(start)
        p0 = sub.current
        if op == "c":
            p1, p2, p3 = points[0], points[1], points[2]
        elif op == "v":
            p1, p2, p3 = p0, points[0], points[1]
        else:  # y
            p1, p2, p3 = points[0], points[1], points[1]
        sub.points.extend(_flatten_bezier(p0, p1, p2, p3))
        sub.beziers += 1

    def _rectangle(self, values: List[float], path: _PathBuilder, state: ContentState) -> None:
        """Append a closed ``re`` subpath, transformed by the CTM."""
        x, y, width, height = values
        corners = (
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
            (x, y),
        )
        sub = _SubPath(state.ctm.apply_xy(*corners[0]))
        for corner in corners[1:]:
            sub.points.append(state.ctm.apply_xy(*corner))
        sub.segments = 4
        sub.closed = True
        # An ``re`` is only an axis-aligned rectangle when the CTM has no rotation/skew.
        sub.from_rect = _close(state.ctm.b, 0.0, 1e-9) and _close(state.ctm.c, 0.0, 1e-9)
        path.subpaths.append(sub)
        path.start = sub.points[0]

    def _paint(
        self, op: str, path: _PathBuilder, state: ContentState, result: ContentResult
    ) -> None:
        """Turn the accumulated path into primitives, if this operator paints.

        ``filled`` and ``stroked`` mirror the operator exactly, which is what gives the
        documented white-fill convention its meaning: a plain ``f`` produces
        ``filled=True, stroked=False``, so a white filled rectangle is unambiguously the
        blank-region signal rather than a drawn box.
        """
        if op == "n":
            # ``W n`` sets a clip and paints nothing; a bare ``n`` paints nothing either.
            return
        if op in _CLOSE_OPS:
            path.close()
        filled = op in _FILL_OPS
        stroked = op in _STROKE_OPS
        if not (filled or stroked):  # pragma: no cover - _PAINT_OPS has no other member
            return
        white = filled and state.fill_is_white()
        width = state.scaled_line_width() if stroked else 0.0
        for sub in path.subpaths:
            if len(sub.points) < 2:
                continue
            rect = sub.bbox()
            if rect.width <= 0.0 and rect.height <= 0.0:
                continue
            kind = self._classify(sub, rect)
            result.primitives.append(
                VectorPrimitive(
                    kind=kind,
                    rect=rect,
                    page=self.page_index,
                    stroke_width=width,
                    filled=filled,
                    stroked=stroked,
                    points=_sample_points(sub.points),
                )
            )
            if white:
                result.white_fills.append(rect)

    def _classify(self, sub: _SubPath, rect: Rect) -> str:
        """Classify one painted subpath as line / rect / circle / path."""
        width, height = rect.width, rect.height
        if sub.beziers == 4 and sub.segments <= 1 and width > 0.0 and height > 0.0:
            ratio = width / height if height > 0.0 else 0.0
            if 1.0 / _SQUARE_RATIO <= ratio <= _SQUARE_RATIO:
                return "circle"
        thin = width < self._thin or height < self._thin
        rectangular = sub.is_axis_aligned_rect()
        if thin and (sub.distinct_count() <= 2 or rectangular):
            return "line"
        if rectangular:
            return "rect"
        return "path"

    # -- text helpers -----------------------------------------------------------------
    def _select_font(self, operands: List[Any], state: ContentState) -> None:
        """Apply ``Tf``: resource name plus size."""
        size: Optional[float] = None
        name: Optional[str] = None
        for value in operands:
            if isinstance(value, PdfName):
                name = value.value
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                size = float(value)
        if name is not None:
            state.font = name
        if size is not None:
            state.font_size = size

    def _next_line(self, state: ContentState, tx: float, ty: float) -> None:
        """Apply ``Td``-style line movement to the text and line matrices."""
        state.line_matrix = Matrix.translation(tx, ty).concat(state.line_matrix)
        state.text_matrix = state.line_matrix

    def _show_array(
        self, items: Sequence[Any], state: ContentState, resources: Any, result: ContentResult
    ) -> None:
        """Apply ``TJ``: strings interleaved with thousandths-of-em kerning numbers."""
        for item in items:
            if isinstance(item, (bytes, bytearray)):
                self._show(bytes(item), state, resources, result)
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                shift = -float(item) / 1000.0 * state.font_size * state.horizontal_factor
                state.text_matrix = Matrix.translation(shift, 0.0).concat(state.text_matrix)

    def _show(
        self, raw: bytes, state: ContentState, resources: Any, result: ContentResult
    ) -> None:
        """Show one string, emitting a single :class:`TextSpan` and advancing the pen."""
        program = self._font_for(resources, state.font)
        size = state.font_size
        factor = state.horizontal_factor
        parts: List[str] = []
        glyph_rects: List[Rect] = []
        boxes: List[Rect] = []
        baseline: Optional[float] = None
        effective_size = 0.0

        # Trm = [Tfs*Th 0 0 Tfs 0 Ts] x Tm x CTM; only Tm changes between glyphs.
        parameters = Matrix(size * factor, 0.0, 0.0, size, 0.0, state.rise)
        for code, text in program.decode(raw):
            advance = program.width(code)
            trm = parameters.concat(state.text_matrix).concat(state.ctm)
            if baseline is None:
                baseline = trm.f
                effective_size = math.hypot(trm.b, trm.d)
            box = trm.transform_rect(Rect(0.0, program.descent, advance, program.ascent))
            boxes.append(box)
            if text:
                parts.append(text)
                glyph_rects.extend([box] * len(text))
            # tx = (w0 * Tfs + Tc + Tw) * Th, with Tw applying to code 32 only.
            step = advance * size + state.char_spacing
            if program.is_space(code):
                step += state.word_spacing
            state.text_matrix = Matrix.translation(step * factor, 0.0).concat(state.text_matrix)

        if not boxes:
            return
        rect = Rect.bounding(boxes)
        if rect is None:  # pragma: no cover - boxes is non-empty here
            return
        invisible = state.render_mode in INVISIBLE_RENDER_MODES
        result.spans.append(
            TextSpan(
                text="".join(parts),
                rect=rect,
                page=self.page_index,
                font_name=program.std_font,
                font_size=effective_size,
                source="native",
                confidence=0.0 if invisible else 1.0,
                glyph_rects=glyph_rects,
                baseline=baseline,
            )
        )

    # -- XObjects ---------------------------------------------------------------------
    def _do_xobject(
        self,
        name: str,
        state: ContentState,
        resources: Any,
        depth: int,
        visited: Set[int],
        result: ContentResult,
    ) -> None:
        """Execute ``Do``: recurse into a Form XObject or record an image's box."""
        xobject = self._resource(resources, "XObject", name)
        if not isinstance(xobject, PdfStream):
            return
        subtype = xobject.dict.get("Subtype")
        subtype_name = subtype.value if isinstance(subtype, PdfName) else ""
        if subtype_name == "Image":
            result.images.append(state.ctm.transform_rect(Rect(0.0, 0.0, 1.0, 1.0)))
            return
        if subtype_name and subtype_name != "Form":
            return
        if depth >= MAX_FORM_DEPTH:
            return
        key = id(xobject)
        if key in visited:
            return

        inner = state.copy()
        matrix = self._resolve(xobject.dict.get("Matrix"))
        values = _numbers(list(matrix), 6) if isinstance(matrix, (list, tuple)) else None
        if values is not None:
            inner.ctm = Matrix(*values).concat(state.ctm)
        inner.text_matrix = Matrix.identity()
        inner.line_matrix = Matrix.identity()

        own = self._resolve(xobject.dict.get("Resources"))
        child_resources = own if isinstance(own, dict) else resources
        try:
            data = xobject.decoded(self.resolver)
        except Exception:  # pragma: no cover - decoded() is lenient
            result.errors += 1
            return
        visited.add(key)
        try:
            self._execute(data, child_resources, inner, depth + 1, visited, result)
        finally:
            visited.discard(key)

    # -- inline images ----------------------------------------------------------------
    def _inline_image(self, lexer: Lexer, state: ContentState, result: ContentResult) -> None:
        """Consume ``BI ... ID <binary> EI`` and record the image's box.

        The payload is binary and may contain the bytes ``EI`` by chance, so the scan
        requires ``EI`` to be preceded by white space and followed by white space, a
        delimiter or end of stream.  When the image is unfiltered its exact length is
        computable from ``/W``, ``/H``, ``/BPC`` and ``/CS``, and that is used to jump
        straight past the data instead of guessing.
        """
        params = PdfDict()
        key: Optional[str] = None
        for _ in range(4096):
            token = lexer.next_token()
            if token.kind == "eof":
                return
            if token.kind == "keyword" and token.value == "ID":
                break
            if key is None:
                if token.kind == "name":
                    key = token.value
                continue
            if token.kind == "name":
                params[key] = PdfName(token.value)
            elif token.kind == "array_open":
                params[key] = self._read_array(lexer)
            elif token.kind == "dict_open":
                params[key] = self._read_dict(lexer)
            elif token.kind == "keyword":
                params[key] = {"true": True, "false": False}.get(token.value, token.value)
            else:
                params[key] = token.value
            key = None

        data = lexer.data
        pos = lexer.pos
        # Exactly one white-space byte separates ID from the data.
        if pos < len(data) and data[pos] in b"\x00\t\n\x0c\r ":
            pos += 1
        skip = _inline_payload_length(params)
        if skip is not None and pos + skip <= len(data):
            pos += skip
        end = _find_inline_end(data, pos)
        lexer.pos = end
        result.images.append(state.ctm.transform_rect(Rect(0.0, 0.0, 1.0, 1.0)))


# --------------------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------------------

_INLINE_CS_COMPONENTS = {
    "DeviceGray": 1, "G": 1, "CalGray": 1, "I": 1, "Indexed": 1,
    "DeviceRGB": 3, "RGB": 3, "CalRGB": 3,
    "DeviceCMYK": 4, "CMYK": 4,
}


#: Every field of :class:`ContentState`, so ``Q`` can never miss one that is added later.
_STATE_FIELDS = tuple(f.name for f in dataclass_fields(ContentState))


def _assign(target: ContentState, source: ContentState) -> None:
    """Copy every field of ``source`` onto ``target`` in place."""
    for name in _STATE_FIELDS:
        setattr(target, name, getattr(source, name))


def _numbers(operands: Sequence[Any], count: int) -> Optional[List[float]]:
    """Return the last ``count`` operands as floats, or ``None`` when they are not numbers."""
    if len(operands) < count:
        return None
    tail = operands[len(operands) - count :]
    out: List[float] = []
    for value in tail:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        out.append(float(value))
    return out


def _tail_numbers(operands: Sequence[Any], count: int) -> Optional[List[float]]:
    """Like :func:`_numbers` but tolerant: skips junk and takes the last numeric run."""
    numbers = [v for v in operands if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(numbers) < count:
        return None
    return [float(v) for v in numbers[len(numbers) - count :]]


def _last_bytes(operands: Sequence[Any]) -> Optional[bytes]:
    """Return the last string operand as bytes."""
    for value in reversed(operands):
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
    return None


def _last_name(operands: Sequence[Any]) -> Optional[str]:
    """Return the last name operand as a plain string."""
    for value in reversed(operands):
        if isinstance(value, PdfName):
            return value.value
    return None


def _sample_points(points: Sequence[Tuple[float, float]]) -> List[Point]:
    """Return at most :data:`_MAX_POINTS` of a subpath's vertices, ends included."""
    total = len(points)
    if total <= _MAX_POINTS:
        return [Point(x, y) for x, y in points]
    step = total / float(_MAX_POINTS - 1)
    out: List[Point] = []
    for index in range(_MAX_POINTS - 1):
        x, y = points[int(index * step)]
        out.append(Point(x, y))
    x, y = points[-1]
    out.append(Point(x, y))
    return out


def _inline_payload_length(params: PdfDict) -> Optional[int]:
    """Exact byte length of an *unfiltered* inline image payload, or ``None``."""
    if params.get("F") is not None or params.get("Filter") is not None:
        length = params.get("L", params.get("Length"))
        if isinstance(length, int) and not isinstance(length, bool) and length >= 0:
            return length
        return None
    width = params.get("W", params.get("Width"))
    height = params.get("H", params.get("Height"))
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    if isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0:
        return None
    if params.get("IM") is True or params.get("ImageMask") is True:
        bits, components = 1, 1
    else:
        depth = params.get("BPC", params.get("BitsPerComponent"))
        bits = depth if isinstance(depth, int) and not isinstance(depth, bool) else 8
        space = params.get("CS", params.get("ColorSpace"))
        space_name = space.value if isinstance(space, PdfName) else ""
        components = _INLINE_CS_COMPONENTS.get(space_name, 1)
    if bits <= 0 or bits > 16:
        return None
    row = (width * components * bits + 7) // 8
    return row * height


def _find_inline_end(data: bytes, start: int) -> int:
    """Return the offset just past the ``EI`` that ends an inline image."""
    total = len(data)
    pos = start
    while True:
        found = data.find(b"EI", pos)
        if found < 0:
            return total
        before_ok = found == 0 or data[found - 1] in b"\x00\t\n\x0c\r "
        after = found + 2
        after_ok = after >= total or data[after] in b"\x00\t\n\x0c\r ()<>[]{}/%"
        if before_ok and after_ok:
            return after
        pos = found + 2


def analyze_page(page: Any, resolver: Any = None, config: Optional[ZfpConfig] = None) -> ContentResult:
    """Interpret one page's content stream.

    A one-line convenience over :class:`ContentStreamInterpreter`.

    Args:
        page: The page to interpret.
        resolver: Reference resolver; defaults to the page's own document.
        config: Detection thresholds; ``ZfpConfig.default()`` when omitted.

    Returns:
        The page's :class:`ContentResult`.
    """
    return ContentStreamInterpreter(page, resolver, config).run()
