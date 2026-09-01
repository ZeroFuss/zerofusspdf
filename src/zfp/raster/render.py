"""Page rasterization.

ZFP needs a gray raster of a page for OCR and for raster shape detection.  Four backends
can produce one; they are tried in the order below, which is a *licensing* order as much
as a quality one (see ``docs/LICENSING.md``):

1. ``pypdfium2`` -- BSD-3-Clause/Apache-2.0 PDFium bindings.  The preferred renderer.
2. ``pdftoppm`` -- Poppler's CLI (GPL-2.0), invoked as a separate **process**, which is a
   distribution question rather than a linking one.
3. ``pymupdf(agpl)`` -- MuPDF is **AGPL-3.0 or commercial**, and AGPL's network clause
   reaches a hosted deployment.  It is never a declared dependency of ZFP and is only
   used when the operator opts in by setting the ``ZFP_ALLOW_AGPL_RENDERER`` environment
   variable to a truthy value, under their own licence terms.  The backend label keeps
   the ``(agpl)`` suffix so it can never be adopted by accident.
4. ``embedded`` -- no dependency at all: the page's own image XObjects are decoded by
   :mod:`zfp.raster.image` and composited onto a white canvas at the requested scale.
   This is *the* case that matters for scanned forms, which are one full-page image, and
   it is always available.

Everything here works on a bare CPython stdlib.  Pixel space never leaves this module
without going back through :class:`~zfp.core.geometry.PageGeometry`.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.errors import UnsupportedFeatureError, ValidationError
from ..core.geometry import EPS, Matrix, Rect
from ..core.logging import get_logger
from ..core.optional import optional_import
from ..pdfio.objects import PdfDict, PdfName, PdfStream
from .image import DecodedImage, decode_image_xobject

__all__ = [
    "RenderedPage",
    "BACKEND_PYPDFIUM2",
    "BACKEND_PDFTOPPM",
    "BACKEND_PYMUPDF",
    "BACKEND_EMBEDDED",
    "AGPL_ENV_VAR",
    "available_backends",
    "render_page",
    "render_available",
    "embedded_page_images",
    "parse_pgm",
]

_log = get_logger(__name__)

BACKEND_PYPDFIUM2 = "pypdfium2"
BACKEND_PDFTOPPM = "pdftoppm"
#: MuPDF is AGPL-3.0 or commercial; the label carries that so nobody adopts it silently.
BACKEND_PYMUPDF = "pymupdf(agpl)"
BACKEND_EMBEDDED = "embedded"

#: Set this to ``1``/``true``/``yes``/``on`` to allow the AGPL MuPDF backend.
AGPL_ENV_VAR = "ZFP_ALLOW_AGPL_RENDERER"

_WHITE = 255
_PDFTOPPM_TIMEOUT = 120


# ======================================================================================
# The raster itself
# ======================================================================================


@dataclass(frozen=True)
class RenderedPage:
    """A rasterized page: row-major 8-bit gray, top-left origin, y down.

    Attributes:
        page: Zero-based page index.
        width: Raster width in pixels.
        height: Raster height in pixels.
        scale: Pixels per PDF point (``dpi / 72``).  Feed this to
            :meth:`~zfp.core.geometry.PageGeometry.pixel_rect_to_user` to get back to
            user space.
        gray: ``width * height`` bytes, 0 = black, 255 = white.
        backend: Which renderer produced it.
    """

    page: int
    width: int
    height: int
    scale: float
    gray: bytes
    backend: str

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0:
            raise ValidationError("RenderedPage size cannot be negative")
        if len(self.gray) != self.width * self.height:
            raise ValidationError(
                "RenderedPage gray is %d bytes but %dx%d needs %d"
                % (len(self.gray), self.width, self.height, self.width * self.height)
            )

    # -- access -----------------------------------------------------------------------
    def pixel(self, x: int, y: int) -> int:
        """Return the gray level at ``(x, y)``; coordinates are clamped to the raster."""
        if self.width <= 0 or self.height <= 0:
            return _WHITE
        cx = 0 if x < 0 else (self.width - 1 if x >= self.width else int(x))
        cy = 0 if y < 0 else (self.height - 1 if y >= self.height else int(y))
        return self.gray[cy * self.width + cx]

    def row(self, y: int) -> bytes:
        """Return row ``y`` as ``width`` bytes; ``y`` is clamped to the raster."""
        if self.width <= 0 or self.height <= 0:
            return b""
        cy = 0 if y < 0 else (self.height - 1 if y >= self.height else int(y))
        return self.gray[cy * self.width : (cy + 1) * self.width]

    def crop(self, rect_px: Rect) -> RenderedPage:
        """Return the sub-raster covered by ``rect_px`` (a rectangle in *pixel* space).

        The rectangle is clamped to the raster.  A rectangle that misses the page
        entirely yields a 1x1 white raster rather than an unusable empty one.
        """
        rect = rect_px.normalized()
        x0 = max(0, int(math.floor(rect.x0)))
        y0 = max(0, int(math.floor(rect.y0)))
        x1 = min(self.width, int(math.ceil(rect.x1)))
        y1 = min(self.height, int(math.ceil(rect.y1)))
        if x1 <= x0 or y1 <= y0:
            return replace(self, width=1, height=1, gray=b"\xff")
        width = x1 - x0
        out = bytearray(width * (y1 - y0))
        for i, y in enumerate(range(y0, y1)):
            base = y * self.width
            out[i * width : (i + 1) * width] = self.gray[base + x0 : base + x1]
        return replace(self, width=width, height=y1 - y0, gray=bytes(out))

    # -- interchange ------------------------------------------------------------------
    def to_pgm(self) -> bytes:
        """Serialize as a binary PGM (``P5``), the debug format the tests round-trip."""
        header = ("P5\n%d %d\n255\n" % (self.width, self.height)).encode("ascii")
        return header + self.gray

    @staticmethod
    def from_pgm(
        data: bytes, page: int = 0, scale: float = 1.0, backend: str = BACKEND_EMBEDDED
    ) -> RenderedPage:
        """Build a :class:`RenderedPage` from binary PGM bytes."""
        width, height, gray = parse_pgm(data)
        return RenderedPage(
            page=page, width=width, height=height, scale=scale, gray=gray, backend=backend
        )

    def __repr__(self) -> str:
        return "RenderedPage(page=%d, %dx%d, scale=%.4f, backend=%s)" % (
            self.page, self.width, self.height, self.scale, self.backend,
        )


def parse_pgm(data: bytes) -> Tuple[int, int, bytes]:
    """Parse a binary ``P5`` PGM into ``(width, height, gray)``.

    Only maxval 255 is accepted, which is what every renderer ZFP shells out to emits.

    Raises:
        ValidationError: The bytes are not a usable 8-bit binary PGM.
    """
    if not data.startswith(b"P5"):
        raise ValidationError("not a binary PGM (missing P5 magic)")
    pos = 2
    fields: List[int] = []
    while len(fields) < 3:
        while pos < len(data) and data[pos : pos + 1].isspace():
            pos += 1
        if pos < len(data) and data[pos : pos + 1] == b"#":
            while pos < len(data) and data[pos : pos + 1] not in (b"\n", b"\r"):
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos : pos + 1].isspace():
            pos += 1
        token = data[start:pos]
        if not token.isdigit():
            raise ValidationError("malformed PGM header near byte %d" % start)
        fields.append(int(token))
    if data[pos : pos + 1].isspace():
        pos += 1
    width, height, maxval = fields
    if maxval != 255:
        raise ValidationError("only 8-bit PGM (maxval 255) is supported, got %d" % maxval)
    gray = data[pos : pos + width * height]
    if len(gray) < width * height:
        gray = gray + b"\xff" * (width * height - len(gray))
    return (width, height, gray)


# ======================================================================================
# Backend discovery
# ======================================================================================


def _truthy(value: Optional[str]) -> bool:
    """True for ``1``, ``true``, ``yes``, ``on`` (case-insensitive)."""
    if not value:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _agpl_opt_in() -> bool:
    """True when the operator explicitly allowed the AGPL MuPDF backend."""
    return _truthy(os.environ.get(AGPL_ENV_VAR))


def _pymupdf_module() -> Any:
    """Return the imported PyMuPDF module (either spelling), or ``None``."""
    for name in ("pymupdf", "fitz"):
        module = optional_import(name)
        if module and hasattr(module.module, "open"):
            return module.module
    return None


def available_backends() -> List[str]:
    """Return the usable rendering backends, best first.

    ``"embedded"`` is always last and always present: it needs nothing but the standard
    library.  ``"pymupdf(agpl)"`` only appears when :data:`AGPL_ENV_VAR` is truthy,
    because MuPDF is AGPL-3.0 or commercial and must never be adopted by accident.
    """
    out: List[str] = []
    if optional_import("pypdfium2"):
        out.append(BACKEND_PYPDFIUM2)
    if shutil.which("pdftoppm"):
        out.append(BACKEND_PDFTOPPM)
    if _agpl_opt_in() and _pymupdf_module() is not None:
        out.append(BACKEND_PYMUPDF)
    out.append(BACKEND_EMBEDDED)
    return out


def _document_bytes(doc: Any) -> bytes:
    """Return the document's current bytes, including any staged edits."""
    writer = getattr(doc, "writer", None)
    try:
        if writer is not None and bool(getattr(writer, "has_changes", False)):
            return doc.to_bytes()
    except Exception:  # pragma: no cover - a broken writer must not stop rendering
        _log.debug("could not serialize staged changes; rendering the original bytes")
    data = getattr(doc, "source_bytes", b"") or b""
    if not data:
        data = doc.to_bytes()
    return data


# ======================================================================================
# Embedded images: placement, decoding, compositing
# ======================================================================================

#: Bytes that end an unquoted content-stream token.
_TOKEN_END = b"\x00\t\n\x0c\r ()<>[]{}/%"

#: Refuse to allocate a canvas bigger than this (120 megapixels).
_MAX_CANVAS = 120_000_000


def _skip_inline_image(data: bytes, pos: int) -> int:
    """Return the offset just past the ``EI`` that closes an inline image."""
    marker = data.find(b"ID", pos)
    if marker < 0:
        return len(data)
    i = marker + 3  # ID plus the single whitespace byte that follows it
    n = len(data)
    while i < n:
        j = data.find(b"EI", i)
        if j < 0:
            return n
        before_ok = j == 0 or data[j - 1] in _TOKEN_END
        after_ok = j + 2 >= n or data[j + 2] in _TOKEN_END
        if before_ok and after_ok:
            return j + 2
        i = j + 2
    return n


def _content_ops(data: bytes, limit: int = 400000):
    """Yield ``(operator, operands)`` for a content stream.

    A deliberately small scanner: it understands enough of the token grammar (numbers,
    names, strings, dictionaries, inline images) to keep operand positions correct, and
    it does not care what any operator except ``q Q cm Do`` means.
    """
    operands: List[Any] = []
    i = 0
    n = len(data)
    emitted = 0
    while i < n and emitted < limit:
        c = data[i]
        if c in b"\x00\t\n\x0c\r ":
            i += 1
            continue
        if c == 0x25:  # % comment
            while i < n and data[i] not in b"\r\n":
                i += 1
            continue
        if c == 0x2F:  # /Name
            j = i + 1
            while j < n and data[j] not in _TOKEN_END:
                j += 1
            operands.append(PdfName.decode(data[i:j]))
            i = j
            continue
        if c == 0x28:  # ( literal string
            depth = 1
            j = i + 1
            while j < n and depth:
                ch = data[j]
                if ch == 0x5C:
                    j += 2
                    continue
                if ch == 0x28:
                    depth += 1
                elif ch == 0x29:
                    depth -= 1
                j += 1
            operands.append(None)
            i = j
            continue
        if c == 0x3C:  # << dictionary or <hex string>
            if data[i : i + 2] == b"<<":
                depth = 1
                j = i + 2
                while j < n and depth:
                    if data[j : j + 2] == b"<<":
                        depth += 1
                        j += 2
                        continue
                    if data[j : j + 2] == b">>":
                        depth -= 1
                        j += 2
                        continue
                    j += 1
                operands.append(None)
                i = j
                continue
            j = data.find(b">", i)
            i = n if j < 0 else j + 1
            operands.append(None)
            continue
        if c in b"[]{}>)":
            i += 1
            continue
        j = i
        while j < n and data[j] not in _TOKEN_END:
            j += 1
        token = data[i:j]
        i = j if j > i else i + 1
        if not token:
            continue
        if token[0:1].isdigit() or token[0:1] in (b"+", b"-", b"."):
            try:
                operands.append(float(token))
            except ValueError:
                operands.append(None)
            continue
        operator = token.decode("latin-1")
        yield (operator, operands)
        emitted += 1
        if operator == "BI":
            i = _skip_inline_image(data, i)
        operands = []


def _scan_image_placements(content: bytes, names: Sequence[str]) -> List[Tuple[str, Rect]]:
    """Return ``(xobject name, placement rect)`` for every image painted by ``Do``.

    Tracks only ``q``, ``Q`` and ``cm``, which is all the common
    ``q W 0 0 H 0 0 cm /Im0 Do Q`` scan shape needs.  Form XObjects are *not* recursed;
    an image used inside a form gets no placement here and falls back to the whole page
    when it is the page's only image.
    """
    wanted = set(names)
    if not wanted:
        return []
    unit = Rect(0.0, 0.0, 1.0, 1.0)
    ctm = Matrix.identity()
    stack: List[Matrix] = []
    out: List[Tuple[str, Rect]] = []
    for operator, operands in _content_ops(content):
        if operator == "q":
            stack.append(ctm)
        elif operator == "Q":
            ctm = stack.pop() if stack else Matrix.identity()
        elif operator == "cm":
            numbers = [v for v in operands[-6:] if isinstance(v, float)]
            if len(numbers) == 6:
                ctm = Matrix(*numbers).concat(ctm)
        elif operator == "Do":
            name = operands[-1] if operands else None
            if isinstance(name, PdfName) and name.value in wanted:
                out.append((name.value, ctm.transform_rect(unit)))
    return out


def _page_image_xobjects(page: Any, resolver: Any) -> Dict[str, PdfStream]:
    """Return the page's image XObjects keyed by resource name, in name order."""
    resources = page.resources()
    xobjects = resources.resolved_get("XObject", None, resolver)
    out: Dict[str, PdfStream] = {}
    if not isinstance(xobjects, PdfDict):
        return out
    for key in sorted(xobjects.keys()):
        value = resolver.resolve(xobjects.get(key)) if resolver is not None else xobjects.get(key)
        if isinstance(value, PdfStream):
            subtype = value.dict.get_name("Subtype", None, resolver)
            if subtype == "Image":
                out[str(key)] = value
    return out


def _placements(doc: Any, index: int) -> List[Tuple[str, Rect, PdfStream]]:
    """Return ``(name, user-space rect, stream)`` for every placeable page image."""
    page = doc.page(index)
    images = _page_image_xobjects(page, doc)
    if not images:
        return []
    out: List[Tuple[str, Rect, PdfStream]] = []
    for name, rect in _scan_image_placements(page.content_bytes(), list(images)):
        normalized = rect.normalized()
        if normalized.width <= EPS or normalized.height <= EPS:
            continue
        out.append((name, normalized, images[name]))
    if not out and len(images) == 1:
        # The overwhelmingly common scanned page: one image, placed over the whole page
        # (possibly from inside a form XObject the minimal scanner does not enter).
        name = next(iter(images))
        out.append((name, page.geometry.crop_box, images[name]))
    return out


def _image_codec(stream: PdfStream, resolver: Any) -> str:
    """Return the image codec still applied to :meth:`PdfStream.decoded` bytes, or ``""``."""
    from ..pdfio.filters import IMAGE_FILTERS, normalize_filter_name

    for name in stream.filter_names(resolver):
        canonical = normalize_filter_name(name)
        if canonical in IMAGE_FILTERS:
            return canonical
    return ""


def embedded_page_images(doc: Any, index: int) -> List[Tuple[Rect, bytes, str]]:
    """Return every image painted on a page, without needing any renderer.

    Args:
        doc: The :class:`~zfp.pdfio.document.Document`.
        index: Zero-based page index.

    Returns:
        A list of ``(rect, data, filter_name)``.  ``rect`` is the image's placement in
        **PDF user space**.  ``data`` is the stream with its non-image filters already
        applied, so it is the JPEG/CCITT payload when ``filter_name`` names an image
        codec and raw samples when ``filter_name`` is ``""``.
    """
    out: List[Tuple[Rect, bytes, str]] = []
    for _name, rect, stream in _placements(doc, index):
        try:
            data = stream.decoded(doc)
        except Exception:  # pragma: no cover - filters are lenient; this is a backstop
            data = stream.raw
        out.append((rect, data, _image_codec(stream, doc)))
    return out


def _box_shrink(gray: bytes, width: int, height: int, factor: int) -> Tuple[bytes, int, int]:
    """Average ``factor x factor`` blocks together; returns ``(gray, width, height)``."""
    new_w = width // factor
    new_h = height // factor
    if factor < 2 or new_w < 1 or new_h < 1:
        return (gray, width, height)
    span = new_w * factor
    divisor = factor * factor
    out = bytearray(new_w * new_h)
    for y in range(new_h):
        acc = [0] * new_w
        for k in range(factor):
            base = (y * factor + k) * width
            row = gray[base : base + span]
            if len(row) < span:
                row = row + b"\xff" * (span - len(row))
            parts = [row[j::factor] for j in range(factor)]
            for i, total in enumerate(map(sum, zip(*parts))):
                acc[i] += total
        out[y * new_w : (y + 1) * new_w] = bytes(v // divisor for v in acc)
    return (bytes(out), new_w, new_h)


def _blit(
    canvas: bytearray,
    canvas_w: int,
    canvas_h: int,
    image: DecodedImage,
    rect: Rect,
    geometry: Any,
    scale: float,
) -> bool:
    """Composite one decoded image onto the canvas at its user-space placement.

    The whole pixel -> image-sample mapping is a single affine transform, so the common
    axis-aligned case reduces to one precomputed column table plus a row slice, and the
    rotated case to one index table -- no per-pixel matrix arithmetic.
    """
    rect = rect.normalized()
    if rect.width <= EPS or rect.height <= EPS or image.width <= 0 or image.height <= 0:
        return False
    dest = geometry.user_rect_to_pixel(rect, scale)
    x0 = max(0, int(math.floor(dest.x0)))
    y0 = max(0, int(math.floor(dest.y0)))
    x1 = min(canvas_w, int(math.ceil(dest.x1)))
    y1 = min(canvas_h, int(math.ceil(dest.y1)))
    if x1 <= x0 or y1 <= y0:
        return False

    source = image.gray
    src_w, src_h = image.width, image.height
    units_w, units_h = float(src_w), float(src_h)
    # Measure the shrink against the *unclipped* placement: an image running off the
    # page edge must not be blurred just because less of it is visible.
    factor = int(
        min(src_w / max(1.0, dest.width), src_h / max(1.0, dest.height))
    )
    if factor >= 2:
        # Downsampling: average first so a 300 dpi scan drawn at 150 dpi keeps its ink.
        source, src_w, src_h = _box_shrink(source, src_w, src_h, factor)
        units_w /= factor
        units_h /= factor

    to_image = Matrix(
        units_w / rect.width,
        0.0,
        0.0,
        -units_h / rect.height,
        -rect.x0 * units_w / rect.width,
        rect.y1 * units_h / rect.height,
    )
    m = geometry.render_matrix(scale).inverted().concat(to_image)
    mask = image.kind == "mask"
    max_x, max_y = src_w - 1, src_h - 1
    span = x1 - x0

    if abs(m.b) < 1e-9 and abs(m.c) < 1e-9:
        columns = []
        for px in range(x0, x1):
            value = int(m.a * (px + 0.5) + m.e)
            columns.append(0 if value < 0 else (max_x if value > max_x else value))
        for py in range(y0, y1):
            value = int(m.d * (py + 0.5) + m.f)
            row_index = 0 if value < 0 else (max_y if value > max_y else value)
            row = source[row_index * src_w : (row_index + 1) * src_w]
            values = bytes(map(row.__getitem__, columns))
            base = py * canvas_w + x0
            if mask:
                values = bytes(map(min, canvas[base : base + span], values))
            canvas[base : base + span] = values
        return True

    if abs(m.a) < 1e-9 and abs(m.d) < 1e-9:
        row_offsets = []
        for px in range(x0, x1):
            value = int(m.b * (px + 0.5) + m.f)
            row_offsets.append((0 if value < 0 else (max_y if value > max_y else value)) * src_w)
        for py in range(y0, y1):
            value = int(m.c * (py + 0.5) + m.e)
            column = 0 if value < 0 else (max_x if value > max_x else value)
            values = bytes(map(source.__getitem__, [r + column for r in row_offsets]))
            base = py * canvas_w + x0
            if mask:
                values = bytes(map(min, canvas[base : base + span], values))
            canvas[base : base + span] = values
        return True

    for py in range(y0, y1):
        yc = py + 0.5
        col_base = m.c * yc + m.e
        row_base = m.d * yc + m.f
        indices = []
        for px in range(x0, x1):
            xc = px + 0.5
            cx = int(m.a * xc + col_base)
            ry = int(m.b * xc + row_base)
            cx = 0 if cx < 0 else (max_x if cx > max_x else cx)
            ry = 0 if ry < 0 else (max_y if ry > max_y else ry)
            indices.append(ry * src_w + cx)
        values = bytes(map(source.__getitem__, indices))
        base = py * canvas_w + x0
        if mask:
            values = bytes(map(min, canvas[base : base + span], values))
        canvas[base : base + span] = values
    return True


def _render_embedded(doc: Any, index: int, dpi: float, scale: float) -> Optional[RenderedPage]:
    """Composite the page's own image XObjects onto a white canvas."""
    page = doc.page(index)
    geometry = page.geometry
    width, height = geometry.pixel_size(scale)
    if width * height > _MAX_CANVAS:
        raise ValidationError(
            "a %dx%d raster is too large; lower the dpi (%.0f)" % (width, height, dpi)
        )
    placements = _placements(doc, index)
    if not placements:
        return None
    canvas = bytearray(b"\xff" * (width * height))
    painted = 0
    for name, rect, stream in placements:
        image = decode_image_xobject(stream, doc)
        if image.width <= 0 or image.height <= 0:
            continue
        if _blit(canvas, width, height, image, rect, geometry, scale):
            painted += 1
            if not image.supported:
                _log.debug("page %d: image %s: %s", index, name, image.detail)
    if not painted:
        return None
    return RenderedPage(
        page=index,
        width=width,
        height=height,
        scale=scale,
        gray=bytes(canvas),
        backend=BACKEND_EMBEDDED,
    )


# ======================================================================================
# Optional external renderers
# ======================================================================================


def _bgr_rows_to_gray(buffer: bytes, width: int, height: int, stride: int, channels: int) -> bytes:
    """Collapse an interleaved BGR/BGRA (or gray) bitmap into 8-bit luminance."""
    out = bytearray(width * height)
    for y in range(height):
        base = y * stride
        row = buffer[base : base + width * channels]
        if len(row) < width * channels:
            row = row + b"\xff" * (width * channels - len(row))
        if channels == 1:
            out[y * width : (y + 1) * width] = row
            continue
        blue = row[0::channels]
        green = row[1::channels]
        red = row[2::channels]
        out[y * width : (y + 1) * width] = bytes(
            (19595 * r + 38470 * g + 7471 * b + 32768) >> 16
            for r, g, b in zip(red, green, blue)
        )
    return bytes(out)


def _render_pypdfium2(doc: Any, index: int, dpi: float, scale: float) -> Optional[RenderedPage]:
    """Render with PDFium (BSD-3-Clause/Apache-2.0), the preferred backend."""
    pdfium = optional_import("pypdfium2").module
    if pdfium is None:
        return None
    pdf = pdfium.PdfDocument(_document_bytes(doc))
    try:
        page = pdf[index]
        try:
            bitmap = page.render(scale=scale, grayscale=True, draw_annots=True)
        except TypeError:  # pragma: no cover - older signature
            bitmap = page.render(scale=scale, grayscale=True)
        width = int(bitmap.width)
        height = int(bitmap.height)
        buffer = bytes(bitmap.buffer)
        stride = int(getattr(bitmap, "stride", 0) or width)
        channels = int(getattr(bitmap, "n_channels", 0) or max(1, stride // max(1, width)))
        gray = _bgr_rows_to_gray(buffer, width, height, stride, channels)
    finally:
        closer = getattr(pdf, "close", None)
        if callable(closer):  # pragma: no cover - version dependent
            try:
                closer()
            except Exception:
                pass
    return RenderedPage(
        page=index, width=width, height=height, scale=scale, gray=gray,
        backend=BACKEND_PYPDFIUM2,
    )


def _render_pymupdf(doc: Any, index: int, dpi: float, scale: float) -> Optional[RenderedPage]:
    """Render with MuPDF.

    MuPDF/PyMuPDF is **AGPL-3.0 or commercial**.  It is not a declared dependency of ZFP
    and is only reached when the operator opts in through :data:`AGPL_ENV_VAR` or names
    the backend explicitly, under their own licence terms.
    """
    module = _pymupdf_module()
    if module is None:
        return None
    handle = module.open(stream=_document_bytes(doc), filetype="pdf")
    try:
        page = handle.load_page(index)
        matrix = module.Matrix(scale, scale)
        try:
            pix = page.get_pixmap(matrix=matrix, colorspace=module.csGRAY, alpha=False)
        except AttributeError:  # pragma: no cover - PyMuPDF < 1.19
            pix = page.getPixmap(matrix=matrix, colorspace=module.csGRAY, alpha=False)
        width, height = int(pix.width), int(pix.height)
        stride = int(getattr(pix, "stride", 0) or width * int(getattr(pix, "n", 1)))
        channels = int(getattr(pix, "n", 1) or 1)
        gray = _bgr_rows_to_gray(bytes(pix.samples), width, height, stride, channels)
    finally:
        closer = getattr(handle, "close", None)
        if callable(closer):  # pragma: no cover - version dependent
            try:
                closer()
            except Exception:
                pass
    return RenderedPage(
        page=index, width=width, height=height, scale=scale, gray=gray,
        backend=BACKEND_PYMUPDF,
    )


def _render_pdftoppm(doc: Any, index: int, dpi: float, scale: float) -> Optional[RenderedPage]:
    """Render by shelling out to Poppler's ``pdftoppm`` (a process boundary, by design)."""
    executable = shutil.which("pdftoppm")
    if not executable:
        return None
    data = _document_bytes(doc)
    page_number = str(index + 1)
    with tempfile.TemporaryDirectory(prefix="zfp-render-") as folder:
        source = os.path.join(folder, "input.pdf")
        with open(source, "wb") as handle:
            handle.write(data)
        command = [
            executable, "-gray", "-r", "%d" % int(round(dpi)),
            "-f", page_number, "-l", page_number, "-singlefile", source, "-",
        ]
        completed = subprocess.run(
            command, capture_output=True, timeout=_PDFTOPPM_TIMEOUT
        )
    if completed.returncode != 0 or not completed.stdout.startswith(b"P5"):
        _log.debug(
            "pdftoppm failed (rc=%s): %s",
            completed.returncode,
            completed.stderr[:200].decode("utf-8", "replace"),
        )
        return None
    width, height, gray = parse_pgm(completed.stdout)
    return RenderedPage(
        page=index, width=width, height=height, scale=scale, gray=gray,
        backend=BACKEND_PDFTOPPM,
    )


_BACKEND_FUNCTIONS = {
    BACKEND_PYPDFIUM2: _render_pypdfium2,
    BACKEND_PDFTOPPM: _render_pdftoppm,
    BACKEND_PYMUPDF: _render_pymupdf,
    BACKEND_EMBEDDED: _render_embedded,
}

#: Spellings accepted for the ``backend`` argument of :func:`render_page`.
_BACKEND_ALIASES = {
    "pymupdf": BACKEND_PYMUPDF,
    "fitz": BACKEND_PYMUPDF,
    "mupdf": BACKEND_PYMUPDF,
    "pdfium": BACKEND_PYPDFIUM2,
    "poppler": BACKEND_PDFTOPPM,
}


def render_page(
    doc: Any, index: int, dpi: int = 300, backend: Optional[str] = None
) -> RenderedPage:
    """Rasterize one page to 8-bit gray.

    Args:
        doc: The :class:`~zfp.pdfio.document.Document` to render.
        index: Zero-based page index.
        dpi: Target resolution; ``scale`` on the result is ``dpi / 72``.
        backend: Force one backend (see :func:`available_backends`).  ``None`` tries them
            in preference order and returns the first that produces a raster.

    Returns:
        The :class:`RenderedPage`, tagged with the backend that produced it.

    Raises:
        ValidationError: ``index`` or ``dpi`` is out of range.
        UnsupportedFeatureError: No backend could rasterize this page -- which, since the
            dependency-free ``embedded`` backend is always tried, means the page has no
            decodable embedded image either.  Callers can still use
            :func:`embedded_page_images`.
    """
    count = int(doc.page_count)
    if index < 0 or index >= count:
        raise ValidationError("page index %d out of range (document has %d)" % (index, count))
    try:
        dpi_value = float(dpi)
    except (TypeError, ValueError) as exc:
        raise ValidationError("dpi must be a number, got %r" % (dpi,)) from exc
    if dpi_value <= 0.0:
        raise ValidationError("dpi must be positive, got %r" % (dpi,))
    scale = dpi_value / 72.0

    if backend is not None:
        name = _BACKEND_ALIASES.get(str(backend).strip().lower(), str(backend).strip())
        if name not in _BACKEND_FUNCTIONS:
            raise UnsupportedFeatureError(
                "unknown render backend %r; available: %s"
                % (backend, ", ".join(sorted(_BACKEND_FUNCTIONS)))
            )
        order = [name]
    else:
        order = available_backends()

    failures: List[str] = []
    for name in order:
        function = _BACKEND_FUNCTIONS[name]
        try:
            result = function(doc, index, dpi_value, scale)
        except ValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken backend must not be fatal
            failures.append("%s: %s" % (name, exc))
            _log.debug("render backend %s failed on page %d: %s", name, index, exc)
            continue
        if result is not None:
            return result
        failures.append("%s: produced nothing" % name)

    raise UnsupportedFeatureError(
        "no rendering backend could rasterize page %d (%s). Install one with "
        "pip install 'zerofusspdf[render]' (pypdfium2) or Poppler's pdftoppm binary; "
        "PyMuPDF is AGPL/commercial and must be enabled with %s=1."
        % (index, "; ".join(failures) or "no backends", AGPL_ENV_VAR)
    )


def render_available(doc: Any, index: int) -> bool:
    """True when :func:`render_page` can produce a raster for this page.

    A real renderer covers every page; with only the dependency-free ``embedded``
    backend, it depends on the page actually carrying an image.
    """
    try:
        count = int(doc.page_count)
    except Exception:
        return False
    if index < 0 or index >= count:
        return False
    for name in available_backends():
        if name != BACKEND_EMBEDDED:
            return True
    try:
        return bool(_placements(doc, index))
    except Exception:  # pragma: no cover - a damaged page is simply not renderable
        return False
