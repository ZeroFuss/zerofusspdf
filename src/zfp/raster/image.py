"""Decoding embedded image XObjects to 8-bit grayscale, on the standard library alone.

A scanned form is, physically, one big image XObject on an otherwise empty page.  Being
able to read that image *without* Pillow, numpy or a PDF renderer is what makes the whole
raster path of ZFP work in a default install, so everything here is pure CPython.

What is supported
-----------------
* Raw samples (unfiltered, ``FlateDecode``, ``LZWDecode``, ``RunLengthDecode``,
  ``ASCIIHexDecode``, ``ASCII85Decode`` -- the whole non-image half of the filter chain is
  already applied by :meth:`zfp.pdfio.objects.PdfStream.decoded`) at
  ``/BitsPerComponent`` 1, 2, 4, 8 and 16, in ``/DeviceGray``, ``/DeviceRGB``,
  ``/DeviceCMYK``, ``/Indexed`` (palette lookup) and ``/ImageMask`` (honouring ``/Decode``).
* ``CCITTFaxDecode``: Group 4 (``K < 0``, two-dimensional), Group 3 one-dimensional
  (``K == 0``) and Group 3 mixed (``K > 0``), honouring ``/Columns``, ``/Rows``,
  ``/BlackIs1`` and ``/EncodedByteAlign``.
* ``DCTDecode``: baseline sequential JPEG (Huffman, 8-bit), interleaved or
  non-interleaved scans, restart intervals, 1/3/4 components.

What is not
-----------
Progressive and arithmetic-coded JPEG, ``JPXDecode`` and ``JBIG2Decode`` return a
:class:`DecodedImage` with ``kind == "unsupported"``, ``supported == False`` and a flat
canvas of the right size, so a caller can composite something sane and report honestly
instead of crashing.

Colour to gray uses the luminance ``0.299 R + 0.587 G + 0.114 B`` (fixed point, so the
result is deterministic across platforms) and CMYK via ``255 * (1 - c) * (1 - k)`` per
channel followed by the same luminance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.errors import UnsupportedFeatureError
from ..core.logging import get_logger
from ..pdfio.filters import IMAGE_FILTERS, normalize_filter_name
from ..pdfio.objects import PdfArray, PdfDict, PdfName, PdfStream, PdfString, resolve_with

__all__ = [
    "DecodedImage",
    "IMAGE_KINDS",
    "decode_image_xobject",
    "ccitt_decode",
    "decode_jpeg_gray",
    "JpegUnsupported",
]

_log = get_logger(__name__)

#: Every value :attr:`DecodedImage.kind` can take.
IMAGE_KINDS = frozenset(
    {"gray", "rgb", "cmyk", "indexed", "mask", "ccitt", "jpeg", "empty", "unsupported"}
)

_WHITE = 255
_MID_GRAY = 128

# Fixed-point luminance weights: 19595 + 38470 + 7471 == 65536 exactly, so pure white
# maps to exactly 255 and the result never depends on floating point rounding.
_LUMA_R = 19595
_LUMA_G = 38470
_LUMA_B = 7471


@dataclass(frozen=True)
class DecodedImage:
    """An image XObject decoded to row-major 8-bit grayscale.

    Attributes:
        width: Sample columns.
        height: Sample rows.
        gray: ``width * height`` bytes, row-major, 0 = black, 255 = white.
        kind: How the samples were interpreted -- one of :data:`IMAGE_KINDS`.
        supported: False when the codec could not be decoded and ``gray`` is a
            placeholder canvas rather than real image data.
        detail: Human-readable note (the codec name, the reason it was refused).
    """

    width: int
    height: int
    gray: bytes
    kind: str
    supported: bool = True
    detail: str = ""

    @property
    def size(self) -> Tuple[int, int]:
        """``(width, height)`` in samples."""
        return (self.width, self.height)

    def pixel(self, x: int, y: int) -> int:
        """Return the gray level at ``(x, y)``, clamped to the image bounds."""
        if self.width <= 0 or self.height <= 0:
            return _WHITE
        cx = 0 if x < 0 else (self.width - 1 if x >= self.width else int(x))
        cy = 0 if y < 0 else (self.height - 1 if y >= self.height else int(y))
        return self.gray[cy * self.width + cx]

    def row(self, y: int) -> bytes:
        """Return row ``y`` as ``width`` bytes, clamped to the image bounds."""
        if self.width <= 0 or self.height <= 0:
            return b""
        cy = 0 if y < 0 else (self.height - 1 if y >= self.height else int(y))
        return self.gray[cy * self.width : (cy + 1) * self.width]


# ======================================================================================
# Dictionary access helpers (image XObjects and inline images share these keys)
# ======================================================================================


def _first(d: Any, resolver: Any, *keys: str) -> Any:
    """Return the first present value among ``keys`` (long form then abbreviation)."""
    if not isinstance(d, PdfDict):
        return None
    for key in keys:
        value = d.resolved_get(key, None, resolver)
        if value is not None:
            return value
    return None


def _as_int(value: Any, default: int) -> int:
    """Coerce a PDF number to ``int``, falling back to ``default``."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _as_bool(value: Any, default: bool) -> bool:
    """Coerce a PDF boolean, falling back to ``default``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _name_of(value: Any) -> str:
    """Return a plain name string for a :class:`PdfName`/str/bytes value."""
    if isinstance(value, PdfName):
        return value.value
    if isinstance(value, str):
        return value[1:] if value.startswith("/") else value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("latin-1").lstrip("/")
    return ""


def _number_list(value: Any, resolver: Any) -> List[float]:
    """Return a list of floats from a PDF array, ignoring anything non-numeric."""
    value = resolve_with(value, resolver)
    if not isinstance(value, (PdfArray, list, tuple)):
        return []
    out: List[float] = []
    for item in value:
        item = resolve_with(item, resolver)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return []
        out.append(float(item))
    return out


# ======================================================================================
# Colour spaces
# ======================================================================================

_FAMILY_COMPONENTS = {
    "DeviceGray": 1,
    "CalGray": 1,
    "G": 1,
    "DeviceRGB": 3,
    "CalRGB": 3,
    "RGB": 3,
    "Lab": 3,
    "DeviceCMYK": 4,
    "CMYK": 4,
}


@dataclass(frozen=True)
class _ColorSpace:
    """The little that a grayscale converter needs to know about a colour space."""

    family: str
    components: int
    palette: Optional[bytes] = None  # index -> gray, 256 entries, Indexed only


def _luma(r: int, g: int, b: int) -> int:
    """Return the 8-bit luminance of an 8-bit RGB triple."""
    return (_LUMA_R * r + _LUMA_G * g + _LUMA_B * b + 32768) >> 16


def _cmyk_to_gray(c: int, m: int, y: int, k: int) -> int:
    """Return the 8-bit luminance of an 8-bit CMYK quadruple."""
    inv_k = 255 - k
    r = ((255 - c) * inv_k) // 255
    g = ((255 - m) * inv_k) // 255
    b = ((255 - y) * inv_k) // 255
    return _luma(r, g, b)


def _resolve_colorspace(cs: Any, resolver: Any, depth: int = 0) -> _ColorSpace:
    """Map a ``/ColorSpace`` value onto a :class:`_ColorSpace`.

    Unknown spaces degrade to single-component DeviceGray, which is the least
    destructive guess: it produces a readable (if wrongly toned) image instead of
    garbage or an exception.
    """
    cs = resolve_with(cs, resolver)
    if depth > 4:
        return _ColorSpace("DeviceGray", 1)
    if cs is None:
        return _ColorSpace("DeviceGray", 1)
    if isinstance(cs, (PdfName, str, bytes, bytearray)):
        name = _name_of(cs)
        if name in _FAMILY_COMPONENTS:
            return _ColorSpace(name, _FAMILY_COMPONENTS[name])
        return _ColorSpace("DeviceGray", 1)
    if isinstance(cs, (PdfArray, list, tuple)):
        if not cs:
            return _ColorSpace("DeviceGray", 1)
        family = _name_of(resolve_with(cs[0], resolver))
        if family in ("Indexed", "I") and len(cs) >= 4:
            base = _resolve_colorspace(cs[1], resolver, depth + 1)
            hival = _as_int(resolve_with(cs[2], resolver), 0)
            lookup = _palette_bytes(resolve_with(cs[3], resolver), resolver)
            return _ColorSpace("Indexed", 1, _palette_to_gray(base, lookup, hival))
        if family in ("ICCBased",) and len(cs) >= 2:
            stream = resolve_with(cs[1], resolver)
            n = 0
            if isinstance(stream, PdfStream):
                n = _as_int(stream.dict.resolved_get("N", None, resolver), 0)
                if n not in (1, 3, 4):
                    alt = stream.dict.resolved_get("Alternate", None, resolver)
                    if alt is not None:
                        return _resolve_colorspace(alt, resolver, depth + 1)
            if n == 3:
                return _ColorSpace("DeviceRGB", 3)
            if n == 4:
                return _ColorSpace("DeviceCMYK", 4)
            return _ColorSpace("DeviceGray", 1)
        if family in ("CalRGB", "Lab"):
            return _ColorSpace("DeviceRGB", 3)
        if family == "CalGray":
            return _ColorSpace("DeviceGray", 1)
        if family == "DeviceN" and len(cs) >= 2:
            names = resolve_with(cs[1], resolver)
            count = len(names) if isinstance(names, (PdfArray, list, tuple)) else 1
            return _ColorSpace("DeviceN", max(1, count))
        if family == "Separation":
            return _ColorSpace("Separation", 1)
        if family in _FAMILY_COMPONENTS:
            return _ColorSpace(family, _FAMILY_COMPONENTS[family])
    return _ColorSpace("DeviceGray", 1)


def _palette_bytes(lookup: Any, resolver: Any) -> bytes:
    """Return the raw palette table of an ``/Indexed`` colour space."""
    if isinstance(lookup, PdfString):
        return bytes(lookup)
    if isinstance(lookup, PdfStream):
        return lookup.decoded(resolver)
    if isinstance(lookup, (bytes, bytearray)):
        return bytes(lookup)
    return b""


def _palette_to_gray(base: _ColorSpace, table: bytes, hival: int) -> bytes:
    """Convert an ``/Indexed`` lookup table into a 256-entry index -> gray table."""
    n = max(1, base.components)
    out = bytearray(256)
    entries = len(table) // n
    limit = min(256, max(0, hival + 1))
    for i in range(256):
        if i >= limit or i >= entries:
            # Out-of-range indices are undefined; black is what most viewers show.
            out[i] = 0
            continue
        off = i * n
        if n == 1:
            out[i] = table[off]
        elif n == 3:
            out[i] = _luma(table[off], table[off + 1], table[off + 2])
        elif n == 4:
            out[i] = _cmyk_to_gray(table[off], table[off + 1], table[off + 2], table[off + 3])
        else:
            out[i] = table[off]
    return bytes(out)


# ======================================================================================
# Bit unpacking
# ======================================================================================

_EXPAND_CACHE: Dict[int, Tuple[bytes, ...]] = {}


def _expansion_table(bpc: int) -> Tuple[bytes, ...]:
    """Return a 256-entry table mapping one packed byte to its samples, one per byte."""
    cached = _EXPAND_CACHE.get(bpc)
    if cached is not None:
        return cached
    per_byte = 8 // bpc
    mask = (1 << bpc) - 1
    table = []
    for value in range(256):
        row = bytearray(per_byte)
        for j in range(per_byte):
            shift = 8 - bpc * (j + 1)
            row[j] = (value >> shift) & mask
        table.append(bytes(row))
    result = tuple(table)
    _EXPAND_CACHE[bpc] = result
    return result


def _unpack_row(data: bytes, bpc: int, count: int) -> bytes:
    """Return ``count`` samples of ``bpc`` bits each, one sample per output byte."""
    if bpc == 8:
        row = data[:count]
    elif bpc == 16:
        row = data[0 : count * 2 : 2]
    elif bpc in (1, 2, 4):
        table = _expansion_table(bpc)
        row = b"".join(map(table.__getitem__, data))[:count]
    else:
        return bytes(count)
    if len(row) < count:
        row = row + bytes(count - len(row))
    return row


def _scale_table(bpc: int, decode: Sequence[float] | None) -> bytes:
    """Return a 256-entry table mapping a raw sample to an 8-bit gray level.

    ``decode`` is the two-element ``/Decode`` pair for a one-component image; it maps the
    sample range onto an output range, which is how ``/Decode [1 0]`` inverts an image.
    """
    maxval = (1 << bpc) - 1 if bpc < 16 else 255
    if maxval <= 0:
        maxval = 1
    lo, hi = (0.0, 1.0)
    if decode and len(decode) >= 2:
        lo, hi = float(decode[0]), float(decode[1])
    out = bytearray(256)
    for value in range(256):
        sample = value if value <= maxval else maxval
        level = lo + (sample * (hi - lo)) / maxval
        pixel = int(round(level * 255.0))
        out[value] = 0 if pixel < 0 else (255 if pixel > 255 else pixel)
    return bytes(out)


def _invert_table() -> bytes:
    """A 256-entry table that inverts a byte."""
    return bytes(255 - i for i in range(256))


def _decode_is_inverted(decode: Sequence[float], components: int) -> bool:
    """True when a ``/Decode`` array is the plain ``[1 0 1 0 ...]`` inversion."""
    if len(decode) < 2 * components:
        return False
    for i in range(components):
        if not (decode[2 * i] > decode[2 * i + 1]):
            return False
    return True


# ======================================================================================
# Raw sample images
# ======================================================================================


def _samples_to_gray(
    data: bytes,
    width: int,
    height: int,
    bpc: int,
    space: _ColorSpace,
    decode: Sequence[float],
    image_mask: bool,
) -> bytes:
    """Convert packed raw samples into ``width * height`` gray bytes."""
    components = 1 if image_mask else max(1, space.components)
    bits_per_row = width * components * bpc
    stride = (bits_per_row + 7) // 8
    out = bytearray(width * height)

    if image_mask:
        # 1 bit per sample: 0 paints, 1 leaves the background alone.  /Decode [1 0]
        # swaps that.  Painted -> black ink, unpainted -> white paper.
        painted_first = not (len(decode) >= 2 and decode[0] > decode[1])
        table = bytes((0 if (i == 0) == painted_first else 255) for i in range(256))
    elif space.family == "Indexed" and space.palette is not None:
        table = space.palette
    elif components == 1:
        table = _scale_table(bpc, decode[:2] if len(decode) >= 2 else None)
    else:
        table = b""

    inverted = bool(decode) and components > 1 and _decode_is_inverted(decode, components)
    invert = _invert_table()
    # Multi-component samples below 8 bits are widened to the full 0..255 range before
    # the colour conversion, so the luminance/CMYK maths always sees 8-bit values.
    widen = b""
    if components > 1 and bpc < 8:
        maxval = (1 << bpc) - 1
        widen = bytes(min(255, (i * 255) // maxval) if i <= maxval else 255 for i in range(256))

    for y in range(height):
        raw = data[y * stride : (y + 1) * stride]
        if not raw:
            # Truncated stream: leave the remaining rows white rather than failing.
            for yy in range(y, height):
                out[yy * width : (yy + 1) * width] = b"\xff" * width
            break
        samples = _unpack_row(raw, bpc, width * components)
        if widen:
            samples = samples.translate(widen)
        if inverted:
            samples = samples.translate(invert)
        if components == 1:
            row = samples.translate(table)
        elif components == 3:
            row = bytes(
                _luma(r, g, b)
                for r, g, b in zip(samples[0::3], samples[1::3], samples[2::3])
            )
        elif components == 4:
            row = bytes(
                _cmyk_to_gray(c, m, yv, k)
                for c, m, yv, k in zip(
                    samples[0::4], samples[1::4], samples[2::4], samples[3::4]
                )
            )
        else:
            row = bytes(samples[i * components] for i in range(width))
        out[y * width : (y + 1) * width] = row
    return bytes(out)


# ======================================================================================
# CCITT Group 3 / Group 4 fax decoding
# ======================================================================================

# ITU-T T.4 terminating and make-up codes, written as (bit string, run length).  The
# tables are prefix-free by construction, which ``tests/unit/test_raster_render`` asserts.
_WHITE_CODES: Tuple[Tuple[str, int], ...] = (
    ("00110101", 0), ("000111", 1), ("0111", 2), ("1000", 3), ("1011", 4),
    ("1100", 5), ("1110", 6), ("1111", 7), ("10011", 8), ("10100", 9),
    ("00111", 10), ("01000", 11), ("001000", 12), ("000011", 13), ("110100", 14),
    ("110101", 15), ("101010", 16), ("101011", 17), ("0100111", 18), ("0001100", 19),
    ("0001000", 20), ("0010111", 21), ("0000011", 22), ("0000100", 23), ("0101000", 24),
    ("0101011", 25), ("0010011", 26), ("0100100", 27), ("0011000", 28), ("00000010", 29),
    ("00000011", 30), ("00011010", 31), ("00011011", 32), ("00010010", 33),
    ("00010011", 34), ("00010100", 35), ("00010101", 36), ("00010110", 37),
    ("00010111", 38), ("00101000", 39), ("00101001", 40), ("00101010", 41),
    ("00101011", 42), ("00101100", 43), ("00101101", 44), ("00000100", 45),
    ("00000101", 46), ("00001010", 47), ("00001011", 48), ("01010010", 49),
    ("01010011", 50), ("01010100", 51), ("01010101", 52), ("00100100", 53),
    ("00100101", 54), ("01011000", 55), ("01011001", 56), ("01011010", 57),
    ("01011011", 58), ("01001010", 59), ("01001011", 60), ("00110010", 61),
    ("00110011", 62), ("00110100", 63),
    # make-up codes
    ("11011", 64), ("10010", 128), ("010111", 192), ("0110111", 256), ("00110110", 320),
    ("00110111", 384), ("01100100", 448), ("01100101", 512), ("01101000", 576),
    ("01100111", 640), ("011001100", 704), ("011001101", 768), ("011010010", 832),
    ("011010011", 896), ("011010100", 960), ("011010101", 1024), ("011010110", 1088),
    ("011010111", 1152), ("011011000", 1216), ("011011001", 1280), ("011011010", 1344),
    ("011011011", 1408), ("010011000", 1472), ("010011001", 1536), ("010011010", 1600),
    ("011000", 1664), ("010011011", 1728),
)

_BLACK_CODES: Tuple[Tuple[str, int], ...] = (
    ("0000110111", 0), ("010", 1), ("11", 2), ("10", 3), ("011", 4),
    ("0011", 5), ("0010", 6), ("00011", 7), ("000101", 8), ("000100", 9),
    ("0000100", 10), ("0000101", 11), ("0000111", 12), ("00000100", 13),
    ("00000111", 14), ("000011000", 15), ("0000010111", 16), ("0000011000", 17),
    ("0000001000", 18), ("00001100111", 19), ("00001101000", 20), ("00001101100", 21),
    ("00000110111", 22), ("00000101000", 23), ("00000010111", 24), ("00000011000", 25),
    ("000011001010", 26), ("000011001011", 27), ("000011001100", 28),
    ("000011001101", 29), ("000001101000", 30), ("000001101001", 31),
    ("000001101010", 32), ("000001101011", 33), ("000011010010", 34),
    ("000011010011", 35), ("000011010100", 36), ("000011010101", 37),
    ("000011010110", 38), ("000011010111", 39), ("000001101100", 40),
    ("000001101101", 41), ("000011011010", 42), ("000011011011", 43),
    ("000001010100", 44), ("000001010101", 45), ("000001010110", 46),
    ("000001010111", 47), ("000001100100", 48), ("000001100101", 49),
    ("000001010010", 50), ("000001010011", 51), ("000000100100", 52),
    ("000000110111", 53), ("000000111000", 54), ("000000100111", 55),
    ("000000101000", 56), ("000001011000", 57), ("000001011001", 58),
    ("000000101011", 59), ("000000101100", 60), ("000001011010", 61),
    ("000001100110", 62), ("000001100111", 63),
    # make-up codes
    ("0000001111", 64), ("000011001000", 128), ("000011001001", 192),
    ("000001011011", 256), ("000000110011", 320), ("000000110100", 384),
    ("000000110101", 448), ("0000001101100", 512), ("0000001101101", 576),
    ("0000001001010", 640), ("0000001001011", 704), ("0000001001100", 768),
    ("0000001001101", 832), ("0000001110010", 896), ("0000001110011", 960),
    ("0000001110100", 1024), ("0000001110101", 1088), ("0000001110110", 1152),
    ("0000001110111", 1216), ("0000001010010", 1280), ("0000001010011", 1344),
    ("0000001010100", 1408), ("0000001010101", 1472), ("0000001011010", 1536),
    ("0000001011011", 1600), ("0000001100100", 1664), ("0000001100101", 1728),
)

#: Extended make-up codes, shared by both colours (T.4 table 3).
_EXT_CODES: Tuple[Tuple[str, int], ...] = (
    ("00000001000", 1792), ("00000001100", 1856), ("00000001101", 1920),
    ("000000010010", 1984), ("000000010011", 2048), ("000000010100", 2112),
    ("000000010101", 2176), ("000000010110", 2240), ("000000010111", 2304),
    ("000000011100", 2368), ("000000011101", 2432), ("000000011110", 2496),
    ("000000011111", 2560),
)

_RUN_LUT_BITS = 13

#: Upper bound on rows decoded when ``/Rows`` is absent -- far above any real fax page,
#: low enough that a corrupt stream cannot allocate unbounded memory.
_MAX_CCITT_ROWS = 65536

#: Two-dimensional mode codes (T.4 table 4).
_MODE_PASS = "P"
_MODE_HORIZ = "H"
_MODE_EXT = "X"
_MODE_CODES: Tuple[Tuple[str, Any], ...] = (
    ("1", 0), ("011", 1), ("000011", 2), ("0000011", 3),
    ("010", -1), ("000010", -2), ("0000010", -3),
    ("001", _MODE_HORIZ), ("0001", _MODE_PASS), ("0000001", _MODE_EXT),
)
_MODE_LUT_BITS = 7

_EOL_CODE = 0b000000000001
_EOL_BITS = 12

_LUT_CACHE: Dict[str, List[Any]] = {}


def _build_lut(codes: Sequence[Tuple[str, Any]], width: int) -> List[Any]:
    """Expand a prefix code table into a flat ``2**width`` lookup list."""
    lut: List[Any] = [None] * (1 << width)
    for bits, value in codes:
        length = len(bits)
        if length > width:
            raise ValueError("code %r is wider than the %d-bit table" % (bits, width))
        base = int(bits, 2) << (width - length)
        for i in range(1 << (width - length)):
            lut[base + i] = (value, length)
    return lut


def _white_lut() -> List[Any]:
    lut = _LUT_CACHE.get("white")
    if lut is None:
        lut = _build_lut(_WHITE_CODES + _EXT_CODES, _RUN_LUT_BITS)
        _LUT_CACHE["white"] = lut
    return lut


def _black_lut() -> List[Any]:
    lut = _LUT_CACHE.get("black")
    if lut is None:
        lut = _build_lut(_BLACK_CODES + _EXT_CODES, _RUN_LUT_BITS)
        _LUT_CACHE["black"] = lut
    return lut


def _mode_lut() -> List[Any]:
    lut = _LUT_CACHE.get("mode")
    if lut is None:
        lut = _build_lut(_MODE_CODES, _MODE_LUT_BITS)
        _LUT_CACHE["mode"] = lut
    return lut


class _BitReader:
    """MSB-first bit reader that pads past the end with zeros instead of raising."""

    __slots__ = ("data", "size", "byte_pos", "acc", "avail", "padded")

    def __init__(self, data: bytes) -> None:
        self.data = bytes(data)
        self.size = len(self.data)
        self.byte_pos = 0
        self.acc = 0
        self.avail = 0
        self.padded = 0

    def _fill(self, n: int) -> None:
        while self.avail < n:
            if self.byte_pos < self.size:
                self.acc = (self.acc << 8) | self.data[self.byte_pos]
                self.byte_pos += 1
            else:
                self.acc <<= 8
                self.padded += 8
            self.avail += 8

    def peek(self, n: int) -> int:
        """Return the next ``n`` bits without consuming them."""
        self._fill(n)
        return (self.acc >> (self.avail - n)) & ((1 << n) - 1)

    def skip(self, n: int) -> None:
        """Consume ``n`` bits."""
        self._fill(n)
        self.avail -= n
        self.acc &= (1 << self.avail) - 1

    def read(self, n: int) -> int:
        """Consume and return the next ``n`` bits."""
        value = self.peek(n)
        self.skip(n)
        return value

    @property
    def bit_pos(self) -> int:
        """Absolute bit offset of the next unread bit."""
        return self.byte_pos * 8 - self.avail

    def align(self) -> None:
        """Advance to the next byte boundary."""
        remainder = self.bit_pos & 7
        if remainder:
            self.skip(8 - remainder)

    def exhausted(self) -> bool:
        """True once every real bit has been consumed."""
        return self.bit_pos >= self.size * 8


def _read_run(reader: _BitReader, lut: List[Any]) -> Optional[int]:
    """Decode one complete run length (make-up codes plus a terminating code)."""
    total = 0
    for _ in range(64):  # a run is at most a handful of make-up codes long
        entry = lut[reader.peek(_RUN_LUT_BITS)]
        if entry is None:
            return None
        run, bits = entry
        reader.skip(bits)
        total += run
        if run < 64:
            return total
    return None


def _at_eol(reader: _BitReader) -> bool:
    """True when the next 12 bits are an end-of-line code."""
    return reader.peek(_EOL_BITS) == _EOL_CODE


def _consume_eols(reader: _BitReader, limit: int = 64) -> int:
    """Skip fill bits and any number of EOL codes; return how many EOLs were consumed."""
    count = 0
    for _ in range(limit):
        if _at_eol(reader):
            reader.skip(_EOL_BITS)
            count += 1
            continue
        if reader.peek(_EOL_BITS) == 0 and not reader.exhausted():
            # Fill bits: a run of zeros that precedes an EOL.  Step over one at a time.
            reader.skip(1)
            continue
        break
    return count


def _find_b1(ref: Sequence[int], a0: int, color: int, columns: int, hint: int) -> Tuple[int, int, int]:
    """Return ``(b1, b2, next_hint)`` for the reference line.

    ``b1`` is the first changing element on the reference line strictly right of ``a0``
    whose colour is the opposite of ``color``; ``b2`` is the next one after it.  Even
    indices in ``ref`` are white-to-black transitions, odd ones black-to-white, so the
    required parity of the index is exactly ``color``.
    """
    n = len(ref)
    i = hint if 0 <= hint <= n else 0
    if i > 0 and ref[i - 1] > a0:
        i = 0  # a0 moved backwards: the cached scan position is no longer valid
    while i < n and ref[i] <= a0:
        i += 1
    hint = i
    if (i & 1) != color:
        i += 1
    b1 = ref[i] if i < n else columns
    b2 = ref[i + 1] if i + 1 < n else columns
    return (b1, b2, hint)


def _decode_1d_row(reader: _BitReader, columns: int) -> Optional[List[int]]:
    """Decode one one-dimensional (MH) row into its changing-element positions."""
    white, black = _white_lut(), _black_lut()
    changes: List[int] = []
    pos = 0
    color = 0
    for _ in range(2 * columns + 16):
        if pos >= columns:
            return changes
        run = _read_run(reader, white if color == 0 else black)
        if run is None:
            return changes if changes else None
        pos += run
        if pos > columns:
            pos = columns
        changes.append(pos)
        color ^= 1
    return changes


def _decode_2d_row(
    reader: _BitReader, ref: Sequence[int], columns: int
) -> Optional[List[int]]:
    """Decode one two-dimensional (MR/MMR) row into its changing-element positions."""
    white, black = _white_lut(), _black_lut()
    modes = _mode_lut()
    changes: List[int] = []
    a0 = -1
    color = 0
    hint = 0
    for _ in range(2 * columns + 16):
        if a0 >= columns:
            return changes
        b1, b2, hint = _find_b1(ref, a0, color, columns, hint)
        entry = modes[reader.peek(_MODE_LUT_BITS)]
        if entry is None:
            return changes if changes else None
        mode, bits = entry
        reader.skip(bits)
        if mode == _MODE_PASS:
            a0 = b2
            continue
        if mode == _MODE_EXT:
            return changes if changes else None
        if mode == _MODE_HORIZ:
            start = 0 if a0 < 0 else a0
            run1 = _read_run(reader, white if color == 0 else black)
            run2 = _read_run(reader, black if color == 0 else white)
            if run1 is None or run2 is None:
                return changes if changes else None
            a1 = min(start + run1, columns)
            a2 = min(a1 + run2, columns)
            changes.append(a1)
            changes.append(a2)
            a0 = a2
            continue
        # Vertical mode: ``mode`` is the signed offset from b1.
        a1 = b1 + mode
        if a1 < 0:
            a1 = 0
        elif a1 > columns:
            a1 = columns
        changes.append(a1)
        if a1 < a0:
            hint = 0  # the reference scan has to restart when a0 moved backwards
        a0 = a1
        color ^= 1
    return changes


def _set_bits(row: bytearray, start: int, end: int) -> None:
    """Set bits ``[start, end)`` of a packed row to 1."""
    if end <= start:
        return
    first = start >> 3
    last = (end - 1) >> 3
    if first == last:
        row[first] |= (0xFF >> (start & 7)) & (0xFF << (7 - ((end - 1) & 7))) & 0xFF
        return
    row[first] |= 0xFF >> (start & 7)
    if last > first + 1:
        row[first + 1 : last] = b"\xff" * (last - first - 1)
    row[last] |= (0xFF << (7 - ((end - 1) & 7))) & 0xFF


def _pack_changes(changes: Sequence[int], columns: int, stride: int) -> bytes:
    """Pack changing-element positions into a row where a 1 bit means black."""
    row = bytearray(stride)
    previous = 0
    ordered: List[int] = []
    for value in changes:
        value = 0 if value < 0 else (columns if value > columns else value)
        if value < previous:
            value = previous
        ordered.append(value)
        previous = value
    i = 0
    n = len(ordered)
    while i < n:
        start = ordered[i]
        end = ordered[i + 1] if i + 1 < n else columns
        _set_bits(row, start, end)
        i += 2
    return bytes(row)


def ccitt_decode(
    data: bytes,
    columns: int = 1728,
    rows: int = 0,
    k: int = 0,
    black_is_1: bool = False,
    byte_align: bool = False,
) -> Tuple[bytes, int]:
    """Decode a CCITT Group 3/4 stream into packed one-bit-per-pixel rows.

    Args:
        data: The encoded bytes.
        columns: Pixels per row (``/Columns``).
        rows: Expected row count (``/Rows``); 0 means "until the data runs out".
        k: ``/K`` -- below zero is Group 4 (pure 2D), zero is Group 3 one-dimensional,
            above zero is Group 3 mixed, where each row carries a 1D/2D flag bit.
        black_is_1: ``/BlackIs1``.  False (the default) means a 0 bit is black.
        byte_align: ``/EncodedByteAlign`` -- every row starts on a byte boundary.

    Returns:
        ``(packed, row_count)`` where ``packed`` holds ``row_count`` rows of
        ``ceil(columns / 8)`` bytes.  Decoding stops at the first unreadable code and
        returns everything recovered so far, so damaged faxes still produce an image.
    """
    columns = max(1, int(columns))
    stride = (columns + 7) // 8
    reader = _BitReader(data)
    out = bytearray()
    ref: List[int] = []
    produced = 0
    limit = int(rows) if int(rows) > 0 else _MAX_CCITT_ROWS

    while produced < limit:
        if byte_align:
            reader.align()
        if reader.exhausted():
            break
        if k < 0:
            if _at_eol(reader):
                break  # EOFB
            two_d = True
        else:
            _consume_eols(reader)
            if reader.exhausted():
                break
            two_d = False
            if k > 0:
                two_d = reader.read(1) == 0
        before = reader.bit_pos
        changes = _decode_2d_row(reader, ref, columns) if two_d else _decode_1d_row(reader, columns)
        if changes is None:
            break
        if reader.bit_pos == before:
            break  # made no progress: refuse to spin
        out += _pack_changes(changes, columns, stride)
        ref = changes
        produced += 1
        if reader.exhausted() and reader.padded > 0:
            break

    if not black_is_1 and out:
        out = bytearray(out.translate(_invert_table()))
    return (bytes(out), produced)


# ======================================================================================
# Baseline JPEG (DCTDecode)
# ======================================================================================

class JpegUnsupported(UnsupportedFeatureError):
    """The JPEG stream uses a mode this pure-python decoder does not implement."""


#: Natural (row-major) position of each zig-zag coefficient index.
_ZIGZAG: Tuple[int, ...] = (
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
)

_IDCT_CACHE: Dict[str, Any] = {}


def _idct_tables() -> Tuple[List[List[float]], float]:
    """Return ``(T_by_x, dc_scale)`` for the 8-point inverse DCT.

    ``T_by_x[x][u] == c(u)/2 * cos((2x+1) u pi / 16)`` with ``c(0) = 1/sqrt(2)``, which
    is the plain mathematical definition of the inverse DCT-II -- no fast factorisation,
    just a precomputed cosine table plus zero-coefficient shortcuts.
    """
    cached = _IDCT_CACHE.get("t")
    if cached is None:
        table = [[0.0] * 8 for _ in range(8)]
        for x in range(8):
            for u in range(8):
                cu = (1.0 / math.sqrt(2.0)) if u == 0 else 1.0
                table[x][u] = 0.5 * cu * math.cos((2 * x + 1) * u * math.pi / 16.0)
        cached = (table, table[0][0])
        _IDCT_CACHE["t"] = cached
    return cached


def _idct_2d(coeffs: List[float]) -> List[float]:
    """Return the 64 spatial samples (natural order, still centred on zero)."""
    table, dc = _idct_tables()
    tmp = [0.0] * 64
    for v in range(8):
        base = v * 8
        row = coeffs[base : base + 8]
        if row[1] == 0.0 and row[2] == 0.0 and row[3] == 0.0 and row[4] == 0.0 \
                and row[5] == 0.0 and row[6] == 0.0 and row[7] == 0.0:
            value = row[0] * dc
            if value:
                for x in range(8):
                    tmp[base + x] = value
            continue
        for x in range(8):
            tx = table[x]
            total = 0.0
            for u in range(8):
                cu = row[u]
                if cu:
                    total += cu * tx[u]
            tmp[base + x] = total
    out = [0.0] * 64
    for x in range(8):
        column = (
            tmp[x], tmp[8 + x], tmp[16 + x], tmp[24 + x],
            tmp[32 + x], tmp[40 + x], tmp[48 + x], tmp[56 + x],
        )
        if column[1] == 0.0 and column[2] == 0.0 and column[3] == 0.0 and column[4] == 0.0 \
                and column[5] == 0.0 and column[6] == 0.0 and column[7] == 0.0:
            value = column[0] * dc
            if value:
                for y in range(8):
                    out[y * 8 + x] = value
            continue
        for y in range(8):
            ty = table[y]
            total = 0.0
            for v in range(8):
                cv = column[v]
                if cv:
                    total += cv * ty[v]
            out[y * 8 + x] = total
    return out


class _HuffTable:
    """A canonical JPEG Huffman table with a 9-bit fast path."""

    __slots__ = ("fast", "slow")

    def __init__(self, counts: Sequence[int], symbols: Sequence[int]) -> None:
        self.fast: List[Any] = [None] * 512
        self.slow: Dict[Tuple[int, int], int] = {}
        code = 0
        k = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                if k >= len(symbols):
                    break
                symbol = symbols[k]
                k += 1
                if length <= 9:
                    base = code << (9 - length)
                    for i in range(1 << (9 - length)):
                        self.fast[base + i] = (symbol, length)
                else:
                    self.slow[(length, code)] = symbol
                code += 1
            code <<= 1


def _decode_huff(reader: _BitReader, table: _HuffTable) -> int:
    """Decode one Huffman symbol."""
    entry = table.fast[reader.peek(9)]
    if entry is not None:
        reader.skip(entry[1])
        return entry[0]
    slow = table.slow
    for length in range(10, 17):
        symbol = slow.get((length, reader.peek(length)))
        if symbol is not None:
            reader.skip(length)
            return symbol
    raise JpegUnsupported("corrupt JPEG: no Huffman code matches")


def _receive_extend(reader: _BitReader, size: int) -> int:
    """Read ``size`` bits and sign-extend them the way the JPEG spec does."""
    if size == 0:
        return 0
    value = reader.read(size)
    if value < (1 << (size - 1)):
        return value - (1 << size) + 1
    return value


@dataclass
class _JpegComponent:
    """One frame component and the plane its blocks are written into."""

    identifier: int
    h: int
    v: int
    tq: int
    plane_w: int = 0
    plane_h: int = 0
    plane: Optional[bytearray] = None
    blocks_x: int = 0
    blocks_y: int = 0
    needed: bool = True
    dc_table: int = 0
    ac_table: int = 0
    pred: int = 0


def _split_entropy(data: bytes, start: int) -> Tuple[List[bytes], int]:
    """Un-stuff an entropy-coded segment, splitting it at restart markers.

    Returns ``(chunks, end)`` where ``end`` is the offset of the marker that terminated
    the segment.
    """
    chunks: List[bytes] = []
    current = bytearray()
    i = start
    n = len(data)
    while i < n:
        j = data.find(b"\xff", i)
        if j < 0:
            current += data[i:n]
            i = n
            break
        current += data[i:j]
        if j + 1 >= n:
            i = n
            break
        marker = data[j + 1]
        if marker == 0x00:
            current.append(0xFF)
            i = j + 2
            continue
        if marker == 0xFF:
            i = j + 1
            continue
        if 0xD0 <= marker <= 0xD7:
            chunks.append(bytes(current))
            current = bytearray()
            i = j + 2
            continue
        i = j
        break
    chunks.append(bytes(current))
    return (chunks, i)


def _read_u16(data: bytes, pos: int) -> int:
    """Read a big-endian unsigned 16-bit value."""
    if pos + 1 >= len(data):
        raise JpegUnsupported("truncated JPEG")
    return (data[pos] << 8) | data[pos + 1]


def decode_jpeg_gray(data: bytes) -> Tuple[int, int, bytes]:
    """Decode a baseline sequential JPEG into ``(width, height, gray)``.

    Grayscale is the *luma* plane for YCbCr images, which is exactly
    ``0.299 R + 0.587 G + 0.114 B``, so the chroma blocks are entropy-decoded (the
    bit stream demands it) but never inverse-transformed.

    Raises:
        JpegUnsupported: progressive, arithmetic-coded, hierarchical or 12-bit JPEG, or
            a stream too damaged to read.
    """
    data = bytes(data)
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise JpegUnsupported("not a JPEG stream (no SOI marker)")

    quant: Dict[int, List[int]] = {}
    dc_tables: Dict[int, _HuffTable] = {}
    ac_tables: Dict[int, _HuffTable] = {}
    components: List[_JpegComponent] = []
    width = height = 0
    hmax = vmax = 1
    restart_interval = 0
    adobe_transform = -1
    mcus_x = mcus_y = 0
    saw_frame = False

    pos = 2
    n = len(data)
    while pos < n:
        if data[pos] != 0xFF:
            pos += 1
            continue
        while pos < n and data[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break
        marker = data[pos]
        pos += 1
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xD9:  # EOI
            break
        length = _read_u16(data, pos)
        segment = data[pos + 2 : pos + length]
        next_pos = pos + length

        if marker in (0xC0, 0xC1):  # baseline / extended sequential, Huffman
            if len(segment) < 6:
                raise JpegUnsupported("truncated SOF segment")
            if segment[0] != 8:
                raise JpegUnsupported("only 8-bit sample precision is supported")
            height = (segment[1] << 8) | segment[2]
            width = (segment[3] << 8) | segment[4]
            count = segment[5]
            if width <= 0 or height <= 0 or count <= 0:
                raise JpegUnsupported("degenerate JPEG frame")
            components = []
            for i in range(count):
                off = 6 + i * 3
                if off + 2 >= len(segment):
                    raise JpegUnsupported("truncated SOF component list")
                components.append(
                    _JpegComponent(
                        identifier=segment[off],
                        h=max(1, (segment[off + 1] >> 4) & 0x0F),
                        v=max(1, segment[off + 1] & 0x0F),
                        tq=segment[off + 2],
                    )
                )
            hmax = max(c.h for c in components)
            vmax = max(c.v for c in components)
            mcus_x = (width + 8 * hmax - 1) // (8 * hmax)
            mcus_y = (height + 8 * vmax - 1) // (8 * vmax)
            for comp in components:
                comp.blocks_x = mcus_x * comp.h
                comp.blocks_y = mcus_y * comp.v
                comp.plane_w = comp.blocks_x * 8
                comp.plane_h = comp.blocks_y * 8
                comp.plane = bytearray(b"\x80" * (comp.plane_w * comp.plane_h))
            if len(components) == 3:
                for index, comp in enumerate(components):
                    comp.needed = index == 0
            saw_frame = True
        elif marker == 0xC2:
            raise JpegUnsupported("progressive JPEG is not supported")
        elif marker in (0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            raise JpegUnsupported("JPEG mode 0x%02X is not supported" % marker)
        elif marker == 0xC4:  # DHT
            i = 0
            while i + 17 <= len(segment):
                tc_th = segment[i]
                counts = list(segment[i + 1 : i + 17])
                total = sum(counts)
                symbols = list(segment[i + 17 : i + 17 + total])
                table = _HuffTable(counts, symbols)
                if (tc_th >> 4) == 0:
                    dc_tables[tc_th & 0x0F] = table
                else:
                    ac_tables[tc_th & 0x0F] = table
                i += 17 + total
        elif marker == 0xDB:  # DQT
            i = 0
            while i < len(segment):
                pq_tq = segment[i]
                precision = pq_tq >> 4
                slot = pq_tq & 0x0F
                i += 1
                values: List[int] = []
                for _ in range(64):
                    if precision:
                        values.append(_read_u16(segment, i))
                        i += 2
                    else:
                        if i >= len(segment):
                            break
                        values.append(segment[i])
                        i += 1
                while len(values) < 64:
                    values.append(1)
                quant[slot] = values
        elif marker == 0xDD:  # DRI
            restart_interval = _read_u16(segment, 0) if len(segment) >= 2 else 0
        elif marker == 0xEE:  # APP14
            if segment[:5] == b"Adobe" and len(segment) >= 12:
                adobe_transform = segment[-1]
        elif marker == 0xDA:  # SOS
            if not saw_frame:
                raise JpegUnsupported("JPEG scan before frame header")
            ns = segment[0] if segment else 0
            scan: List[_JpegComponent] = []
            for i in range(ns):
                off = 1 + i * 2
                if off + 1 >= len(segment):
                    raise JpegUnsupported("truncated SOS component list")
                identifier = segment[off]
                tables = segment[off + 1]
                match = None
                for comp in components:
                    if comp.identifier == identifier:
                        match = comp
                        break
                if match is None:
                    match = components[min(i, len(components) - 1)]
                match.dc_table = (tables >> 4) & 0x0F
                match.ac_table = tables & 0x0F
                scan.append(match)
            chunks, end = _split_entropy(data, next_pos)
            _decode_jpeg_scan(
                chunks, scan, components, quant, dc_tables, ac_tables,
                mcus_x, mcus_y, hmax, vmax, width, height, restart_interval,
            )
            next_pos = end
        pos = next_pos

    if not saw_frame or not components:
        raise JpegUnsupported("JPEG has no frame header")
    return (width, height, _jpeg_to_gray(components, width, height, hmax, vmax, adobe_transform))


def _decode_jpeg_scan(
    chunks: List[bytes],
    scan: List[_JpegComponent],
    components: List[_JpegComponent],
    quant: Dict[int, List[int]],
    dc_tables: Dict[int, _HuffTable],
    ac_tables: Dict[int, _HuffTable],
    mcus_x: int,
    mcus_y: int,
    hmax: int,
    vmax: int,
    width: int,
    height: int,
    restart_interval: int,
) -> None:
    """Entropy-decode one sequential scan into the component planes."""
    interleaved = len(scan) > 1
    if interleaved:
        total_mcus = mcus_x * mcus_y
        across = mcus_x
    else:
        comp = scan[0]
        comp_w = (width * comp.h + hmax - 1) // hmax
        comp_h = (height * comp.v + vmax - 1) // vmax
        across = max(1, (comp_w + 7) // 8)
        down = max(1, (comp_h + 7) // 8)
        total_mcus = across * down

    chunk_index = 0
    reader = _BitReader(chunks[0] if chunks else b"")
    for comp in components:
        comp.pred = 0

    coeffs = [0.0] * 64
    for mcu in range(total_mcus):
        if restart_interval and mcu and mcu % restart_interval == 0:
            chunk_index += 1
            if chunk_index >= len(chunks):
                return
            reader = _BitReader(chunks[chunk_index])
            for comp in components:
                comp.pred = 0
        try:
            if interleaved:
                my, mx = divmod(mcu, across)
                for comp in scan:
                    for by in range(comp.v):
                        for bx in range(comp.h):
                            _decode_jpeg_block(
                                reader, comp, quant, dc_tables, ac_tables, coeffs,
                                mx * comp.h + bx, my * comp.v + by,
                            )
            else:
                comp = scan[0]
                by, bx = divmod(mcu, across)
                _decode_jpeg_block(
                    reader, comp, quant, dc_tables, ac_tables, coeffs, bx, by
                )
        except JpegUnsupported:
            # Corrupt tail: keep everything decoded so far rather than losing the page.
            return


def _decode_jpeg_block(
    reader: _BitReader,
    comp: _JpegComponent,
    quant: Dict[int, List[int]],
    dc_tables: Dict[int, _HuffTable],
    ac_tables: Dict[int, _HuffTable],
    coeffs: List[float],
    bx: int,
    by: int,
) -> None:
    """Decode one 8x8 block and blit it into the component plane."""
    dc = dc_tables.get(comp.dc_table)
    ac = ac_tables.get(comp.ac_table)
    if dc is None or ac is None:
        raise JpegUnsupported("JPEG scan references a missing Huffman table")
    qt = quant.get(comp.tq) or [1] * 64

    for i in range(64):
        coeffs[i] = 0.0
    size = _decode_huff(reader, dc)
    comp.pred += _receive_extend(reader, size)
    coeffs[0] = float(comp.pred * qt[0])
    k = 1
    while k < 64:
        rs = _decode_huff(reader, ac)
        run, size = rs >> 4, rs & 0x0F
        if size == 0:
            if run == 15:
                k += 16
                continue
            break
        k += run
        if k > 63:
            break
        coeffs[_ZIGZAG[k]] = float(_receive_extend(reader, size) * qt[k])
        k += 1

    if not comp.needed or comp.plane is None:
        return
    if bx >= comp.blocks_x or by >= comp.blocks_y:
        return
    samples = _idct_2d(coeffs)
    plane = comp.plane
    stride = comp.plane_w
    base = by * 8 * stride + bx * 8
    for y in range(8):
        offset = base + y * stride
        row = samples[y * 8 : y * 8 + 8]
        for x in range(8):
            value = int(row[x] + 128.5)
            plane[offset + x] = 0 if value < 0 else (255 if value > 255 else value)


def _upsample(comp: _JpegComponent, width: int, height: int, hmax: int, vmax: int) -> bytes:
    """Return the component plane resampled to the full image size (nearest neighbour)."""
    plane = comp.plane or bytearray()
    if comp.h == hmax and comp.v == vmax and comp.plane_w == width and comp.plane_h == height:
        return bytes(plane)
    out = bytearray(width * height)
    x_map = [min(comp.plane_w - 1, (x * comp.h) // hmax) for x in range(width)]
    for y in range(height):
        sy = min(comp.plane_h - 1, (y * comp.v) // vmax)
        row = bytes(plane[sy * comp.plane_w : (sy + 1) * comp.plane_w])
        if not row:
            row = b"\x80" * comp.plane_w
        out[y * width : (y + 1) * width] = bytes(map(row.__getitem__, x_map))
    return bytes(out)


def _jpeg_to_gray(
    components: List[_JpegComponent],
    width: int,
    height: int,
    hmax: int,
    vmax: int,
    adobe_transform: int,
) -> bytes:
    """Collapse the decoded component planes into one 8-bit gray image."""
    count = len(components)
    if count == 0:
        return b"\xff" * (width * height)
    if count in (1, 3):
        # One component is already gray; for YCbCr the luma plane *is* the luminance.
        return _upsample(components[0], width, height, hmax, vmax)
    planes = [_upsample(comp, width, height, hmax, vmax) for comp in components[:4]]
    if adobe_transform == 2:
        # YCCK: undo the chroma transform, then treat the result as CMY.
        out = bytearray(width * height)
        for i in range(width * height):
            y_val = planes[0][i]
            cb = planes[1][i] - 128
            cr = planes[2][i] - 128
            r = int(max(0.0, min(255.0, y_val + 1.402 * cr)))
            g = int(max(0.0, min(255.0, y_val - 0.344136 * cb - 0.714136 * cr)))
            b = int(max(0.0, min(255.0, y_val + 1.772 * cb)))
            # Adobe stores YCCK over inverted ink, so R'G'B' is (255-C, 255-M, 255-Y)
            # and the K plane is inverted too.
            out[i] = _cmyk_to_gray(255 - r, 255 - g, 255 - b, 255 - planes[3][i])
        return bytes(out)
    # Adobe CMYK JPEGs store inverted ink values; non-Adobe ones do not.
    inverted = adobe_transform >= 0
    out = bytearray(width * height)
    if inverted:
        for i in range(width * height):
            out[i] = _cmyk_to_gray(
                255 - planes[0][i], 255 - planes[1][i], 255 - planes[2][i], 255 - planes[3][i]
            )
    else:
        for i in range(width * height):
            out[i] = _cmyk_to_gray(planes[0][i], planes[1][i], planes[2][i], planes[3][i])
    return bytes(out)


# ======================================================================================
# The dispatcher
# ======================================================================================

#: Refuse to allocate a canvas larger than this; a bogus ``/Width`` must not be able to
#: exhaust memory.  120 megapixels is well past A0 at 300 dpi.
_MAX_PIXELS = 120_000_000


def _parm(parms: Any, key: str, default: int, resolver: Any = None) -> int:
    """Read an integer decode parameter."""
    if not isinstance(parms, PdfDict):
        return default
    return _as_int(parms.resolved_get(key, None, resolver), default)


def _parm_bool(parms: Any, key: str, default: bool, resolver: Any = None) -> bool:
    """Read a boolean decode parameter."""
    if not isinstance(parms, PdfDict):
        return default
    return _as_bool(parms.resolved_get(key, None, resolver), default)


def _flat(width: int, height: int, level: int) -> bytes:
    """Return a solid canvas of ``level``."""
    return bytes([level]) * (width * height)


def _crop_rows(data: bytes, src_width: int, dst_width: int, height: int) -> bytes:
    """Trim each row of a gray buffer from ``src_width`` to ``dst_width`` columns."""
    if src_width == dst_width:
        return data
    out = bytearray(dst_width * height)
    for y in range(height):
        row = data[y * src_width : y * src_width + dst_width]
        if len(row) < dst_width:
            row = row + b"\xff" * (dst_width - len(row))
        out[y * dst_width : (y + 1) * dst_width] = row
    return bytes(out)


def _resize_nearest(data: bytes, src_w: int, src_h: int, dst_w: int, dst_h: int) -> bytes:
    """Nearest-neighbour resample of a gray buffer."""
    if (src_w, src_h) == (dst_w, dst_h):
        return data
    if src_w <= 0 or src_h <= 0:
        return _flat(dst_w, dst_h, _WHITE)
    x_map = [min(src_w - 1, (x * src_w) // dst_w) for x in range(dst_w)]
    out = bytearray(dst_w * dst_h)
    for y in range(dst_h):
        sy = min(src_h - 1, (y * src_h) // dst_h)
        row = data[sy * src_w : (sy + 1) * src_w]
        if len(row) < src_w:
            row = row + b"\xff" * (src_w - len(row))
        out[y * dst_w : (y + 1) * dst_w] = bytes(map(row.__getitem__, x_map))
    return bytes(out)


def decode_image_xobject(stream: Any, resolver: Any = None) -> DecodedImage:
    """Decode an image XObject (or an inline image dictionary) to 8-bit grayscale.

    Args:
        stream: The :class:`~zfp.pdfio.objects.PdfStream` holding the image.
        resolver: Anything with a ``resolve`` method (a
            :class:`~zfp.pdfio.document.Document` is one) or a plain callable, used to
            follow indirect references in the image dictionary.

    Returns:
        A :class:`DecodedImage`.  Never raises for a malformed or unsupported image: the
        result carries ``supported == False`` and a flat canvas instead, because a
        perception pipeline would rather composite a blank region than lose the page.

    ``/Decode`` is applied in full for one-component images and image masks, and as a
    plain inversion for multi-component ones; an ``/Indexed`` index remap and a soft
    mask (``/SMask``) are both ignored, since neither changes where the ink is.
    """
    if not isinstance(stream, PdfStream):
        return DecodedImage(0, 0, b"", "empty", False, "not a stream")
    d = stream.dict
    width = _as_int(_first(d, resolver, "Width", "W"), 0)
    height = _as_int(_first(d, resolver, "Height", "H"), 0)
    if width <= 0 or height <= 0:
        return DecodedImage(0, 0, b"", "empty", False, "zero-sized image")
    if width * height > _MAX_PIXELS:
        return DecodedImage(
            1, 1, bytes([_MID_GRAY]), "unsupported", False,
            "image is %dx%d, beyond the %d pixel ceiling" % (width, height, _MAX_PIXELS),
        )

    image_mask = _as_bool(_first(d, resolver, "ImageMask", "IM"), False)
    bpc = 1 if image_mask else _as_int(_first(d, resolver, "BitsPerComponent", "BPC"), 8)
    if bpc not in (1, 2, 4, 8, 16):
        bpc = 8
    decode = _number_list(_first(d, resolver, "Decode", "D"), resolver)
    if image_mask:
        space = _ColorSpace("DeviceGray", 1)
    else:
        space = _resolve_colorspace(_first(d, resolver, "ColorSpace", "CS"), resolver)

    names = [normalize_filter_name(name) for name in stream.filter_names(resolver)]
    parms = stream.decode_parms(resolver)
    codec = ""
    codec_parms: Any = None
    for index, name in enumerate(names):
        if name in IMAGE_FILTERS:
            codec = name
            codec_parms = parms[index] if index < len(parms) else None
            break
    data = stream.decoded(resolver)

    if codec == "CCITTFaxDecode":
        columns = _parm(codec_parms, "Columns", 1728, resolver)
        if columns <= 0:
            columns = width
        rows = _parm(codec_parms, "Rows", 0, resolver) or height
        packed, produced = ccitt_decode(
            data,
            columns=columns,
            rows=rows,
            k=_parm(codec_parms, "K", 0, resolver),
            black_is_1=_parm_bool(codec_parms, "BlackIs1", False, resolver),
            byte_align=_parm_bool(codec_parms, "EncodedByteAlign", False, resolver),
        )
        stride = (columns + 7) // 8
        if produced < height:
            filler = b"\xff" if not _parm_bool(codec_parms, "BlackIs1", False, resolver) else b"\x00"
            packed = packed + filler * (stride * (height - produced))
        gray = _samples_to_gray(packed, columns, height, 1, space, decode, image_mask)
        gray = _crop_rows(gray, columns, width, height)
        detail = "K=%d %dx%d, %d rows decoded" % (
            _parm(codec_parms, "K", 0, resolver), columns, height, produced,
        )
        return DecodedImage(width, height, gray, "ccitt", produced > 0, detail)

    if codec == "DCTDecode":
        try:
            jw, jh, gray = decode_jpeg_gray(data)
        except JpegUnsupported as exc:
            _log.debug("DCTDecode fell back to a flat canvas: %s", exc)
            return DecodedImage(
                width, height, _flat(width, height, _MID_GRAY), "unsupported", False, str(exc)
            )
        except Exception as exc:  # pragma: no cover - defensive: never lose the page
            _log.debug("DCTDecode failed: %s", exc)
            return DecodedImage(
                width, height, _flat(width, height, _MID_GRAY), "unsupported", False, repr(exc)
            )
        if (jw, jh) != (width, height):
            gray = _resize_nearest(gray, jw, jh, width, height)
        if space.components != 4 and decode and _decode_is_inverted(decode, 1):
            gray = gray.translate(_invert_table())
        return DecodedImage(width, height, gray, "jpeg", True, "%dx%d baseline" % (jw, jh))

    if codec in ("JPXDecode", "JBIG2Decode"):
        return DecodedImage(
            width, height, _flat(width, height, _WHITE), "unsupported", False,
            "%s is not decodable without an optional backend" % codec,
        )

    if codec:
        return DecodedImage(
            width, height, _flat(width, height, _WHITE), "unsupported", False,
            "unknown image codec %s" % codec,
        )

    gray = _samples_to_gray(data, width, height, bpc, space, decode, image_mask)
    if image_mask:
        kind = "mask"
    elif space.family == "Indexed":
        kind = "indexed"
    elif space.components == 3:
        kind = "rgb"
    elif space.components == 4:
        kind = "cmyk"
    else:
        kind = "gray"
    expected = ((width * (1 if image_mask else max(1, space.components)) * bpc + 7) // 8) * height
    complete = len(data) >= expected
    return DecodedImage(
        width, height, gray, kind, True,
        "%s %dbpc%s" % (space.family, bpc, "" if complete else " (truncated stream)"),
    )
