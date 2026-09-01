"""Font encoding and CMap handling for native text extraction.

A content stream shows *bytes*.  Turning those bytes into characters -- and, just as
importantly for ZFP, into per-glyph advance widths -- needs the font dictionary that was
selected by ``Tf``.  This module owns that translation and nothing else:

* :func:`decode_string` splits a shown string into ``(charcode, unicode)`` pairs,
  honouring ``/Encoding`` as a name, ``/Encoding`` as a dictionary with ``/BaseEncoding``
  and ``/Differences``, an embedded ``/ToUnicode`` CMap, and two-byte ``/Type0``
  composite fonts;
* :func:`font_widths` returns ``(code -> advance, default advance)`` where every advance
  is expressed in **text space units per unit font size** -- that is, the PDF ``/Widths``
  value already divided by 1000, so the horizontal displacement of a glyph is simply
  ``width * font_size``.

Both are thin wrappers over :func:`load_font`, which builds a :class:`FontProgram` once
and is what the interpreter actually caches.

Nothing here raises: a font dictionary can be absent, circular, truncated or plain wrong
and the worst that happens is a fallback to Helvetica metrics and WinAnsi text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.logging import get_logger
from ..pdfio import fonts as _fonts
from ..pdfio.lexer import Lexer
from ..pdfio.objects import PdfDict, PdfName, PdfNull, PdfStream, resolve_with

__all__ = [
    "GLYPH_LIST",
    "STANDARD_ENCODING",
    "WINANSI_ENCODING",
    "MACROMAN_ENCODING",
    "PDF_DOC_ENCODING",
    "ZAPF_DINGBATS_OVERRIDES",
    "FontProgram",
    "glyph_to_unicode",
    "base_encoding_table",
    "parse_to_unicode",
    "parse_codespace_lengths",
    "load_font",
    "decode_string",
    "font_widths",
]

_log = get_logger(__name__)

#: Advance used when a font gives no usable width at all, in em units.
FALLBACK_ADVANCE = 0.5
#: ``/DW`` default for composite fonts, per the PDF specification, in em units.
DEFAULT_CID_WIDTH = 1.0


# --------------------------------------------------------------------------------------
# Glyph names
# --------------------------------------------------------------------------------------

# Codes 32..64, in order.
_ASCII_NAMES_32_64 = (
    "space", "exclam", "quotedbl", "numbersign", "dollar", "percent", "ampersand",
    "quotesingle", "parenleft", "parenright", "asterisk", "plus", "comma", "hyphen",
    "period", "slash", "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "colon", "semicolon", "less", "equal", "greater", "question", "at",
)
# Codes 91..96, in order.
_ASCII_NAMES_91_96 = (
    "bracketleft", "backslash", "bracketright", "asciicircum", "underscore", "grave",
)
# Codes 123..126, in order.
_ASCII_NAMES_123_126 = ("braceleft", "bar", "braceright", "asciitilde")
# Codes 161..255, in order (Adobe's Latin-1 glyph names).
_LATIN1_NAMES_161_255 = (
    "exclamdown", "cent", "sterling", "currency", "yen", "brokenbar", "section",
    "dieresis", "copyright", "ordfeminine", "guillemotleft", "logicalnot", "hyphen",
    "registered", "macron", "degree", "plusminus", "twosuperior", "threesuperior",
    "acute", "mu", "paragraph", "periodcentered", "cedilla", "onesuperior",
    "ordmasculine", "guillemotright", "onequarter", "onehalf", "threequarters",
    "questiondown", "Agrave", "Aacute", "Acircumflex", "Atilde", "Adieresis", "Aring",
    "AE", "Ccedilla", "Egrave", "Eacute", "Ecircumflex", "Edieresis", "Igrave", "Iacute",
    "Icircumflex", "Idieresis", "Eth", "Ntilde", "Ograve", "Oacute", "Ocircumflex",
    "Otilde", "Odieresis", "multiply", "Oslash", "Ugrave", "Uacute", "Ucircumflex",
    "Udieresis", "Yacute", "Thorn", "germandbls", "agrave", "aacute", "acircumflex",
    "atilde", "adieresis", "aring", "ae", "ccedilla", "egrave", "eacute", "ecircumflex",
    "edieresis", "igrave", "iacute", "icircumflex", "idieresis", "eth", "ntilde",
    "ograve", "oacute", "ocircumflex", "otilde", "odieresis", "divide", "oslash",
    "ugrave", "uacute", "ucircumflex", "udieresis", "yacute", "thorn", "ydieresis",
)

#: Glyph names that are not simply "the Latin-1 character with this name".
_EXTRA_GLYPHS: Dict[str, str] = {
    "nbspace": " ",
    "nonbreakingspace": " ",
    "sfthyphen": "­",
    "softhyphen": "­",
    "quoteright": "’",
    "quoteleft": "‘",
    "quotedblleft": "“",
    "quotedblright": "”",
    "quotesinglbase": "‚",
    "quotedblbase": "„",
    "endash": "–",
    "emdash": "—",
    "dagger": "†",
    "daggerdbl": "‡",
    "bullet": "•",
    "ellipsis": "…",
    "perthousand": "‰",
    "guilsinglleft": "‹",
    "guilsinglright": "›",
    "fraction": "⁄",
    "florin": "ƒ",
    "fi": "ﬁ",
    "fl": "ﬂ",
    "ff": "ﬀ",
    "ffi": "ﬃ",
    "ffl": "ﬄ",
    "trademark": "™",
    "Euro": "€",
    "euro": "€",
    "OE": "Œ",
    "oe": "œ",
    "Scaron": "Š",
    "scaron": "š",
    "Zcaron": "Ž",
    "zcaron": "ž",
    "Ydieresis": "Ÿ",
    "dotlessi": "ı",
    "Lslash": "Ł",
    "lslash": "ł",
    "minus": "−",
    "tilde": "˜",
    "circumflex": "ˆ",
    "caron": "ˇ",
    "breve": "˘",
    "dotaccent": "˙",
    "ring": "˚",
    "ogonek": "˛",
    "hungarumlaut": "˝",
    "commaaccent": ",",
    "Delta": "∆",
    "Omega": "Ω",
    "pi": "π",
    "check": "✓",
    "checkmark": "✓",
    "cross": "✗",
    "square": "□",
    "filledbox": "■",
    "filledsquare": "■",
    "circle": "○",
    "filledcircle": "●",
    "star": "★",
    "arrowright": "→",
    "arrowleft": "←",
    "arrowup": "↑",
    "arrowdown": "↓",
}


def _build_glyph_list() -> Dict[str, str]:
    """Return the glyph-name -> unicode table used to resolve ``/Differences``."""
    table: Dict[str, str] = {}
    for offset, name in enumerate(_ASCII_NAMES_32_64):
        table[name] = chr(32 + offset)
    for code in range(65, 91):
        table[chr(code)] = chr(code)
    for offset, name in enumerate(_ASCII_NAMES_91_96):
        table[name] = chr(91 + offset)
    for code in range(97, 123):
        table[chr(code)] = chr(code)
    for offset, name in enumerate(_ASCII_NAMES_123_126):
        table[name] = chr(123 + offset)
    # Latin-1 names never displace an ASCII name (``hyphen`` stays U+002D).
    for offset, name in enumerate(_LATIN1_NAMES_161_255):
        table.setdefault(name, chr(161 + offset))
    for name, value in _EXTRA_GLYPHS.items():
        table[name] = value
    return table


#: ``glyph name -> unicode text``.  Covers ASCII, Latin-1 and the common typographic and
#: accent names; anything else goes through the ``uniXXXX`` conventions in
#: :func:`glyph_to_unicode`.
GLYPH_LIST: Dict[str, str] = _build_glyph_list()


def glyph_to_unicode(name: str) -> str:
    """Return the text a glyph name stands for, or ``""`` when it is unknowable.

    Recognizes the Adobe glyph list subset in :data:`GLYPH_LIST`, the ``uniXXXX`` and
    ``uXXXX``..``uXXXXXX`` conventions, dot-suffixed variants (``one.oldstyle``), and
    multi-part ligature names (``f_i``).  Pure index names (``g23``, ``cid45``, ``G7``)
    carry no character information and return ``""``.

    Args:
        name: A glyph name, with or without a leading ``/``.

    Returns:
        The unicode text, possibly more than one character, or the empty string.

    Examples:
        >>> glyph_to_unicode("eacute")
        'é'
        >>> glyph_to_unicode("uni20AC")
        '€'
        >>> glyph_to_unicode("g14")
        ''
    """
    if not name:
        return ""
    if name.startswith("/"):
        name = name[1:]
    hit = GLYPH_LIST.get(name)
    if hit is not None:
        return hit
    # ``one.oldstyle`` / ``a.sc`` -- the base name carries the meaning.
    if "." in name:
        base = name.split(".", 1)[0]
        if base and base != name:
            return glyph_to_unicode(base)
    # ``uniXXXX`` (one or more four-digit values).
    if name.startswith("uni") and len(name) >= 7:
        body = name[3:]
        if len(body) % 4 == 0:
            try:
                return "".join(chr(int(body[i : i + 4], 16)) for i in range(0, len(body), 4))
            except ValueError:
                pass
    # ``uXXXX`` .. ``uXXXXXXXX``
    if name.startswith("u") and 5 <= len(name) <= 9:
        try:
            value = int(name[1:], 16)
        except ValueError:
            value = -1
        if 0 <= value <= 0x10FFFF:
            return chr(value)
    # ``f_i`` style ligature names.
    if "_" in name:
        parts = [glyph_to_unicode(part) for part in name.split("_")]
        if all(parts):
            return "".join(parts)
    return ""


# --------------------------------------------------------------------------------------
# Base encodings
# --------------------------------------------------------------------------------------

def _decode_via_codec(codec: str) -> Dict[int, str]:
    """Build a ``code -> character`` table from a stdlib single-byte codec."""
    table: Dict[int, str] = {}
    for code in range(32, 256):
        try:
            char = bytes((code,)).decode(codec)
        except UnicodeDecodeError:
            continue
        table[code] = char
    return table


def _build_winansi() -> Dict[int, str]:
    table = _decode_via_codec("cp1252")
    # WinAnsiEncoding treats these as plain space and hyphen for extraction purposes.
    table[0xA0] = " "
    table[0xAD] = "-"
    return table


def _build_standard() -> Dict[int, str]:
    table: Dict[int, str] = {}
    for offset, name in enumerate(_ASCII_NAMES_32_64):
        table[32 + offset] = GLYPH_LIST[name]
    for code in range(65, 91):
        table[code] = chr(code)
    for offset, name in enumerate(_ASCII_NAMES_91_96):
        table[91 + offset] = GLYPH_LIST[name]
    for code in range(97, 123):
        table[code] = chr(code)
    for offset, name in enumerate(_ASCII_NAMES_123_126):
        table[123 + offset] = GLYPH_LIST[name]
    table[39] = "’"  # quoteright
    table[96] = "‘"  # quoteleft
    high = {
        161: "exclamdown", 162: "cent", 163: "sterling", 164: "fraction", 165: "yen",
        166: "florin", 167: "section", 168: "currency", 169: "quotesingle",
        170: "quotedblleft", 171: "guillemotleft", 172: "guilsinglleft",
        173: "guilsinglright", 174: "fi", 175: "fl", 177: "endash", 178: "dagger",
        179: "daggerdbl", 180: "periodcentered", 182: "paragraph", 183: "bullet",
        184: "quotesinglbase", 185: "quotedblbase", 186: "quotedblright",
        187: "guillemotright", 188: "ellipsis", 189: "perthousand", 191: "questiondown",
        193: "grave", 194: "acute", 195: "circumflex", 196: "tilde", 197: "macron",
        198: "breve", 199: "dotaccent", 200: "dieresis", 202: "ring", 203: "cedilla",
        205: "hungarumlaut", 206: "ogonek", 207: "caron", 208: "emdash", 225: "AE",
        227: "ordfeminine", 232: "Lslash", 233: "Oslash", 234: "OE", 235: "ordmasculine",
        241: "ae", 245: "dotlessi", 248: "lslash", 249: "oslash", 250: "oe",
        251: "germandbls",
    }
    for code, name in high.items():
        table[code] = GLYPH_LIST.get(name, "")
    return table


#: ``WinAnsiEncoding`` -- CP1252, with 0xA0/0xAD folded onto space and hyphen.
WINANSI_ENCODING: Dict[int, str] = _build_winansi()
#: ``MacRomanEncoding``.
MACROMAN_ENCODING: Dict[int, str] = _decode_via_codec("mac_roman")
#: Adobe ``StandardEncoding`` (note 39/96 are the curly quotes, not the ASCII marks).
STANDARD_ENCODING: Dict[int, str] = _build_standard()
#: ``PDFDocEncoding`` is close enough to WinAnsi for text extraction.
PDF_DOC_ENCODING: Dict[int, str] = dict(WINANSI_ENCODING)

#: The handful of ZapfDingbats codes AcroForm actually uses for check styles.
ZAPF_DINGBATS_OVERRIDES: Dict[int, str] = {
    0x34: "✔",  # a20  check
    0x35: "✘",  # a21  heavy ballot X
    0x36: "✗",  # a22  ballot X
    0x38: "✘",  # a24  cross
    0x48: "★",  # a35  star
    0x6C: "●",  # a71  filled circle
    0x6E: "■",  # a73  filled square
    0x75: "◆",  # a117 filled diamond
}

_BASE_ENCODINGS: Dict[str, Dict[int, str]] = {
    "WinAnsiEncoding": WINANSI_ENCODING,
    "MacRomanEncoding": MACROMAN_ENCODING,
    "MacExpertEncoding": STANDARD_ENCODING,
    "StandardEncoding": STANDARD_ENCODING,
    "PDFDocEncoding": PDF_DOC_ENCODING,
}


def base_encoding_table(name: Optional[str]) -> Dict[int, str]:
    """Return a copy of the named base encoding, defaulting to ``StandardEncoding``.

    Args:
        name: ``"WinAnsiEncoding"``, ``"MacRomanEncoding"``, ``"StandardEncoding"``,
            ``"PDFDocEncoding"``, ``None`` or anything unrecognized.

    Returns:
        A fresh ``code -> character`` dictionary the caller may mutate.
    """
    return dict(_BASE_ENCODINGS.get(name or "", STANDARD_ENCODING))


# --------------------------------------------------------------------------------------
# CMaps
# --------------------------------------------------------------------------------------

def _utf16be(data: bytes) -> str:
    """Decode a ``/ToUnicode`` destination string (UTF-16BE, possibly surrogate pairs)."""
    if not data:
        return ""
    if len(data) % 2:
        data = data + b"\x00"
    try:
        text = data.decode("utf-16-be", "ignore")
    except (UnicodeDecodeError, LookupError):  # pragma: no cover - 'ignore' never raises
        return ""
    # Producers pad with NULs rather than trimming; a trailing NUL is never real text.
    return text.replace("\x00", "")


def _code_of(data: bytes) -> int:
    """Big-endian integer value of a CMap code string."""
    value = 0
    for byte in data:
        value = (value << 8) | byte
    return value


def parse_to_unicode(data: bytes) -> Dict[int, str]:
    """Parse a ``/ToUnicode`` CMap stream into ``{charcode: text}``.

    Understands ``beginbfchar``/``endbfchar`` and ``beginbfrange``/``endbfrange``,
    including the array form of a bfrange destination.  Malformed sections are skipped
    rather than aborting the parse, because a partially usable CMap still beats none.

    Args:
        data: The *decoded* bytes of the CMap stream.

    Returns:
        A mapping from character code to the text it stands for.

    Examples:
        >>> cmap = b"1 beginbfchar <0041> <0061> endbfchar"
        >>> parse_to_unicode(cmap)
        {65: 'a'}
    """
    out: Dict[int, str] = {}
    if not data:
        return out
    lexer = Lexer(data)
    operands: List[Any] = []
    guard = 0
    limit = len(data) * 4 + 1024
    while guard < limit:
        guard += 1
        token = lexer.next_token()
        if token.kind == "eof":
            break
        if token.kind == "keyword":
            word = token.value
            try:
                if word == "beginbfchar":
                    _parse_bfchar(lexer, out)
                elif word == "beginbfrange":
                    _parse_bfrange(lexer, out)
            except Exception:  # pragma: no cover - defensive, the helpers are lenient
                _log.debug("ToUnicode: %s section failed", word)
            operands = []
            continue
        operands.append(token.value)
        if len(operands) > 32:
            del operands[:-8]
    return out


def _parse_bfchar(lexer: Lexer, out: Dict[int, str]) -> None:
    """Consume ``<src> <dst>`` pairs up to ``endbfchar``."""
    pending: List[Any] = []
    for _ in range(200000):
        token = lexer.next_token()
        if token.kind == "eof":
            return
        if token.kind == "keyword":
            if token.value == "endbfchar":
                return
            continue
        if token.kind in ("hexstring", "string"):
            pending.append(token.value)
        elif token.kind == "name":
            pending.append(PdfName(token.value))
        else:
            continue
        if len(pending) == 2:
            src, dst = pending
            pending = []
            if not isinstance(src, bytes):
                continue
            code = _code_of(src)
            if isinstance(dst, PdfName):
                text = glyph_to_unicode(dst.value)
            else:
                text = _utf16be(dst)
            if text:
                out[code] = text


def _parse_bfrange(lexer: Lexer, out: Dict[int, str]) -> None:
    """Consume ``<lo> <hi> <dst>`` triples up to ``endbfrange``."""
    pending: List[Any] = []
    for _ in range(200000):
        token = lexer.next_token()
        if token.kind == "eof":
            return
        if token.kind == "keyword":
            if token.value == "endbfrange":
                return
            continue
        if token.kind in ("hexstring", "string"):
            pending.append(token.value)
        elif token.kind == "name":
            pending.append(PdfName(token.value))
        elif token.kind == "array_open":
            pending.append(_collect_array(lexer))
        else:
            continue
        if len(pending) == 3:
            lo_raw, hi_raw, dst = pending
            pending = []
            if not isinstance(lo_raw, bytes) or not isinstance(hi_raw, bytes):
                continue
            lo = _code_of(lo_raw)
            hi = _code_of(hi_raw)
            if hi < lo:
                lo, hi = hi, lo
            if hi - lo > 65535:  # a runaway range would eat all memory
                hi = lo + 65535
            if isinstance(dst, list):
                for offset, item in enumerate(dst):
                    if lo + offset > hi:
                        break
                    if isinstance(item, bytes):
                        text = _utf16be(item)
                    elif isinstance(item, PdfName):
                        text = glyph_to_unicode(item.value)
                    else:
                        text = ""
                    if text:
                        out[lo + offset] = text
                continue
            if isinstance(dst, PdfName):
                text = glyph_to_unicode(dst.value)
                if text:
                    out[lo] = text
                continue
            if not isinstance(dst, bytes) or not dst:
                continue
            start = _utf16be(dst)
            if not start:
                continue
            base = ord(start[-1])
            prefix = start[:-1]
            for offset in range(hi - lo + 1):
                point = base + offset
                if point > 0x10FFFF:
                    break
                out[lo + offset] = prefix + chr(point)


def _collect_array(lexer: Lexer) -> List[Any]:
    """Collect tokens up to the matching ``]`` (the ``[`` is already consumed)."""
    items: List[Any] = []
    for _ in range(100000):
        token = lexer.next_token()
        if token.kind in ("eof", "array_close"):
            return items
        if token.kind == "name":
            items.append(PdfName(token.value))
        elif token.kind == "array_open":
            items.append(_collect_array(lexer))
        else:
            items.append(token.value)
    return items


def parse_codespace_lengths(data: bytes) -> List[int]:
    """Return the distinct code byte lengths declared by a CMap's codespace ranges.

    Args:
        data: The decoded bytes of a CMap stream.

    Returns:
        Sorted byte lengths, e.g. ``[2]`` for a UCS-2 CMap or ``[1, 2]`` for a mixed
        one.  Empty when the CMap declares no codespace range.
    """
    lengths: set = set()
    if not data:
        return []
    lexer = Lexer(data)
    inside = False
    for _ in range(len(data) + 16):
        token = lexer.next_token()
        if token.kind == "eof":
            break
        if token.kind == "keyword":
            if token.value == "begincodespacerange":
                inside = True
            elif token.value == "endcodespacerange":
                inside = False
            continue
        if inside and token.kind in ("hexstring", "string") and token.value:
            lengths.add(len(token.value))
    return sorted(lengths)


# --------------------------------------------------------------------------------------
# The font model
# --------------------------------------------------------------------------------------

@dataclass
class FontProgram:
    """Everything the interpreter needs to know about one ``Tf`` font.

    Attributes:
        base_font: The raw ``/BaseFont`` value, subset tag included.
        std_font: ``base_font`` resolved onto one of the standard 14, used for the
            metric fallback and for :attr:`~zfp.core.types.TextSpan.font_name`.
        subtype: ``/Subtype`` (``Type1``, ``TrueType``, ``Type0``, ``Type3`` ...).
        code_bytes: 1 for a simple font, 2 for a composite one.
        widths: ``code -> advance`` in em units (``/Widths`` value / 1000).
        default_width: The advance used for a code absent from :attr:`widths`.
        to_unicode: ``code -> text`` from an embedded ``/ToUnicode`` CMap.
        encoding: ``code -> text`` from ``/Encoding`` (simple fonts only).
        ascent: Ascender in em units, for the glyph box top.
        descent: Descender in em units (negative), for the glyph box bottom.
    """

    base_font: str = ""
    std_font: str = "Helvetica"
    subtype: str = ""
    code_bytes: int = 1
    widths: Dict[int, float] = field(default_factory=dict)
    default_width: float = FALLBACK_ADVANCE
    to_unicode: Dict[int, str] = field(default_factory=dict)
    encoding: Dict[int, str] = field(default_factory=dict)
    ascent: float = 0.718
    descent: float = -0.207

    # -- decoding ---------------------------------------------------------------------
    def codes(self, raw: bytes) -> List[int]:
        """Split a shown string into character codes.

        A composite font consumes two bytes at a time; a trailing odd byte is still
        returned as a one-byte code rather than dropped.
        """
        if not raw:
            return []
        if self.code_bytes == 2:
            out: List[int] = []
            length = len(raw)
            index = 0
            while index + 1 < length:
                out.append((raw[index] << 8) | raw[index + 1])
                index += 2
            if index < length:
                out.append(raw[index])
            return out
        return list(raw)

    def text_for(self, code: int) -> str:
        """Return the text one character code stands for (``""`` when unknowable)."""
        hit = self.to_unicode.get(code)
        if hit is not None:
            return hit
        hit = self.encoding.get(code)
        if hit is not None:
            return hit
        if self.code_bytes == 2:
            return ""
        return WINANSI_ENCODING.get(code, "")

    def decode(self, raw: bytes) -> List[Tuple[int, str]]:
        """Return ``(charcode, text)`` for every code in ``raw``."""
        return [(code, self.text_for(code)) for code in self.codes(raw)]

    # -- metrics ----------------------------------------------------------------------
    def width(self, code: int) -> float:
        """Advance of one code in em units (multiply by the font size for points)."""
        hit = self.widths.get(code)
        if hit is not None:
            return hit
        return self.default_width

    def is_space(self, code: int) -> bool:
        """True when ``Tw`` word spacing applies to this code.

        Word spacing applies to the single-byte code 32 only -- never to a two-byte
        composite code, even when its value happens to be 32.
        """
        return self.code_bytes == 1 and code == 32


def _resolve(value: Any, resolver: Any) -> Any:
    """Resolve ``value`` and fold PDF ``null`` onto ``None``."""
    out = resolve_with(value, resolver)
    if out is None or isinstance(out, PdfNull):
        return None
    return out


def _as_dict(value: Any) -> Optional[PdfDict]:
    """Return ``value`` as a :class:`PdfDict`, unwrapping a stream, or ``None``."""
    if isinstance(value, PdfStream):
        return value.dict
    if isinstance(value, PdfDict):
        return value
    if isinstance(value, dict):
        return PdfDict(value)
    return None


def _number(value: Any, default: float = 0.0) -> float:
    """Coerce a PDF number, tolerating booleans and junk."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _name_of(value: Any) -> Optional[str]:
    """Return a plain name string from a :class:`PdfName` (or ``None``)."""
    if isinstance(value, PdfName):
        return value.value
    if isinstance(value, str):
        return value.lstrip("/")
    return None


def _sequence(value: Any) -> Sequence[Any]:
    """Return ``value`` as a sequence; a non-sequence becomes an empty tuple."""
    if isinstance(value, (list, tuple)):
        return value
    return ()


def _simple_encoding(font: PdfDict, std_font: str, resolver: Any) -> Dict[int, str]:
    """Build the ``code -> text`` table for a simple font."""
    encoding_obj = _resolve(font.get("Encoding"), resolver)
    name = _name_of(encoding_obj)
    if name is not None:
        table = base_encoding_table(name)
    elif isinstance(encoding_obj, (PdfDict, dict)):
        holder = _as_dict(encoding_obj) or PdfDict()
        base_name = _name_of(_resolve(holder.get("BaseEncoding"), resolver))
        table = base_encoding_table(base_name)
        _apply_differences(table, _resolve(holder.get("Differences"), resolver), resolver)
    else:
        table = base_encoding_table(None)
    if std_font == "ZapfDingbats":
        # The built-in encoding is not Latin at all; overlay the handful of codes whose
        # meaning is worth knowing (the AcroForm check styles) on top of whatever the
        # /Encoding said, and leave the rest as the best guess available.
        table.update(ZAPF_DINGBATS_OVERRIDES)
    return table


def _apply_differences(table: Dict[int, str], differences: Any, resolver: Any) -> None:
    """Overlay an ``/Encoding /Differences`` array onto ``table`` in place."""
    code = 0
    for item in _sequence(differences):
        item = _resolve(item, resolver)
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            code = int(item)
            continue
        name = _name_of(item)
        if name is None:
            continue
        if 0 <= code <= 0xFFFF:
            text = glyph_to_unicode(name)
            if text:
                table[code] = text
            else:
                table.pop(code, None)
        code += 1


def _metric_widths(std_font: str, encoding: Dict[int, str]) -> Dict[int, float]:
    """Fall back to standard-14 metrics, keyed by this font's own encoding."""
    out: Dict[int, float] = {}
    for code, text in encoding.items():
        if not text:
            continue
        advance = _fonts.text_width(text[0], std_font, 1.0)
        if advance > 0.0:
            out[code] = advance
    return out


def _simple_widths(
    font: PdfDict, std_font: str, encoding: Dict[int, str], resolver: Any
) -> Tuple[Dict[int, float], float]:
    """Return ``(code -> em advance, default advance)`` for a simple font."""
    widths = _metric_widths(std_font, encoding)
    default = _fonts.text_width(" ", std_font, 1.0) or FALLBACK_ADVANCE

    descriptor = _as_dict(_resolve(font.get("FontDescriptor"), resolver))
    declared_missing = False
    if descriptor is not None and "MissingWidth" in descriptor:
        missing = _number(_resolve(descriptor.get("MissingWidth"), resolver), -1.0)
        if missing >= 0.0:
            default = missing / 1000.0
            declared_missing = True

    array = _resolve(font.get("Widths"), resolver)
    first = int(_number(_resolve(font.get("FirstChar"), resolver), 0.0))
    items = _sequence(array)
    if items:
        declared: Dict[int, float] = {}
        for offset, item in enumerate(items):
            value = _resolve(item, resolver)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            declared[first + offset] = float(value) / 1000.0
        # A zero width is legal (combining marks) but a whole table of zeros is a broken
        # producer; only trust the declared table when it carries some real advances.
        if any(value > 0.0 for value in declared.values()):
            if declared_missing:
                # The font states what an out-of-range code costs, so the standard-14
                # metric fallback must not shadow it.
                last = first + len(items) - 1
                widths = {
                    code: value
                    for code, value in widths.items()
                    if first <= code <= last
                }
            widths.update(declared)
    return widths, default


def _cid_widths(descendant: PdfDict, resolver: Any) -> Tuple[Dict[int, float], float]:
    """Return ``(cid -> em advance, default advance)`` from ``/W`` and ``/DW``."""
    default = _number(_resolve(descendant.get("DW"), resolver), 1000.0) / 1000.0
    if default <= 0.0:
        default = DEFAULT_CID_WIDTH
    widths: Dict[int, float] = {}
    array = list(_sequence(_resolve(descendant.get("W"), resolver)))
    index = 0
    count = len(array)
    while index < count:
        first = _resolve(array[index], resolver)
        index += 1
        if isinstance(first, bool) or not isinstance(first, (int, float)):
            continue
        if index >= count:
            break
        second = _resolve(array[index], resolver)
        index += 1
        if isinstance(second, (list, tuple)):
            start = int(first)
            for offset, item in enumerate(second):
                value = _resolve(item, resolver)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                widths[start + offset] = float(value) / 1000.0
            continue
        if isinstance(second, bool) or not isinstance(second, (int, float)):
            continue
        if index >= count:
            break
        third = _resolve(array[index], resolver)
        index += 1
        if isinstance(third, bool) or not isinstance(third, (int, float)):
            continue
        low = int(first)
        high = int(second)
        if high < low:
            low, high = high, low
        if high - low > 65535:
            high = low + 65535
        advance = float(third) / 1000.0
        for cid in range(low, high + 1):
            widths[cid] = advance
    return widths, default


def _composite_code_bytes(font: PdfDict, resolver: Any) -> int:
    """Decide whether a ``/Type0`` font uses one-byte or two-byte codes."""
    encoding = _resolve(font.get("Encoding"), resolver)
    if isinstance(encoding, PdfStream):
        try:
            lengths = parse_codespace_lengths(encoding.decoded(resolver))
        except Exception:  # pragma: no cover - decoded() is already lenient
            lengths = []
        if lengths:
            return 2 if max(lengths) >= 2 else 1
    # Identity-H/V and every predefined CJK CMap in common use are two-byte.
    return 2


def _descriptor_metrics(descriptor: Optional[PdfDict], std_font: str, resolver: Any) -> Tuple[float, float]:
    """Return ``(ascent, descent)`` in em units, preferring the embedded descriptor."""
    ascent = _fonts.font_ascent(std_font) / 1000.0
    descent = _fonts.font_descent(std_font) / 1000.0
    if descriptor is None:
        return ascent, descent
    declared_up = _number(_resolve(descriptor.get("Ascent"), resolver), 0.0)
    declared_down = _number(_resolve(descriptor.get("Descent"), resolver), 0.0)
    if declared_up > 0.0:
        ascent = declared_up / 1000.0
    if declared_down < 0.0:
        descent = declared_down / 1000.0
    return ascent, descent


def load_font(font_dict: Any, resolver: Any = None) -> FontProgram:
    """Build the :class:`FontProgram` for a ``/Font`` resource entry.

    Every branch degrades: a missing dictionary, an unresolvable descendant font or a
    corrupt ``/ToUnicode`` stream all end at Helvetica metrics and WinAnsi text rather
    than an exception.

    Args:
        font_dict: The resolved ``/Font`` dictionary (a :class:`PdfDict`), or anything
            else, in which case a Helvetica default is returned.
        resolver: Anything with ``.resolve`` (a :class:`~zfp.pdfio.document.Document`
            works) or a plain callable; ``None`` disables reference following.

    Returns:
        A ready :class:`FontProgram`.

    Examples:
        >>> load_font(None).std_font
        'Helvetica'
    """
    font = _as_dict(_resolve(font_dict, resolver))
    if font is None:
        program = FontProgram()
        program.encoding = base_encoding_table("WinAnsiEncoding")
        program.widths = _metric_widths("Helvetica", program.encoding)
        return program

    base_font = _name_of(_resolve(font.get("BaseFont"), resolver)) or ""
    subtype = _name_of(_resolve(font.get("Subtype"), resolver)) or ""
    std_font = _fonts.resolve_base_font(base_font)

    to_unicode: Dict[int, str] = {}
    stream = _resolve(font.get("ToUnicode"), resolver)
    if isinstance(stream, PdfStream):
        try:
            to_unicode = parse_to_unicode(stream.decoded(resolver))
        except Exception:  # pragma: no cover - decoded() and the parser are lenient
            _log.debug("font %s: unusable /ToUnicode", base_font)

    if subtype == "Type0":
        descendants = _sequence(_resolve(font.get("DescendantFonts"), resolver))
        descendant = _as_dict(_resolve(descendants[0], resolver)) if descendants else None
        if descendant is None:
            descendant = PdfDict()
        if not base_font:
            base_font = _name_of(_resolve(descendant.get("BaseFont"), resolver)) or ""
            std_font = _fonts.resolve_base_font(base_font)
        widths, default = _cid_widths(descendant, resolver)
        descriptor = _as_dict(_resolve(descendant.get("FontDescriptor"), resolver))
        ascent, descent = _descriptor_metrics(descriptor, std_font, resolver)
        return FontProgram(
            base_font=base_font,
            std_font=std_font,
            subtype=subtype,
            code_bytes=_composite_code_bytes(font, resolver),
            widths=widths,
            default_width=default,
            to_unicode=to_unicode,
            encoding={},
            ascent=ascent,
            descent=descent,
        )

    if subtype == "Type3":
        # A Type3 glyph space is defined by /FontMatrix, not by 1/1000 em.  Widths are
        # in glyph space, so scale them by the matrix's horizontal component.
        matrix = _sequence(_resolve(font.get("FontMatrix"), resolver))
        scale = _number(matrix[0], 0.001) if len(matrix) >= 6 else 0.001
        encoding = _simple_encoding(font, std_font, resolver)
        widths: Dict[int, float] = {}
        first = int(_number(_resolve(font.get("FirstChar"), resolver), 0.0))
        for offset, item in enumerate(_sequence(_resolve(font.get("Widths"), resolver))):
            value = _resolve(item, resolver)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            widths[first + offset] = float(value) * scale
        default = FALLBACK_ADVANCE if not widths else 0.0
        return FontProgram(
            base_font=base_font,
            std_font=std_font,
            subtype=subtype,
            code_bytes=1,
            widths=widths,
            default_width=default,
            to_unicode=to_unicode,
            encoding=encoding,
        )

    encoding = _simple_encoding(font, std_font, resolver)
    widths, default = _simple_widths(font, std_font, encoding, resolver)
    descriptor = _as_dict(_resolve(font.get("FontDescriptor"), resolver))
    ascent, descent = _descriptor_metrics(descriptor, std_font, resolver)
    return FontProgram(
        base_font=base_font,
        std_font=std_font,
        subtype=subtype,
        code_bytes=1,
        widths=widths,
        default_width=default,
        to_unicode=to_unicode,
        encoding=encoding,
        ascent=ascent,
        descent=descent,
    )


def decode_string(raw: bytes, font_dict: Any, resolver: Any = None) -> List[Tuple[int, str]]:
    """Split a shown string into ``(charcode, unicode)`` pairs.

    The text of a pair is ``""`` when the code cannot be mapped -- a subset ``Identity-H``
    font with no ``/ToUnicode`` is genuinely undecodable and guessing would poison label
    matching.  The code is always reported, so the caller can still advance the pen.

    Args:
        raw: The bytes of a ``Tj``/``TJ``/``'``/``"`` operand.
        font_dict: The ``/Font`` resource entry in effect (``Tf``).
        resolver: Reference resolver, or ``None``.

    Returns:
        One ``(code, text)`` pair per glyph, in show order.

    Examples:
        >>> decode_string(b"Hi", None)
        [(72, 'H'), (105, 'i')]
    """
    if not isinstance(raw, (bytes, bytearray)):
        return []
    return load_font(font_dict, resolver).decode(bytes(raw))


def font_widths(font_dict: Any, resolver: Any = None) -> Tuple[Dict[int, float], float]:
    """Return ``(code -> advance, default advance)`` for a font, in em units.

    "Em units" means the raw PDF width divided by 1000, so a glyph's horizontal
    displacement in text space is ``advance * font_size`` with no further scaling.  For
    a ``/Type0`` font the keys are CIDs, which equal the character codes under the
    ``Identity`` CMaps that dominate in practice.

    Args:
        font_dict: The ``/Font`` resource entry.
        resolver: Reference resolver, or ``None``.

    Returns:
        ``(widths, default_width)``.

    Examples:
        >>> widths, default = font_widths(None)
        >>> round(widths[ord('A')], 3)
        0.667
    """
    program = load_font(font_dict, resolver)
    return program.widths, program.default_width
