"""Adobe Core 14 (base-14) font metrics, keyed by WinAnsiEncoding code point.

This module is pure data plus the two accessors :func:`metrics_for` and
:func:`has_metrics`.  It carries the published Adobe Font Metrics (AFM) glyph
advance widths -- in 1/1000 em units, the unit PDF text space uses -- for every
font a conforming PDF viewer is required to have built in:

``Helvetica`` (regular / Bold / Oblique / BoldOblique), ``Times`` (Roman / Bold /
Italic / BoldItalic), ``Courier`` (regular / Bold / Oblique / BoldOblique),
``Symbol`` and ``ZapfDingbats``.

Widths are stored as one comma-separated literal per distinct metric set, listing
codes 32..255 in order.  An empty entry means the code is unmapped in that font's
encoding (Symbol and ZapfDingbats have gaps; the WinAnsi fonts do not, because
PDF 32000-1 Annex D.2 maps every otherwise-unused WinAnsi code above 40 to
``bullet``).  The literals expand into ``{code: width}`` dictionaries once, at
import time.

Alongside the widths each font carries its published ``Ascender``, ``Descender``,
``CapHeight``, ``XHeight``, ``FontBBox`` and ``ItalicAngle``.  ``Symbol.afm`` and
``ZapfDingbats.afm`` publish no vertical stem metrics at all; those four fields are
``None`` for them and :mod:`zfp.pdfio.fonts` substitutes a documented fallback.

Nothing here reads a file or imports a third-party library: the tables are the
dependency-free substitute for a font engine, and everything ZFP does to fit a
value into a blank rectangle is computed from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..core.errors import ValidationError

__all__ = [
    "AfmMetrics",
    "BASE_FONTS",
    "DEFAULT_BASE_FONT",
    "FIRST_CODE",
    "LAST_CODE",
    "WIN_ANSI_GLYPH_NAMES",
    "has_metrics",
    "metrics_for",
]

#: First and last WinAnsi code point covered by the width literals (inclusive).
FIRST_CODE: int = 32
LAST_CODE: int = 255

#: The face every unrecognized font name degrades to.
DEFAULT_BASE_FONT: str = "Helvetica"

#: The fourteen canonical base font names, in the order PDF 32000-1 Table 111 lists them.
BASE_FONTS: Tuple[str, ...] = (
    "Times-Roman",
    "Helvetica",
    "Courier",
    "Symbol",
    "Times-Bold",
    "Helvetica-Bold",
    "Courier-Bold",
    "ZapfDingbats",
    "Times-Italic",
    "Helvetica-Oblique",
    "Courier-Oblique",
    "Times-BoldItalic",
    "Helvetica-BoldOblique",
    "Courier-BoldOblique",
)

#: Glyph name for every WinAnsi code from :data:`FIRST_CODE` to :data:`LAST_CODE`.
#: Index ``0`` is code 32.  Unused codes carry ``"bullet"`` per Annex D.2.
WIN_ANSI_GLYPH_NAMES: Tuple[str, ...] = (
    "space", "exclam", "quotedbl", "numbersign", "dollar", "percent", "ampersand",
    "quotesingle", "parenleft", "parenright", "asterisk", "plus", "comma", "hyphen", "period",
    "slash", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "colon", "semicolon", "less", "equal", "greater", "question", "at", "A", "B", "C", "D",
    "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "bracketleft", "backslash", "bracketright", "asciicircum",
    "underscore", "grave", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "braceleft", "bar",
    "braceright", "asciitilde", "bullet", "Euro", "bullet", "quotesinglbase", "florin",
    "quotedblbase", "ellipsis", "dagger", "daggerdbl", "circumflex", "perthousand", "Scaron",
    "guilsinglleft", "OE", "bullet", "Zcaron", "bullet", "bullet", "quoteleft", "quoteright",
    "quotedblleft", "quotedblright", "bullet", "endash", "emdash", "tilde", "trademark",
    "scaron", "guilsinglright", "oe", "bullet", "zcaron", "Ydieresis", "space", "exclamdown",
    "cent", "sterling", "currency", "yen", "brokenbar", "section", "dieresis", "copyright",
    "ordfeminine", "guillemotleft", "logicalnot", "hyphen", "registered", "macron", "degree",
    "plusminus", "twosuperior", "threesuperior", "acute", "mu", "paragraph", "periodcentered",
    "cedilla", "onesuperior", "ordmasculine", "guillemotright", "onequarter", "onehalf",
    "threequarters", "questiondown", "Agrave", "Aacute", "Acircumflex", "Atilde", "Adieresis",
    "Aring", "AE", "Ccedilla", "Egrave", "Eacute", "Ecircumflex", "Edieresis", "Igrave",
    "Iacute", "Icircumflex", "Idieresis", "Eth", "Ntilde", "Ograve", "Oacute", "Ocircumflex",
    "Otilde", "Odieresis", "multiply", "Oslash", "Ugrave", "Uacute", "Ucircumflex",
    "Udieresis", "Yacute", "Thorn", "germandbls", "agrave", "aacute", "acircumflex", "atilde",
    "adieresis", "aring", "ae", "ccedilla", "egrave", "eacute", "ecircumflex", "edieresis",
    "igrave", "iacute", "icircumflex", "idieresis", "eth", "ntilde", "ograve", "oacute",
    "ocircumflex", "otilde", "odieresis", "divide", "oslash", "ugrave", "uacute",
    "ucircumflex", "udieresis", "yacute", "thorn", "ydieresis",
)

# Helvetica / Helvetica-Oblique -- advances for WinAnsi codes 32..255.
_HELVETICA = (
    "278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,"
    "556,556,556,278,278,584,584,584,556,1015,667,667,722,722,667,611,778,722,278,500,667,556,"
    "833,722,778,667,778,722,667,611,722,667,944,667,667,611,278,278,278,469,556,333,556,556,500,"
    "556,556,278,556,556,222,222,500,222,833,556,556,556,556,333,500,278,556,500,722,500,500,500,"
    "334,260,334,584,350,556,350,222,556,333,1000,556,556,333,1000,667,333,1000,350,611,350,350,"
    "333,222,333,333,350,556,1000,333,1000,500,333,944,350,500,667,278,333,556,556,556,556,260,"
    "556,333,737,370,556,584,333,737,333,400,584,333,333,333,556,537,278,333,333,365,556,834,834,"
    "834,611,667,667,667,667,667,667,1000,722,667,667,667,667,278,278,278,278,722,722,778,778,"
    "778,778,778,584,778,722,722,722,722,667,667,611,556,556,556,556,556,556,889,500,556,556,556,"
    "556,278,278,278,278,556,556,556,556,556,556,556,584,611,556,556,556,556,500,556,500"
)

# Helvetica-Bold / Helvetica-BoldOblique -- advances for WinAnsi codes 32..255.
_HELVETICA_BOLD = (
    "278,333,474,556,556,889,722,238,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,"
    "556,556,556,333,333,584,584,584,611,975,722,722,722,722,667,611,778,722,278,556,722,611,833,"
    "722,778,667,778,722,667,611,722,667,944,667,667,611,333,278,333,584,556,333,556,611,556,611,"
    "556,333,611,611,278,278,556,278,889,611,611,611,611,389,556,333,611,556,778,556,556,500,389,"
    "280,389,584,350,556,350,278,556,500,1000,556,556,333,1000,667,333,1000,350,611,350,350,278,"
    "278,500,500,350,556,1000,333,1000,556,333,944,350,500,667,278,333,556,556,556,556,280,556,"
    "333,737,370,556,584,333,737,333,400,584,333,333,333,611,556,278,333,333,365,556,834,834,834,"
    "611,722,722,722,722,722,722,1000,722,667,667,667,667,278,278,278,278,722,722,778,778,778,"
    "778,778,584,778,722,722,722,722,667,667,611,556,556,556,556,556,556,889,556,556,556,556,556,"
    "278,278,278,278,611,611,611,611,611,611,611,584,611,611,611,611,611,556,611,556"
)

# Times-Roman -- advances for WinAnsi codes 32..255.
_TIMES_ROMAN = (
    "250,333,408,500,500,833,778,180,333,333,500,564,250,333,250,278,500,500,500,500,500,500,500,"
    "500,500,500,278,278,564,564,564,444,921,722,667,667,722,611,556,722,722,333,389,722,611,889,"
    "722,722,556,722,667,556,611,722,722,944,722,722,611,333,278,333,469,500,333,444,500,444,500,"
    "444,333,500,500,278,278,500,278,778,500,500,500,500,333,389,278,500,500,722,500,500,444,480,"
    "200,480,541,350,500,350,333,500,444,1000,500,500,333,1000,556,333,889,350,611,350,350,333,"
    "333,444,444,350,500,1000,333,980,389,333,722,350,444,722,250,333,500,500,500,500,200,500,"
    "333,760,276,500,564,333,760,333,400,564,300,300,333,500,453,250,333,300,310,500,750,750,750,"
    "444,722,722,722,722,722,722,889,667,611,611,611,611,333,333,333,333,722,722,722,722,722,722,"
    "722,564,722,722,722,722,722,722,556,500,444,444,444,444,444,444,667,444,444,444,444,444,278,"
    "278,278,278,500,500,500,500,500,500,500,564,500,500,500,500,500,500,500,500"
)

# Times-Bold -- advances for WinAnsi codes 32..255.
_TIMES_BOLD = (
    "250,333,555,500,500,1000,833,278,333,333,500,570,250,333,250,278,500,500,500,500,500,500,"
    "500,500,500,500,333,333,570,570,570,500,930,722,667,722,722,667,611,778,778,389,500,778,667,"
    "944,722,778,611,778,722,556,667,722,722,1000,722,722,667,333,278,333,581,500,333,500,556,"
    "444,556,444,333,500,556,278,333,556,278,833,556,500,556,556,444,389,333,556,500,722,500,500,"
    "444,394,220,394,520,350,500,350,333,500,500,1000,500,500,333,1000,556,333,1000,350,667,350,"
    "350,333,333,500,500,350,500,1000,333,1000,389,333,722,350,444,722,250,333,500,500,500,500,"
    "220,500,333,747,300,500,570,333,747,333,400,570,300,300,333,556,540,250,333,300,330,500,750,"
    "750,750,500,722,722,722,722,722,722,1000,722,667,667,667,667,389,389,389,389,722,722,778,"
    "778,778,778,778,570,778,722,722,722,722,722,611,556,500,500,500,500,500,500,722,444,444,444,"
    "444,444,278,278,278,278,500,556,500,500,500,500,500,570,500,556,556,556,556,500,556,500"
)

# Times-Italic -- advances for WinAnsi codes 32..255.
_TIMES_ITALIC = (
    "250,333,420,500,500,833,778,214,333,333,500,675,250,333,250,278,500,500,500,500,500,500,500,"
    "500,500,500,333,333,675,675,675,500,920,611,611,667,722,611,611,722,722,333,444,667,556,833,"
    "667,722,611,722,611,500,556,722,611,833,611,556,556,389,278,389,422,500,333,500,500,444,500,"
    "444,278,500,500,278,278,444,278,722,500,500,500,500,389,389,278,500,444,667,444,444,389,400,"
    "275,400,541,350,500,350,333,500,556,889,500,500,333,1000,500,333,944,350,556,350,350,333,"
    "333,556,556,350,500,889,333,980,389,333,667,350,389,556,250,389,500,500,500,500,275,500,333,"
    "760,276,500,675,333,760,333,400,675,300,300,333,500,523,250,333,300,310,500,750,750,750,500,"
    "611,611,611,611,611,611,889,667,611,611,611,611,333,333,333,333,722,667,722,722,722,722,722,"
    "675,722,722,722,722,722,556,611,500,500,500,500,500,500,500,667,444,444,444,444,444,278,278,"
    "278,278,500,500,500,500,500,500,500,675,500,500,500,500,500,444,500,444"
)

# Times-BoldItalic -- advances for WinAnsi codes 32..255.
_TIMES_BOLD_ITALIC = (
    "250,389,555,500,500,833,778,278,333,333,500,570,250,333,250,278,500,500,500,500,500,500,500,"
    "500,500,500,333,333,570,570,570,500,832,667,667,667,722,667,667,722,778,389,500,667,611,889,"
    "722,722,611,722,667,556,611,722,667,889,667,611,611,333,278,333,570,500,333,500,500,444,500,"
    "444,333,500,556,278,278,500,278,778,556,500,500,500,389,389,278,556,444,667,500,444,389,348,"
    "220,348,570,350,500,350,333,500,500,1000,500,500,333,1000,556,333,944,350,611,350,350,333,"
    "333,500,500,350,500,1000,333,1000,389,333,722,350,389,611,250,389,500,500,500,500,220,500,"
    "333,747,266,500,606,333,747,333,400,570,300,300,333,576,500,250,333,300,300,500,750,750,750,"
    "500,667,667,667,667,667,667,944,667,667,667,667,667,389,389,389,389,722,722,722,722,722,722,"
    "722,570,722,722,722,722,722,611,611,500,500,500,500,500,500,500,722,444,444,444,444,444,278,"
    "278,278,278,500,556,500,500,500,500,500,570,500,556,556,556,556,444,500,444"
)

# Courier (all four faces are a fixed 600-unit grid) -- advances for WinAnsi codes 32..255.
_COURIER = (
    "600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,"
    "600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,"
    "600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,"
    "600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,"
    "600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,"
    "600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,"
    "600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,"
    "600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,"
    "600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,"
    "600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600,600"
)

# Symbol, in its own built-in encoding (codes 127..159 and 255 unused) -- advances for WinAnsi codes 32..255.
_SYMBOL = (
    "250,333,713,500,549,833,778,439,333,333,500,549,250,549,250,278,500,500,500,500,500,500,500,"
    "500,500,500,278,278,549,549,549,444,549,722,667,722,612,611,763,603,722,333,631,722,686,889,"
    "722,722,768,741,556,592,611,690,439,768,645,795,611,333,863,333,658,500,500,631,549,549,494,"
    "439,521,411,603,329,603,549,549,576,521,549,549,521,549,603,439,576,713,686,493,686,494,480,"
    "200,480,549,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,750,620,247,549,167,713,500,753,753,753,753,"
    "1042,987,603,987,603,400,549,411,549,549,713,494,460,549,549,549,549,1000,603,1000,658,823,"
    "686,795,987,768,768,823,768,768,713,713,713,713,713,713,713,768,713,790,790,890,823,549,250,"
    "713,603,603,1042,987,603,987,603,494,329,790,790,786,713,384,384,384,384,384,384,494,494,"
    "494,494,790,329,274,686,686,686,384,384,384,384,384,384,494,494,494,"
)

# ZapfDingbats, built-in encoding (127..160, 240 and 255 unused) -- advances for WinAnsi codes 32..255.
_ZAPF_DINGBATS = (
    "278,974,961,974,980,719,789,790,791,690,960,939,549,855,911,933,911,945,974,755,846,762,761,"
    "571,677,763,760,759,754,494,552,537,577,692,786,788,788,790,793,794,816,823,789,841,823,833,"
    "816,831,923,744,723,749,790,792,695,776,768,792,759,707,708,682,701,826,815,789,789,707,687,"
    "696,689,786,787,713,791,785,791,873,761,762,762,759,759,892,892,788,784,438,138,277,415,392,"
    "392,668,668,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,732,544,544,910,667,760,760,776,595,694,626,"
    "788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,"
    "788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,788,894,838,1016,458,748,"
    "924,748,918,927,928,928,834,873,828,924,924,917,930,931,463,883,836,836,867,867,696,696,874,"
    ",874,760,946,771,865,771,888,967,888,831,873,927,970,918,"
)


def _expand(literal: str) -> Dict[int, int]:
    """Expand a comma-separated width literal into ``{code: width}``.

    Args:
        literal: Widths for codes :data:`FIRST_CODE`..:data:`LAST_CODE` in order;
            an empty field marks a code the font does not encode.

    Returns:
        A dictionary holding only the codes the font actually encodes.

    Raises:
        ValidationError: The literal does not hold exactly ``LAST_CODE - FIRST_CODE + 1``
            fields, which would silently shift every glyph.
    """
    fields = literal.split(",")
    expected = LAST_CODE - FIRST_CODE + 1
    if len(fields) != expected:
        raise ValidationError(
            "AFM width literal must hold %d fields, got %d" % (expected, len(fields))
        )
    out: Dict[int, int] = {}
    for offset, field_text in enumerate(fields):
        if not field_text:
            continue
        out[FIRST_CODE + offset] = int(field_text)
    return out


def _average(widths: Dict[int, int]) -> int:
    """Mean advance across every encoded glyph, rounded to the nearest unit.

    Used as the stand-in advance for characters that fall outside the font's
    encoding, so measuring text with an exotic character never under-counts to zero.
    """
    if not widths:
        return 0
    return int(round(sum(widths.values()) / float(len(widths))))


@dataclass(frozen=True)
class AfmMetrics:
    """Published metrics for one base-14 face.

    Attributes:
        name: Canonical PDF base font name, e.g. ``"Helvetica-BoldOblique"``.
        widths: WinAnsi (or, for Symbol/ZapfDingbats, built-in) code -> advance width
            in 1/1000 em.
        ascender: Published ``Ascender``, or ``None`` when the AFM omits it.
        descender: Published ``Descender`` (negative), or ``None``.
        cap_height: Published ``CapHeight``, or ``None``.
        x_height: Published ``XHeight``, or ``None``.
        font_bbox: ``(llx, lly, urx, ury)`` glyph-space bounding box.
        italic_angle: Published ``ItalicAngle`` in degrees; negative slopes right.
        fixed_pitch: True only for the Courier faces, where every glyph is 600 wide.
        average_width: Mean encoded advance, the fallback for unencoded characters.
        missing_width: Advance charged to a code the font does not encode at all.
    """

    name: str
    widths: Dict[int, int]
    ascender: Optional[float]
    descender: Optional[float]
    cap_height: Optional[float]
    x_height: Optional[float]
    font_bbox: Tuple[float, float, float, float]
    italic_angle: float
    fixed_pitch: bool
    average_width: int
    missing_width: int = 0

    def width_of(self, code: int) -> int:
        """Advance width for one code, falling back to :attr:`average_width`."""
        return self.widths.get(int(code), self.average_width)


def _build(
    name: str,
    literal: str,
    ascender: Optional[float],
    descender: Optional[float],
    cap_height: Optional[float],
    x_height: Optional[float],
    font_bbox: Tuple[float, float, float, float],
    italic_angle: float,
    fixed_pitch: bool = False,
) -> AfmMetrics:
    """Expand one font's literal and wrap it with its vertical metrics."""
    widths = _expand(literal)
    return AfmMetrics(
        name=name,
        widths=widths,
        ascender=ascender,
        descender=descender,
        cap_height=cap_height,
        x_height=x_height,
        font_bbox=font_bbox,
        italic_angle=italic_angle,
        fixed_pitch=fixed_pitch,
        average_width=_average(widths),
    )


# Helvetica-Oblique repeats Helvetica's advances exactly (the oblique is a shear, not a
# redraw); likewise Helvetica-BoldOblique repeats Helvetica-Bold, and each Courier face
# repeats the 600-unit grid.  Only the bounding box and italic angle differ.
_METRICS: Dict[str, AfmMetrics] = {
    "Helvetica": _build(
        "Helvetica", _HELVETICA, 718.0, -207.0, 718.0, 523.0, (-166.0, -225.0, 1000.0, 931.0), 0.0
    ),
    "Helvetica-Bold": _build(
        "Helvetica-Bold", _HELVETICA_BOLD, 718.0, -207.0, 718.0, 532.0,
        (-170.0, -228.0, 1003.0, 962.0), 0.0,
    ),
    "Helvetica-Oblique": _build(
        "Helvetica-Oblique", _HELVETICA, 718.0, -207.0, 718.0, 523.0,
        (-170.0, -225.0, 1116.0, 931.0), -12.0,
    ),
    "Helvetica-BoldOblique": _build(
        "Helvetica-BoldOblique", _HELVETICA_BOLD, 718.0, -207.0, 718.0, 532.0,
        (-174.0, -228.0, 1114.0, 962.0), -12.0,
    ),
    "Times-Roman": _build(
        "Times-Roman", _TIMES_ROMAN, 683.0, -217.0, 662.0, 450.0,
        (-168.0, -218.0, 1000.0, 898.0), 0.0,
    ),
    "Times-Bold": _build(
        "Times-Bold", _TIMES_BOLD, 683.0, -217.0, 676.0, 461.0,
        (-168.0, -218.0, 1000.0, 935.0), 0.0,
    ),
    "Times-Italic": _build(
        "Times-Italic", _TIMES_ITALIC, 683.0, -217.0, 653.0, 441.0,
        (-169.0, -217.0, 1010.0, 883.0), -15.5,
    ),
    "Times-BoldItalic": _build(
        "Times-BoldItalic", _TIMES_BOLD_ITALIC, 683.0, -217.0, 669.0, 462.0,
        (-200.0, -218.0, 996.0, 921.0), -15.0,
    ),
    "Courier": _build(
        "Courier", _COURIER, 629.0, -157.0, 562.0, 426.0, (-6.0, -249.0, 639.0, 803.0),
        0.0, True,
    ),
    "Courier-Bold": _build(
        "Courier-Bold", _COURIER, 629.0, -157.0, 562.0, 439.0,
        (-88.0, -249.0, 697.0, 811.0), 0.0, True,
    ),
    "Courier-Oblique": _build(
        "Courier-Oblique", _COURIER, 629.0, -157.0, 562.0, 426.0,
        (-27.0, -250.0, 849.0, 805.0), -12.0, True,
    ),
    "Courier-BoldOblique": _build(
        "Courier-BoldOblique", _COURIER, 629.0, -157.0, 562.0, 439.0,
        (-100.0, -250.0, 859.0, 811.0), -12.0, True,
    ),
    # Symbol.afm and ZapfDingbats.afm publish FontBBox and ItalicAngle but no
    # Ascender/Descender/CapHeight/XHeight -- hence the four Nones.
    "Symbol": _build(
        "Symbol", _SYMBOL, None, None, None, None, (-180.0, -293.0, 1090.0, 1010.0), 0.0
    ),
    "ZapfDingbats": _build(
        "ZapfDingbats", _ZAPF_DINGBATS, None, None, None, None,
        (-1.0, -143.0, 981.0, 820.0), 0.0,
    ),
}


def has_metrics(base_font: str) -> bool:
    """True when ``base_font`` is one of the fourteen canonical base font names."""
    return base_font in _METRICS


def metrics_for(base_font: str) -> AfmMetrics:
    """Return the :class:`AfmMetrics` for a canonical base font name.

    Args:
        base_font: An exact name from :data:`BASE_FONTS`.  Callers that hold a
            user-supplied name should run it through
            :func:`zfp.pdfio.fonts.resolve_base_font` first.

    Returns:
        The shared, immutable metrics record for that face.

    Raises:
        ValidationError: ``base_font`` is not a canonical base-14 name.
    """
    try:
        return _METRICS[base_font]
    except KeyError:
        raise ValidationError(
            "%r is not a base-14 font name; expected one of %s"
            % (base_font, ", ".join(BASE_FONTS))
        ) from None
