"""Base-14 font metrics, text measurement, fitting and PDF text escaping.

This is ZFP's substitute for a font engine.  Every appearance stream the writer
produces has to answer three questions without a third-party library:

* how wide is this string at this size (:func:`text_width`, :func:`char_widths`),
* what is the largest size that still fits this blank rectangle
  (:func:`fit_font_size`, :func:`wrap_text`, :func:`measure_lines`),
* how do I put the string in a content stream (:func:`escape_pdf_text`) and how do
  I make the font available to the form (:func:`ensure_standard_font`).

All of it is computed from the published Adobe Core 14 metrics in
:mod:`zfp.pdfio._afm_data`, so the answers are byte-identical to what a viewer
computes from its own built-in copies of those fonts.

Two conventions run through the module:

*Advance units.*  AFM advances are in 1/1000 em.  Anything returning points
(:func:`text_width`, :func:`char_widths`, :func:`measure_lines`) has already been
multiplied by ``size / 1000``.  :func:`font_ascent` and :func:`font_descent` are the
exception: they return the *published* 1/1000-em values (718.0 and -207.0 for
Helvetica), so scale them yourself with ``size / 1000``.

*Encoding.*  Text is measured and escaped as WinAnsiEncoding, the encoding ZFP asks
for whenever it embeds a standard font.  A character WinAnsi cannot represent is
charged the font's average advance when measuring and written as ``?`` when escaping.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from ..core.geometry import EPS, Rect
from ._afm_data import (
    BASE_FONTS,
    DEFAULT_BASE_FONT,
    FIRST_CODE,
    LAST_CODE,
    AfmMetrics,
    metrics_for,
)

if TYPE_CHECKING:  # pragma: no cover - imports for type checkers only
    from .document import Document
    from .objects import PdfRef

__all__ = [
    "LEADING_FACTOR",
    "FALLBACK_ASCENDER",
    "FALLBACK_DESCENDER",
    "STANDARD_14",
    "RESOURCE_NAMES",
    "resolve_base_font",
    "resource_name",
    "is_fixed_pitch",
    "widths_for",
    "text_width",
    "char_widths",
    "measure_lines",
    "font_ascent",
    "font_descent",
    "fit_font_size",
    "wrap_text",
    "escape_pdf_text",
    "ensure_standard_font",
]

#: Baseline-to-baseline distance as a multiple of the font size.  1.16 is the ratio
#: Acrobat uses for auto-sized form text and what ZFP lays multiline fields out on.
LEADING_FACTOR: float = 1.16

#: Substituted for Symbol and ZapfDingbats, whose AFM files publish no vertical stem
#: metrics.  These are Helvetica's values, which is what viewers effectively lay
#: dingbat runs out on.
FALLBACK_ASCENDER: float = 718.0
FALLBACK_DESCENDER: float = -207.0


# --------------------------------------------------------------------------------------
# Name resolution
# --------------------------------------------------------------------------------------

#: Alias -> canonical base font name.  Covers the four-letter resource names Acrobat
#: writes into ``/DR``, the PostScript names, and the Arial / Times New Roman /
#: Courier New substitutes that flat forms name instead of the base-14 originals.
STANDARD_14: Dict[str, str] = {
    # -- Helvetica -------------------------------------------------------------------
    "Helv": "Helvetica",
    "Helvetica": "Helvetica",
    "Helvetica-Regular": "Helvetica",
    "Arial": "Helvetica",
    "ArialMT": "Helvetica",
    "Arial-Regular": "Helvetica",
    "Sans": "Helvetica",
    "HeBo": "Helvetica-Bold",
    "Helvetica-Bold": "Helvetica-Bold",
    "Arial-Bold": "Helvetica-Bold",
    "Arial,Bold": "Helvetica-Bold",
    "Arial-BoldMT": "Helvetica-Bold",
    "HeOb": "Helvetica-Oblique",
    "Helvetica-Oblique": "Helvetica-Oblique",
    "Helvetica-Italic": "Helvetica-Oblique",
    "Arial-Italic": "Helvetica-Oblique",
    "Arial,Italic": "Helvetica-Oblique",
    "Arial-ItalicMT": "Helvetica-Oblique",
    "HeBO": "Helvetica-BoldOblique",
    "Helvetica-BoldOblique": "Helvetica-BoldOblique",
    "Helvetica-BoldItalic": "Helvetica-BoldOblique",
    "Arial-BoldItalic": "Helvetica-BoldOblique",
    "Arial,BoldItalic": "Helvetica-BoldOblique",
    "Arial-BoldItalicMT": "Helvetica-BoldOblique",
    # -- Times -----------------------------------------------------------------------
    "TiRo": "Times-Roman",
    "Times": "Times-Roman",
    "Times-Roman": "Times-Roman",
    "TimesNewRoman": "Times-Roman",
    "TimesNewRomanPSMT": "Times-Roman",
    "Serif": "Times-Roman",
    "TiBo": "Times-Bold",
    "Times-Bold": "Times-Bold",
    "TimesNewRoman-Bold": "Times-Bold",
    "TimesNewRoman,Bold": "Times-Bold",
    "TimesNewRomanPS-BoldMT": "Times-Bold",
    "TiIt": "Times-Italic",
    "Times-Italic": "Times-Italic",
    "TimesNewRoman-Italic": "Times-Italic",
    "TimesNewRoman,Italic": "Times-Italic",
    "TimesNewRomanPS-ItalicMT": "Times-Italic",
    "TiBI": "Times-BoldItalic",
    "Times-BoldItalic": "Times-BoldItalic",
    "TimesNewRoman-BoldItalic": "Times-BoldItalic",
    "TimesNewRoman,BoldItalic": "Times-BoldItalic",
    "TimesNewRomanPS-BoldItalicMT": "Times-BoldItalic",
    # -- Courier ---------------------------------------------------------------------
    "Cour": "Courier",
    "Courier": "Courier",
    "CourierNew": "Courier",
    "CourierNewPSMT": "Courier",
    "Mono": "Courier",
    "Monospace": "Courier",
    "CoBo": "Courier-Bold",
    "Courier-Bold": "Courier-Bold",
    "CourierNew-Bold": "Courier-Bold",
    "CourierNew,Bold": "Courier-Bold",
    "CourierNewPS-BoldMT": "Courier-Bold",
    "CoOb": "Courier-Oblique",
    "Courier-Oblique": "Courier-Oblique",
    "Courier-Italic": "Courier-Oblique",
    "CourierNew-Italic": "Courier-Oblique",
    "CourierNew,Italic": "Courier-Oblique",
    "CourierNewPS-ItalicMT": "Courier-Oblique",
    "CoBO": "Courier-BoldOblique",
    "Courier-BoldOblique": "Courier-BoldOblique",
    "Courier-BoldItalic": "Courier-BoldOblique",
    "CourierNew-BoldItalic": "Courier-BoldOblique",
    "CourierNew,BoldItalic": "Courier-BoldOblique",
    "CourierNewPS-BoldItalicMT": "Courier-BoldOblique",
    # -- pi fonts --------------------------------------------------------------------
    "Symb": "Symbol",
    "Symbol": "Symbol",
    "ZaDb": "ZapfDingbats",
    "ZapfDingbats": "ZapfDingbats",
    "Dingbats": "ZapfDingbats",
    "Zapf Dingbats": "ZapfDingbats",
}

#: Canonical base font name -> the short name it is filed under in ``/DR /Font``.
#: These are the names Acrobat itself uses, so a form ZFP writes stays editable in it.
RESOURCE_NAMES: Dict[str, str] = {
    "Helvetica": "Helv",
    "Helvetica-Bold": "HeBo",
    "Helvetica-Oblique": "HeOb",
    "Helvetica-BoldOblique": "HeBO",
    "Times-Roman": "TiRo",
    "Times-Bold": "TiBo",
    "Times-Italic": "TiIt",
    "Times-BoldItalic": "TiBI",
    "Courier": "Cour",
    "Courier-Bold": "CoBo",
    "Courier-Oblique": "CoOb",
    "Courier-BoldOblique": "CoBO",
    "Symbol": "Symb",
    "ZapfDingbats": "ZaDb",
}

#: Fonts that must not be given ``/WinAnsiEncoding``: their glyphs live in a built-in
#: encoding of their own and re-encoding them yields the wrong (or no) glyph.
_BUILTIN_ENCODING_FONTS = frozenset(("Symbol", "ZapfDingbats"))

_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")

# Deliberately conservative: a marker only earns its place if it cannot show up
# inside an unrelated family name (which rules out bare "it" -- cf. "Bitstream").
_BOLD_MARKERS = ("bold", "black", "heavy", "semibold", "demibold")
_ITALIC_MARKERS = ("italic", "oblique", "slanted")


def _normalize(name: str) -> str:
    """Fold a font name to its comparison key: no subset tag, no case, no separators."""
    return _NON_ALNUM.sub("", _SUBSET_PREFIX.sub("", name.strip()).lower())


_ALIAS_INDEX: Dict[str, str] = {}
for _alias, _canonical in STANDARD_14.items():
    _ALIAS_INDEX.setdefault(_normalize(_alias), _canonical)
for _canonical in BASE_FONTS:
    _ALIAS_INDEX.setdefault(_normalize(_canonical), _canonical)


def _infer_family(key: str) -> str:
    """Guess a base-14 family from an unknown font name's normalized key."""
    if "zapf" in key or "dingbat" in key:
        return "ZapfDingbats"
    if "symbol" in key:
        return "Symbol"
    if "courier" in key or "mono" in key or "consol" in key or "typewriter" in key:
        return "Courier"
    for marker in ("times", "serif", "roman", "georgia", "garamond", "minion", "book"):
        if marker in key and "sansserif" not in key:
            return "Times"
    return "Helvetica"


def _compose(family: str, bold: bool, italic: bool) -> str:
    """Assemble a canonical base-14 name from a family plus style flags."""
    if family in _BUILTIN_ENCODING_FONTS:
        return family                       # neither has a bold or italic cut
    if family == "Times":
        if bold and italic:
            return "Times-BoldItalic"
        if bold:
            return "Times-Bold"
        if italic:
            return "Times-Italic"
        return "Times-Roman"
    slant = "Oblique"
    if bold and italic:
        return "%s-Bold%s" % (family, slant)
    if bold:
        return "%s-Bold" % family
    if italic:
        return "%s-%s" % (family, slant)
    return family


def resolve_base_font(name: str) -> str:
    """Map any font name onto one of the fourteen canonical base font names.

    Resolution runs in three steps: an exact hit in :data:`STANDARD_14`, then a hit on
    the normalized key (case, subset tag such as ``ABCDEF+`` and separators removed),
    then a style inference over the remaining text -- family from ``times`` /
    ``courier`` / ``symbol`` / ``dingbat`` substrings, weight and slant from markers
    like ``Bold``, ``Black``, ``Italic`` or ``Oblique``.  Anything still unrecognized
    degrades to ``"Helvetica"``; this function never raises.

    Args:
        name: A font name of any provenance -- ``/BaseFont`` value, ``/DR`` resource
            name, a ``/DA`` string's font, or a user-supplied label.

    Returns:
        A member of :data:`~zfp.pdfio._afm_data.BASE_FONTS`.

    Examples:
        >>> resolve_base_font("Helv")
        'Helvetica'
        >>> resolve_base_font("ABCDEF+Arial,BoldItalic")
        'Helvetica-BoldOblique'
        >>> resolve_base_font("Wingdings")
        'Helvetica'
    """
    if not isinstance(name, str) or not name.strip():
        return DEFAULT_BASE_FONT
    direct = STANDARD_14.get(name.strip())
    if direct is not None:
        return direct
    key = _normalize(name)
    if not key:
        return DEFAULT_BASE_FONT
    hit = _ALIAS_INDEX.get(key)
    if hit is not None:
        return hit
    family = _infer_family(key)
    bold = any(marker in key for marker in _BOLD_MARKERS)
    italic = any(marker in key for marker in _ITALIC_MARKERS)
    return _compose(family, bold, italic)


def resource_name(base_font: str) -> str:
    """Return the short ``/DR /Font`` key for a font name, e.g. ``"Helv"``."""
    return RESOURCE_NAMES[resolve_base_font(base_font)]


def is_fixed_pitch(name: str) -> bool:
    """True when ``name`` resolves to a monospaced face (the four Courier cuts)."""
    return metrics_for(resolve_base_font(name)).fixed_pitch


_METRICS_CACHE: Dict[str, AfmMetrics] = {}


def _metrics(name: str) -> AfmMetrics:
    """Resolve ``name`` and return its metrics, memoizing the resolution."""
    cached = _METRICS_CACHE.get(name)
    if cached is None:
        cached = metrics_for(resolve_base_font(name))
        _METRICS_CACHE[name] = cached
    return cached


# --------------------------------------------------------------------------------------
# WinAnsi encoding
# --------------------------------------------------------------------------------------

def _build_char_to_code() -> Dict[str, int]:
    """Build the character -> WinAnsi code table from the cp1252 codec.

    cp1252 *is* WinAnsiEncoding for every code that maps to a character, so the codec
    is the authority here rather than a hand-typed table.  Codes 0x81, 0x8D, 0x8F,
    0x90, 0x9D are unmapped in both.  Code 127 is skipped: WinAnsi draws a bullet
    there, but U+007F is a control character and must not silently become one.
    """
    table: Dict[str, int] = {}
    for code in range(FIRST_CODE, LAST_CODE + 1):
        if code == 127:
            continue
        try:
            ch = bytes((code,)).decode("cp1252")
        except UnicodeDecodeError:
            continue
        table.setdefault(ch, code)
    return table


_CHAR_TO_CODE: Dict[str, int] = _build_char_to_code()

#: Characters WinAnsi cannot hold, mapped to the closest run of characters it can.
#: An empty replacement means the character is zero-width and simply disappears.
_CHAR_FALLBACK: Dict[str, str] = {
    "\u2010": "-",      # hyphen
    "\u2011": "-",      # non-breaking hyphen
    "\u2012": "-",      # figure dash
    "\u2015": "\u2014",  # horizontal bar -> em dash
    "\u2212": "-",      # minus sign
    "\u2002": " ",      # en space
    "\u2003": " ",      # em space
    "\u2004": " ",      # three-per-em space
    "\u2005": " ",      # four-per-em space
    "\u2006": " ",      # six-per-em space
    "\u2007": " ",      # figure space
    "\u2008": " ",      # punctuation space
    "\u2009": " ",      # thin space
    "\u200a": " ",      # hair space
    "\u202f": " ",      # narrow no-break space
    "\u205f": " ",      # medium mathematical space
    "\u3000": " ",      # ideographic space
    "\u200b": "",       # zero width space
    "\u200c": "",       # zero width non-joiner
    "\u200d": "",       # zero width joiner
    "\ufeff": "",       # zero width no-break space / BOM
    "\u2032": "'",      # prime
    "\u2033": '"',      # double prime
    "\u2035": "'",      # reversed prime
    "\u2044": "/",      # fraction slash
    "\ufb01": "fi",     # fi ligature
    "\ufb02": "fl",     # fl ligature
    "\u0130": "I",      # dotted capital I
    "\u0131": "i",      # dotless i
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
}

#: Control characters that get their own backslash escape inside a PDF literal string.
_CONTROL_ESCAPES: Dict[str, bytes] = {
    "\n": b"\\n",
    "\r": b"\\r",
    "\t": b"\\t",
    "\b": b"\\b",
    "\x0c": b"\\f",
}

_BACKSLASH_ESCAPES: Dict[int, bytes] = {
    0x28: b"\\(",
    0x29: b"\\)",
    0x5C: b"\\\\",
}

_CODES_CACHE: Dict[str, Optional[Tuple[int, ...]]] = {}


def _codes_for(ch: str) -> Optional[Tuple[int, ...]]:
    """WinAnsi codes for one character.

    Returns:
        A tuple of codes (possibly empty for a zero-width character), or ``None`` when
        the character has no WinAnsi representation at all.
    """
    cached = _CODES_CACHE.get(ch)
    if cached is not None or ch in _CODES_CACHE:
        return cached
    result: Optional[Tuple[int, ...]]
    if ord(ch) < 32 or ord(ch) == 127:
        result = ()                     # control characters carry no advance
    else:
        code = _CHAR_TO_CODE.get(ch)
        if code is not None:
            result = (code,)
        else:
            replacement = _CHAR_FALLBACK.get(ch)
            if replacement is None:
                result = None
            else:
                codes = [_CHAR_TO_CODE[c] for c in replacement if c in _CHAR_TO_CODE]
                result = tuple(codes)
    _CODES_CACHE[ch] = result
    return result


def _advance(ch: str, m: AfmMetrics) -> int:
    """Advance width of one character in 1/1000 em for the given face.

    Tab counts as one space, because a tab in a form value is nearly always standing
    in for horizontal padding.  Other control characters are zero.  A character with
    no WinAnsi representation is charged the font's average advance so an exotic
    string never measures as narrower than it will draw.
    """
    if ch == "\t":
        return m.width_of(32)
    codes = _codes_for(ch)
    if codes is None:
        return m.average_width
    total = 0
    for code in codes:
        total += m.width_of(code)
    return total


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------

def widths_for(base_font: str) -> Dict[int, int]:
    """Return ``{code point: advance width in 1/1000 em}`` for a font.

    The keys are WinAnsi code points for the twelve text faces and built-in encoding
    code points for Symbol and ZapfDingbats.  The result is a fresh dictionary, so
    callers may mutate it without disturbing the shared tables.

    Args:
        base_font: Any font name; resolved through :func:`resolve_base_font`.

    Returns:
        A new dictionary mapping each encoded code to its advance.

    Examples:
        >>> widths_for("Helvetica")[65]
        667
        >>> sorted(set(widths_for("Courier").values()))
        [600]
    """
    return dict(_metrics(base_font).widths)


def text_width(text: str, base_font: str, size: float) -> float:
    """Width of ``text`` in points when set in ``base_font`` at ``size``.

    Args:
        text: The string to measure; may be empty.
        base_font: Any font name; resolved through :func:`resolve_base_font`.
        size: Font size in points.

    Returns:
        The summed advance in points.  Empty text is exactly ``0.0``.

    Examples:
        >>> text_width("", "Helvetica", 12.0)
        0.0
        >>> round(text_width("A", "Helvetica", 12.0), 4)
        8.004
    """
    if not text:
        return 0.0
    m = _metrics(base_font)
    total = 0
    for ch in text:
        total += _advance(ch, m)
    return total * float(size) / 1000.0


def char_widths(text: str, base_font: str, size: float) -> List[float]:
    """Per-character advance widths in points, one entry per character of ``text``.

    The comb-field appearance writer needs the individual advances to centre each
    glyph in its cell, which a single total cannot give it.

    Args:
        text: The string to measure.
        base_font: Any font name; resolved through :func:`resolve_base_font`.
        size: Font size in points.

    Returns:
        A list the same length as ``text``.  ``sum()`` of it equals
        :func:`text_width` of the same arguments up to floating point.
    """
    if not text:
        return []
    m = _metrics(base_font)
    scale = float(size) / 1000.0
    return [_advance(ch, m) * scale for ch in text]


def measure_lines(lines: Sequence[str], base_font: str, size: float) -> Tuple[float, float]:
    """Bounding size of a block of already-wrapped lines.

    Args:
        lines: The lines, in order; may be empty.
        base_font: Any font name; resolved through :func:`resolve_base_font`.
        size: Font size in points.

    Returns:
        ``(max_width, total_height)`` in points, where the height is
        ``len(lines) * size * LEADING_FACTOR``.  An empty sequence gives
        ``(0.0, 0.0)``.
    """
    widest = 0.0
    count = 0
    for line in lines:
        count += 1
        width = text_width(line, base_font, size)
        if width > widest:
            widest = width
    if count == 0:
        return (0.0, 0.0)
    return (widest, count * float(size) * LEADING_FACTOR)


def font_ascent(base_font: str) -> float:
    """Published ``Ascender`` for a font, in 1/1000 em (Helvetica: ``718.0``).

    Multiply by ``size / 1000`` for points.  Symbol and ZapfDingbats publish no
    ascender; they return :data:`FALLBACK_ASCENDER`.
    """
    value = _metrics(base_font).ascender
    return FALLBACK_ASCENDER if value is None else float(value)


def font_descent(base_font: str) -> float:
    """Published ``Descender`` for a font, in 1/1000 em and negative (Helvetica: ``-207.0``).

    Multiply by ``size / 1000`` for points.  Symbol and ZapfDingbats publish no
    descender; they return :data:`FALLBACK_DESCENDER`.
    """
    value = _metrics(base_font).descender
    return FALLBACK_DESCENDER if value is None else float(value)


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------

def fit_font_size(
    text: str,
    base_font: str,
    rect: Rect,
    *,
    max_size: float = 12.0,
    min_size: float = 4.0,
    padding: float = 2.0,
) -> float:
    """Largest size in ``[min_size, max_size]`` at which ``text`` fits ``rect``.

    Both axes constrain the answer: the string's advance must fit
    ``rect.width - 2 * padding`` and one line of leading (``size * LEADING_FACTOR``)
    must fit ``rect.height - 2 * padding``.  The search runs over hundredths of a
    point, so the result is exactly representable and the same on every run.

    Args:
        text: The value that has to sit in the rectangle; empty text is constrained
            by height alone.
        base_font: Any font name; resolved through :func:`resolve_base_font`.
        rect: The blank the value has to land in, in PDF user space.  Normalized
            internally, so a rectangle with swapped corners is handled.
        max_size: Upper bound; never exceeded.
        min_size: Lower bound, and the answer when nothing fits -- an over-long value
            is drawn small and clipped rather than silently dropped.
        padding: Inset applied to *both* sides of *both* axes.

    Returns:
        A size in points rounded to two decimals, always within the bounds.

    Examples:
        >>> box = Rect(0.0, 0.0, 200.0, 20.0)
        >>> fit_font_size("Jane", "Helvetica", box)
        12.0
        >>> fit_font_size("Jane Q. Public, 1234 Long Winding Street", "Helvetica", box) < 12.0
        True
    """
    low_bound = round(float(min_size), 2)
    high_bound = round(float(max_size), 2)
    if high_bound < low_bound:
        low_bound, high_bound = high_bound, low_bound

    box = rect.normalized()
    pad = float(padding)
    avail_w = box.width - 2.0 * pad
    avail_h = box.height - 2.0 * pad
    base = resolve_base_font(base_font)

    def fits(size: float) -> bool:
        if size * LEADING_FACTOR > avail_h + EPS:
            return False
        return text_width(text, base, size) <= avail_w + EPS

    if not fits(low_bound):
        return low_bound
    if fits(high_bound):
        return high_bound

    # Binary search the hundredth-of-a-point grid.  `fits` is monotonically
    # decreasing in size, so the invariant "low fits, high+1 does not" holds.
    low = int(round(low_bound * 100.0))
    high = int(round(high_bound * 100.0))
    while low < high:
        mid = (low + high + 1) // 2
        if fits(mid / 100.0):
            low = mid
        else:
            high = mid - 1
    return round(low / 100.0, 2)


_TOKEN_RE = re.compile(r"\s+|\S+")


def _hard_split(word: str, m: AfmMetrics, size: float, width: float) -> List[str]:
    """Break a single unbreakable run into pieces that each fit ``width``.

    Always makes progress: a character wider than ``width`` on its own gets a line to
    itself rather than looping.  ``"".join(result)`` is exactly ``word``.
    """
    scale = float(size) / 1000.0
    pieces: List[str] = []
    current = ""
    current_w = 0.0
    for ch in word:
        advance = _advance(ch, m) * scale
        if current and current_w + advance > width + EPS:
            pieces.append(current)
            current = ch
            current_w = advance
        else:
            current += ch
            current_w += advance
    pieces.append(current)
    return pieces


def _wrap_paragraph(para: str, m: AfmMetrics, size: float, width: float) -> List[str]:
    """Greedy-wrap one newline-free paragraph into lines no wider than ``width``."""
    if para == "":
        return [""]
    scale = float(size) / 1000.0

    def measure(chunk: str) -> float:
        total = 0
        for ch in chunk:
            total += _advance(ch, m)
        return total * scale

    lines: List[str] = []
    current = ""
    current_w = 0.0
    pending = ""            # whitespace held back; dropped only at a line break
    pending_w = 0.0

    for token in _TOKEN_RE.findall(para):
        token_w = measure(token)
        if token.isspace():
            pending += token
            pending_w += token_w
            continue

        if current == "":
            # Start of a paragraph: keep any leading indent attached to the word.
            candidate = pending + token
            candidate_w = pending_w + token_w
            pending, pending_w = "", 0.0
            if candidate_w <= width + EPS:
                current, current_w = candidate, candidate_w
            else:
                pieces = _hard_split(candidate, m, size, width)
                lines.extend(pieces[:-1])
                current = pieces[-1]
                current_w = measure(current)
            continue

        if current_w + pending_w + token_w <= width + EPS:
            current += pending + token
            current_w += pending_w + token_w
            pending, pending_w = "", 0.0
            continue

        lines.append(current)
        pending, pending_w = "", 0.0        # the break consumes the separator
        if token_w <= width + EPS:
            current, current_w = token, token_w
        else:
            pieces = _hard_split(token, m, size, width)
            lines.extend(pieces[:-1])
            current = pieces[-1]
            current_w = measure(current)

    if pending:
        current += pending                  # trailing whitespace is never discarded
    lines.append(current)
    return lines


def wrap_text(text: str, base_font: str, size: float, width: float) -> List[str]:
    """Greedy word wrap of ``text`` to lines no wider than ``width`` points.

    Explicit line breaks are honoured: ``\\r\\n`` and ``\\r`` are normalized to
    ``\\n``, then each paragraph is wrapped on its own, so a blank line stays a blank
    line.  Nothing but the whitespace consumed at a break point is lost -- every word,
    every interior space run and any leading or trailing whitespace survives.  A word
    too long for a line is hard-split across lines rather than overflowing.

    Args:
        text: The value to lay out; may be empty.
        base_font: Any font name; resolved through :func:`resolve_base_font`.
        size: Font size in points.
        width: Line width in points.  A non-positive width has no wrap points, so the
            paragraphs are returned unwrapped.

    Returns:
        The lines, in order.  Always at least one entry (``[""]`` for empty text).

    Examples:
        >>> wrap_text("a\\nb", "Helvetica", 10.0, 100.0)
        ['a', 'b']
        >>> len(wrap_text("one two three four five", "Helvetica", 10.0, 40.0)) > 1
        True
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = normalized.split("\n")
    if width <= 0.0:
        return paragraphs
    m = _metrics(base_font)
    out: List[str] = []
    for para in paragraphs:
        out.extend(_wrap_paragraph(para, m, size, float(width)))
    return out


# --------------------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------------------

def escape_pdf_text(s: str) -> bytes:
    """Encode ``s`` as WinAnsi and escape it for a PDF literal string.

    The result is the bytes that go *between* the parentheses of a ``(...) Tj``
    operand.  ``(``, ``)`` and ``\\`` get backslash escapes; the five control
    characters PDF names get theirs (``\\n \\r \\t \\b \\f``); every other byte below
    32 or at/above 127 is written as a three-digit octal escape, so the operand stays
    inside printable ASCII and survives tools that reflow content streams.  A
    character with no WinAnsi representation becomes ``?``.

    Args:
        s: The text to embed.

    Returns:
        Escaped bytes, ready to be wrapped in parentheses.

    Examples:
        >>> escape_pdf_text("a(b)c")
        b'a\\\\(b\\\\)c'
        >>> escape_pdf_text("caf\\u00e9")
        b'caf\\\\351'
    """
    out = bytearray()
    for ch in s:
        point = ord(ch)
        if point < 32 or point == 127:
            control = _CONTROL_ESCAPES.get(ch)
            out += control if control is not None else b"\\%03o" % point
            continue
        codes = _codes_for(ch)
        if codes is None:
            codes = (0x3F,)                 # '?'
        for code in codes:
            escape = _BACKSLASH_ESCAPES.get(code)
            if escape is not None:
                out += escape
            elif code < 32 or code >= 127:
                out += b"\\%03o" % code
            else:
                out.append(code)
    return bytes(out)


def ensure_standard_font(doc: "Document", base_font: str = "Helvetica") -> Tuple[str, "PdfRef"]:
    """Make a base-14 font available in the document's AcroForm resources.

    Adds ``/Type /Font /Subtype /Type1 /BaseFont <name>`` (plus
    ``/Encoding /WinAnsiEncoding``, except for Symbol and ZapfDingbats, whose built-in
    encodings must not be overridden) to ``/AcroForm /DR /Font`` under its standard
    short name, and stages the enclosing object with the writer so an incremental save
    persists it.

    The call is idempotent: if the short name already points at a font dictionary with
    the right ``/BaseFont``, that existing reference is returned and no object is
    allocated.

    Args:
        doc: The open document.  Its AcroForm is created if it has none.
        base_font: Any font name; resolved through :func:`resolve_base_font`.

    Returns:
        ``(short_name, ref)`` -- e.g. ``("Helv", PdfRef(12, 0))``.  ``short_name`` is
        what a ``/DA`` string names, ``ref`` points at the font dictionary.

    Raises:
        PdfWriteError: The AcroForm could not be created or staged for writing.
    """
    # Imported here, not at module scope: the object layer pulls in the parser and
    # writer, and `fonts` must stay importable (and cheap) on its own.
    from .objects import PdfDict, PdfName, PdfRef

    canonical = resolve_base_font(base_font)
    short = RESOURCE_NAMES[canonical]

    acroform = doc.ensure_acroform()
    acroform_ref = getattr(doc, "acroform_ref", None)

    raw_dr = acroform.get("DR")
    dr_ref = raw_dr if isinstance(raw_dr, PdfRef) else None
    resources = doc.resolve(raw_dr) if raw_dr is not None else None
    if not isinstance(resources, PdfDict):
        resources = PdfDict()
        acroform["DR"] = resources
        dr_ref = None

    raw_fonts = resources.get("Font")
    font_dir_ref = raw_fonts if isinstance(raw_fonts, PdfRef) else None
    fonts = doc.resolve(raw_fonts) if raw_fonts is not None else None
    if not isinstance(fonts, PdfDict):
        fonts = PdfDict()
        resources["Font"] = fonts
        font_dir_ref = None

    existing = fonts.get(short)
    if isinstance(existing, PdfRef):
        current = doc.resolve(existing)
        if isinstance(current, PdfDict) and current.get_name("BaseFont") == canonical:
            return (short, existing)

    font_dict = PdfDict(
        {
            "Type": PdfName("Font"),
            "Subtype": PdfName("Type1"),
            "BaseFont": PdfName(canonical),
        }
    )
    if canonical not in _BUILTIN_ENCODING_FONTS:
        font_dict["Encoding"] = PdfName("WinAnsiEncoding")
    ref = doc.writer.add_object(font_dict)
    fonts[short] = ref

    # Stage the outermost object that actually changed.  Resolving an indirect object
    # can hand back a fresh instance every time, so mutating in place is not enough.
    if font_dir_ref is not None:
        doc.writer.set_object(font_dir_ref.num, fonts)
    elif dr_ref is not None:
        doc.writer.set_object(dr_ref.num, resources)
    elif acroform_ref is not None:
        doc.writer.set_object(acroform_ref.num, acroform)
    return (short, ref)
