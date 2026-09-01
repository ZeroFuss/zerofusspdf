"""A byte-level tokenizer for PDF syntax.

The lexer is deliberately dumb and deliberately unbreakable.  It knows nothing about
objects, xrefs or streams; it turns bytes into :class:`Token` values and guarantees two
things that everything above it depends on:

* every call to :meth:`Lexer.next_token` either advances ``pos`` or returns ``eof``, so
  no malformed input can spin the parser in a loop;
* :meth:`Lexer.peek` never consumes, and assigning to ``pos`` invalidates the peeked
  token, so a parser can seek freely.

Token values are primitives, not PDF objects: a ``name`` token carries the decoded name
as a plain ``str`` **without** the leading ``/`` (wrap it in ``PdfName`` if you need the
object), ``string``/``hexstring`` carry ``bytes``, and ``num`` carries ``int`` or
``float``.
"""

from __future__ import annotations

from typing import Any, Iterator, List, NamedTuple, Optional

from .objects import PdfName

__all__ = [
    "Token",
    "Lexer",
    "KEYWORDS",
    "WHITESPACE",
    "DELIMITERS",
    "tokenize",
]

#: Bytes PDF treats as white space (including NUL).
WHITESPACE = frozenset(b"\x00\t\n\x0c\r ")
#: Bytes PDF treats as delimiters; they terminate names, numbers and keywords.
DELIMITERS = frozenset(b"()<>[]{}/%")
#: Keywords the object layer cares about.  Any other regular-character run is still
#: returned as a ``keyword`` token -- this set is for callers that want to validate.
KEYWORDS = frozenset(
    {
        "obj",
        "endobj",
        "stream",
        "endstream",
        "R",
        "true",
        "false",
        "null",
        "xref",
        "trailer",
        "startxref",
        "n",
        "f",
    }
)

_TOKEN_KINDS = (
    "num",
    "name",
    "string",
    "hexstring",
    "dict_open",
    "dict_close",
    "array_open",
    "array_close",
    "keyword",
    "eof",
)

_NUMBER_START = frozenset(b"0123456789+-.")
_NUMBER_BODY = frozenset(b"0123456789+-.")
_HEX_DIGITS = frozenset(b"0123456789abcdefABCDEF")
_OCTAL_DIGITS = frozenset(b"01234567")

_SIMPLE_ESCAPES = {
    0x6E: 0x0A,  # n
    0x72: 0x0D,  # r
    0x74: 0x09,  # t
    0x62: 0x08,  # b
    0x66: 0x0C,  # f
    0x28: 0x28,  # (
    0x29: 0x29,  # )
    0x5C: 0x5C,  # backslash
}


class Token(NamedTuple):
    """One lexical token.

    ``kind`` is one of ``num, name, string, hexstring, dict_open, dict_close,
    array_open, array_close, keyword, eof``; ``pos`` is the offset of the token's first
    byte in the buffer.
    """

    kind: str
    value: Any
    pos: int

    def is_keyword(self, *names: str) -> bool:
        """True when this is a ``keyword`` token whose text is one of ``names``."""
        return self.kind == "keyword" and self.value in names

    def __bool__(self) -> bool:
        return self.kind != "eof"


class Lexer:
    """Tokenizer over a PDF byte buffer.

    ``Lexer(data, pos)`` starts at ``pos``; ``pos`` stays readable and writable at all
    times so a parser can jump to an xref offset and keep lexing.
    """

    __slots__ = ("data", "_pos", "_peeked", "_peek_end")

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data: bytes = data if isinstance(data, (bytes, bytearray)) else bytes(data)
        self._pos: int = max(0, int(pos))
        self._peeked: Optional[Token] = None
        self._peek_end: int = self._pos

    # -- cursor -----------------------------------------------------------------------
    @property
    def pos(self) -> int:
        """Current byte offset.  Assigning to it discards any peeked token."""
        return self._pos

    @pos.setter
    def pos(self, value: int) -> None:
        self._pos = max(0, int(value))
        self._peeked = None
        self._peek_end = self._pos

    def at_end(self) -> bool:
        """True when only white space and comments remain."""
        return self.peek().kind == "eof"

    # -- token access -----------------------------------------------------------------
    def next_token(self) -> Token:
        """Consume and return the next token; ``eof`` forever once the buffer is spent."""
        if self._peeked is not None:
            token = self._peeked
            self._pos = self._peek_end
            self._peeked = None
            return token
        return self._scan()

    def peek(self) -> Token:
        """Return the next token **without** consuming it.  Idempotent."""
        if self._peeked is None:
            start = self._pos
            token = self._scan()
            self._peek_end = self._pos
            self._pos = start
            self._peeked = token
        return self._peeked

    def __iter__(self) -> Iterator[Token]:
        """Yield tokens until (and excluding) ``eof``."""
        while True:
            token = self.next_token()
            if token.kind == "eof":
                return
            yield token

    # -- scanning ---------------------------------------------------------------------
    def _skip_space(self) -> None:
        data = self.data
        n = len(data)
        pos = self._pos
        while pos < n:
            byte = data[pos]
            if byte in WHITESPACE:
                pos += 1
                continue
            if byte == 0x25:  # '%' comment runs to end of line
                pos += 1
                while pos < n and data[pos] not in (0x0A, 0x0D):
                    pos += 1
                continue
            break
        self._pos = pos

    def _scan(self) -> Token:
        self._skip_space()
        data = self.data
        n = len(data)
        start = self._pos
        if start >= n:
            return Token("eof", None, start)
        byte = data[start]

        if byte == 0x2F:  # '/'
            return self._scan_name(start)
        if byte == 0x28:  # '('
            return self._scan_literal_string(start)
        if byte == 0x3C:  # '<'
            if start + 1 < n and data[start + 1] == 0x3C:
                self._pos = start + 2
                return Token("dict_open", "<<", start)
            return self._scan_hex_string(start)
        if byte == 0x3E:  # '>'
            if start + 1 < n and data[start + 1] == 0x3E:
                self._pos = start + 2
                return Token("dict_close", ">>", start)
            self._pos = start + 1
            return Token("keyword", ">", start)
        if byte == 0x5B:  # '['
            self._pos = start + 1
            return Token("array_open", "[", start)
        if byte == 0x5D:  # ']'
            self._pos = start + 1
            return Token("array_close", "]", start)
        if byte in (0x7B, 0x7D, 0x29):  # '{', '}', stray ')'
            self._pos = start + 1
            return Token("keyword", chr(byte), start)
        if byte in _NUMBER_START:
            return self._scan_number(start)
        return self._scan_keyword(start)

    # -- individual token scanners ----------------------------------------------------
    def _scan_regular_run(self, start: int) -> bytes:
        """Consume a run of regular characters (not white space, not a delimiter)."""
        data = self.data
        n = len(data)
        pos = start
        while pos < n:
            byte = data[pos]
            if byte in WHITESPACE or byte in DELIMITERS:
                break
            pos += 1
        self._pos = pos
        return data[start:pos]

    def _scan_name(self, start: int) -> Token:
        raw = self._scan_regular_run(start + 1)
        return Token("name", PdfName.decode(raw).value, start)

    def _scan_keyword(self, start: int) -> Token:
        raw = self._scan_regular_run(start)
        if not raw:
            # A delimiter byte we do not otherwise handle: consume it so the caller can
            # never be stuck on the same offset twice.
            self._pos = start + 1
            return Token("keyword", chr(self.data[start]), start)
        return Token("keyword", raw.decode("latin-1"), start)

    def _scan_number(self, start: int) -> Token:
        data = self.data
        n = len(data)
        pos = start
        while pos < n and data[pos] in _NUMBER_BODY:
            pos += 1
        self._pos = pos
        return Token("num", _parse_number(data[start:pos]), start)

    def _scan_literal_string(self, start: int) -> Token:
        data = self.data
        n = len(data)
        pos = start + 1
        depth = 1
        out = bytearray()
        while pos < n:
            byte = data[pos]
            pos += 1
            if byte == 0x5C:  # backslash
                if pos >= n:
                    break
                esc = data[pos]
                pos += 1
                simple = _SIMPLE_ESCAPES.get(esc)
                if simple is not None:
                    out.append(simple)
                elif esc in _OCTAL_DIGITS:
                    value = esc - 0x30
                    for _ in range(2):
                        if pos < n and data[pos] in _OCTAL_DIGITS:
                            value = value * 8 + (data[pos] - 0x30)
                            pos += 1
                        else:
                            break
                    out.append(value & 0xFF)
                elif esc == 0x0D:  # line continuation, CR or CRLF
                    if pos < n and data[pos] == 0x0A:
                        pos += 1
                elif esc == 0x0A:  # line continuation, LF
                    pass
                else:
                    out.append(esc)  # unknown escape: the backslash is dropped
                continue
            if byte == 0x28:  # '('
                depth += 1
                out.append(byte)
                continue
            if byte == 0x29:  # ')'
                depth -= 1
                if depth == 0:
                    break
                out.append(byte)
                continue
            if byte == 0x0D:  # CR and CRLF both mean a single LF inside a string
                out.append(0x0A)
                if pos < n and data[pos] == 0x0A:
                    pos += 1
                continue
            out.append(byte)
        self._pos = pos
        return Token("string", bytes(out), start)

    def _scan_hex_string(self, start: int) -> Token:
        data = self.data
        n = len(data)
        pos = start + 1
        digits = bytearray()
        while pos < n:
            byte = data[pos]
            pos += 1
            if byte == 0x3E:  # '>'
                break
            if byte in _HEX_DIGITS:
                digits.append(byte)
            # anything else (white space or junk) is ignored, per the spec's leniency
        if len(digits) % 2:
            digits.append(0x30)  # an odd final digit is padded with '0'
        self._pos = pos
        try:
            value = bytes.fromhex(digits.decode("ascii"))
        except ValueError:  # pragma: no cover - digits is hex by construction
            value = b""
        return Token("hexstring", value, start)


def _parse_number(raw: bytes) -> Any:
    """Parse a PDF numeric token leniently.

    ``4.``, ``.5`` and ``-.002`` are all legal PDF; ``--5``, ``1.2.3`` and a bare sign
    are not, but appear in the wild.  Anything unparseable degrades to ``0`` rather than
    raising, matching what real viewers do.
    """
    text = raw.decode("latin-1")
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        pass
    # Salvage the longest well-formed prefix: an optional sign, digits, one dot, digits.
    pos = 0
    n = len(text)
    if pos < n and text[pos] in "+-":
        pos += 1
    mantissa_start = pos
    while pos < n and text[pos].isdigit():
        pos += 1
    if pos < n and text[pos] == ".":
        pos += 1
        while pos < n and text[pos].isdigit():
            pos += 1
    prefix = text[:pos]
    if pos == mantissa_start or prefix in ("", "+", "-", ".", "+.", "-."):
        return 0
    try:
        if "." in prefix:
            return float(prefix)
        return int(prefix)
    except ValueError:  # pragma: no cover - prefix is well formed by construction
        return 0


def tokenize(data: bytes, pos: int = 0) -> List[Token]:
    """Tokenize a whole buffer.  Convenience wrapper used mostly by tests."""
    return list(Lexer(data, pos))
