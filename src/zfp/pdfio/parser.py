"""PDF file-structure parser: xref tables, xref streams, object streams, repair.

This module turns a byte buffer into a navigable object graph.  It is the only place
in ZFP that knows how a PDF *file* is laid out, as opposed to how a PDF *object* is
spelled (:mod:`zfp.pdfio.objects`) or tokenized (:mod:`zfp.pdfio.lexer`).

Two classes matter to callers:

:class:`ObjectParser`
    A token-stream-to-object translator.  Handles dictionaries, arrays, numbers,
    names, strings, booleans, ``null``, indirect references and streams.

:class:`PdfFile`
    The document.  :meth:`PdfFile.load` reads the header, walks the ``startxref`` /
    ``/Prev`` chain merging classic tables, cross-reference streams and hybrid
    ``/XRefStm`` sections, resolves objects out of ``/Type /ObjStm`` containers, and
    falls back to :meth:`PdfFile.rebuild_xref` -- a brute-force scan of the whole file
    -- whenever any of that turns out to be a lie.

The design rule throughout is *never fail on a file a viewer would open*.  Real PDFs
have wrong ``/Length`` values, offsets shifted by a junk prefix, truncated xref tables
and dangling ``/Root`` references; every one of those degrades to a repair path rather
than an exception.
"""

from __future__ import annotations

import os
import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

try:  # pragma: no cover - Protocol exists on every supported interpreter
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment,misc]

from ..core.errors import EncryptedDocumentError, PdfParseError
from ..core.logging import get_logger
from .lexer import Lexer, Token
from .objects import (
    PdfArray,
    PdfDict,
    PdfName,
    PdfNull,
    PdfRef,
    PdfStream,
    PdfString,
)

__all__ = [
    "Resolver",
    "ObjectParser",
    "XrefEntry",
    "PdfFile",
    "LockedString",
    "LockedStream",
]

logger = get_logger(__name__)

#: An indirect-object header anywhere in the file.  ``(?<![0-9])`` stops the scanner
#: from matching the tail of a longer number.
_OBJ_HEADER_RE = re.compile(rb"(?<![0-9])(\d{1,10})[\x00\t\r\n\f ]{1,8}(\d{1,5})[\x00\t\r\n\f ]{1,8}obj\b")
_TRAILER_RE = re.compile(rb"trailer\b")
_XREF_TYPE_RE = re.compile(rb"/Type[\x00\t\r\n\f ]*/XRef\b")
_CATALOG_RE = re.compile(rb"/Type[\x00\t\r\n\f ]*/Catalog\b")
_STANDARD_FILTER_RE = re.compile(rb"/Filter[\x00\t\r\n\f ]*/Standard\b")
_HEADER_RE = re.compile(rb"%PDF-(\d+)\.(\d+)")

_EOL_BYTES = b"\r\n"
_SPACE_BYTES = b"\x00\t\n\x0c\r "

#: Guard rails.  A PDF that trips one of these is broken, not big.
_MAX_DEPTH = 96
_MAX_SECTIONS = 512
_MAX_PAGES = 65536
_MAX_PAGE_DEPTH = 64


class Resolver(Protocol):
    """Anything that can turn a :class:`PdfRef` into the object it names."""

    def resolve(self, obj: Any) -> Any:  # pragma: no cover - structural protocol
        ...


# --------------------------------------------------------------------------------------
# Locked objects (encrypted document, no usable password)
# --------------------------------------------------------------------------------------


class LockedString(PdfString):
    """A string whose plaintext is unavailable because the document stayed encrypted.

    The ciphertext is still readable through :attr:`raw` so structural tooling keeps
    working; asking for the *text* raises.
    """

    def text(self) -> str:
        raise EncryptedDocumentError(
            "cannot decode a string: the document is encrypted and no valid "
            "password has been supplied"
        )


class LockedStream(PdfStream):
    """A stream whose plaintext is unavailable because the document stayed encrypted."""

    def decoded(self, resolver: Any = None) -> bytes:
        raise EncryptedDocumentError(
            "cannot decode a stream: the document is encrypted and no valid "
            "password has been supplied"
        )


# --------------------------------------------------------------------------------------
# Object parsing
# --------------------------------------------------------------------------------------


class ObjectParser:
    """Parse PDF objects out of a byte buffer starting at ``pos``.

    ``resolver`` is optional and used for exactly one thing: turning an indirect
    ``/Length`` into an integer when reading a stream.  When it is absent, or when the
    resolved length is wrong, the parser scans forward for the next ``endstream``
    keyword, which is what makes damaged files readable.
    """

    __slots__ = ("data", "lexer", "resolver", "_buf")

    def __init__(self, data: bytes, pos: int = 0, resolver: Optional[Resolver] = None) -> None:
        self.data: bytes = bytes(data)
        self.lexer = Lexer(self.data, pos)
        self.resolver = resolver
        self._buf: List[Token] = []

    # -- cursor -----------------------------------------------------------------------
    @property
    def pos(self) -> int:
        """Offset of the next unconsumed token."""
        if self._buf:
            return self._buf[0].pos
        return self.lexer.pos

    def seek(self, pos: int) -> None:
        """Move the cursor, discarding any buffered look-ahead."""
        self._buf = []
        self.lexer.pos = pos

    # -- token plumbing ---------------------------------------------------------------
    def _next(self) -> Token:
        if self._buf:
            return self._buf.pop(0)
        return self.lexer.next_token()

    def _peek(self) -> Token:
        if not self._buf:
            self._buf.append(self.lexer.next_token())
        return self._buf[0]

    def _push(self, token: Token) -> None:
        self._buf.insert(0, token)

    # -- public API -------------------------------------------------------------------
    def parse_object(self, depth: int = 0) -> Any:
        """Parse and return the next object.

        Unexpected tokens degrade to :data:`PdfNull.NULL` after being consumed, so a
        caller looping over a malformed array can never spin.
        """
        token = self._next()
        kind = token.kind

        if kind == "num":
            return self._maybe_reference(token)
        if kind == "name":
            return PdfName(token.value)
        if kind == "string":
            return PdfString(token.value, False)
        if kind == "hexstring":
            return PdfString(token.value, True)
        if kind == "array_open":
            return self._parse_array(depth)
        if kind == "dict_open":
            return self._parse_dict_or_stream(depth)
        if kind == "keyword":
            value = token.value
            if value == "true":
                return True
            if value == "false":
                return False
            return PdfNull.NULL
        # dict_close / array_close arriving here are stray delimiters; eof ends the run.
        return PdfNull.NULL

    def parse_indirect_object(self) -> Tuple[int, int, Any]:
        """Parse ``N G obj ... endobj`` at the cursor and return ``(num, gen, object)``."""
        t_num = self._next()
        t_gen = self._next()
        t_obj = self._next()
        if t_num.kind != "num" or t_gen.kind != "num" or not t_obj.is_keyword("obj"):
            raise PdfParseError(
                f"expected an indirect object header at offset {t_num.pos}"
            )
        obj = self.parse_object()
        token = self._peek()
        if token.is_keyword("endobj"):
            self._next()
        return int(t_num.value), int(t_gen.value), obj

    # -- composites -------------------------------------------------------------------
    def _maybe_reference(self, token: Token) -> Any:
        """Return ``PdfRef`` for ``N G R``, otherwise the number itself."""
        value = token.value
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return value
        second = self._next()
        if second.kind == "num" and isinstance(second.value, int) and second.value >= 0:
            third = self._next()
            if third.is_keyword("R"):
                return PdfRef(int(value), int(second.value))
            self._push(third)
        self._push(second)
        return value

    def _parse_array(self, depth: int) -> PdfArray:
        out = PdfArray()
        if depth > _MAX_DEPTH:
            return out
        while True:
            token = self._peek()
            if token.kind in ("array_close", "eof"):
                if token.kind == "array_close":
                    self._next()
                return out
            if token.kind == "dict_close" or token.is_keyword("endobj", "endstream"):
                # An unterminated array: stop before swallowing the enclosing object.
                return out
            out.append(self.parse_object(depth + 1))

    def _parse_dict_or_stream(self, depth: int) -> Any:
        result = PdfDict()
        if depth > _MAX_DEPTH:
            return result
        while True:
            token = self._next()
            if token.kind in ("dict_close", "eof"):
                break
            if token.is_keyword("endobj", "stream", "endstream"):
                # A dictionary that forgot its ``>>``.  Put the keyword back so the
                # stream check below still sees it.
                self._push(token)
                break
            if token.kind != "name":
                continue  # junk between entries; the token is consumed, so we progress
            key = token.value
            value = self.parse_object(depth + 1)
            result[key] = value
        token = self._peek()
        if token.is_keyword("stream"):
            self._next()
            return self._read_stream(result, token.pos + len("stream"))
        return result

    # -- streams ----------------------------------------------------------------------
    def _read_stream(self, header: PdfDict, after_keyword: int) -> PdfStream:
        """Read the stream body that follows the ``stream`` keyword at ``after_keyword``."""
        data = self.data
        start = after_keyword
        # Exactly one EOL may follow the keyword: CRLF or LF per the spec, bare CR in
        # the wild.  Anything else is part of the payload.
        if data[start : start + 2] == b"\r\n":
            start += 2
        elif data[start : start + 1] in (b"\n", b"\r"):
            start += 1

        declared = self._declared_length(header)
        end, resume = self._stream_extent(start, declared)
        raw = data[start:end]
        self.seek(resume)
        return PdfStream(header, raw)

    def _declared_length(self, header: PdfDict) -> Optional[int]:
        value = header.get("Length")
        if isinstance(value, PdfRef):
            if self.resolver is None:
                return None
            try:
                value = self.resolver.resolve(value)
            except Exception:  # pragma: no cover - resolver failures are not fatal
                return None
        if isinstance(value, bool) or value is None or isinstance(value, PdfNull):
            return None
        if isinstance(value, (int, float)):
            length = int(value)
            return length if length >= 0 else None
        return None

    def _stream_extent(self, start: int, declared: Optional[int]) -> Tuple[int, int]:
        """Return ``(body_end, resume_pos)`` for a stream body beginning at ``start``.

        A declared length is trusted only when ``endstream`` actually follows it.
        """
        data = self.data
        size = len(data)
        if declared is not None and 0 <= start + declared <= size:
            tail = data[start + declared : start + declared + 24]
            if tail.lstrip(_SPACE_BYTES).startswith(b"endstream"):
                end = start + declared
                marker = data.find(b"endstream", end)
                resume = marker + len(b"endstream") if marker >= 0 else size
                return end, resume
        marker = data.find(b"endstream", start)
        if marker < 0:
            return size, size
        end = marker
        # ``endstream`` is preceded by an EOL that belongs to the delimiter, not the data.
        if end > start and data[end - 1 : end] == b"\n":
            end -= 1
        if end > start and data[end - 1 : end] == b"\r":
            end -= 1
        return end, marker + len(b"endstream")


# --------------------------------------------------------------------------------------
# Cross-reference bookkeeping
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class XrefEntry:
    """One cross-reference entry.

    ``kind`` follows the xref-stream field-1 encoding: ``0`` free, ``1`` an offset in
    the file, ``2`` an object living inside an object stream.
    """

    num: int
    kind: int
    offset: int = 0
    gen: int = 0
    stream_num: int = 0
    stream_index: int = 0

    @property
    def in_use(self) -> bool:
        """True for entries that name a real object."""
        return self.kind in (1, 2)


# --------------------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------------------


class PdfFile:
    """A parsed PDF file: trailer, cross-reference map and object access.

    Construct through :meth:`load` or :meth:`open`; the constructor only sets up empty
    state.  The instance itself satisfies the :class:`Resolver` protocol, so it can be
    handed to :meth:`PdfStream.decoded` and friends.
    """

    def __init__(self, data: bytes) -> None:
        self.data: bytes = bytes(data)
        self.trailer: PdfDict = PdfDict()
        self.version: str = "1.4"
        self.header_offset: int = 0
        self.startxref: int = -1
        self.xref: Dict[int, XrefEntry] = {}
        self.warnings: List[str] = []
        self.rebuilt: bool = False
        self.security: Optional[Any] = None
        self.encrypt_dict: Optional[PdfDict] = None

        self._cache: Dict[int, Any] = {}
        self._objstm_cache: Dict[int, Tuple[bytes, int, List[Tuple[int, int]]]] = {}
        self._parsing: Set[int] = set()
        self._catalog: Optional[PdfDict] = None
        self._pages_cache: Optional[Tuple[List[PdfDict], List[Any]]] = None
        self._encrypt_num: int = -1
        self._doc_id: bytes = b""
        self._rebuilding: bool = False

    # -- construction -----------------------------------------------------------------
    @staticmethod
    def load(data: bytes, password: str = "") -> PdfFile:
        """Parse ``data`` into a :class:`PdfFile`.

        Raises :class:`PdfParseError` only when the bytes cannot be a PDF at all.
        Everything short of that -- a wrong ``startxref``, a truncated table, a broken
        ``/Root`` -- routes through :meth:`rebuild_xref`.

        ``password`` is an additive convenience for encrypted files; the empty user
        password is always tried automatically.
        """
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)
        if not isinstance(data, bytes):
            raise PdfParseError(f"PdfFile.load expects bytes, got {type(data).__name__!r}")
        if not data:
            raise PdfParseError("cannot parse an empty buffer as a PDF")

        pdf = PdfFile(data)
        pdf._read_header()
        try:
            pdf._read_xref()
        except Exception as exc:  # noqa: BLE001 - any failure means "repair it"
            pdf.warnings.append(f"cross-reference unusable ({exc}); rebuilding")
            pdf.rebuild_xref()
        pdf._setup_encryption()
        if not pdf._structure_ok():
            pdf.warnings.append("document structure unusable; rebuilding")
            pdf.rebuild_xref()
            if pdf.encrypt_dict is None:
                pdf._setup_encryption()
        if password:
            pdf.authenticate(password)
        if not pdf.xref:
            raise PdfParseError(f"no PDF objects found in {len(data)} bytes")
        return pdf

    @staticmethod
    def open(path: os.PathLike[str] | str, password: str = "") -> PdfFile:
        """Read ``path`` from disk and :meth:`load` it."""
        with open(os.fspath(path), "rb") as handle:
            return PdfFile.load(handle.read(), password)

    # -- header -----------------------------------------------------------------------
    def _read_header(self) -> None:
        """Locate ``%PDF-x.y`` and record the byte offset every xref offset is relative to."""
        window = self.data[:1024]
        index = window.find(b"%PDF-")
        if index < 0:
            index = self.data.find(b"%PDF-")
        if index < 0:
            if _OBJ_HEADER_RE.search(self.data) is None:
                raise PdfParseError("not a PDF: no %PDF- header and no indirect objects")
            self.warnings.append("missing %PDF- header; assuming version 1.4")
            self.header_offset = 0
            self.version = "1.4"
            return
        self.header_offset = index
        match = _HEADER_RE.match(self.data, index)
        if match is not None:
            major = match.group(1).decode("ascii")
            minor = match.group(2).decode("ascii")
            self.version = f"{major}.{minor}"
        if index:
            self.warnings.append(f"{index} junk bytes before the %PDF- header")

    # -- cross-reference chain --------------------------------------------------------
    def _read_xref(self) -> None:
        """Follow ``startxref`` and the ``/Prev`` chain, merging newest-first."""
        index = self.data.rfind(b"startxref")
        if index < 0:
            raise PdfParseError("no startxref keyword")
        token = Lexer(self.data, index + len(b"startxref")).next_token()
        if token.kind != "num":
            raise PdfParseError(f"malformed startxref at offset {index}")
        declared = int(token.value)
        # Only advertise ``startxref`` once a section at that offset has actually
        # parsed: an incremental writer uses it for /Prev and must not chain onto a
        # value that turned out to be a lie.
        self.startxref = -1

        queue: List[int] = [declared]
        seen: Set[int] = set()
        sections = 0
        first_error: Optional[str] = None
        while queue:
            offset = queue.pop(0)
            if offset in seen:
                continue
            seen.add(offset)
            sections += 1
            if sections > _MAX_SECTIONS:
                self.warnings.append("cross-reference chain too long; truncated")
                break
            try:
                entries, trailer = self._xref_section_at(offset)
            except Exception as exc:  # noqa: BLE001
                if first_error is None:
                    first_error = str(exc)
                self.warnings.append(
                    f"unreadable cross-reference section at {offset} ({exc})"
                )
                continue
            if self.startxref < 0 and offset == declared:
                self.startxref = declared
            # Hybrid files hide compressed objects from classic readers by marking them
            # free; the /XRefStm section holds the truth and therefore merges first.
            hybrid = trailer.get("XRefStm")
            if isinstance(hybrid, (int, float)) and not isinstance(hybrid, bool):
                try:
                    h_entries, h_trailer = self._xref_section_at(int(hybrid))
                except Exception as exc:  # noqa: BLE001
                    self.warnings.append(f"unreadable /XRefStm at {hybrid} ({exc})")
                else:
                    self._merge_entries(h_entries)
                    self._merge_trailer(h_trailer, skip=("Prev", "XRefStm"))
            self._merge_entries(entries)
            self._merge_trailer(trailer)
            previous = trailer.get("Prev")
            if isinstance(previous, (int, float)) and not isinstance(previous, bool):
                queue.append(int(previous))
        if not self.xref:
            raise PdfParseError(first_error or "cross-reference table held no entries")

    def _merge_entries(self, entries: Dict[int, XrefEntry]) -> None:
        """Merge a section's entries; already-known (newer) entries win."""
        for num, entry in entries.items():
            if num not in self.xref:
                self.xref[num] = entry

    def _merge_trailer(self, trailer: PdfDict, skip: Sequence[str] = ()) -> None:
        """Merge a section's trailer; already-known (newer) keys win."""
        for key in trailer.keys():
            if key in skip or key in ("Prev", "XRefStm"):
                continue
            if key not in self.trailer:
                self.trailer[key] = trailer[key]

    def _xref_section_at(self, offset: int) -> Tuple[Dict[int, XrefEntry], PdfDict]:
        """Parse one cross-reference section -- classic table or xref stream."""
        position = self._normalize_section_offset(offset)
        if position is None:
            raise PdfParseError(f"no cross-reference section at offset {offset}")
        if Lexer(self.data, position).peek().is_keyword("xref"):
            return self._parse_classic_xref(position)
        return self._parse_xref_stream(position)

    def _normalize_section_offset(self, offset: int) -> Optional[int]:
        """Return the real offset of a section, compensating for a junk file prefix."""
        for candidate in (offset, offset + self.header_offset, offset - self.header_offset):
            if candidate < 0 or candidate >= len(self.data):
                continue
            lexer = Lexer(self.data, candidate)
            first = lexer.peek()
            if first.is_keyword("xref"):
                return candidate
            if first.kind == "num":
                lexer.next_token()
                second = lexer.next_token()
                third = lexer.next_token()
                if second.kind == "num" and third.is_keyword("obj"):
                    return candidate
        return None

    def _parse_classic_xref(self, position: int) -> Tuple[Dict[int, XrefEntry], PdfDict]:
        """Parse ``xref`` subsections followed by ``trailer <<...>>``.

        Entries are read as tokens rather than fixed 20-byte records, which makes the
        common 19/20/21-byte line-ending variations a non-issue.
        """
        lexer = Lexer(self.data, position)
        if not lexer.next_token().is_keyword("xref"):
            raise PdfParseError(f"expected 'xref' at offset {position}")
        entries: Dict[int, XrefEntry] = {}
        limit = len(self.data) // 18 + 64
        while True:
            token = lexer.peek()
            if token.kind == "eof":
                break
            if token.is_keyword("trailer"):
                lexer.next_token()
                trailer = ObjectParser(self.data, lexer.pos).parse_object()
                if not isinstance(trailer, PdfDict):
                    trailer = PdfDict()
                return entries, trailer
            if token.kind != "num":
                break
            lexer.next_token()
            count_token = lexer.next_token()
            if count_token.kind != "num":
                break
            start = int(token.value)
            count = int(count_token.value)
            if count < 0:
                break
            count = min(count, limit)
            for index in range(count):
                if lexer.peek().kind != "num":
                    break
                field1 = lexer.next_token()
                field2 = lexer.next_token()
                if field2.kind != "num":
                    break
                marker = lexer.next_token()
                free = marker.kind == "keyword" and str(marker.value).startswith("f")
                num = start + index
                if num in entries:
                    continue
                if free:
                    entries[num] = XrefEntry(num, 0)
                else:
                    entries[num] = XrefEntry(
                        num, 1, offset=int(field1.value), gen=int(field2.value)
                    )
        return entries, PdfDict()

    def _parse_xref_stream(self, position: int) -> Tuple[Dict[int, XrefEntry], PdfDict]:
        """Parse a ``/Type /XRef`` cross-reference stream at ``position``."""
        _num, _gen, obj = ObjectParser(self.data, position).parse_indirect_object()
        if not isinstance(obj, PdfStream):
            raise PdfParseError(f"offset {position} is not a cross-reference stream")
        header = obj.dict
        type_name = header.get_name("Type")
        if type_name not in (None, "XRef"):
            raise PdfParseError(f"expected /Type /XRef, found /{type_name}")
        widths_raw = header.get_array("W")
        if not widths_raw:
            raise PdfParseError("cross-reference stream has no /W")
        widths = [int(w) if isinstance(w, (int, float)) else 0 for w in widths_raw]
        size = header.get_int("Size", 0) or 0
        index_raw = header.get_array("Index")
        pairs: List[Tuple[int, int]] = []
        if index_raw and len(index_raw) >= 2:
            for i in range(0, len(index_raw) - 1, 2):
                try:
                    pairs.append((int(index_raw[i]), int(index_raw[i + 1])))
                except (TypeError, ValueError):
                    continue
        if not pairs:
            pairs = [(0, size)]

        payload = obj.decoded(None)
        row_length = sum(widths)
        if row_length <= 0:
            raise PdfParseError("cross-reference stream has a zero-width /W")
        entries: Dict[int, XrefEntry] = {}
        cursor = 0
        for start, count in pairs:
            for i in range(max(0, count)):
                if cursor + row_length > len(payload):
                    break
                fields: List[Optional[int]] = []
                offset = cursor
                for width in widths:
                    if width <= 0:
                        fields.append(None)
                        continue
                    fields.append(int.from_bytes(payload[offset : offset + width], "big"))
                    offset += width
                cursor += row_length
                kind = fields[0] if fields and fields[0] is not None else 1
                field2 = fields[1] if len(fields) > 1 and fields[1] is not None else 0
                field3 = fields[2] if len(fields) > 2 and fields[2] is not None else 0
                num = start + i
                if num in entries:
                    continue
                if kind == 0:
                    entries[num] = XrefEntry(num, 0)
                elif kind == 1:
                    entries[num] = XrefEntry(num, 1, offset=field2, gen=field3)
                elif kind == 2:
                    entries[num] = XrefEntry(
                        num, 2, stream_num=field2, stream_index=field3
                    )
        return entries, PdfDict(header)

    # -- object access ----------------------------------------------------------------
    def resolve(self, obj: Any) -> Any:
        """Follow a chain of indirect references.

        A reference cycle yields :data:`PdfNull.NULL` rather than recursing forever.
        Non-reference values are returned untouched.
        """
        seen: Set[Tuple[int, int]] = set()
        while isinstance(obj, PdfRef):
            key = (obj.num, obj.gen)
            if key in seen:
                return PdfNull.NULL
            seen.add(key)
            obj = self.get_object(obj.num, obj.gen)
        return obj

    def get_object(self, num: int, gen: int = 0) -> Any:
        """Return object ``num``, parsing and caching it on first access."""
        num = int(num)
        if num in self._cache:
            return self._cache[num]
        if num in self._parsing:
            # A self-referential /Length (or similar); break the cycle.
            return PdfNull.NULL
        self._parsing.add(num)
        try:
            try:
                obj = self._load_object(num, gen)
            except EncryptedDocumentError:
                raise
            except Exception as exc:  # noqa: BLE001
                if not self.rebuilt and not self._rebuilding:
                    self.warnings.append(
                        f"object {num} unreadable ({exc}); rebuilding cross-reference"
                    )
                    self._parsing.discard(num)
                    self.rebuild_xref()
                    self._parsing.add(num)
                    try:
                        obj = self._load_object(num, gen)
                    except Exception:  # noqa: BLE001
                        obj = PdfNull.NULL
                else:
                    obj = PdfNull.NULL
        finally:
            self._parsing.discard(num)
        self._cache[num] = obj
        return obj

    def _load_object(self, num: int, gen: int) -> Any:
        entry = self.xref.get(num)
        if entry is None:
            if not self.rebuilt and not self._rebuilding:
                self.rebuild_xref()
                entry = self.xref.get(num)
            if entry is None:
                return PdfNull.NULL
        if entry.kind == 0:
            return PdfNull.NULL
        if entry.kind == 2:
            return self._object_from_stream(entry.stream_num, entry.stream_index, num)
        position = self._object_start(entry.offset, num)
        if position is None:
            raise PdfParseError(
                f"object {num} is not at its cross-reference offset {entry.offset}"
            )
        parser = ObjectParser(self.data, position, resolver=self)
        actual, actual_gen, obj = parser.parse_indirect_object()
        if actual != num:
            raise PdfParseError(
                f"object header mismatch at {position}: wanted {num}, found {actual}"
            )
        return self._apply_security(obj, num, actual_gen)

    def _object_start(self, offset: int, num: int) -> Optional[int]:
        """Return the offset where ``num 0 obj`` really begins, or ``None``."""
        data = self.data
        for candidate in (offset, offset + self.header_offset, offset - self.header_offset):
            if candidate < 0 or candidate >= len(data):
                continue
            lexer = Lexer(data, candidate)
            first = lexer.next_token()
            second = lexer.next_token()
            third = lexer.next_token()
            if (
                first.kind == "num"
                and second.kind == "num"
                and third.is_keyword("obj")
                and int(first.value) == num
            ):
                return candidate
        # Producers sometimes miss by a handful of bytes; look in a small window.
        pattern = re.compile(
            rb"(?<![0-9])" + str(num).encode("ascii") + rb"[\x00\t\r\n\f ]{1,8}\d{1,5}[\x00\t\r\n\f ]{1,8}obj\b"
        )
        window_start = max(0, offset - 512)
        match = pattern.search(data, window_start, offset + 1024)
        if match is not None:
            return match.start()
        return None

    def _object_from_stream(self, stream_num: int, index: int, want: int) -> Any:
        """Pull object ``want`` out of the ``/Type /ObjStm`` container ``stream_num``."""
        payload, first, pairs = self._objstm_index(stream_num)
        entry: Optional[Tuple[int, int]] = None
        if 0 <= index < len(pairs) and pairs[index][0] == want:
            entry = pairs[index]
        else:
            for pair in pairs:
                if pair[0] == want:
                    entry = pair
                    break
        if entry is None:
            return PdfNull.NULL
        position = first + entry[1]
        if position < 0 or position >= len(payload):
            return PdfNull.NULL
        # Objects inside an object stream are covered by the container's own
        # decryption; they are never encrypted a second time.
        return ObjectParser(payload, position, resolver=self).parse_object()

    def _objstm_index(self, num: int) -> Tuple[bytes, int, List[Tuple[int, int]]]:
        """Return ``(payload, first, [(objnum, offset), ...])`` for an object stream."""
        cached = self._objstm_cache.get(num)
        if cached is not None:
            return cached
        stream = self.get_object(num)
        if not isinstance(stream, PdfStream):
            raise PdfParseError(f"object {num} is not an object stream")
        type_name = stream.dict.get_name("Type")
        if type_name not in (None, "ObjStm"):
            raise PdfParseError(f"object {num} is /{type_name}, not /ObjStm")
        payload = stream.decoded(self)
        count = stream.dict.get_int("N", 0, self) or 0
        first = stream.dict.get_int("First", 0, self) or 0
        first = max(0, min(first, len(payload)))
        pairs: List[Tuple[int, int]] = []
        lexer = Lexer(payload, 0)
        for _ in range(max(0, count)):
            key = lexer.next_token()
            value = lexer.next_token()
            if key.kind != "num" or value.kind != "num":
                break
            if key.pos >= first:
                break
            pairs.append((int(key.value), int(value.value)))
        result = (payload, first, pairs)
        self._objstm_cache[num] = result
        return result

    def object_numbers(self) -> List[int]:
        """Sorted object numbers that the cross-reference marks as in use."""
        return sorted(num for num, entry in self.xref.items() if entry.in_use)

    def max_object_number(self) -> int:
        """Highest object number known to the file (0 when there are none)."""
        numbers = list(self.xref)
        return max(numbers) if numbers else 0

    # -- encryption -------------------------------------------------------------------
    @property
    def is_encrypted(self) -> bool:
        """True when the trailer carries an ``/Encrypt`` dictionary."""
        return self.encrypt_dict is not None

    @property
    def is_authenticated(self) -> bool:
        """True when content can actually be decrypted (always true when unencrypted)."""
        if self.encrypt_dict is None:
            return True
        return bool(self.security is not None and self.security.authenticated)

    def authenticate(self, password: str) -> bool:
        """Try ``password`` (user or owner) against the security handler.

        Returns ``True`` when the document is not encrypted or the password works, and
        clears the object cache so previously locked objects are re-read.
        """
        if self.encrypt_dict is None:
            return True
        if self.security is None:
            from .crypt import StandardSecurityHandler

            try:
                self.security = StandardSecurityHandler.from_encrypt_dict(
                    self.encrypt_dict, self._doc_id, password
                )
            except EncryptedDocumentError as exc:
                self.warnings.append(str(exc))
                return False
        else:
            self.security.authenticate(password)
        ok = self.is_authenticated
        if ok:
            self._invalidate_objects()
        return ok

    def _setup_encryption(self) -> None:
        """Detect ``/Encrypt``, build the handler and try the empty user password."""
        raw = self.trailer.get("Encrypt")
        if raw is None or isinstance(raw, PdfNull):
            return
        if isinstance(raw, PdfRef):
            self._encrypt_num = raw.num
        try:
            enc = self.resolve(raw)
        except EncryptedDocumentError:  # pragma: no cover - defensive
            enc = None
        if isinstance(enc, PdfStream):
            enc = enc.dict
        if not isinstance(enc, PdfDict):
            self.warnings.append("/Encrypt does not resolve to a dictionary")
            return
        self.encrypt_dict = enc
        self._doc_id = self._first_document_id()
        from .crypt import StandardSecurityHandler

        try:
            self.security = StandardSecurityHandler.from_encrypt_dict(
                enc, self._doc_id, ""
            )
        except EncryptedDocumentError as exc:
            self.security = None
            self.warnings.append(f"encrypted document: {exc}")
            return
        if not self.security.authenticated:
            self.warnings.append(
                "encrypted document: the empty user password was rejected"
            )
        self._invalidate_objects()

    def _first_document_id(self) -> bytes:
        """Return the raw bytes of the first ``/ID`` string, or ``b''``."""
        ids = self.trailer.get("ID")
        if isinstance(ids, PdfRef):
            ids = self.resolve(ids)
        if isinstance(ids, (PdfArray, list, tuple)) and ids:
            first = ids[0]
            if isinstance(first, PdfRef):
                first = self.resolve(first)
            if isinstance(first, PdfString):
                return first.raw
            if isinstance(first, (bytes, bytearray)):
                return bytes(first)
        return b""

    def _invalidate_objects(self) -> None:
        """Drop every cached object so they are re-parsed under the new crypt state."""
        self._cache.clear()
        self._objstm_cache.clear()
        self._catalog = None
        self._pages_cache = None

    def _apply_security(self, obj: Any, num: int, gen: int) -> Any:
        """Decrypt (or lock) a freshly parsed top-level object."""
        if self.encrypt_dict is None or num == self._encrypt_num:
            return obj
        if self.security is not None and self.security.authenticated:
            return self._decrypt_tree(obj, num, gen, 0)
        return self._lock_tree(obj, 0)

    def _decrypt_tree(self, obj: Any, num: int, gen: int, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            return obj
        if isinstance(obj, PdfString):
            obj.raw = self.security.decrypt(obj.raw, num, gen, True)
            return obj
        if isinstance(obj, PdfStream):
            if not self._stream_is_encrypted(obj):
                return obj
            self._decrypt_tree(obj.dict, num, gen, depth + 1)
            obj.raw = self.security.decrypt(obj.raw, num, gen, False)
            return obj
        if isinstance(obj, PdfDict):
            for key in list(obj.keys()):
                obj[key] = self._decrypt_tree(obj[key], num, gen, depth + 1)
            return obj
        if isinstance(obj, (PdfArray, list)):
            for index in range(len(obj)):
                obj[index] = self._decrypt_tree(obj[index], num, gen, depth + 1)
            return obj
        return obj

    def _stream_is_encrypted(self, stream: PdfStream) -> bool:
        """False for streams the spec exempts: xref streams, /Identity crypt, metadata."""
        type_name = stream.dict.get_name("Type")
        if type_name == "XRef":
            return False
        if type_name == "Metadata" and self.security is not None:
            if not self.security.encrypt_metadata:
                return False
        names = stream.filter_names()
        if "Crypt" in names:
            parms = stream.decode_parms()
            for name, parm in zip(names, parms):
                if name != "Crypt":
                    continue
                filter_name = parm.get_name("Name") if isinstance(parm, PdfDict) else None
                if filter_name in (None, "Identity"):
                    return False
        return True

    def _lock_tree(self, obj: Any, depth: int) -> Any:
        """Replace strings and streams with objects that refuse to hand out plaintext."""
        if depth > _MAX_DEPTH:
            return obj
        if isinstance(obj, LockedString) or isinstance(obj, LockedStream):
            return obj
        if isinstance(obj, PdfString):
            return LockedString(obj.raw, obj.hexform)
        if isinstance(obj, PdfStream):
            if not self._stream_is_encrypted(obj):
                return obj
            return LockedStream(self._lock_tree(obj.dict, depth + 1), obj.raw)
        if isinstance(obj, PdfDict):
            for key in list(obj.keys()):
                obj[key] = self._lock_tree(obj[key], depth + 1)
            return obj
        if isinstance(obj, (PdfArray, list)):
            for index in range(len(obj)):
                obj[index] = self._lock_tree(obj[index], depth + 1)
            return obj
        return obj

    # -- repair -----------------------------------------------------------------------
    def rebuild_xref(self) -> None:
        """Brute-force the cross-reference table by scanning the whole file.

        Every ``N G obj`` header is recorded, the **last** one for each object number
        winning (that is the newest incremental revision).  A trailer is then
        synthesised from any ``trailer`` dictionaries, any ``/Type /XRef`` stream
        dictionaries and, failing both, a direct scan for the document catalog.
        """
        if self._rebuilding:
            return
        self._rebuilding = True
        try:
            data = self.data
            offsets: Dict[int, int] = {}
            gens: Dict[int, int] = {}
            for match in _OBJ_HEADER_RE.finditer(data):
                try:
                    num = int(match.group(1))
                    gen = int(match.group(2))
                except ValueError:  # pragma: no cover - the regex only matches digits
                    continue
                offsets[num] = match.start()
                gens[num] = gen
            self.xref = {
                num: XrefEntry(num, 1, offset=offset, gen=gens.get(num, 0))
                for num, offset in offsets.items()
            }
            self._invalidate_objects()
            self.rebuilt = True
            previous_trailer = self.trailer
            self.trailer = self._rebuild_trailer(offsets, previous_trailer)
        finally:
            self._rebuilding = False

    def _rebuild_trailer(
        self, offsets: Dict[int, int], previous: PdfDict
    ) -> PdfDict:
        data = self.data
        trailer = PdfDict()
        for match in _TRAILER_RE.finditer(data):
            try:
                candidate = ObjectParser(data, match.end()).parse_object()
            except Exception:  # noqa: BLE001
                continue
            if isinstance(candidate, PdfDict):
                for key in candidate.keys():
                    if key in ("Prev", "XRefStm"):
                        continue
                    trailer[key] = candidate[key]

        sorted_offsets = sorted(offsets.items(), key=lambda item: item[1])
        positions = [item[1] for item in sorted_offsets]

        def owner(position: int) -> Optional[int]:
            index = bisect_right(positions, position) - 1
            if index < 0:
                return None
            return sorted_offsets[index][0]

        # Cross-reference stream dictionaries double as trailers.
        for match in _XREF_TYPE_RE.finditer(data):
            num = owner(match.start())
            if num is None:
                continue
            obj = self.get_object(num)
            source = obj.dict if isinstance(obj, PdfStream) else obj
            if not isinstance(source, PdfDict):
                continue
            if source.get_name("Type") != "XRef":
                continue
            for key in ("Root", "Info", "Encrypt", "ID"):
                if key in source and key not in trailer:
                    trailer[key] = source[key]

        for key in ("Root", "Info", "Encrypt", "ID"):
            if key not in trailer and key in previous:
                trailer[key] = previous[key]

        if not self._is_catalog(trailer.get("Root")):
            root = self._scan_for_catalog(owner)
            if root is not None:
                trailer["Root"] = root
            elif "Root" in trailer:
                del trailer["Root"]
        if "Info" in trailer and not isinstance(self.resolve(trailer.get("Info")), PdfDict):
            del trailer["Info"]
        if "Encrypt" not in trailer:
            match = _STANDARD_FILTER_RE.search(data)
            if match is not None:
                num = owner(match.start())
                if num is not None:
                    candidate = self.get_object(num)
                    if isinstance(candidate, PdfDict) and "O" in candidate and "U" in candidate:
                        trailer["Encrypt"] = PdfRef(num, gens_of(self.xref, num))
        trailer["Size"] = (max(offsets) + 1) if offsets else 1
        if "Root" not in trailer:
            self.warnings.append("rebuild found no document catalog")
        return trailer

    def _scan_for_catalog(self, owner: Callable[[int], Optional[int]]) -> Optional[PdfRef]:
        """Find the newest object that is a ``/Type /Catalog`` dictionary."""
        found: Optional[PdfRef] = None
        for match in _CATALOG_RE.finditer(self.data):
            num = owner(match.start())
            if num is None:
                continue
            obj = self.get_object(num)
            if isinstance(obj, PdfDict) and obj.get_name("Type") == "Catalog":
                found = PdfRef(num, gens_of(self.xref, num))
        return found

    def _is_catalog(self, ref: Any) -> bool:
        if ref is None or isinstance(ref, PdfNull):
            return False
        try:
            obj = self.resolve(ref)
        except EncryptedDocumentError:  # pragma: no cover - dicts are never locked
            return False
        if not isinstance(obj, PdfDict):
            return False
        return obj.get_name("Type") == "Catalog" or "Pages" in obj

    def _structure_ok(self) -> bool:
        """Cheap sanity check deciding whether a rebuild is warranted."""
        if not self.xref:
            return False
        if self.encrypt_dict is not None and not self.is_authenticated:
            return True  # nothing deeper can be checked without a password
        return self._is_catalog(self.trailer.get("Root"))

    # -- document structure -----------------------------------------------------------
    @property
    def catalog(self) -> PdfDict:
        """The document catalog.

        Resolves the trailer's ``/Root``; when that is missing or dangling, scans the
        file for a ``/Type /Catalog`` object.  Returns an empty dictionary rather than
        raising so callers can report a finding instead of crashing.
        """
        if self._catalog is not None:
            return self._catalog
        candidate = None
        root = self.trailer.get("Root")
        if root is not None and not isinstance(root, PdfNull):
            try:
                resolved = self.resolve(root)
            except EncryptedDocumentError:  # pragma: no cover - defensive
                resolved = None
            if isinstance(resolved, PdfDict) and (
                resolved.get_name("Type") == "Catalog" or "Pages" in resolved
            ):
                candidate = resolved
        if candidate is None:
            candidate = self._catalog_by_scan()
        if candidate is None:
            self.warnings.append("no document catalog could be located")
            candidate = PdfDict()
        self._catalog = candidate
        return candidate

    def _catalog_by_scan(self) -> Optional[PdfDict]:
        for num in self.object_numbers():
            try:
                obj = self.get_object(num)
            except EncryptedDocumentError:  # pragma: no cover - defensive
                continue
            if isinstance(obj, PdfDict) and obj.get_name("Type") == "Catalog":
                return obj
        return None

    def page_dicts(self) -> List[PdfDict]:
        """Every page dictionary, in document order."""
        return list(self._pages()[0])

    def page_refs(self) -> List[Any]:
        """The reference for each page, aligned with :meth:`page_dicts`.

        Entries are :class:`PdfRef` except where a page was written inline inside its
        parent's ``/Kids``, which yields :data:`PdfNull.NULL`.
        """
        return list(self._pages()[1])

    def _pages(self) -> Tuple[List[PdfDict], List[Any]]:
        if self._pages_cache is not None:
            return self._pages_cache
        dicts: List[PdfDict] = []
        refs: List[Any] = []
        seen: Set[Tuple[Any, Any]] = set()

        def walk(node: Any, depth: int) -> None:
            if depth > _MAX_PAGE_DEPTH or len(dicts) >= _MAX_PAGES:
                return
            ref = node if isinstance(node, PdfRef) else None
            key: Tuple[Any, Any] = (
                (ref.num, ref.gen) if ref is not None else ("inline", id(node))
            )
            if key in seen:
                return
            seen.add(key)
            try:
                resolved = self.resolve(node)
            except EncryptedDocumentError:  # pragma: no cover - dicts are never locked
                return
            if not isinstance(resolved, PdfDict):
                return
            type_name = resolved.get_name("Type")
            kids = resolved.get("Kids")
            if isinstance(kids, PdfRef):
                kids = self.resolve(kids)
            has_kids = isinstance(kids, (PdfArray, list, tuple))
            if type_name == "Page" or (
                not has_kids
                and type_name != "Pages"
                and ("Contents" in resolved or "MediaBox" in resolved)
            ):
                dicts.append(resolved)
                refs.append(ref if ref is not None else PdfNull.NULL)
                return
            if has_kids:
                for kid in kids:
                    walk(kid, depth + 1)

        catalog = self.catalog
        root = catalog.get("Pages")
        if root is not None and not isinstance(root, PdfNull):
            walk(root, 0)
        if not dicts:
            dicts, refs = self._pages_by_scan()
        self._pages_cache = (dicts, refs)
        return self._pages_cache

    def _pages_by_scan(self) -> Tuple[List[PdfDict], List[Any]]:
        """Last resort: every ``/Type /Page`` object, in object-number order."""
        dicts: List[PdfDict] = []
        refs: List[Any] = []
        for num in self.object_numbers():
            try:
                obj = self.get_object(num)
            except EncryptedDocumentError:  # pragma: no cover - defensive
                continue
            if isinstance(obj, PdfDict) and obj.get_name("Type") == "Page":
                dicts.append(obj)
                entry = self.xref.get(num)
                refs.append(PdfRef(num, entry.gen if entry else 0))
        if dicts:
            self.warnings.append(
                f"page tree unusable; {len(dicts)} page objects recovered by scanning"
            )
        return dicts, refs

    # -- misc -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"PdfFile(version={self.version}, objects={len(self.xref)}, "
            f"encrypted={self.is_encrypted}, rebuilt={self.rebuilt})"
        )


def gens_of(xref: Dict[int, XrefEntry], num: int) -> int:
    """Generation number recorded for ``num``, or 0."""
    entry = xref.get(num)
    return entry.gen if entry is not None else 0
