"""PDF serialization: incremental updates and full rewrites.

Two guarantees drive every design decision in this module.

**Byte preservation.**  :meth:`PdfWriter.write_incremental` returns the original file
bytes *verbatim* followed by a new revision.  Nothing in the original substrate is
rewritten, re-compressed or re-ordered, which is how ZFP proves it did not disturb a
document's visual appearance: a reader can diff the produced file against the input and
see that the first ``len(original)`` bytes are identical.

**Determinism.**  The same object graph always serializes to the same bytes.  Numbers
have one canonical spelling, dictionaries are written in insertion order, and the
``/ID`` trailer entry is derived from a BLAKE2b digest of the appended bytes rather
than from a clock or a random source.

The module is usable in three ways::

    writer = PdfWriter(pdf)                 # incremental edits to a parsed file
    ref = writer.add_object(PdfDict(...))
    data = writer.write_incremental()

    data = writer.write_full()              # flatten everything into one revision

    data = build_document(objects, root)    # a brand new file, no PdfFile needed
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.errors import PdfWriteError
from ..core.logging import get_logger
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
    "PdfWriter",
    "serialize_object",
    "format_number",
    "build_document",
    "DEFAULT_VERSION",
    "BINARY_COMMENT",
    "XREF_ENTRY_WIDTH",
]

_log = get_logger(__name__)

#: Written into the header of files produced from scratch.
DEFAULT_VERSION = "1.7"

#: The four high-bit bytes every PDF puts on line 2 so transfer agents treat the file
#: as binary (PDF 32000-1 section 7.5.2).
BINARY_COMMENT = b"%\xe2\xe3\xcf\xd3"

#: Maximum number of decimals kept when serializing a real number.
_FLOAT_DIGITS = 6

#: A classic cross-reference entry is exactly 20 bytes wide: ten offset digits, a
#: space, five generation digits, a space, the ``n``/``f`` type byte and a two-byte
#: end-of-line.  Readers that seek into a table by index depend on it.
XREF_ENTRY_WIDTH = 20


# --------------------------------------------------------------------------------------
# Scalar formatting
# --------------------------------------------------------------------------------------


def format_number(value: Any) -> bytes:
    """Return the canonical PDF spelling of a number.

    Integers are written plainly.  Reals get at most six decimals with trailing zeros
    removed, never scientific notation (which no PDF reader accepts), and negative zero
    is normalized to ``0``.  Non-finite values degrade to ``0`` rather than emitting the
    unparseable ``inf``/``nan`` tokens.

    Args:
        value: An :class:`int` or :class:`float` (``bool`` is rejected by the caller).

    Returns:
        The on-wire bytes, e.g. ``b'0'``, ``b'-12'``, ``b'1.5'``, ``b'612'``.
    """
    if isinstance(value, int):
        return b"%d" % value
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return b"0"
    text = "%.*f" % (_FLOAT_DIGITS, number)
    if "." in text:
        text = text.rstrip("0")
        if text.endswith("."):
            text = text[:-1]
    if text in ("", "-", "-0"):
        text = "0"
    return text.encode("ascii")


def _serialize_stream(stream: PdfStream) -> bytes:
    """Serialize a stream, forcing ``/Length`` to agree with the raw bytes."""
    raw = stream.raw
    body = PdfDict(stream.dict)
    body["Length"] = len(raw)
    # ``/Filter`` stays untouched: ``raw`` is by definition still encoded.
    return b"".join(
        (serialize_object(body), b"\nstream\n", raw, b"\nendstream")
    )


def serialize_object(obj: Any) -> bytes:
    """Serialize any PDF object to its on-wire byte representation.

    Handles the full object model: ``null``, booleans, integers, reals, names, strings,
    arrays, dictionaries, streams and indirect references.  Plain Python containers
    (``list``, ``tuple``, ``dict``) and ``str``/``bytes`` are accepted as conveniences
    and coerced to the corresponding PDF type -- a ``str`` always becomes a *string*,
    never a name, so a value such as ``"/Helv 0 Tf 0 g"`` survives intact.

    Args:
        obj: The object to serialize.

    Returns:
        The serialized bytes, with no leading or trailing whitespace.

    Raises:
        PdfWriteError: The value has no PDF representation.
    """
    if obj is None or isinstance(obj, PdfNull):
        return b"null"
    # bool must precede int: ``isinstance(True, int)`` is True.
    if isinstance(obj, bool):
        return b"true" if obj else b"false"
    if isinstance(obj, (int, float)):
        return format_number(obj)
    if isinstance(obj, PdfName):
        return obj.encoded
    if isinstance(obj, PdfString):
        return obj.serialize()
    if isinstance(obj, PdfRef):
        return obj.encoded
    if isinstance(obj, PdfStream):
        return _serialize_stream(obj)
    if isinstance(obj, dict):
        parts: List[bytes] = [b"<<"]
        for key, value in obj.items():
            parts.append(PdfName(_key_text(key)).encoded)
            parts.append(b" ")
            parts.append(serialize_object(value))
            parts.append(b" ")
        if len(parts) > 1:
            parts.pop()  # drop the trailing separator
        parts.append(b">>")
        return b"".join(parts)
    if isinstance(obj, (list, tuple)):
        return b"[" + b" ".join(serialize_object(item) for item in obj) + b"]"
    if isinstance(obj, (bytes, bytearray)):
        return PdfString(bytes(obj)).serialize()
    if isinstance(obj, str):
        return PdfString.from_text(obj).serialize()
    raise PdfWriteError("cannot serialize %r of type %s" % (obj, type(obj).__name__))


def _is_dead_container(value: Any) -> bool:
    """True for ``/ObjStm`` and ``/XRef`` streams, which a full rewrite makes obsolete."""
    if not isinstance(value, PdfStream):
        return False
    return value.dict.get_name("Type") in ("ObjStm", "XRef")


def _key_text(key: Any) -> str:
    """Normalize a dictionary key to the plain name text (no leading ``/``)."""
    if isinstance(key, PdfName):
        return key.value
    if isinstance(key, (bytes, bytearray)):
        return PdfName.decode(bytes(key)).value
    text = str(key)
    return text[1:] if text.startswith("/") else text


# --------------------------------------------------------------------------------------
# Cross-reference helpers
# --------------------------------------------------------------------------------------


def _contiguous_runs(numbers: Sequence[int]) -> List[List[int]]:
    """Split a sorted, de-duplicated sequence into runs of consecutive integers."""
    runs: List[List[int]] = []
    for number in numbers:
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    return runs


def _xref_in_use(offset: int, gen: int = 0) -> bytes:
    """Return a 20-byte in-use cross-reference entry."""
    return b"%010d %05d n \n" % (offset, gen)


def _xref_free(next_free: int, gen: int = 65535) -> bytes:
    """Return a 20-byte free cross-reference entry pointing at ``next_free``."""
    return b"%010d %05d f \n" % (next_free, gen)


def _derive_id_half(seed: bytes) -> PdfString:
    """Return the 16-byte hex ``/ID`` half derived from ``seed``.

    BLAKE2b keeps this deterministic: identical inputs always yield identical files,
    which is what the QA layer's byte-for-byte reproducibility checks rely on.
    """
    digest = hashlib.blake2b(seed, digest_size=16).digest()
    return PdfString(digest, hexform=True)


def _trailer_id(first: Any, seed: bytes) -> PdfArray:
    """Build the two-element ``/ID`` array, preserving ``first`` when it is usable."""
    if not isinstance(first, PdfString):
        first = _derive_id_half(b"zfp-original|" + seed)
    return PdfArray([first, _derive_id_half(seed)])


# --------------------------------------------------------------------------------------
# Writing a body + classic xref from scratch
# --------------------------------------------------------------------------------------


def _emit_full_body(
    objects: Mapping[int, Any],
    version: str,
    generations: Optional[Mapping[int, int]] = None,
) -> Tuple[bytearray, Dict[int, int], int]:
    """Emit a header and every object, returning ``(buffer, offsets, size)``.

    ``size`` is the ``/Size`` the trailer must declare: one past the highest object
    number written.  ``generations`` preserves non-zero generation numbers so that
    references already embedded in the object bodies (``5 2 R``) keep matching the
    objects they name.
    """
    out = bytearray()
    out += b"%PDF-" + str(version).encode("ascii") + b"\n"
    out += BINARY_COMMENT + b"\n"
    offsets: Dict[int, int] = {}
    for num in sorted(objects):
        if num < 1:
            continue
        offsets[num] = len(out)
        out += b"%d %d obj\n" % (num, (generations or {}).get(num, 0))
        out += serialize_object(objects[num])
        out += b"\nendobj\n"
    size = (max(offsets) + 1) if offsets else 1
    return out, offsets, size


def _emit_full_xref(
    out: bytearray,
    offsets: Mapping[int, int],
    size: int,
    generations: Optional[Mapping[int, int]] = None,
) -> int:
    """Append a single-subsection classic xref table.  Returns its byte offset.

    Object numbers below ``size`` with no object get properly chained free entries, so
    the free list terminates at object 0 exactly as the specification requires.
    """
    missing = [n for n in range(1, size) if n not in offsets]
    next_free = {}
    for position, number in enumerate(missing):
        next_free[number] = missing[position + 1] if position + 1 < len(missing) else 0
    head_free = missing[0] if missing else 0

    xref_offset = len(out)
    out += b"xref\n"
    out += b"0 %d\n" % size
    out += _xref_free(head_free)
    for num in range(1, size):
        offset = offsets.get(num)
        if offset is None:
            out += _xref_free(next_free[num])
        else:
            out += _xref_in_use(offset, (generations or {}).get(num, 0))
    return xref_offset


def _emit_trailer(out: bytearray, trailer: PdfDict, xref_offset: int) -> None:
    """Append ``trailer <<...>> startxref <offset> %%EOF``."""
    out += b"trailer\n"
    out += serialize_object(trailer)
    out += b"\nstartxref\n"
    out += b"%d\n" % xref_offset
    out += b"%%EOF\n"


def build_document(
    objects: Mapping[int, Any],
    root: PdfRef,
    *,
    info: Optional[PdfRef] = None,
    version: str = DEFAULT_VERSION,
    extra_trailer: Optional[Mapping[str, Any]] = None,
) -> bytes:
    """Serialize a brand new PDF from an object table.

    Used by :meth:`zfp.pdfio.document.Document.from_pages_blank` and by the synthetic
    corpus generator, both of which need a real parseable file without first having a
    file to update.

    Args:
        objects: Object number (>= 1) to object.  Numbers need not be contiguous.
        root: Reference to the document catalog.
        info: Optional reference to the ``/Info`` dictionary.
        version: PDF version for the ``%PDF-x.y`` header.
        extra_trailer: Additional trailer entries (``/Encrypt``, ``/ID``, ...).

    Returns:
        Complete PDF file bytes.

    Raises:
        PdfWriteError: ``root`` is not an indirect reference.
    """
    if not isinstance(root, PdfRef):
        raise PdfWriteError("build_document requires an indirect /Root reference")
    out, offsets, size = _emit_full_body(objects, version)
    xref_offset = _emit_full_xref(out, offsets, size)

    trailer = PdfDict()
    trailer["Size"] = size
    trailer["Root"] = root
    if info is not None:
        trailer["Info"] = info
    trailer["ID"] = _trailer_id(None, bytes(out))
    for key, value in (extra_trailer or {}).items():
        trailer[key] = value
    _emit_trailer(out, trailer, xref_offset)
    return bytes(out)


# --------------------------------------------------------------------------------------
# PdfWriter
# --------------------------------------------------------------------------------------


class PdfWriter:
    """Accumulates object updates against a parsed :class:`~zfp.pdfio.parser.PdfFile`.

    Nothing is written until :meth:`write_incremental` or :meth:`write_full` is called,
    so a caller can stage an arbitrary number of edits and decide the output strategy
    at the very end.

    Attributes:
        pdf: The source file the updates apply to.
        updates: Object number -> replacement object.  New objects live here too.
        trailer_overrides: Trailer entries to force into the produced revision.
    """

    def __init__(self, pdf: Any) -> None:
        self.pdf = pdf
        self.updates: Dict[int, Any] = {}
        self.trailer_overrides: Dict[str, Any] = {}
        self._next: int = self._first_free_number(pdf)

    # -- object numbering -------------------------------------------------------------
    @staticmethod
    def _first_free_number(pdf: Any) -> int:
        """Return one past the highest object number the source file already uses."""
        highest = 0
        try:
            numbers = pdf.object_numbers()
        except Exception:  # pragma: no cover - defensive against a broken xref
            numbers = ()
        for num in numbers or ():
            if isinstance(num, int) and num > highest:
                highest = num
        trailer = getattr(pdf, "trailer", None)
        if isinstance(trailer, dict):
            size = trailer.get("Size")
            if isinstance(size, int) and not isinstance(size, bool) and size - 1 > highest:
                highest = size - 1
        return highest + 1

    def allocate(self) -> int:
        """Reserve and return the next free object number."""
        num = self._next
        self._next += 1
        return num

    def set_object(self, num: int, obj: Any) -> None:
        """Stage ``obj`` as the new content of object ``num``."""
        num = int(num)
        if num < 1:
            raise PdfWriteError("object numbers start at 1, got %d" % num)
        self.updates[num] = obj
        if num >= self._next:
            self._next = num + 1

    def add_object(self, obj: Any) -> PdfRef:
        """Append ``obj`` as a brand new indirect object and return its reference."""
        num = self.allocate()
        self.updates[num] = obj
        return PdfRef(num, 0)

    def update_trailer(self, key: str, value: Any) -> None:
        """Force a trailer entry into the revision this writer produces."""
        self.trailer_overrides[_key_text(key)] = value

    @property
    def has_changes(self) -> bool:
        """True when anything at all would be appended by an incremental write."""
        return bool(self.updates) or bool(self.trailer_overrides)

    # -- serialization ----------------------------------------------------------------
    def serialize(self, obj: Any) -> bytes:
        """Serialize a single PDF object.  See :func:`serialize_object`."""
        return serialize_object(obj)

    # -- shared trailer construction --------------------------------------------------
    def _source_trailer(self) -> PdfDict:
        trailer = getattr(self.pdf, "trailer", None)
        return trailer if isinstance(trailer, PdfDict) else PdfDict(trailer or {})

    def _declared_size(self) -> int:
        """The ``/Size`` the produced revision must declare."""
        size = self._next
        source = self._source_trailer().get("Size")
        if isinstance(source, int) and not isinstance(source, bool) and source > size:
            size = source
        if self.updates:
            size = max(size, max(self.updates) + 1)
        return size

    def _base_trailer(self, seed: bytes) -> PdfDict:
        """Build the trailer common to both writers (``/Size /Root /Info /ID``)."""
        source = self._source_trailer()
        trailer = PdfDict()
        trailer["Size"] = self._declared_size()
        root = source.get("Root")
        if root is None or isinstance(root, PdfNull):
            raise PdfWriteError("source trailer has no /Root; cannot write a valid PDF")
        trailer["Root"] = root
        info = source.get("Info")
        if info is not None and not isinstance(info, PdfNull):
            trailer["Info"] = info
        encrypt = source.get("Encrypt")
        if encrypt is not None and not isinstance(encrypt, PdfNull):
            trailer["Encrypt"] = encrypt
        original_id = source.get("ID")
        first = original_id[0] if isinstance(original_id, (list, tuple)) and original_id else None
        trailer["ID"] = _trailer_id(first, seed)
        return trailer

    def _apply_overrides(self, trailer: PdfDict) -> PdfDict:
        for key, value in self.trailer_overrides.items():
            trailer[key] = value
        return trailer

    def _generation_for(self, num: int) -> int:
        """The generation an object must keep.

        Replacing an existing object preserves its generation (the number only changes
        when a *freed* slot is reused), and objects this writer created are generation
        zero.  Keeping it right matters because references already embedded in other
        objects spell the generation out.
        """
        xref = getattr(self.pdf, "xref", None)
        if isinstance(xref, dict):
            entry = xref.get(num)
            gen = getattr(entry, "gen", None)
            if isinstance(gen, int) and not isinstance(gen, bool) and 0 <= gen <= 65535:
                return gen
        return 0

    def _previous_startxref(self) -> Optional[int]:
        """Return the byte offset of the source file's most recent xref section."""
        recorded = getattr(self.pdf, "startxref", None)
        if isinstance(recorded, int) and not isinstance(recorded, bool) and recorded >= 0:
            return recorded
        data = getattr(self.pdf, "data", b"") or b""
        index = data.rfind(b"startxref")
        if index < 0:
            return None
        tail = data[index + len(b"startxref") : index + len(b"startxref") + 64]
        digits = bytearray()
        for byte in tail:
            char = bytes((byte,))
            if char.isdigit():
                digits += char
            elif digits:
                break
            elif char not in b" \t\r\n\f\x00":
                break
        if not digits:
            return None
        offset = int(digits)
        return offset if 0 <= offset < len(data) else None

    # -- incremental ------------------------------------------------------------------
    def write_incremental(self) -> bytes:
        """Return the original bytes followed by an appended revision.

        The first ``len(self.pdf.data)`` bytes of the result are byte-for-byte identical
        to the input; everything new is appended.  When nothing has been staged the
        original bytes are returned unchanged rather than growing the file with an empty
        revision.

        Returns:
            The complete updated PDF.

        Raises:
            PdfWriteError: The source trailer has no ``/Root``.
        """
        data = getattr(self.pdf, "data", b"") or b""
        if not self.has_changes:
            return bytes(data)

        out = bytearray(data)
        if out and not out.endswith((b"\n", b"\r")):
            out += b"\n"

        numbers = sorted(self.updates)
        offsets: Dict[int, int] = {}
        for num in numbers:
            offsets[num] = len(out)
            out += b"%d %d obj\n" % (num, self._generation_for(num))
            out += serialize_object(self.updates[num])
            out += b"\nendobj\n"

        xref_offset = len(out)
        out += b"xref\n"
        for run in _contiguous_runs(numbers):
            out += b"%d %d\n" % (run[0], len(run))
            for num in run:
                out += _xref_in_use(offsets[num], self._generation_for(num))

        trailer = self._base_trailer(bytes(out[len(data) :]))
        previous = self._previous_startxref()
        if previous is not None:
            trailer["Prev"] = previous
        self._apply_overrides(trailer)
        _emit_trailer(out, trailer, xref_offset)

        _log.debug(
            "incremental revision: %d objects, %d appended bytes",
            len(numbers),
            len(out) - len(data),
        )
        return bytes(out)

    # -- full rewrite -----------------------------------------------------------------
    def _collect_all_objects(self) -> Dict[int, Any]:
        """Return every in-use object of the source file with the updates applied.

        Object numbers are preserved exactly -- nothing is renumbered, so references
        held by callers stay valid.  An object the parser cannot produce is written as
        ``null``, which keeps the file loadable instead of failing the whole write.
        """
        objects: Dict[int, Any] = {}
        try:
            source_numbers = list(self.pdf.object_numbers() or ())
        except Exception:  # pragma: no cover - defensive
            source_numbers = []
        for num in source_numbers:
            if not isinstance(num, int) or num < 1 or num in self.updates:
                continue
            try:
                value = self.pdf.get_object(num)
            except Exception:
                _log.debug("object %d could not be resolved; writing null", num)
                value = PdfNull.NULL
            if _is_dead_container(value):
                # An /ObjStm or /XRef stream only exists to serve the cross-reference
                # machinery of the revision it came from.  A classic single-revision
                # rewrite re-emits their contents as plain objects, so the containers
                # themselves are dead weight and nothing may reference them.
                continue
            objects[num] = PdfNull.NULL if value is None else value
        for num, value in self.updates.items():
            if num >= 1:
                objects[num] = PdfNull.NULL if value is None else value
        return objects

    def write_full(self) -> bytes:
        """Rewrite the whole document as a single revision with a classic xref.

        Object numbers are preserved, so the output is a drop-in replacement whose
        references match the input.  Unlike :meth:`write_incremental` this discards the
        original byte layout, so use it only when a clean single-revision file is
        wanted (final delivery, or after many stacked updates).

        Returns:
            The complete PDF file bytes.

        Raises:
            PdfWriteError: The source trailer has no ``/Root``.
        """
        if getattr(self.pdf, "is_encrypted", False):
            _log.warning(
                "full rewrite of an encrypted document: object data is written in the "
                "clear while /Encrypt is retained; use the crypt layer to re-encrypt"
            )
        objects = self._collect_all_objects()
        generations = {num: self._generation_for(num) for num in objects}
        version = str(getattr(self.pdf, "version", None) or DEFAULT_VERSION)
        out, offsets, body_size = _emit_full_body(objects, version, generations)
        size = max(body_size, self._declared_size())
        xref_offset = _emit_full_xref(out, offsets, size, generations)

        trailer = self._base_trailer(bytes(out))
        trailer["Size"] = size
        trailer.pop("Prev", None)
        self._apply_overrides(trailer)
        _emit_trailer(out, trailer, xref_offset)

        _log.debug("full rewrite: %d objects, %d bytes", len(objects), len(out))
        return bytes(out)

    def __repr__(self) -> str:
        return "PdfWriter(updates=%d, next=%d)" % (len(self.updates), self._next)
