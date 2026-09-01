"""The dependency-free PDF object model.

Every value that can appear inside a PDF file body is represented here:

===========================  =========================================
PDF syntax                   Python representation
===========================  =========================================
``null``                     :data:`PdfNull.NULL`
``true`` / ``false``         :class:`bool`
``42`` / ``-1.5``            :class:`int` / :class:`float`
``/Name``                    :class:`PdfName`
``(text)`` / ``<AB>``        :class:`PdfString`
``[1 2 3]``                  :class:`PdfArray` (a ``list`` subclass)
``<< /K /V >>``              :class:`PdfDict` (a ``dict`` subclass)
``12 0 R``                   :class:`PdfRef`
``<<...>> stream ... endstream``  :class:`PdfStream`
===========================  =========================================

Two conventions run through the whole package:

* names never carry their leading ``/`` in Python; ``PdfName('Type').value == 'Type'``
  and :class:`PdfDict` keys are plain :class:`str` for the same reason;
* strings keep their **raw** bytes.  Text decoding (UTF-16BE / PDFDocEncoding) is an
  explicit call to :meth:`PdfString.text`, never an implicit guess.
"""

from __future__ import annotations

import binascii
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, List, Mapping, Optional, Union

__all__ = [
    "PdfObject",
    "PdfNull",
    "PdfName",
    "PdfRef",
    "PdfString",
    "PdfArray",
    "PdfDict",
    "PdfStream",
    "pdfdoc_decode",
    "pdfdoc_encode",
    "resolve_with",
    "as_list",
    "make_dict",
    "make_array",
]

# A resolver is anything that turns a PdfRef into the object it points at: either an
# object exposing ``.resolve(obj)`` (``zfp.pdfio.parser.Resolver``) or a bare callable.
# Typed as ``Any`` on purpose so this module never imports the parser.
ResolverLike = Any


def resolve_with(value: Any, resolver: ResolverLike = None) -> Any:
    """Return ``value`` with indirect references followed, if a resolver was supplied.

    Accepts either a ``Resolver`` protocol object or a plain callable, and is a no-op
    when ``resolver`` is ``None`` so every call site can stay resolver-agnostic.
    """
    if resolver is None:
        return value
    method = getattr(resolver, "resolve", None)
    if callable(method):
        return method(value)
    if callable(resolver):
        return resolver(value)
    return value


class PdfObject:
    """Marker base class shared by every non-primitive PDF object."""

    __slots__ = ()


class PdfNull(PdfObject):
    """The PDF ``null`` object.

    A singleton: ``PdfNull() is PdfNull.NULL``.  It is falsy and compares equal to
    Python's ``None`` so ``obj == None`` style checks keep working across the boundary
    between "key absent" and "key present but null".
    """

    NULL: ClassVar[PdfNull]
    _instance: ClassVar[Optional[PdfNull]] = None

    def __new__(cls) -> PdfNull:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "PdfNull.NULL"

    def __str__(self) -> str:
        return "null"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return other is None or isinstance(other, PdfNull)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(None)


PdfNull.NULL = PdfNull()


# --------------------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------------------

# Bytes that must be written as ``#xx`` inside a name token: whitespace, the delimiters,
# and ``#`` itself.  Everything outside 0x21..0x7e is escaped as well.
_NAME_SPECIALS = frozenset(b"()<>[]{}/%#")


@dataclass(frozen=True)
class PdfName(PdfObject):
    """A PDF name object, stored **without** its leading ``/``.

    ``PdfName('Type').value == 'Type'`` while ``str(PdfName('Type')) == '/Type'``.
    Frozen and hashable so names work as dictionary keys and set members.
    """

    value: str

    def __str__(self) -> str:
        return "/" + self.value

    def __bool__(self) -> bool:
        # ``PdfName('')`` is a legal (if useless) name; treat every name as truthy so
        # ``if page.get('Type'):`` does not silently misfire.
        return True

    @staticmethod
    def escape(value: str) -> bytes:
        """Return the on-wire body of a name (no leading ``/``) with ``#xx`` escaping."""
        out = bytearray()
        for byte in value.encode("utf-8"):
            if byte < 0x21 or byte > 0x7E or byte in _NAME_SPECIALS:
                out += b"#%02X" % byte
            else:
                out.append(byte)
        return bytes(out)

    @property
    def encoded(self) -> bytes:
        """The complete on-wire token, leading ``/`` included: ``/A#20B``."""
        return b"/" + self.escape(self.value)

    @staticmethod
    def decode(raw: Union[str, bytes, bytearray, PdfName]) -> PdfName:
        """Build a name from its on-wire spelling, resolving ``#xx`` escapes.

        The leading ``/`` is optional.  A malformed escape (``#`` not followed by two
        hex digits) is kept verbatim rather than raising, because broken producers are
        common and a name is never worth failing a whole document over.
        """
        if isinstance(raw, PdfName):
            return raw
        if isinstance(raw, str):
            data = raw.encode("utf-8", "surrogateescape")
        else:
            data = bytes(raw)
        if data.startswith(b"/"):
            data = data[1:]
        if b"#" not in data:
            return PdfName(_decode_name_bytes(data))
        out = bytearray()
        i = 0
        n = len(data)
        while i < n:
            byte = data[i]
            if byte == 0x23 and i + 2 < n:
                try:
                    out.append(int(data[i + 1 : i + 3], 16))
                except ValueError:
                    out.append(byte)
                    i += 1
                    continue
                i += 3
                continue
            out.append(byte)
            i += 1
        return PdfName(_decode_name_bytes(bytes(out)))


def _decode_name_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


@dataclass(frozen=True)
class PdfRef(PdfObject):
    """An indirect reference, ``num gen R``."""

    num: int
    gen: int = 0

    def __str__(self) -> str:
        return f"{self.num} {self.gen} R"

    @property
    def encoded(self) -> bytes:
        """The on-wire token, e.g. ``b'12 0 R'``."""
        return b"%d %d R" % (self.num, self.gen)

    def as_tuple(self) -> tuple:
        """``(num, gen)`` -- the key shape used by xref tables."""
        return (self.num, self.gen)


# --------------------------------------------------------------------------------------
# Strings
# --------------------------------------------------------------------------------------

# PDFDocEncoding (PDF 32000-1 Annex D.2).  It agrees with Latin-1 everywhere except the
# ranges below, so the table is built from Latin-1 and then patched.
_PDFDOC_OVERRIDES = {
    0x18: "˘",  # breve
    0x19: "ˇ",  # caron
    0x1A: "ˆ",  # modifier circumflex
    0x1B: "˙",  # dot above
    0x1C: "˝",  # double acute
    0x1D: "˛",  # ogonek
    0x1E: "˚",  # ring above
    0x1F: "˜",  # small tilde
    0x80: "•",  # bullet
    0x81: "†",  # dagger
    0x82: "‡",  # double dagger
    0x83: "…",  # ellipsis
    0x84: "—",  # em dash
    0x85: "–",  # en dash
    0x86: "ƒ",  # florin
    0x87: "⁄",  # fraction slash
    0x88: "‹",  # single left angle quote
    0x89: "›",  # single right angle quote
    0x8A: "−",  # minus sign
    0x8B: "‰",  # per mille
    0x8C: "„",  # double low-9 quote
    0x8D: "“",  # left double quote
    0x8E: "”",  # right double quote
    0x8F: "‘",  # left single quote
    0x90: "’",  # right single quote
    0x91: "‚",  # single low-9 quote
    0x92: "™",  # trade mark
    0x93: "ﬁ",  # fi ligature
    0x94: "ﬂ",  # fl ligature
    0x95: "Ł",  # Lslash
    0x96: "Œ",  # OE
    0x97: "Š",  # Scaron
    0x98: "Ÿ",  # Ydieresis
    0x99: "Ž",  # Zcaron
    0x9A: "ı",  # dotlessi
    0x9B: "ł",  # lslash
    0x9C: "œ",  # oe
    0x9D: "š",  # scaron
    0x9E: "ž",  # zcaron
    0xA0: "€",  # Euro sign (PDFDocEncoding only; Latin-1 has NBSP here)
}

_PDFDOC_CHARS: List[str] = [chr(i) for i in range(256)]
for _code, _char in _PDFDOC_OVERRIDES.items():
    _PDFDOC_CHARS[_code] = _char
_PDFDOC_REVERSE = {c: i for i, c in enumerate(_PDFDOC_CHARS)}

_UTF16_BE_BOM = b"\xfe\xff"
_UTF16_LE_BOM = b"\xff\xfe"
_UTF8_BOM = b"\xef\xbb\xbf"

_LITERAL_ESCAPES = {
    0x0A: b"\\n",
    0x0D: b"\\r",
    0x09: b"\\t",
    0x08: b"\\b",
    0x0C: b"\\f",
    0x28: b"\\(",
    0x29: b"\\)",
    0x5C: b"\\\\",
}


def pdfdoc_decode(data: bytes) -> str:
    """Decode PDFDocEncoded bytes to text.  Never raises; every byte has a mapping."""
    chars = _PDFDOC_CHARS
    return "".join([chars[b] for b in data])


def pdfdoc_encode(text: str) -> Optional[bytes]:
    """Encode text as PDFDocEncoding, or return ``None`` when a character has no slot."""
    out = bytearray()
    reverse = _PDFDOC_REVERSE
    for char in text:
        code = reverse.get(char)
        if code is None:
            return None
        out.append(code)
    return bytes(out)


class PdfString(PdfObject):
    """A PDF string, stored as the raw bytes it carries plus how it was written.

    ``hexform`` records whether the string came from (or should go out as) ``<AB CD>``
    rather than ``(literal)``.  It is a serialization detail only: two strings with the
    same ``raw`` compare equal regardless of it.
    """

    __slots__ = ("raw", "hexform")

    def __init__(self, raw: bytes, hexform: bool = False) -> None:
        if isinstance(raw, str):
            encoded = pdfdoc_encode(raw)
            raw = encoded if encoded is not None else raw.encode("utf-8")
        self.raw: bytes = bytes(raw)
        self.hexform: bool = bool(hexform)

    # -- text -------------------------------------------------------------------------
    def text(self) -> str:
        """Decode the raw bytes to text.

        A leading UTF-16BE byte-order mark selects UTF-16; UTF-16LE and UTF-8 marks are
        honoured too (out of spec, but produced in the wild).  Everything else is
        PDFDocEncoding.
        """
        raw = self.raw
        if raw.startswith(_UTF16_BE_BOM):
            body = raw[2:]
            if len(body) % 2:
                body = body[:-1]
            return body.decode("utf-16-be", "replace")
        if raw.startswith(_UTF16_LE_BOM):
            body = raw[2:]
            if len(body) % 2:
                body = body[:-1]
            return body.decode("utf-16-le", "replace")
        if raw.startswith(_UTF8_BOM):
            return raw[3:].decode("utf-8", "replace")
        return pdfdoc_decode(raw)

    @staticmethod
    def from_text(s: str) -> PdfString:
        """Build a string from text, choosing the narrowest faithful representation.

        Pure ASCII becomes a plain literal string; anything else becomes UTF-16BE with
        a byte-order mark, which every conforming reader understands.
        """
        try:
            return PdfString(s.encode("ascii"))
        except UnicodeEncodeError:
            return PdfString(_UTF16_BE_BOM + s.encode("utf-16-be", "surrogatepass"))

    # -- serialization ----------------------------------------------------------------
    def serialize(self) -> bytes:
        """Return the on-wire bytes, ``<hex>`` when :attr:`hexform` else ``(literal)``."""
        if self.hexform:
            return b"<" + binascii.hexlify(self.raw).upper() + b">"
        out = bytearray(b"(")
        for byte in self.raw:
            escape = _LITERAL_ESCAPES.get(byte)
            if escape is not None:
                out += escape
            elif byte < 0x20 or byte == 0x7F:
                out += b"\\%03o" % byte
            else:
                out.append(byte)
        out += b")"
        return bytes(out)

    @property
    def encoded(self) -> bytes:
        """Alias for :meth:`serialize`, matching :attr:`PdfName.encoded`."""
        return self.serialize()

    # -- protocol ---------------------------------------------------------------------
    def __bytes__(self) -> bytes:
        return self.raw

    def __len__(self) -> int:
        return len(self.raw)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PdfString):
            return self.raw == other.raw
        if isinstance(other, (bytes, bytearray)):
            return self.raw == bytes(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self) -> int:
        return hash(self.raw)

    def __repr__(self) -> str:
        return f"PdfString({self.raw!r}, hexform={self.hexform!r})"

    def __str__(self) -> str:
        return self.text()


# --------------------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------------------


class PdfArray(PdfObject, list):
    """A PDF array.  A plain ``list`` that is also a :class:`PdfObject`."""

    def resolved(self, resolver: ResolverLike = None) -> PdfArray:
        """Return a copy with every top-level element run through ``resolver``."""
        return PdfArray([resolve_with(item, resolver) for item in self])

    def __repr__(self) -> str:
        return f"PdfArray({list.__repr__(self)})"


def _coerce_key(key: Any) -> str:
    """Normalize a dictionary key to a plain ``str`` without a leading ``/``."""
    if isinstance(key, PdfName):
        return key.value
    if isinstance(key, str):
        return key[1:] if key.startswith("/") else key
    if isinstance(key, (bytes, bytearray)):
        return PdfName.decode(bytes(key)).value
    return str(key)


class PdfDict(PdfObject, dict):
    """A PDF dictionary keyed by plain ``str`` (no leading ``/``).

    :class:`PdfName` keys, ``'/Type'``-style strings and ``bytes`` are all accepted and
    coerced on the way in, so ``d[PdfName('Type')]``, ``d['/Type']`` and ``d['Type']``
    address the same slot.  The ``get_*`` readers additionally take an optional resolver
    so callers can pull typed values straight out of a dictionary full of references.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        if args or kwargs:
            self.update(dict(*args, **kwargs))

    # -- key coercion -----------------------------------------------------------------
    def __setitem__(self, key: Any, value: Any) -> None:
        dict.__setitem__(self, _coerce_key(key), value)

    def __getitem__(self, key: Any) -> Any:
        return dict.__getitem__(self, _coerce_key(key))

    def __delitem__(self, key: Any) -> None:
        dict.__delitem__(self, _coerce_key(key))

    def __contains__(self, key: Any) -> bool:
        return dict.__contains__(self, _coerce_key(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return dict.get(self, _coerce_key(key), default)

    def pop(self, key: Any, *default: Any) -> Any:
        return dict.pop(self, _coerce_key(key), *default)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        return dict.setdefault(self, _coerce_key(key), default)

    def update(self, other: Any = (), **kwargs: Any) -> None:  # type: ignore[override]
        if hasattr(other, "keys"):
            for key in other.keys():
                self[key] = other[key]
        else:
            for key, value in other:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def copy(self) -> PdfDict:
        return PdfDict(self)

    def __repr__(self) -> str:
        return f"PdfDict({dict.__repr__(self)})"

    # -- typed readers ----------------------------------------------------------------
    def resolved_get(self, key: Any, default: Any = None, resolver: ResolverLike = None) -> Any:
        """``get`` with the result passed through ``resolver`` (and ``null`` folded away)."""
        value = resolve_with(self.get(key, default), resolver)
        if isinstance(value, PdfNull):
            return default
        return value

    def get_name(self, key: Any, default: Any = None, resolver: ResolverLike = None) -> Optional[str]:
        """Return a name value as a plain ``str`` without ``/``, else ``default``."""
        value = self.resolved_get(key, None, resolver)
        if isinstance(value, PdfName):
            return value.value
        if isinstance(value, str):
            return value[1:] if value.startswith("/") else value
        if isinstance(value, (bytes, bytearray)):
            return PdfName.decode(bytes(value)).value
        return default

    def get_int(self, key: Any, default: Any = None, resolver: ResolverLike = None) -> Optional[int]:
        """Return an integer value, coercing floats and numeric strings."""
        value = self.resolved_get(key, None, resolver)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return default
        return default

    def get_float(self, key: Any, default: Any = None, resolver: ResolverLike = None) -> Optional[float]:
        """Return a float value, coercing integers and numeric strings."""
        value = self.resolved_get(key, None, resolver)
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    def get_bool(self, key: Any, default: Any = None, resolver: ResolverLike = None) -> Optional[bool]:
        """Return a boolean value, else ``default``."""
        value = self.resolved_get(key, None, resolver)
        if isinstance(value, bool):
            return value
        return default

    def get_dict(self, key: Any, default: Any = None, resolver: ResolverLike = None) -> Optional[PdfDict]:
        """Return a dictionary value.  A stream value yields its stream dictionary."""
        value = self.resolved_get(key, None, resolver)
        if isinstance(value, PdfDict):
            return value
        if isinstance(value, PdfStream):
            return value.dict
        if isinstance(value, dict):
            return PdfDict(value)
        return default

    def get_array(self, key: Any, default: Any = None, resolver: ResolverLike = None) -> Optional[PdfArray]:
        """Return an array value, wrapping plain lists and tuples."""
        value = self.resolved_get(key, None, resolver)
        if isinstance(value, PdfArray):
            return value
        if isinstance(value, (list, tuple)):
            return PdfArray(value)
        return default

    def get_text(self, key: Any, default: Any = None, resolver: ResolverLike = None) -> Optional[str]:
        """Return a string value decoded to text via :meth:`PdfString.text`."""
        value = self.resolved_get(key, None, resolver)
        if isinstance(value, PdfString):
            return value.text()
        if isinstance(value, str):
            return value
        if isinstance(value, (bytes, bytearray)):
            return PdfString(bytes(value)).text()
        return default

    def get_stream(self, key: Any, default: Any = None, resolver: ResolverLike = None) -> Optional[PdfStream]:
        """Return a stream value, else ``default``."""
        value = self.resolved_get(key, None, resolver)
        if isinstance(value, PdfStream):
            return value
        return default


# --------------------------------------------------------------------------------------
# Streams
# --------------------------------------------------------------------------------------


class PdfStream(PdfObject):
    """A PDF stream: a dictionary plus the raw (still encoded) bytes that follow it.

    :meth:`decoded` runs the ``/Filter`` chain and caches the result.  The raw bytes are
    never mutated, so a writer can always emit the original substrate untouched.
    """

    __slots__ = ("dict", "_raw", "_decoded")

    def __init__(self, d: PdfDict, raw: bytes) -> None:
        self.dict: PdfDict = d if isinstance(d, PdfDict) else PdfDict(d or {})
        self._raw: bytes = bytes(raw)
        self._decoded: Optional[bytes] = None

    # -- raw bytes --------------------------------------------------------------------
    @property
    def raw(self) -> bytes:
        """The stream's bytes exactly as they appear in the file."""
        return self._raw

    @raw.setter
    def raw(self, value: bytes) -> None:
        self._raw = bytes(value)
        self._decoded = None

    # -- filters ----------------------------------------------------------------------
    def filter_names(self, resolver: ResolverLike = None) -> List[str]:
        """Return the ``/Filter`` chain as plain names, in application order."""
        value = self.dict.resolved_get("Filter", None, resolver)
        if value is None:
            alt = self.dict.resolved_get("F", None, resolver)
            # ``/F`` is a file specification unless it holds names (inline-image style).
            if isinstance(alt, (PdfName, PdfArray, list)):
                value = alt
        return _names_of(value, resolver)

    def decode_parms(self, resolver: ResolverLike = None) -> List[Optional[PdfDict]]:
        """Return the ``/DecodeParms`` chain, padded to the length of the filter chain."""
        value = self.dict.resolved_get("DecodeParms", None, resolver)
        if value is None:
            value = self.dict.resolved_get("DP", None, resolver)
        count = len(self.filter_names(resolver))
        parms: List[Optional[PdfDict]] = []
        if isinstance(value, (PdfArray, list, tuple)):
            for item in value:
                item = resolve_with(item, resolver)
                parms.append(item if isinstance(item, dict) else None)
        elif isinstance(value, dict):
            parms.append(value if isinstance(value, PdfDict) else PdfDict(value))
        while len(parms) < count:
            parms.append(None)
        return parms

    def is_image(self, resolver: ResolverLike = None) -> bool:
        """True when the filter chain contains an image codec ZFP does not decode."""
        from . import filters as _filters

        return any(_filters.is_image_filter(name) for name in self.filter_names(resolver))

    # -- decoding ---------------------------------------------------------------------
    def decoded(self, resolver: ResolverLike = None) -> bytes:
        """Return the stream data with its ``/Filter`` chain applied.

        The result is cached.  If the chain reaches an image codec (``DCTDecode`` and
        friends) decoding stops there and the bytes are handed back as-is, which is what
        a JPEG consumer wants anyway.
        """
        if self._decoded is not None:
            return self._decoded
        from . import filters as _filters

        names = self.filter_names(resolver)
        if not names:
            self._decoded = self._raw
            return self._decoded
        try:
            result = _filters.decode(self._raw, names, self.decode_parms(resolver))
        except Exception:  # pragma: no cover - filters are lenient, this is a backstop
            result = self._raw
        self._decoded = result
        return result

    # -- protocol ---------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._raw)

    def __repr__(self) -> str:
        return f"PdfStream({self.dict!r}, {len(self._raw)} raw bytes)"


def _names_of(value: Any, resolver: ResolverLike = None) -> List[str]:
    """Coerce a name, an array of names, or ``None`` into a list of plain name strings."""
    value = resolve_with(value, resolver)
    if value is None or isinstance(value, PdfNull):
        return []
    if isinstance(value, PdfName):
        return [value.value]
    if isinstance(value, str):
        return [value[1:] if value.startswith("/") else value]
    if isinstance(value, (bytes, bytearray)):
        return [PdfName.decode(bytes(value)).value]
    if isinstance(value, (PdfArray, list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(_names_of(item, resolver))
        return out
    return []


def as_list(value: Any) -> List[Any]:
    """Return ``value`` as a list: arrays pass through, scalars are wrapped, null is empty."""
    if value is None or isinstance(value, PdfNull):
        return []
    if isinstance(value, (PdfArray, list, tuple)):
        return list(value)
    return [value]


def make_dict(mapping: Optional[Mapping[Any, Any]] = None, **kwargs: Any) -> PdfDict:
    """Convenience constructor: ``make_dict(Type='Page', MediaBox=[...])``."""
    result = PdfDict(mapping or {})
    for key, value in kwargs.items():
        result[key] = value
    return result


def make_array(items: Iterable[Any] = ()) -> PdfArray:
    """Convenience constructor for :class:`PdfArray`."""
    return PdfArray(items)
