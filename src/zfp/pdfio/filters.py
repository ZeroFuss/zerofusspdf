"""PDF stream filters, implemented on the standard library alone.

Everything a form-processing pipeline actually needs to read is here: ``FlateDecode``
(with PNG and TIFF predictors), ``LZWDecode``, ``ASCIIHexDecode``, ``ASCII85Decode`` and
``RunLengthDecode``.  The four image codecs (``DCTDecode``, ``JPXDecode``,
``CCITTFaxDecode``, ``JBIG2Decode``) are *not* decoded: :func:`decode` stops when it
reaches one and hands back the bytes, which is exactly what an image consumer wants.

Every decoder is lenient by design.  Real-world PDFs contain truncated Flate streams,
LZW streams that run off the end of their table, and ASCII85 data with no terminator.
A partially recovered stream is worth far more than an exception, so the decoders return
what they could salvage instead of raising.
"""

from __future__ import annotations

import binascii
import zlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .objects import PdfDict, PdfName, PdfNull

__all__ = [
    "IMAGE_FILTERS",
    "FILTER_ALIASES",
    "is_image_filter",
    "normalize_filter_name",
    "decode",
    "decode_one",
    "flate_decode",
    "lzw_decode",
    "ascii_hex_decode",
    "ascii85_decode",
    "run_length_decode",
    "apply_predictor",
    "encode_flate",
    "encode_ascii_hex",
    "encode_ascii85",
    "encode_runlength",
    "encode_lzw",
]

#: Filters whose payload ZFP deliberately leaves encoded.
IMAGE_FILTERS = frozenset(
    {"DCTDecode", "JPXDecode", "CCITTFaxDecode", "JBIG2Decode"}
)

#: Inline-image and legacy abbreviations mapped to their canonical filter names.
FILTER_ALIASES = {
    "AHx": "ASCIIHexDecode",
    "A85": "ASCII85Decode",
    "LZW": "LZWDecode",
    "Fl": "FlateDecode",
    "RL": "RunLengthDecode",
    "CCF": "CCITTFaxDecode",
    "DCT": "DCTDecode",
}

_WHITESPACE = b"\x00\t\n\x0c\r "


def normalize_filter_name(name: Any) -> str:
    """Return the canonical filter name for a name object, string or abbreviation."""
    if isinstance(name, PdfName):
        text = name.value
    elif isinstance(name, (bytes, bytearray)):
        text = bytes(name).decode("latin-1")
    elif isinstance(name, str):
        text = name
    elif name is None or isinstance(name, PdfNull):
        return ""
    else:
        text = str(name)
    if text.startswith("/"):
        text = text[1:]
    return FILTER_ALIASES.get(text, text)


def is_image_filter(name: Any) -> bool:
    """True for the image codecs whose payload is returned untouched."""
    return normalize_filter_name(name) in IMAGE_FILTERS


# --------------------------------------------------------------------------------------
# Decode parameter access
# --------------------------------------------------------------------------------------


def _parm_int(parms: Any, key: str, default: int) -> int:
    """Read an integer decode parameter, tolerating missing or malformed values."""
    if parms is None or isinstance(parms, PdfNull):
        return default
    if isinstance(parms, PdfDict):
        value = parms.get(key, None)
    elif isinstance(parms, dict):
        value = parms.get(key, parms.get("/" + key, None))
    else:
        value = getattr(parms, key, None)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


# --------------------------------------------------------------------------------------
# FlateDecode
# --------------------------------------------------------------------------------------


def _inflate_best_effort(data: bytes) -> bytes:
    """Inflate ``data``, recovering as much as possible from a corrupt stream.

    Tries, in order: a clean zlib inflate; a chunked inflate that keeps whatever came
    out before the error; raw deflate (some producers omit the zlib header); and the
    same two after stripping leading white space.  Returns the longest result found,
    which may legitimately be empty.
    """
    if not data:
        return b""
    try:
        return zlib.decompress(data)  # the overwhelmingly common case
    except zlib.error:
        pass
    best = b""
    candidates = [data]
    stripped = data.lstrip(_WHITESPACE)
    if stripped and stripped != data:
        candidates.append(stripped)
    for candidate in candidates:
        if not candidate:
            continue
        for wbits in (15, -15, 47):
            try:
                out = zlib.decompress(candidate, wbits)
            except zlib.error:
                out = _inflate_chunked(candidate, wbits)
            if len(out) > len(best):
                best = out
        if best:
            break
    return best


def _inflate_chunked(data: bytes, wbits: int) -> bytes:
    """Feed ``data`` to a decompressor in chunks, keeping everything decoded so far."""
    obj = zlib.decompressobj(wbits)
    out = bytearray()
    step = 512
    index = 0
    n = len(data)
    while index < n:
        chunk = data[index : index + step]
        try:
            out += obj.decompress(chunk)
        except zlib.error:
            # Squeeze the final bytes out one at a time before giving up.
            for offset in range(len(chunk)):
                try:
                    out += obj.decompress(chunk[offset : offset + 1])
                except zlib.error:
                    return bytes(out)
            return bytes(out)
        index += step
    try:
        out += obj.flush()
    except zlib.error:
        pass
    return bytes(out)


def flate_decode(data: bytes, parms: Any = None) -> bytes:
    """``FlateDecode``: zlib inflate followed by any ``/Predictor`` post-processing."""
    return apply_predictor(_inflate_best_effort(bytes(data)), parms)


def encode_flate(data: bytes, level: int = 9) -> bytes:
    """Deflate ``data`` with a zlib header.  Deterministic for a given ``level``."""
    return zlib.compress(bytes(data), level)


# --------------------------------------------------------------------------------------
# LZWDecode
# --------------------------------------------------------------------------------------

_LZW_CLEAR = 256
_LZW_EOD = 257


def _lzw_base_table() -> List[bytes]:
    table = [bytes([i]) for i in range(256)]
    table.append(b"")  # 256 clear
    table.append(b"")  # 257 end-of-data
    return table


def _lzw_width(table_size: int, early: int) -> int:
    """Code width for a table of ``table_size`` entries under the EarlyChange rule."""
    size = table_size + early
    if size >= 2048:
        return 12
    if size >= 1024:
        return 11
    if size >= 512:
        return 10
    return 9


def lzw_decode(data: bytes, parms: Any = None, early: Optional[int] = None) -> bytes:
    """``LZWDecode``: variable-width LZW with the PDF ``EarlyChange`` convention.

    Codes are 9 to 12 bits, MSB first.  ``EarlyChange`` (default 1) makes the width grow
    one code sooner than the table strictly requires; passing ``early=0`` (or
    ``/EarlyChange 0``) selects the other convention.  Corrupt input truncates the
    output rather than raising.
    """
    data = bytes(data)
    if early is None:
        early = 1 if _parm_int(parms, "EarlyChange", 1) else 0
    early = 1 if early else 0

    table = _lzw_base_table()
    out = bytearray()
    width = 9
    previous: Optional[bytes] = None
    bitpos = 0
    total_bits = len(data) * 8

    while bitpos + width <= total_bits:
        byte_index = bitpos >> 3
        window = data[byte_index : byte_index + 3]
        if len(window) < 3:
            window = window + b"\x00" * (3 - len(window))
        packed = int.from_bytes(window, "big")
        shift = 24 - (bitpos & 7) - width
        code = (packed >> shift) & ((1 << width) - 1)
        bitpos += width

        if code == _LZW_EOD:
            break
        if code == _LZW_CLEAR:
            table = _lzw_base_table()
            width = 9
            previous = None
            continue

        if previous is None:
            if code >= len(table):
                break
            entry = table[code]
            out += entry
            previous = entry
        else:
            if code < len(table):
                entry = table[code]
            elif code == len(table):
                entry = previous + previous[:1]
            else:
                break  # corrupt: a code beyond the next free slot cannot be resolved
            out += entry
            if len(table) < 4096:
                table.append(previous + entry[:1])
            previous = entry
        width = _lzw_width(len(table), early)

    return apply_predictor(bytes(out), parms)


def encode_lzw(data: bytes, early: int = 1) -> bytes:
    """LZW-compress ``data``.  Provided so writers and tests can round-trip a stream."""
    data = bytes(data)
    table: Dict[bytes, int] = {bytes([i]): i for i in range(256)}
    next_code = 258
    width = 9
    bits: List[int] = []

    def emit(code: int, nbits: int) -> None:
        for shift in range(nbits - 1, -1, -1):
            bits.append((code >> shift) & 1)

    early = 1 if early else 0
    emit(_LZW_CLEAR, width)
    window = b""
    for byte in data:
        candidate = window + bytes([byte])
        if candidate in table:
            window = candidate
            continue
        emit(table[window], width)
        if next_code < 4096:
            table[candidate] = next_code
            next_code += 1
            # The decoder's table always lags the encoder's by exactly one entry (it
            # cannot add a row until it has seen the following code), so the width for
            # the next code is derived from ``next_code - 1``.
            width = _lzw_width(next_code - 1, early)
        else:
            emit(_LZW_CLEAR, width)
            table = {bytes([i]): i for i in range(256)}
            next_code = 258
            width = 9
        window = bytes([byte])
    if window:
        emit(table[window], width)
    emit(_LZW_EOD, width)

    while len(bits) % 8:
        bits.append(0)
    out = bytearray()
    for index in range(0, len(bits), 8):
        value = 0
        for bit in bits[index : index + 8]:
            value = (value << 1) | bit
        out.append(value)
    return bytes(out)


# --------------------------------------------------------------------------------------
# ASCIIHexDecode / ASCII85Decode / RunLengthDecode
# --------------------------------------------------------------------------------------


def ascii_hex_decode(data: bytes, parms: Any = None) -> bytes:
    """``ASCIIHexDecode``: hex digits up to ``>``; an odd final digit is padded with 0."""
    digits = bytearray()
    for byte in bytes(data):
        if byte == 0x3E:  # '>'
            break
        if 0x30 <= byte <= 0x39 or 0x41 <= byte <= 0x46 or 0x61 <= byte <= 0x66:
            digits.append(byte)
        # white space and stray bytes are simply skipped
    if len(digits) % 2:
        digits.append(0x30)
    try:
        return binascii.unhexlify(bytes(digits))
    except (binascii.Error, ValueError):  # pragma: no cover - digits are valid hex
        return b""


def encode_ascii_hex(data: bytes) -> bytes:
    """Encode ``data`` as upper-case hex terminated by ``>``."""
    return binascii.hexlify(bytes(data)).upper() + b">"


def ascii85_decode(data: bytes, parms: Any = None) -> bytes:
    """``ASCII85Decode``: base-85 with the ``z`` all-zero shorthand and ``~>`` terminator.

    A leading ``<~`` (the Adobe ``btoa`` convention) is skipped if present, a missing
    terminator is tolerated, and out-of-range characters are ignored.
    """
    data = bytes(data)
    if data.startswith(b"<~"):
        data = data[2:]
    out = bytearray()
    group: List[int] = []
    index = 0
    n = len(data)
    while index < n:
        byte = data[index]
        index += 1
        if byte in _WHITESPACE:
            continue
        if byte == 0x7E:  # '~' begins the '~>' terminator
            break
        if byte == 0x7A and not group:  # 'z' == four zero bytes
            out += b"\x00\x00\x00\x00"
            continue
        if byte < 0x21 or byte > 0x75:
            continue  # not a base-85 digit; be lenient
        group.append(byte - 0x21)
        if len(group) == 5:
            out += _ascii85_group(group, 4)
            group = []
    if group:
        count = len(group) - 1
        while len(group) < 5:
            group.append(84)  # pad with 'u'
        out += _ascii85_group(group, count)
    return bytes(out)


def _ascii85_group(group: Sequence[int], count: int) -> bytes:
    value = 0
    for digit in group:
        value = value * 85 + digit
    value &= 0xFFFFFFFF
    return value.to_bytes(4, "big")[:count]


def encode_ascii85(data: bytes, terminator: bytes = b"~>") -> bytes:
    """Encode ``data`` as ASCII85, using ``z`` for all-zero groups."""
    data = bytes(data)
    out = bytearray()
    for index in range(0, len(data) - len(data) % 4, 4):
        chunk = data[index : index + 4]
        if chunk == b"\x00\x00\x00\x00":
            out.append(0x7A)
            continue
        out += _ascii85_digits(int.from_bytes(chunk, "big"), 5)
    remainder = len(data) % 4
    if remainder:
        chunk = data[len(data) - remainder :] + b"\x00" * (4 - remainder)
        out += _ascii85_digits(int.from_bytes(chunk, "big"), 5)[: remainder + 1]
    return bytes(out) + terminator


def _ascii85_digits(value: int, count: int) -> bytes:
    digits = bytearray(count)
    for position in range(count - 1, -1, -1):
        digits[position] = 0x21 + (value % 85)
        value //= 85
    return bytes(digits)


def run_length_decode(data: bytes, parms: Any = None) -> bytes:
    """``RunLengthDecode``: ``0..127`` copies n+1 literals, ``129..255`` repeats, ``128`` ends."""
    data = bytes(data)
    out = bytearray()
    index = 0
    n = len(data)
    while index < n:
        length = data[index]
        index += 1
        if length == 128:
            break
        if length < 128:
            count = length + 1
            out += data[index : index + count]
            index += count
        else:
            if index >= n:
                break
            out += bytes([data[index]]) * (257 - length)
            index += 1
    return bytes(out)


def encode_runlength(data: bytes) -> bytes:
    """Encode ``data`` with run-length compression, terminated by the ``128`` EOD byte."""
    data = bytes(data)
    out = bytearray()
    index = 0
    n = len(data)
    while index < n:
        run = 1
        while run < 128 and index + run < n and data[index + run] == data[index]:
            run += 1
        if run > 1:
            out.append(257 - run)
            out.append(data[index])
            index += run
            continue
        # Gather literals up to the point where a run of 3 or more starts.
        start = index
        while index < n and index - start < 128:
            if index + 2 < n and data[index] == data[index + 1] == data[index + 2]:
                break
            index += 1
        count = index - start
        out.append(count - 1)
        out += data[start:index]
    out.append(128)
    return bytes(out)


# --------------------------------------------------------------------------------------
# Predictors
# --------------------------------------------------------------------------------------


def _paeth(a: int, b: int, c: int) -> int:
    """The PNG Paeth predictor: pick whichever of left/up/upper-left is closest to a+b-c."""
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def apply_predictor(data: bytes, parms: Any = None) -> bytes:
    """Undo the ``/Predictor`` transform described by a ``/DecodeParms`` dictionary.

    ``/Predictor`` 1 (or absent) is a no-op, 2 is the TIFF horizontal differencing
    predictor, and 10..15 are the PNG row filters (the exact value does not matter: each
    row carries its own filter type byte).
    """
    predictor = _parm_int(parms, "Predictor", 1)
    if predictor <= 1:
        return data
    colors = max(1, _parm_int(parms, "Colors", 1))
    bpc = _parm_int(parms, "BitsPerComponent", 8)
    if bpc not in (1, 2, 4, 8, 16):
        bpc = 8
    columns = max(1, _parm_int(parms, "Columns", 1))
    if predictor == 2:
        return _tiff_predictor(data, colors, bpc, columns)
    return _png_predictor(data, colors, bpc, columns)


def _png_predictor(data: bytes, colors: int, bpc: int, columns: int) -> bytes:
    row_length = (columns * colors * bpc + 7) // 8
    if row_length <= 0:
        return data
    bytes_per_pixel = max(1, (colors * bpc + 7) // 8)
    out = bytearray()
    previous = bytearray(row_length)
    index = 0
    n = len(data)
    while index < n:
        filter_type = data[index]
        index += 1
        row = bytearray(data[index : index + row_length])
        index += row_length
        if not row:
            break
        if len(row) < row_length:
            row += bytearray(row_length - len(row))  # tolerate a truncated final row
        if filter_type == 1:  # Sub
            for i in range(bytes_per_pixel, row_length):
                row[i] = (row[i] + row[i - bytes_per_pixel]) & 0xFF
        elif filter_type == 2:  # Up
            for i in range(row_length):
                row[i] = (row[i] + previous[i]) & 0xFF
        elif filter_type == 3:  # Average
            for i in range(row_length):
                left = row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                row[i] = (row[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            for i in range(row_length):
                left = row[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                upper_left = previous[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                row[i] = (row[i] + _paeth(left, previous[i], upper_left)) & 0xFF
        # filter_type 0 (None) and anything unknown are left alone
        out += row
        previous = row
    return bytes(out)


def _tiff_predictor(data: bytes, colors: int, bpc: int, columns: int) -> bytes:
    row_length = (columns * colors * bpc + 7) // 8
    if row_length <= 0:
        return data
    out = bytearray(data)
    row_count = len(out) // row_length
    if bpc == 8:
        for row in range(row_count):
            base = row * row_length
            for i in range(colors, row_length):
                out[base + i] = (out[base + i] + out[base + i - colors]) & 0xFF
        return bytes(out)
    if bpc == 16:
        stride = colors * 2
        for row in range(row_count):
            base = row * row_length
            for i in range(stride, row_length - 1, 2):
                previous = (out[base + i - stride] << 8) | out[base + i - stride + 1]
                current = (out[base + i] << 8) | out[base + i + 1]
                value = (current + previous) & 0xFFFF
                out[base + i] = value >> 8
                out[base + i + 1] = value & 0xFF
        return bytes(out)
    # Sub-byte samples: unpack the row, difference it, repack it.
    mask = (1 << bpc) - 1
    samples_per_row = columns * colors
    for row in range(row_count):
        base = row * row_length
        chunk = out[base : base + row_length]
        samples = _unpack_bits(chunk, bpc, samples_per_row)
        for i in range(colors, samples_per_row):
            samples[i] = (samples[i] + samples[i - colors]) & mask
        packed = _pack_bits(samples, bpc, row_length)
        out[base : base + row_length] = packed
    return bytes(out)


def _unpack_bits(chunk: bytes, bpc: int, count: int) -> List[int]:
    values: List[int] = []
    mask = (1 << bpc) - 1
    for index in range(count):
        bit = index * bpc
        byte_index = bit >> 3
        if byte_index >= len(chunk):
            values.append(0)
            continue
        shift = 8 - (bit & 7) - bpc
        values.append((chunk[byte_index] >> shift) & mask)
    return values


def _pack_bits(values: Sequence[int], bpc: int, row_length: int) -> bytearray:
    out = bytearray(row_length)
    for index, value in enumerate(values):
        bit = index * bpc
        byte_index = bit >> 3
        if byte_index >= row_length:
            break
        shift = 8 - (bit & 7) - bpc
        out[byte_index] |= (value & ((1 << bpc) - 1)) << shift
    return out


# --------------------------------------------------------------------------------------
# The dispatcher
# --------------------------------------------------------------------------------------

_DECODERS: Dict[str, Callable[[bytes, Any], bytes]] = {
    "FlateDecode": flate_decode,
    "LZWDecode": lzw_decode,
    "ASCIIHexDecode": ascii_hex_decode,
    "ASCII85Decode": ascii85_decode,
    "RunLengthDecode": run_length_decode,
    "Crypt": lambda data, parms: data,  # identity; real decryption happens in crypt.py
}


def decode_one(data: bytes, name: Any, parms: Any = None) -> bytes:
    """Apply a single named filter.  Unknown and image filters return ``data`` unchanged."""
    canonical = normalize_filter_name(name)
    decoder = _DECODERS.get(canonical)
    if decoder is None:
        return data
    return decoder(data, parms)


def decode(
    data: bytes,
    filters: Optional[Iterable[Any]] = None,
    parms: Optional[Sequence[Any]] = None,
) -> bytes:
    """Apply a whole ``/Filter`` chain to ``data``.

    ``filters`` may be a list of names, a single name, or ``None``; ``parms`` is the
    matching ``/DecodeParms`` list and may be shorter (or absent).  Decoding stops and
    returns the bytes as they stand as soon as an image codec or an unrecognized filter
    is reached, so ``decode`` is always safe to call on any stream.
    """
    if filters is None:
        return data
    if isinstance(filters, (str, bytes, bytearray, PdfName)):
        names: List[Any] = [filters]
    else:
        names = list(filters)
    parm_list: List[Any] = list(parms) if parms else []
    out = bytes(data)
    for index, name in enumerate(names):
        canonical = normalize_filter_name(name)
        if not canonical:
            continue
        if canonical in IMAGE_FILTERS:
            return out
        decoder = _DECODERS.get(canonical)
        if decoder is None:
            return out
        parm = parm_list[index] if index < len(parm_list) else None
        if isinstance(parm, PdfNull):
            parm = None
        out = decoder(out, parm)
    return out
