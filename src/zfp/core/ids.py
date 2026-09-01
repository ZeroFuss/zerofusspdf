"""Deterministic identifiers.

Identifiers must be identical across runs, processes and machines: they are computed
with ``blake2b`` over a canonical textual encoding that never contains memory addresses
or dictionary ordering artifacts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from enum import Enum
from typing import Any, Mapping, Sequence

from .geometry import Matrix, Point, Rect

__all__ = ["stable_id", "candidate_id", "canonical_repr", "DEFAULT_LENGTH"]

DEFAULT_LENGTH = 12

_ADDRESS_RE = re.compile(r"0x[0-9a-fA-F]+")
#: Floats are rounded before hashing so sub-micron jitter cannot change an id.
_FLOAT_DIGITS = 6


def _float_repr(value: float) -> str:
    """Canonical text for a float: fixed precision, no signed zero, inf/nan-safe."""
    if value != value:  # NaN
        return "nan"
    if value in (float("inf"), float("-inf")):
        return "inf" if value > 0 else "-inf"
    rounded = round(float(value), _FLOAT_DIGITS) + 0.0
    return "%.*f" % (_FLOAT_DIGITS, rounded)


def canonical_repr(obj: Any) -> str:
    """Return a stable, address-free textual encoding of ``obj``."""
    if obj is None:
        return "None"
    # Enums first: str/int-valued enums would otherwise be encoded as bare scalars.
    if isinstance(obj, Enum):
        return "%s.%s" % (type(obj).__name__, canonical_repr(obj.value))
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return _float_repr(obj)
    if isinstance(obj, str):
        return "s:" + obj
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return "b:" + bytes(obj).hex()
    if isinstance(obj, Rect):
        return "Rect(%s)" % ",".join(_float_repr(v) for v in obj.as_list())
    if isinstance(obj, Point):
        return "Point(%s)" % ",".join(_float_repr(v) for v in obj.as_tuple())
    if isinstance(obj, Matrix):
        return "Matrix(%s)" % ",".join(_float_repr(v) for v in obj.as_tuple())
    if isinstance(obj, Mapping):
        items = sorted((str(k), v) for k, v in obj.items())
        return "{%s}" % ",".join("%s=%s" % (k, canonical_repr(v)) for k, v in items)
    if isinstance(obj, (set, frozenset)):
        return "{%s}" % ",".join(sorted(canonical_repr(v) for v in obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        parts = [
            "%s=%s" % (f.name, canonical_repr(getattr(obj, f.name)))
            for f in dataclasses.fields(obj)
        ]
        return "%s(%s)" % (type(obj).__name__, ",".join(parts))
    if isinstance(obj, (list, tuple)) or (
        isinstance(obj, Sequence) and not isinstance(obj, (str, bytes))
    ):
        return "[%s]" % ",".join(canonical_repr(v) for v in obj)
    return _ADDRESS_RE.sub("0xX", repr(obj))


def stable_id(*parts: Any, prefix: str = "", length: int = DEFAULT_LENGTH) -> str:
    """Return a deterministic identifier derived from ``parts``.

    Args:
        parts: Any values; encoded with :func:`canonical_repr` and joined with ``|``.
        prefix: Optional human-readable prefix, joined with ``_``.
        length: Number of hex digits kept from the digest (1..64).

    Returns:
        ``"<prefix>_<hex>"`` when a prefix is given, otherwise ``"<hex>"``.
    """
    if length < 1:
        raise ValueError("stable_id length must be >= 1")
    length = min(int(length), 64)
    payload = "|".join(canonical_repr(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=32).hexdigest()[:length]
    return "%s_%s" % (prefix, digest) if prefix else digest


def candidate_id(page: int, rect: Rect, kind: str, prefix: str = "fc") -> str:
    """Return the stable identifier of a field candidate.

    The rectangle is rounded to three decimals first so that geometrically identical
    candidates produced by different detectors collapse onto the same id.
    """
    return stable_id(int(page), rect.normalized().rounded(3), str(kind), prefix=prefix)
