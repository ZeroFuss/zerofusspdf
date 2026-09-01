"""JSON serialization helpers.

:func:`to_jsonable` converts ZFP's dataclasses, enums, geometry primitives, byte
strings, sets and tuples into structures the stdlib ``json`` module accepts.
Round-tripping is possible for any type registered through :func:`register_decoder`
whose encoded form carries a ``"__type__"`` marker.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from enum import Enum
from typing import Any, Callable, Dict, Mapping, Optional, Type

from .geometry import Matrix, Point, Rect

__all__ = [
    "to_jsonable",
    "dumps",
    "loads",
    "register_decoder",
    "decoders",
    "BYTES_MARKER",
    "TYPE_MARKER",
]

#: Key marking a base64-encoded byte string.
BYTES_MARKER = "__bytes__"
#: Key naming the registered class an object should be decoded back into.
TYPE_MARKER = "__type__"

_DECODERS: Dict[str, Callable[[Mapping[str, Any]], Any]] = {}


def register_decoder(cls: Type[Any], fn: Callable[[Mapping[str, Any]], Any]) -> None:
    """Register ``fn`` as the reconstructor for dicts marked ``{"__type__": cls.__name__}``.

    Args:
        cls: The class whose ``__name__`` is used as the marker value.
        fn: Callable taking the decoded mapping and returning an instance.
    """
    if not callable(fn):
        raise TypeError("decoder for %r must be callable" % (cls,))
    _DECODERS[getattr(cls, "__name__", str(cls))] = fn


def decoders() -> Dict[str, Callable[[Mapping[str, Any]], Any]]:
    """Return a copy of the decoder registry."""
    return dict(_DECODERS)


def _encode_bytes(data: bytes) -> Dict[str, str]:
    """Encode raw bytes as an ASCII-safe base64 payload with a marker."""
    return {BYTES_MARKER: base64.b64encode(bytes(data)).decode("ascii")}


def _sort_key(value: Any) -> str:
    """Deterministic ordering key for set members of mixed types."""
    return "%s:%s" % (type(value).__name__, value)


def to_jsonable(obj: Any) -> Any:
    """Convert ``obj`` into JSON-compatible primitives.

    Handles ``None``/``bool``/``int``/``float``/``str`` verbatim, ``bytes`` as a
    base64 marker dict, :class:`enum.Enum` as its value, :class:`Rect`/:class:`Point`/
    :class:`Matrix` as flat lists, any object exposing ``as_dict()``, dataclass
    instances (recursively), mappings, sets (sorted for determinism) and any other
    iterable as a list. Unknown objects fall back to ``str(obj)``.
    """
    if obj is None:
        return None
    # Enums first: str/int-valued enums would otherwise pass through as themselves.
    if isinstance(obj, Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return _encode_bytes(bytes(obj))
    if isinstance(obj, Rect):
        return obj.as_list()
    if isinstance(obj, Point):
        return list(obj.as_tuple())
    if isinstance(obj, Matrix):
        return list(obj.as_tuple())
    as_dict = getattr(obj, "as_dict", None)
    if callable(as_dict) and not isinstance(obj, type):
        return to_jsonable(as_dict())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(obj, key=_sort_key)]
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    try:
        iterator = iter(obj)
    except TypeError:
        return str(obj)
    return [to_jsonable(v) for v in iterator]


def dumps(obj: Any, indent: Optional[int] = 2) -> str:
    """Serialize ``obj`` to a JSON string with deterministic key ordering."""
    return json.dumps(to_jsonable(obj), indent=indent, sort_keys=True, ensure_ascii=False)


def _object_hook(d: Dict[str, Any]) -> Any:
    """Decode byte markers and registered types while parsing JSON."""
    if len(d) == 1 and BYTES_MARKER in d:
        raw = d[BYTES_MARKER]
        if isinstance(raw, str):
            try:
                return base64.b64decode(raw.encode("ascii"))
            except Exception:
                return d
    marker = d.get(TYPE_MARKER)
    if isinstance(marker, str):
        decoder = _DECODERS.get(marker)
        if decoder is not None:
            return decoder(d)
    return d


def loads(s: str) -> Any:
    """Parse JSON produced by :func:`dumps`, restoring bytes and registered types."""
    return json.loads(s, object_hook=_object_hook)
