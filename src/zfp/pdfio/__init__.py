"""``zfp.pdfio`` -- the dependency-free PDF object layer.

The object model, tokenizer and stream filters are imported eagerly because nothing
else in ZFP works without them.  The higher layers built on top -- ``parser`` (
:class:`PdfFile`), ``writer`` (:class:`PdfWriter`) and ``document`` (:class:`Document`,
:class:`Page`) -- are resolved lazily through :func:`__getattr__`, so importing this
package never drags in (or fails on) a module a caller does not need.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .filters import decode, encode_flate, is_image_filter
from .lexer import Lexer, Token
from .objects import (
    PdfArray,
    PdfDict,
    PdfName,
    PdfNull,
    PdfObject,
    PdfRef,
    PdfStream,
    PdfString,
)

__all__ = [
    "PdfObject",
    "PdfName",
    "PdfRef",
    "PdfString",
    "PdfArray",
    "PdfDict",
    "PdfStream",
    "PdfNull",
    "Lexer",
    "Token",
    "decode",
    "encode_flate",
    "is_image_filter",
]

# Attribute name -> (submodule, attribute in that submodule).  Resolved on first access
# so ``import zfp.pdfio`` stays cheap and never hard-fails on an absent higher layer.
_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PdfFile": (".parser", "PdfFile"),
    "Resolver": (".parser", "Resolver"),
    "PdfWriter": (".writer", "PdfWriter"),
    "Document": (".document", "Document"),
    "Page": (".document", "Page"),
}


def __getattr__(name: str) -> Any:
    """Resolve the lazily exported names from their submodules on first access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    from importlib import import_module

    try:
        module = import_module(module_name, __name__)
    except ImportError as exc:
        raise AttributeError(
            f"{name!r} is provided by {__name__}{module_name}, "
            f"which is not available: {exc}"
        ) from exc
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(__all__) | set(_LAZY_EXPORTS) | set(globals()))
