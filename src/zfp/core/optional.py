"""Runtime discovery of optional third-party adapters.

ZFP has zero mandatory third-party dependencies. Every optional backend is looked up
through :func:`optional_import`, which never raises: a missing dependency yields a
falsy :class:`OptionalModule` carrying the import error string, so callers can degrade
instead of crashing.
"""

from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .errors import UnsupportedFeatureError

__all__ = [
    "OptionalModule",
    "optional_import",
    "have",
    "capability_report",
    "clear_cache",
    "EXTRA_FOR_MODULE",
    "OPTIONAL_MODULES",
    "OPTIONAL_BINARIES",
]

#: module name -> the ``pip install zerofusspdf[<extra>]`` extra that provides it.
EXTRA_FOR_MODULE: Dict[str, str] = {
    "pikepdf": "pdfbackends",
    "pypdf": "pdfbackends",
    "fitz": "render",
    "pymupdf": "render",
    "pypdfium2": "render",
    "numpy": "vision",
    "cv2": "vision",
    "PIL": "ocr",
    "pytesseract": "ocr",
    "paddleocr": "paddle",
    "fastapi": "api",
    "uvicorn": "api",
}

#: module name -> distribution name used for metadata version lookups.
_DIST_FOR_MODULE: Dict[str, str] = {
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "fitz": "PyMuPDF",
    "pymupdf": "PyMuPDF",
}

#: The modules reported by ``zfp doctor``.
OPTIONAL_MODULES: Tuple[str, ...] = (
    "pikepdf",
    "pypdf",
    "fitz",
    "pypdfium2",
    "numpy",
    "cv2",
    "PIL",
    "pytesseract",
    "paddleocr",
    "fastapi",
)

#: The external binaries reported by ``zfp doctor``.
OPTIONAL_BINARIES: Tuple[str, ...] = ("tesseract", "pdftoppm", "qpdf")

_CACHE: Dict[Tuple[str, Optional[str]], OptionalModule] = {}


@dataclass(frozen=True)
class OptionalModule:
    """The result of an optional import: either a live module/attribute, or an error."""

    name: str
    module: Optional[Any]
    version: Optional[str]
    error: Optional[str]

    def __bool__(self) -> bool:
        """True when the dependency was imported successfully."""
        return self.module is not None

    def require(self, feature: str) -> Any:
        """Return the module, or raise :class:`UnsupportedFeatureError` naming the extra.

        Args:
            feature: Human-readable name of the feature that needs the dependency.
        """
        if self.module is None:
            extra = EXTRA_FOR_MODULE.get(self.name, "all")
            raise UnsupportedFeatureError(
                "%s requires the optional dependency '%s' "
                "(install it with: pip install 'zerofusspdf[%s]'). Import error: %s"
                % (feature, self.name, extra, self.error or "not installed")
            )
        return self.module


def _detect_version(mod: Any, name: str) -> Optional[str]:
    """Best-effort version string for an imported module."""
    for attr in ("__version__", "VERSION", "version", "__VERSION__"):
        value = getattr(mod, attr, None)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, (tuple, list)) and value:
            return ".".join(str(p) for p in value)
    try:  # pragma: no cover - depends on the installed environment
        from importlib import metadata as _metadata

        return _metadata.version(_DIST_FOR_MODULE.get(name, name))
    except Exception:
        return None


def optional_import(name: str, *, attr: Optional[str] = None) -> OptionalModule:
    """Import ``name`` (optionally an attribute of it) without ever raising.

    Results are cached per ``(name, attr)`` for the lifetime of the process, so a
    missing dependency costs one failed import, not one per call site.
    """
    key = (name, attr)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    module: Optional[Any] = None
    version: Optional[str] = None
    error: Optional[str] = None
    try:
        imported = importlib.import_module(name)
    except BaseException as exc:  # noqa: BLE001 - broken installs raise anything
        error = "%s: %s" % (type(exc).__name__, exc)
    else:
        version = _detect_version(imported, name)
        if attr is None:
            module = imported
        else:
            try:
                module = getattr(imported, attr)
            except AttributeError as exc:
                module = None
                error = "%s: %s" % (type(exc).__name__, exc)

    result = OptionalModule(name=name, module=module, version=version, error=error)
    _CACHE[key] = result
    return result


def have(name: str) -> bool:
    """True when the optional module ``name`` can be imported."""
    return bool(optional_import(name))


def clear_cache() -> None:
    """Drop the memoized import results (used by tests and ``zfp doctor --refresh``)."""
    _CACHE.clear()


def capability_report() -> Dict[str, Dict[str, Any]]:
    """Return the capability matrix backing the ``zfp doctor`` command.

    Keys are module names then binary names; each value describes availability,
    version/path, the failure reason, and the pip extra that provides it.
    """
    report: Dict[str, Dict[str, Any]] = {}
    for name in OPTIONAL_MODULES:
        mod = optional_import(name)
        report[name] = {
            "kind": "module",
            "available": bool(mod),
            "version": mod.version,
            "error": mod.error,
            "extra": EXTRA_FOR_MODULE.get(name, "all"),
            "distribution": _DIST_FOR_MODULE.get(name, name),
        }
    for binary in OPTIONAL_BINARIES:
        path = shutil.which(binary)
        report[binary] = {
            "kind": "binary",
            "available": path is not None,
            "path": path,
            "error": None if path else "'%s' not found on PATH" % binary,
        }
    return report
