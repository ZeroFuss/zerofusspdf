"""OCR engines: one uniform interface over Tesseract, PaddleOCR, and nothing at all.

Every engine sees a :class:`~zfp.raster.render.RenderedPage` (top-left origin, y down,
8-bit gray) and answers with :class:`~zfp.core.types.RasterWord` boxes **in PDF user
space**.  Engines themselves only ever produce :class:`PixelWord` boxes; the single
conversion from pixels to user space lives in :meth:`BaseEngine.to_user_space`, which
calls :meth:`~zfp.core.geometry.PageGeometry.pixel_rect_to_user` with the raster's own
scale.  That is deliberate: pixel space must never leak past this package, and a
coordinate bug is only fixable in one place if there is only one place.

Robustness is a hard requirement rather than a nicety.  A missing binary, a subprocess
that times out, a third-party library that changed its return shape, one malformed row
in a TSV dump -- each of those yields *fewer words and a logged warning*, never an
exception escaping into the pipeline.  A page that fails to OCR is a page the cascade
escalates; it is not a run that dies.

Nothing here imports a third-party library at module import time.  ``pytesseract`` and
``paddleocr`` are resolved inside the functions that need them, through
:func:`~zfp.core.optional.optional_import`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

try:  # pragma: no cover - typing.runtime_checkable exists on every supported version
    from typing import runtime_checkable
except ImportError:  # pragma: no cover
    def runtime_checkable(cls):  # type: ignore[misc]
        """Fallback no-op when the runtime is too old to check protocols."""
        return cls

from ..core.config import OcrConfig
from ..core.errors import UnsupportedFeatureError
from ..core.geometry import PageGeometry, Rect
from ..core.logging import get_logger
from ..core.optional import optional_import
from ..core.types import RasterWord

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance, types only
    from ..raster.render import RenderedPage

__all__ = [
    "PixelWord",
    "OcrEngine",
    "BaseEngine",
    "TesseractEngine",
    "PaddleEngine",
    "NullEngine",
    "TSV_COLUMNS",
    "TSV_WORD_LEVEL",
    "TESSERACT_BINARY",
    "TESSERACT_TIMEOUT_S",
    "TESSERACT_PSM",
    "parse_tesseract_tsv",
    "parse_paddle_result",
    "resolve_ocr_config",
    "register_engine",
    "unregister_engine",
    "get_engine",
    "engine_names",
    "available_engines",
    "clear_engine_cache",
]

_log = get_logger(__name__)

#: Column order of a Tesseract TSV dump, used when the header row is absent.
TSV_COLUMNS: Tuple[str, ...] = (
    "level",
    "page_num",
    "block_num",
    "par_num",
    "line_num",
    "word_num",
    "left",
    "top",
    "width",
    "height",
    "conf",
    "text",
)
#: TSV hierarchy level of a single word.
TSV_WORD_LEVEL = 5
#: Name of the Tesseract executable looked up on ``PATH``.
TESSERACT_BINARY = "tesseract"
#: Wall-clock ceiling for one Tesseract subprocess, in seconds.
TESSERACT_TIMEOUT_S = 120.0
#: Page segmentation mode: 3 == fully automatic, no orientation detection.
TESSERACT_PSM = 3


def _clamp01(value: float) -> float:
    """Clamp ``value`` into ``[0, 1]``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _int_or(value: Any, fallback: int) -> int:
    """Best-effort int conversion that never raises."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def resolve_ocr_config(config: Any) -> OcrConfig:
    """Return an :class:`~zfp.core.config.OcrConfig` from whatever the caller passed.

    Accepts an :class:`OcrConfig`, a :class:`~zfp.core.config.ZfpConfig` (its ``.ocr``
    section is used), or ``None`` (defaults).  Callers deep in the pipeline get handed
    one or the other depending on the layer they sit in, and neither should have to care.
    """
    if config is None:
        return OcrConfig()
    inner = getattr(config, "ocr", None)
    if isinstance(inner, OcrConfig):
        return inner
    return config  # type: ignore[return-value]


# ======================================================================================
# The word an engine actually produces
# ======================================================================================


@dataclass
class PixelWord:
    """One recognized word in **pixel** space, before conversion to user space.

    This type never crosses a package boundary: :meth:`BaseEngine.to_user_space` turns it
    into a :class:`~zfp.core.types.RasterWord`.

    Attributes:
        text: The recognized characters.
        rect: Box in raster pixels, top-left origin, y down.
        confidence: 0..1 (engines report 0..100; the adapters divide).
        line_id: Engine line index, or ``-1``.
        block_id: Engine block index, or ``-1``.
        alternatives: ``(text, confidence)`` runners-up, when the engine offers them.
    """

    text: str
    rect: Rect
    confidence: float
    line_id: int = -1
    block_id: int = -1
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


# ======================================================================================
# Interface
# ======================================================================================


@runtime_checkable
class OcrEngine(Protocol):
    """Structural interface every OCR backend satisfies.

    Implementations are free to subclass :class:`BaseEngine` (which is what the bundled
    ones do, so that the pixel -> user-space conversion is shared) or to satisfy the
    protocol structurally, which is what test doubles do.
    """

    name: str

    def available(self) -> bool:
        """True when this engine can actually run on this machine right now."""
        ...

    def recognize(
        self, page: "RenderedPage", geometry: PageGeometry, config: OcrConfig
    ) -> List[RasterWord]:
        """Recognize ``page``, returning words in PDF user space."""
        ...


class BaseEngine(ABC):
    """Shared plumbing: availability guard, error containment, coordinate conversion.

    Subclasses implement :meth:`_recognize_pixels` and return :class:`PixelWord` boxes in
    raster coordinates.  :meth:`recognize` is deliberately ``final`` in spirit: it is the
    only path to a :class:`~zfp.core.types.RasterWord`, so every engine converts
    coordinates identically and no engine can leak pixels.
    """

    #: Registry key and the string that shows up in :class:`~zfp.ocr.cascade.OcrResult`.
    name: str = "base"

    def available(self) -> bool:
        """True when the backend can run; the base implementation says no."""
        return False

    def recognize(
        self, page: "RenderedPage", geometry: PageGeometry, config: OcrConfig
    ) -> List[RasterWord]:
        """Recognize ``page`` and return user-space words; never raises.

        Args:
            page: The rasterized page (or a crop of one, with a matching ``geometry``).
            geometry: Page geometry whose crop box corresponds to ``page``'s pixel (0,0).
            config: OCR settings (languages are the part engines care about).

        Returns:
            Words in PDF user space, in the engine's own reading order.  An unavailable
            engine, an empty raster, or any backend failure yields ``[]``.
        """
        cfg = resolve_ocr_config(config)
        try:
            usable = bool(self.available())
        except Exception as exc:  # noqa: BLE001 - a broken backend must not kill a run
            _log.warning("ocr engine %r availability check failed: %s", self.name, exc)
            return []
        if not usable:
            _log.warning("ocr engine %r is unavailable; recognized nothing", self.name)
            return []
        width = int(getattr(page, "width", 0) or 0)
        height = int(getattr(page, "height", 0) or 0)
        if width <= 0 or height <= 0:
            _log.warning("ocr engine %r got an empty raster; recognized nothing", self.name)
            return []
        try:
            raw = self._recognize_pixels(page, cfg)
        except Exception as exc:  # noqa: BLE001 - subprocess/vendor failures are data
            _log.warning(
                "ocr engine %r failed on page %s: %s: %s",
                self.name,
                getattr(page, "page", "?"),
                type(exc).__name__,
                exc,
            )
            return []
        return self.to_user_space(raw or [], page, geometry)

    def to_user_space(
        self,
        words: Sequence[PixelWord],
        page: "RenderedPage",
        geometry: PageGeometry,
    ) -> List[RasterWord]:
        """Convert pixel boxes to user space -- **the only place this conversion happens**.

        The raster's own :attr:`~zfp.raster.render.RenderedPage.scale` drives the
        transform, so a crop re-recognized at the page scale converts correctly as long
        as its ``geometry`` was built by :func:`~zfp.ocr.cascade.crop_geometry`.  Boxes
        are clamped to the crop box: an engine may report a box a pixel off the raster,
        but a widget must never be placed off the page.
        """
        scale = float(getattr(page, "scale", 1.0) or 1.0)
        page_index = int(getattr(page, "page", 0) or 0)
        out: List[RasterWord] = []
        for word in words:
            text = (word.text or "").strip()
            if not text:
                continue
            try:
                user = geometry.clamp(geometry.pixel_rect_to_user(word.rect, scale))
            except Exception as exc:  # noqa: BLE001 - a bad box is one dropped word
                _log.warning("ocr engine %r produced an unusable box: %s", self.name, exc)
                continue
            out.append(
                RasterWord(
                    text=text,
                    rect=user,
                    confidence=_clamp01(word.confidence),
                    page=page_index,
                    line_id=int(word.line_id),
                    block_id=int(word.block_id),
                    alternatives=[(str(t), _clamp01(c)) for t, c in word.alternatives],
                )
            )
        return out

    @abstractmethod
    def _recognize_pixels(self, page: "RenderedPage", config: OcrConfig) -> List[PixelWord]:
        """Run the backend and return words in pixel space."""

    def __repr__(self) -> str:
        return "%s(name=%r)" % (type(self).__name__, self.name)


# ======================================================================================
# Tesseract
# ======================================================================================


def parse_tesseract_tsv(text: str) -> List[PixelWord]:
    """Parse a Tesseract TSV dump into word-level :class:`PixelWord` boxes.

    The dump has one row per hierarchy node; only level-5 rows are words.  Rows whose
    ``conf`` is ``-1`` carry no text (they are structural nodes) and are dropped, as are
    rows with empty text, non-positive size, or unparseable numbers.  A header row, when
    present, defines the column order; otherwise :data:`TSV_COLUMNS` is assumed.

    Args:
        text: The raw TSV, as printed by ``tesseract ... tsv`` or ``image_to_data``.

    Returns:
        Words in file order.  A malformed row costs that row, not the dump.
    """
    words: List[PixelWord] = []
    if not text:
        return words
    columns: List[str] = list(TSV_COLUMNS)
    for lineno, raw in enumerate(text.splitlines()):
        if not raw.strip():
            continue
        cells = raw.split("\t")
        if cells[0].strip().lower() == "level":
            columns = [c.strip().lower() for c in cells]
            continue
        row = _tsv_row(columns, cells)
        if row is None:
            _log.debug("tesseract tsv: dropping short row %d", lineno)
            continue
        if _int_or(row.get("level"), -1) != TSV_WORD_LEVEL:
            continue
        word = _tsv_word(row)
        if word is None:
            continue
        words.append(word)
    return words


def _tsv_row(columns: Sequence[str], cells: Sequence[str]) -> Optional[Dict[str, str]]:
    """Zip a TSV row against its columns, tolerating tabs inside the text column."""
    if len(cells) < len(columns) - 1:
        return None
    row: Dict[str, str] = {}
    last = len(columns) - 1
    for i, name in enumerate(columns):
        if i < last:
            row[name] = cells[i] if i < len(cells) else ""
        else:
            row[name] = "\t".join(cells[last:]) if len(cells) > last else ""
    return row


def _tsv_word(row: Dict[str, str]) -> Optional[PixelWord]:
    """Turn one level-5 TSV row into a :class:`PixelWord`, or ``None`` if unusable."""
    text = (row.get("text") or "").strip()
    if not text:
        return None
    try:
        left = float(row["left"])
        top = float(row["top"])
        width = float(row["width"])
        height = float(row["height"])
        conf = float(row["conf"])
    except (KeyError, TypeError, ValueError):
        _log.debug("tesseract tsv: dropping row with unparseable geometry: %r", row)
        return None
    if conf < 0.0:  # -1 == "this node has no recognized text"
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return PixelWord(
        text=text,
        rect=Rect(left, top, left + width, top + height),
        confidence=_clamp01(conf / 100.0),
        line_id=_int_or(row.get("line_num"), -1),
        block_id=_int_or(row.get("block_num"), -1),
    )


class TesseractEngine(BaseEngine):
    """Tesseract, through the ``pytesseract`` module when present, else the binary.

    Both paths speak TSV, so both land in :func:`parse_tesseract_tsv`.  The binary path
    is fed the page as PGM bytes on **stdin** and reads TSV from **stdout**, which keeps
    the whole exchange in memory except for the temporary file ``pytesseract`` insists on
    (it is handed a path, not a PIL image, so Pillow is not required).
    """

    name = "tesseract"

    def __init__(
        self,
        binary: str = TESSERACT_BINARY,
        timeout_s: float = TESSERACT_TIMEOUT_S,
        psm: int = TESSERACT_PSM,
    ) -> None:
        self._binary = binary
        self._timeout_s = float(timeout_s)
        self._psm = int(psm)

    # -- discovery ---------------------------------------------------------------------
    def binary_path(self) -> Optional[str]:
        """Absolute path of the ``tesseract`` executable, or ``None``."""
        return shutil.which(self._binary)

    def available(self) -> bool:
        """True when either the Python binding or the executable can be reached."""
        if optional_import("pytesseract"):
            return True
        return self.binary_path() is not None

    # -- recognition -------------------------------------------------------------------
    def _recognize_pixels(self, page: "RenderedPage", config: OcrConfig) -> List[PixelWord]:
        """Produce TSV from whichever Tesseract front end answers first, then parse it."""
        tsv = self._via_pytesseract(page, config)
        if tsv is None:
            tsv = self._via_binary(page, config)
        if not tsv:
            return []
        return parse_tesseract_tsv(tsv)

    def _language(self, config: OcrConfig) -> str:
        """Tesseract's ``-l`` argument, e.g. ``eng+deu``."""
        langs = [str(code).strip() for code in (config.languages or []) if str(code).strip()]
        return "+".join(langs) if langs else "eng"

    def _via_pytesseract(self, page: "RenderedPage", config: OcrConfig) -> Optional[str]:
        """Ask ``pytesseract.image_to_data`` for TSV, or ``None`` to fall through."""
        module = optional_import("pytesseract")
        if not module:
            return None
        image_to_data = getattr(module.module, "image_to_data", None)
        if image_to_data is None:
            _log.warning("pytesseract has no image_to_data(); falling back to the binary")
            return None
        try:
            with tempfile.TemporaryDirectory(prefix="zfp-ocr-") as tmp:
                path = os.path.join(tmp, "page.pgm")
                with open(path, "wb") as handle:
                    handle.write(page.to_pgm())
                out = image_to_data(
                    path,
                    lang=self._language(config),
                    config="--psm %d" % self._psm,
                )
        except Exception as exc:  # noqa: BLE001 - fall through to the binary
            _log.warning("pytesseract failed (%s: %s); trying the binary", type(exc).__name__, exc)
            return None
        if isinstance(out, bytes):
            return out.decode("utf-8", "replace")
        if isinstance(out, str):
            return out
        _log.warning("pytesseract returned %s, not TSV text", type(out).__name__)
        return None

    def _via_binary(self, page: "RenderedPage", config: OcrConfig) -> Optional[str]:
        """Run ``tesseract stdin stdout`` over the page's PGM bytes."""
        path = self.binary_path()
        if path is None:
            return None
        command = [
            path,
            "stdin",
            "stdout",
            "--psm",
            str(self._psm),
            "-l",
            self._language(config),
            "-c",
            "tessedit_create_tsv=1",
        ]
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
                command,
                input=page.to_pgm(),
                capture_output=True,
                timeout=self._timeout_s,
            )
        except subprocess.TimeoutExpired:
            _log.warning("tesseract timed out after %.0fs on page %s", self._timeout_s, getattr(page, "page", "?"))
            return None
        except (OSError, ValueError) as exc:
            _log.warning("tesseract could not be executed: %s: %s", type(exc).__name__, exc)
            return None
        if proc.returncode != 0:
            _log.warning(
                "tesseract exited %d: %s",
                proc.returncode,
                (proc.stderr or b"").decode("utf-8", "replace").strip()[:200],
            )
            return None
        return (proc.stdout or b"").decode("utf-8", "replace")


# ======================================================================================
# PaddleOCR
# ======================================================================================


def _points_to_rect(box: Any) -> Optional[Rect]:
    """Axis-aligned bounding box of a quad (or any point list); ``None`` if unusable."""
    xs: List[float] = []
    ys: List[float] = []
    try:
        for point in box:
            values = [float(v) for v in point]
            if len(values) < 2:
                return None
            xs.append(values[0])
            ys.append(values[1])
    except (TypeError, ValueError):
        return None
    if len(xs) < 2:
        return None
    return Rect(min(xs), min(ys), max(xs), max(ys))


def _is_paddle_line(item: Any) -> bool:
    """True when ``item`` looks like Paddle's ``[box, (text, score)]`` pair."""
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return False
    box, payload = item
    if not isinstance(box, (list, tuple)) or len(box) < 3:
        return False
    if not isinstance(box[0], (list, tuple)) or len(box[0]) < 2:
        return False
    if isinstance(payload, str):
        return True
    return isinstance(payload, (list, tuple)) and len(payload) >= 1


def _paddle_entries(result: Any) -> List[Any]:
    """Flatten Paddle's several historical return shapes into a list of entries."""
    if result is None:
        return []
    if isinstance(result, dict):
        return [result]
    if not isinstance(result, (list, tuple)):
        return []
    entries: List[Any] = []
    for item in result:
        if item is None:
            continue
        if isinstance(item, dict) or _is_paddle_line(item):
            entries.append(item)
            continue
        if isinstance(item, (list, tuple)):
            entries.extend(sub for sub in item if sub is not None)
    return entries


def parse_paddle_result(result: Any) -> List[PixelWord]:
    """Convert a PaddleOCR result into :class:`PixelWord` boxes.

    Paddle has shipped at least three result shapes: ``[[[box, (text, score)], ...]]``
    (one list per image), the same list unwrapped, and 3.x's ``[{"dt_polys": ...,
    "rec_texts": ..., "rec_scores": ...}]`` prediction dicts.  All three are accepted;
    anything unrecognized is skipped with a debug line rather than raising.

    Quad boxes become axis-aligned rectangles, which is what a form-field rectangle is.
    """
    words: List[PixelWord] = []
    for entry in _paddle_entries(result):
        if isinstance(entry, dict):
            words.extend(_paddle_dict_words(entry))
            continue
        if not _is_paddle_line(entry):
            _log.debug("paddleocr: skipping unrecognized entry %r", type(entry).__name__)
            continue
        box, payload = entry
        rect = _points_to_rect(box)
        if rect is None:
            continue
        if isinstance(payload, str):
            text, score = payload, 1.0
        else:
            text = str(payload[0])
            score = float(payload[1]) if len(payload) > 1 else 1.0
        if not text.strip():
            continue
        words.append(PixelWord(text=text, rect=rect, confidence=_clamp01(score)))
    return words


def _paddle_dict_words(entry: Dict[str, Any]) -> List[PixelWord]:
    """Read PaddleOCR 3.x prediction dicts (``rec_texts`` / ``rec_scores`` / polygons)."""
    texts = entry.get("rec_texts") or entry.get("texts") or []
    scores = entry.get("rec_scores") or entry.get("scores") or []
    polys = entry.get("dt_polys") or entry.get("rec_polys") or entry.get("boxes") or []
    words: List[PixelWord] = []
    for i, text in enumerate(texts):
        label = str(text)
        if not label.strip():
            continue
        try:
            score = float(scores[i])
        except (IndexError, TypeError, ValueError):
            score = 1.0
        rect = _points_to_rect(polys[i]) if i < len(polys) else None
        if rect is None:
            continue
        words.append(PixelWord(text=label, rect=rect, confidence=_clamp01(score)))
    return words


class PaddleEngine(BaseEngine):
    """PaddleOCR, imported lazily and built once per instance.

    Constructing a ``PaddleOCR`` object loads models and is expensive, so it is created
    on first use and cached on the instance, keyed by language.  The page is handed over
    as a numpy array when numpy is present and as a temporary PGM path otherwise, so the
    engine still works when only Paddle's own dependencies are installed.
    """

    name = "paddle"

    def __init__(self) -> None:
        self._ocr: Optional[Any] = None
        self._ocr_language: Optional[str] = None

    def available(self) -> bool:
        """True when ``paddleocr`` imports."""
        return bool(optional_import("paddleocr"))

    def _language(self, config: OcrConfig) -> str:
        """Paddle takes one language code; the first configured one wins."""
        langs = [str(code).strip() for code in (config.languages or []) if str(code).strip()]
        first = langs[0] if langs else "eng"
        return {"eng": "en", "deu": "german", "fra": "fr", "spa": "es"}.get(first, first)

    def _ocr_object(self, config: OcrConfig) -> Optional[Any]:
        """Build (once) and return the cached ``PaddleOCR`` instance, or ``None``."""
        language = self._language(config)
        if self._ocr is not None and self._ocr_language == language:
            return self._ocr
        module = optional_import("paddleocr")
        if not module:
            return None
        factory = getattr(module.module, "PaddleOCR", None)
        if factory is None:
            _log.warning("paddleocr has no PaddleOCR class; engine disabled")
            return None
        for kwargs in (
            {"use_angle_cls": True, "lang": language, "show_log": False},
            {"use_angle_cls": True, "lang": language},
            {"lang": language},
            {},
        ):
            try:
                self._ocr = factory(**kwargs)
            except Exception as exc:  # noqa: BLE001 - vendor kwargs drift between releases
                _log.debug("PaddleOCR(**%r) failed: %s", sorted(kwargs), exc)
                continue
            self._ocr_language = language
            return self._ocr
        _log.warning("PaddleOCR could not be constructed; engine disabled")
        return None

    def _numpy_image(self, page: "RenderedPage") -> Optional[Any]:
        """Return the page as an ``HxWx3`` uint8 array, or ``None`` when numpy is absent."""
        np = optional_import("numpy").module
        if np is None:
            return None
        try:
            flat = np.frombuffer(page.gray, dtype=np.uint8)
            if int(flat.size) != int(page.width) * int(page.height):
                return None
            plane = flat.reshape((int(page.height), int(page.width)))
            return np.stack([plane, plane, plane], axis=-1)
        except Exception as exc:  # noqa: BLE001 - fall back to the file path
            _log.debug("numpy view of the raster failed: %s", exc)
            return None

    def _invoke(self, ocr: Any, image: Any) -> Any:
        """Call whichever recognition entry point this Paddle release exposes."""
        run = getattr(ocr, "ocr", None)
        if run is not None:
            try:
                return run(image, cls=True)
            except TypeError:
                return run(image)
        predict = getattr(ocr, "predict", None)
        if predict is not None:
            return predict(image)
        raise UnsupportedFeatureError("the PaddleOCR object exposes neither ocr() nor predict()")

    def _recognize_pixels(self, page: "RenderedPage", config: OcrConfig) -> List[PixelWord]:
        """Run Paddle over the page and normalize whatever it hands back."""
        ocr = self._ocr_object(config)
        if ocr is None:
            return []
        image = self._numpy_image(page)
        if image is not None:
            return parse_paddle_result(self._invoke(ocr, image))
        with tempfile.TemporaryDirectory(prefix="zfp-ocr-") as tmp:
            path = os.path.join(tmp, "page.pgm")
            with open(path, "wb") as handle:
                handle.write(page.to_pgm())
            return parse_paddle_result(self._invoke(ocr, path))


# ======================================================================================
# Null
# ======================================================================================


class NullEngine(BaseEngine):
    """Always available, recognizes nothing.

    It is what keeps the cascade honest on a machine with no OCR installed: the pipeline
    still runs end to end, the report says plainly that no words were found, and no
    caller has to special-case ``None``.
    """

    name = "null"

    def available(self) -> bool:
        """Always true."""
        return True

    def _recognize_pixels(self, page: "RenderedPage", config: OcrConfig) -> List[PixelWord]:
        """Recognize nothing, successfully."""
        return []


# ======================================================================================
# Registry
# ======================================================================================

EngineFactory = Callable[[], Any]

_REGISTRY: Dict[str, EngineFactory] = {}
_INSTANCES: Dict[str, Any] = {}


def register_engine(name: str, factory: EngineFactory) -> None:
    """Register (or replace) an engine factory under ``name``.

    Args:
        name: Registry key; matched case-insensitively against ``OcrConfig.engines``.
        factory: Zero-argument callable returning an :class:`OcrEngine`.

    Raises:
        ValidationError: ``name`` is empty or ``factory`` is not callable.
    """
    from ..core.errors import ValidationError

    key = (name or "").strip().lower()
    if not key:
        raise ValidationError("an OCR engine name cannot be empty")
    if not callable(factory):
        raise ValidationError("an OCR engine factory must be callable")
    _REGISTRY[key] = factory
    _INSTANCES.pop(key, None)


def unregister_engine(name: str) -> bool:
    """Remove ``name`` from the registry; returns whether it was there."""
    key = (name or "").strip().lower()
    _INSTANCES.pop(key, None)
    return _REGISTRY.pop(key, None) is not None


def clear_engine_cache() -> None:
    """Drop memoized engine instances (used by tests and long-lived services)."""
    _INSTANCES.clear()


def get_engine(name: str) -> OcrEngine:
    """Return the engine registered under ``name``.

    Instances are memoized, because an engine may cache an expensive model object on
    itself (see :class:`PaddleEngine`).

    Raises:
        UnsupportedFeatureError: No engine is registered under that name, or its factory
            raised.
    """
    key = (name or "").strip().lower()
    cached = _INSTANCES.get(key)
    if cached is not None:
        return cached
    factory = _REGISTRY.get(key)
    if factory is None:
        raise UnsupportedFeatureError(
            "unknown OCR engine %r (registered: %s)" % (name, ", ".join(engine_names()) or "none")
        )
    try:
        engine = factory()
    except Exception as exc:  # noqa: BLE001 - a broken factory is an unsupported feature
        raise UnsupportedFeatureError(
            "OCR engine %r could not be constructed: %s: %s" % (name, type(exc).__name__, exc)
        ) from exc
    _INSTANCES[key] = engine
    return engine


def engine_names() -> List[str]:
    """Every registered engine name, in registration order."""
    return list(_REGISTRY)


def available_engines() -> List[str]:
    """Registered engines that can actually run here, in registration order.

    On a machine with no OCR installed this is ``["null"]`` -- never an empty list, so
    the cascade always has something to call.
    """
    out: List[str] = []
    for key in _REGISTRY:
        try:
            engine = get_engine(key)
            if engine.available():
                out.append(key)
        except Exception as exc:  # noqa: BLE001 - discovery never raises
            _log.warning("ocr engine %r could not be probed: %s", key, exc)
    return out


register_engine(TesseractEngine.name, TesseractEngine)
register_engine(PaddleEngine.name, PaddleEngine)
register_engine(NullEngine.name, NullEngine)
