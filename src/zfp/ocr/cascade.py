"""The OCR cascade: native text first, then the cheapest engine that is good enough.

The governing rule, straight out of the research, is **never OCR text the PDF already
contains perfectly**.  A born-digital page hands over exact glyph boxes and exact
characters; running a recognizer over a rendering of that page can only lose information,
and it costs a second per page to do it.  So the first branch of :func:`ocr_cascade` is
also its most important one: given non-empty ``native_spans``, it returns immediately
with ``engine="skipped_native_text"`` and no words.

Everything after that branch is an escalation ladder, cheapest rung first:

1. **native text** -- free, exact, already in user space.
2. **a local engine** (Tesseract) -- fast, offline, good on clean scans.
3. **a modern engine** (PaddleOCR) -- slower, better on noisy or dense pages.
4. **cropped re-recognition** -- re-run the best engine on just the regions it was
   unsure about, where the recognizer sees the word without the rest of the page.

A rung is only climbed when the one below it came back under
:attr:`~zfp.core.config.OcrConfig.escalate_below`; the first engine whose mean confidence
clears that bar wins, and the rest are never called.  Words that still sit below
:attr:`~zfp.core.config.OcrConfig.min_word_confidence` are not thrown away and not
trusted either: they go to :attr:`OcrResult.suspects`, which is what Acrobat calls
"suspect words" and what the QA dashboard shows a human.

Every decision is recorded in :attr:`OcrResult.report` as a plain-language line, because
"why did this page come out badly" is the question this module will be asked most.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from statistics import median
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.errors import ZfpError
from ..core.geometry import PageGeometry, Rect
from ..core.logging import get_logger
from ..core.types import RasterWord, TextSpan
from .engine import OcrEngine, get_engine, resolve_ocr_config

if TYPE_CHECKING:  # pragma: no cover - types only, never imported at runtime
    from ..raster.render import RenderedPage

__all__ = [
    "OcrResult",
    "SKIPPED_NATIVE_TEXT",
    "ENGINE_NONE",
    "ENGINE_DISABLED",
    "MIN_NATIVE_TEXT_CHARS",
    "CROP_PAD_PT",
    "MAX_ESCALATION_CROPS",
    "MERGE_MIN_IOU",
    "LINE_OVERLAP_RATIO",
    "has_native_text",
    "mean_confidence",
    "ocr_cascade",
    "recognize_regions",
    "crop_geometry",
    "merge_words",
    "group_words_into_lines",
    "words_to_spans",
    "words_to_word_spans",
]

_log = get_logger(__name__)

#: Reported as the engine when the page already had usable native text.
SKIPPED_NATIVE_TEXT = "skipped_native_text"
#: Reported when no engine ran at all.
ENGINE_NONE = "none"
#: Reported when OCR is switched off in the configuration.
ENGINE_DISABLED = "disabled"

#: How many visible characters of native text are enough to skip OCR entirely.
MIN_NATIVE_TEXT_CHARS = 1
#: Padding, in points, added around a word before it is re-cropped.
CROP_PAD_PT = 4.0
#: Ceiling on the number of crops one escalation pass will run.
MAX_ESCALATION_CROPS = 64
#: Minimum IoU for a re-recognized word to be considered the same word as an original.
MERGE_MIN_IOU = 0.30
#: Fraction of a word's height that must overlap a line's band to join that line.
LINE_OVERLAP_RATIO = 0.4


# ======================================================================================
# Result
# ======================================================================================


@dataclass
class OcrResult:
    """Everything one page's OCR produced, including how it got there.

    Attributes:
        words: Recognized words at or above ``min_word_confidence``, in user space.
        engine: Name of the engine whose output was kept, or one of
            :data:`SKIPPED_NATIVE_TEXT`, :data:`ENGINE_DISABLED`, :data:`ENGINE_NONE`.
        mean_confidence: Word-count-weighted mean confidence over **every** recognized
            word, suspects included -- it describes the page, not the surviving subset.
            It is ``1.0`` for :data:`SKIPPED_NATIVE_TEXT`, because the page's text came
            from the content stream and is exact.
        escalated: True when the cascade had to re-recognize crops.
        suspects: Recognized words below ``min_word_confidence``.
        per_engine: Word count produced by each engine that ran, keyed by name; a
            cropped re-recognition pass is recorded as ``"<engine>+crops"``.
        report: The decision trail, one line per decision, in order.
    """

    words: List[RasterWord] = field(default_factory=list)
    engine: str = ENGINE_NONE
    mean_confidence: float = 0.0
    escalated: bool = False
    suspects: List[RasterWord] = field(default_factory=list)
    per_engine: Dict[str, int] = field(default_factory=dict)
    report: List[str] = field(default_factory=list)

    @property
    def all_words(self) -> List[RasterWord]:
        """Kept words plus suspects, i.e. everything the engine actually read."""
        return list(self.words) + list(self.suspects)

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "words": [w.as_dict() for w in self.words],
            "engine": self.engine,
            "mean_confidence": self.mean_confidence,
            "escalated": self.escalated,
            "suspects": [w.as_dict() for w in self.suspects],
            "per_engine": dict(self.per_engine),
            "report": list(self.report),
        }


# ======================================================================================
# Predicates and statistics
# ======================================================================================


def has_native_text(
    spans: Optional[Sequence[TextSpan]], min_chars: int = MIN_NATIVE_TEXT_CHARS
) -> bool:
    """True when ``spans`` carry real visible text, so OCR must not run.

    Whitespace-only spans do not count: a page whose content stream draws three spaces
    and a raster is a scan, not a text page.

    Args:
        spans: Native spans from :mod:`zfp.native.content`, or ``None``.
        min_chars: How many non-whitespace characters are enough.
    """
    if not spans:
        return False
    seen = 0
    for span in spans:
        text = getattr(span, "text", "") or ""
        seen += sum(1 for ch in text if not ch.isspace())
        if seen >= max(1, int(min_chars)):
            return True
    return False


def mean_confidence(words: Sequence[RasterWord]) -> float:
    """Word-count-weighted mean confidence of ``words`` (0.0 when there are none).

    Weighting by word count is what makes per-engine means comparable: an engine that
    reads three words at 0.9 has not done better than one that read ninety at 0.85.
    """
    items = list(words)
    if not items:
        return 0.0
    return sum(float(w.confidence) for w in items) / float(len(items))


# ======================================================================================
# The cascade
# ======================================================================================


def ocr_cascade(
    page: "RenderedPage",
    geometry: PageGeometry,
    config: Any = None,
    *,
    native_spans: Optional[Sequence[TextSpan]] = None,
) -> OcrResult:
    """Recognize a page, escalating only as far as the confidence forces.

    Args:
        page: The rasterized page.
        geometry: Geometry of the same page; every returned rect is converted through it.
        config: An :class:`~zfp.core.config.OcrConfig`, a
            :class:`~zfp.core.config.ZfpConfig`, or ``None`` for defaults.
        native_spans: Text the PDF already contains for this page.  If this carries any
            visible characters, **no engine is called at all**.

    Returns:
        An :class:`OcrResult`.  Failure is expressed in the result (empty words, a
        report line), never as an exception.
    """
    cfg = resolve_ocr_config(config)
    page_index = int(getattr(page, "page", getattr(geometry, "index", 0)) or 0)
    report: List[str] = []

    # 1. The rule that matters most: never OCR text the PDF already contains.
    if has_native_text(native_spans):
        visible = sum(1 for s in (native_spans or []) if (getattr(s, "text", "") or "").strip())
        report.append(
            "page %d: %d visible native text span(s) present; OCR skipped entirely "
            "(never OCR text the PDF already contains)" % (page_index, visible)
        )
        _log.debug("page %d has native text; OCR skipped", page_index)
        return OcrResult(
            words=[],
            engine=SKIPPED_NATIVE_TEXT,
            mean_confidence=1.0,
            escalated=False,
            suspects=[],
            per_engine={},
            report=report,
        )

    if not cfg.enabled:
        report.append("page %d: OCR is disabled by configuration" % page_index)
        return OcrResult(engine=ENGINE_DISABLED, report=report)

    names = [str(n).strip().lower() for n in (cfg.engines or []) if str(n).strip()]
    if not names:
        report.append("page %d: no OCR engines are configured" % page_index)
        return OcrResult(engine=ENGINE_NONE, report=report)

    # 2. Run engines in order; stop at the first one that clears the bar.
    per_engine: Dict[str, int] = {}
    best_name: Optional[str] = None
    best_words: List[RasterWord] = []
    best_mean = -1.0
    accepted = False
    last_ran: Optional[str] = None

    for name in names:
        try:
            engine = get_engine(name)
        except ZfpError as exc:
            report.append("engine %r is not registered (%s)" % (name, exc))
            continue
        try:
            usable = bool(engine.available())
        except Exception as exc:  # noqa: BLE001 - discovery never kills a run
            _log.warning("engine %r availability check failed: %s", name, exc)
            usable = False
        if not usable:
            report.append("engine %r is unavailable here; skipped" % name)
            continue

        try:
            words = list(engine.recognize(page, geometry, cfg))
        except Exception as exc:  # noqa: BLE001 - third-party engines may still throw
            _log.warning("engine %r raised %s: %s", name, type(exc).__name__, exc)
            report.append("engine %r raised %s; treated as no words" % (name, type(exc).__name__))
            words = []
        last_ran = name
        per_engine[name] = len(words)

        if not words:
            report.append("engine %r returned no words" % name)
            continue

        current = mean_confidence(words)
        if current > best_mean or (current == best_mean and len(words) > len(best_words)):
            best_name, best_words, best_mean = name, words, current

        if current >= cfg.escalate_below:
            report.append(
                "engine %r accepted: %d word(s), mean confidence %.3f >= %.3f"
                % (name, len(words), current, cfg.escalate_below)
            )
            accepted = True
            break
        report.append(
            "engine %r under the bar: %d word(s), mean confidence %.3f < %.3f; escalating"
            % (name, len(words), current, cfg.escalate_below)
        )

    if best_name is None:
        engine_name = last_ran or ENGINE_NONE
        report.append("page %d: no engine produced any words" % page_index)
        return OcrResult(
            words=[],
            engine=engine_name,
            mean_confidence=0.0,
            escalated=False,
            suspects=[],
            per_engine=per_engine,
            report=report,
        )

    words = best_words
    escalated = False

    # 3. Still under the bar: re-recognize the low-confidence regions on their own.
    if not accepted:
        escalated = True
        low = [w.rect for w in words if w.confidence < cfg.escalate_below]
        report.append(
            "best engine %r at mean %.3f < %.3f; escalating to cropped re-recognition of "
            "%d low-confidence word(s)" % (best_name, best_mean, cfg.escalate_below, len(low))
        )
        report.append(
            "crops are re-recognized at the page's own scale %.4f: a rendered page cannot "
            "be re-rasterized at a higher dpi from here, so the gain comes from context "
            "isolation rather than resolution"
            % float(getattr(page, "scale", 1.0) or 1.0)
        )
        if low:
            try:
                engine = get_engine(best_name)
                improved = recognize_regions(engine, page, geometry, low, cfg)
            except ZfpError as exc:
                improved = []
                report.append("cropped re-recognition unavailable: %s" % exc)
            if improved:
                per_engine["%s+crops" % best_name] = len(improved)
                words, changed = _merge(words, improved)
                report.append(
                    "cropped re-recognition returned %d word(s) and improved %d"
                    % (len(improved), changed)
                )
            else:
                report.append("cropped re-recognition returned nothing usable")
        else:
            report.append("no individual word was under the bar; nothing to re-crop")

    # 4. Split off the suspects rather than dropping them.
    kept = [w for w in words if w.confidence >= cfg.min_word_confidence]
    suspects = [w for w in words if w.confidence < cfg.min_word_confidence]
    overall = mean_confidence(words)
    report.append(
        "page %d final: %d word(s) kept, %d suspect(s) below %.3f, mean confidence %.3f"
        % (page_index, len(kept), len(suspects), cfg.min_word_confidence, overall)
    )

    return OcrResult(
        words=kept,
        engine=best_name,
        mean_confidence=overall,
        escalated=escalated,
        suspects=suspects,
        per_engine=per_engine,
        report=report,
    )


# ======================================================================================
# Cropped re-recognition
# ======================================================================================


def crop_geometry(geometry: PageGeometry, rect_px: Rect, scale: float) -> PageGeometry:
    """Geometry for a crop, such that the crop's pixel (0,0) lands where it really is.

    A crop taken at pixel ``rect_px`` is its own little raster whose origin is not the
    page origin, so feeding it the page geometry would place every recognized word at the
    top-left of the page.  The fix is exactly one line of algebra: the sub-geometry is the
    page geometry with its **crop box replaced by the user-space rectangle the pixel crop
    covers**.  That identity holds for all four ``/Rotate`` values, because
    ``pixel_rect_to_user`` already applies the rotation.

    Args:
        geometry: Geometry of the full page.
        rect_px: The crop, in whole pixels of the full-page raster.
        scale: The raster's pixels-per-point.
    """
    return PageGeometry(
        index=geometry.index,
        media_box=geometry.media_box,
        crop_box=geometry.pixel_rect_to_user(rect_px, scale),
        rotation=geometry.rotation,
    )


def _coalesce_rects(rects: Sequence[Rect], pad: float) -> List[Rect]:
    """Union rectangles that touch once padded, so overlapping crops are cropped once."""
    groups: List[Rect] = []
    for rect in rects:
        grown = rect.normalized().inflated(pad)
        merged = False
        for i, existing in enumerate(groups):
            if existing.intersects(grown):
                groups[i] = existing.union(grown)
                merged = True
                break
        if not merged:
            groups.append(grown)
    # One more settling pass: a late rectangle can bridge two earlier groups.
    settled: List[Rect] = []
    for rect in groups:
        merged = False
        for i, existing in enumerate(settled):
            if existing.intersects(rect):
                settled[i] = existing.union(rect)
                merged = True
                break
        if not merged:
            settled.append(rect)
    return settled


def recognize_regions(
    engine: OcrEngine,
    page: "RenderedPage",
    geometry: PageGeometry,
    rects: Sequence[Rect],
    config: Any = None,
    *,
    pad_pt: float = CROP_PAD_PT,
    max_regions: int = MAX_ESCALATION_CROPS,
) -> List[RasterWord]:
    """Re-run ``engine`` on crops around ``rects`` and return the words, in user space.

    Args:
        engine: The engine to re-run -- normally the one that did best on the full page.
        page: The full-page raster.
        geometry: Geometry of the full page.
        rects: Regions of interest in **PDF user space** (this is a module boundary).
        config: OCR configuration.
        pad_pt: Padding around each region, in points; recognizers need some margin.
        max_regions: Ceiling on the number of crops, so a catastrophic page cannot turn
            into thousands of subprocess launches.

    Returns:
        Words in user space.  Regions that fall off the page, or that the engine chokes
        on, contribute nothing.
    """
    cfg = resolve_ocr_config(config)
    scale = float(getattr(page, "scale", 1.0) or 1.0)
    if scale <= 0.0:
        scale = 1.0
    width = int(getattr(page, "width", 0) or 0)
    height = int(getattr(page, "height", 0) or 0)
    out: List[RasterWord] = []
    if width <= 0 or height <= 0:
        return out

    regions = _coalesce_rects([r for r in rects if r is not None], float(pad_pt))
    for rect in regions[: max(0, int(max_regions))]:
        clamped = geometry.clamp(rect)
        if clamped.width <= 0.0 or clamped.height <= 0.0:
            continue
        px = geometry.user_rect_to_pixel(clamped, scale)
        x0 = max(0, int(math.floor(px.x0)))
        y0 = max(0, int(math.floor(px.y0)))
        x1 = min(width, int(math.ceil(px.x1)))
        y1 = min(height, int(math.ceil(px.y1)))
        if x1 <= x0 or y1 <= y0:
            continue
        box = Rect(float(x0), float(y0), float(x1), float(y1))
        try:
            crop = page.crop(box)
            sub = crop_geometry(geometry, box, scale)
            out.extend(engine.recognize(crop, sub, cfg))
        except Exception as exc:  # noqa: BLE001 - one bad crop is not a failed page
            _log.warning(
                "cropped re-recognition of %s failed: %s: %s", box.as_list(), type(exc).__name__, exc
            )
            continue
    return out


def _best_match(words: Sequence[RasterWord], candidate: RasterWord) -> Optional[int]:
    """Index of the word ``candidate`` most plausibly *is*, or ``None``."""
    best_index: Optional[int] = None
    best_iou = MERGE_MIN_IOU
    for i, word in enumerate(words):
        if word.page != candidate.page:
            continue
        score = word.rect.iou(candidate.rect)
        if score > best_iou:
            best_iou = score
            best_index = i
    return best_index


def _dedup_alternatives(
    alternatives: Iterable[Tuple[str, float]], exclude: str
) -> List[Tuple[str, float]]:
    """Keep the first occurrence of each alternative text, dropping ``exclude``."""
    seen = {exclude}
    out: List[Tuple[str, float]] = []
    for text, score in alternatives:
        if text in seen:
            continue
        seen.add(text)
        out.append((text, float(score)))
    return out


def _merge(
    base: Sequence[RasterWord], improved: Sequence[RasterWord]
) -> Tuple[List[RasterWord], int]:
    """Merge re-recognized words into the originals; returns the list and a change count."""
    merged: List[RasterWord] = list(base)
    changed = 0
    for candidate in improved:
        index = _best_match(merged, candidate)
        if index is None:
            merged.append(candidate)
            changed += 1
            continue
        current = merged[index]
        if candidate.confidence > current.confidence:
            merged[index] = replace(
                current,
                text=candidate.text,
                rect=candidate.rect,
                confidence=candidate.confidence,
                alternatives=_dedup_alternatives(
                    list(current.alternatives)
                    + [(current.text, current.confidence)]
                    + list(candidate.alternatives),
                    candidate.text,
                ),
            )
            changed += 1
        elif candidate.text != current.text:
            merged[index] = replace(
                current,
                alternatives=_dedup_alternatives(
                    list(current.alternatives) + [(candidate.text, candidate.confidence)],
                    current.text,
                ),
            )
    return merged, changed


def merge_words(
    base: Sequence[RasterWord], improved: Sequence[RasterWord]
) -> List[RasterWord]:
    """Merge re-recognized words back into the originals by position.

    Where a re-recognized word overlaps an original (IoU >= :data:`MERGE_MIN_IOU`), the
    higher-confidence reading wins and the loser is kept as an alternative, which is what
    lets a human or the council see the second opinion.  A re-recognized word that
    overlaps nothing is a word the full-page pass missed, so it is appended.

    Neither input list nor any word in it is mutated.
    """
    return _merge(base, improved)[0]


# ======================================================================================
# Words to spans
# ======================================================================================


def group_words_into_lines(words: Sequence[RasterWord]) -> List[List[RasterWord]]:
    """Group words into text lines by vertical overlap, ordered top-down then left-right.

    Two words share a line when their boxes overlap vertically by at least
    :data:`LINE_OVERLAP_RATIO` of the shorter of the two heights, which tolerates a
    lowercase ``o`` sitting next to a capital ``H`` while still splitting lines that are
    merely close.  Words from different pages never share a line.
    """
    items = [w for w in words if (w.text or "").strip()]
    order = sorted(range(len(items)), key=lambda i: (items[i].page, -items[i].rect.y1, items[i].rect.x0, i))
    lines: List[Dict[str, Any]] = []
    for i in order:
        word = items[i]
        placed = False
        for line in lines:
            if line["page"] != word.page:
                continue
            band: Rect = line["band"]
            overlap = band.vertical_overlap(word.rect)
            shorter = min(band.height, word.rect.height)
            if shorter <= 0.0:
                continue
            if overlap >= LINE_OVERLAP_RATIO * shorter:
                line["words"].append(word)
                line["band"] = band.union(word.rect)
                placed = True
                break
        if not placed:
            lines.append({"page": word.page, "band": word.rect, "words": [word]})
    for line in lines:
        line["words"].sort(key=lambda w: (w.rect.x0, -w.rect.y1))
    lines.sort(key=lambda item: (item["page"], -item["band"].y1, item["band"].x0))
    return [list(line["words"]) for line in lines]


def _line_span(line: Sequence[RasterWord]) -> Optional[TextSpan]:
    """Build one OCR :class:`~zfp.core.types.TextSpan` from one line of words."""
    if not line:
        return None
    rect = Rect.bounding([w.rect for w in line])
    if rect is None:
        return None
    heights = [w.rect.height for w in line if w.rect.height > 0.0]
    return TextSpan(
        text=" ".join(w.text.strip() for w in line),
        rect=rect,
        page=line[0].page,
        font_name="",
        font_size=float(median(heights)) if heights else 0.0,
        source="ocr",
        confidence=min(float(w.confidence) for w in line),
        glyph_rects=[w.rect for w in line],
        baseline=None,
    )


def words_to_spans(words: Sequence[RasterWord]) -> List[TextSpan]:
    """Turn OCR words into one :class:`~zfp.core.types.TextSpan` per text line.

    The span's confidence is the **minimum** of its words: a line is only as trustworthy
    as its worst word, and a label linker that trusts ``"Date of Birth"`` because two of
    its three words were clean would be lying to itself.  ``glyph_rects`` keeps the
    per-word boxes, so downstream geometry can still work at word resolution, and
    ``font_size`` is the median word height, which is the closest thing OCR offers.
    """
    spans: List[TextSpan] = []
    for line in group_words_into_lines(words):
        span = _line_span(line)
        if span is not None:
            spans.append(span)
    return spans


def words_to_word_spans(words: Sequence[RasterWord]) -> List[TextSpan]:
    """Turn OCR words into one span per **word**, in reading order.

    Detectors that care about individual tokens (comb cells, checkbox captions, single
    digits) want this rather than :func:`words_to_spans`.
    """
    spans: List[TextSpan] = []
    for line in group_words_into_lines(words):
        for word in line:
            spans.append(
                TextSpan(
                    text=word.text.strip(),
                    rect=word.rect,
                    page=word.page,
                    font_name="",
                    font_size=word.rect.height,
                    source="ocr",
                    confidence=float(word.confidence),
                    glyph_rects=[word.rect],
                    baseline=None,
                )
            )
    return spans
