"""The end-to-end perception path: an open document in, field candidates out.

This module is the single place where the eight perception packages are wired into one
system.  Every stage below it is deliberately ignorant of the others -- the content-stream
interpreter does not know OCR exists, the archetype detectors do not know whether their
spans were parsed or recognised -- and this module is what makes that separation pay off:
it decides, per page, which sensing path applies, normalises the result into the one shape
:mod:`zfp.candidates` consumes, and hands the union to :mod:`zfp.fusion`.

The routing rule is the spec's cascade, and it is not a heuristic:

* A page that carries **native text** is read with
  :class:`~zfp.native.content.ContentStreamInterpreter`.  Its spans are exact, its
  primitives are exact, and it is **never** rasterized or OCR'd -- see
  :func:`page_context`, which never calls :func:`~zfp.ocr.cascade.ocr_cascade` on such a
  page, and :func:`~zfp.ocr.cascade.ocr_cascade` itself, which refuses a second time.
* A page that carries **no native text but does carry raster** is rendered
  (:func:`~zfp.raster.render.render_page`, which falls back to compositing the page's own
  image XObjects when no renderer is installed), recognised
  (:func:`~zfp.ocr.cascade.ocr_cascade`) and measured
  (:func:`~zfp.vision.raster_shapes.detect_shapes_from_image`).
* A page that is **neither** -- an empty page, an unreadable one, a scan whose codec
  nobody can decode -- yields an empty context and no candidates.  It never raises.

Vector primitives are always collected, even on a scanned page: a hybrid page with a
stamped vector grid over a photographed background is common, and the vector geometry is
free and exact wherever it exists.

    >>> from zfp.pdfio.document import Document
    >>> from zfp.synth import SynthOptions, generate
    >>> from zfp.pipeline.detect import detect_document
    >>> doc = Document.open(generate(SynthOptions(kind="underline", seed=3)).pdf_bytes)
    >>> profile, candidates = detect_document(doc)
    >>> profile.page_count
    1
    >>> len(candidates) > 0
    True
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..candidates.archetypes import generate_candidates
from ..candidates.context import CandidateContext, build_context
from ..core.config import ZfpConfig
from ..core.errors import ZfpError
from ..core.geometry import PageGeometry, Rect
from ..core.logging import get_logger
from ..core.types import (
    DocumentProfile,
    FieldCandidate,
    PageMode,
    PageProfile,
    RasterWord,
    TextSpan,
    VectorPrimitive,
)
from ..fusion.geometry_fusion import fuse
from ..native.content import analyze_page
from ..ocr.cascade import ocr_cascade, words_to_spans
from ..preflight.classifier import classify_page, profile_document
from ..raster.render import RenderedPage, embedded_page_images, render_page
from ..vision.raster_shapes import detect_shapes_from_image

__all__ = [
    "NATIVE_MODES",
    "RASTER_MODES",
    "PageSensing",
    "detect_document",
    "detect_page",
    "page_context",
    "page_sensing",
    "wants_raster",
]

LOG = get_logger(__name__)

#: Page modes whose geometry and text come from the content stream.
NATIVE_MODES = frozenset(
    (
        PageMode.NATIVE_DOCUMENT,
        PageMode.FLAT_NATIVE_FORM,
        PageMode.INTERACTIVE_FORM,
        PageMode.HYBRID,
    )
)

#: Page modes whose geometry and text come from pixels.
RASTER_MODES = frozenset((PageMode.SCANNED_FORM, PageMode.SCANNED_DOCUMENT))


# =======================================================================================
# Per-page sensing
# =======================================================================================
class PageSensing:
    """What one page's sensors produced, before any candidate is proposed.

    Attributes:
        profile: The page's :class:`~zfp.core.types.PageProfile`.
        spans: Text spans, native or recognised (``span.source`` says which).
        primitives: Vector primitives, from the content stream and/or the raster shapes.
        words: OCR words in user space; empty on a native page.
        images: User-space rectangles of the raster images painted on the page.
        path: ``"native"``, ``"raster"``, ``"hybrid"`` or ``"empty"``.
        ocr_engine: The engine the cascade used, or ``""`` when it was never called.
        render_backend: The renderer that produced the raster, or ``""``.
        notes: Human-readable degradations (a missing renderer, an undecodable scan).
    """

    __slots__ = (
        "profile",
        "spans",
        "primitives",
        "words",
        "images",
        "path",
        "ocr_engine",
        "render_backend",
        "notes",
    )

    def __init__(
        self,
        profile: PageProfile,
        spans: Optional[List[TextSpan]] = None,
        primitives: Optional[List[VectorPrimitive]] = None,
        words: Optional[List[RasterWord]] = None,
        images: Optional[List[Rect]] = None,
        path: str = "empty",
        ocr_engine: str = "",
        render_backend: str = "",
        notes: Optional[List[str]] = None,
    ) -> None:
        self.profile = profile
        self.spans: List[TextSpan] = list(spans or ())
        self.primitives: List[VectorPrimitive] = list(primitives or ())
        self.words: List[RasterWord] = list(words or ())
        self.images: List[Rect] = list(images or ())
        self.path = path
        self.ocr_engine = ocr_engine
        self.render_backend = render_backend
        self.notes: List[str] = list(notes or ())

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            "PageSensing(page=%d, path=%r, spans=%d, primitives=%d, words=%d, ocr=%r)"
            % (
                self.profile.index,
                self.path,
                len(self.spans),
                len(self.primitives),
                len(self.words),
                self.ocr_engine,
            )
        )


def wants_raster(profile: PageProfile) -> bool:
    """True when this page must be sensed through pixels.

    The cascade rule in one predicate: a page with native text is *never* rasterized for
    recognition, whatever else it contains.  A page without native text is rasterized
    when it holds raster content or was classified into a scanned mode.
    """
    if profile.has_native_text:
        return False
    return bool(profile.has_raster) or profile.mode in RASTER_MODES


def _page_profile(
    doc: object, index: int, config: ZfpConfig, profile: object = None
) -> PageProfile:
    """Resolve ``profile`` (a page profile, a document profile, or ``None``)."""
    if isinstance(profile, PageProfile):
        return profile
    if isinstance(profile, DocumentProfile):
        for page in profile.pages:
            if page.index == index:
                return page
    if profile is not None:
        LOG.debug("page %d: ignoring unusable profile %r", index, type(profile).__name__)
    return classify_page(doc, index, config)


def _native_pass(
    doc: object, index: int, config: ZfpConfig
) -> Tuple[List[TextSpan], List[VectorPrimitive], List[Rect], int]:
    """Interpret the content stream; never raises."""
    try:
        result = analyze_page(doc.page(index), doc, config)
    except Exception as exc:  # noqa: BLE001 - a broken page must not kill the document
        LOG.warning("page %d: content stream unreadable (%s)", index, exc)
        return ([], [], [], 0)
    return (list(result.spans), list(result.primitives), list(result.images), result.errors)


def _render(
    doc: object, index: int, config: ZfpConfig, notes: List[str]
) -> Optional[RenderedPage]:
    """Rasterize one page, degrading to ``None`` instead of raising.

    ``render_page`` already tries the dependency-free ``embedded`` backend last, so a
    failure here means no installed renderer *and* no decodable page image.  The
    contract's explicit fallback is still exercised, because the image placements it
    returns are usable evidence even when their pixels are not.
    """
    dpi = int(getattr(config.ocr, "dpi", 300) or 300)
    try:
        return render_page(doc, index, dpi=dpi)
    except ZfpError as exc:
        notes.append("render_failed: %s" % exc)
        LOG.info("page %d: no raster available (%s)", index, exc)
    except Exception as exc:  # noqa: BLE001 - defensive; a backend must not be fatal
        notes.append("render_error: %s" % exc)
        LOG.warning("page %d: renderer raised (%s)", index, exc)
    return None


def _embedded_image_rects(doc: object, index: int, notes: List[str]) -> List[Rect]:
    """The contract's fallback path: image placements without a renderer."""
    try:
        placements = embedded_page_images(doc, index)
    except Exception as exc:  # noqa: BLE001
        notes.append("embedded_images_error: %s" % exc)
        return []
    if placements:
        notes.append("embedded_images: %d" % len(placements))
    return [rect for rect, _data, _codec in placements]


def _raster_pass(
    doc: object,
    index: int,
    geometry: PageGeometry,
    config: ZfpConfig,
    sensing: PageSensing,
) -> None:
    """Render, recognise and measure a page that has no native text.

    Mutates ``sensing`` in place.  Every step degrades independently: a page that renders
    but has no OCR engine still contributes raster shapes, and a page that neither renders
    nor decodes still contributes its image placements.
    """
    rendered = _render(doc, index, config, sensing.notes)
    if rendered is None:
        sensing.images.extend(_embedded_image_rects(doc, index, sensing.notes))
        return
    sensing.render_backend = rendered.backend

    # -- recognition -------------------------------------------------------------------
    # Reached only when the page has no native text, so the cascade's own guard is a
    # second line of defence rather than the first.
    if getattr(config.ocr, "enabled", True):
        try:
            ocr = ocr_cascade(rendered, geometry, config.ocr, native_spans=())
        except Exception as exc:  # noqa: BLE001
            sensing.notes.append("ocr_error: %s" % exc)
            LOG.warning("page %d: OCR cascade failed (%s)", index, exc)
        else:
            sensing.ocr_engine = ocr.engine
            sensing.words.extend(ocr.words)
            sensing.spans.extend(words_to_spans(ocr.words))
    else:
        sensing.notes.append("ocr_disabled")

    # -- shapes ------------------------------------------------------------------------
    try:
        shapes = detect_shapes_from_image(rendered, geometry, config, rendered.scale)
    except ZfpError as exc:
        sensing.notes.append("raster_shapes_unavailable: %s" % exc)
    except Exception as exc:  # noqa: BLE001
        sensing.notes.append("raster_shapes_error: %s" % exc)
        LOG.warning("page %d: raster shape detection failed (%s)", index, exc)
    else:
        sensing.primitives.extend(shapes.as_primitives(index))


def page_sensing(
    doc: object, index: int, config: Optional[ZfpConfig] = None, *, profile: object = None
) -> PageSensing:
    """Run every sensor that applies to one page and return the raw evidence.

    Args:
        doc: An open :class:`~zfp.pdfio.document.Document`.
        index: Zero-based page index.
        config: Run configuration; ``ZfpConfig.default()`` when omitted.
        profile: A previously computed :class:`~zfp.core.types.PageProfile` (or the whole
            :class:`~zfp.core.types.DocumentProfile`) to save a re-classification.

    Returns:
        The :class:`PageSensing` for the page.  Never raises for a malformed page.
    """
    cfg = config or ZfpConfig.default()
    page_profile = _page_profile(doc, index, cfg, profile)

    spans, primitives, images, errors = _native_pass(doc, index, cfg)
    sensing = PageSensing(
        page_profile, spans=spans, primitives=primitives, images=images, path="empty"
    )
    if errors:
        sensing.notes.append("content_stream_errors: %d" % errors)

    native_text = bool(spans) and page_profile.has_native_text
    if wants_raster(page_profile):
        _raster_pass(doc, index, page_profile.geometry, cfg, sensing)
        sensing.path = "raster" if (sensing.words or sensing.primitives) else "empty"
    elif native_text:
        sensing.path = "hybrid" if page_profile.mode is PageMode.HYBRID else "native"
    elif sensing.primitives:
        # Vector geometry with no text at all: still worth detecting on.
        sensing.path = "native"

    LOG.debug(
        "page %d: path=%s spans=%d primitives=%d words=%d",
        index,
        sensing.path,
        len(sensing.spans),
        len(sensing.primitives),
        len(sensing.words),
    )
    return sensing


def page_context(
    doc: object, index: int, config: Optional[ZfpConfig] = None, *, profile: object = None
) -> Tuple[CandidateContext, PageProfile]:
    """Build a :class:`~zfp.candidates.context.CandidateContext` for one page.

    Chooses the native or the raster sensing path from the page profile, then derives the
    shared page structure every archetype detector reads.

    Args:
        doc: An open :class:`~zfp.pdfio.document.Document`.
        index: Zero-based page index.
        config: Run configuration; ``ZfpConfig.default()`` when omitted.
        profile: A pre-computed page or document profile, to avoid re-classifying.

    Returns:
        ``(context, page_profile)``.  A page that is neither native nor raster yields an
        empty-but-valid context rather than an error.
    """
    cfg = config or ZfpConfig.default()
    sensing = page_sensing(doc, index, cfg, profile=profile)
    ctx = build_context(
        index,
        sensing.profile.geometry,
        sensing.spans,
        sensing.primitives,
        sensing.words,
        cfg,
        _existing_widgets(doc, index) if sensing.profile.has_widgets else (),
    )
    return (ctx, sensing.profile)


def _existing_widgets(doc: object, index: int) -> Sequence[Rect]:
    """Rectangles of the widget annotations already on the page (never raises)."""
    try:
        specs = doc.existing_fields()
    except Exception:  # noqa: BLE001 - a broken AcroForm must not stop detection
        return ()
    out: List[Rect] = []
    for spec in specs:
        if int(getattr(spec, "page", -1)) == index:
            out.append(spec.rect)
        for extra_page, extra_rect in getattr(spec, "extra_widgets", ()) or ():
            if int(extra_page) == index:
                out.append(extra_rect)
    return out


# =======================================================================================
# Detection
# =======================================================================================
def detect_page(
    doc: object, index: int, config: Optional[ZfpConfig] = None, *, profile: object = None
) -> List[FieldCandidate]:
    """Detect the fields on one page: sense, propose, fuse.

    Args:
        doc: An open :class:`~zfp.pdfio.document.Document`.
        index: Zero-based page index.
        config: Run configuration; ``ZfpConfig.default()`` when omitted.
        profile: A pre-computed page or document profile.

    Returns:
        The fused candidates for that page, in reading order.  Empty for a page with no
        detectable structure.
    """
    cfg = config or ZfpConfig.default()
    ctx, _page_profile = page_context(doc, index, cfg, profile=profile)
    raw = generate_candidates(ctx)
    return fuse(raw, cfg, {index: list(ctx.primitives)}, {index: ctx.geometry})


def detect_document(
    doc: object, config: Optional[ZfpConfig] = None
) -> Tuple[DocumentProfile, List[FieldCandidate]]:
    """Profile a document and detect every field on every page.

    Fusion runs once over the whole document, so a candidate proposed on page 3 is scored
    against the same weights as one proposed on page 1 and cross-page duplicates (a
    repeated header field, say) are ranked consistently.  Fusion never merges across
    pages: :func:`~zfp.fusion.geometry_fusion.deduplicate` compares page numbers first.

    Args:
        doc: An open :class:`~zfp.pdfio.document.Document`.
        config: Run configuration; ``ZfpConfig.default()`` when omitted.

    Returns:
        ``(document_profile, candidates)`` with candidates sorted by page and then in
        reading order.
    """
    cfg = config or ZfpConfig.default()
    profile = profile_document(doc, cfg)

    raw: List[FieldCandidate] = []
    primitives_by_page: Dict[int, List[VectorPrimitive]] = {}
    geometry_by_page: Dict[int, PageGeometry] = {}
    for page_profile in profile.pages:
        index = page_profile.index
        ctx, _ = page_context(doc, index, cfg, profile=page_profile)
        primitives_by_page[index] = list(ctx.primitives)
        geometry_by_page[index] = ctx.geometry
        raw.extend(generate_candidates(ctx))

    fused = fuse(raw, cfg, primitives_by_page, geometry_by_page)
    fused.sort(key=lambda c: (c.page, c.order))
    return (profile, fused)
