"""``zfp.ocr`` -- recognition, but only where the PDF does not already know the answer.

Three modules:

``engine``
    One interface (:class:`~zfp.ocr.engine.OcrEngine`) over Tesseract, PaddleOCR and a
    null backend, with the pixel -> user-space conversion shared by all of them in
    :meth:`~zfp.ocr.engine.BaseEngine.to_user_space`.
``cascade``
    :func:`~zfp.ocr.cascade.ocr_cascade`: native text, then the cheapest engine that
    clears the confidence bar, then cropped re-recognition of what is left. Plus
    :func:`~zfp.ocr.cascade.words_to_spans`, which hands the rest of the pipeline OCR
    output in the same :class:`~zfp.core.types.TextSpan` shape the native parser
    produces, so no detector downstream needs to know where the text came from.
``suspects``
    Confidence and confusion-set analysis of the result, for the QA dashboard and for a
    human reviewer.

The rule that shapes all of it: **never OCR text the PDF already contains perfectly.**
OCR is lossy, slow and non-exact; a content stream is none of those things.

Nothing here imports a third-party library at import time, and no OCR backend needs to be
installed for the package to import, run, and report honestly that it found nothing.
"""

from __future__ import annotations

from .cascade import (
    CROP_PAD_PT,
    ENGINE_DISABLED,
    ENGINE_NONE,
    LINE_OVERLAP_RATIO,
    MAX_ESCALATION_CROPS,
    MERGE_MIN_IOU,
    MIN_NATIVE_TEXT_CHARS,
    SKIPPED_NATIVE_TEXT,
    OcrResult,
    crop_geometry,
    group_words_into_lines,
    has_native_text,
    mean_confidence,
    merge_words,
    ocr_cascade,
    recognize_regions,
    words_to_spans,
    words_to_word_spans,
)
from .engine import (
    BaseEngine,
    NullEngine,
    OcrEngine,
    PaddleEngine,
    PixelWord,
    TesseractEngine,
    available_engines,
    clear_engine_cache,
    engine_names,
    get_engine,
    parse_paddle_result,
    parse_tesseract_tsv,
    register_engine,
    resolve_ocr_config,
    unregister_engine,
)
from .suspects import (
    CONFUSABLE_CHARS,
    CONFUSION_MAP,
    MAX_ALTERNATIVES,
    REASON_LOW_CONFIDENCE,
    REASON_MIXED_ALNUM,
    Suspect,
    apply_correction,
    confidence_report,
    find_suspects,
    is_implausible_mix,
    suggest_alternatives,
)

__all__ = [
    # engine
    "OcrEngine",
    "BaseEngine",
    "PixelWord",
    "TesseractEngine",
    "PaddleEngine",
    "NullEngine",
    "parse_tesseract_tsv",
    "parse_paddle_result",
    "resolve_ocr_config",
    "register_engine",
    "unregister_engine",
    "get_engine",
    "engine_names",
    "available_engines",
    "clear_engine_cache",
    # cascade
    "OcrResult",
    "ocr_cascade",
    "recognize_regions",
    "crop_geometry",
    "merge_words",
    "has_native_text",
    "mean_confidence",
    "group_words_into_lines",
    "words_to_spans",
    "words_to_word_spans",
    "SKIPPED_NATIVE_TEXT",
    "ENGINE_NONE",
    "ENGINE_DISABLED",
    "MIN_NATIVE_TEXT_CHARS",
    "CROP_PAD_PT",
    "MAX_ESCALATION_CROPS",
    "MERGE_MIN_IOU",
    "LINE_OVERLAP_RATIO",
    # suspects
    "Suspect",
    "find_suspects",
    "suggest_alternatives",
    "is_implausible_mix",
    "apply_correction",
    "confidence_report",
    "CONFUSION_MAP",
    "CONFUSABLE_CHARS",
    "MAX_ALTERNATIVES",
    "REASON_LOW_CONFIDENCE",
    "REASON_MIXED_ALNUM",
]
