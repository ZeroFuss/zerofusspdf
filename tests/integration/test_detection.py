"""End-to-end detection quality against the synthetic corpus' exact ground truth.

This is the first test in the suite that asks the whole perception layer a question it
cannot answer one module at a time: *given a real PDF, do we find the fields that are
actually on it, in the right places?*  The corpus is :mod:`zfp.synth`, which knows the
answer by construction -- it placed the label, drew the rule and recorded the rectangle
-- so recall, precision and IoU are all exactly computable with no hand-labelling.

The rectangle conventions being measured are the ones ``zfp/synth/layouts.py`` documents
and ``docs/CONTRACT.md`` parameterises, and every detector, the fusion stage and the
generator now agree on them:

* **Underline** -- the rule's x-range, starting ``DetectionConfig.underline_gap_pt``
  above the rule and ``DetectionConfig.field_height_pt`` tall.
* **Box / table cell / comb** -- the drawn rectangle deflated by the stroke width: the
  *inside* of the ink, because the ink is not writable.
* **Checkbox / radio** -- the glyph's own bounding box.
* **Borderless** -- the blank region under or beside the label.

:func:`measure_all` prints the whole measured table and is the hook the next phase will
tighten against ``docs/QA.md``.  Run it directly with::

    PYTHONPATH=src python3 -m tests.integration.test_detection
"""

from __future__ import annotations

import unittest
from typing import Dict, List, Optional, Sequence, Tuple

from tests.fixtures.factory import build
from zfp.candidates.context import CandidateContext
from zfp.core.config import ZfpConfig
from zfp.core.geometry import Rect
from zfp.core.types import FieldCandidate, PageMode
from zfp.pdfio.document import Document
from zfp.pdfio.filters import encode_flate
from zfp.pdfio.objects import PdfDict, PdfName, PdfStream
from zfp.pipeline.detect import (
    detect_document,
    detect_page,
    page_context,
    page_sensing,
    wants_raster,
)
from zfp.synth.generator import GroundTruthField

#: Kinds measured by :func:`measure_all` and by :class:`DetectionQualityTests`.
KINDS: Tuple[str, ...] = ("underline", "boxed", "checkbox", "comb", "table", "mixed")

#: Documents per kind.  Seeds are consecutive from zero, so the corpus is reproducible.
SEEDS = 6

#: The IoU at which a prediction is allowed to claim a true field.
MATCH_IOU = 0.50

#: Per-kind gates, set just under what this pass actually measures so a regression is
#: caught rather than absorbed.  Every one of them is at or above the corresponding
#: ``docs/QA.md`` threshold, which is the number the next phase will assert directly.
#:
#: ``{kind: (min_recall, min_precision, min_mean_iou)}``
GATES: Dict[str, Tuple[float, float, float]] = {
    "underline": (0.95, 0.90, 0.92),
    "boxed": (0.95, 0.90, 0.92),
    "checkbox": (0.95, 0.90, 0.92),
    "comb": (0.95, 0.90, 0.92),
    "table": (0.95, 0.90, 0.92),
    "mixed": (0.92, 0.90, 0.90),
}

#: The thresholds ``docs/QA.md`` enforces, for reference in the printed table.
QA_TARGETS: Dict[str, Tuple[float, float]] = {
    "underline": (0.90, 0.80),
    "boxed": (0.90, 0.80),
    "checkbox": (0.85, 0.70),
    "comb": (0.85, 0.75),
    "mixed": (0.80, 0.72),
}


# =======================================================================================
# The scorer
# =======================================================================================
class Score:
    """One kind's detection quality.

    Attributes:
        kind: The corpus kind measured.
        documents: How many documents went into the numbers.
        true_positives: Predictions greedily matched to a true field.
        false_positives: Predictions that matched nothing.
        false_negatives: True fields that nothing matched.
        ious: The IoU of every matched pair.
    """

    __slots__ = ("kind", "documents", "true_positives", "false_positives", "false_negatives", "ious")

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.documents = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.ious: List[float] = []

    @property
    def recall(self) -> float:
        """Fraction of true fields that were found."""
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total else 1.0

    @property
    def precision(self) -> float:
        """Fraction of predictions that were real fields."""
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total else 1.0

    @property
    def mean_iou(self) -> float:
        """Mean IoU over the matched pairs (zero when nothing matched)."""
        return sum(self.ious) / len(self.ious) if self.ious else 0.0

    def add(self, matches: Sequence[Tuple[int, int, float]], truth: int, predicted: int) -> None:
        """Fold one document's result in."""
        self.documents += 1
        self.true_positives += len(matches)
        self.false_negatives += truth - len(matches)
        self.false_positives += predicted - len(matches)
        self.ious.extend(iou for _t, _p, iou in matches)

    def row(self) -> str:
        """One formatted line of the printed table."""
        target = QA_TARGETS.get(self.kind)
        qa = "%.2f / %.2f" % target if target else "     -     "
        return "%-11s %5d %5d %8.3f %10.3f %8.3f %6d %5d %5d   %s" % (
            self.kind,
            self.documents,
            self.true_positives + self.false_negatives,
            self.recall,
            self.precision,
            self.mean_iou,
            self.true_positives,
            self.false_positives,
            self.false_negatives,
            qa,
        )


def greedy_match(
    truth: Sequence[GroundTruthField],
    predicted: Sequence[FieldCandidate],
    threshold: float = MATCH_IOU,
) -> List[Tuple[int, int, float]]:
    """Greedily pair true fields with predictions by IoU, best pair first.

    Only rectangles on the same page may pair, and each true field and each prediction is
    used at most once.  Greedy-by-best-IoU is not optimal in general, but it is
    deterministic, obvious, and identical to what ``qa.metrics.recall_precision`` will
    need to do -- and on a form, where fields do not overlap, it *is* optimal.

    Args:
        truth: The generator's ground-truth fields.
        predicted: The detected candidates.
        threshold: The minimum IoU for a pair to count.

    Returns:
        ``(truth_index, predicted_index, iou)`` triples, strongest match first.
    """
    pairs: List[Tuple[float, int, int]] = []
    for t_index, field in enumerate(truth):
        for p_index, candidate in enumerate(predicted):
            if int(field.page) != int(candidate.page):
                continue
            iou = field.rect.iou(candidate.rect)
            if iou >= threshold:
                pairs.append((iou, t_index, p_index))
    # -iou first, then the indices, so ties resolve the same way on every machine.
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_truth = set()
    used_pred = set()
    out: List[Tuple[int, int, float]] = []
    for iou, t_index, p_index in pairs:
        if t_index in used_truth or p_index in used_pred:
            continue
        used_truth.add(t_index)
        used_pred.add(p_index)
        out.append((t_index, p_index, iou))
    return out


def score_kind(kind: str, seeds: int = SEEDS, config: Optional[ZfpConfig] = None) -> Score:
    """Detect over ``seeds`` documents of one kind and score the result."""
    cfg = config or ZfpConfig.default()
    score = Score(kind)
    for seed in range(seeds):
        form = build(kind, seed)
        doc = Document.open(form.pdf_bytes)
        _profile, candidates = detect_document(doc, cfg)
        score.add(greedy_match(form.fields, candidates), len(form.fields), len(candidates))
    return score


def measure_all(seeds: int = SEEDS, kinds: Sequence[str] = KINDS) -> Dict[str, Score]:
    """Measure every kind, print the table, and return the scores.

    The printed table is the artefact this phase exists to produce; the next phase
    replaces :data:`GATES` with the ``docs/QA.md`` thresholds and asserts against those.

    Args:
        seeds: Documents per kind.
        kinds: Which corpus kinds to measure.

    Returns:
        ``{kind: Score}``.
    """
    scores = {kind: score_kind(kind, seeds) for kind in kinds}
    header = "%-11s %5s %5s %8s %10s %8s %6s %5s %5s   %s" % (
        "kind", "docs", "true", "recall", "precision", "meanIoU", "tp", "fp", "fn", "QA R/IoU"
    )
    print()
    print(header)
    print("-" * len(header))
    for kind in kinds:
        print(scores[kind].row())
    print()
    return scores


# =======================================================================================
# Fixtures shared by the behavioural tests
# =======================================================================================
def _scanned_document(width: int = 240, height: int = 320) -> Document:
    """A one-page document holding nothing but a full-page grayscale image.

    Three dark rules are painted into the samples so the raster path has something to
    find; there is no text and no vector operator anywhere in the content stream.
    """
    gray = bytearray([255] * (width * height))
    for row in (int(height * 0.25), int(height * 0.5), int(height * 0.75)):
        for column in range(int(width * 0.1), int(width * 0.9)):
            for thickness in range(2):
                gray[(row + thickness) * width + column] = 0
    image = PdfStream(
        PdfDict(
            {
                "Type": PdfName("XObject"),
                "Subtype": PdfName("Image"),
                "Width": width,
                "Height": height,
                "Filter": PdfName("FlateDecode"),
                "ColorSpace": PdfName("DeviceGray"),
                "BitsPerComponent": 8,
            }
        ),
        encode_flate(bytes(gray)),
    )
    doc = Document.from_pages_blank(1, float(width), float(height))
    page = doc.page(0)
    reference = doc.writer.add_object(image)
    contents = doc.writer.add_object(
        PdfStream(PdfDict({}), b"q %d 0 0 %d 0 0 cm /Im0 Do Q" % (width, height))
    )
    page.dict["Resources"] = PdfDict({"XObject": PdfDict({"Im0": reference})})
    page.dict["Contents"] = contents
    page.touch()
    return Document.open(doc.to_bytes(incremental=False))


# =======================================================================================
# 1. Detection quality
# =======================================================================================
class DetectionQualityTests(unittest.TestCase):
    """Recall, precision and IoU against the generator's exact rectangles."""

    scores: Dict[str, Score] = {}

    @classmethod
    def setUpClass(cls) -> None:
        cls.scores = {kind: score_kind(kind) for kind in KINDS}

    def test_every_kind_clears_its_gate(self) -> None:
        for kind in KINDS:
            score = self.scores[kind]
            min_recall, min_precision, min_iou = GATES[kind]
            with self.subTest(kind=kind):
                self.assertGreaterEqual(
                    score.recall, min_recall, "%s recall: %s" % (kind, score.row())
                )
                self.assertGreaterEqual(
                    score.precision, min_precision, "%s precision: %s" % (kind, score.row())
                )
                self.assertGreaterEqual(
                    score.mean_iou, min_iou, "%s mean IoU: %s" % (kind, score.row())
                )

    def test_gates_are_at_or_above_the_published_qa_thresholds(self) -> None:
        # A gate that drifts below docs/QA.md would let the build go green on a build
        # that the quality-gate phase will fail.
        for kind, (qa_recall, qa_iou) in QA_TARGETS.items():
            with self.subTest(kind=kind):
                self.assertGreaterEqual(GATES[kind][0], qa_recall)
                self.assertGreaterEqual(GATES[kind][2], qa_iou)

    def test_something_was_actually_measured(self) -> None:
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.assertEqual(self.scores[kind].documents, SEEDS)
                self.assertGreater(self.scores[kind].true_positives, 10)

    def test_detection_is_deterministic(self) -> None:
        doc = Document.open(build("mixed", 0).pdf_bytes)
        first = [c.as_dict() for c in detect_document(doc)[1]]
        second = [c.as_dict() for c in detect_document(Document.open(build("mixed", 0).pdf_bytes))[1]]
        self.assertEqual(first, second)

    def test_every_candidate_lies_inside_its_crop_box(self) -> None:
        # An absolute invariant from docs/QA.md, checked here on real geometry.
        for kind in ("mixed", "table", "multipage"):
            form = build(kind, 3)
            doc = Document.open(form.pdf_bytes)
            profile, candidates = detect_document(doc)
            boxes = {page.index: page.geometry.crop_box for page in profile.pages}
            for candidate in candidates:
                with self.subTest(kind=kind, candidate=candidate.id):
                    self.assertTrue(
                        boxes[candidate.page].contains_rect(candidate.rect),
                        "%s escapes %s" % (candidate.rect.as_list(), boxes[candidate.page].as_list()),
                    )

    def test_rotated_pages_are_detected_too(self) -> None:
        # /Rotate does not move user-space rectangles, so detection must not move either.
        for rotation in (90, 180, 270):
            form = build("underline", 2, rotation=rotation)
            doc = Document.open(form.pdf_bytes)
            _profile, candidates = detect_document(doc)
            matches = greedy_match(form.fields, candidates)
            with self.subTest(rotation=rotation):
                self.assertGreaterEqual(len(matches), int(0.9 * len(form.fields)))


# =======================================================================================
# 2. The cascade rule
# =======================================================================================
class CascadeRuleTests(unittest.TestCase):
    """**Never OCR a page that already has native text.**"""

    def test_a_native_page_is_never_ocred_or_rendered(self) -> None:
        doc = Document.open(build("mixed", 1).pdf_bytes)
        sensing = page_sensing(doc, 0)
        self.assertTrue(sensing.profile.has_native_text)
        self.assertEqual(sensing.path, "native")
        # An empty engine name is the proof: ocr_cascade was never called at all.
        self.assertEqual(sensing.ocr_engine, "")
        self.assertEqual(sensing.render_backend, "")
        self.assertEqual(sensing.words, [])
        self.assertFalse(wants_raster(sensing.profile))
        self.assertTrue(all(span.source == "native" for span in sensing.spans))

    def test_the_cascade_is_not_consulted_for_any_native_page(self) -> None:
        import zfp.pipeline.detect as detect_module

        calls = []
        original = detect_module.ocr_cascade

        def spy(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        detect_module.ocr_cascade = spy
        try:
            for kind in KINDS:
                doc = Document.open(build(kind, 0).pdf_bytes)
                detect_document(doc)
        finally:
            detect_module.ocr_cascade = original
        self.assertEqual(calls, [])

    def test_a_scanned_page_does_take_the_raster_path(self) -> None:
        doc = _scanned_document()
        sensing = page_sensing(doc, 0)
        self.assertFalse(sensing.profile.has_native_text)
        self.assertTrue(wants_raster(sensing.profile))
        self.assertEqual(sensing.path, "raster")
        # No renderer and no OCR engine need be installed; the embedded-image fallback
        # and the null engine are always available, and both report honestly.
        self.assertTrue(sensing.render_backend)
        self.assertTrue(sensing.ocr_engine)
        self.assertTrue(sensing.primitives, "raster shapes found nothing on a ruled scan")

    def test_a_scanned_page_yields_candidates_from_its_pixels(self) -> None:
        doc = _scanned_document()
        profile, candidates = detect_document(doc)
        self.assertIs(profile.pages[0].mode, PageMode.SCANNED_FORM)
        # Three ruled lines in the image, three fields -- found without a renderer, an
        # OCR engine or numpy, through the embedded-image fallback and the pure-python
        # raster shape pass.
        self.assertEqual(len(candidates), 3)
        for candidate in candidates:
            self.assertEqual(candidate.page, 0)
            self.assertGreater(candidate.rect.width, 0.0)

    def test_recognised_words_reach_the_detectors_as_labels(self) -> None:
        """The whole raster branch, with an engine installed: pixels -> labelled fields."""
        from zfp.ocr.engine import (
            BaseEngine,
            PixelWord,
            clear_engine_cache,
            register_engine,
            unregister_engine,
        )

        class StubEngine(BaseEngine):
            """Reads one word to the left of each painted rule, in pixel space."""

            name = "stub"

            def available(self) -> bool:
                return True

            def _recognize_pixels(self, page, config):  # noqa: ANN001 - test double
                words = []
                for index, row in enumerate((0.25, 0.5, 0.75)):
                    top = page.height * row - 12.0
                    words.append(
                        PixelWord(
                            text=("Name", "City", "Phone")[index],
                            rect=Rect(4.0, top, 40.0, top + 10.0),
                            confidence=0.95,
                            line_id=index,
                        )
                    )
                return words

        config = ZfpConfig.default()
        config.ocr.engines = ["stub"]
        register_engine("stub", StubEngine)
        clear_engine_cache()
        try:
            doc = _scanned_document()
            sensing = page_sensing(doc, 0, config)
            self.assertEqual(sensing.ocr_engine, "stub")
            self.assertEqual(len(sensing.words), 3)
            self.assertTrue(sensing.spans)
            self.assertTrue(all(span.source == "ocr" for span in sensing.spans))
            labels = {c.visible_label for c in detect_page(doc, 0, config)}
            self.assertTrue(
                {"Name", "City", "Phone"} & labels,
                "OCR words never reached the detectors: %r" % (labels,),
            )
        finally:
            unregister_engine("stub")
            clear_engine_cache()


# =======================================================================================
# 3. Degradation
# =======================================================================================
class EmptyAndBrokenPageTests(unittest.TestCase):
    """A page that is neither native nor raster must cost nothing and crash nothing."""

    def test_a_blank_page_yields_an_empty_context_and_no_candidates(self) -> None:
        doc = Document.open(Document.from_pages_blank(2).to_bytes(incremental=False))
        for index in (0, 1):
            with self.subTest(page=index):
                context, profile = page_context(doc, index, None)
                self.assertIsInstance(context, CandidateContext)
                self.assertIs(profile.mode, PageMode.EMPTY)
                self.assertEqual(context.spans, [])
                self.assertEqual(context.primitives, [])
                self.assertEqual(detect_page(doc, index), [])

    def test_a_blank_document_profiles_and_detects_without_raising(self) -> None:
        doc = Document.open(Document.from_pages_blank(3).to_bytes(incremental=False))
        profile, candidates = detect_document(doc)
        self.assertEqual(profile.page_count, 3)
        self.assertEqual(candidates, [])

    def test_a_broken_content_stream_degrades_to_no_candidates(self) -> None:
        doc = Document.from_pages_blank(1)
        page = doc.page(0)
        page.dict["Contents"] = doc.writer.add_object(
            PdfStream(PdfDict({}), b"BT /Nope 12 Tf 1 2 3 nonsense ET q q q 5 5 re")
        )
        page.touch()
        doc = Document.open(doc.to_bytes(incremental=False))
        self.assertEqual(detect_page(doc, 0), [])

    def test_page_context_accepts_a_precomputed_profile(self) -> None:
        doc = Document.open(build("boxed", 0).pdf_bytes)
        profile, _candidates = detect_document(doc)
        context, page_profile = page_context(doc, 0, None, profile=profile)
        self.assertEqual(page_profile.index, 0)
        self.assertTrue(context.spans)
        # A single PageProfile works too, as does junk (which is ignored, not fatal).
        again, _ = page_context(doc, 0, None, profile=profile.pages[0])
        self.assertEqual(len(again.spans), len(context.spans))
        junk, _ = page_context(doc, 0, None, profile="not a profile")
        self.assertEqual(len(junk.spans), len(context.spans))


# =======================================================================================
# 4. Geometry conventions
# =======================================================================================
class GeometryConventionTests(unittest.TestCase):
    """The rectangle conventions the whole layer has to share to score at all."""

    def test_an_underline_field_sits_above_its_rule_at_the_configured_height(self) -> None:
        config = ZfpConfig.default()
        detection = config.detection
        form = build("underline", 0)
        doc = Document.open(form.pdf_bytes)
        context, _profile = page_context(doc, 0, config)
        rule_levels = sorted({round(rule.rect.y1, 3) for rule in context.h_rules})
        matched = 0
        for candidate in detect_page(doc, 0, config):
            expected = round(candidate.rect.y0 - detection.underline_gap_pt, 3)
            if expected in rule_levels:
                matched += 1
                self.assertAlmostEqual(
                    candidate.rect.height, detection.field_height_pt, places=3
                )
        self.assertGreater(matched, 5, "no candidate was anchored to a rule")

    def test_a_box_field_is_the_inside_of_the_ink(self) -> None:
        form = build("boxed", 0)
        doc = Document.open(form.pdf_bytes)
        candidates = detect_page(doc, 0)
        context, _profile = page_context(doc, 0, None)
        drawn = [p.rect for p in context.primitives if p.kind == "rect"]
        self.assertTrue(drawn)
        for candidate in candidates:
            enclosing = [box for box in drawn if box.contains_rect(candidate.rect)]
            if not enclosing:
                continue
            box = enclosing[0]
            with self.subTest(candidate=candidate.id):
                # Inside the ink, and not by much: the deflation is the stroke width.
                self.assertGreater(candidate.rect.x0, box.x0)
                self.assertLess(candidate.rect.x1, box.x1)
                self.assertLess(box.x0, candidate.rect.x0 - 0.0)
                self.assertLess(candidate.rect.x0 - box.x0, 3.0)

    def test_fusion_does_not_move_a_measured_rectangle(self) -> None:
        # The regression that cost ~0.30 mean IoU: fusion re-snapping and padding a
        # rectangle a detector had already placed by its archetype's convention.
        from zfp.candidates.archetypes import generate_candidates
        from zfp.fusion.geometry_fusion import fuse, has_exact_geometry

        for kind in ("boxed", "table", "comb", "checkbox"):
            doc = Document.open(build(kind, 0).pdf_bytes)
            context, _profile = page_context(doc, 0, None)
            raw = {c.id: c.rect for c in generate_candidates(context) if has_exact_geometry(c)}
            self.assertTrue(raw, kind)
            fused = fuse(
                generate_candidates(context),
                ZfpConfig.default(),
                {0: list(context.primitives)},
                {0: context.geometry},
            )
            for candidate in fused:
                if candidate.id in raw:
                    with self.subTest(kind=kind, candidate=candidate.id):
                        self.assertEqual(
                            candidate.rect.rounded(3).as_list(),
                            raw[candidate.id].rounded(3).as_list(),
                        )


if __name__ == "__main__":  # pragma: no cover - manual measurement entry point
    measure_all()
