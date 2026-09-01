"""Unit tests for :mod:`zfp.fusion.calibration`.

Three separate things are checked here: the fixed per-type padding, the sanity clamps
that keep a runaway detection out of the writer, and the *learned* corrections -- a
corpus whose predictions are all systematically 2pt too far left must produce a
calibrator that puts them back exactly where they belong.
"""

from __future__ import annotations

import unittest

from zfp.core.config import SCORING_WEIGHT_NAMES, DetectionConfig, ScoringWeights, ZfpConfig
from zfp.core.errors import ValidationError
from zfp.core.geometry import Rect
from zfp.core.types import Confidence, Evidence, EvidenceKind, FieldCandidate, FieldType
from zfp.fusion.calibration import (
    DEFAULT_PADDING,
    MAX_SIZE,
    MIN_SIZE,
    Calibrator,
    EdgeAdjustment,
    FieldPadding,
    _f1,
    _truth_by_page,
    calibrate,
    calibrate_weights,
    padding_for,
    size_bounds,
)

# ======================================================================================
# Padding
# ======================================================================================


class FieldPaddingTests(unittest.TestCase):
    def test_padding_grows_each_edge_outward(self):
        padded = FieldPadding(left=1.0, right=2.0, top=3.0, bottom=4.0).applied(
            Rect(100.0, 200.0, 300.0, 214.0)
        )
        self.assertEqual(padded.as_list(), [99.0, 196.0, 302.0, 217.0])

    def test_negative_padding_never_inverts(self):
        start = Rect(100.0, 200.0, 104.0, 204.0)
        padded = FieldPadding(left=-10.0, right=-10.0).applied(start)
        self.assertEqual(padded.as_list(), start.as_list())

    def test_round_trip(self):
        padding = FieldPadding(left=1.0, right=0.0, top=2.5, bottom=1.0)
        self.assertEqual(FieldPadding.from_dict(padding.as_dict()), padding)

    def test_every_field_type_has_padding_and_bounds(self):
        for field_type in FieldType:
            self.assertIn(field_type, DEFAULT_PADDING, field_type.value)
            self.assertIn(field_type, MIN_SIZE, field_type.value)
            self.assertIn(field_type, MAX_SIZE, field_type.value)

    def test_documented_defaults_are_the_identity(self):
        # Every detector already reports the writable area under its archetype's
        # convention -- the inside of a stroked box, the band above a rule, the glyph
        # box of a check -- so a constant unlearned offset can only move the widget off
        # it.  Real per-edge corrections are fitted by Calibrator, never hard-coded.
        for field_type in FieldType:
            self.assertEqual(
                padding_for(field_type).as_tuple(), (0.0, 0.0, 0.0, 0.0), field_type.value
            )

    def test_padding_is_still_a_table_a_deployment_can_edit(self):
        saved = DEFAULT_PADDING[FieldType.TEXT]
        DEFAULT_PADDING[FieldType.TEXT] = FieldPadding(left=1.0, right=0.0, top=1.0, bottom=1.0)
        try:
            self.assertEqual(padding_for(FieldType.TEXT).as_tuple(), (1.0, 0.0, 1.0, 1.0))
            out = calibrate(Rect(100.0, 700.0, 300.0, 712.0), FieldType.TEXT, None)
            self.assertEqual(out.as_list(), [99.0, 699.0, 300.0, 713.0])
        finally:
            DEFAULT_PADDING[FieldType.TEXT] = saved

    def test_padding_lookup_tolerates_a_string(self):
        self.assertEqual(padding_for("checkbox").as_tuple(), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(padding_for("nonsense"), padding_for(FieldType.UNKNOWN))


class CalibrateTests(unittest.TestCase):
    def test_clamps_a_seven_hundred_point_tall_field(self):
        out = calibrate(Rect(100.0, 50.0, 300.0, 750.0), FieldType.TEXT, DetectionConfig())
        self.assertLessEqual(out.height, MAX_SIZE[FieldType.TEXT][1] + 1e-9)
        # The top edge is the trustworthy one and is preserved exactly.
        self.assertAlmostEqual(out.y1, 750.0, places=6)
        self.assertAlmostEqual(out.y0, 710.0, places=6)

    def test_clamps_an_absurd_width_about_the_centre(self):
        out = calibrate(Rect(0.0, 100.0, 2000.0, 120.0), FieldType.SIGNATURE, None)
        self.assertLessEqual(out.width, MAX_SIZE[FieldType.SIGNATURE][0] + 1e-9)
        self.assertAlmostEqual(out.center.x, 1000.0, places=6)

    def test_grows_a_degenerate_rect_to_the_minimum(self):
        out = calibrate(Rect(100.0, 100.0, 100.5, 100.5), FieldType.TEXT, None)
        min_w, min_h = MIN_SIZE[FieldType.TEXT]
        self.assertGreaterEqual(out.width + 1e-9, min_w)
        self.assertGreaterEqual(out.height + 1e-9, min_h)

    def test_a_sane_rect_passes_through_untouched(self):
        out = calibrate(Rect(100.0, 700.0, 300.0, 712.0), FieldType.TEXT, None)
        self.assertEqual(out.as_list(), [100.0, 700.0, 300.0, 712.0])

    def test_checkbox_bounds_follow_the_detection_config(self):
        generous = DetectionConfig(checkbox_max_pt=40.0)
        _lo, hi = size_bounds(FieldType.CHECKBOX, generous)
        self.assertEqual(hi, (40.0, 40.0))
        out = calibrate(Rect(0.0, 0.0, 36.0, 36.0), FieldType.CHECKBOX, generous)
        self.assertEqual(out.as_list(), [0.0, 0.0, 36.0, 36.0])

    def test_accepts_a_zfp_config(self):
        out = calibrate(Rect(100.0, 700.0, 300.0, 712.0), FieldType.TEXT, ZfpConfig.default())
        self.assertEqual(out.as_list(), [100.0, 700.0, 300.0, 712.0])

    def test_result_is_rounded(self):
        out = calibrate(Rect(1.00048, 2.00048, 90.00048, 20.00048), FieldType.CHECKBOX, None)
        for value in out.as_list():
            self.assertEqual(value, round(value, 3))


# ======================================================================================
# Learned per-edge calibration
# ======================================================================================


class CalibratorTests(unittest.TestCase):
    def _shifted_corpus(self, dx=-2.0, dy=0.0, count=5):
        truth = [
            Rect(100.0 + 10 * i, 700.0 - 20 * i, 300.0 + 10 * i, 714.0 - 20 * i)
            for i in range(count)
        ]
        pred = [r.translated(dx, dy) for r in truth]
        return pred, truth

    def test_fit_recovers_a_systematic_two_point_left_shift(self):
        pred, truth = self._shifted_corpus(dx=-2.0)
        calibrator = Calibrator.fit(pred, truth)
        for predicted, expected in zip(pred, truth):
            recovered = calibrator.apply(predicted, FieldType.UNKNOWN)
            self.assertEqual(recovered.rounded(6).as_list(), expected.rounded(6).as_list())

    def test_the_learned_adjustment_is_the_mean_signed_error(self):
        pred, truth = self._shifted_corpus(dx=-2.0, dy=1.5)
        adjustment = Calibrator.fit(pred, truth).adjustment_for(FieldType.UNKNOWN)
        self.assertAlmostEqual(adjustment.x0, -2.0, places=6)
        self.assertAlmostEqual(adjustment.x1, -2.0, places=6)
        self.assertAlmostEqual(adjustment.y0, 1.5, places=6)
        self.assertAlmostEqual(adjustment.y1, 1.5, places=6)
        self.assertEqual(adjustment.count, 5)

    def test_adjustments_are_learned_per_field_type(self):
        pred = [
            (FieldType.TEXT, Rect(98.0, 700.0, 298.0, 714.0)),
            (FieldType.TEXT, Rect(98.0, 600.0, 298.0, 614.0)),
            (FieldType.CHECKBOX, Rect(100.0, 500.0, 114.0, 514.0)),
        ]
        truth = [
            (FieldType.TEXT, Rect(100.0, 700.0, 300.0, 714.0)),
            (FieldType.TEXT, Rect(100.0, 600.0, 300.0, 614.0)),
            (FieldType.CHECKBOX, Rect(100.0, 500.0, 112.0, 512.0)),
        ]
        calibrator = Calibrator.fit(pred, truth)
        self.assertAlmostEqual(calibrator.adjustment_for(FieldType.TEXT).x0, -2.0, places=6)
        self.assertAlmostEqual(calibrator.adjustment_for(FieldType.CHECKBOX).x0, 0.0, places=6)
        self.assertAlmostEqual(calibrator.adjustment_for(FieldType.CHECKBOX).x1, 2.0, places=6)
        recovered = calibrator.apply(Rect(100.0, 500.0, 114.0, 514.0), FieldType.CHECKBOX)
        self.assertEqual(recovered.as_list(), [100.0, 500.0, 112.0, 512.0])

    def test_unseen_field_type_falls_back_to_the_pooled_adjustment(self):
        pred, truth = self._shifted_corpus(dx=-2.0)
        typed_pred = [(FieldType.TEXT, r) for r in pred]
        typed_truth = [(FieldType.TEXT, r) for r in truth]
        calibrator = Calibrator.fit(typed_pred, typed_truth)
        self.assertAlmostEqual(calibrator.adjustment_for(FieldType.SIGNATURE).x0, -2.0, places=6)

    def test_fit_accepts_field_candidates(self):
        def cand(rect, field_type):
            return FieldCandidate(id="x", page=0, rect=rect, field_type=field_type)

        calibrator = Calibrator.fit(
            [cand(Rect(98.0, 0.0, 298.0, 14.0), FieldType.TEXT)],
            [cand(Rect(100.0, 0.0, 300.0, 14.0), FieldType.TEXT)],
        )
        self.assertAlmostEqual(calibrator.adjustment_for(FieldType.TEXT).x0, -2.0, places=6)

    def test_empty_fit_is_a_no_op(self):
        calibrator = Calibrator.fit([], [])
        self.assertTrue(calibrator.is_empty())
        start = Rect(1.0, 2.0, 3.0, 4.0)
        self.assertIs(calibrator.apply(start, FieldType.TEXT), start)

    def test_default_calibrator_is_a_no_op(self):
        start = Rect(1.0, 2.0, 3.0, 4.0)
        self.assertIs(Calibrator().apply(start, FieldType.TEXT), start)

    def test_a_perfect_corpus_learns_nothing(self):
        _pred, truth = self._shifted_corpus(dx=0.0)
        calibrator = Calibrator.fit(list(truth), list(truth))
        self.assertTrue(calibrator.is_empty())

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValidationError):
            Calibrator.fit([Rect(0, 0, 1, 1)], [])

    def test_an_inverting_adjustment_is_refused(self):
        calibrator = Calibrator(
            adjustments={FieldType.TEXT: EdgeAdjustment(x0=-50.0, x1=50.0, count=1)},
            overall=EdgeAdjustment(x0=-50.0, x1=50.0, count=1),
        )
        start = Rect(100.0, 0.0, 140.0, 14.0)
        self.assertIs(calibrator.apply(start, FieldType.TEXT), start)

    def test_round_trip_through_a_dict(self):
        pred, truth = self._shifted_corpus(dx=-2.0, dy=0.5)
        typed_pred = [(FieldType.TEXT, r) for r in pred]
        typed_truth = [(FieldType.TEXT, r) for r in truth]
        calibrator = Calibrator.fit(typed_pred, typed_truth)
        restored = Calibrator.from_dict(calibrator.as_dict())
        self.assertEqual(restored.as_dict(), calibrator.as_dict())
        self.assertEqual(
            restored.apply(pred[0], FieldType.TEXT).as_list(),
            calibrator.apply(pred[0], FieldType.TEXT).as_list(),
        )

    def test_fit_is_deterministic(self):
        pred, truth = self._shifted_corpus(dx=-1.25, dy=0.75)
        first = Calibrator.fit(pred, truth).as_dict()
        second = Calibrator.fit(pred, truth).as_dict()
        self.assertEqual(first, second)


# ======================================================================================
# Weight calibration
# ======================================================================================


def _cand(cid, rect, buckets):
    """A candidate whose evidence collapses to the requested bucket scores."""
    kinds = {
        "geometric_evidence": EvidenceKind.VECTOR_LINE,
        "blank_region_evidence": EvidenceKind.BLANK_REGION,
        "nearby_label_evidence": EvidenceKind.LABEL_LINK,
        "layout_consistency": EvidenceKind.LAYOUT,
        "repeated_pattern_evidence": EvidenceKind.REPEAT,
        "semantic_type_confidence": EvidenceKind.PATTERN,
        "model_consensus": EvidenceKind.MODEL,
    }
    cand = FieldCandidate(
        id=cid, page=0, rect=rect, field_type=FieldType.TEXT, confidence=Confidence(geometry=0.5)
    )
    for name, score in sorted(buckets.items()):
        cand.add_evidence(Evidence(kind=kinds[name], score=score, detail=name))
    return cand


class CalibrateWeightsTests(unittest.TestCase):
    """A tiny corpus the contract defaults get wrong, and a step that fixes it."""

    def setUp(self):
        self.truth_rect = Rect(100.0, 700.0, 300.0, 714.0)
        self.true_field = _cand(
            "true",
            self.truth_rect,
            {"blank_region_evidence": 1.0, "nearby_label_evidence": 0.9},
        )
        self.false_field = _cand(
            "false",
            Rect(100.0, 100.0, 300.0, 114.0),
            {"geometric_evidence": 1.0, "layout_consistency": 1.0},
        )
        self.candidates = [self.true_field, self.false_field]
        self.truth = [self.truth_rect]

    def _f1_for(self, weights):
        scored = [(c, c.evidence_scores()) for c in self.candidates]
        return _f1(scored, _truth_by_page(self.truth), weights, 0.35, 0.5)

    def test_the_defaults_miss_the_true_field(self):
        self.assertAlmostEqual(self._f1_for(ScoringWeights().normalized()), 0.0, places=9)

    def test_calibration_improves_f1(self):
        base = ScoringWeights()
        tuned = calibrate_weights(self.candidates, self.truth, base)
        self.assertGreater(self._f1_for(tuned), self._f1_for(base.normalized()))

    def test_result_is_normalized(self):
        tuned = calibrate_weights(self.candidates, self.truth, ScoringWeights())
        self.assertAlmostEqual(sum(tuned.as_tuple()), 1.0, places=9)
        for name in SCORING_WEIGHT_NAMES:
            self.assertGreaterEqual(getattr(tuned, name), 0.0)

    def test_never_degrades_f1(self):
        # A corpus no single step can improve: the weights must be held, not wrecked.
        easy = [
            _cand(
                "t",
                self.truth_rect,
                {"geometric_evidence": 1.0, "blank_region_evidence": 1.0},
            )
        ]
        base = ScoringWeights()
        tuned = calibrate_weights(easy, self.truth, base)
        scored = [(c, c.evidence_scores()) for c in easy]
        before = _f1(scored, _truth_by_page(self.truth), base.normalized(), 0.35, 0.5)
        after = _f1(scored, _truth_by_page(self.truth), tuned, 0.35, 0.5)
        self.assertGreaterEqual(after, before)

    def test_is_deterministic(self):
        first = calibrate_weights(self.candidates, self.truth, ScoringWeights()).as_tuple()
        second = calibrate_weights(self.candidates, self.truth, ScoringWeights()).as_tuple()
        self.assertEqual(first, second)

    def test_empty_inputs_return_the_normalized_base(self):
        base = ScoringWeights(geometric_evidence=3.0, blank_region_evidence=1.0)
        self.assertEqual(
            calibrate_weights([], self.truth, base).as_tuple(), base.normalized().as_tuple()
        )
        self.assertEqual(
            calibrate_weights(self.candidates, [], base).as_tuple(),
            base.normalized().as_tuple(),
        )

    def test_zero_passes_is_a_no_op(self):
        tuned = calibrate_weights(self.candidates, self.truth, ScoringWeights(), passes=0)
        self.assertEqual(tuned.as_tuple(), ScoringWeights().normalized().as_tuple())

    def test_truth_accepts_pages_candidates_and_rects(self):
        grouped = _truth_by_page(
            [
                self.truth_rect,
                (1, Rect(0.0, 0.0, 10.0, 10.0)),
                FieldCandidate(id="t", page=2, rect=Rect(1.0, 1.0, 2.0, 2.0)),
            ]
        )
        self.assertEqual(sorted(grouped), [0, 1, 2])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
