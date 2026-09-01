"""Unit tests for :mod:`zfp.fusion.geometry_fusion`.

The fixtures are the situations the module exists for: an OCR estimate a few points off
the vector rule that actually drew the line, three detectors describing one field, two
detectors describing overlapping halves of a row, and a blank-region detector that ran
away down the page.
"""

from __future__ import annotations

import random
import unittest

from zfp.core.config import DetectionConfig, ScoringWeights, ZfpConfig
from zfp.core.geometry import PageGeometry, Rect
from zfp.core.types import (
    Confidence,
    Evidence,
    EvidenceKind,
    FieldCandidate,
    FieldType,
    VectorPrimitive,
)
from zfp.fusion.geometry_fusion import (
    MAX_WIDGET_IOU,
    SNAP_EXPANSION_FACTOR,
    TYPE_SPECIFICITY,
    calibrate_rect,
    deduplicate,
    fuse,
    fused_score,
    merge_cluster,
    rank,
    score_candidates,
    snap_to_primitive,
    snap_tolerance,
    suppress_overlaps,
)

# ======================================================================================
# Fixture helpers
# ======================================================================================


def rule(x0: float, y: float, x1: float, thickness: float = 0.8, page: int = 0):
    """A horizontal rule the way the content-stream interpreter emits one."""
    return VectorPrimitive(
        kind="line",
        rect=Rect(x0, y, x1, y + thickness),
        page=page,
        stroke_width=thickness,
        stroked=True,
    )


def vrule(x: float, y0: float, y1: float, thickness: float = 0.8, page: int = 0):
    """A vertical rule."""
    return VectorPrimitive(
        kind="line",
        rect=Rect(x, y0, x + thickness, y1),
        page=page,
        stroke_width=thickness,
        stroked=True,
    )


def box(x0: float, y0: float, x1: float, y1: float, page: int = 0):
    """A stroked rectangle: a boxed field."""
    return VectorPrimitive(
        kind="rect", rect=Rect(x0, y0, x1, y1), page=page, stroke_width=0.7, stroked=True
    )


def candidate(
    cid: str,
    rect: Rect,
    kind: EvidenceKind = EvidenceKind.VECTOR_LINE,
    evidence_score: float = 0.9,
    geometry: float = 0.7,
    field_type: FieldType = FieldType.TEXT,
    page: int = 0,
    label=None,
    label_link: float = 0.0,
    detail: str = "",
    context=None,
):
    """A candidate carrying exactly one piece of evidence."""
    cand = FieldCandidate(
        id=cid,
        page=page,
        rect=rect,
        field_type=field_type,
        visible_label=label,
        parent_context=list(context or []),
        confidence=Confidence(geometry=geometry, label_link=label_link),
    )
    cand.add_evidence(Evidence(kind=kind, score=evidence_score, detail=detail or cid))
    return cand


def max_pair_iou(cands):
    """Largest IoU between any two candidates on the same page."""
    worst = 0.0
    for i, a in enumerate(cands):
        for b in cands[i + 1 :]:
            if a.page != b.page:
                continue
            worst = max(worst, a.rect.iou(b.rect))
    return worst


# ======================================================================================
# Snapping
# ======================================================================================


class SnapToPrimitiveTests(unittest.TestCase):
    """The headline case: OCR says approximately, the vector path says exactly."""

    def setUp(self):
        # The rule the printer actually drew.
        self.rule = rule(598.0, 300.0, 1130.0)

    def test_adopts_vector_x_range_when_ocr_is_three_points_off(self):
        ocr = Rect(601.0, 312.0, 1127.0, 326.0)
        snapped = snap_to_primitive(ocr, [self.rule], 4.0)
        self.assertAlmostEqual(snapped.x0, 598.0, places=6)
        self.assertAlmostEqual(snapped.x1, 1130.0, places=6)

    def test_ignores_a_primitive_thirty_points_away(self):
        ocr = Rect(571.0, 312.0, 1157.0, 326.0)
        snapped = snap_to_primitive(ocr, [self.rule], 4.0)
        self.assertIs(snapped, ocr)
        self.assertEqual(snapped.as_list(), [571.0, 312.0, 1157.0, 326.0])

    def test_per_edge_snapping_does_not_change_the_height(self):
        ocr = Rect(601.0, 312.0, 1127.0, 326.0)
        snapped = snap_to_primitive(ocr, [self.rule], 4.0)
        self.assertAlmostEqual(snapped.height, ocr.height, places=9)
        self.assertAlmostEqual(snapped.y0, 312.0, places=9)
        self.assertAlmostEqual(snapped.y1, 326.0, places=9)

    def test_one_edge_may_snap_while_the_other_stays(self):
        # Left edge is 2pt out, right edge is 25pt out.
        ocr = Rect(600.0, 312.0, 1105.0, 326.0)
        snapped = snap_to_primitive(ocr, [self.rule], 4.0)
        self.assertAlmostEqual(snapped.x0, 598.0, places=6)
        self.assertAlmostEqual(snapped.x1, 1105.0, places=6)

    def test_rule_donates_the_bottom_edge_when_the_field_sits_on_it(self):
        # A field whose bottom is 1pt below the top of the rule adopts the stroke edge.
        ocr = Rect(600.0, 299.9, 1128.0, 314.0)
        snapped = snap_to_primitive(ocr, [self.rule], 4.0)
        self.assertIn(round(snapped.y0, 6), (300.0, 300.8))
        self.assertAlmostEqual(snapped.y1, 314.0, places=9)

    def test_box_donates_all_four_edges(self):
        target = box(100.0, 200.0, 300.0, 216.0)
        snapped = snap_to_primitive(Rect(102.0, 202.0, 298.0, 214.0), [target], 4.0)
        self.assertEqual(snapped.rounded(6).as_list(), [100.0, 200.0, 300.0, 216.0])

    def test_vertical_rule_donates_the_nearer_x_edge_only(self):
        left = vrule(100.0, 200.0, 260.0)
        start = Rect(102.0, 210.0, 300.0, 224.0)
        snapped = snap_to_primitive(start, [left], 4.0)
        # The nearer side of the stroke wins: the rule is 0.8pt thick.
        self.assertAlmostEqual(snapped.x0, 100.8, places=6)
        self.assertAlmostEqual(snapped.x1, 300.0, places=6)

    def test_far_away_primitive_at_the_right_height_does_not_donate(self):
        # Same y as the field, but on the far side of the page: not associated.
        elsewhere = rule(60.0, 312.0, 120.0)
        start = Rect(600.0, 312.0, 700.0, 326.0)
        self.assertIs(snap_to_primitive(start, [elsewhere], 4.0), start)

    def test_nothing_close_enough_returns_the_rect_unchanged(self):
        start = Rect(10.0, 10.0, 50.0, 24.0)
        self.assertIs(snap_to_primitive(start, [rule(400.0, 500.0, 500.0)], 4.0), start)
        self.assertIs(snap_to_primitive(start, [], 4.0), start)
        self.assertIs(snap_to_primitive(start, [rule(10.0, 10.0, 50.0)], 0.0), start)

    def test_closest_primitive_wins_when_several_are_in_range(self):
        near = rule(599.0, 300.0, 1129.0)
        start = Rect(601.0, 312.0, 1127.0, 326.0)
        snapped = snap_to_primitive(start, [self.rule, near], 4.0)
        self.assertAlmostEqual(snapped.x0, 599.0, places=6)
        self.assertAlmostEqual(snapped.x1, 1129.0, places=6)

    def test_never_expands_by_more_than_three_tolerances(self):
        start = Rect(601.0, 312.0, 1127.0, 326.0)
        tol = 4.0
        snapped = snap_to_primitive(start, [self.rule], tol)
        limit = SNAP_EXPANSION_FACTOR * tol
        self.assertGreaterEqual(snapped.x0, start.x0 - limit)
        self.assertGreaterEqual(snapped.y0, start.y0 - limit)
        self.assertLessEqual(snapped.x1, start.x1 + limit)
        self.assertLessEqual(snapped.y1, start.y1 + limit)

    def test_snapping_never_inverts_a_narrow_rect(self):
        # Two vertical rules straddling a 3pt-wide candidate: the axis must survive.
        start = Rect(100.0, 200.0, 103.0, 214.0)
        snapped = snap_to_primitive(start, [vrule(101.0, 195.0, 220.0)], 4.0)
        self.assertGreater(snapped.width, 0.0)
        self.assertGreater(snapped.height, 0.0)

    def test_snap_tolerance_follows_the_config(self):
        self.assertAlmostEqual(snap_tolerance(None), 4.0, places=9)
        self.assertAlmostEqual(
            snap_tolerance(DetectionConfig(line_merge_tolerance_pt=6.0)), 12.0, places=9
        )


# ======================================================================================
# Deduplication and merging
# ======================================================================================


class DeduplicateTests(unittest.TestCase):
    """Three detectors agreeing must produce one field, and a better one."""

    def _trio(self):
        return [
            candidate(
                "vec",
                Rect(100.0, 700.0, 300.0, 714.0),
                EvidenceKind.VECTOR_LINE,
                0.90,
                geometry=0.60,
                field_type=FieldType.TEXT,
                label="Full name",
                context=["Applicant"],
            ),
            candidate(
                "ocr",
                Rect(101.0, 699.0, 299.0, 715.0),
                EvidenceKind.OCR_TEXT,
                0.70,
                geometry=0.50,
                field_type=FieldType.UNKNOWN,
                context=["Section 1"],
            ),
            candidate(
                "blank",
                Rect(99.0, 701.0, 301.0, 713.0),
                EvidenceKind.BLANK_REGION,
                0.60,
                geometry=0.40,
                field_type=FieldType.UNKNOWN,
            ),
        ]

    def test_three_agreeing_detectors_become_one_candidate(self):
        merged = deduplicate(self._trio(), 0.55)
        self.assertEqual(len(merged), 1)

    def test_corroboration_raises_geometry_above_every_input(self):
        inputs = self._trio()
        merged = deduplicate(inputs, 0.55)[0]
        for cand in inputs:
            self.assertGreater(merged.confidence.geometry, cand.confidence.geometry)
        # 1 - (1-.6)(1-.5)(1-.4) = 0.88
        self.assertAlmostEqual(merged.confidence.geometry, 0.88, places=9)

    def test_corroboration_is_capped(self):
        strong = [
            candidate("a", Rect(0, 0, 100, 14), EvidenceKind.VECTOR_LINE, 1.0, geometry=0.999),
            candidate("b", Rect(0, 0, 100, 14), EvidenceKind.OCR_TEXT, 1.0, geometry=0.999),
            candidate("c", Rect(0, 0, 100, 14), EvidenceKind.BLANK_REGION, 1.0, geometry=0.999),
        ]
        merged = deduplicate(strong, 0.55)[0]
        self.assertLessEqual(merged.confidence.geometry, 0.999)

    def test_same_evidence_kind_earns_no_corroboration_bonus(self):
        same = [
            candidate("a", Rect(0, 0, 100, 14), EvidenceKind.VECTOR_LINE, 0.9, geometry=0.60),
            candidate("b", Rect(1, 0, 99, 14), EvidenceKind.VECTOR_LINE, 0.8, geometry=0.50),
        ]
        merged = deduplicate(same, 0.55)[0]
        self.assertAlmostEqual(merged.confidence.geometry, 0.60, places=9)

    def test_survivor_keeps_the_best_geometry_rect(self):
        merged = deduplicate(self._trio(), 0.55)[0]
        self.assertEqual(merged.rect.as_list(), [100.0, 700.0, 300.0, 714.0])
        self.assertEqual(merged.id, "vec")

    def test_sources_evidence_and_context_are_unioned(self):
        merged = deduplicate(self._trio(), 0.55)[0]
        self.assertEqual(
            sorted(merged.sources), ["blank_region", "ocr_text", "vector_line"]
        )
        self.assertEqual(len(merged.evidence), 3)
        self.assertEqual(sorted(merged.parent_context), ["Applicant", "Section 1"])

    def test_identical_evidence_is_deduplicated(self):
        twins = [
            candidate("a", Rect(0, 0, 100, 14), EvidenceKind.VECTOR_LINE, 0.9, detail="rule"),
            candidate("b", Rect(0, 0, 100, 14), EvidenceKind.VECTOR_LINE, 0.9, detail="rule"),
        ]
        merged = deduplicate(twins, 0.55)[0]
        self.assertEqual(len(merged.evidence), 1)

    def test_first_non_none_label_survives(self):
        merged = deduplicate(self._trio(), 0.55)[0]
        self.assertEqual(merged.visible_label, "Full name")

    def test_confidence_is_the_per_axis_maximum(self):
        left = candidate("a", Rect(0, 0, 100, 14), EvidenceKind.VECTOR_LINE, 0.9, geometry=0.6)
        left.confidence.label_link = 0.2
        left.confidence.autofill_value = 0.8
        right = candidate("b", Rect(0, 0, 100, 14), EvidenceKind.OCR_TEXT, 0.7, geometry=0.5)
        right.confidence.label_link = 0.9
        right.confidence.semantic_type = 0.4
        merged = deduplicate([left, right], 0.55)[0]
        self.assertAlmostEqual(merged.confidence.label_link, 0.9, places=9)
        self.assertAlmostEqual(merged.confidence.semantic_type, 0.4, places=9)
        self.assertAlmostEqual(merged.confidence.autofill_value, 0.8, places=9)

    def test_specificity_keeps_checkbox_over_text(self):
        text = candidate(
            "text",
            Rect(100.0, 500.0, 112.0, 512.0),
            EvidenceKind.BLANK_REGION,
            0.6,
            geometry=0.90,
            field_type=FieldType.TEXT,
        )
        check = candidate(
            "check",
            Rect(100.0, 500.0, 112.0, 512.0),
            EvidenceKind.CHECKBOX_GLYPH,
            0.8,
            geometry=0.30,
            field_type=FieldType.CHECKBOX,
        )
        merged = deduplicate([text, check], 0.55)[0]
        self.assertEqual(merged.field_type, FieldType.CHECKBOX)
        # The rect still comes from the better-measured member.
        self.assertEqual(merged.id, "text")

    def test_specificity_ranking_is_ordered_as_documented(self):
        self.assertGreater(TYPE_SPECIFICITY[FieldType.CHECKBOX], TYPE_SPECIFICITY[FieldType.TEXT])
        self.assertGreater(TYPE_SPECIFICITY[FieldType.RADIO], TYPE_SPECIFICITY[FieldType.TEXT])
        self.assertGreater(TYPE_SPECIFICITY[FieldType.COMB], TYPE_SPECIFICITY[FieldType.TEXT])
        self.assertGreater(TYPE_SPECIFICITY[FieldType.TEXT], TYPE_SPECIFICITY[FieldType.UNKNOWN])

    def test_clusters_are_transitive(self):
        chain = [
            candidate("a", Rect(0.0, 0.0, 100.0, 14.0), EvidenceKind.VECTOR_LINE, 0.9),
            candidate("b", Rect(20.0, 0.0, 120.0, 14.0), EvidenceKind.OCR_TEXT, 0.8),
            candidate("c", Rect(40.0, 0.0, 140.0, 14.0), EvidenceKind.BLANK_REGION, 0.7),
        ]
        # a-b and b-c exceed the threshold; a-c does not, but all three are one field.
        self.assertGreater(chain[0].rect.iou(chain[1].rect), 0.55)
        self.assertLess(chain[0].rect.iou(chain[2].rect), 0.55)
        self.assertEqual(len(deduplicate(chain, 0.55)), 1)

    def test_different_pages_never_merge(self):
        pair = [
            candidate("a", Rect(0.0, 0.0, 100.0, 14.0), page=0),
            candidate("b", Rect(0.0, 0.0, 100.0, 14.0), page=1),
        ]
        self.assertEqual(len(deduplicate(pair, 0.55)), 2)

    def test_distant_candidates_are_left_alone(self):
        pair = [
            candidate("a", Rect(0.0, 0.0, 100.0, 14.0)),
            candidate("b", Rect(300.0, 0.0, 400.0, 14.0)),
        ]
        self.assertEqual(len(deduplicate(pair, 0.55)), 2)

    def test_inputs_are_not_mutated(self):
        inputs = self._trio()
        before = [c.as_dict() for c in inputs]
        deduplicate(inputs, 0.55)
        self.assertEqual([c.as_dict() for c in inputs], before)

    def test_empty_and_single_inputs(self):
        self.assertEqual(deduplicate([], 0.55), [])
        one = candidate("only", Rect(0.0, 0.0, 10.0, 10.0))
        self.assertEqual(len(deduplicate([one], 0.55)), 1)

    def test_merge_cluster_rejects_an_empty_cluster(self):
        with self.assertRaises(ValueError):
            merge_cluster([])


# ======================================================================================
# Overlap suppression
# ======================================================================================


class SuppressOverlapsTests(unittest.TestCase):
    """The 10% IoU invariant from docs/QA.md, enforced by construction."""

    def setUp(self):
        self.config = ZfpConfig.default()

    def test_loser_is_shrunk_to_the_free_remainder(self):
        strong = candidate("strong", Rect(100.0, 700.0, 300.0, 714.0), geometry=0.9)
        weak = candidate(
            "weak", Rect(240.0, 700.0, 440.0, 714.0), evidence_score=0.5, geometry=0.5
        )
        out = suppress_overlaps([strong, weak], self.config)
        self.assertEqual(len(out), 2)
        kept = {c.id: c.rect.as_list() for c in out}
        self.assertEqual(kept["strong"], [100.0, 700.0, 300.0, 714.0])
        self.assertEqual(kept["weak"], [300.0, 700.0, 440.0, 714.0])

    def test_loser_is_dropped_when_the_remainder_is_not_viable(self):
        strong = candidate("strong", Rect(100.0, 700.0, 300.0, 714.0), geometry=0.9)
        weak = candidate(
            "weak", Rect(105.0, 701.0, 315.0, 713.0), evidence_score=0.4, geometry=0.3
        )
        out = suppress_overlaps([strong, weak], self.config)
        self.assertEqual([c.id for c in out], ["strong"])

    def test_small_overlaps_are_left_alone(self):
        a = candidate("a", Rect(100.0, 700.0, 300.0, 714.0), geometry=0.9)
        b = candidate("b", Rect(290.0, 700.0, 490.0, 714.0), geometry=0.5)
        self.assertLess(a.rect.iou(b.rect), MAX_WIDGET_IOU)
        out = suppress_overlaps([a, b], self.config)
        self.assertEqual(len(out), 2)
        self.assertEqual([c.rect.as_list() for c in out], [a.rect.as_list(), b.rect.as_list()])

    def test_never_leaves_a_pair_above_ten_percent_iou(self):
        rng = random.Random(20240517)
        cands = []
        for i in range(80):
            x0 = rng.uniform(40.0, 500.0)
            y0 = rng.uniform(40.0, 700.0)
            cands.append(
                candidate(
                    "c%03d" % i,
                    Rect(x0, y0, x0 + rng.uniform(45.0, 180.0), y0 + rng.uniform(10.0, 30.0)),
                    evidence_score=rng.uniform(0.2, 1.0),
                    geometry=rng.uniform(0.2, 1.0),
                )
            )
        out = suppress_overlaps(cands, self.config)
        self.assertTrue(out)
        self.assertLessEqual(max_pair_iou(out), MAX_WIDGET_IOU + 1e-9)

    def test_identical_stack_collapses_to_one(self):
        stack = [
            candidate("a", Rect(100.0, 700.0, 300.0, 714.0), geometry=0.9),
            candidate("b", Rect(100.0, 700.0, 300.0, 714.0), geometry=0.8),
            candidate("c", Rect(100.0, 700.0, 300.0, 714.0), geometry=0.7),
        ]
        out = suppress_overlaps(stack, self.config)
        self.assertEqual(len(out), 1)

    def test_checkboxes_are_judged_by_checkbox_minimums(self):
        big = candidate(
            "big",
            Rect(100.0, 500.0, 140.0, 520.0),
            geometry=0.9,
            field_type=FieldType.CHECKBOX,
        )
        small = candidate(
            "small",
            Rect(130.0, 500.0, 150.0, 520.0),
            evidence_score=0.4,
            geometry=0.4,
            field_type=FieldType.CHECKBOX,
        )
        out = suppress_overlaps([big, small], self.config)
        self.assertEqual(len(out), 2)
        kept = {c.id: c.rect for c in out}
        self.assertAlmostEqual(kept["small"].x0, 140.0, places=6)

    def test_pages_are_independent(self):
        a = candidate("a", Rect(100.0, 700.0, 300.0, 714.0), page=0, geometry=0.9)
        b = candidate("b", Rect(100.0, 700.0, 300.0, 714.0), page=1, geometry=0.5)
        out = suppress_overlaps([a, b], self.config)
        self.assertEqual(len(out), 2)

    def test_output_is_in_reading_order(self):
        low = candidate("low", Rect(100.0, 100.0, 300.0, 114.0))
        high = candidate("high", Rect(100.0, 700.0, 300.0, 714.0))
        out = suppress_overlaps([low, high], self.config)
        self.assertEqual([c.id for c in out], ["high", "low"])


# ======================================================================================
# Calibration entry point
# ======================================================================================


class CalibrateRectTests(unittest.TestCase):
    def test_clamps_a_seven_hundred_point_tall_field(self):
        absurd = Rect(100.0, 50.0, 300.0, 750.0)
        out = calibrate_rect(absurd, FieldType.TEXT, DetectionConfig())
        self.assertLessEqual(out.height, 40.0 + 1e-9)
        # The top edge is the trustworthy one and is preserved exactly.
        self.assertAlmostEqual(out.y1, 750.0, places=6)

    def test_a_sane_text_rect_is_not_moved(self):
        # DEFAULT_PADDING is the identity: the detector already reported the writable
        # area, so calibration only clamps insane sizes.
        start = Rect(100.0, 700.0, 300.0, 712.0)
        out = calibrate_rect(start, FieldType.TEXT, None)
        self.assertEqual(out.as_list(), start.as_list())

    def test_a_checkbox_is_not_padded_at_all(self):
        start = Rect(100.0, 500.0, 112.0, 512.0)
        out = calibrate_rect(start, FieldType.CHECKBOX, None)
        self.assertEqual(out.as_list(), start.as_list())


# ======================================================================================
# Scoring
# ======================================================================================


class ScoringTests(unittest.TestCase):
    def test_fused_score_is_the_weighted_evidence(self):
        cand = candidate("a", Rect(0.0, 0.0, 100.0, 14.0), EvidenceKind.VECTOR_LINE, 1.0)
        cand.add_evidence(Evidence(kind=EvidenceKind.LABEL_LINK, score=1.0, detail="label"))
        weights = ScoringWeights()
        self.assertAlmostEqual(fused_score(cand, weights), 0.30 + 0.15, places=9)

    def test_fused_score_is_reproducible(self):
        cand = candidate("a", Rect(0.0, 0.0, 100.0, 14.0))
        self.assertEqual(fused_score(cand), fused_score(cand))

    def test_score_candidates_writes_into_semantic_type_only(self):
        cand = candidate("a", Rect(0.0, 0.0, 100.0, 14.0), EvidenceKind.VECTOR_LINE, 1.0)
        scored = score_candidates([cand], ScoringWeights())[0]
        self.assertAlmostEqual(scored.confidence.semantic_type, 0.30, places=9)
        self.assertAlmostEqual(scored.confidence.geometry, cand.confidence.geometry, places=9)

    def test_score_candidates_never_lowers_an_existing_value(self):
        cand = candidate("a", Rect(0.0, 0.0, 100.0, 14.0), EvidenceKind.VECTOR_LINE, 0.1)
        cand.confidence.semantic_type = 0.95
        scored = score_candidates([cand], ScoringWeights())[0]
        self.assertAlmostEqual(scored.confidence.semantic_type, 0.95, places=9)

    def test_score_candidates_does_not_mutate_its_input(self):
        cand = candidate("a", Rect(0.0, 0.0, 100.0, 14.0), EvidenceKind.VECTOR_LINE, 1.0)
        score_candidates([cand], ScoringWeights())
        self.assertAlmostEqual(cand.confidence.semantic_type, 0.0, places=9)

    def test_rank_sorts_by_fused_score_then_reading_order(self):
        weak = candidate("weak", Rect(0.0, 700.0, 100.0, 714.0), evidence_score=0.2)
        strong = candidate("strong", Rect(0.0, 100.0, 100.0, 114.0), evidence_score=1.0)
        self.assertEqual([c.id for c in rank([weak, strong])], ["strong", "weak"])

    def test_rank_tiebreak_is_reading_order(self):
        top = candidate("top", Rect(0.0, 700.0, 100.0, 714.0), evidence_score=0.5)
        bottom = candidate("bottom", Rect(0.0, 100.0, 100.0, 114.0), evidence_score=0.5)
        self.assertEqual([c.id for c in rank([bottom, top])], ["top", "bottom"])


# ======================================================================================
# The pipeline
# ======================================================================================


class FuseTests(unittest.TestCase):
    def setUp(self):
        self.config = ZfpConfig.default()
        page = PageGeometry(
            index=0, media_box=Rect(0, 0, 612, 792), crop_box=Rect(0, 0, 612, 792)
        )
        self.geometry = {0: page}

    def _corpus(self):
        return [
            candidate(
                "vec",
                Rect(100.0, 700.0, 300.0, 714.0),
                EvidenceKind.VECTOR_LINE,
                0.95,
                geometry=0.7,
                label="Name",
                label_link=0.8,
            ),
            candidate(
                "ocr",
                Rect(102.0, 699.0, 297.0, 715.0),
                EvidenceKind.OCR_TEXT,
                0.7,
                geometry=0.5,
            ),
            candidate(
                "blank",
                Rect(100.0, 660.0, 400.0, 676.0),
                EvidenceKind.BLANK_REGION,
                0.8,
                geometry=0.6,
                label_link=0.6,
            ),
            candidate(
                "check",
                Rect(100.0, 600.0, 112.0, 612.0),
                EvidenceKind.CHECKBOX_GLYPH,
                0.9,
                geometry=0.8,
                field_type=FieldType.CHECKBOX,
                label_link=0.7,
            ),
            candidate(
                "weak",
                Rect(100.0, 560.0, 300.0, 574.0),
                EvidenceKind.LAYOUT,
                0.05,
                geometry=0.05,
            ),
        ]

    def test_is_deterministic_across_two_runs(self):
        prims = {0: [rule(98.0, 698.0, 302.0), box(100.0, 600.0, 112.0, 612.0)]}
        first = fuse(self._corpus(), self.config, prims, self.geometry)
        second = fuse(self._corpus(), self.config, prims, self.geometry)
        self.assertEqual([c.as_dict() for c in first], [c.as_dict() for c in second])

    def test_snapping_corrects_an_estimated_candidate(self):
        # "blank" is a blank-region guess, so the rule two points off its left edge is
        # the exact geometry it was guessing at, and fusion adopts it.
        prims = {0: [rule(98.0, 659.0, 402.0)]}
        without = fuse(self._corpus(), self.config, None, self.geometry)
        with_prims = fuse(self._corpus(), self.config, prims, self.geometry)
        plain = {c.id: c.rect for c in without}["blank"]
        snapped = {c.id: c.rect for c in with_prims}["blank"]
        self.assertNotEqual(plain.as_list(), snapped.as_list())
        self.assertAlmostEqual(snapped.x0, 98.0, places=6)
        self.assertAlmostEqual(snapped.x1, 402.0, places=6)

    def test_a_measured_candidate_is_never_re_snapped(self):
        # "vec" carries VECTOR_LINE evidence: its rectangle is the rule's own geometry
        # already mapped through the underline convention (a gap above the rule, a
        # field_height_pt band).  Snapping it back onto the raw path coordinates would
        # undo exactly that convention, so fusion leaves it alone.
        prims = {0: [rule(98.0, 698.0, 302.0)]}
        without = fuse(self._corpus(), self.config, None, self.geometry)
        with_prims = fuse(self._corpus(), self.config, prims, self.geometry)
        self.assertEqual(
            {c.id: c.rect for c in without}["vec"].as_list(),
            {c.id: c.rect for c in with_prims}["vec"].as_list(),
        )

    def test_agreeing_detectors_produce_one_field(self):
        out = fuse(self._corpus(), self.config, None, self.geometry)
        ids = [c.id for c in out]
        self.assertIn("vec", ids)
        self.assertNotIn("ocr", ids)

    def test_weak_candidates_are_filtered_out(self):
        out = fuse(self._corpus(), self.config, None, self.geometry)
        self.assertNotIn("weak", [c.id for c in out])
        for cand in out:
            self.assertGreaterEqual(
                cand.confidence.overall() + 1e-9, self.config.detection.min_candidate_confidence
            )

    def test_result_is_sorted_in_reading_order_and_renumbered(self):
        out = fuse(self._corpus(), self.config, None, self.geometry)
        keys = [(c.page, -c.rect.y1, c.rect.x0) for c in out]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual([c.order for c in out], list(range(len(out))))

    def test_output_honours_the_ten_percent_invariant(self):
        out = fuse(self._corpus(), self.config, None, self.geometry)
        self.assertLessEqual(max_pair_iou(out), MAX_WIDGET_IOU + 1e-9)

    def test_rects_are_clipped_to_the_page(self):
        off_page = candidate(
            "off", Rect(-50.0, 700.0, 900.0, 714.0), geometry=0.9, label_link=0.8
        )
        out = fuse([off_page], self.config, None, self.geometry)
        self.assertEqual(len(out), 1)
        crop = self.geometry[0].crop_box
        self.assertGreaterEqual(out[0].rect.x0, crop.x0)
        self.assertLessEqual(out[0].rect.x1, crop.x1)

    def test_works_without_geometry_or_primitives(self):
        out = fuse(self._corpus(), self.config)
        self.assertTrue(out)

    def test_empty_input(self):
        self.assertEqual(fuse([], self.config), [])
        self.assertEqual(fuse(None, self.config), [])

    def test_accepts_a_bare_detection_config(self):
        out = fuse(self._corpus(), DetectionConfig())
        self.assertTrue(out)

    def test_does_not_mutate_its_input(self):
        corpus = self._corpus()
        before = [c.as_dict() for c in corpus]
        fuse(corpus, self.config, {0: [rule(98.0, 698.0, 302.0)]}, self.geometry)
        self.assertEqual([c.as_dict() for c in corpus], before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
