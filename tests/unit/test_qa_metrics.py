"""Unit tests for :mod:`zfp.qa.metrics`."""

from __future__ import annotations

import unittest

from zfp.core.geometry import Rect
from zfp.core.types import FieldType
from zfp.qa import metrics as M


class _T:
    def __init__(self, page=0, rect=None, field_type=FieldType.TEXT,
                canonical_key=None, visible_label=None):
        self.page = page
        self.rect = rect or Rect(0, 0, 10, 10)
        self.field_type = field_type
        self.canonical_key = canonical_key
        self.visible_label = visible_label


class RecallPrecisionTests(unittest.TestCase):
    def test_one_tp_one_fp_one_fn(self):
        pred = [_T(rect=Rect(0, 0, 10, 10)), _T(rect=Rect(200, 200, 210, 210))]
        truth = [_T(rect=Rect(0, 0, 10, 10)), _T(rect=Rect(400, 400, 410, 410))]
        result = M.recall_precision(pred, truth)
        self.assertAlmostEqual(result["recall"], 0.5)
        self.assertAlmostEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["f1"], 0.5)


class MatchFieldsTests(unittest.TestCase):
    def test_greedy_optimal_and_deterministic(self):
        pred = [_T(rect=Rect(0, 0, 10, 10)), _T(rect=Rect(5, 5, 15, 15))]
        truth = [_T(rect=Rect(0, 0, 10, 10))]
        m1 = M.match_fields(pred, truth)
        m2 = M.match_fields(pred, truth)
        self.assertEqual(m1, m2)
        self.assertEqual(len(m1), 1)
        self.assertEqual(m1[0][0], 0)  # the exact-match prediction wins, not the looser one


class OcrCerWerTests(unittest.TestCase):
    def test_kitten_sitting_edit_distance_three(self):
        result = M.ocr_cer_wer("sitting", "kitten")
        self.assertEqual(result["char_distance"], 3)
        self.assertAlmostEqual(result["cer"], 3 / 6)


class TypeMacroF1Tests(unittest.TestCase):
    def test_matches_hand_computation(self):
        pred = [_T(rect=Rect(0, 0, 10, 10), field_type=FieldType.TEXT),
               _T(rect=Rect(20, 0, 30, 10), field_type=FieldType.CHECKBOX)]
        truth = [_T(rect=Rect(0, 0, 10, 10), field_type=FieldType.TEXT),
                _T(rect=Rect(20, 0, 30, 10), field_type=FieldType.TEXT)]
        result = M.type_macro_f1(pred, truth)
        # TEXT: 1 tp, 1 fn (the second truth was predicted as CHECKBOX); CHECKBOX: 1 fp
        text_row = result["per_type"][str(FieldType.TEXT)]
        self.assertEqual(text_row["tp"], 1)
        self.assertEqual(text_row["fn"], 1)


class DashboardTests(unittest.TestCase):
    def test_render_text_contains_every_metric_name(self):
        dash = M.evaluate([], [])
        text = dash.render_text()
        for name in ("recall", "precision", "IoU", "macro-F1", "Label association",
                    "Canonical semantics", "exact match", "consistency"):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
