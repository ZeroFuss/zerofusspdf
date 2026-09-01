"""Unit tests for :mod:`zfp.vision.primitives`.

Every fixture is a hand-built list of :class:`VectorPrimitive` objects: a rule at a
known y, a box with known corners, nine comb cells, a 3x4 table lattice.  The point of
this module is exactness, so the assertions are on exact rectangles, not on counts
alone.
"""

from __future__ import annotations

import random
import unittest

from zfp.core.config import DetectionConfig, ZfpConfig
from zfp.core.geometry import PageGeometry, Point, Rect
from zfp.core.types import TextSpan, VectorPrimitive
from zfp.vision.primitives import (
    GLYPH_CHECKBOXES,
    detect_boxes,
    detect_checkbox_glyphs,
    detect_circles,
    detect_comb_cells,
    detect_table_cells,
    horizontal_rules,
    merge_collinear,
    normalize_primitives,
    rule_spans,
    vertical_rules,
)

# ======================================================================================
# Fixture helpers
# ======================================================================================


def line(x0, y0, x1, y1, width=0.0, page=0):
    """A stroked line primitive with real endpoints, the way the interpreter emits one."""
    return VectorPrimitive(
        kind="line",
        rect=Rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
        page=page,
        stroke_width=width,
        stroked=True,
        points=[Point(x0, y0), Point(x1, y1)],
    )


def rect_prim(x0, y0, x1, y1, filled=False, kind="rect", page=0):
    """A painted rectangle primitive."""
    return VectorPrimitive(
        kind=kind, rect=Rect(x0, y0, x1, y1), page=page, filled=filled, stroked=not filled
    )


def box_rules(x0, y0, x1, y1):
    """The four rules a box is drawn with."""
    return [
        line(x0, y0, x1, y0),
        line(x0, y1, x1, y1),
        line(x0, y0, x0, y1),
        line(x1, y0, x1, y1),
    ]


def rects(items):
    """Normalize a list of rectangles to comparable tuples."""
    return [tuple(round(v, 3) for v in r.as_list()) for r in items]


# ======================================================================================
# Normalization
# ======================================================================================


class NormalizePrimitivesTests(unittest.TestCase):
    def test_drops_degenerate_points(self):
        prims = [line(100, 500, 100, 500), line(100, 500, 300, 500)]
        out = normalize_primitives(prims, None)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].rect.as_list(), [100.0, 500.0, 300.0, 500.0])

    def test_drops_unpainted(self):
        ghost = VectorPrimitive(
            kind="rect", rect=Rect(10, 10, 100, 100), page=0, filled=False, stroked=False
        )
        self.assertEqual(normalize_primitives([ghost], None), [])

    def test_rounds_to_three_decimals(self):
        prim = line(100.00049, 500.12345678, 300.9, 500.12345678)
        out = normalize_primitives([prim], None)[0]
        self.assertEqual(out.rect.as_list(), [100.0, 500.123, 300.9, 500.123])

    def test_clamps_to_page(self):
        geometry = PageGeometry(0, Rect(0, 0, 612, 792), Rect(0, 0, 612, 792), 0)
        out = normalize_primitives([line(-40, 500, 900, 500)], None, geometry)
        self.assertEqual(out[0].rect.as_list(), [0.0, 500.0, 612.0, 500.0])

    def test_thin_rect_becomes_a_rule(self):
        out = normalize_primitives([rect_prim(100, 499, 300, 500, filled=True)], None)
        self.assertEqual(out[0].kind, "line")
        self.assertEqual(len(out[0].points), 2)
        self.assertAlmostEqual(out[0].points[0].y, 499.5)
        self.assertAlmostEqual(out[0].points[1].y, 499.5)

    def test_fat_rect_stays_a_rect(self):
        out = normalize_primitives([rect_prim(100, 400, 300, 500, filled=True)], None)
        self.assertEqual(out[0].kind, "rect")

    def test_does_not_mutate_input(self):
        prim = rect_prim(100, 499, 300, 500, filled=True)
        normalize_primitives([prim], None)
        self.assertEqual(prim.kind, "rect")
        self.assertEqual(prim.points, [])

    def test_accepts_every_config_shape(self):
        prims = [line(100, 500, 300, 500)]
        for config in (None, DetectionConfig(), ZfpConfig.default()):
            self.assertEqual(len(normalize_primitives(prims, config)), 1)


# ======================================================================================
# Rules
# ======================================================================================


class RuleSelectionTests(unittest.TestCase):
    def test_horizontal_rule_is_found(self):
        rule = line(100, 500, 300, 500)
        self.assertEqual(horizontal_rules([rule], None), [rule])
        self.assertEqual(vertical_rules([rule], None), [])

    def test_vertical_rule_is_found(self):
        rule = line(100, 400, 100, 600)
        self.assertEqual(vertical_rules([rule], None), [rule])
        self.assertEqual(horizontal_rules([rule], None), [])

    def test_short_line_is_rejected(self):
        self.assertEqual(horizontal_rules([line(100, 500, 110, 500)], None), [])

    def test_explicit_line_qualifies_at_the_relaxed_length(self):
        # 15pt is under min_line_length_pt (24) but over 0.6 * 24 = 14.4.
        self.assertEqual(len(horizontal_rules([line(100, 500, 115, 500)], None)), 1)
        # The same extent as a plain rectangle does not qualify.
        self.assertEqual(len(horizontal_rules([rect_prim(100, 499, 115, 500)], None)), 0)

    def test_thick_bar_is_not_a_rule(self):
        self.assertEqual(horizontal_rules([rect_prim(100, 495, 300, 500)], None), [])

    def test_thick_pen_is_not_a_rule(self):
        self.assertEqual(horizontal_rules([line(100, 500, 300, 500, width=8.0)], None), [])

    def test_dashes_qualify_as_a_group(self):
        dashes = [line(100 + i * 10, 500, 108 + i * 10, 500) for i in range(10)]
        self.assertEqual(len(horizontal_rules(dashes, None)), 10)

    def test_a_lonely_dash_does_not_qualify(self):
        self.assertEqual(horizontal_rules([line(100, 500, 108, 500)], None), [])
        pair = [line(100, 500, 108, 500), line(110, 500, 118, 500)]
        self.assertEqual(horizontal_rules(pair, None), [])

    def test_circle_is_never_a_rule(self):
        circle = VectorPrimitive(kind="circle", rect=Rect(100, 500, 300, 501), page=0)
        self.assertEqual(horizontal_rules([circle], None), [])


class MergeCollinearTests(unittest.TestCase):
    def test_dashed_rule_becomes_one(self):
        dashes = [line(100 + i * 10, 500, 108 + i * 10, 500) for i in range(10)]
        merged = merge_collinear(horizontal_rules(dashes, None), None)
        self.assertEqual(rects([p.rect for p in merged]), [(100.0, 500.0, 198.0, 500.0)])

    def test_near_collinear_rules_merge(self):
        # 1.0pt apart, inside line_merge_tolerance_pt (1.5).
        parts = [line(100, 500, 200, 500), line(201, 501, 300, 501)]
        merged = merge_collinear(parts, None)
        self.assertEqual(rects([p.rect for p in merged]), [(100.0, 500.0, 300.0, 501.0)])

    def test_distant_rules_stay_apart(self):
        parts = [line(100, 500, 200, 500), line(100, 480, 200, 480)]
        merged = merge_collinear(parts, None)
        self.assertEqual(
            rects([p.rect for p in merged]),
            [(100.0, 500.0, 200.0, 500.0), (100.0, 480.0, 200.0, 480.0)],
        )

    def test_wide_gap_is_not_merged(self):
        parts = [line(100, 500, 200, 500), line(210, 500, 300, 500)]
        merged = merge_collinear(parts, None)
        self.assertEqual(len(merged), 2)

    def test_horizontal_and_vertical_are_independent(self):
        merged = merge_collinear([line(100, 500, 300, 500), line(100, 400, 100, 600)], None)
        self.assertEqual(len(merged), 2)
        self.assertEqual(rects([merged[0].rect]), [(100.0, 500.0, 300.0, 500.0)])
        self.assertEqual(rects([merged[1].rect]), [(100.0, 400.0, 100.0, 600.0)])

    def test_merged_rule_keeps_endpoints(self):
        merged = merge_collinear([line(100, 500, 300, 500)], None)[0]
        self.assertEqual([p.as_tuple() for p in merged.points], [(100.0, 500.0), (300.0, 500.0)])


# ======================================================================================
# Boxes, circles, checkboxes
# ======================================================================================


class BoxTests(unittest.TestCase):
    def test_painted_rectangle_is_a_box(self):
        self.assertEqual(
            rects(detect_boxes([rect_prim(50, 50, 150, 100)], None)), [(50.0, 50.0, 150.0, 100.0)]
        )

    def test_tiny_rectangle_is_not_a_box(self):
        self.assertEqual(detect_boxes([rect_prim(50, 50, 53, 53)], None), [])

    def test_four_rules_make_a_box(self):
        found = detect_boxes(box_rules(100, 500, 300, 560), None)
        self.assertEqual(rects(found), [(100.0, 500.0, 300.0, 560.0)])

    def test_crossing_rules_do_not_make_a_box(self):
        # A plus sign: the rules cross but no endpoints meet.
        prims = [line(100, 500, 300, 500), line(200, 400, 200, 600)]
        self.assertEqual(detect_boxes(prims, None), [])

    def test_duplicate_box_is_reported_once(self):
        prims = box_rules(100, 500, 300, 560) + [rect_prim(100, 500, 300, 560)]
        self.assertEqual(rects(detect_boxes(prims, None)), [(100.0, 500.0, 300.0, 560.0)])

    def test_thick_border_still_closes(self):
        prims = [
            rect_prim(100, 500, 300, 502, filled=True),
            rect_prim(100, 558, 300, 560, filled=True),
            rect_prim(100, 500, 102, 560, filled=True),
            rect_prim(298, 500, 300, 560, filled=True),
        ]
        found = detect_boxes(normalize_primitives(prims, None), None)
        self.assertEqual(rects(found), [(100.0, 500.0, 300.0, 560.0)])


class CircleTests(unittest.TestCase):
    def test_circle_primitive(self):
        circle = VectorPrimitive(kind="circle", rect=Rect(100, 500, 112, 512), page=0)
        self.assertEqual(rects(detect_circles([circle], None)), [(100.0, 500.0, 112.0, 512.0)])

    def test_bezier_path_is_a_circle(self):
        curve = VectorPrimitive(
            kind="path",
            rect=Rect(100, 500, 112, 512),
            page=0,
            points=[
                Point(106, 500),
                Point(112, 506),
                Point(106, 512),
                Point(100, 506),
                Point(103, 502),
            ],
        )
        self.assertEqual(rects(detect_circles([curve], None)), [(100.0, 500.0, 112.0, 512.0)])

    def test_rectangular_path_is_not_a_circle(self):
        square = VectorPrimitive(
            kind="path",
            rect=Rect(100, 500, 112, 512),
            page=0,
            points=[Point(100, 500), Point(112, 500), Point(112, 512), Point(100, 512)],
        )
        self.assertEqual(detect_circles([square], None), [])

    def test_elongated_path_is_not_a_circle(self):
        blob = VectorPrimitive(
            kind="path",
            rect=Rect(100, 500, 200, 512),
            page=0,
            points=[Point(150, 500), Point(200, 506), Point(150, 512), Point(100, 506)],
        )
        self.assertEqual(detect_circles([blob], None), [])


class CheckboxGlyphTests(unittest.TestCase):
    def test_small_square_box(self):
        found = detect_checkbox_glyphs([rect_prim(100, 500, 110, 510)], [], None)
        self.assertEqual(rects(found), [(100.0, 500.0, 110.0, 510.0)])

    def test_large_box_is_not_a_checkbox(self):
        self.assertEqual(detect_checkbox_glyphs([rect_prim(100, 470, 140, 510)], [], None), [])

    def test_oblong_box_is_not_a_checkbox(self):
        self.assertEqual(detect_checkbox_glyphs([rect_prim(100, 500, 120, 508)], [], None), [])

    def test_circle_checkbox(self):
        circle = VectorPrimitive(kind="circle", rect=Rect(100, 500, 110, 510), page=0)
        self.assertEqual(rects(detect_checkbox_glyphs([circle], [], None)), [(100.0, 500.0, 110.0, 510.0)])

    def test_ballot_glyphs(self):
        for text in ("□", "☐", "❑", "○", "◯", "[ ]", "( )"):
            span = TextSpan(text=text, rect=Rect(200, 500, 210, 510), page=0)
            self.assertEqual(
                rects(detect_checkbox_glyphs([], [span], None)),
                [(200.0, 500.0, 210.0, 510.0)],
                text,
            )

    def test_underscores_are_not_a_checkbox(self):
        span = TextSpan(text="____", rect=Rect(200, 500, 240, 510), page=0)
        self.assertEqual(detect_checkbox_glyphs([], [span], None), [])

    def test_prose_is_not_a_checkbox(self):
        span = TextSpan(text="Yes", rect=Rect(200, 500, 210, 510), page=0)
        self.assertEqual(detect_checkbox_glyphs([], [span], None), [])

    def test_zapf_dingbats_box(self):
        span = TextSpan(text="q", rect=Rect(200, 500, 210, 510), page=0, font_name="ZapfDingbats")
        self.assertEqual(len(detect_checkbox_glyphs([], [span], None)), 1)
        plain = TextSpan(text="q", rect=Rect(200, 500, 210, 510), page=0, font_name="Helvetica")
        self.assertEqual(detect_checkbox_glyphs([], [plain], None), [])

    def test_underscore_string_is_not_in_the_glyph_set(self):
        self.assertNotIn("____", GLYPH_CHECKBOXES)
        self.assertIn("□", GLYPH_CHECKBOXES)


# ======================================================================================
# Table cells
# ======================================================================================


class TableCellTests(unittest.TestCase):
    def _lattice(self, xs, ys):
        h = [line(xs[0], y, xs[-1], y) for y in ys]
        v = [line(x, ys[-1], x, ys[0]) for x in xs]
        return (
            merge_collinear(horizontal_rules(h, None), None),
            merge_collinear(vertical_rules(v, None), None),
        )

    def test_three_by_four_grid(self):
        xs = [100, 200, 300, 400, 500]
        ys = [700, 680, 660, 640]
        h, v = self._lattice(xs, ys)
        cells = detect_table_cells(h, v, None)
        self.assertEqual(len(cells), 12)
        self.assertIn((100.0, 680.0, 200.0, 700.0), rects(cells))
        self.assertIn((400.0, 640.0, 500.0, 660.0), rects(cells))

    def test_cells_are_in_reading_order(self):
        xs = [100, 200, 300, 400, 500]
        ys = [700, 680, 660, 640]
        h, v = self._lattice(xs, ys)
        cells = detect_table_cells(h, v, None)
        self.assertEqual(cells[0].as_list(), [100.0, 680.0, 200.0, 700.0])
        self.assertEqual(cells[-1].as_list(), [400.0, 640.0, 500.0, 660.0])

    def test_no_cell_contains_another(self):
        xs = [100, 200, 300, 400, 500]
        ys = [700, 680, 660, 640]
        h, v = self._lattice(xs, ys)
        cells = detect_table_cells(h, v, None)
        for outer in cells:
            for inner in cells:
                if inner is outer:
                    continue
                self.assertFalse(outer.contains_rect(inner) and outer.area > inner.area)

    def test_merged_cell_is_emitted_once_at_full_width(self):
        # Top row has no internal column rules: it is one cell spanning three columns.
        xs = [100, 200, 300, 400]
        ys = [700, 680, 660]
        h = [line(xs[0], y, xs[-1], y) for y in ys]
        v = [line(100, 660, 100, 700), line(400, 660, 400, 700)]
        v += [line(200, 660, 200, 680), line(300, 660, 300, 680)]
        cells = detect_table_cells(
            merge_collinear(horizontal_rules(h, None), None),
            merge_collinear(vertical_rules(v, None), None),
            None,
        )
        found = rects(cells)
        self.assertEqual(len(found), 4)
        self.assertIn((100.0, 680.0, 400.0, 700.0), found)
        self.assertIn((100.0, 660.0, 200.0, 680.0), found)
        self.assertIn((200.0, 660.0, 300.0, 680.0), found)
        self.assertIn((300.0, 660.0, 400.0, 680.0), found)

    def test_needs_two_rules_per_axis(self):
        self.assertEqual(detect_table_cells([line(100, 700, 500, 700)], [], None), [])

    def test_open_lattice_yields_nothing(self):
        # Two horizontals, one vertical: nothing is bounded on four sides.
        h = [line(100, 700, 500, 700), line(100, 680, 500, 680)]
        v = [line(100, 680, 100, 700)]
        self.assertEqual(detect_table_cells(h, v, None), [])

    def test_oversized_input_is_capped_not_fatal(self):
        h = [line(100, 700 - i * 0.9, 500, 700 - i * 0.9) for i in range(260)]
        v = [line(100 + i * 1.6, 400, 100 + i * 1.6, 700) for i in range(260)]
        cells = detect_table_cells(h, v, None)
        self.assertIsInstance(cells, list)


# ======================================================================================
# Comb cells
# ======================================================================================


class CombCellTests(unittest.TestCase):
    def _comb(self, count, x0=100.0, y0=500.0, width=16.0, gap=4.0, height=16.0):
        prims = []
        for index in range(count):
            x = x0 + index * (width + gap)
            prims.extend(box_rules(x, y0, x + width, y0 + height))
        return prims

    def test_nine_cells_are_one_run(self):
        boxes = detect_boxes(self._comb(9), None)
        self.assertEqual(len(boxes), 9)
        runs = detect_comb_cells(boxes, None)
        self.assertEqual([len(run) for run in runs], [9])
        self.assertEqual(runs[0][0].as_list(), [100.0, 500.0, 116.0, 516.0])
        self.assertEqual(runs[0][-1].as_list(), [260.0, 500.0, 276.0, 516.0])

    def test_two_cells_are_not_a_run(self):
        boxes = detect_boxes(self._comb(2), None)
        self.assertEqual(detect_comb_cells(boxes, None), [])

    def test_two_rows_give_two_runs(self):
        boxes = detect_boxes(self._comb(4) + self._comb(3, y0=450.0), None)
        runs = detect_comb_cells(boxes, None)
        self.assertEqual([len(run) for run in runs], [4, 3])
        self.assertGreater(runs[0][0].y0, runs[1][0].y0)

    def test_uneven_gap_breaks_the_run(self):
        prims = self._comb(3)
        prims.extend(box_rules(300, 500, 316, 516))  # far away on the same row
        boxes = detect_boxes(prims, None)
        runs = detect_comb_cells(boxes, None)
        self.assertEqual([len(run) for run in runs], [3])

    def test_uneven_width_breaks_the_run(self):
        prims = self._comb(3)
        prims.extend(box_rules(160, 500, 200, 516))
        runs = detect_comb_cells(detect_boxes(prims, None), None)
        self.assertEqual([len(run) for run in runs], [3])

    def test_accepts_bare_rectangles(self):
        boxes = [Rect(100 + i * 20, 500, 116 + i * 20, 516) for i in range(5)]
        self.assertEqual([len(run) for run in detect_comb_cells(boxes, None)], [5])


# ======================================================================================
# Text sitting on a rule
# ======================================================================================


class RuleSpanTests(unittest.TestCase):
    def test_text_on_the_rule_is_reported(self):
        rules = [line(100, 500, 300, 500), line(100, 400, 300, 400)]
        spans = [
            TextSpan(text="Employment", rect=Rect(100, 501, 180, 511), page=0),
            TextSpan(text="History", rect=Rect(185, 501, 240, 511), page=0),
            TextSpan(text="far above", rect=Rect(100, 450, 180, 460), page=0),
        ]
        found = rule_spans(rules, spans, None)
        self.assertEqual([s.text for s in found[0]], ["Employment", "History"])
        self.assertNotIn(1, found)

    def test_label_beside_the_rule_is_not_on_it(self):
        rules = [line(200, 500, 400, 500)]
        spans = [TextSpan(text="Name:", rect=Rect(100, 501, 190, 511), page=0)]
        self.assertEqual(rule_spans(rules, spans, None), {})

    def test_blank_spans_are_ignored(self):
        rules = [line(100, 500, 300, 500)]
        spans = [TextSpan(text="   ", rect=Rect(100, 501, 180, 511), page=0)]
        self.assertEqual(rule_spans(rules, spans, None), {})

    def test_distant_text_is_ignored(self):
        rules = [line(100, 500, 300, 500)]
        spans = [TextSpan(text="Above", rect=Rect(100, 506, 180, 516), page=0)]
        self.assertEqual(rule_spans(rules, spans, None), {})


# ======================================================================================
# Determinism
# ======================================================================================


class DeterminismTests(unittest.TestCase):
    def _page(self):
        prims = []
        prims.extend(box_rules(100, 500, 300, 560))
        prims.extend(box_rules(320, 500, 336, 516))
        prims.extend(box_rules(340, 500, 356, 516))
        prims.extend(box_rules(360, 500, 376, 516))
        prims.append(line(100, 400, 400, 400))
        prims.append(line(100, 380, 400, 380))
        prims.append(rect_prim(60, 600, 80, 620))
        prims.append(VectorPrimitive(kind="circle", rect=Rect(100, 300, 110, 310), page=0))
        return prims

    def test_same_input_gives_identical_output(self):
        prims = self._page()
        spans = [TextSpan(text="☐", rect=Rect(500, 500, 510, 510), page=0)]
        first = (
            rects(detect_boxes(prims, None)),
            rects(detect_circles(prims, None)),
            rects(detect_checkbox_glyphs(prims, spans, None)),
            [len(run) for run in detect_comb_cells(detect_boxes(prims, None), None)],
        )
        second = (
            rects(detect_boxes(prims, None)),
            rects(detect_circles(prims, None)),
            rects(detect_checkbox_glyphs(prims, spans, None)),
            [len(run) for run in detect_comb_cells(detect_boxes(prims, None), None)],
        )
        self.assertEqual(first, second)

    def test_input_order_does_not_change_the_result(self):
        prims = self._page()
        shuffled = list(prims)
        random.Random(1234).shuffle(shuffled)
        self.assertEqual(rects(detect_boxes(prims, None)), rects(detect_boxes(shuffled, None)))
        self.assertEqual(
            rects([p.rect for p in merge_collinear(horizontal_rules(prims, None), None)]),
            rects([p.rect for p in merge_collinear(horizontal_rules(shuffled, None), None)]),
        )

    def test_empty_input_is_safe(self):
        self.assertEqual(normalize_primitives([], None), [])
        self.assertEqual(horizontal_rules([], None), [])
        self.assertEqual(vertical_rules([], None), [])
        self.assertEqual(merge_collinear([], None), [])
        self.assertEqual(detect_boxes([], None), [])
        self.assertEqual(detect_circles([], None), [])
        self.assertEqual(detect_checkbox_glyphs([], [], None), [])
        self.assertEqual(detect_table_cells([], [], None), [])
        self.assertEqual(detect_comb_cells([], None), [])
        self.assertEqual(rule_spans([], [], None), {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
