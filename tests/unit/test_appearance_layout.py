"""Unit tests for :mod:`zfp.appearance.layout`."""

from __future__ import annotations

import unittest

from zfp.appearance import layout
from zfp.core.geometry import Rect
from zfp.core.types import FieldSpec, FieldType
from zfp.pdfio import fonts


def _spec(field_type=FieldType.TEXT, rect=None, **kw):
    rect = rect or Rect(400.0, 600.0, 500.0, 614.0)
    return FieldSpec(name="f", field_type=field_type, page=0, rect=rect, **kw)


class TextLayoutTests(unittest.TestCase):
    def test_single_line_centres_using_real_metrics(self):
        # In a rect tall enough that the metric-derived centre fits inside the padded
        # box (no clamping engaged), the baseline must equal the metric formula exactly.
        rect = Rect(0, 0, 100, 30)
        spec = _spec(rect=rect, font_size=10.0)
        result = layout.layout_text("Jane", spec, rect)
        ascent = fonts.font_ascent("Helvetica") * 10.0 / 1000.0
        descent = fonts.font_descent("Helvetica") * 10.0 / 1000.0
        expected = (rect.height - (ascent + descent)) / 2.0 - descent
        self.assertAlmostEqual(result.origins[0][1], expected, delta=0.5)

    def test_baseline_stays_within_the_padded_box_even_when_tight(self):
        # A short rect clamps the metric centre so the glyph never draws above the top
        # padding or below the bottom padding.
        rect = Rect(0, 0, 100, 14)
        spec = _spec(rect=rect, font_size=10.0)
        result = layout.layout_text("Jane", spec, rect)
        pad = layout.DEFAULT_PADDING
        self.assertGreaterEqual(result.origins[0][1], pad - 1e-6)
        self.assertLessEqual(result.origins[0][1], rect.height - pad + 1e-6)

    def test_auto_size_shrinks_long_value_and_stays_bounded(self):
        rect = Rect(0, 0, 60, 14)
        spec = _spec(rect=rect)
        short = layout.layout_text("Jo", spec, rect)
        long = layout.layout_text("Jane Q. Public of Somewhere", spec, rect)
        self.assertLess(long.font_size, short.font_size)
        self.assertGreaterEqual(long.font_size, 4.0)
        self.assertLessEqual(long.font_size, 12.0)

    def test_fixed_size_value_too_long_is_clipped_not_shrunk(self):
        rect = Rect(0, 0, 40, 14)
        spec = _spec(rect=rect, font_size=12.0)
        result = layout.layout_text("a very very long value indeed", spec, rect)
        self.assertEqual(result.font_size, 12.0)
        self.assertTrue(result.clipped)

    def test_multiline_wraps_and_flags_overflow(self):
        rect = Rect(0, 0, 60, 20)
        spec = _spec(field_type=FieldType.MULTILINE_TEXT, rect=rect, multiline=True,
                     font_size=10.0)
        result = layout.layout_text(
            "one two three four five six seven eight nine ten eleven twelve", spec, rect)
        self.assertGreater(len(result.lines), 1)
        self.assertTrue(result.clipped)

    def test_empty_value_yields_no_lines(self):
        rect = Rect(0, 0, 100, 14)
        result = layout.layout_text("", _spec(rect=rect), rect)
        self.assertEqual(result.lines, [])
        self.assertEqual(result.origins, [])

    def test_alignment_moves_origin_right_for_right_align(self):
        rect = Rect(0, 0, 100, 14)
        left = layout.layout_text("Hi", _spec(rect=rect, font_size=10.0, alignment=0), rect)
        right = layout.layout_text("Hi", _spec(rect=rect, font_size=10.0, alignment=2), rect)
        self.assertLess(left.origins[0][0], right.origins[0][0])

    def test_coordinates_are_xobject_local_not_page_space(self):
        # A field placed far up the page must still produce small local coordinates.
        rect = Rect(400.0, 600.0, 500.0, 614.0)
        result = layout.layout_text("X", _spec(rect=rect, font_size=10.0), rect)
        x, y = result.origins[0]
        self.assertLess(x, rect.width)
        self.assertLess(y, rect.height)


class CombLayoutTests(unittest.TestCase):
    def test_places_one_character_per_cell_at_cell_centres(self):
        rect = Rect(0, 0, 90, 14)
        spec = _spec(field_type=FieldType.COMB, rect=rect, comb_cells=9)
        result = layout.layout_comb("123456789", spec, rect)
        self.assertEqual(result.cells, 9)
        cell_width = rect.width / 9
        for i, (ch, x) in enumerate(zip(result.characters, result.positions)):
            centre = (i + 0.5) * cell_width
            width = fonts.text_width(ch, result.base_font, result.font_size)
            self.assertAlmostEqual(x + width / 2.0, centre, delta=0.05)

    def test_overflow_characters_are_dropped_and_flagged(self):
        rect = Rect(0, 0, 40, 14)
        spec = _spec(field_type=FieldType.COMB, rect=rect, comb_cells=4)
        result = layout.layout_comb("123456", spec, rect)
        self.assertEqual(len(result.characters), 4)
        self.assertTrue(result.clipped)


class MarkPathTests(unittest.TestCase):
    def test_check_mark_paths_are_nonempty_for_every_style(self):
        rect = Rect(0, 0, 12, 12)
        for style in ("check", "cross", "square", "diamond", "circle", "star"):
            path = layout.check_mark_path(rect, style)
            self.assertTrue(path, style)

    def test_circle_path_is_closed(self):
        path = layout.circle_path(5, 5, 4)
        self.assertEqual(path[-1][0], "h")
        self.assertEqual(path[0][0], "m")


if __name__ == "__main__":
    unittest.main()
