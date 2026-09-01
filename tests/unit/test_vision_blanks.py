"""Unit tests for :mod:`zfp.vision.blanks`.

The fixtures are small pages built by hand -- a label and some paper -- so that the
maximal empty rectangles can be written down and asserted exactly.
"""

from __future__ import annotations

import unittest

from zfp.core.geometry import PageGeometry, Rect
from zfp.core.types import TextSpan, VectorPrimitive
from zfp.vision.blanks import (
    MAX_BLANKS,
    blank_regions,
    line_gaps,
    maximal_empty_cells,
    occupancy_grid,
    suppress_redundant,
    whitespace_profile,
)

PAGE = PageGeometry(0, Rect(0, 0, 300, 300), Rect(0, 0, 300, 300), 0)


def span(text, x0, y0, x1, y1, page=0):
    """A native text span."""
    return TextSpan(text=text, rect=Rect(x0, y0, x1, y1), page=page)


def tuples(items):
    """Rectangles as comparable tuples."""
    return [tuple(round(v, 3) for v in r.as_list()) for r in items]


# ======================================================================================
# Occupancy grid
# ======================================================================================


class OccupancyGridTests(unittest.TestCase):
    def test_resolution_is_two_points(self):
        grid = occupancy_grid([], Rect(0, 0, 300, 300))
        self.assertEqual((grid.cols, grid.rows), (150, 150))
        self.assertAlmostEqual(grid.cell_w, 2.0)
        self.assertAlmostEqual(grid.cell_h, 2.0)

    def test_resolution_is_capped(self):
        grid = occupancy_grid([], Rect(0, 0, 4000, 4000))
        self.assertEqual((grid.cols, grid.rows), (400, 400))
        self.assertAlmostEqual(grid.cell_w, 10.0)

    def test_marking_is_conservative(self):
        grid = occupancy_grid([Rect(1.0, 1.0, 3.0, 3.0)], Rect(0, 0, 10, 10))
        # A rectangle straddling cell boundaries occupies every cell it touches.
        self.assertTrue(grid.data[0 * grid.cols + 0])
        self.assertTrue(grid.data[1 * grid.cols + 1])
        self.assertFalse(grid.data[2 * grid.cols + 2])

    def test_cells_round_trip(self):
        grid = occupancy_grid([], Rect(0, 0, 300, 300))
        self.assertEqual(grid.cells_to_rect(0, 0, 0, 0).as_list(), [0.0, 0.0, 2.0, 2.0])
        self.assertEqual(grid.cells_to_rect(5, 10, 5, 10).as_list(), [10.0, 20.0, 12.0, 22.0])

    def test_row_zero_is_the_bottom_of_the_page(self):
        grid = occupancy_grid([Rect(0, 0, 300, 2)], Rect(0, 0, 300, 300))
        self.assertTrue(grid.data[0])
        self.assertFalse(grid.data[(grid.rows - 1) * grid.cols])

    def test_empty_page_is_one_rectangle(self):
        grid = occupancy_grid([], Rect(0, 0, 100, 100))
        cells = maximal_empty_cells(grid, 1, 1)
        self.assertEqual(cells, [(0, 0, grid.cols - 1, grid.rows - 1)])

    def test_a_bar_splits_the_page_in_two(self):
        grid = occupancy_grid([Rect(0, 48, 100, 52)], Rect(0, 0, 100, 100))
        cells = maximal_empty_cells(grid, 1, 1)
        rectangles = sorted(grid.cells_to_rect(*cell).as_list() for cell in cells)
        self.assertEqual(rectangles, [[0.0, 0.0, 100.0, 48.0], [0.0, 52.0, 100.0, 100.0]])

    def test_minimum_size_is_honoured(self):
        grid = occupancy_grid([Rect(0, 48, 100, 52)], Rect(0, 0, 100, 100))
        self.assertEqual(maximal_empty_cells(grid, 60, 1), [])


# ======================================================================================
# Blank regions
# ======================================================================================


class BlankRegionTests(unittest.TestCase):
    def test_blank_beside_a_label(self):
        spans = [span("Name:", 20, 200, 60, 210)]
        found = blank_regions(spans, [], PAGE, None)
        self.assertIn((60.0, 0.0, 300.0, 300.0), tuples(found))

    def test_blank_below_a_label(self):
        spans = [span("Notes", 20, 200, 60, 210)]
        found = blank_regions(spans, [], PAGE, None)
        self.assertIn((0.0, 0.0, 300.0, 200.0), tuples(found))

    def test_region_above_a_label_is_margin(self):
        spans = [span("Footer", 100, 10, 140, 20)]
        found = blank_regions(spans, [], PAGE, None)
        self.assertTrue(found)
        self.assertNotIn((0.0, 20.0, 300.0, 300.0), tuples(found))

    def test_no_text_means_no_fields(self):
        self.assertEqual(blank_regions([], [], PAGE, None), [])

    def test_no_geometry_is_not_fatal(self):
        self.assertEqual(blank_regions([span("Name", 0, 0, 10, 10)], [], None, None), [])

    def test_narrow_gap_is_not_a_blank(self):
        # Two labels 10pt apart: below blank_min_width_pt (40).
        spans = [span("A", 100, 200, 140, 210), span("B", 150, 200, 190, 210)]
        for rect in blank_regions(spans, [], PAGE, None):
            self.assertGreaterEqual(rect.width, 40.0)
            self.assertGreaterEqual(rect.height, 9.0)

    def test_filled_primitive_occupies_its_interior(self):
        spans = [span("Name:", 20, 200, 60, 210)]
        panel = VectorPrimitive(
            kind="rect", rect=Rect(80, 100, 220, 200), page=0, filled=True, stroked=False
        )
        found = blank_regions(spans, [panel], PAGE, None)
        for rect in found:
            self.assertIsNone(rect.intersection(Rect(82, 102, 218, 198)))

    def test_stroked_primitive_occupies_only_its_border(self):
        spans = [span("Name:", 20, 200, 60, 210)]
        frame = VectorPrimitive(
            kind="rect", rect=Rect(60, 100, 300, 300), page=0, filled=False, stroked=True
        )
        found = blank_regions(spans, [frame], PAGE, None)
        inside = [
            r
            for r in found
            if Rect(60, 100, 300, 300).contains_rect(r) and r.area > 20000.0
        ]
        self.assertTrue(inside, "the inside of a drawn frame is still blank")

    def test_images_occupy_the_page(self):
        spans = [span("Photo:", 20, 200, 60, 210)]
        image = Rect(60, 0, 300, 300)
        found = blank_regions(spans, [], PAGE, None, images=[image])
        for rect in found:
            self.assertIsNone(rect.intersection(Rect(62, 2, 298, 298)))

    def test_result_is_sorted_by_area(self):
        spans = [span("Name:", 20, 200, 60, 210), span("Date:", 20, 100, 60, 110)]
        found = blank_regions(spans, [], PAGE, None)
        areas = [r.area for r in found]
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_result_is_capped(self):
        spans = []
        for row in range(20):
            for col in range(6):
                spans.append(span("x", 10 + col * 48, 10 + row * 14, 30 + col * 48, 20 + row * 14))
        found = blank_regions(spans, [], PAGE, None)
        self.assertLessEqual(len(found), MAX_BLANKS)

    def test_deterministic(self):
        spans = [span("Name:", 20, 200, 60, 210), span("Date:", 20, 100, 60, 110)]
        self.assertEqual(
            tuples(blank_regions(spans, [], PAGE, None)),
            tuples(blank_regions(spans, [], PAGE, None)),
        )

    def test_suppression_keeps_the_larger(self):
        big = Rect(0, 0, 100, 100)
        small = Rect(10, 10, 50, 50)
        self.assertEqual(tuples(suppress_redundant([small, big])), [(0.0, 0.0, 100.0, 100.0)])


# ======================================================================================
# Profiles
# ======================================================================================


class WhitespaceProfileTests(unittest.TestCase):
    def test_shape_matches_the_grid(self):
        rows, cols = whitespace_profile([span("A", 0, 0, 10, 10)], PAGE)
        self.assertEqual(len(rows), 150)
        self.assertEqual(len(cols), 150)

    def test_row_zero_is_the_top_of_the_page(self):
        rows, _cols = whitespace_profile([span("Header", 0, 292, 300, 300)], PAGE)
        self.assertGreater(rows[0], 0.9)
        self.assertEqual(rows[-1], 0.0)

    def test_column_zero_is_the_left_edge(self):
        _rows, cols = whitespace_profile([span("Left", 0, 0, 8, 300)], PAGE)
        self.assertGreater(cols[0], 0.9)
        self.assertEqual(cols[-1], 0.0)

    def test_blank_page_profiles_are_zero(self):
        rows, cols = whitespace_profile([], PAGE)
        self.assertEqual(set(rows), {0.0})
        self.assertEqual(set(cols), {0.0})

    def test_no_geometry(self):
        self.assertEqual(whitespace_profile([span("A", 0, 0, 10, 10)], None), ([], []))


class LineGapTests(unittest.TestCase):
    def test_finds_the_oversized_gap(self):
        spans = [
            span("one", 10, 280, 50, 290),
            span("two", 10, 268, 50, 278),
            span("three", 10, 200, 50, 210),
        ]
        found = line_gaps(spans, None)
        self.assertEqual(tuples(found), [(10.0, 210.0, 50.0, 268.0)])

    def test_uniform_leading_has_no_gaps(self):
        spans = [span("l%d" % i, 10, 280 - i * 14, 50, 290 - i * 14) for i in range(6)]
        self.assertEqual(line_gaps(spans, None), [])

    def test_one_line_has_no_gaps(self):
        self.assertEqual(line_gaps([span("only", 10, 280, 50, 290)], None), [])

    def test_empty_input(self):
        self.assertEqual(line_gaps([], None), [])

    def test_spans_on_one_line_are_one_line(self):
        spans = [
            span("left", 10, 280, 50, 290),
            span("right", 60, 280, 100, 290),
            span("below", 10, 200, 50, 210),
        ]
        found = line_gaps(spans, None)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].x0, 10.0)
        self.assertEqual(found[0].x1, 100.0)

    def test_deterministic(self):
        spans = [
            span("one", 10, 280, 50, 290),
            span("two", 10, 268, 50, 278),
            span("three", 10, 200, 50, 210),
        ]
        self.assertEqual(tuples(line_gaps(spans, None)), tuples(line_gaps(spans, None)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
