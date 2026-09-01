"""Unit tests for :mod:`zfp.vision.raster_shapes`.

Every fixture is drawn here, pixel by pixel, onto a white canvas: rules, a box, a ring.
The assertions are on the *user-space* rectangles that come back, at scale 2.0, to
within a point -- which is the only thing that matters about a raster detector.

Neither numpy nor OpenCV is installed in the reference environment, so what runs here is
the pure-python path.  When they are installed the OpenCV path runs instead and these
same assertions apply.
"""

from __future__ import annotations

import unittest

from zfp.core.geometry import PageGeometry, Rect
from zfp.vision.raster_shapes import RasterShapes, binarize_ink, detect_shapes_from_image

try:  # the renderer is a sibling module; the detector only duck-types it
    from zfp.raster.render import RenderedPage
except ImportError:  # pragma: no cover - exercised only if render.py is absent
    RenderedPage = None


class StandInPage(object):
    """The four attributes :func:`detect_shapes_from_image` actually reads."""

    def __init__(self, page, width, height, scale, gray, backend="stand-in"):
        self.page = page
        self.width = width
        self.height = height
        self.scale = scale
        self.gray = gray
        self.backend = backend


class Canvas(object):
    """A small mutable gray raster the tests draw their fixtures on."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.data = bytearray(b"\xff" * (width * height))

    def fill(self, x0, y0, x1, y1):
        """Paint a black rectangle in *pixel* space (x1/y1 exclusive)."""
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self.width, x1)
        y1 = min(self.height, y1)
        for y in range(y0, y1):
            self.data[y * self.width + x0 : y * self.width + x1] = b"\x00" * (x1 - x0)

    def outline(self, x0, y0, x1, y1, thickness=1):
        """Paint a hollow rectangle -- a ring, not a blob."""
        self.fill(x0, y0, x1, y0 + thickness)
        self.fill(x0, y1 - thickness, x1, y1)
        self.fill(x0, y0, x0 + thickness, y1)
        self.fill(x1 - thickness, y0, x1, y1)

    def page(self, scale=2.0, index=0):
        """Return a RenderedPage when one is importable, else the stand-in."""
        gray = bytes(self.data)
        if RenderedPage is not None:
            return RenderedPage(
                page=index,
                width=self.width,
                height=self.height,
                scale=scale,
                gray=gray,
                backend="test",
            )
        return StandInPage(index, self.width, self.height, scale, gray)

    def stand_in(self, scale=2.0, index=0):
        """Always return the duck-typed stand-in."""
        return StandInPage(index, self.width, self.height, scale, bytes(self.data))


#: A 100 x 130 point page rendered at scale 2.0 is a 200 x 260 pixel raster.
GEOMETRY = PageGeometry(0, Rect(0, 0, 100, 130), Rect(0, 0, 100, 130), 0)
SCALE = 2.0


def near(test, rect, expected, tolerance=1.0):
    """Assert a rectangle matches ``expected`` within ``tolerance`` points."""
    for got, want, name in zip(rect.as_list(), expected, ("x0", "y0", "x1", "y1")):
        test.assertAlmostEqual(got, want, delta=tolerance, msg="%s of %s" % (name, rect))


def matching(rects, expected, tolerance=1.0):
    """Return the rectangles within ``tolerance`` of ``expected``."""
    out = []
    for rect in rects:
        if all(abs(a - b) <= tolerance for a, b in zip(rect.as_list(), expected)):
            out.append(rect)
    return out


def drawn_page():
    """A canvas with a box, a rule beneath it and a small ring."""
    canvas = Canvas(200, 260)
    canvas.fill(20, 40, 180, 42)  # box top
    canvas.fill(20, 140, 180, 142)  # box bottom
    canvas.fill(20, 40, 22, 142)  # box left
    canvas.fill(178, 40, 180, 142)  # box right
    canvas.fill(20, 170, 180, 172)  # a lone rule
    canvas.outline(40, 200, 53, 213)  # a 13px ring
    return canvas


# ======================================================================================
# Binarization
# ======================================================================================


class BinarizeTests(unittest.TestCase):
    def test_ink_is_one_and_paper_is_zero(self):
        mask, _threshold = binarize_ink(b"\xff\x00\xff\x00")
        self.assertEqual(mask, b"\x00\x01\x00\x01")

    def test_a_flat_page_has_no_ink(self):
        mask, _threshold = binarize_ink(b"\xff" * 64)
        self.assertEqual(mask.count(1), 0)

    def test_empty_input(self):
        self.assertEqual(binarize_ink(b""), (b"", 128))


# ======================================================================================
# Detection
# ======================================================================================


class RasterRuleTests(unittest.TestCase):
    def setUp(self):
        self.shapes = detect_shapes_from_image(drawn_page().page(SCALE), GEOMETRY, None)

    def test_horizontal_rules_land_where_they_were_drawn(self):
        # Rows 170..172 of a 260px raster are user y = 130 - 86 .. 130 - 85.
        self.assertTrue(matching(self.shapes.h_rules, [10.0, 44.0, 90.0, 45.0]))
        self.assertTrue(matching(self.shapes.h_rules, [10.0, 109.0, 90.0, 110.0]))
        self.assertTrue(matching(self.shapes.h_rules, [10.0, 59.0, 90.0, 60.0]))
        self.assertEqual(len(self.shapes.h_rules), 3)

    def test_vertical_rules_land_where_they_were_drawn(self):
        self.assertEqual(len(self.shapes.v_rules), 2)
        self.assertTrue(matching(self.shapes.v_rules, [10.0, 59.0, 11.0, 110.0]))
        self.assertTrue(matching(self.shapes.v_rules, [89.0, 59.0, 90.0, 110.0]))

    def test_the_box_is_reconstructed(self):
        self.assertEqual(len(self.shapes.boxes), 1)
        near(self, self.shapes.boxes[0], [10.0, 59.0, 90.0, 110.0])

    def test_the_ring_is_a_circle(self):
        self.assertEqual(len(self.shapes.circles), 1)
        near(self, self.shapes.circles[0], [20.0, 23.5, 26.5, 30.0])

    def test_the_box_outline_is_not_a_circle(self):
        for rect in self.shapes.circles:
            self.assertLess(rect.width, 22.0)

    def test_nothing_escapes_in_pixels(self):
        for rect in self.shapes.all_rects():
            self.assertGreaterEqual(rect.x0, 0.0)
            self.assertGreaterEqual(rect.y0, 0.0)
            self.assertLessEqual(rect.x1, 100.0)
            self.assertLessEqual(rect.y1, 130.0)

    def test_blank_inside_the_box_is_found(self):
        inside = [r for r in self.shapes.blanks if Rect(10, 59, 90, 110).contains_rect(r)]
        self.assertTrue(inside)

    def test_rules_are_in_reading_order(self):
        tops = [r.y1 for r in self.shapes.h_rules]
        self.assertEqual(tops, sorted(tops, reverse=True))

    def test_primitives_view(self):
        prims = self.shapes.as_primitives(page=3)
        self.assertTrue(prims)
        self.assertEqual({p.page for p in prims}, {3})
        self.assertIn("line", {p.kind for p in prims})
        self.assertIn("rect", {p.kind for p in prims})

    def test_as_dict_is_jsonable(self):
        payload = self.shapes.as_dict()
        self.assertEqual(len(payload["boxes"][0]), 4)
        self.assertIn(payload["backend"], ("pure", "opencv"))


class RasterScaleTests(unittest.TestCase):
    def test_scale_one_gives_the_same_geometry(self):
        canvas = Canvas(100, 130)
        canvas.fill(10, 85, 90, 86)
        geometry = PageGeometry(0, Rect(0, 0, 100, 130), Rect(0, 0, 100, 130), 0)
        shapes = detect_shapes_from_image(canvas.page(1.0), geometry, None)
        self.assertEqual(len(shapes.h_rules), 1)
        near(self, shapes.h_rules[0], [10.0, 44.0, 90.0, 45.0])

    def test_explicit_scale_overrides_the_page(self):
        canvas = Canvas(200, 260)
        canvas.fill(20, 170, 180, 172)
        page = canvas.page(1.0)  # the page claims 1.0 ...
        shapes = detect_shapes_from_image(page, GEOMETRY, None, scale=SCALE)  # ... we say 2.0
        self.assertEqual(len(shapes.h_rules), 1)
        near(self, shapes.h_rules[0], [10.0, 44.0, 90.0, 45.0])

    def test_legacy_positional_scale(self):
        canvas = Canvas(200, 260)
        canvas.fill(20, 170, 180, 172)
        shapes = detect_shapes_from_image(canvas.page(1.0), GEOMETRY, SCALE, None)
        self.assertEqual(len(shapes.h_rules), 1)
        near(self, shapes.h_rules[0], [10.0, 44.0, 90.0, 45.0])

    def test_rotated_page_reclassifies_orientation(self):
        # /Rotate 90: the raster is 130x100 points wide, and a rule drawn horizontally
        # across the raster is a vertical rule in user space.
        geometry = PageGeometry(0, Rect(0, 0, 100, 130), Rect(0, 0, 100, 130), 90)
        canvas = Canvas(260, 200)
        canvas.fill(20, 100, 240, 102)
        shapes = detect_shapes_from_image(canvas.page(SCALE), geometry, None)
        self.assertEqual(len(shapes.h_rules), 0)
        self.assertEqual(len(shapes.v_rules), 1)
        for rect in shapes.all_rects():
            self.assertTrue(geometry.crop_box.contains_rect(rect), rect)


class RasterDegenerateTests(unittest.TestCase):
    def test_no_page(self):
        self.assertTrue(detect_shapes_from_image(None, GEOMETRY, None).is_empty())

    def test_no_geometry(self):
        self.assertTrue(detect_shapes_from_image(Canvas(20, 20).page(), None, None).is_empty())

    def test_blank_page(self):
        self.assertTrue(
            detect_shapes_from_image(Canvas(200, 260).page(SCALE), GEOMETRY, None).is_empty()
        )

    def test_all_black_page(self):
        canvas = Canvas(60, 60)
        canvas.fill(0, 0, 60, 60)
        shapes = detect_shapes_from_image(canvas.page(SCALE), GEOMETRY, None)
        self.assertIsInstance(shapes, RasterShapes)

    def test_bad_buffer_length(self):
        page = StandInPage(0, 10, 10, 2.0, b"\xff" * 5)
        self.assertTrue(detect_shapes_from_image(page, GEOMETRY, None).is_empty())

    def test_zero_scale(self):
        page = StandInPage(0, 10, 10, 0.0, b"\xff" * 100)
        self.assertTrue(detect_shapes_from_image(page, GEOMETRY, None).is_empty())

    def test_accepts_any_duck_typed_page(self):
        canvas = drawn_page()
        shapes = detect_shapes_from_image(canvas.stand_in(SCALE), GEOMETRY, None)
        self.assertEqual(len(shapes.boxes), 1)
        near(self, shapes.boxes[0], [10.0, 59.0, 90.0, 110.0])

    def test_short_marks_are_not_rules(self):
        canvas = Canvas(200, 260)
        canvas.fill(20, 100, 40, 101)  # 20px = 10pt, under min_line_length_pt
        shapes = detect_shapes_from_image(canvas.page(SCALE), GEOMETRY, None)
        self.assertEqual(shapes.h_rules, [])

    def test_thick_bar_is_not_a_rule(self):
        canvas = Canvas(200, 260)
        canvas.fill(20, 100, 180, 120)  # 20px = 10pt thick
        shapes = detect_shapes_from_image(canvas.page(SCALE), GEOMETRY, None)
        self.assertEqual(shapes.h_rules, [])

    def test_dashed_rule_is_recovered(self):
        canvas = Canvas(200, 260)
        for index in range(16):
            canvas.fill(20 + index * 10, 100, 28 + index * 10, 102)
        shapes = detect_shapes_from_image(canvas.page(SCALE), GEOMETRY, None)
        self.assertEqual(len(shapes.h_rules), 1)
        near(self, shapes.h_rules[0], [10.0, 79.0, 89.0, 80.0])


class RasterDeterminismTests(unittest.TestCase):
    def test_repeat_runs_are_identical(self):
        page = drawn_page().page(SCALE)
        first = detect_shapes_from_image(page, GEOMETRY, None).as_dict()
        second = detect_shapes_from_image(page, GEOMETRY, None).as_dict()
        self.assertEqual(first, second)

    def test_blanks_are_sorted_by_area(self):
        shapes = detect_shapes_from_image(drawn_page().page(SCALE), GEOMETRY, None)
        areas = [r.area for r in shapes.blanks]
        self.assertEqual(areas, sorted(areas, reverse=True))

    def test_empty_result_is_a_dataclass(self):
        shapes = RasterShapes()
        self.assertTrue(shapes.is_empty())
        self.assertEqual(shapes.all_rects(), [])
        self.assertEqual(shapes.as_primitives(), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
