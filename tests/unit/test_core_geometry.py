"""Unit tests for :mod:`zfp.core.geometry` and :mod:`zfp.core.units`."""

from __future__ import annotations

import math
import unittest

from zfp.core import units
from zfp.core.errors import ValidationError
from zfp.core.geometry import EPS, Matrix, PageGeometry, Point, Rect


class PointTest(unittest.TestCase):
    def test_translated_and_tuple(self) -> None:
        p = Point(1.5, -2.0)
        self.assertEqual(p.translated(2.5, 4.0), Point(4.0, 2.0))
        self.assertEqual(p.as_tuple(), (1.5, -2.0))

    def test_distance(self) -> None:
        self.assertAlmostEqual(Point(0, 0).distance_to(Point(3, 4)), 5.0)
        self.assertAlmostEqual(Point(2, 2).distance_to(Point(2, 2)), 0.0)

    def test_is_hashable_and_frozen(self) -> None:
        self.assertEqual(len({Point(1, 1), Point(1, 1), Point(2, 2)}), 2)
        with self.assertRaises(Exception):
            Point(1, 1).x = 5  # type: ignore[misc]


class RectConstructionTest(unittest.TestCase):
    def test_from_points_normalizes(self) -> None:
        r = Rect.from_points(Point(10, 20), Point(4, 6))
        self.assertEqual(r, Rect(4, 6, 10, 20))

    def test_from_list_normalizes(self) -> None:
        self.assertEqual(Rect.from_list([10, 20, 4, 6]), Rect(4, 6, 10, 20))
        self.assertEqual(Rect.from_list((0, 0, 1, 1, 99)), Rect(0, 0, 1, 1))

    def test_from_list_rejects_short_input(self) -> None:
        with self.assertRaises(ValidationError):
            Rect.from_list([1, 2, 3])

    def test_bounding(self) -> None:
        self.assertIsNone(Rect.bounding([]))
        rects = [Rect(0, 0, 5, 5), Rect(-2, 3, 1, 9), Rect(4, 4, 4, 4)]
        self.assertEqual(Rect.bounding(rects), Rect(-2, 0, 5, 9))

    def test_bounding_normalizes_inputs(self) -> None:
        self.assertEqual(Rect.bounding([Rect(5, 5, 0, 0)]), Rect(0, 0, 5, 5))


class RectPropertyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.r = Rect(10, 20, 40, 60)

    def test_dimensions(self) -> None:
        self.assertEqual(self.r.width, 30)
        self.assertEqual(self.r.height, 40)
        self.assertEqual(self.r.area, 1200)
        self.assertEqual(self.r.center, Point(25, 40))

    def test_dimensions_are_absolute_for_inverted_rects(self) -> None:
        inverted = Rect(40, 60, 10, 20)
        self.assertEqual(inverted.width, 30)
        self.assertEqual(inverted.height, 40)
        self.assertEqual(inverted.area, 1200)

    def test_normalized(self) -> None:
        self.assertEqual(Rect(40, 60, 10, 20).normalized(), self.r)
        self.assertEqual(Rect(10, 60, 40, 20).normalized(), self.r)
        self.assertEqual(self.r.normalized(), self.r)

    def test_degenerate_rule_has_zero_height(self) -> None:
        rule = Rect(100, 500, 300, 500)
        self.assertEqual(rule.height, 0.0)
        self.assertEqual(rule.area, 0.0)
        self.assertEqual(rule.normalized(), rule)


class RectTransformTest(unittest.TestCase):
    def test_inflated_single_argument(self) -> None:
        self.assertEqual(Rect(0, 0, 10, 10).inflated(2), Rect(-2, -2, 12, 12))

    def test_inflated_two_arguments(self) -> None:
        self.assertEqual(Rect(0, 0, 10, 10).inflated(2, 5), Rect(-2, -5, 12, 15))

    def test_inflated_normalizes_first(self) -> None:
        self.assertEqual(Rect(10, 10, 0, 0).inflated(1), Rect(-1, -1, 11, 11))

    def test_translated(self) -> None:
        self.assertEqual(Rect(1, 2, 3, 4).translated(-1, 10), Rect(0, 12, 2, 14))

    def test_scaled(self) -> None:
        self.assertEqual(Rect(1, 2, 3, 4).scaled(2), Rect(2, 4, 6, 8))
        self.assertEqual(Rect(1, 2, 3, 4).scaled(2, 0.5), Rect(2, 1, 6, 2))

    def test_scaled_negative_normalizes(self) -> None:
        self.assertEqual(Rect(1, 2, 3, 4).scaled(-1, 1), Rect(-3, 2, -1, 4))

    def test_rounded(self) -> None:
        self.assertEqual(Rect(1.23456, 2.0, 3.0, 4.98765).rounded(3), Rect(1.235, 2.0, 3.0, 4.988))
        self.assertEqual(Rect(1.4, 1.6, 2.4, 2.6).rounded(0), Rect(1.0, 2.0, 2.0, 3.0))

    def test_as_list(self) -> None:
        self.assertEqual(Rect(1, 2, 3, 4).as_list(), [1, 2, 3, 4])


class RectSetOpTest(unittest.TestCase):
    def test_union_is_bounding_box(self) -> None:
        self.assertEqual(Rect(0, 0, 2, 2).union(Rect(10, 10, 12, 12)), Rect(0, 0, 12, 12))

    def test_union_normalizes(self) -> None:
        self.assertEqual(Rect(2, 2, 0, 0).union(Rect(3, 3, 1, 1)), Rect(0, 0, 3, 3))

    def test_intersection_overlapping(self) -> None:
        self.assertEqual(Rect(0, 0, 10, 10).intersection(Rect(5, 5, 20, 20)), Rect(5, 5, 10, 10))

    def test_intersection_disjoint_is_none(self) -> None:
        self.assertIsNone(Rect(0, 0, 10, 10).intersection(Rect(11, 0, 20, 10)))
        self.assertIsNone(Rect(0, 0, 10, 10).intersection(Rect(0, 11, 10, 20)))

    def test_intersection_touching_is_zero_area(self) -> None:
        touching = Rect(0, 0, 10, 10).intersection(Rect(10, 0, 20, 10))
        self.assertIsNotNone(touching)
        assert touching is not None
        self.assertEqual(touching.area, 0.0)

    def test_intersection_of_contained_rect(self) -> None:
        self.assertEqual(Rect(0, 0, 10, 10).intersection(Rect(2, 2, 4, 4)), Rect(2, 2, 4, 4))

    def test_intersection_with_degenerate_rule(self) -> None:
        rule = Rect(2, 5, 8, 5)
        self.assertEqual(Rect(0, 0, 10, 10).intersection(rule), rule)

    def test_iou(self) -> None:
        a = Rect(0, 0, 10, 10)
        self.assertAlmostEqual(a.iou(a), 1.0)
        self.assertAlmostEqual(a.iou(Rect(20, 20, 30, 30)), 0.0)
        # 5x10 overlap; union = 100 + 100 - 50 = 150
        self.assertAlmostEqual(a.iou(Rect(5, 0, 15, 10)), 50.0 / 150.0)

    def test_iou_of_zero_area_rects_is_zero(self) -> None:
        self.assertEqual(Rect(0, 0, 0, 0).iou(Rect(0, 0, 0, 0)), 0.0)

    def test_iou_is_symmetric(self) -> None:
        a, b = Rect(0, 0, 10, 4), Rect(3, 1, 12, 9)
        self.assertAlmostEqual(a.iou(b), b.iou(a))

    def test_predicates(self) -> None:
        r = Rect(0, 0, 10, 10)
        self.assertTrue(r.contains_point(Point(5, 5)))
        self.assertTrue(r.contains_point(Point(0, 10)))
        self.assertFalse(r.contains_point(Point(-0.1, 5)))
        self.assertTrue(r.contains_rect(Rect(1, 1, 9, 9)))
        self.assertTrue(r.contains_rect(r))
        self.assertFalse(r.contains_rect(Rect(1, 1, 11, 9)))
        self.assertTrue(r.intersects(Rect(9, 9, 20, 20)))
        self.assertFalse(r.intersects(Rect(10.5, 0, 20, 10)))

    def test_overlap_lengths(self) -> None:
        a = Rect(0, 0, 10, 10)
        self.assertAlmostEqual(a.horizontal_overlap(Rect(5, 100, 30, 200)), 5.0)
        self.assertAlmostEqual(a.vertical_overlap(Rect(100, 5, 200, 30)), 5.0)
        self.assertAlmostEqual(a.horizontal_overlap(Rect(11, 0, 20, 10)), 0.0)
        self.assertAlmostEqual(a.vertical_overlap(Rect(0, 11, 10, 20)), 0.0)
        self.assertAlmostEqual(a.horizontal_overlap(Rect(-5, 0, 15, 10)), 10.0)


class MatrixTest(unittest.TestCase):
    def test_identity(self) -> None:
        self.assertEqual(Matrix.identity().as_tuple(), (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        self.assertEqual(Matrix.identity().apply(Point(3, 4)), Point(3, 4))

    def test_translation_and_scaling(self) -> None:
        self.assertEqual(Matrix.translation(5, -3).apply(Point(1, 1)), Point(6, -2))
        self.assertEqual(Matrix.scaling(2, 3).apply(Point(2, 2)), Point(4, 6))

    def test_concat_is_self_then_other(self) -> None:
        m = Matrix.translation(10, 0).concat(Matrix.scaling(2, 2))
        self.assertEqual(m.apply(Point(1, 1)), Point(22, 2))

    def test_concat_is_not_commutative(self) -> None:
        other = Matrix.scaling(2, 2).concat(Matrix.translation(10, 0))
        self.assertEqual(other.apply(Point(1, 1)), Point(12, 2))

    def test_concat_with_identity(self) -> None:
        m = Matrix(2, 0.5, -1, 3, 7, -2)
        self.assertEqual(m.concat(Matrix.identity()), m)
        self.assertEqual(Matrix.identity().concat(m), m)

    def test_concat_is_associative(self) -> None:
        a = Matrix.translation(3, 4)
        b = Matrix.rotation(30)
        c = Matrix.scaling(2, 5)
        left = a.concat(b).concat(c)
        right = a.concat(b.concat(c))
        for x, y in zip(left.as_tuple(), right.as_tuple()):
            self.assertAlmostEqual(x, y)

    def test_rotation_is_counter_clockwise(self) -> None:
        p = Matrix.rotation(90).apply(Point(1, 0))
        self.assertAlmostEqual(p.x, 0.0)
        self.assertAlmostEqual(p.y, 1.0)
        p = Matrix.rotation(180).apply(Point(1, 0))
        self.assertAlmostEqual(p.x, -1.0)
        self.assertAlmostEqual(p.y, 0.0)

    def test_rotation_preserves_length(self) -> None:
        for deg in (0, 17, 45, 90, 180, 270, 359):
            p = Matrix.rotation(deg).apply(Point(3, 4))
            self.assertAlmostEqual(math.hypot(p.x, p.y), 5.0)

    def test_apply_xy(self) -> None:
        self.assertEqual(Matrix(2, 0, 0, 2, 1, 1).apply_xy(3, 4), (7.0, 9.0))

    def test_transform_rect_is_axis_aligned_bbox(self) -> None:
        r = Rect(0, 0, 10, 4)
        out = Matrix.rotation(90).transform_rect(r)
        self.assertAlmostEqual(out.x0, -4.0)
        self.assertAlmostEqual(out.y0, 0.0)
        self.assertAlmostEqual(out.x1, 0.0)
        self.assertAlmostEqual(out.y1, 10.0)

    def test_inverted_round_trip(self) -> None:
        for m in (
            Matrix.translation(13, -7),
            Matrix.scaling(2.5, -4),
            Matrix.rotation(37),
            Matrix.translation(5, 5).concat(Matrix.rotation(120)).concat(Matrix.scaling(3, 3)),
        ):
            inv = m.inverted()
            p = Point(11.25, -3.5)
            back = inv.apply(m.apply(p))
            self.assertAlmostEqual(back.x, p.x, places=9)
            self.assertAlmostEqual(back.y, p.y, places=9)

    def test_inverted_composed_with_self_is_identity(self) -> None:
        m = Matrix(2, 1, -1, 3, 9, -4)
        ident = m.concat(m.inverted())
        for got, want in zip(ident.as_tuple(), Matrix.identity().as_tuple()):
            self.assertAlmostEqual(got, want, places=9)

    def test_singular_matrix_raises(self) -> None:
        with self.assertRaises(ValidationError):
            Matrix(0, 0, 0, 0, 0, 0).inverted()

    def test_determinant(self) -> None:
        self.assertAlmostEqual(Matrix.scaling(3, 4).determinant(), 12.0)


class PageGeometryTest(unittest.TestCase):
    MEDIA = Rect(0, 0, 650, 850)
    CROP = Rect(20, 30, 620, 830)  # 600 x 800, non-zero origin
    SCALE = 2.0

    def geom(self, rotation: int) -> PageGeometry:
        return PageGeometry(index=0, media_box=self.MEDIA, crop_box=self.CROP, rotation=rotation)

    def test_rotation_normalization(self) -> None:
        self.assertEqual(self.geom(0).rotation, 0)
        self.assertEqual(self.geom(360).rotation, 0)
        self.assertEqual(self.geom(-90).rotation, 270)
        self.assertEqual(self.geom(450).rotation, 90)
        self.assertEqual(PageGeometry(0, self.MEDIA, self.CROP, 44).rotation, 0)
        self.assertEqual(PageGeometry(0, self.MEDIA, self.CROP, 46).rotation, 90)

    def test_boxes_are_normalized_on_construction(self) -> None:
        g = PageGeometry(0, Rect(650, 850, 0, 0), Rect(620, 830, 20, 30))
        self.assertEqual(g.media_box, self.MEDIA)
        self.assertEqual(g.crop_box, self.CROP)

    def test_dimensions(self) -> None:
        g = self.geom(0)
        self.assertEqual(g.width, 600)
        self.assertEqual(g.height, 800)
        self.assertEqual(g.display_size, (600, 800))
        self.assertEqual(self.geom(90).display_size, (800, 600))
        self.assertEqual(self.geom(180).display_size, (600, 800))
        self.assertEqual(self.geom(270).display_size, (800, 600))

    def test_pixel_size(self) -> None:
        self.assertEqual(self.geom(0).pixel_size(2.0), (1200, 1600))
        self.assertEqual(self.geom(90).pixel_size(2.0), (1600, 1200))

    def test_render_matrix_maps_crop_origin_top_left_for_rotation_0(self) -> None:
        g = self.geom(0)
        # user-space top-left of the crop box -> pixel (0, 0)
        self.assertEqual(g.user_to_pixel(20, 830, self.SCALE), (0.0, 0.0))
        # user-space bottom-right -> pixel (W, H)
        self.assertEqual(g.user_to_pixel(620, 30, self.SCALE), (1200.0, 1600.0))

    def test_top_left_corner_rotates_clockwise(self) -> None:
        """The displayed page turns clockwise, so the top-left corner walks TL->TR->BR->BL."""
        expected = {
            0: (0.0, 0.0),
            90: (1600.0, 0.0),
            180: (1200.0, 1600.0),
            270: (0.0, 1200.0),
        }
        for rotation, want in expected.items():
            got = self.geom(rotation).user_to_pixel(20, 830, self.SCALE)
            self.assertEqual(got, want, "rotation %d" % rotation)

    def test_render_matrix_fills_the_raster_exactly(self) -> None:
        for rotation in (0, 90, 180, 270):
            g = self.geom(rotation)
            w, h = g.pixel_size(self.SCALE)
            box = g.user_rect_to_pixel(self.CROP, self.SCALE)
            self.assertAlmostEqual(box.x0, 0.0, places=6)
            self.assertAlmostEqual(box.y0, 0.0, places=6)
            self.assertAlmostEqual(box.x1, float(w), places=6)
            self.assertAlmostEqual(box.y1, float(h), places=6)

    def test_pixel_rect_round_trip_all_rotations(self) -> None:
        source = Rect(120.25, 245.75, 388.5, 301.125)
        for rotation in (0, 90, 180, 270):
            for scale in (1.0, 2.0, 300.0 / 72.0):
                g = self.geom(rotation)
                pixels = g.user_rect_to_pixel(source, scale)
                back = g.pixel_rect_to_user(pixels, scale)
                for got, want in zip(back.as_list(), source.as_list()):
                    self.assertAlmostEqual(got, want, delta=1e-6)

    def test_point_round_trip_all_rotations(self) -> None:
        for rotation in (0, 90, 180, 270):
            g = self.geom(rotation)
            px, py = g.user_to_pixel(333.5, 411.25, self.SCALE)
            back = g.pixel_to_user(px, py, self.SCALE)
            self.assertAlmostEqual(back.x, 333.5, delta=1e-6)
            self.assertAlmostEqual(back.y, 411.25, delta=1e-6)

    def test_pixels_stay_inside_the_raster(self) -> None:
        for rotation in (0, 90, 180, 270):
            g = self.geom(rotation)
            w, h = g.pixel_size(self.SCALE)
            for x, y in ((20, 30), (620, 830), (320, 430)):
                px, py = g.user_to_pixel(x, y, self.SCALE)
                self.assertGreaterEqual(px, -EPS)
                self.assertGreaterEqual(py, -EPS)
                self.assertLessEqual(px, w + EPS)
                self.assertLessEqual(py, h + EPS)

    def test_clamp(self) -> None:
        g = self.geom(0)
        self.assertEqual(g.clamp(Rect(0, 0, 1000, 1000)), self.CROP)
        self.assertEqual(g.clamp(Rect(100, 100, 200, 200)), Rect(100, 100, 200, 200))
        self.assertEqual(g.clamp(Rect(10, 10, 30, 40)), Rect(20, 30, 30, 40))
        # entirely outside -> a degenerate rect on the boundary, never inverted
        outside = g.clamp(Rect(700, 900, 800, 1000))
        self.assertEqual(outside, Rect(620, 830, 620, 830))
        self.assertLessEqual(outside.x0, outside.x1)


class UnitsTest(unittest.TestCase):
    def test_constants(self) -> None:
        self.assertEqual(units.PT_PER_INCH, 72.0)
        self.assertEqual(units.MM_PER_INCH, 25.4)

    def test_points_and_pixels(self) -> None:
        self.assertAlmostEqual(units.pt_to_px(72, 300), 300.0)
        self.assertAlmostEqual(units.px_to_pt(300, 300), 72.0)
        self.assertAlmostEqual(units.px_to_pt(units.pt_to_px(123.5, 150), 150), 123.5)
        self.assertEqual(units.px_to_pt(10, 0), 0.0)

    def test_millimetres(self) -> None:
        self.assertAlmostEqual(units.mm_to_pt(25.4), 72.0)
        self.assertAlmostEqual(units.pt_to_mm(72.0), 25.4)
        self.assertAlmostEqual(units.pt_to_mm(units.mm_to_pt(210.0)), 210.0)

    def test_inches(self) -> None:
        self.assertAlmostEqual(units.inch_to_pt(8.5), 612.0)
        self.assertAlmostEqual(units.pt_to_inch(792.0), 11.0)

    def test_scale(self) -> None:
        self.assertAlmostEqual(units.dpi_to_scale(72), 1.0)
        self.assertAlmostEqual(units.dpi_to_scale(300), 300.0 / 72.0)
        self.assertAlmostEqual(units.scale_to_dpi(units.dpi_to_scale(400)), 400.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
