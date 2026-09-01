"""Unit tests for :mod:`zfp.raster.preprocess`.

Every fixture is drawn into a byte buffer here in the test: shaded gradients, text-like
bars, salt-and-pepper speckle.  No scan fixtures, no optional dependencies, no clock.
"""

from __future__ import annotations

import unittest

from zfp.core.errors import ValidationError
from zfp.raster.preprocess import (
    PreprocessReport,
    binarize,
    denoise,
    deskew,
    detect_orientation,
    estimate_skew,
    histogram,
    ink_mask,
    normalize_contrast,
    otsu_threshold,
    preprocess,
    rotate_quarter,
)
from zfp.raster.render import RenderedPage

# ======================================================================================
# Canvas helpers
# ======================================================================================


class Canvas:
    """A tiny mutable gray raster, so the tests can draw their own fixtures."""

    def __init__(self, width, height, level=255):
        self.width = width
        self.height = height
        self.data = bytearray([level]) * (width * height)

    def set(self, x, y, value):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y * self.width + x] = value

    def get(self, x, y):
        return self.data[y * self.width + x]

    def rect(self, x0, y0, x1, y1, value):
        for y in range(max(0, y0), min(self.height, y1)):
            for x in range(max(0, x0), min(self.width, x1)):
                self.data[y * self.width + x] = value

    def page(self, scale=1.0):
        return RenderedPage(
            page=0,
            width=self.width,
            height=self.height,
            scale=scale,
            gray=bytes(self.data),
            backend="test",
        )


def text_bars(width=240, height=180, spacing=20, thickness=3, margin=16):
    """A page of horizontal bars: a stand-in for lines of text."""
    canvas = Canvas(width, height)
    y = margin
    while y + thickness <= height - margin:
        canvas.rect(margin, y, width - margin, y + thickness, 0)
        y += spacing
    return canvas


def shaded_text(width=128, height=96):
    """Bars that stay 60 levels darker than a strong left-to-right background gradient."""
    canvas = Canvas(width, height)
    background = [60 + (x * 160) // max(1, width - 1) for x in range(width)]
    truth = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            canvas.data[y * width + x] = background[x]
    for top in (10, 30, 50, 70):
        if top + 4 > height - 4:
            break
        for y in range(top, top + 4):
            for x in range(10, width - 10):
                canvas.data[y * width + x] = max(0, background[x] - 60)
                truth[y * width + x] = 1
    return canvas, bytes(truth)


def accuracy(gray, truth):
    """Fraction of pixels whose ink/paper classification matches ``truth``."""
    correct = 0
    for value, expected in zip(gray, truth):
        if (value < 128) == bool(expected):
            correct += 1
    return correct / float(len(truth))


# ======================================================================================
# Histogram, contrast
# ======================================================================================


class HistogramTests(unittest.TestCase):
    def test_histogram_counts_every_level(self):
        bins = histogram(bytes([0, 0, 5, 255]))
        self.assertEqual(bins[0], 2)
        self.assertEqual(bins[5], 1)
        self.assertEqual(bins[255], 1)
        self.assertEqual(sum(bins), 4)

    def test_normalize_contrast_stretches_a_low_contrast_page(self):
        canvas = Canvas(40, 40, level=200)
        canvas.rect(5, 5, 35, 35, 140)
        page = canvas.page()
        stretched = normalize_contrast(page)
        self.assertEqual(min(stretched.gray), 0)
        self.assertEqual(max(stretched.gray), 255)
        # The ordering of the two populations survives the stretch.
        self.assertLess(stretched.pixel(10, 10), stretched.pixel(0, 0))

    def test_normalize_contrast_leaves_a_flat_page_alone(self):
        page = Canvas(8, 8, level=77).page()
        self.assertEqual(normalize_contrast(page).gray, page.gray)

    def test_normalize_contrast_ignores_outliers(self):
        canvas = Canvas(20, 20, level=100)
        canvas.rect(10, 10, 14, 14, 180)
        canvas.set(0, 0, 0)      # a single black speck
        canvas.set(19, 19, 255)  # a single white speck
        stretched = normalize_contrast(canvas.page())
        self.assertEqual(stretched.pixel(1, 1), 0)
        self.assertEqual(stretched.pixel(11, 11), 255)


# ======================================================================================
# Binarization
# ======================================================================================


class BinarizeTests(unittest.TestCase):
    def test_otsu_finds_the_valley_of_a_bimodal_image(self):
        canvas = Canvas(64, 64, level=40)
        canvas.rect(0, 0, 64, 32, 200)
        threshold = otsu_threshold(histogram(canvas.page().gray))
        self.assertGreaterEqual(threshold, 40)
        self.assertLess(threshold, 200)
        self.assertTrue(90 <= threshold <= 150, "unexpected threshold %d" % threshold)

    def test_otsu_binarize_splits_the_two_populations(self):
        canvas = Canvas(32, 32, level=30)
        canvas.rect(0, 0, 32, 16, 220)
        result = binarize(canvas.page(), method="otsu")
        self.assertEqual(set(result.gray), {0, 255})
        self.assertEqual(result.pixel(4, 4), 255)
        self.assertEqual(result.pixel(4, 20), 0)

    def test_sauvola_beats_a_global_threshold_on_a_shaded_page(self):
        canvas, truth = shaded_text()
        page = canvas.page()
        sauvola = accuracy(binarize(page, method="sauvola").gray, truth)
        otsu = accuracy(binarize(page, method="otsu").gray, truth)
        self.assertGreater(sauvola, 0.95)
        self.assertLess(otsu, 0.75)
        self.assertGreater(sauvola, otsu + 0.2)

    def test_sauvola_output_is_bilevel_and_sized(self):
        canvas, _truth = shaded_text(64, 48)
        result = binarize(canvas.page(), method="sauvola")
        self.assertEqual(len(result.gray), 64 * 48)
        self.assertLessEqual(set(result.gray), {0, 255})

    def test_method_aliases_and_validation(self):
        page = Canvas(16, 16, level=128).page()
        self.assertEqual(binarize(page, "SAUVOLA").gray, binarize(page, "adaptive").gray)
        self.assertEqual(binarize(page, "otsu").gray, binarize(page, "global").gray)
        with self.assertRaises(ValidationError):
            binarize(page, method="telepathy")

    def test_ink_mask_marks_dark_pixels(self):
        canvas = Canvas(8, 2, level=255)
        canvas.rect(0, 0, 4, 1, 0)
        mask, threshold = ink_mask(canvas.page())
        self.assertEqual(mask.count(1), 4)
        self.assertLess(threshold, 255)


# ======================================================================================
# Denoising
# ======================================================================================


class DenoiseTests(unittest.TestCase):
    def test_median_removes_salt_and_pepper(self):
        canvas = Canvas(48, 48, level=255)
        canvas.rect(10, 10, 38, 38, 0)
        speckles = [(3, 3), (44, 5), (5, 44), (44, 44), (25, 2)]
        holes = [(20, 20), (30, 30), (15, 25)]
        for x, y in speckles:
            canvas.set(x, y, 0)
        for x, y in holes:
            canvas.set(x, y, 255)
        cleaned = denoise(canvas.page())
        for x, y in speckles:
            self.assertEqual(cleaned.pixel(x, y), 255, "speck at %d,%d survived" % (x, y))
        for x, y in holes:
            self.assertEqual(cleaned.pixel(x, y), 0, "hole at %d,%d survived" % (x, y))
        # The block itself is untouched away from its edges.
        self.assertEqual(cleaned.pixel(24, 24), 0)
        self.assertEqual(cleaned.pixel(1, 1), 255)

    def test_small_components_are_removed_but_real_marks_survive(self):
        from zfp.raster.preprocess import _median3

        canvas = Canvas(60, 60, level=255)
        # A plus: the median filter erodes it to its single centre pixel, and only the
        # connected-component pass can remove what is left.
        for x, y in ((10, 9), (9, 10), (10, 10), (11, 10), (10, 11)):
            canvas.set(x, y, 0)
        canvas.rect(30, 30, 40, 40, 0)  # a real mark
        page = canvas.page()
        filtered = _median3(page.gray, page.width, page.height)
        self.assertEqual(filtered[10 * 60 + 10], 0, "the median alone should leave a dot")
        cleaned = denoise(page)
        for y in range(8, 13):
            for x in range(8, 13):
                self.assertEqual(cleaned.pixel(x, y), 255)
        self.assertEqual(cleaned.pixel(35, 35), 0)
        self.assertEqual(cleaned.pixel(31, 31), 0)

    def test_tiny_blobs_never_survive_the_median(self):
        canvas = Canvas(40, 40, level=255)
        canvas.rect(5, 5, 7, 7, 0)  # a 2x2 speck
        cleaned = denoise(canvas.page())
        self.assertEqual(cleaned.gray.count(0), 0)

    def test_denoise_preserves_the_raster_shape(self):
        page = Canvas(17, 13, level=200).page()
        cleaned = denoise(page)
        self.assertEqual((cleaned.width, cleaned.height), (17, 13))
        self.assertEqual(len(cleaned.gray), 17 * 13)

    def test_denoise_on_a_grayscale_page(self):
        canvas = Canvas(32, 32, level=180)
        canvas.rect(8, 8, 24, 24, 40)
        canvas.set(3, 3, 0)
        cleaned = denoise(canvas.page())
        self.assertEqual(cleaned.pixel(3, 3), 180)
        self.assertEqual(cleaned.pixel(16, 16), 40)


# ======================================================================================
# Skew and orientation
# ======================================================================================


class SkewTests(unittest.TestCase):
    def test_straight_text_has_no_skew(self):
        page = text_bars().page()
        self.assertLessEqual(abs(estimate_skew(page)), 0.25)

    def test_a_known_two_degree_rotation_is_recovered(self):
        page = text_bars().page()
        for angle in (2.0, -2.0):
            rotated = deskew(page, angle)
            estimated = estimate_skew(rotated)
            self.assertAlmostEqual(estimated, angle, delta=0.5)

    def test_deskew_undoes_itself(self):
        page = text_bars().page()
        restored = deskew(deskew(page, 3.0), -3.0)
        self.assertEqual((restored.width, restored.height), (page.width, page.height))
        # Compare the interior, where nothing was rotated in from outside the canvas.
        total = 0
        count = 0
        for y in range(20, page.height - 20):
            for x in range(20, page.width - 20):
                total += abs(restored.pixel(x, y) - page.pixel(x, y))
                count += 1
        self.assertLess(total / float(count), 12.0)

    def test_deskew_correction_straightens_the_profile(self):
        page = text_bars().page()
        skewed = deskew(page, 2.5)
        corrected = deskew(skewed, -estimate_skew(skewed))
        self.assertLessEqual(abs(estimate_skew(corrected)), 0.5)

    def test_deskew_of_zero_is_a_no_op(self):
        page = text_bars(40, 40).page()
        self.assertIs(deskew(page, 0.0), page)

    def test_blank_pages_have_no_skew(self):
        self.assertEqual(estimate_skew(Canvas(40, 40).page()), 0.0)


class OrientationTests(unittest.TestCase):
    def test_horizontal_text_is_upright(self):
        self.assertEqual(detect_orientation(text_bars().page()), 0)

    def test_vertical_text_is_sideways(self):
        page = rotate_quarter(text_bars().page(), 90)
        self.assertEqual(detect_orientation(page), 90)

    def test_blank_page_is_upright(self):
        self.assertEqual(detect_orientation(Canvas(20, 20).page()), 0)

    def test_rotate_quarter_round_trips(self):
        canvas = Canvas(6, 4, level=255)
        canvas.rect(0, 0, 2, 1, 0)
        page = canvas.page()
        self.assertIs(rotate_quarter(page, 0), page)
        turned = rotate_quarter(page, 90)
        self.assertEqual((turned.width, turned.height), (4, 6))
        self.assertEqual(rotate_quarter(turned, 270).gray, page.gray)
        self.assertEqual(rotate_quarter(rotate_quarter(page, 180), 180).gray, page.gray)

    def test_rotate_quarter_moves_the_corner_clockwise(self):
        canvas = Canvas(4, 3, level=255)
        canvas.set(0, 0, 0)  # top-left
        turned = rotate_quarter(canvas.page(), 90)
        self.assertEqual((turned.width, turned.height), (3, 4))
        self.assertEqual(turned.pixel(2, 0), 0)  # now top-right
        self.assertEqual(turned.gray.count(0), 1)


# ======================================================================================
# The pipeline
# ======================================================================================


class PipelineTests(unittest.TestCase):
    def test_preprocess_reports_every_step(self):
        canvas, _truth = shaded_text(96, 72)
        canvas.set(3, 3, 0)
        page, report = preprocess(canvas.page())
        self.assertIsInstance(report, PreprocessReport)
        self.assertEqual(set(page.gray), {0, 255})
        self.assertEqual((page.width, page.height), (96, 72))
        self.assertEqual(report.method, "sauvola")
        self.assertEqual(report.orientation, 0)
        self.assertTrue(any(step.startswith("orientation") for step in report.steps))
        self.assertTrue(any(step.startswith("denoise") for step in report.steps))
        self.assertTrue(any(step.startswith("binarize") for step in report.steps))
        self.assertGreater(report.ink_ratio, 0.0)
        self.assertLess(report.ink_ratio, 1.0)
        self.assertGreaterEqual(report.threshold, 0)
        self.assertLessEqual(report.threshold, 255)

    def test_preprocess_corrects_a_skewed_page(self):
        page = deskew(text_bars(120, 96).page(), 2.0)
        cleaned, report = preprocess(page)
        self.assertAlmostEqual(report.skew_angle, 2.0, delta=0.6)
        self.assertTrue(any(step.startswith("deskew") for step in report.steps))
        self.assertLessEqual(abs(estimate_skew(cleaned)), 0.6)

    def test_preprocess_is_deterministic(self):
        canvas, _truth = shaded_text(64, 48)
        first, report_a = preprocess(canvas.page())
        second, report_b = preprocess(canvas.page())
        self.assertEqual(first.gray, second.gray)
        self.assertEqual(report_a.as_dict(), report_b.as_dict())

    def test_preprocess_with_otsu(self):
        canvas = Canvas(48, 48, level=210)
        canvas.rect(10, 10, 40, 20, 30)
        _page, report = preprocess(canvas.page(), method="otsu")
        self.assertEqual(report.method, "otsu")

    def test_preprocess_of_an_empty_raster(self):
        page = RenderedPage(page=0, width=0, height=0, scale=1.0, gray=b"", backend="test")
        result, report = preprocess(page)
        self.assertIs(result, page)
        self.assertIn("empty raster", report.notes)

    def test_report_serializes(self):
        canvas, _truth = shaded_text(48, 48)
        _page, report = preprocess(canvas.page())
        data = report.as_dict()
        self.assertIn("steps", data)
        self.assertIsInstance(data["skew_angle"], float)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
