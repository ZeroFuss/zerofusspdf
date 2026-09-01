"""Unit tests for :mod:`zfp.ocr.engine`.

No OCR engine is installed in this environment and the tests must not need one, so the
backends are exercised at their seams: the TSV parser is fed a real Tesseract dump as a
string, the Paddle adapter is fed each of the result shapes Paddle has shipped, and the
shared pixel -> user-space conversion is driven by a :class:`BaseEngine` subclass that
returns scripted pixel boxes.

The conversion tests are the ones that matter most.  Field placement is won or lost on
that transform, so the expected user-space rectangles are written out as literals rather
than recomputed from the geometry -- a test that calls ``pixel_rect_to_user`` to check
that ``pixel_rect_to_user`` was called proves nothing.
"""

from __future__ import annotations

import math
import shutil
import unittest
from typing import List

from zfp.core.config import OcrConfig
from zfp.core.errors import UnsupportedFeatureError, ValidationError
from zfp.core.geometry import PageGeometry, Rect
from zfp.core.optional import have
from zfp.ocr.engine import (
    BaseEngine,
    NullEngine,
    OcrEngine,
    PaddleEngine,
    PixelWord,
    TesseractEngine,
    available_engines,
    clear_engine_cache,
    engine_names,
    get_engine,
    parse_paddle_result,
    parse_tesseract_tsv,
    register_engine,
    resolve_ocr_config,
    unregister_engine,
)

LETTER = PageGeometry(
    index=0, media_box=Rect(0, 0, 612, 792), crop_box=Rect(0, 0, 612, 792), rotation=0
)
LETTER_90 = PageGeometry(
    index=0, media_box=Rect(0, 0, 612, 792), crop_box=Rect(0, 0, 612, 792), rotation=90
)


class StubPage:
    """A stand-in for :class:`zfp.raster.render.RenderedPage`.

    The real class is written by another module; duplicating its tiny surface here keeps
    these tests from depending on a renderer, and documents exactly which four attributes
    and two methods the OCR package actually uses.
    """

    def __init__(self, width: int, height: int, scale: float = 2.0, page: int = 0) -> None:
        self.page = page
        self.width = width
        self.height = height
        self.scale = scale
        self.gray = b"\xff" * (width * height)
        self.backend = "stub"

    def crop(self, rect_px: Rect) -> "StubPage":
        """Mimic ``RenderedPage.crop``: floor/ceil, clamp, 1x1 white when it misses."""
        rect = rect_px.normalized()
        x0 = max(0, int(math.floor(rect.x0)))
        y0 = max(0, int(math.floor(rect.y0)))
        x1 = min(self.width, int(math.ceil(rect.x1)))
        y1 = min(self.height, int(math.ceil(rect.y1)))
        if x1 <= x0 or y1 <= y0:
            return StubPage(1, 1, self.scale, self.page)
        return StubPage(x1 - x0, y1 - y0, self.scale, self.page)

    def to_pgm(self) -> bytes:
        """Binary PGM, exactly as the real page serializes itself."""
        return ("P5\n%d %d\n255\n" % (self.width, self.height)).encode("ascii") + self.gray


class ScriptedPixelEngine(BaseEngine):
    """A :class:`BaseEngine` whose recognizer is a fixed list of pixel boxes."""

    name = "scripted-pixel"

    def __init__(self, words: List[PixelWord], available: bool = True) -> None:
        self._words = list(words)
        self._available = available
        self.calls = 0

    def available(self) -> bool:
        return self._available

    def _recognize_pixels(self, page, config):
        self.calls += 1
        return list(self._words)


class ExplodingEngine(BaseEngine):
    """A backend that fails the way a broken subprocess wrapper fails."""

    name = "exploding"

    def available(self) -> bool:
        return True

    def _recognize_pixels(self, page, config):
        raise RuntimeError("the vendor library exploded")


TSV_HEADER = "\t".join(
    [
        "level",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "word_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    ]
)


def tsv_row(level, block, line, left, top, width, height, conf, text):
    """Build one TSV row in Tesseract's column order."""
    return "\t".join(
        str(v)
        for v in [level, 1, block, 1, line, 1, left, top, width, height, conf, text]
    )


class TesseractTsvTests(unittest.TestCase):
    """The TSV parser is the whole Tesseract adapter; it is tested without Tesseract."""

    def test_parses_word_rows_into_pixel_boxes(self):
        text = "\n".join(
            [
                TSV_HEADER,
                tsv_row(1, 1, 0, 0, 0, 612, 792, -1, ""),
                tsv_row(5, 1, 1, 100, 100, 50, 20, 95.5, "Name"),
                tsv_row(5, 1, 1, 160, 100, 40, 20, 88, "Smith"),
            ]
        )
        words = parse_tesseract_tsv(text)
        self.assertEqual([w.text for w in words], ["Name", "Smith"])
        self.assertEqual(words[0].rect, Rect(100, 100, 150, 120))
        self.assertAlmostEqual(words[0].confidence, 0.955)
        self.assertAlmostEqual(words[1].confidence, 0.88)
        self.assertEqual(words[0].line_id, 1)
        self.assertEqual(words[0].block_id, 1)

    def test_drops_conf_minus_one_and_empty_text(self):
        text = "\n".join(
            [
                TSV_HEADER,
                tsv_row(5, 1, 1, 100, 100, 50, 20, -1, ""),
                tsv_row(5, 1, 1, 100, 100, 50, 20, -1, "ghost"),
                tsv_row(5, 1, 1, 100, 100, 50, 20, 90, "   "),
                tsv_row(5, 1, 1, 200, 100, 50, 20, 90, "real"),
            ]
        )
        words = parse_tesseract_tsv(text)
        self.assertEqual([w.text for w in words], ["real"])

    def test_malformed_rows_cost_only_themselves(self):
        text = "\n".join(
            [
                TSV_HEADER,
                "5\t1\t1",
                tsv_row(5, 1, 1, "x", 100, 50, 20, 90, "bad-left"),
                tsv_row(5, 1, 1, 100, 100, 0, 20, 90, "zero-width"),
                tsv_row(5, 1, 1, 300, 100, 50, 20, 90, "good"),
                "",
            ]
        )
        words = parse_tesseract_tsv(text)
        self.assertEqual([w.text for w in words], ["good"])

    def test_header_is_optional_and_non_word_levels_are_ignored(self):
        text = "\n".join(
            [
                tsv_row(4, 1, 1, 0, 0, 612, 40, -1, ""),
                tsv_row(5, 2, 3, 10, 20, 30, 40, 70, "Date"),
            ]
        )
        words = parse_tesseract_tsv(text)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].block_id, 2)
        self.assertEqual(words[0].line_id, 3)

    def test_empty_input_is_empty_output(self):
        self.assertEqual(parse_tesseract_tsv(""), [])
        self.assertEqual(parse_tesseract_tsv(TSV_HEADER), [])


class PaddleParsingTests(unittest.TestCase):
    """Paddle has changed its return shape across releases; all of them are accepted."""

    LINE = [[[10.0, 20.0], [70.0, 22.0], [70.0, 44.0], [10.0, 42.0]], ("Name", 0.97)]

    def test_parses_the_nested_per_image_shape(self):
        words = parse_paddle_result([[self.LINE]])
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].text, "Name")
        self.assertAlmostEqual(words[0].confidence, 0.97)
        self.assertEqual(words[0].rect, Rect(10, 20, 70, 44))

    def test_parses_the_flat_shape(self):
        words = parse_paddle_result([self.LINE])
        self.assertEqual([w.text for w in words], ["Name"])

    def test_parses_the_prediction_dict_shape(self):
        result = [
            {
                "rec_texts": ["Name", "Smith"],
                "rec_scores": [0.9, 0.5],
                "dt_polys": [
                    [[0, 0], [10, 0], [10, 5], [0, 5]],
                    [[20, 0], [30, 0], [30, 5], [20, 5]],
                ],
            }
        ]
        words = parse_paddle_result(result)
        self.assertEqual([w.text for w in words], ["Name", "Smith"])
        self.assertEqual(words[1].rect, Rect(20, 0, 30, 5))

    def test_junk_is_skipped_rather_than_raised(self):
        self.assertEqual(parse_paddle_result(None), [])
        self.assertEqual(parse_paddle_result([None]), [])
        self.assertEqual(parse_paddle_result("not a result"), [])
        self.assertEqual(parse_paddle_result([[["not", "a", "box"], ("x", 1.0)]]), [])
        self.assertEqual(parse_paddle_result([[self.LINE[0], ("   ", 0.9)]]), [])


class CoordinateConversionTests(unittest.TestCase):
    """Pixel space must not escape this package, and must be converted exactly once."""

    def test_pixel_boxes_become_user_space_boxes(self):
        engine = ScriptedPixelEngine(
            [PixelWord(text="Name", rect=Rect(100, 100, 200, 140), confidence=0.9)]
        )
        page = StubPage(1224, 1584, scale=2.0)
        words = engine.recognize(page, LETTER, OcrConfig())
        self.assertEqual(len(words), 1)
        # scale 2, origin at the crop box: x = px/2, y = 792 - py/2.
        self.assertEqual(words[0].rect, Rect(50.0, 722.0, 100.0, 742.0))
        self.assertTrue(LETTER.crop_box.contains_rect(words[0].rect))
        self.assertEqual(words[0].page, 0)
        self.assertAlmostEqual(words[0].confidence, 0.9)

    def test_conversion_happens_exactly_once(self):
        # A second application of the same transform would land near the page origin;
        # asserting the literal single-conversion result catches that.
        engine = ScriptedPixelEngine(
            [PixelWord(text="A", rect=Rect(100, 100, 200, 140), confidence=0.5)]
        )
        page = StubPage(1224, 1584, scale=2.0)
        rect = engine.recognize(page, LETTER, OcrConfig())[0].rect
        twice = LETTER.pixel_rect_to_user(rect, 2.0)
        self.assertNotEqual(rect, twice)
        self.assertEqual(rect, Rect(50.0, 722.0, 100.0, 742.0))

    def test_rotation_is_honoured_by_the_shared_conversion(self):
        engine = ScriptedPixelEngine(
            [PixelWord(text="A", rect=Rect(100, 100, 200, 140), confidence=0.9)]
        )
        page = StubPage(1584, 1224, scale=2.0)
        words = engine.recognize(page, LETTER_90, OcrConfig())
        self.assertEqual(words[0].rect, Rect(50.0, 50.0, 70.0, 100.0))
        self.assertTrue(LETTER_90.crop_box.contains_rect(words[0].rect))

    def test_crop_box_origin_is_applied(self):
        geometry = PageGeometry(
            index=0, media_box=Rect(0, 0, 612, 792), crop_box=Rect(20, 30, 592, 762)
        )
        engine = ScriptedPixelEngine(
            [PixelWord(text="A", rect=Rect(0, 0, 20, 20), confidence=0.9)]
        )
        words = engine.recognize(StubPage(1144, 1464), geometry, OcrConfig())
        # x = 20 + px/2, y = 762 - py/2.
        self.assertEqual(words[0].rect, Rect(20.0, 752.0, 30.0, 762.0))

    def test_boxes_are_clamped_into_the_page(self):
        engine = ScriptedPixelEngine(
            [PixelWord(text="A", rect=Rect(-50, -50, 40, 40), confidence=0.9)]
        )
        words = engine.recognize(StubPage(1224, 1584), LETTER, OcrConfig())
        self.assertTrue(LETTER.crop_box.contains_rect(words[0].rect))
        self.assertEqual(words[0].rect.y1, 792.0)

    def test_blank_words_and_out_of_range_confidence_are_normalized(self):
        engine = ScriptedPixelEngine(
            [
                PixelWord(text="   ", rect=Rect(0, 0, 10, 10), confidence=0.9),
                PixelWord(text=" A ", rect=Rect(0, 0, 10, 10), confidence=1.7),
                PixelWord(text="B", rect=Rect(0, 0, 10, 10), confidence=-3.0),
            ]
        )
        words = engine.recognize(StubPage(100, 100), LETTER, OcrConfig())
        self.assertEqual([w.text for w in words], ["A", "B"])
        self.assertEqual(words[0].confidence, 1.0)
        self.assertEqual(words[1].confidence, 0.0)


class RobustnessTests(unittest.TestCase):
    """A failing backend produces fewer words, never an exception."""

    def test_backend_exception_yields_no_words(self):
        words = ExplodingEngine().recognize(StubPage(100, 100), LETTER, OcrConfig())
        self.assertEqual(words, [])

    def test_unavailable_engine_yields_no_words(self):
        engine = ScriptedPixelEngine(
            [PixelWord(text="A", rect=Rect(0, 0, 10, 10), confidence=0.9)], available=False
        )
        self.assertEqual(engine.recognize(StubPage(100, 100), LETTER, OcrConfig()), [])
        self.assertEqual(engine.calls, 0)

    def test_empty_raster_yields_no_words(self):
        engine = ScriptedPixelEngine(
            [PixelWord(text="A", rect=Rect(0, 0, 10, 10), confidence=0.9)]
        )
        self.assertEqual(engine.recognize(StubPage(0, 0), LETTER, OcrConfig()), [])
        self.assertEqual(engine.calls, 0)

    def test_a_zfp_config_is_accepted_where_an_ocr_config_is_expected(self):
        from zfp.core.config import ZfpConfig

        self.assertIsInstance(resolve_ocr_config(ZfpConfig.default()), OcrConfig)
        self.assertIsInstance(resolve_ocr_config(None), OcrConfig)
        cfg = OcrConfig()
        self.assertIs(resolve_ocr_config(cfg), cfg)


class NullEngineTests(unittest.TestCase):
    """The engine that is always there so the pipeline always runs."""

    def test_null_engine_is_available_and_silent(self):
        engine = NullEngine()
        self.assertEqual(engine.name, "null")
        self.assertTrue(engine.available())
        self.assertEqual(engine.recognize(StubPage(100, 100), LETTER, OcrConfig()), [])

    def test_null_engine_satisfies_the_protocol(self):
        self.assertIsInstance(NullEngine(), OcrEngine)


class RegistryTests(unittest.TestCase):
    """Discovery must be honest on a machine with nothing installed."""

    def tearDown(self):
        unregister_engine("registry-test")
        clear_engine_cache()

    def test_builtin_engines_are_registered_in_cascade_order(self):
        self.assertEqual(engine_names(), ["tesseract", "paddle", "null"])

    def test_available_engines_always_includes_null(self):
        names = available_engines()
        self.assertIn("null", names)
        self.assertEqual(names, [n for n in engine_names() if n in names])

    def test_available_engines_agrees_with_the_probes(self):
        names = available_engines()
        expect_tesseract = have("pytesseract") or shutil.which("tesseract") is not None
        self.assertEqual("tesseract" in names, expect_tesseract)
        self.assertEqual("paddle" in names, have("paddleocr"))

    def test_get_engine_is_memoized_and_case_insensitive(self):
        first = get_engine("null")
        self.assertIs(first, get_engine("NULL"))
        clear_engine_cache()
        self.assertIsNot(first, get_engine("null"))

    def test_unknown_engine_raises_a_zfp_error(self):
        with self.assertRaises(UnsupportedFeatureError):
            get_engine("no-such-engine")

    def test_register_and_unregister(self):
        register_engine("registry-test", NullEngine)
        self.assertIn("registry-test", engine_names())
        self.assertIsInstance(get_engine("registry-test"), NullEngine)
        self.assertTrue(unregister_engine("registry-test"))
        self.assertFalse(unregister_engine("registry-test"))
        with self.assertRaises(UnsupportedFeatureError):
            get_engine("registry-test")

    def test_register_validates_its_arguments(self):
        with self.assertRaises(ValidationError):
            register_engine("", NullEngine)
        with self.assertRaises(ValidationError):
            register_engine("registry-test", "not callable")

    def test_a_broken_factory_becomes_an_unsupported_feature(self):
        def factory():
            raise RuntimeError("no model files")

        register_engine("registry-test", factory)
        with self.assertRaises(UnsupportedFeatureError):
            get_engine("registry-test")
        # Discovery still succeeds; the broken engine is simply not available.
        self.assertNotIn("registry-test", available_engines())


class TesseractEngineTests(unittest.TestCase):
    """Everything that can be checked without the binary installed."""

    def test_availability_matches_the_environment(self):
        engine = TesseractEngine()
        expected = have("pytesseract") or shutil.which("tesseract") is not None
        self.assertEqual(engine.available(), expected)

    def test_language_argument_joins_configured_languages(self):
        engine = TesseractEngine()
        self.assertEqual(engine._language(OcrConfig(languages=["eng", "deu"])), "eng+deu")
        self.assertEqual(engine._language(OcrConfig(languages=[])), "eng")

    @unittest.skipIf(have("pytesseract"), "pytesseract present; the binary path is bypassed")
    def test_a_missing_binary_yields_no_words_and_no_exception(self):
        engine = TesseractEngine(binary="zfp-tesseract-does-not-exist")
        self.assertIsNone(engine.binary_path())
        self.assertFalse(engine.available())
        self.assertEqual(engine.recognize(StubPage(100, 100), LETTER, OcrConfig()), [])


class PaddleEngineTests(unittest.TestCase):
    """Everything that can be checked without paddleocr installed."""

    def test_availability_matches_the_environment(self):
        self.assertEqual(PaddleEngine().available(), have("paddleocr"))

    def test_language_codes_are_mapped_to_paddles_spelling(self):
        engine = PaddleEngine()
        self.assertEqual(engine._language(OcrConfig(languages=["eng"])), "en")
        self.assertEqual(engine._language(OcrConfig(languages=["fra"])), "fr")
        self.assertEqual(engine._language(OcrConfig(languages=["nld"])), "nld")

    @unittest.skipIf(have("paddleocr"), "paddleocr present")
    def test_missing_paddle_yields_no_words(self):
        engine = PaddleEngine()
        self.assertEqual(engine.recognize(StubPage(100, 100), LETTER, OcrConfig()), [])
        self.assertIsNone(engine._ocr_object(OcrConfig()))


class RenderedPageIntegrationTests(unittest.TestCase):
    """The engines are written against the real RenderedPage, so prove they fit it."""

    def real_page(self):
        try:
            from zfp.raster.render import RenderedPage
        except Exception as exc:  # pragma: no cover - the renderer is another module
            self.skipTest("zfp.raster.render unavailable: %s" % exc)
        return RenderedPage(
            page=0, width=8, height=4, scale=2.0, gray=b"\xff" * 32, backend="test"
        )

    def test_a_real_rendered_page_converts_the_same_way(self):
        page = self.real_page()
        engine = ScriptedPixelEngine(
            [PixelWord(text="A", rect=Rect(2, 0, 6, 2), confidence=0.8)]
        )
        words = engine.recognize(page, LETTER, OcrConfig())
        self.assertEqual(words[0].rect, Rect(1.0, 791.0, 3.0, 792.0))

    def test_pgm_serialization_is_what_the_binary_path_feeds_tesseract(self):
        page = self.real_page()
        self.assertTrue(page.to_pgm().startswith(b"P5\n8 4\n255\n"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
