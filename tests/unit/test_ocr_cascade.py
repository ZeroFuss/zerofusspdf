"""Unit tests for :mod:`zfp.ocr.cascade` and :mod:`zfp.ocr.suspects`.

No OCR engine is installed here, so the cascade is driven by a ``FakeEngine`` that
satisfies the :class:`~zfp.ocr.engine.OcrEngine` protocol, counts its calls and returns a
script of pre-built words.  That is not a workaround for the missing engines: the cascade
*is* a control-flow decision -- which engine runs, whether the next one runs at all,
whether crops are re-recognized -- and a call counter is the only honest way to assert
"the second engine was never touched".

The first test in the file is the one the module exists for: a page whose PDF already
contains text must never reach an engine.
"""

from __future__ import annotations

import math
import unittest
from typing import Any, List, Sequence

from zfp.core.config import OcrConfig, ZfpConfig
from zfp.core.errors import ValidationError
from zfp.core.geometry import PageGeometry, Rect
from zfp.core.types import RasterWord, TextSpan
from zfp.ocr.cascade import (
    ENGINE_DISABLED,
    ENGINE_NONE,
    SKIPPED_NATIVE_TEXT,
    OcrResult,
    crop_geometry,
    group_words_into_lines,
    has_native_text,
    mean_confidence,
    merge_words,
    ocr_cascade,
    recognize_regions,
    words_to_spans,
    words_to_word_spans,
)
from zfp.ocr.engine import (
    NullEngine,
    OcrEngine,
    clear_engine_cache,
    register_engine,
    unregister_engine,
)
from zfp.ocr.suspects import (
    REASON_LOW_CONFIDENCE,
    REASON_MIXED_ALNUM,
    Suspect,
    apply_correction,
    confidence_report,
    find_suspects,
    is_implausible_mix,
    suggest_alternatives,
)

LETTER = PageGeometry(
    index=0, media_box=Rect(0, 0, 612, 792), crop_box=Rect(0, 0, 612, 792), rotation=0
)


class StubPage:
    """Stand-in for :class:`zfp.raster.render.RenderedPage` (see test_ocr_engine.py)."""

    def __init__(self, width: int = 1224, height: int = 1584, scale: float = 2.0, page: int = 0):
        self.page = page
        self.width = width
        self.height = height
        self.scale = scale
        self.gray = b"\xff" * (width * height) if width * height <= 4096 else b""
        self.backend = "stub"

    def crop(self, rect_px: Rect) -> "StubPage":
        rect = rect_px.normalized()
        x0 = max(0, int(math.floor(rect.x0)))
        y0 = max(0, int(math.floor(rect.y0)))
        x1 = min(self.width, int(math.ceil(rect.x1)))
        y1 = min(self.height, int(math.ceil(rect.y1)))
        if x1 <= x0 or y1 <= y0:
            return StubPage(1, 1, self.scale, self.page)
        return StubPage(x1 - x0, y1 - y0, self.scale, self.page)

    def to_pgm(self) -> bytes:
        return ("P5\n%d %d\n255\n" % (self.width, self.height)).encode("ascii") + self.gray


class FakeEngine:
    """A scripted OCR engine: call *n* returns ``script[n]`` (then nothing).

    It implements the protocol structurally rather than subclassing
    :class:`~zfp.ocr.engine.BaseEngine`, which keeps the cascade tests honest about what
    the cascade actually requires of an engine.
    """

    def __init__(self, name: str, script: Sequence[Sequence[RasterWord]], available: bool = True):
        self.name = name
        self.script = [list(batch) for batch in script]
        self.available_flag = available
        self.calls = 0
        self.pages: List[Any] = []
        self.geometries: List[PageGeometry] = []

    def available(self) -> bool:
        return self.available_flag

    def recognize(self, page, geometry, config) -> List[RasterWord]:
        index = self.calls
        self.calls += 1
        self.pages.append(page)
        self.geometries.append(geometry)
        if index < len(self.script):
            return list(self.script[index])
        return []


class RaisingEngine:
    """An engine that throws, the way a half-installed vendor wrapper does."""

    name = "raising"

    def __init__(self):
        self.calls = 0

    def available(self) -> bool:
        return True

    def recognize(self, page, geometry, config):
        self.calls += 1
        raise RuntimeError("model files are missing")


def word(text: str, x0: float, y0: float, conf: float, width: float = 40.0, page: int = 0):
    """Build a user-space OCR word 12 pt tall at ``(x0, y0)``."""
    return RasterWord(text=text, rect=Rect(x0, y0, x0 + width, y0 + 12.0), confidence=conf, page=page)


def config(**kwargs) -> OcrConfig:
    """An OcrConfig with the scripted engines and the contract's default thresholds."""
    base = dict(
        enabled=True,
        dpi=300,
        engines=["fake"],
        min_word_confidence=0.55,
        escalate_below=0.70,
        languages=["eng"],
    )
    base.update(kwargs)
    return OcrConfig(**base)


class CascadeTestCase(unittest.TestCase):
    """Base class that registers scripted engines and always cleans them up."""

    def setUp(self):
        self.page = StubPage()
        self.registered: List[str] = []

    def tearDown(self):
        for name in self.registered:
            unregister_engine(name)
        clear_engine_cache()

    def install(self, engine) -> Any:
        """Register ``engine`` under its own name for the duration of one test."""
        register_engine(engine.name, lambda: engine)
        self.registered.append(engine.name)
        return engine


class NativeTextShortCircuitTests(CascadeTestCase):
    """The most important branch in the module: never OCR what the PDF already has."""

    def test_native_spans_skip_ocr_entirely(self):
        fake = self.install(FakeEngine("fake", [[word("never", 10, 10, 0.99)]]))
        spans = [
            TextSpan(text="Applicant Name:", rect=Rect(72, 700, 200, 712), page=0),
            TextSpan(text="Date of Birth:", rect=Rect(72, 680, 190, 692), page=0),
        ]
        result = ocr_cascade(self.page, LETTER, config(), native_spans=spans)

        self.assertEqual(result.engine, SKIPPED_NATIVE_TEXT)
        self.assertEqual(result.words, [])
        self.assertEqual(result.suspects, [])
        self.assertFalse(result.escalated)
        self.assertEqual(result.mean_confidence, 1.0)
        self.assertEqual(fake.calls, 0, "an engine ran on a page that already had text")
        self.assertEqual(result.per_engine, {})
        self.assertTrue(any("never OCR" in line for line in result.report))

    def test_blank_native_spans_do_not_count_as_text(self):
        fake = self.install(FakeEngine("fake", [[word("Smith", 100, 700, 0.9)]]))
        spans = [TextSpan(text="   ", rect=Rect(0, 0, 10, 10), page=0)]
        result = ocr_cascade(self.page, LETTER, config(), native_spans=spans)

        self.assertEqual(result.engine, "fake")
        self.assertEqual(fake.calls, 1)

    def test_no_native_spans_at_all_runs_ocr(self):
        fake = self.install(FakeEngine("fake", [[word("Smith", 100, 700, 0.9)]]))
        for spans in (None, []):
            fake.calls = 0
            result = ocr_cascade(self.page, LETTER, config(), native_spans=spans)
            self.assertEqual(result.engine, "fake")
            self.assertEqual(fake.calls, 1)

    def test_has_native_text_predicate(self):
        self.assertFalse(has_native_text(None))
        self.assertFalse(has_native_text([]))
        self.assertFalse(has_native_text([TextSpan(text="\t \n", rect=Rect(0, 0, 1, 1), page=0)]))
        self.assertTrue(has_native_text([TextSpan(text="x", rect=Rect(0, 0, 1, 1), page=0)]))
        self.assertFalse(
            has_native_text([TextSpan(text="ab", rect=Rect(0, 0, 1, 1), page=0)], min_chars=5)
        )


class EngineOrderTests(CascadeTestCase):
    """Rung two: run engines in order, stop at the first good enough one."""

    def test_high_confidence_first_engine_short_circuits_the_second(self):
        first = self.install(
            FakeEngine("fake", [[word("Name", 100, 700, 0.9), word("Smith", 150, 700, 0.92)]])
        )
        second = self.install(FakeEngine("fake2", [[word("Other", 100, 600, 0.99)]]))

        result = ocr_cascade(self.page, LETTER, config(engines=["fake", "fake2"]))

        self.assertEqual(result.engine, "fake")
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 0, "the second engine ran despite a confident first")
        self.assertFalse(result.escalated)
        self.assertEqual(result.per_engine, {"fake": 2})
        self.assertAlmostEqual(result.mean_confidence, 0.91)
        self.assertTrue(any("accepted" in line for line in result.report))

    def test_low_confidence_first_engine_escalates_to_the_second(self):
        first = self.install(
            FakeEngine("fake", [[word("Nome", 100, 700, 0.40), word("Smth", 150, 700, 0.42)]])
        )
        second = self.install(
            FakeEngine("fake2", [[word("Name", 100, 700, 0.95), word("Smith", 150, 700, 0.93)]])
        )

        result = ocr_cascade(self.page, LETTER, config(engines=["fake", "fake2"]))

        self.assertEqual(result.engine, "fake2")
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertEqual([w.text for w in result.words], ["Name", "Smith"])
        self.assertEqual(result.per_engine, {"fake": 2, "fake2": 2})
        self.assertTrue(
            any("escalating" in line and "'fake'" in line for line in result.report),
            result.report,
        )
        # Falling through to a better engine is not the crop escalation.
        self.assertFalse(result.escalated)

    def test_the_best_engine_wins_when_none_clears_the_bar(self):
        self.install(FakeEngine("fake", [[word("a", 100, 700, 0.30)]]))
        self.install(FakeEngine("fake2", [[word("b", 100, 700, 0.60)], []]))

        result = ocr_cascade(self.page, LETTER, config(engines=["fake", "fake2"]))

        self.assertEqual(result.engine, "fake2")
        self.assertTrue(result.escalated)

    def test_an_engine_returning_nothing_does_not_win(self):
        empty = self.install(FakeEngine("fake", [[]]))
        good = self.install(FakeEngine("fake2", [[word("Name", 100, 700, 0.9)]]))

        result = ocr_cascade(self.page, LETTER, config(engines=["fake", "fake2"]))

        self.assertEqual(result.engine, "fake2")
        self.assertEqual(empty.calls, 1)
        self.assertEqual(good.calls, 1)
        self.assertEqual(result.per_engine["fake"], 0)
        self.assertTrue(any("no words" in line for line in result.report))

    def test_unknown_engine_names_are_reported_and_stepped_over(self):
        self.install(FakeEngine("fake", [[word("Name", 100, 700, 0.9)]]))
        result = ocr_cascade(self.page, LETTER, config(engines=["no-such-engine", "fake"]))

        self.assertEqual(result.engine, "fake")
        self.assertTrue(any("not registered" in line for line in result.report))

    def test_unavailable_engines_are_skipped(self):
        absent = self.install(FakeEngine("fake", [[word("x", 1, 1, 0.99)]], available=False))
        present = self.install(FakeEngine("fake2", [[word("Name", 100, 700, 0.9)]]))

        result = ocr_cascade(self.page, LETTER, config(engines=["fake", "fake2"]))

        self.assertEqual(result.engine, "fake2")
        self.assertEqual(absent.calls, 0)
        self.assertEqual(present.calls, 1)
        self.assertTrue(any("unavailable" in line for line in result.report))

    def test_an_engine_that_raises_is_contained(self):
        raising = self.install(RaisingEngine())
        good = self.install(FakeEngine("fake", [[word("Name", 100, 700, 0.9)]]))

        result = ocr_cascade(self.page, LETTER, config(engines=["raising", "fake"]))

        self.assertEqual(raising.calls, 1)
        self.assertEqual(good.calls, 1)
        self.assertEqual(result.engine, "fake")
        self.assertTrue(any("raised" in line for line in result.report))

    def test_disabled_ocr_runs_nothing(self):
        fake = self.install(FakeEngine("fake", [[word("x", 1, 1, 0.9)]]))
        result = ocr_cascade(self.page, LETTER, config(enabled=False))

        self.assertEqual(result.engine, ENGINE_DISABLED)
        self.assertEqual(fake.calls, 0)
        self.assertEqual(result.words, [])

    def test_no_configured_engines_is_reported_not_raised(self):
        result = ocr_cascade(self.page, LETTER, config(engines=[]))
        self.assertEqual(result.engine, ENGINE_NONE)
        self.assertTrue(any("no OCR engines" in line for line in result.report))

    def test_a_zfp_config_is_accepted(self):
        self.install(FakeEngine("fake", [[word("Name", 100, 700, 0.9)]]))
        cfg = ZfpConfig.default()
        cfg.ocr.engines = ["fake"]
        result = ocr_cascade(self.page, LETTER, cfg)
        self.assertEqual(result.engine, "fake")

    def test_the_null_engine_produces_an_honest_empty_result(self):
        result = ocr_cascade(self.page, LETTER, config(engines=["null"]))
        self.assertEqual(result.engine, NullEngine.name)
        self.assertEqual(result.words, [])
        self.assertEqual(result.mean_confidence, 0.0)
        self.assertFalse(result.escalated)
        self.assertTrue(any("no words" in line for line in result.report))


class SuspectSplitTests(CascadeTestCase):
    """Low-confidence words are kept and quarantined, never silently dropped."""

    def test_words_below_min_confidence_land_in_suspects(self):
        self.install(
            FakeEngine(
                "fake",
                [
                    [
                        word("Name", 100, 700, 0.95),
                        word("Smith", 150, 700, 0.95),
                        word("l23", 100, 680, 0.30),
                    ]
                ],
            )
        )
        result = ocr_cascade(self.page, LETTER, config())

        self.assertEqual([w.text for w in result.words], ["Name", "Smith"])
        self.assertEqual([w.text for w in result.suspects], ["l23"])
        self.assertFalse(result.escalated, "mean 0.73 clears the bar; no crops needed")
        # The mean describes the page, so it includes the suspect.
        self.assertAlmostEqual(result.mean_confidence, (0.95 + 0.95 + 0.30) / 3.0)
        self.assertEqual(len(result.all_words), 3)


class EscalationTests(CascadeTestCase):
    """Rung four: re-recognize the regions the engine was unsure about."""

    def build(self):
        low = [word("l23", 100, 700, 0.40), word("Smth", 100, 500, 0.50, width=60)]
        improved = [RasterWord(text="123", rect=low[0].rect, confidence=0.95, page=0)]
        engine = self.install(FakeEngine("fake", [low, improved, []]))
        return engine, ocr_cascade(self.page, LETTER, config())

    def test_low_confidence_triggers_cropped_re_recognition(self):
        engine, result = self.build()

        self.assertTrue(result.escalated)
        self.assertEqual(engine.calls, 3, "one full page plus one crop per low word")
        self.assertEqual([w.text for w in result.words], ["123"])
        self.assertEqual([w.text for w in result.suspects], ["Smth"])
        self.assertEqual(result.per_engine, {"fake": 2, "fake+crops": 1})
        self.assertTrue(any("cropped re-recognition" in line for line in result.report))
        self.assertTrue(any("re-rasterized at a higher dpi" in line for line in result.report))

    def test_the_replaced_reading_survives_as_an_alternative(self):
        _engine, result = self.build()
        self.assertEqual(result.words[0].alternatives, [("l23", 0.4)])

    def test_crops_are_smaller_rasters_with_their_own_geometry(self):
        engine, _result = self.build()
        full_page, first_crop = engine.pages[0], engine.pages[1]
        self.assertEqual((full_page.width, full_page.height), (1224, 1584))
        # 40 pt wide, 12 pt tall, padded by 4 pt, at 2 px/pt.
        self.assertEqual((first_crop.width, first_crop.height), (96, 40))
        self.assertEqual(first_crop.scale, full_page.scale)

        crop_box = engine.geometries[1].crop_box
        self.assertEqual(crop_box, Rect(96, 696, 144, 716))
        self.assertTrue(LETTER.crop_box.contains_rect(crop_box))

    def test_a_worse_reading_is_recorded_but_does_not_win(self):
        low = [word("Nome", 100, 700, 0.40)]
        worse = [RasterWord(text="N0me", rect=low[0].rect, confidence=0.20, page=0)]
        self.install(FakeEngine("fake", [low, worse]))

        result = ocr_cascade(self.page, LETTER, config(min_word_confidence=0.1))

        self.assertEqual([w.text for w in result.words], ["Nome"])
        self.assertEqual(result.words[0].alternatives, [("N0me", 0.2)])


class RecognizeRegionsTests(CascadeTestCase):
    """The crop helper on its own: user space in, user space out."""

    def test_each_region_becomes_one_crop(self):
        engine = FakeEngine("fake", [[], [], []])
        rects = [Rect(100, 700, 140, 712), Rect(100, 500, 160, 512)]
        recognize_regions(engine, self.page, LETTER, rects, config())
        self.assertEqual(engine.calls, 2)

    def test_overlapping_regions_are_cropped_once(self):
        engine = FakeEngine("fake", [[], []])
        rects = [Rect(100, 700, 140, 712), Rect(138, 700, 180, 712)]
        recognize_regions(engine, self.page, LETTER, rects, config())
        self.assertEqual(engine.calls, 1)
        self.assertEqual(engine.geometries[0].crop_box, Rect(96, 696, 184, 716))

    def test_regions_off_the_page_are_dropped(self):
        engine = FakeEngine("fake", [[]])
        recognize_regions(engine, self.page, LETTER, [Rect(-100, -100, -50, -50)], config())
        self.assertEqual(engine.calls, 0)

    def test_the_region_ceiling_is_honoured(self):
        engine = FakeEngine("fake", [[]] * 10)
        rects = [Rect(100, 100 + 40 * i, 140, 112 + 40 * i) for i in range(10)]
        recognize_regions(engine, self.page, LETTER, rects, config(), max_regions=3)
        self.assertEqual(engine.calls, 3)

    def test_a_failing_engine_costs_only_that_crop(self):
        engine = RaisingEngine()
        out = recognize_regions(engine, self.page, LETTER, [Rect(100, 700, 140, 712)], config())
        self.assertEqual(out, [])
        self.assertEqual(engine.calls, 1)


class CropGeometryTests(unittest.TestCase):
    """A crop's geometry must place its words where they really are, at any rotation."""

    def test_crop_pixels_map_to_the_same_user_points_for_every_rotation(self):
        scale = 1.5
        box = Rect(40, 60, 200, 180)
        for rotation in (0, 90, 180, 270):
            geometry = PageGeometry(
                index=0,
                media_box=Rect(0, 0, 612, 792),
                crop_box=Rect(20, 30, 592, 762),
                rotation=rotation,
            )
            sub = crop_geometry(geometry, box, scale)
            self.assertEqual(sub.rotation, geometry.rotation)
            for px, py in ((40, 60), (100, 90), (200, 180), (137, 121)):
                expected = geometry.pixel_to_user(px, py, scale)
                actual = sub.pixel_to_user(px - box.x0, py - box.y0, scale)
                self.assertAlmostEqual(actual.x, expected.x, places=6)
                self.assertAlmostEqual(actual.y, expected.y, places=6)

    def test_the_sub_geometry_covers_exactly_the_cropped_region(self):
        geometry = PageGeometry(index=0, media_box=Rect(0, 0, 612, 792), crop_box=Rect(0, 0, 612, 792))
        sub = crop_geometry(geometry, Rect(192, 152, 288, 192), 2.0)
        self.assertEqual(sub.crop_box, Rect(96, 696, 144, 716))


class MergeTests(unittest.TestCase):
    """Merging re-recognized words back in, by position."""

    def test_higher_confidence_wins_and_the_loser_becomes_an_alternative(self):
        base = [word("l23", 100, 700, 0.4)]
        better = [RasterWord(text="123", rect=base[0].rect, confidence=0.9, page=0)]
        merged = merge_words(base, better)
        self.assertEqual([w.text for w in merged], ["123"])
        self.assertEqual(merged[0].alternatives, [("l23", 0.4)])

    def test_inputs_are_never_mutated(self):
        base = [word("l23", 100, 700, 0.4)]
        better = [RasterWord(text="123", rect=base[0].rect, confidence=0.9, page=0)]
        merge_words(base, better)
        self.assertEqual(base[0].text, "l23")
        self.assertEqual(base[0].alternatives, [])

    def test_an_unmatched_word_is_appended(self):
        base = [word("Name", 100, 700, 0.9)]
        extra = [word("Smith", 300, 400, 0.8)]
        merged = merge_words(base, extra)
        self.assertEqual([w.text for w in merged], ["Name", "Smith"])

    def test_words_from_another_page_never_match(self):
        base = [word("Name", 100, 700, 0.4)]
        other = [word("Nome", 100, 700, 0.9, page=1)]
        merged = merge_words(base, other)
        self.assertEqual(len(merged), 2)


class SpanTests(unittest.TestCase):
    """OCR words become the same TextSpan shape the native parser produces."""

    def test_two_words_on_one_line_become_one_span(self):
        words = [word("Name", 100, 700, 0.9), word("Smith", 150, 700, 0.7, width=50)]
        spans = words_to_spans(words)

        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].text, "Name Smith")
        self.assertEqual(spans[0].rect, Rect(100, 700, 200, 712))
        self.assertEqual(spans[0].source, "ocr")
        self.assertAlmostEqual(spans[0].confidence, 0.7, msg="a line is as good as its worst word")
        self.assertEqual(spans[0].glyph_rects, [w.rect for w in words])
        self.assertEqual(spans[0].page, 0)
        self.assertAlmostEqual(spans[0].font_size, 12.0)

    def test_two_lines_become_two_spans_in_reading_order(self):
        words = [
            word("second", 100, 680, 0.9),
            word("first", 100, 700, 0.9),
        ]
        spans = words_to_spans(words)
        self.assertEqual([s.text for s in spans], ["first", "second"])

    def test_words_are_ordered_left_to_right_inside_a_line(self):
        words = [word("Smith", 150, 700, 0.9), word("Name", 100, 700, 0.9)]
        self.assertEqual(words_to_spans(words)[0].text, "Name Smith")

    def test_mixed_heights_still_share_a_line(self):
        tall = RasterWord(text="H", rect=Rect(100, 700, 110, 716), confidence=0.9, page=0)
        short = RasterWord(text="o", rect=Rect(112, 702, 120, 710), confidence=0.9, page=0)
        self.assertEqual(len(group_words_into_lines([tall, short])), 1)

    def test_blank_words_are_not_spans(self):
        self.assertEqual(words_to_spans([word("   ", 100, 700, 0.9)]), [])
        self.assertEqual(words_to_spans([]), [])

    def test_pages_are_never_merged_into_one_line(self):
        words = [word("a", 100, 700, 0.9), word("b", 150, 700, 0.9, page=1)]
        self.assertEqual(len(words_to_spans(words)), 2)

    def test_word_spans_are_one_per_word_in_reading_order(self):
        words = [word("Smith", 150, 700, 0.7), word("Name", 100, 700, 0.9)]
        spans = words_to_word_spans(words)
        self.assertEqual([s.text for s in spans], ["Name", "Smith"])
        self.assertEqual([s.confidence for s in spans], [0.9, 0.7])
        self.assertEqual(spans[0].glyph_rects, [words[1].rect])
        self.assertTrue(all(s.source == "ocr" for s in spans))


class StatisticsTests(unittest.TestCase):
    """The mean is word-count weighted, which is what makes engines comparable."""

    def test_mean_confidence(self):
        self.assertEqual(mean_confidence([]), 0.0)
        self.assertAlmostEqual(
            mean_confidence([word("a", 0, 0, 0.9), word("b", 0, 0, 0.5)]), 0.7
        )

    def test_result_serializes(self):
        result = OcrResult(words=[word("a", 0, 0, 0.9)], engine="fake", mean_confidence=0.9)
        data = result.as_dict()
        self.assertEqual(data["engine"], "fake")
        self.assertEqual(data["words"][0]["text"], "a")
        self.assertEqual(data["report"], [])


class ConfusionTests(unittest.TestCase):
    """Confusable glyphs are wrong at high confidence, which is what makes them dangerous."""

    def test_the_all_digit_reading_comes_first(self):
        alternatives = suggest_alternatives("l23")
        self.assertEqual(alternatives[0], "123")
        self.assertIn("I23", alternatives)
        self.assertNotIn("l23", alternatives)

    def test_the_all_letter_reading_is_offered(self):
        self.assertIn("HELlO", suggest_alternatives("HEL1O"))
        self.assertIn("HEL10", suggest_alternatives("HEL1O"))

    def test_alternatives_are_bounded_and_deterministic(self):
        text = "l0S8" * 4
        first = suggest_alternatives(text)
        self.assertEqual(first, suggest_alternatives(text))
        self.assertLessEqual(len(first), 16)
        self.assertEqual(suggest_alternatives(""), [])
        self.assertEqual(suggest_alternatives("Name"), [])

    def test_substitutions_never_make_a_token_less_homogeneous(self):
        self.assertEqual(suggest_alternatives("plain"), ["pIain"])
        self.assertEqual(suggest_alternatives("12345"), [])

    def test_implausible_mixes(self):
        for text in ("l23", "HEL1O", "1O0"):
            self.assertTrue(is_implausible_mix(text), text)

    def test_plausible_mixes_are_left_alone(self):
        for text in ("A1", "3D", "F-150", "S5", "", "Smith", "12345", "N/A"):
            self.assertFalse(is_implausible_mix(text), text)


class FindSuspectsTests(unittest.TestCase):
    """Both kinds of suspect: what the engine doubted and what it should have."""

    def test_low_confidence_words_are_flagged(self):
        words = [word("Name", 100, 700, 0.9), word("Smth", 150, 700, 0.2)]
        suspects = find_suspects(words, config())
        self.assertEqual([s.word.text for s in suspects], ["Smth"])
        self.assertIn(REASON_LOW_CONFIDENCE, suspects[0].reason)
        self.assertEqual(suspects[0].index, 1)

    def test_confident_but_implausible_words_are_flagged(self):
        words = [word("l23", 100, 700, 0.99)]
        suspects = find_suspects(words, config())
        self.assertEqual(len(suspects), 1)
        self.assertEqual(suspects[0].reason, REASON_MIXED_ALNUM)
        self.assertEqual(suspects[0].alternatives[0], "123")

    def test_both_reasons_are_reported(self):
        suspects = find_suspects([word("l23", 100, 700, 0.2)], config())
        self.assertIn(REASON_LOW_CONFIDENCE, suspects[0].reason)
        self.assertIn(REASON_MIXED_ALNUM, suspects[0].reason)

    def test_engine_alternatives_come_before_generated_ones(self):
        item = word("l23", 100, 700, 0.2)
        item.alternatives = [("I23", 0.18)]
        suspects = find_suspects([item], config())
        self.assertEqual(suspects[0].alternatives[0], "I23")
        self.assertIn("123", suspects[0].alternatives)

    def test_clean_words_produce_no_suspects(self):
        self.assertEqual(find_suspects([word("Name", 100, 700, 0.99)], config()), [])

    def test_suspect_serializes(self):
        data = Suspect(word=word("l23", 1, 1, 0.2), alternatives=["123"], reason="x", index=0).as_dict()
        self.assertEqual(data["alternatives"], ["123"])
        self.assertEqual(data["word"]["text"], "l23")


class ApplyCorrectionTests(unittest.TestCase):
    """A correction is reversible and never mutates the list it was given."""

    def test_correction_replaces_the_text_and_keeps_the_old_reading(self):
        words = [word("Name", 100, 700, 0.9), word("l23", 150, 700, 0.4)]
        corrected = apply_correction(words, 1, "123")

        self.assertEqual([w.text for w in corrected], ["Name", "123"])
        self.assertEqual(corrected[1].alternatives, [("l23", 0.4)])
        self.assertAlmostEqual(corrected[1].confidence, 0.4, msg="a correction is not a re-scoring")
        self.assertEqual(corrected[1].rect, words[1].rect)
        self.assertEqual(words[1].text, "l23")
        self.assertEqual(words[1].alternatives, [])

    def test_a_no_op_correction_is_a_no_op(self):
        words = [word("Name", 100, 700, 0.9)]
        self.assertEqual(apply_correction(words, 0, "Name")[0].alternatives, [])

    def test_bad_arguments_raise_validation_errors(self):
        words = [word("Name", 100, 700, 0.9)]
        with self.assertRaises(ValidationError):
            apply_correction(words, 5, "x")
        with self.assertRaises(ValidationError):
            apply_correction(words, -1, "x")
        with self.assertRaises(ValidationError):
            apply_correction(words, 0, "   ")


class ConfidenceReportTests(unittest.TestCase):
    """The QA dashboard's view of one page."""

    def build(self) -> OcrResult:
        return OcrResult(
            words=[word("Name", 100, 700, 0.95), word("Smith", 150, 700, 0.75)],
            engine="fake",
            mean_confidence=0.7,
            escalated=True,
            suspects=[word("l23", 100, 680, 0.40)],
            per_engine={"fake": 3},
            report=["line one"],
        )

    def test_report_summarizes_the_distribution(self):
        data = confidence_report(self.build())
        self.assertEqual(data["engine"], "fake")
        self.assertTrue(data["escalated"])
        self.assertEqual(data["word_count"], 2)
        self.assertEqual(data["suspect_count"], 1)
        self.assertEqual(data["total_words"], 3)
        self.assertAlmostEqual(data["min_confidence"], 0.40)
        self.assertAlmostEqual(data["max_confidence"], 0.95)
        self.assertAlmostEqual(data["median_confidence"], 0.75)
        self.assertAlmostEqual(data["suspect_ratio"], 1.0 / 3.0)
        self.assertEqual(data["per_engine"], {"fake": 3})
        self.assertEqual(data["report"], ["line one"])

    def test_histogram_covers_every_word_exactly_once(self):
        data = confidence_report(self.build())
        self.assertEqual(sum(data["histogram"].values()), 3)
        self.assertEqual(data["histogram"]["0.9-1.0"], 1)
        self.assertEqual(data["histogram"]["0.7-0.8"], 1)
        self.assertEqual(data["histogram"]["0.4-0.5"], 1)
        self.assertEqual(len(data["histogram"]), 10)

    def test_worst_words_are_listed_worst_first(self):
        data = confidence_report(self.build())
        self.assertEqual(data["worst_words"][0]["text"], "l23")

    def test_an_empty_result_reports_zeros(self):
        data = confidence_report(OcrResult())
        self.assertEqual(data["total_words"], 0)
        self.assertEqual(data["min_confidence"], 0.0)
        self.assertEqual(sum(data["histogram"].values()), 0)
        self.assertEqual(data["suspect_ratio"], 0.0)


class ProtocolTests(unittest.TestCase):
    """The fakes really do satisfy the interface the cascade is written against."""

    def test_fake_engines_satisfy_the_protocol(self):
        self.assertIsInstance(FakeEngine("fake", []), OcrEngine)
        self.assertIsInstance(NullEngine(), OcrEngine)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
