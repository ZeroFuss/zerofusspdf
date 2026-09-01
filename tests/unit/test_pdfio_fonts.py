"""Unit tests for :mod:`zfp.pdfio.fonts` and :mod:`zfp.pdfio._afm_data`.

The width tables are the part of ZFP that cannot be "nearly right": every appearance
stream position is derived from them, so the published Adobe anchors are asserted
directly rather than compared against a golden file that could drift with the code.
"""

from __future__ import annotations

import unittest

from zfp.core.errors import ValidationError
from zfp.core.geometry import Rect
from zfp.pdfio import _afm_data, fonts
from zfp.pdfio.document import Document
from zfp.pdfio.objects import PdfDict, PdfName, PdfRef

WINANSI_FACES = (
    "Helvetica",
    "Helvetica-Bold",
    "Helvetica-Oblique",
    "Helvetica-BoldOblique",
    "Times-Roman",
    "Times-Bold",
    "Times-Italic",
    "Times-BoldItalic",
    "Courier",
    "Courier-Bold",
    "Courier-Oblique",
    "Courier-BoldOblique",
)


def code(ch: str) -> int:
    """WinAnsi code point of a single ASCII/Latin-1 character."""
    return ord(ch)


# --------------------------------------------------------------------------------------
# Published metrics
# --------------------------------------------------------------------------------------


class AfmAnchorTests(unittest.TestCase):
    """The published Adobe advance widths every other assertion leans on."""

    def test_helvetica_anchor_widths(self):
        w = fonts.widths_for("Helvetica")
        self.assertEqual(w[code(" ")], 278)
        self.assertEqual(w[code("A")], 667)
        self.assertEqual(w[code("a")], 556)
        self.assertEqual(w[code("W")], 944)
        self.assertEqual(w[code("i")], 222)
        self.assertEqual(w[code(".")], 278)

    def test_times_roman_anchor_widths(self):
        w = fonts.widths_for("Times-Roman")
        self.assertEqual(w[code(" ")], 250)
        self.assertEqual(w[code("A")], 722)
        self.assertEqual(w[code("a")], 444)
        self.assertEqual(w[code("W")], 944)

    def test_helvetica_bold_anchor_widths(self):
        w = fonts.widths_for("Helvetica-Bold")
        self.assertEqual(w[code("A")], 722)
        self.assertEqual(w[code(" ")], 278)

    def test_courier_is_exactly_fixed_pitch(self):
        for face in ("Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique"):
            widths = fonts.widths_for(face)
            self.assertEqual(set(widths.values()), {600}, face)
            self.assertEqual(len(widths), 224, face)
            self.assertTrue(fonts.is_fixed_pitch(face), face)

    def test_only_courier_is_fixed_pitch(self):
        for face in _afm_data.BASE_FONTS:
            expected = face.startswith("Courier")
            self.assertEqual(fonts.is_fixed_pitch(face), expected, face)

    def test_oblique_cuts_share_upright_advances(self):
        # The oblique faces are a shear of the upright, not a redraw: identical advances.
        self.assertEqual(fonts.widths_for("Helvetica-Oblique"), fonts.widths_for("Helvetica"))
        self.assertEqual(
            fonts.widths_for("Helvetica-BoldOblique"), fonts.widths_for("Helvetica-Bold")
        )
        self.assertEqual(fonts.widths_for("Courier-Oblique"), fonts.widths_for("Courier"))

    def test_italic_cuts_do_not_share_upright_advances(self):
        # Times-Italic is a separate design, so its metrics must differ from Times-Roman.
        self.assertNotEqual(fonts.widths_for("Times-Italic"), fonts.widths_for("Times-Roman"))
        self.assertEqual(fonts.widths_for("Times-Italic")[code("A")], 611)
        self.assertEqual(fonts.widths_for("Times-BoldItalic")[code("A")], 667)
        self.assertEqual(fonts.widths_for("Times-Bold")[code("A")], 722)

    def test_winansi_faces_cover_every_code_from_32_to_255(self):
        for face in WINANSI_FACES:
            widths = fonts.widths_for(face)
            self.assertEqual(len(widths), 224, face)
            self.assertEqual(min(widths), 32, face)
            self.assertEqual(max(widths), 255, face)

    def test_winansi_high_range_anchors(self):
        w = fonts.widths_for("Helvetica")
        self.assertEqual(w[0xA0], 278)     # no-break space is a space
        self.assertEqual(w[0xAD], 333)     # soft hyphen is a hyphen
        self.assertEqual(w[0x80], 556)     # Euro
        self.assertEqual(w[0x97], 1000)    # em dash
        self.assertEqual(w[0xE9], 556)     # eacute matches 'e'
        self.assertEqual(w[0xC4], 667)     # Adieresis matches 'A'
        self.assertEqual(w[0xF8], 611)     # oslash is wider than 'o' in Helvetica
        self.assertEqual(w[0x7F], 350)     # unused codes draw a bullet (Annex D.2)

    def test_symbol_and_zapf_use_their_own_encodings(self):
        symbol = fonts.widths_for("Symbol")
        zapf = fonts.widths_for("ZapfDingbats")
        self.assertEqual(symbol[32], 250)
        self.assertEqual(symbol[code("a")], 631)      # alpha
        self.assertEqual(symbol[code("W")], 768)      # Omega
        self.assertEqual(zapf[32], 278)
        self.assertEqual(zapf[code("4")], 846)        # a20, the AcroForm check mark
        self.assertEqual(zapf[code("8")], 677)        # a24, the cross
        # Both encodings have holes; the WinAnsi faces do not.
        self.assertNotIn(130, symbol)
        self.assertNotIn(255, symbol)
        self.assertNotIn(240, zapf)

    def test_vertical_metrics(self):
        helv = _afm_data.metrics_for("Helvetica")
        self.assertEqual(helv.ascender, 718.0)
        self.assertEqual(helv.descender, -207.0)
        self.assertEqual(helv.cap_height, 718.0)
        self.assertEqual(helv.x_height, 523.0)
        self.assertEqual(helv.font_bbox, (-166.0, -225.0, 1000.0, 931.0))
        self.assertEqual(helv.italic_angle, 0.0)

        times = _afm_data.metrics_for("Times-Roman")
        self.assertEqual(times.ascender, 683.0)
        self.assertEqual(times.descender, -217.0)
        self.assertEqual(times.cap_height, 662.0)
        self.assertEqual(times.x_height, 450.0)

        self.assertEqual(_afm_data.metrics_for("Helvetica-Oblique").italic_angle, -12.0)
        self.assertEqual(_afm_data.metrics_for("Times-Italic").italic_angle, -15.5)
        self.assertEqual(_afm_data.metrics_for("Courier").font_bbox[0], -6.0)

    def test_pi_fonts_publish_no_stem_metrics(self):
        for face in ("Symbol", "ZapfDingbats"):
            m = _afm_data.metrics_for(face)
            self.assertIsNone(m.ascender, face)
            self.assertIsNone(m.descender, face)
            self.assertIsNone(m.cap_height, face)
            self.assertIsNone(m.x_height, face)

    def test_metrics_for_rejects_unknown_names(self):
        with self.assertRaises(ValidationError):
            _afm_data.metrics_for("Helv")           # an alias, not a canonical name
        self.assertTrue(_afm_data.has_metrics("Helvetica"))
        self.assertFalse(_afm_data.has_metrics("Wingdings"))

    def test_all_fourteen_faces_are_present(self):
        self.assertEqual(len(_afm_data.BASE_FONTS), 14)
        self.assertEqual(len(set(_afm_data.BASE_FONTS)), 14)
        for name in _afm_data.BASE_FONTS:
            self.assertEqual(_afm_data.metrics_for(name).name, name)
            self.assertIn(name, fonts.RESOURCE_NAMES)

    def test_glyph_name_table_lines_up_with_the_codes(self):
        names = _afm_data.WIN_ANSI_GLYPH_NAMES
        self.assertEqual(len(names), 224)
        self.assertEqual(names[0], "space")
        self.assertEqual(names[code("A") - 32], "A")
        self.assertEqual(names[0x80 - 32], "Euro")
        self.assertEqual(names[0xFF - 32], "ydieresis")

    def test_widths_for_returns_an_independent_copy(self):
        first = fonts.widths_for("Helvetica")
        first[code("A")] = 1
        self.assertEqual(fonts.widths_for("Helvetica")[code("A")], 667)


# --------------------------------------------------------------------------------------
# Name resolution
# --------------------------------------------------------------------------------------


class ResolveBaseFontTests(unittest.TestCase):
    """Every font name a PDF can name has to land on one of the fourteen faces."""

    def test_documented_aliases(self):
        cases = {
            "Helv": "Helvetica",
            "Helvetica": "Helvetica",
            "Arial": "Helvetica",
            "HeBo": "Helvetica-Bold",
            "Helvetica-Bold": "Helvetica-Bold",
            "Arial-Bold": "Helvetica-Bold",
            "Arial,Bold": "Helvetica-Bold",
            "TiRo": "Times-Roman",
            "Times": "Times-Roman",
            "Times-Roman": "Times-Roman",
            "TimesNewRoman": "Times-Roman",
            "Cour": "Courier",
            "Courier": "Courier",
            "CourierNew": "Courier",
            "ZaDb": "ZapfDingbats",
            "ZapfDingbats": "ZapfDingbats",
            "Symb": "Symbol",
            "Symbol": "Symbol",
        }
        for alias, expected in cases.items():
            self.assertEqual(fonts.resolve_base_font(alias), expected, alias)

    def test_every_family_has_its_four_style_variants(self):
        cases = {
            "Arial,BoldItalic": "Helvetica-BoldOblique",
            "Arial-Italic": "Helvetica-Oblique",
            "Helvetica-BoldItalic": "Helvetica-BoldOblique",
            "TimesNewRoman,Bold": "Times-Bold",
            "TimesNewRoman,Italic": "Times-Italic",
            "TimesNewRoman,BoldItalic": "Times-BoldItalic",
            "CourierNew,Bold": "Courier-Bold",
            "CourierNew,Italic": "Courier-Oblique",
            "CourierNew,BoldItalic": "Courier-BoldOblique",
        }
        for alias, expected in cases.items():
            self.assertEqual(fonts.resolve_base_font(alias), expected, alias)

    def test_normalization_is_case_and_separator_insensitive(self):
        for name in ("helvetica", "HELVETICA", "Helvetica ", " helv ", "Helvetica-Regular"):
            self.assertEqual(fonts.resolve_base_font(name), "Helvetica", name)
        self.assertEqual(fonts.resolve_base_font("times new roman"), "Times-Roman")

    def test_subset_prefixes_are_stripped(self):
        self.assertEqual(fonts.resolve_base_font("ABCDEF+Helvetica"), "Helvetica")
        self.assertEqual(fonts.resolve_base_font("BAAAAA+Arial,BoldItalic"), "Helvetica-BoldOblique")

    def test_style_inference_for_unknown_families(self):
        self.assertEqual(fonts.resolve_base_font("Calibri-Bold"), "Helvetica-Bold")
        self.assertEqual(fonts.resolve_base_font("Verdana-Italic"), "Helvetica-Oblique")
        self.assertEqual(fonts.resolve_base_font("Consolas"), "Courier")
        self.assertEqual(fonts.resolve_base_font("Georgia-Bold"), "Times-Bold")
        self.assertEqual(fonts.resolve_base_font("SomeFont-Black"), "Helvetica-Bold")

    def test_unknown_names_fall_back_to_helvetica(self):
        for name in ("Wingdings", "", "   ", "!!!", None, 42):
            self.assertEqual(fonts.resolve_base_font(name), "Helvetica", repr(name))

    def test_pi_fonts_have_no_style_variants(self):
        self.assertEqual(fonts.resolve_base_font("Symbol-Bold"), "Symbol")
        self.assertEqual(fonts.resolve_base_font("ZapfDingbats-Italic"), "ZapfDingbats")

    def test_resolution_is_idempotent(self):
        for name in _afm_data.BASE_FONTS:
            self.assertEqual(fonts.resolve_base_font(fonts.resolve_base_font(name)), name)

    def test_resource_names(self):
        self.assertEqual(fonts.resource_name("Helvetica"), "Helv")
        self.assertEqual(fonts.resource_name("Arial,Bold"), "HeBo")
        self.assertEqual(fonts.resource_name("Times"), "TiRo")
        self.assertEqual(fonts.resource_name("CourierNew"), "Cour")
        self.assertEqual(fonts.resource_name("ZapfDingbats"), "ZaDb")
        self.assertEqual(fonts.resource_name("Symbol"), "Symb")
        self.assertEqual(len(set(fonts.RESOURCE_NAMES.values())), 14)

    def test_standard_14_maps_only_onto_canonical_names(self):
        for alias, canonical in fonts.STANDARD_14.items():
            self.assertIn(canonical, _afm_data.BASE_FONTS, alias)


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


class TextWidthTests(unittest.TestCase):
    """Advance measurement, the input to every fit decision."""

    def test_empty_text_is_zero(self):
        self.assertEqual(fonts.text_width("", "Helvetica", 12.0), 0.0)
        self.assertEqual(fonts.text_width("", "Courier", 100.0), 0.0)

    def test_single_glyph_matches_the_afm_table(self):
        self.assertAlmostEqual(fonts.text_width("A", "Helvetica", 1000.0), 667.0, places=6)
        self.assertAlmostEqual(fonts.text_width("A", "Helvetica", 12.0), 667 * 12 / 1000.0, places=9)
        self.assertAlmostEqual(fonts.text_width("W", "Times-Roman", 10.0), 9.44, places=9)

    def test_width_is_additive_and_linear_in_size(self):
        one = fonts.text_width("Hello, world", "Helvetica", 10.0)
        two = fonts.text_width("Hello, world", "Helvetica", 20.0)
        self.assertAlmostEqual(two, 2 * one, places=9)
        self.assertAlmostEqual(
            fonts.text_width("ab", "Helvetica", 10.0),
            fonts.text_width("a", "Helvetica", 10.0) + fonts.text_width("b", "Helvetica", 10.0),
            places=9,
        )

    def test_courier_width_depends_only_on_length(self):
        for text in ("iiii", "WWWW", "12.4"):
            self.assertAlmostEqual(fonts.text_width(text, "Courier", 10.0), 4 * 6.0, places=9)

    def test_aliases_measure_identically_to_canonical_names(self):
        self.assertEqual(
            fonts.text_width("Sample 123", "Helv", 11.0),
            fonts.text_width("Sample 123", "Helvetica", 11.0),
        )

    def test_accented_and_high_winansi_characters_measure(self):
        # 'e' and 'eacute' share an advance in Helvetica, so the accent costs nothing.
        self.assertEqual(
            fonts.text_width("café", "Helvetica", 10.0),
            fonts.text_width("cafe", "Helvetica", 10.0),
        )
        self.assertGreater(fonts.text_width("€", "Helvetica", 10.0), 0.0)

    def test_out_of_encoding_characters_use_the_average_advance(self):
        average = _afm_data.metrics_for("Helvetica").average_width
        self.assertAlmostEqual(
            fonts.text_width("中", "Helvetica", 10.0), average / 100.0, places=9
        )

    def test_control_characters_carry_no_advance_but_tab_is_a_space(self):
        self.assertEqual(fonts.text_width("\n", "Helvetica", 12.0), 0.0)
        self.assertEqual(fonts.text_width("a\nb", "Helvetica", 12.0),
                         fonts.text_width("ab", "Helvetica", 12.0))
        self.assertEqual(fonts.text_width("\t", "Helvetica", 12.0),
                         fonts.text_width(" ", "Helvetica", 12.0))

    def test_char_widths_length_and_sum(self):
        text = "Comb 123"
        widths = fonts.char_widths(text, "Helvetica", 9.0)
        self.assertEqual(len(widths), len(text))
        self.assertAlmostEqual(sum(widths), fonts.text_width(text, "Helvetica", 9.0), places=9)
        self.assertEqual(fonts.char_widths("", "Helvetica", 9.0), [])

    def test_char_widths_are_uniform_for_courier(self):
        widths = fonts.char_widths("aWi.", "Courier", 10.0)
        self.assertEqual(widths, [6.0, 6.0, 6.0, 6.0])

    def test_measure_lines(self):
        lines = ["ab", "abcd", "a"]
        width, height = fonts.measure_lines(lines, "Helvetica", 10.0)
        self.assertAlmostEqual(width, fonts.text_width("abcd", "Helvetica", 10.0), places=9)
        self.assertAlmostEqual(height, 3 * 10.0 * fonts.LEADING_FACTOR, places=9)

    def test_measure_lines_on_an_empty_block(self):
        self.assertEqual(fonts.measure_lines([], "Helvetica", 10.0), (0.0, 0.0))
        self.assertEqual(fonts.measure_lines([""], "Helvetica", 10.0), (0.0, 11.6))

    def test_font_ascent_and_descent(self):
        self.assertEqual(fonts.font_ascent("Helvetica"), 718.0)
        self.assertEqual(fonts.font_descent("Helvetica"), -207.0)
        self.assertEqual(fonts.font_ascent("TiRo"), 683.0)
        self.assertEqual(fonts.font_descent("Times-Roman"), -217.0)
        self.assertEqual(fonts.font_ascent("Courier"), 629.0)
        self.assertLess(fonts.font_descent("Courier"), 0.0)

    def test_pi_fonts_fall_back_to_the_documented_vertical_metrics(self):
        for face in ("Symbol", "ZapfDingbats"):
            self.assertEqual(fonts.font_ascent(face), fonts.FALLBACK_ASCENDER, face)
            self.assertEqual(fonts.font_descent(face), fonts.FALLBACK_DESCENDER, face)


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


class FitFontSizeTests(unittest.TestCase):
    """Choosing the largest size a value can be drawn at inside a known blank."""

    BOX = Rect(0.0, 0.0, 200.0, 20.0)

    def _fits(self, text, size, rect, padding=2.0):
        box = rect.normalized()
        if size * fonts.LEADING_FACTOR > box.height - 2 * padding + 1e-6:
            return False
        return fonts.text_width(text, "Helvetica", size) <= box.width - 2 * padding + 1e-6

    def test_short_text_gets_the_maximum(self):
        self.assertEqual(fonts.fit_font_size("Jane", "Helvetica", self.BOX), 12.0)

    def test_long_text_is_shrunk(self):
        long_value = "Jane Q. Public of 1234 Long Winding Street, Springfield"
        size = fonts.fit_font_size(long_value, "Helvetica", self.BOX)
        self.assertLess(size, 12.0)
        self.assertGreaterEqual(size, 4.0)
        self.assertTrue(self._fits(long_value, size, self.BOX))

    def test_result_never_exceeds_max_size(self):
        for text in ("", "i", "Jane", "a" * 50):
            for max_size in (6.0, 12.0, 30.0):
                size = fonts.fit_font_size(text, "Helvetica", self.BOX, max_size=max_size)
                self.assertLessEqual(size, max_size, (text, max_size))

    def test_result_never_drops_below_min_size(self):
        tiny = Rect(0.0, 0.0, 6.0, 4.0)
        self.assertEqual(fonts.fit_font_size("x" * 200, "Helvetica", tiny), 4.0)
        self.assertEqual(fonts.fit_font_size("x" * 200, "Helvetica", tiny, min_size=7.0), 7.0)

    def test_degenerate_rectangles_return_min_size(self):
        self.assertEqual(fonts.fit_font_size("a", "Helvetica", Rect(0.0, 0.0, 0.0, 0.0)), 4.0)
        self.assertEqual(fonts.fit_font_size("a", "Helvetica", Rect(5.0, 5.0, 5.0, 5.0)), 4.0)

    def test_height_alone_can_be_the_binding_constraint(self):
        flat = Rect(0.0, 0.0, 400.0, 10.0)
        size = fonts.fit_font_size("Hi", "Helvetica", flat)
        self.assertAlmostEqual(size, 5.17, places=2)
        self.assertTrue(self._fits("Hi", size, flat))
        self.assertFalse(self._fits("Hi", round(size + 0.01, 2), flat))

    def test_empty_text_is_bounded_by_height_only(self):
        # 10pt of width could not hold a single 12pt glyph, but empty text has none.
        self.assertEqual(fonts.fit_font_size("", "Helvetica", Rect(0.0, 0.0, 10.0, 40.0)), 12.0)

    def test_a_rect_narrower_than_its_padding_fits_nothing(self):
        self.assertEqual(fonts.fit_font_size("", "Helvetica", Rect(0.0, 0.0, 1.0, 40.0)), 4.0)

    def test_padding_is_applied_to_both_sides(self):
        loose = fonts.fit_font_size("Some value here", "Helvetica", self.BOX, padding=0.0)
        tight = fonts.fit_font_size("Some value here", "Helvetica", self.BOX, padding=6.0)
        self.assertGreaterEqual(loose, tight)

    def test_result_is_rounded_to_two_decimals_and_deterministic(self):
        text = "Determinism matters here"
        first = fonts.fit_font_size(text, "Helvetica", Rect(0.0, 0.0, 97.0, 13.0))
        second = fonts.fit_font_size(text, "Helvetica", Rect(0.0, 0.0, 97.0, 13.0))
        self.assertEqual(first, second)
        self.assertEqual(first, round(first, 2))

    def test_answer_is_maximal_on_the_hundredth_grid(self):
        text = "Maximal fit"
        box = Rect(0.0, 0.0, 80.0, 40.0)
        size = fonts.fit_font_size(text, "Helvetica", box, max_size=40.0)
        self.assertTrue(self._fits(text, size, box))
        self.assertFalse(self._fits(text, round(size + 0.01, 2), box))

    def test_unnormalized_rectangles_are_handled(self):
        flipped = Rect(200.0, 20.0, 0.0, 0.0)
        self.assertEqual(
            fonts.fit_font_size("Jane", "Helvetica", flipped),
            fonts.fit_font_size("Jane", "Helvetica", self.BOX),
        )

    def test_swapped_bounds_are_tolerated(self):
        size = fonts.fit_font_size("Jane", "Helvetica", self.BOX, max_size=4.0, min_size=12.0)
        self.assertEqual(size, 12.0)

    def test_wider_font_yields_a_smaller_size(self):
        text = "Comparative widths"
        box = Rect(0.0, 0.0, 90.0, 30.0)
        helvetica = fonts.fit_font_size(text, "Helvetica", box, max_size=30.0)
        courier = fonts.fit_font_size(text, "Courier", box, max_size=30.0)
        self.assertLess(courier, helvetica)


# --------------------------------------------------------------------------------------
# Wrapping
# --------------------------------------------------------------------------------------


class WrapTextTests(unittest.TestCase):
    """Greedy word wrap for multiline fields."""

    def test_explicit_newlines_are_honoured(self):
        self.assertEqual(fonts.wrap_text("a\nb", "Helvetica", 10.0, 200.0), ["a", "b"])
        self.assertEqual(fonts.wrap_text("a\r\nb", "Helvetica", 10.0, 200.0), ["a", "b"])
        self.assertEqual(fonts.wrap_text("a\rb", "Helvetica", 10.0, 200.0), ["a", "b"])

    def test_blank_lines_survive(self):
        self.assertEqual(fonts.wrap_text("a\n\nb", "Helvetica", 10.0, 200.0), ["a", "", "b"])
        self.assertEqual(fonts.wrap_text("", "Helvetica", 10.0, 200.0), [""])

    def test_every_word_is_preserved(self):
        text = "The quick brown fox jumps over the lazy dog near the riverbank"
        lines = fonts.wrap_text(text, "Helvetica", 10.0, 70.0)
        self.assertGreater(len(lines), 1)
        self.assertEqual(" ".join(lines).split(), text.split())

    def test_words_are_preserved_across_explicit_breaks(self):
        text = "Name: Jane Public\nAddress: 1234 Long Winding Street\n\nNotes: none"
        lines = fonts.wrap_text(text, "Helvetica", 9.0, 60.0)
        self.assertEqual(" ".join(lines).split(), text.split())

    def test_lines_fit_the_requested_width(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        width = 80.0
        for line in fonts.wrap_text(text, "Helvetica", 10.0, width):
            self.assertLessEqual(fonts.text_width(line, "Helvetica", 10.0), width + 1e-6, line)

    def test_greedy_packing_puts_as_much_as_possible_on_each_line(self):
        lines = fonts.wrap_text("one two three four five six", "Helvetica", 10.0, 45.0)
        self.assertEqual(lines[0], "one two")
        for index, line in enumerate(lines[:-1]):
            following = lines[index + 1].split()[0]
            candidate = line + " " + following
            self.assertGreater(fonts.text_width(candidate, "Helvetica", 10.0), 45.0)

    def test_a_single_long_word_is_hard_split_without_loss(self):
        word = "supercalifragilisticexpialidocious"
        lines = fonts.wrap_text(word, "Helvetica", 12.0, 20.0)
        self.assertGreater(len(lines), 1)
        self.assertEqual("".join(lines), word)

    def test_hard_split_makes_progress_on_an_impossibly_narrow_line(self):
        lines = fonts.wrap_text("WWWW", "Helvetica", 12.0, 1.0)
        self.assertEqual(lines, ["W", "W", "W", "W"])

    def test_leading_and_trailing_whitespace_is_kept(self):
        lines = fonts.wrap_text("   indented", "Helvetica", 10.0, 200.0)
        self.assertEqual(lines, ["   indented"])
        self.assertEqual(fonts.wrap_text("tail   ", "Helvetica", 10.0, 200.0), ["tail   "])
        self.assertEqual(fonts.wrap_text("   ", "Helvetica", 10.0, 200.0), ["   "])

    def test_interior_whitespace_runs_survive_when_they_fit(self):
        self.assertEqual(fonts.wrap_text("a    b", "Helvetica", 10.0, 200.0), ["a    b"])

    def test_non_positive_width_returns_the_paragraphs_unwrapped(self):
        self.assertEqual(fonts.wrap_text("a b\nc d", "Helvetica", 10.0, 0.0), ["a b", "c d"])
        self.assertEqual(fonts.wrap_text("a b", "Helvetica", 10.0, -5.0), ["a b"])

    def test_wrapping_is_deterministic(self):
        text = "repeatable output for identical inputs every single time"
        first = fonts.wrap_text(text, "Times-Roman", 11.0, 90.0)
        second = fonts.wrap_text(text, "Times-Roman", 11.0, 90.0)
        self.assertEqual(first, second)

    def test_wrapped_block_measures_back(self):
        text = "one two three four five six seven eight nine ten"
        lines = fonts.wrap_text(text, "Helvetica", 10.0, 60.0)
        width, height = fonts.measure_lines(lines, "Helvetica", 10.0)
        self.assertLessEqual(width, 60.0 + 1e-6)
        self.assertAlmostEqual(height, len(lines) * 10.0 * fonts.LEADING_FACTOR, places=9)


# --------------------------------------------------------------------------------------
# Escaping
# --------------------------------------------------------------------------------------


class EscapePdfTextTests(unittest.TestCase):
    """Turning a value into the operand of a ``(...) Tj``."""

    def test_plain_ascii_passes_through(self):
        self.assertEqual(fonts.escape_pdf_text("Jane Public"), b"Jane Public")
        self.assertIsInstance(fonts.escape_pdf_text("x"), bytes)
        self.assertEqual(fonts.escape_pdf_text(""), b"")

    def test_delimiters_are_escaped(self):
        self.assertEqual(fonts.escape_pdf_text("a(b)c"), b"a\\(b\\)c")
        self.assertEqual(fonts.escape_pdf_text("back\\slash"), b"back\\\\slash")
        self.assertEqual(fonts.escape_pdf_text("(("), b"\\(\\(")

    def test_control_characters_use_the_named_escapes(self):
        self.assertEqual(fonts.escape_pdf_text("a\nb"), b"a\\nb")
        self.assertEqual(fonts.escape_pdf_text("a\rb"), b"a\\rb")
        self.assertEqual(fonts.escape_pdf_text("a\tb"), b"a\\tb")
        self.assertEqual(fonts.escape_pdf_text("a\bb"), b"a\\bb")
        self.assertEqual(fonts.escape_pdf_text("a\fb"), b"a\\fb")

    def test_other_control_characters_become_octal(self):
        self.assertEqual(fonts.escape_pdf_text("\x01"), b"\\001")
        self.assertEqual(fonts.escape_pdf_text("\x1b"), b"\\033")

    def test_high_winansi_characters_become_octal(self):
        self.assertEqual(fonts.escape_pdf_text("café"), b"caf\\351")
        self.assertEqual(fonts.escape_pdf_text("€"), b"\\200")      # Euro -> 0x80
        self.assertEqual(fonts.escape_pdf_text("—"), b"\\227")      # em dash -> 0x97

    def test_output_stays_printable_ascii(self):
        escaped = fonts.escape_pdf_text("Zurück — café (100%) \\ done\n")
        for byte in escaped:
            self.assertTrue(32 <= byte < 127, escaped)

    def test_unrepresentable_characters_degrade_to_a_question_mark(self):
        self.assertEqual(fonts.escape_pdf_text("a中b"), b"a?b")

    def test_unicode_punctuation_falls_back_to_winansi(self):
        self.assertEqual(fonts.escape_pdf_text("a‑b"), b"a-b")      # NB hyphen
        self.assertEqual(fonts.escape_pdf_text("a b"), b"a b")      # thin space
        self.assertEqual(fonts.escape_pdf_text("a​b"), b"ab")       # zero width space
        self.assertEqual(fonts.escape_pdf_text("ﬁ"), b"fi")

    def test_curly_quotes_are_real_winansi_codes(self):
        self.assertEqual(fonts.escape_pdf_text("‘’“”"), b"\\221\\222\\223\\224")


# --------------------------------------------------------------------------------------
# Resource installation
# --------------------------------------------------------------------------------------


class EnsureStandardFontTests(unittest.TestCase):
    """Installing a base-14 font into the AcroForm's default resources."""

    def _blank(self):
        return Document.from_pages_blank(1)

    def _dr_fonts(self, doc):
        acroform = doc.acroform()
        self.assertIsNotNone(acroform)
        resources = doc.resolve(acroform.get("DR"))
        self.assertIsInstance(resources, PdfDict)
        fonts_dict = doc.resolve(resources.get("Font"))
        self.assertIsInstance(fonts_dict, PdfDict)
        return fonts_dict

    def test_installs_a_helvetica_resource(self):
        doc = self._blank()
        name, ref = fonts.ensure_standard_font(doc, "Helvetica")
        self.assertEqual(name, "Helv")
        self.assertIsInstance(ref, PdfRef)

        font_dict = doc.resolve(self._dr_fonts(doc).get("Helv"))
        self.assertEqual(font_dict.get_name("Type"), "Font")
        self.assertEqual(font_dict.get_name("Subtype"), "Type1")
        self.assertEqual(font_dict.get_name("BaseFont"), "Helvetica")
        self.assertEqual(font_dict.get_name("Encoding"), "WinAnsiEncoding")

    def test_is_idempotent(self):
        doc = self._blank()
        first_name, first_ref = fonts.ensure_standard_font(doc, "Helvetica")
        objects_after_first = len(doc.writer.updates)
        second_name, second_ref = fonts.ensure_standard_font(doc, "Helv")
        self.assertEqual(first_name, second_name)
        self.assertEqual(first_ref, second_ref)
        self.assertEqual(len(doc.writer.updates), objects_after_first)
        self.assertEqual(list(self._dr_fonts(doc).keys()), ["Helv"])

    def test_aliases_share_one_resource(self):
        doc = self._blank()
        fonts.ensure_standard_font(doc, "Arial")
        fonts.ensure_standard_font(doc, "ArialMT")
        fonts.ensure_standard_font(doc, "ABCDEF+Helvetica")
        self.assertEqual(list(self._dr_fonts(doc).keys()), ["Helv"])

    def test_several_faces_coexist(self):
        doc = self._blank()
        installed = {}
        for face in ("Helvetica", "Helvetica-Bold", "Times-Roman", "Courier", "ZapfDingbats"):
            short, ref = fonts.ensure_standard_font(doc, face)
            installed[short] = ref
        self.assertEqual(sorted(installed), ["Cour", "HeBo", "Helv", "TiRo", "ZaDb"])
        self.assertEqual(len(set(installed.values())), 5)

    def test_pi_fonts_keep_their_builtin_encoding(self):
        doc = self._blank()
        for face, short in (("ZapfDingbats", "ZaDb"), ("Symbol", "Symb")):
            name, _ref = fonts.ensure_standard_font(doc, face)
            self.assertEqual(name, short)
            font_dict = doc.resolve(self._dr_fonts(doc).get(short))
            self.assertNotIn("Encoding", font_dict)
            self.assertEqual(font_dict.get_name("BaseFont"), face)

    def test_default_argument_installs_helvetica(self):
        doc = self._blank()
        self.assertEqual(fonts.ensure_standard_font(doc)[0], "Helv")

    def test_unknown_font_names_install_helvetica(self):
        doc = self._blank()
        name, _ref = fonts.ensure_standard_font(doc, "Wingdings")
        self.assertEqual(name, "Helv")
        self.assertEqual(
            doc.resolve(self._dr_fonts(doc).get("Helv")).get_name("BaseFont"), "Helvetica"
        )

    def test_change_survives_an_incremental_save(self):
        doc = self._blank()
        fonts.ensure_standard_font(doc, "Helvetica")
        fonts.ensure_standard_font(doc, "ZapfDingbats")
        data = doc.to_bytes(incremental=True)

        reopened = Document.open(data)
        font_dict = self._dr_fonts(reopened)
        self.assertEqual(sorted(font_dict.keys()), ["Helv", "ZaDb"])
        helv = reopened.resolve(font_dict.get("Helv"))
        self.assertEqual(helv.get_name("BaseFont"), "Helvetica")
        self.assertEqual(helv.get_name("Encoding"), "WinAnsiEncoding")
        zadb = reopened.resolve(font_dict.get("ZaDb"))
        self.assertEqual(zadb.get_name("BaseFont"), "ZapfDingbats")
        self.assertNotIn("Encoding", zadb)

    def test_reuses_an_existing_matching_resource(self):
        doc = self._blank()
        acroform = doc.ensure_acroform()
        existing = PdfDict(
            {
                "Type": PdfName("Font"),
                "Subtype": PdfName("Type1"),
                "BaseFont": PdfName("Helvetica"),
                "Encoding": PdfName("WinAnsiEncoding"),
            }
        )
        existing_ref = doc.writer.add_object(existing)
        doc.resolve(acroform.get("DR"))["Font"] = PdfDict({"Helv": existing_ref})

        name, ref = fonts.ensure_standard_font(doc, "Helvetica")
        self.assertEqual(name, "Helv")
        self.assertEqual(ref, existing_ref)

    def test_replaces_a_resource_that_names_a_different_face(self):
        doc = self._blank()
        acroform = doc.ensure_acroform()
        wrong = doc.writer.add_object(
            PdfDict({"Type": PdfName("Font"), "BaseFont": PdfName("Times-Roman")})
        )
        doc.resolve(acroform.get("DR"))["Font"] = PdfDict({"Helv": wrong})

        name, ref = fonts.ensure_standard_font(doc, "Helvetica")
        self.assertEqual(name, "Helv")
        self.assertNotEqual(ref, wrong)
        self.assertEqual(doc.resolve(ref).get_name("BaseFont"), "Helvetica")

    def test_creates_dr_and_font_dictionaries_when_absent(self):
        doc = self._blank()
        acroform = doc.ensure_acroform()
        acroform.pop("DR", None)
        name, _ref = fonts.ensure_standard_font(doc, "Times-Bold")
        self.assertEqual(name, "TiBo")
        self.assertIn("TiBo", self._dr_fonts(doc))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
