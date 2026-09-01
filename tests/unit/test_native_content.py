"""Unit tests for :mod:`zfp.native.content` and :mod:`zfp.native.encoding`.

The interpreter is the only place ZFP learns where anything on a native page actually
is, so the assertions here are numeric: a span's box is compared against
:func:`zfp.pdfio.fonts.text_width`, a rule's rectangle against the coordinates written
into the stream, and a transformed span against the transform applied by hand.  A test
that only checked "one span came back" would let a silently wrong matrix through.

Content streams are hand-written byte strings on a blank document, so nothing here
depends on a fixture file or on another agent's module.
"""

from __future__ import annotations

import unittest
from typing import Any, List, Optional

from zfp.core.config import ZfpConfig
from zfp.core.geometry import Rect
from zfp.native import encoding
from zfp.native.content import (
    ContentResult,
    ContentState,
    ContentStreamInterpreter,
    analyze_page,
)
from zfp.pdfio import fonts
from zfp.pdfio.document import Document
from zfp.pdfio.objects import PdfArray, PdfDict, PdfName, PdfStream


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def make_page(stream: bytes, resources: Optional[PdfDict] = None):
    """Return ``(document, page)`` for a one-page PDF carrying ``stream``."""
    doc = Document.from_pages_blank(1)
    page = doc.page(0)
    doc.writer.set_object(page.dict["Contents"].num, PdfStream(PdfDict({}), stream))
    if resources is not None:
        page.dict["Resources"] = resources
        page.touch()
    return doc, page


def run(stream: bytes, resources: Optional[PdfDict] = None, config: Any = None) -> ContentResult:
    """Interpret ``stream`` on a blank Letter page."""
    doc, page = make_page(stream, resources)
    return ContentStreamInterpreter(page, doc, config).run()


def helvetica_font() -> PdfDict:
    """A minimal standard-14 font dictionary."""
    return PdfDict(
        {
            "Type": PdfName("Font"),
            "Subtype": PdfName("Type1"),
            "BaseFont": PdfName("Helvetica"),
            "Encoding": PdfName("WinAnsiEncoding"),
        }
    )


def font_resources(font: Optional[PdfDict] = None, name: str = "F1") -> PdfDict:
    """A ``/Resources`` dictionary holding one font under ``name``."""
    return PdfDict({"Font": PdfDict({name: font if font is not None else helvetica_font()})})


class _BrokenPage:
    """A page whose content cannot be read, for the failure path of ``run``."""

    index = 3

    def content_bytes(self) -> bytes:
        raise RuntimeError("content is unreadable")

    def resources(self) -> PdfDict:  # pragma: no cover - never reached
        return PdfDict()


class _ExplodingResources(PdfDict):
    """A resource dictionary that raises on every lookup."""

    def get(self, key: Any, default: Any = None) -> Any:
        raise RuntimeError("resource lookup exploded")


# --------------------------------------------------------------------------------------
# Text: the basic show operation
# --------------------------------------------------------------------------------------

class ShowTextTests(unittest.TestCase):
    def test_single_show_yields_one_span_with_exact_geometry(self):
        result = run(b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET", font_resources())
        self.assertEqual(len(result.spans), 1)
        span = result.spans[0]
        self.assertEqual(span.text, "Hello")
        self.assertEqual(span.page, 0)
        self.assertEqual(span.source, "native")
        self.assertAlmostEqual(span.baseline, 700.0, places=6)
        self.assertAlmostEqual(span.rect.x0, 100.0, places=6)
        expected = fonts.text_width("Hello", "Helvetica", 12.0)
        self.assertAlmostEqual(span.rect.width, expected, places=4)
        self.assertAlmostEqual(span.font_size, 12.0, places=6)
        self.assertEqual(span.font_name, "Helvetica")
        self.assertEqual(span.confidence, 1.0)

    def test_span_box_straddles_the_baseline(self):
        result = run(b"BT /F1 12 Tf 100 700 Td (Hg) Tj ET", font_resources())
        span = result.spans[0]
        ascent = fonts.font_ascent("Helvetica") / 1000.0 * 12.0
        descent = fonts.font_descent("Helvetica") / 1000.0 * 12.0
        self.assertAlmostEqual(span.rect.y1, 700.0 + ascent, places=4)
        self.assertAlmostEqual(span.rect.y0, 700.0 + descent, places=4)

    def test_glyph_rects_are_one_per_character_and_left_to_right(self):
        result = run(b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET", font_resources())
        span = result.spans[0]
        self.assertEqual(len(span.glyph_rects), len("Hello"))
        xs = [rect.x0 for rect in span.glyph_rects]
        self.assertEqual(xs, sorted(xs))
        widths = fonts.char_widths("Hello", "Helvetica", 12.0)
        self.assertAlmostEqual(span.glyph_rects[0].width, widths[0], places=4)
        self.assertAlmostEqual(span.glyph_rects[1].x0, 100.0 + widths[0], places=4)

    def test_no_font_selected_still_produces_geometry(self):
        result = run(b"BT 12 Tf 100 700 Td (Hi) Tj ET")
        self.assertEqual(len(result.spans), 1)
        self.assertEqual(result.spans[0].text, "Hi")

    def test_empty_string_emits_nothing(self):
        result = run(b"BT /F1 12 Tf 100 700 Td () Tj ET", font_resources())
        self.assertEqual(result.spans, [])

    def test_empty_page_is_empty(self):
        result = run(b"")
        self.assertEqual(result.spans, [])
        self.assertEqual(result.primitives, [])
        self.assertEqual(result.op_count, 0)
        self.assertEqual(result.errors, 0)


# --------------------------------------------------------------------------------------
# Text: positioning and the text state
# --------------------------------------------------------------------------------------

class TextPositioningTests(unittest.TestCase):
    def test_tm_scale_scales_the_span(self):
        plain = run(b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET", font_resources()).spans[0]
        scaled = run(
            b"BT /F1 12 Tf 2 0 0 2 100 700 Tm (Hello) Tj ET", font_resources()
        ).spans[0]
        self.assertAlmostEqual(scaled.rect.width, plain.rect.width * 2.0, places=4)
        self.assertAlmostEqual(scaled.rect.height, plain.rect.height * 2.0, places=4)
        self.assertAlmostEqual(scaled.baseline, 700.0, places=6)
        self.assertAlmostEqual(scaled.rect.x0, 100.0, places=6)
        self.assertAlmostEqual(scaled.font_size, 24.0, places=6)

    def test_cm_translates_the_span(self):
        result = run(
            b"1 0 0 1 30 -40 cm BT /F1 12 Tf 100 700 Td (Hello) Tj ET", font_resources()
        )
        span = result.spans[0]
        self.assertAlmostEqual(span.rect.x0, 130.0, places=6)
        self.assertAlmostEqual(span.baseline, 660.0, places=6)

    def test_tj_array_applies_kerning_in_thousandths(self):
        result = run(b"BT /F1 10 Tf 50 500 Td [(A) -1000 (B)] TJ ET", font_resources())
        self.assertEqual([span.text for span in result.spans], ["A", "B"])
        advance = fonts.text_width("A", "Helvetica", 10.0)
        self.assertAlmostEqual(result.spans[1].rect.x0, 50.0 + advance + 10.0, places=4)

    def test_char_spacing_widens_the_run(self):
        plain = run(b"BT /F1 10 Tf 0 0 Td (AAA) Tj ET", font_resources()).spans[0]
        spaced = run(b"BT /F1 10 Tf 2 Tc 0 0 Td (AAA) Tj ET", font_resources()).spans[0]
        # Tc applies after every glyph, so the box grows by 2 x (n - 1) plus the trailing
        # advance is unchanged: the last glyph's box still ends at its own right edge.
        self.assertAlmostEqual(spaced.rect.width, plain.rect.width + 4.0, places=4)

    def test_word_spacing_applies_to_code_32_only(self):
        plain = run(b"BT /F1 10 Tf 0 0 Td (A B) Tj ET", font_resources()).spans[0]
        spaced = run(b"BT /F1 10 Tf 5 Tw 0 0 Td (A B) Tj ET", font_resources()).spans[0]
        self.assertAlmostEqual(spaced.rect.width, plain.rect.width + 5.0, places=4)

    def test_horizontal_scaling_stretches_only_x(self):
        plain = run(b"BT /F1 10 Tf 0 0 Td (AAA) Tj ET", font_resources()).spans[0]
        wide = run(b"BT /F1 10 Tf 200 Tz 0 0 Td (AAA) Tj ET", font_resources()).spans[0]
        self.assertAlmostEqual(wide.rect.width, plain.rect.width * 2.0, places=4)
        self.assertAlmostEqual(wide.rect.height, plain.rect.height, places=4)

    def test_rise_lifts_the_baseline(self):
        result = run(b"BT /F1 10 Tf 100 700 Td 4 Ts (x) Tj ET", font_resources())
        self.assertAlmostEqual(result.spans[0].baseline, 704.0, places=6)

    def test_td_sets_leading_and_star_moves_down(self):
        result = run(
            b"BT /F1 10 Tf 100 700 TD 0 -14 TD (a) Tj T* (b) Tj ET", font_resources()
        )
        self.assertEqual([span.text for span in result.spans], ["a", "b"])
        self.assertAlmostEqual(result.spans[0].baseline, 686.0, places=6)
        self.assertAlmostEqual(result.spans[1].baseline, 672.0, places=6)

    def test_quote_operators_move_to_the_next_line(self):
        result = run(
            b"BT /F1 10 Tf 12 TL 100 700 Td (a) ' 1 2 (b) \" ET", font_resources()
        )
        self.assertEqual([span.text for span in result.spans], ["a", "b"])
        self.assertAlmostEqual(result.spans[0].baseline, 688.0, places=6)
        self.assertAlmostEqual(result.spans[1].baseline, 676.0, places=6)

    def test_bt_resets_the_text_matrix(self):
        result = run(
            b"BT /F1 10 Tf 300 300 Td (a) Tj ET BT /F1 10 Tf (b) Tj ET", font_resources()
        )
        self.assertAlmostEqual(result.spans[1].rect.x0, 0.0, places=6)
        self.assertAlmostEqual(result.spans[1].baseline, 0.0, places=6)


# --------------------------------------------------------------------------------------
# Invisible text
# --------------------------------------------------------------------------------------

class RenderModeTests(unittest.TestCase):
    def test_render_mode_3_marks_the_span_invisible(self):
        result = run(b"BT /F1 12 Tf 3 Tr 100 700 Td (Hidden) Tj ET", font_resources())
        self.assertEqual(len(result.spans), 1)
        self.assertEqual(result.spans[0].confidence, 0.0)
        self.assertEqual(result.spans[0].text, "Hidden")
        self.assertEqual(result.visible_spans, [])

    def test_render_mode_7_is_clip_only_and_invisible(self):
        result = run(b"BT /F1 12 Tf 7 Tr 100 700 Td (Clip) Tj ET", font_resources())
        self.assertEqual(result.spans[0].confidence, 0.0)
        self.assertEqual(result.visible_spans, [])

    def test_visible_spans_keeps_ordinary_modes(self):
        stream = (
            b"BT /F1 12 Tf 100 700 Td (Shown) Tj ET "
            b"BT /F1 12 Tf 3 Tr 100 680 Td (OCR) Tj ET"
        )
        result = run(stream, font_resources())
        self.assertEqual(len(result.spans), 2)
        self.assertEqual([span.text for span in result.visible_spans], ["Shown"])

    def test_stroke_render_mode_stays_visible(self):
        result = run(b"BT /F1 12 Tf 1 Tr 100 700 Td (Outline) Tj ET", font_resources())
        self.assertEqual(result.spans[0].confidence, 1.0)


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

class PathTests(unittest.TestCase):
    def test_horizontal_segment_is_a_line_with_the_right_rect(self):
        result = run(b"100 700 m 300 700 l S")
        self.assertEqual(len(result.primitives), 1)
        prim = result.primitives[0]
        self.assertEqual(prim.kind, "line")
        self.assertEqual(prim.rect, Rect(100.0, 700.0, 300.0, 700.0))
        self.assertEqual(prim.orientation(), "horizontal")
        self.assertTrue(prim.stroked)
        self.assertFalse(prim.filled)
        self.assertEqual([p.as_tuple() for p in prim.points], [(100.0, 700.0), (300.0, 700.0)])

    def test_vertical_segment_orientation(self):
        result = run(b"100 100 m 100 400 l S")
        self.assertEqual(result.primitives[0].kind, "line")
        self.assertEqual(result.primitives[0].orientation(), "vertical")

    def test_re_fill_is_a_filled_rect(self):
        result = run(b"100 100 200 50 re f")
        self.assertEqual(len(result.primitives), 1)
        prim = result.primitives[0]
        self.assertEqual(prim.kind, "rect")
        self.assertEqual(prim.rect, Rect(100.0, 100.0, 300.0, 150.0))
        self.assertTrue(prim.filled)
        self.assertFalse(prim.stroked)
        self.assertEqual(len(prim.points), 5)

    def test_thin_rectangle_is_classified_as_a_line(self):
        result = run(b"100 500 200 0.75 re f")
        self.assertEqual(result.primitives[0].kind, "line")
        self.assertEqual(result.primitives[0].rect, Rect(100.0, 500.0, 300.0, 500.75))

    def test_line_thickness_threshold_comes_from_config(self):
        config = ZfpConfig.default()
        config.detection.max_line_thickness_pt = 0.1
        result = run(b"100 500 200 0.75 re f", config=config)
        self.assertEqual(result.primitives[0].kind, "rect")

    def test_stroke_width_is_scaled_by_the_ctm(self):
        result = run(b"3 w 2 0 0 2 0 0 cm 10 10 m 100 10 l S")
        self.assertAlmostEqual(result.primitives[0].stroke_width, 6.0, places=6)

    def test_ext_gstate_line_width(self):
        resources = PdfDict({"ExtGState": PdfDict({"GS0": PdfDict({"LW": 4})})})
        result = run(b"/GS0 gs 10 10 m 100 10 l S", resources)
        self.assertAlmostEqual(result.primitives[0].stroke_width, 4.0, places=6)

    def test_closed_polygon_is_a_rect(self):
        stream = b"100 100 m 300 100 l 300 200 l 100 200 l h S"
        result = run(stream)
        self.assertEqual(result.primitives[0].kind, "rect")
        self.assertEqual(result.primitives[0].rect, Rect(100.0, 100.0, 300.0, 200.0))

    def test_non_axis_aligned_polygon_is_a_path(self):
        stream = b"100 100 m 300 140 l 280 240 l 90 200 l h S"
        result = run(stream)
        self.assertEqual(result.primitives[0].kind, "path")

    def test_four_beziers_make_a_circle(self):
        result = run(_circle_stream(100.0, 100.0, 20.0) + b" S")
        self.assertEqual(len(result.primitives), 1)
        prim = result.primitives[0]
        self.assertEqual(prim.kind, "circle")
        self.assertAlmostEqual(prim.rect.x0, 80.0, places=3)
        self.assertAlmostEqual(prim.rect.x1, 120.0, places=3)
        self.assertAlmostEqual(prim.rect.y0, 80.0, places=3)
        self.assertAlmostEqual(prim.rect.y1, 120.0, places=3)

    def test_bezier_bbox_uses_the_curve_not_the_control_hull(self):
        # A control point far above the curve must not inflate the box.
        result = run(b"0 0 m 0 100 100 100 100 0 c S")
        self.assertLess(result.primitives[0].rect.y1, 80.0)

    def test_each_subpath_becomes_its_own_primitive(self):
        result = run(b"10 10 m 100 10 l 10 50 m 100 50 l S")
        self.assertEqual(len(result.primitives), 2)
        self.assertEqual({p.kind for p in result.primitives}, {"line"})

    def test_clip_only_path_is_not_ink(self):
        result = run(b"100 100 200 50 re W n")
        self.assertEqual(result.primitives, [])
        self.assertEqual(result.white_fills, [])

    def test_clip_then_paint_still_paints(self):
        result = run(b"100 100 200 50 re W f")
        self.assertEqual(len(result.primitives), 1)
        self.assertTrue(result.primitives[0].filled)

    def test_bare_n_paints_nothing(self):
        self.assertEqual(run(b"100 100 200 50 re n").primitives, [])

    def test_path_is_reset_between_paints(self):
        result = run(b"10 10 m 100 10 l S 10 50 m 100 50 l S")
        self.assertEqual(len(result.primitives), 2)

    def test_fill_and_stroke_sets_both_flags(self):
        prim = run(b"100 100 200 50 re B").primitives[0]
        self.assertTrue(prim.filled)
        self.assertTrue(prim.stroked)

    def test_rotated_re_is_not_reported_as_an_axis_aligned_rect(self):
        # 45 degrees: cos = sin = 0.7071
        stream = b"0.7071 0.7071 -0.7071 0.7071 0 0 cm 100 100 60 60 re f"
        self.assertEqual(run(stream).primitives[0].kind, "path")


def _circle_stream(cx: float, cy: float, radius: float) -> bytes:
    """Four cubic Beziers approximating a circle, as content-stream bytes."""
    k = radius * 0.5523
    parts = [
        "%f %f m" % (cx + radius, cy),
        "%f %f %f %f %f %f c" % (cx + radius, cy + k, cx + k, cy + radius, cx, cy + radius),
        "%f %f %f %f %f %f c" % (cx - k, cy + radius, cx - radius, cy + k, cx - radius, cy),
        "%f %f %f %f %f %f c" % (cx - radius, cy - k, cx - k, cy - radius, cx, cy - radius),
        "%f %f %f %f %f %f c" % (cx + k, cy - radius, cx + radius, cy - k, cx + radius, cy),
    ]
    return " ".join(parts).encode("ascii")


# --------------------------------------------------------------------------------------
# Graphics state stack
# --------------------------------------------------------------------------------------

class GraphicsStateTests(unittest.TestCase):
    def test_q_and_Q_restore_the_ctm(self):
        stream = b"q 1 0 0 1 50 50 cm 100 700 m 300 700 l S Q 100 700 m 300 700 l S"
        result = run(stream)
        self.assertEqual(len(result.primitives), 2)
        self.assertEqual(result.primitives[0].rect, Rect(150.0, 750.0, 350.0, 750.0))
        self.assertEqual(result.primitives[1].rect, Rect(100.0, 700.0, 300.0, 700.0))

    def test_q_and_Q_restore_the_line_width(self):
        result = run(b"q 8 w 0 0 m 10 0 l S Q 0 20 m 10 20 l S")
        self.assertAlmostEqual(result.primitives[0].stroke_width, 8.0, places=6)
        self.assertAlmostEqual(result.primitives[1].stroke_width, 1.0, places=6)

    def test_unbalanced_Q_is_ignored(self):
        result = run(b"Q Q 100 700 m 300 700 l S")
        self.assertEqual(result.primitives[0].rect, Rect(100.0, 700.0, 300.0, 700.0))

    def test_nested_cm_composes(self):
        result = run(b"2 0 0 2 0 0 cm 1 0 0 1 10 10 cm 0 0 m 10 0 l S")
        self.assertEqual(result.primitives[0].rect, Rect(20.0, 20.0, 40.0, 20.0))

    def test_content_state_defaults_match_the_specification(self):
        state = ContentState()
        self.assertEqual(state.horizontal_scale, 100.0)
        self.assertEqual(state.horizontal_factor, 1.0)
        self.assertEqual(state.render_mode, 0)
        self.assertEqual(state.stroke_width, 1.0)
        self.assertFalse(state.fill_is_white())

    def test_content_state_copy_is_independent(self):
        state = ContentState()
        clone = state.copy()
        clone.font_size = 42.0
        self.assertEqual(state.font_size, 0.0)


# --------------------------------------------------------------------------------------
# Colour and white fills
# --------------------------------------------------------------------------------------

class ColourTests(unittest.TestCase):
    def test_white_rgb_fill_is_recorded(self):
        result = run(b"1 1 1 rg 100 100 200 50 re f")
        self.assertEqual(result.white_fills, [Rect(100.0, 100.0, 300.0, 150.0)])
        prim = result.primitives[0]
        self.assertEqual(prim.kind, "rect")
        self.assertTrue(prim.filled)
        self.assertFalse(prim.stroked)

    def test_white_gray_fill_is_recorded(self):
        self.assertEqual(len(run(b"1 g 100 100 200 50 re f").white_fills), 1)

    def test_zero_ink_cmyk_is_white(self):
        self.assertEqual(len(run(b"0 0 0 0 k 100 100 200 50 re f").white_fills), 1)

    def test_black_fill_is_not_white(self):
        result = run(b"0 g 100 100 200 50 re f")
        self.assertEqual(result.white_fills, [])
        self.assertTrue(result.primitives[0].filled)

    def test_default_fill_colour_is_black(self):
        self.assertEqual(run(b"100 100 200 50 re f").white_fills, [])

    def test_stroking_white_does_not_make_a_fill_white(self):
        self.assertEqual(run(b"1 1 1 RG 100 100 200 50 re f").white_fills, [])

    def test_separation_tint_zero_is_white(self):
        resources = PdfDict(
            {
                "ColorSpace": PdfDict(
                    {
                        "CS0": PdfArray(
                            [PdfName("Separation"), PdfName("Spot"), PdfName("DeviceGray")]
                        )
                    }
                )
            }
        )
        white = run(b"/CS0 cs 0 scn 100 100 200 50 re f", resources)
        inked = run(b"/CS0 cs 1 scn 100 100 200 50 re f", resources)
        self.assertEqual(len(white.white_fills), 1)
        self.assertEqual(inked.white_fills, [])

    def test_pattern_fill_is_never_white(self):
        resources = PdfDict({"ColorSpace": PdfDict({"P0": PdfArray([PdfName("Pattern")])})})
        result = run(b"/P0 cs /Pat0 scn 100 100 200 50 re f", resources)
        self.assertEqual(result.white_fills, [])

    def test_icc_based_three_component_space_reads_as_rgb(self):
        icc = PdfStream(PdfDict({"N": 3}), b"")
        resources = PdfDict(
            {"ColorSpace": PdfDict({"CS1": PdfArray([PdfName("ICCBased"), icc])})}
        )
        result = run(b"/CS1 cs 1 1 1 sc 100 100 200 50 re f", resources)
        self.assertEqual(len(result.white_fills), 1)

    def test_q_restores_the_fill_colour(self):
        result = run(b"q 1 1 1 rg 0 0 10 10 re f Q 20 20 10 10 re f")
        self.assertEqual(len(result.white_fills), 1)
        self.assertEqual(result.white_fills[0], Rect(0.0, 0.0, 10.0, 10.0))


# --------------------------------------------------------------------------------------
# XObjects and images
# --------------------------------------------------------------------------------------

class XObjectTests(unittest.TestCase):
    def _form_page(self, content: bytes, matrix: Optional[List[float]] = None, **extra: Any):
        doc = Document.from_pages_blank(1)
        page = doc.page(0)
        form_dict = PdfDict(
            {
                "Type": PdfName("XObject"),
                "Subtype": PdfName("Form"),
                "BBox": PdfArray([0, 0, 400, 400]),
            }
        )
        if matrix is not None:
            form_dict["Matrix"] = PdfArray(list(matrix))
        form_dict.update(extra)
        form = PdfStream(form_dict, content)
        ref = doc.writer.add_object(form)
        page.dict["Resources"] = PdfDict({"XObject": PdfDict({"Fm0": ref})})
        page.touch()
        return doc, page, form, ref

    def test_form_xobject_content_is_picked_up_with_its_matrix(self):
        doc, page, _form, _ref = self._form_page(b"0 0 m 50 0 l S", [1, 0, 0, 1, 100, 500])
        doc.writer.set_object(page.dict["Contents"].num, PdfStream(PdfDict({}), b"/Fm0 Do"))
        result = ContentStreamInterpreter(page, doc).run()
        self.assertEqual(len(result.primitives), 1)
        self.assertEqual(result.primitives[0].rect, Rect(100.0, 500.0, 150.0, 500.0))

    def test_form_matrix_composes_with_the_page_ctm(self):
        doc, page, _form, _ref = self._form_page(b"0 0 m 50 0 l S", [2, 0, 0, 2, 0, 0])
        doc.writer.set_object(
            page.dict["Contents"].num, PdfStream(PdfDict({}), b"1 0 0 1 10 20 cm /Fm0 Do")
        )
        result = ContentStreamInterpreter(page, doc).run()
        self.assertEqual(result.primitives[0].rect, Rect(10.0, 20.0, 110.0, 20.0))

    def test_form_text_uses_the_forms_own_resources(self):
        doc, page, _form, _ref = self._form_page(
            b"BT /FF 12 Tf 10 10 Td (Inner) Tj ET",
            [1, 0, 0, 1, 0, 0],
            Resources=font_resources(name="FF"),
        )
        doc.writer.set_object(page.dict["Contents"].num, PdfStream(PdfDict({}), b"/Fm0 Do"))
        result = ContentStreamInterpreter(page, doc).run()
        self.assertEqual([span.text for span in result.spans], ["Inner"])
        self.assertAlmostEqual(result.spans[0].rect.x0, 10.0, places=6)

    def test_self_referencing_form_terminates(self):
        doc = Document.from_pages_blank(1)
        page = doc.page(0)
        number = doc.writer.allocate()
        form = PdfStream(
            PdfDict(
                {
                    "Type": PdfName("XObject"),
                    "Subtype": PdfName("Form"),
                    "BBox": PdfArray([0, 0, 400, 400]),
                    "Resources": PdfDict({"XObject": PdfDict({"Fm0": None})}),
                }
            ),
            b"0 0 m 10 0 l S /Fm0 Do",
        )
        doc.writer.set_object(number, form)
        from zfp.pdfio.objects import PdfRef

        form.dict["Resources"]["XObject"]["Fm0"] = PdfRef(number)
        page.dict["Resources"] = PdfDict({"XObject": PdfDict({"Fm0": PdfRef(number)})})
        page.touch()
        doc.writer.set_object(page.dict["Contents"].num, PdfStream(PdfDict({}), b"/Fm0 Do"))
        result = ContentStreamInterpreter(page, doc).run()
        self.assertEqual(len(result.primitives), 1)

    def test_nested_forms_stop_at_the_depth_limit(self):
        doc = Document.from_pages_blank(1)
        page = doc.page(0)
        from zfp.pdfio.objects import PdfRef

        # A chain of 12 distinct forms, each drawing a segment then calling the next.
        refs: List[PdfRef] = []
        previous: Optional[PdfRef] = None
        for level in range(12):
            body = b"0 %d m 10 %d l S" % (level, level)
            resources = PdfDict()
            if previous is not None:
                resources["XObject"] = PdfDict({"Next": previous})
                body += b" /Next Do"
            stream = PdfStream(
                PdfDict(
                    {
                        "Type": PdfName("XObject"),
                        "Subtype": PdfName("Form"),
                        "BBox": PdfArray([0, 0, 400, 400]),
                        "Resources": resources,
                    }
                ),
                body,
            )
            previous = doc.writer.add_object(stream)
            refs.append(previous)
        page.dict["Resources"] = PdfDict({"XObject": PdfDict({"Fm0": refs[-1]})})
        page.touch()
        doc.writer.set_object(page.dict["Contents"].num, PdfStream(PdfDict({}), b"/Fm0 Do"))
        result = ContentStreamInterpreter(page, doc).run()
        # The outermost call is depth 1, so at most MAX_FORM_DEPTH forms execute.
        self.assertEqual(len(result.primitives), ContentStreamInterpreter.MAX_FORM_DEPTH)

    def test_image_xobject_records_its_transformed_box(self):
        doc = Document.from_pages_blank(1)
        page = doc.page(0)
        image = PdfStream(
            PdfDict({"Type": PdfName("XObject"), "Subtype": PdfName("Image"), "Width": 4,
                     "Height": 4}),
            b"\x00" * 16,
        )
        ref = doc.writer.add_object(image)
        page.dict["Resources"] = PdfDict({"XObject": PdfDict({"Im0": ref})})
        page.touch()
        doc.writer.set_object(
            page.dict["Contents"].num,
            PdfStream(PdfDict({}), b"q 200 0 0 100 50 600 cm /Im0 Do Q"),
        )
        result = ContentStreamInterpreter(page, doc).run()
        self.assertEqual(result.images, [Rect(50.0, 600.0, 250.0, 700.0)])
        self.assertEqual(result.primitives, [])

    def test_missing_xobject_is_ignored(self):
        result = run(b"/Nope Do 10 10 m 20 10 l S")
        self.assertEqual(len(result.primitives), 1)


# --------------------------------------------------------------------------------------
# Inline images
# --------------------------------------------------------------------------------------

class InlineImageTests(unittest.TestCase):
    def test_inline_image_box_and_resumption(self):
        payload = bytes(range(4))
        stream = (
            b"q 200 0 0 100 50 600 cm BI /W 2 /H 2 /CS /G /BPC 8 ID "
            + payload
            + b" EI Q 10 10 m 20 10 l S"
        )
        result = run(stream)
        self.assertEqual(result.images, [Rect(50.0, 600.0, 250.0, 700.0)])
        self.assertEqual(len(result.primitives), 1)
        self.assertEqual(result.primitives[0].rect, Rect(10.0, 10.0, 20.0, 10.0))

    def test_payload_containing_the_bytes_EI_is_skipped_correctly(self):
        # 8 bytes of payload, one of which is the literal 'EI' preceded by a space.
        payload = b"\x01 EI\x02\x03\x04\x05"
        stream = (
            b"BI /W 8 /H 1 /CS /G /BPC 8 ID " + payload + b" EI 10 10 m 20 10 l S"
        )
        result = run(stream)
        self.assertEqual(len(result.images), 1)
        self.assertEqual(len(result.primitives), 1)

    def test_filtered_inline_image_scans_for_the_terminator(self):
        stream = b"BI /W 2 /H 2 /F /AHx ID 00112233> EI 10 10 m 20 10 l S"
        result = run(stream)
        self.assertEqual(len(result.images), 1)
        self.assertEqual(len(result.primitives), 1)

    def test_truncated_inline_image_does_not_hang(self):
        result = run(b"BI /W 2 /H 2 /CS /G /BPC 8 ID \x00\x01")
        self.assertEqual(len(result.images), 1)


# --------------------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------------------

class RobustnessTests(unittest.TestCase):
    def test_garbage_content_does_not_raise(self):
        result = run(b"\x00\x01 ] >> ) foo bar 1 2 3 baz BT ET Q Q Q")
        self.assertIsInstance(result, ContentResult)
        self.assertEqual(result.spans, [])

    def test_operator_failure_is_counted_and_execution_continues(self):
        resources = _ExplodingResources({"Font": PdfDict({"F1": helvetica_font()})})
        result = run(b"BT /F1 12 Tf 0 0 Td (a) Tj ET 10 10 m 20 10 l S", resources)
        self.assertGreaterEqual(result.errors, 1)
        self.assertEqual(len(result.primitives), 1)

    def test_unreadable_content_returns_an_empty_result_with_an_error(self):
        result = ContentStreamInterpreter(_BrokenPage(), None).run()
        self.assertEqual(result.spans, [])
        self.assertEqual(result.errors, 1)

    def test_operator_with_too_few_operands_is_skipped(self):
        result = run(b"1 0 0 cm 10 10 m 20 10 l S")
        self.assertEqual(result.primitives[0].rect, Rect(10.0, 10.0, 20.0, 10.0))

    def test_op_count_counts_every_executed_operator(self):
        result = run(b"10 10 m 20 10 l S")
        self.assertEqual(result.op_count, 3)

    def test_unknown_operators_are_skipped_safely(self):
        result = run(b"/OC /MC0 BDC 10 10 m 20 10 l S EMC")
        self.assertEqual(len(result.primitives), 1)
        self.assertEqual(result.errors, 0)

    def test_as_dict_is_json_shaped(self):
        payload = run(b"BT /F1 12 Tf 0 0 Td (a) Tj ET 1 1 1 rg 0 0 5 5 re f").as_dict()
        self.assertEqual(sorted(payload), ["errors", "images", "op_count", "primitives",
                                           "spans", "white_fills"])
        self.assertEqual(payload["white_fills"], [[0.0, 0.0, 5.0, 5.0]])

    def test_analyze_page_matches_the_interpreter(self):
        doc, page = make_page(b"10 10 m 20 10 l S")
        self.assertEqual(
            analyze_page(page).primitives[0].rect,
            ContentStreamInterpreter(page, doc).run().primitives[0].rect,
        )

    def test_results_are_deterministic(self):
        stream = b"BT /F1 12 Tf 100 700 Td (Hello) Tj ET 10 10 m 20 10 l S"
        first = run(stream, font_resources()).as_dict()
        second = run(stream, font_resources()).as_dict()
        self.assertEqual(first, second)


# --------------------------------------------------------------------------------------
# Embedded font metrics through the interpreter
# --------------------------------------------------------------------------------------

class EmbeddedFontTests(unittest.TestCase):
    def test_widths_array_drives_the_advance(self):
        font = PdfDict(
            {
                "Type": PdfName("Font"),
                "Subtype": PdfName("TrueType"),
                "BaseFont": PdfName("ABCDEF+CustomSans"),
                "Encoding": PdfName("WinAnsiEncoding"),
                "FirstChar": 65,
                "Widths": PdfArray([1000, 1000]),
            }
        )
        result = run(b"BT /F1 10 Tf 0 0 Td (AB) Tj ET", font_resources(font))
        self.assertAlmostEqual(result.spans[0].rect.width, 20.0, places=4)

    def test_missing_width_covers_codes_outside_the_widths_array(self):
        font = PdfDict(
            {
                "Type": PdfName("Font"),
                "Subtype": PdfName("TrueType"),
                "BaseFont": PdfName("ABCDEF+CustomSans"),
                "Encoding": PdfName("WinAnsiEncoding"),
                "FirstChar": 65,
                "Widths": PdfArray([1000]),
                "FontDescriptor": PdfDict({"MissingWidth": 500}),
            }
        )
        result = run(b"BT /F1 10 Tf 0 0 Td (AB) Tj ET", font_resources(font))
        self.assertAlmostEqual(result.spans[0].rect.width, 15.0, places=4)

    def test_differences_rename_a_code(self):
        font = PdfDict(
            {
                "Type": PdfName("Font"),
                "Subtype": PdfName("Type1"),
                "BaseFont": PdfName("Helvetica"),
                "Encoding": PdfDict(
                    {
                        "BaseEncoding": PdfName("WinAnsiEncoding"),
                        "Differences": PdfArray([65, PdfName("eacute"), PdfName("bullet")]),
                    }
                ),
            }
        )
        result = run(b"BT /F1 10 Tf 0 0 Td (AB) Tj ET", font_resources(font))
        self.assertEqual(result.spans[0].text, "é•")

    def test_type0_identity_h_two_byte_codes(self):
        to_unicode = PdfStream(
            PdfDict({}),
            b"/CIDInit /ProcSet findresource begin\n"
            b"1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
            b"2 beginbfchar <0001> <0041> <0002> <0042> endbfchar\n"
            b"end",
        )
        descendant = PdfDict(
            {
                "Type": PdfName("Font"),
                "Subtype": PdfName("CIDFontType2"),
                "BaseFont": PdfName("ABCDEF+Noto"),
                "DW": 1000,
                "W": PdfArray([1, PdfArray([500, 600])]),
            }
        )
        font = PdfDict(
            {
                "Type": PdfName("Font"),
                "Subtype": PdfName("Type0"),
                "BaseFont": PdfName("ABCDEF+Noto"),
                "Encoding": PdfName("Identity-H"),
                "DescendantFonts": PdfArray([descendant]),
                "ToUnicode": to_unicode,
            }
        )
        result = run(b"BT /F1 10 Tf 0 0 Td <00010002> Tj ET", font_resources(font))
        span = result.spans[0]
        self.assertEqual(span.text, "AB")
        self.assertAlmostEqual(span.rect.width, 11.0, places=4)
        self.assertEqual(len(span.glyph_rects), 2)

    def test_type0_without_to_unicode_keeps_geometry_but_no_text(self):
        descendant = PdfDict({"Subtype": PdfName("CIDFontType2"), "DW": 1000})
        font = PdfDict(
            {
                "Type": PdfName("Font"),
                "Subtype": PdfName("Type0"),
                "BaseFont": PdfName("ABCDEF+Noto"),
                "Encoding": PdfName("Identity-H"),
                "DescendantFonts": PdfArray([descendant]),
            }
        )
        result = run(b"BT /F1 10 Tf 0 0 Td <00010002> Tj ET", font_resources(font))
        span = result.spans[0]
        self.assertEqual(span.text, "")
        self.assertAlmostEqual(span.rect.width, 20.0, places=4)


# --------------------------------------------------------------------------------------
# zfp.native.encoding
# --------------------------------------------------------------------------------------

class EncodingTests(unittest.TestCase):
    def test_decode_string_without_a_font_uses_winansi(self):
        self.assertEqual(encoding.decode_string(b"Hi", None), [(72, "H"), (105, "i")])

    def test_font_widths_default_font_matches_helvetica(self):
        widths, default = encoding.font_widths(None)
        self.assertAlmostEqual(widths[ord("A")], 0.667, places=3)
        self.assertGreater(default, 0.0)

    def test_glyph_names(self):
        self.assertEqual(encoding.glyph_to_unicode("eacute"), "é")
        self.assertEqual(encoding.glyph_to_unicode("uni20AC"), "€")
        self.assertEqual(encoding.glyph_to_unicode("u1F600"), "\U0001f600")
        self.assertEqual(encoding.glyph_to_unicode("one.oldstyle"), "1")
        self.assertEqual(encoding.glyph_to_unicode("f_i"), "fi")
        self.assertEqual(encoding.glyph_to_unicode("g14"), "")
        self.assertEqual(encoding.glyph_to_unicode(""), "")

    def test_standard_encoding_uses_curly_quotes(self):
        self.assertEqual(encoding.STANDARD_ENCODING[39], "’")
        self.assertEqual(encoding.STANDARD_ENCODING[96], "‘")
        self.assertEqual(encoding.WINANSI_ENCODING[39], "'")

    def test_winansi_high_codes(self):
        self.assertEqual(encoding.WINANSI_ENCODING[0x80], "€")
        self.assertEqual(encoding.WINANSI_ENCODING[0xA0], " ")
        self.assertEqual(encoding.WINANSI_ENCODING[0xE9], "é")

    def test_base_encoding_table_is_a_copy(self):
        table = encoding.base_encoding_table("WinAnsiEncoding")
        table[65] = "?"
        self.assertEqual(encoding.WINANSI_ENCODING[65], "A")

    def test_to_unicode_bfchar_and_bfrange(self):
        cmap = (
            b"2 beginbfchar <0003> <0020> <0004> <002C> endbfchar\n"
            b"1 beginbfrange <0010> <0012> <0041> endbfrange\n"
            b"1 beginbfrange <0020> <0021> [<0058> <0059>] endbfrange\n"
        )
        table = encoding.parse_to_unicode(cmap)
        self.assertEqual(table[3], " ")
        self.assertEqual(table[4], ",")
        self.assertEqual([table[c] for c in (0x10, 0x11, 0x12)], ["A", "B", "C"])
        self.assertEqual([table[0x20], table[0x21]], ["X", "Y"])

    def test_to_unicode_handles_surrogate_pairs(self):
        table = encoding.parse_to_unicode(b"1 beginbfchar <0001> <D83DDE00> endbfchar")
        self.assertEqual(table[1], "\U0001f600")

    def test_to_unicode_survives_a_truncated_cmap(self):
        self.assertEqual(encoding.parse_to_unicode(b"1 beginbfchar <0001>"), {})
        self.assertEqual(encoding.parse_to_unicode(b""), {})

    def test_codespace_lengths(self):
        cmap = b"1 begincodespacerange <00> <80> <8140> <9ffc> endcodespacerange"
        self.assertEqual(encoding.parse_codespace_lengths(cmap), [1, 2])
        self.assertEqual(encoding.parse_codespace_lengths(b""), [])

    def test_load_font_of_junk_degrades_to_helvetica(self):
        program = encoding.load_font(42)
        self.assertEqual(program.std_font, "Helvetica")
        self.assertEqual(program.code_bytes, 1)

    def test_word_spacing_never_applies_to_a_composite_code(self):
        descendant = PdfDict({"Subtype": PdfName("CIDFontType2"), "DW": 1000})
        font = PdfDict(
            {
                "Subtype": PdfName("Type0"),
                "BaseFont": PdfName("X"),
                "Encoding": PdfName("Identity-H"),
                "DescendantFonts": PdfArray([descendant]),
            }
        )
        program = encoding.load_font(font)
        self.assertEqual(program.code_bytes, 2)
        self.assertFalse(program.is_space(32))
        self.assertTrue(encoding.load_font(helvetica_font()).is_space(32))

    def test_odd_trailing_byte_of_a_composite_string_is_kept(self):
        descendant = PdfDict({"Subtype": PdfName("CIDFontType2"), "DW": 1000})
        font = PdfDict(
            {
                "Subtype": PdfName("Type0"),
                "Encoding": PdfName("Identity-H"),
                "DescendantFonts": PdfArray([descendant]),
            }
        )
        program = encoding.load_font(font)
        self.assertEqual(program.codes(b"\x00\x01\x02"), [1, 2])

    def test_cid_widths_from_a_range_triple(self):
        descendant = PdfDict(
            {"Subtype": PdfName("CIDFontType0"), "DW": 800, "W": PdfArray([5, 7, 250])}
        )
        font = PdfDict(
            {
                "Subtype": PdfName("Type0"),
                "Encoding": PdfName("Identity-H"),
                "DescendantFonts": PdfArray([descendant]),
            }
        )
        widths, default = encoding.font_widths(font)
        self.assertAlmostEqual(widths[5], 0.25, places=6)
        self.assertAlmostEqual(widths[7], 0.25, places=6)
        self.assertNotIn(8, widths)
        self.assertAlmostEqual(default, 0.8, places=6)

    def test_zapf_dingbats_check_glyph(self):
        font = PdfDict({"Subtype": PdfName("Type1"), "BaseFont": PdfName("ZapfDingbats")})
        pairs = encoding.decode_string(b"4", font)
        self.assertEqual(pairs, [(0x34, "✔")])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
