"""Unit tests for :mod:`zfp.native.text` -- line grouping, run merging, columns, order.

The functions here decide what "a label" is, so the tests are built from spans placed at
deliberate coordinates rather than from a rendered page: a 0.4 pt baseline jitter must
stay on one line, a 20 pt drop must not, and a 100 pt gutter must split a page while a
3 pt word gap must not.

One end-to-end case drives the layout functions from a real content stream, so a change
in the interpreter that broke the spans it hands over would be caught here too.
"""

from __future__ import annotations

import unittest
from typing import List

from zfp.core.geometry import PageGeometry, Rect
from zfp.core.types import TextSpan
from zfp.native.content import ContentStreamInterpreter
from zfp.native.text import (
    GUTTER_RATIO,
    baseline_of,
    detect_columns,
    group_spans_into_lines,
    merge_adjacent_spans,
    reading_order,
    span_size,
)
from zfp.pdfio import fonts
from zfp.pdfio.document import Document
from zfp.pdfio.objects import PdfDict, PdfName, PdfStream

LETTER = PageGeometry(
    index=0, media_box=Rect(0, 0, 612, 792), crop_box=Rect(0, 0, 612, 792), rotation=0
)


def span(
    text: str,
    x0: float,
    baseline: float,
    width: float = 30.0,
    size: float = 10.0,
    *,
    with_baseline: bool = True,
    source: str = "native",
    confidence: float = 1.0,
    glyphs: bool = False,
) -> TextSpan:
    """Build a text span the way the interpreter would: box straddling the baseline."""
    rect = Rect(x0, baseline - 0.2 * size, x0 + width, baseline + 0.72 * size)
    glyph_rects: List[Rect] = []
    if glyphs and text:
        step = width / len(text)
        glyph_rects = [
            Rect(x0 + i * step, rect.y0, x0 + (i + 1) * step, rect.y1)
            for i in range(len(text))
        ]
    return TextSpan(
        text=text,
        rect=rect,
        page=0,
        font_name="Helvetica",
        font_size=size,
        source=source,
        confidence=confidence,
        glyph_rects=glyph_rects,
        baseline=baseline if with_baseline else None,
    )


# --------------------------------------------------------------------------------------
# Accessors
# --------------------------------------------------------------------------------------

class AccessorTests(unittest.TestCase):
    def test_baseline_of_prefers_the_baseline(self):
        self.assertEqual(baseline_of(span("a", 0, 700)), 700.0)

    def test_baseline_of_falls_back_to_the_box_centre(self):
        item = span("a", 0, 700, with_baseline=False)
        self.assertAlmostEqual(baseline_of(item), item.rect.center.y, places=9)

    def test_span_size_prefers_the_font_size(self):
        self.assertEqual(span_size(span("a", 0, 700, size=14.0)), 14.0)

    def test_span_size_falls_back_to_the_box_height(self):
        item = TextSpan("a", Rect(0, 0, 10, 8), 0)
        self.assertEqual(span_size(item), 8.0)

    def test_span_size_never_returns_zero(self):
        self.assertGreater(span_size(TextSpan("a", Rect(0, 0, 0, 0), 0)), 0.0)


# --------------------------------------------------------------------------------------
# group_spans_into_lines
# --------------------------------------------------------------------------------------

class GroupLinesTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(group_spans_into_lines([]), [])

    def test_same_baseline_groups_and_sorts_left_to_right(self):
        lines = group_spans_into_lines([span("b", 200, 700), span("a", 50, 700)])
        self.assertEqual([[s.text for s in line] for line in lines], [["a", "b"]])

    def test_lines_come_back_top_to_bottom(self):
        spans = [span("bottom", 50, 100), span("top", 50, 700), span("middle", 50, 400)]
        lines = group_spans_into_lines(spans)
        self.assertEqual([line[0].text for line in lines], ["top", "middle", "bottom"])

    def test_baseline_jitter_inside_the_tolerance_stays_on_one_line(self):
        # tolerance is 0.5 x median font size = 5 pt.
        lines = group_spans_into_lines([span("a", 50, 700), span("b", 200, 703.9)])
        self.assertEqual(len(lines), 1)

    def test_baseline_gap_beyond_the_tolerance_splits(self):
        lines = group_spans_into_lines([span("a", 50, 700), span("b", 200, 690)])
        self.assertEqual(len(lines), 2)

    def test_tolerance_follows_the_font_size(self):
        big = [span("a", 50, 700, size=40.0), span("b", 200, 712.0, size=40.0)]
        self.assertEqual(len(group_spans_into_lines(big)), 1)
        small = [span("a", 50, 700, size=4.0), span("b", 200, 712.0, size=4.0)]
        self.assertEqual(len(group_spans_into_lines(small)), 2)

    def test_spans_without_a_baseline_use_the_box_centre(self):
        spans = [
            span("a", 50, 700, with_baseline=False),
            span("b", 200, 700, with_baseline=False),
        ]
        self.assertEqual(len(group_spans_into_lines(spans)), 1)

    def test_every_span_appears_exactly_once(self):
        spans = [span(str(i), 50 * (i % 4), 700 - 20 * (i // 4)) for i in range(12)]
        lines = group_spans_into_lines(spans)
        flat = [s for line in lines for s in line]
        self.assertEqual(len(flat), 12)
        self.assertEqual({id(s) for s in flat}, {id(s) for s in spans})

    def test_grouping_is_deterministic_regardless_of_input_order(self):
        spans = [span("a", 50, 700), span("b", 200, 700), span("c", 50, 660)]
        first = [[s.text for s in line] for line in group_spans_into_lines(spans)]
        second = [[s.text for s in line] for line in group_spans_into_lines(list(reversed(spans)))]
        self.assertEqual(first, second)

    def test_zero_sized_spans_do_not_produce_a_zero_tolerance(self):
        spans = [
            TextSpan("a", Rect(0, 0, 0, 0), 0, baseline=100.0),
            TextSpan("b", Rect(0, 0, 0, 0), 0, baseline=100.2),
        ]
        self.assertEqual(len(group_spans_into_lines(spans)), 1)


# --------------------------------------------------------------------------------------
# merge_adjacent_spans
# --------------------------------------------------------------------------------------

class MergeSpansTests(unittest.TestCase):
    def test_touching_spans_merge_without_a_space(self):
        a = span("Name", 10, 700, width=30)
        b = span(":", 40.5, 700, width=3)
        merged = merge_adjacent_spans([a, b])
        self.assertEqual([s.text for s in merged], ["Name:"])

    def test_a_word_gap_inserts_a_space(self):
        a = span("First", 10, 700, width=30)
        b = span("Name", 43, 700, width=32)
        merged = merge_adjacent_spans([a, b])
        self.assertEqual([s.text for s in merged], ["First Name"])
        self.assertEqual(merged[0].rect, Rect(10, a.rect.y0, 75, a.rect.y1))

    def test_a_wide_gap_does_not_merge(self):
        a = span("Label", 10, 700, width=30)
        b = span("Value", 200, 700, width=30)
        self.assertEqual([s.text for s in merge_adjacent_spans([a, b])], ["Label", "Value"])

    def test_gap_ratio_is_honoured(self):
        a = span("A", 10, 700, width=30)
        b = span("B", 45, 700, width=30)  # 5 pt gap, font size 10
        self.assertEqual(len(merge_adjacent_spans([a, b], gap_ratio=0.3)), 2)
        self.assertEqual(len(merge_adjacent_spans([a, b], gap_ratio=0.6)), 1)

    def test_different_lines_never_merge(self):
        a = span("A", 10, 700)
        b = span("B", 42, 600)
        self.assertEqual(len(merge_adjacent_spans([a, b])), 2)

    def test_glyph_rects_are_preserved_and_stay_character_aligned(self):
        a = span("First", 10, 700, width=30, glyphs=True)
        b = span("Name", 43, 700, width=32, glyphs=True)
        merged = merge_adjacent_spans([a, b])[0]
        self.assertEqual(merged.text, "First Name")
        self.assertEqual(len(merged.glyph_rects), len(merged.text))
        self.assertAlmostEqual(merged.glyph_rects[5].x0, 40.0, places=6)
        self.assertAlmostEqual(merged.glyph_rects[5].x1, 43.0, places=6)

    def test_spans_without_glyph_rects_still_merge(self):
        a = span("First", 10, 700, width=30)
        b = span("Name", 43, 700, width=32)
        merged = merge_adjacent_spans([a, b])[0]
        self.assertEqual(merged.text, "First Name")
        self.assertEqual(merged.glyph_rects, [])

    def test_inputs_are_not_mutated(self):
        a = span("First", 10, 700, width=30, glyphs=True)
        b = span("Name", 43, 700, width=32, glyphs=True)
        before = (a.text, a.rect, len(a.glyph_rects))
        merge_adjacent_spans([a, b])
        self.assertEqual((a.text, a.rect, len(a.glyph_rects)), before)

    def test_invisible_text_never_merges_into_visible_text(self):
        visible = span("Hello", 10, 700, width=30)
        hidden = span("Hello", 41, 700, width=30, confidence=0.0)
        merged = merge_adjacent_spans([visible, hidden])
        self.assertEqual([s.text for s in merged], ["Hello", "Hello"])

    def test_ocr_and_native_never_merge(self):
        native = span("Hello", 10, 700, width=30)
        ocr = span("World", 41, 700, width=30, source="ocr")
        self.assertEqual(len(merge_adjacent_spans([native, ocr])), 2)

    def test_merged_span_keeps_the_largest_font_size(self):
        a = span("A", 10, 700, width=10, size=10.0)
        b = span("B", 21, 700, width=10, size=14.0)
        self.assertEqual(merge_adjacent_spans([a, b])[0].font_size, 14.0)

    def test_result_is_ordered_by_line_then_left_to_right(self):
        spans = [span("c", 10, 600), span("b", 200, 700), span("a", 10, 700)]
        self.assertEqual([s.text for s in merge_adjacent_spans(spans)], ["a", "b", "c"])

    def test_empty_input(self):
        self.assertEqual(merge_adjacent_spans([]), [])


# --------------------------------------------------------------------------------------
# detect_columns
# --------------------------------------------------------------------------------------

class DetectColumnsTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(detect_columns([], LETTER), [])

    def test_single_column_page_returns_one_rect(self):
        spans = [span("line %d" % i, 72, 700 - 14 * i, width=400) for i in range(10)]
        columns = detect_columns(spans, LETTER)
        self.assertEqual(len(columns), 1)
        self.assertAlmostEqual(columns[0].x0, 72.0, places=6)
        self.assertAlmostEqual(columns[0].x1, 472.0, places=6)

    def test_two_columns_are_split_at_the_gutter(self):
        spans: List[TextSpan] = []
        for i in range(8):
            spans.append(span("l%d" % i, 60, 700 - 14 * i, width=200))
            spans.append(span("r%d" % i, 340, 700 - 14 * i, width=200))
        columns = detect_columns(spans, LETTER)
        self.assertEqual(len(columns), 2)
        self.assertAlmostEqual(columns[0].x0, 60.0, places=6)
        self.assertAlmostEqual(columns[0].x1, 260.0, places=6)
        self.assertAlmostEqual(columns[1].x0, 340.0, places=6)
        self.assertAlmostEqual(columns[1].x1, 540.0, places=6)

    def test_a_gap_narrower_than_the_threshold_is_not_a_gutter(self):
        threshold = GUTTER_RATIO * 612.0
        spans = [
            span("l", 60, 700, width=200),
            span("r", 260 + threshold * 0.5, 700, width=200),
        ]
        self.assertEqual(len(detect_columns(spans, LETTER)), 1)

    def test_a_full_width_headline_does_not_weld_the_columns(self):
        spans = [span("A WIDE BANNER HEADLINE", 60, 740, width=480, size=18.0)]
        for i in range(8):
            spans.append(span("l%d" % i, 60, 700 - 14 * i, width=200))
            spans.append(span("r%d" % i, 340, 700 - 14 * i, width=200))
        self.assertEqual(len(detect_columns(spans, LETTER)), 2)

    def test_column_rects_cover_the_vertical_extent_of_their_text(self):
        spans = [
            span("l", 60, 700, width=200),
            span("l2", 60, 300, width=200),
            span("r", 340, 700, width=200),
        ]
        columns = detect_columns(spans, LETTER)
        self.assertLess(columns[0].y0, columns[1].y0)

    def test_columns_come_back_left_to_right(self):
        spans = [span("r", 340, 700, width=200), span("l", 60, 700, width=200)]
        columns = detect_columns(spans, LETTER)
        self.assertLess(columns[0].x0, columns[1].x0)

    def test_geometry_is_optional(self):
        spans = [span("l", 60, 700, width=200), span("r", 340, 700, width=200)]
        self.assertEqual(len(detect_columns(spans)), 2)

    def test_three_columns(self):
        spans: List[TextSpan] = []
        for i in range(6):
            spans.append(span("a", 40, 700 - 14 * i, width=140))
            spans.append(span("b", 240, 700 - 14 * i, width=140))
            spans.append(span("c", 440, 700 - 14 * i, width=140))
        self.assertEqual(len(detect_columns(spans, LETTER)), 3)


# --------------------------------------------------------------------------------------
# reading_order
# --------------------------------------------------------------------------------------

class ReadingOrderTests(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(reading_order([], LETTER), [])

    def test_single_column_is_plain_line_order(self):
        spans = [span("b", 200, 700), span("c", 50, 660), span("a", 50, 700)]
        self.assertEqual([s.text for s in reading_order(spans, LETTER)], ["a", "b", "c"])

    def test_two_columns_are_read_left_column_first(self):
        spans: List[TextSpan] = []
        for i in range(3):
            spans.append(span("r%d" % i, 340, 700 - 14 * i, width=200))
            spans.append(span("l%d" % i, 60, 700 - 14 * i, width=200))
        order = [s.text for s in reading_order(spans, LETTER)]
        self.assertEqual(order, ["l0", "l1", "l2", "r0", "r1", "r2"])

    def test_every_span_appears_exactly_once(self):
        spans: List[TextSpan] = []
        for i in range(5):
            spans.append(span("l%d" % i, 60, 700 - 14 * i, width=200))
            spans.append(span("r%d" % i, 340, 700 - 14 * i, width=200))
        order = reading_order(spans, LETTER)
        self.assertEqual(len(order), len(spans))
        self.assertEqual({id(s) for s in order}, {id(s) for s in spans})

    def test_geometry_is_optional(self):
        spans = [span("b", 200, 700), span("a", 50, 700)]
        self.assertEqual([s.text for s in reading_order(spans)], ["a", "b"])

    def test_order_is_deterministic(self):
        spans = [span("l", 60, 700, width=200), span("r", 340, 700, width=200)]
        first = [s.text for s in reading_order(spans, LETTER)]
        second = [s.text for s in reading_order(list(reversed(spans)), LETTER)]
        self.assertEqual(first, second)


# --------------------------------------------------------------------------------------
# End to end, from a real content stream
# --------------------------------------------------------------------------------------

class EndToEndTests(unittest.TestCase):
    def _spans(self, stream: bytes) -> List[TextSpan]:
        doc = Document.from_pages_blank(1)
        page = doc.page(0)
        page.dict["Resources"] = PdfDict(
            {
                "Font": PdfDict(
                    {
                        "F1": PdfDict(
                            {
                                "Type": PdfName("Font"),
                                "Subtype": PdfName("Type1"),
                                "BaseFont": PdfName("Helvetica"),
                                "Encoding": PdfName("WinAnsiEncoding"),
                            }
                        )
                    }
                )
            }
        )
        page.touch()
        doc.writer.set_object(page.dict["Contents"].num, PdfStream(PdfDict({}), stream))
        return ContentStreamInterpreter(page, doc).run().spans

    def test_label_fragments_from_a_stream_merge_into_one_run(self):
        # Three show operations: a word gap of 2.5 pt before "Name" (a real space) and a
        # 0.3 pt gap before the colon (no space).
        first_x = 72.0
        name_x = first_x + fonts.text_width("First", "Helvetica", 10.0) + 2.5
        colon_x = name_x + fonts.text_width("Name", "Helvetica", 10.0) + 0.3
        stream = (
            "BT /F1 10 Tf %f 700 Td (First) Tj ET "
            "BT /F1 10 Tf %f 700 Td (Name) Tj ET "
            "BT /F1 10 Tf %f 700 Td (:) Tj ET" % (first_x, name_x, colon_x)
        ).encode("ascii")
        spans = self._spans(stream)
        self.assertEqual(len(spans), 3)
        merged = merge_adjacent_spans(spans)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "First Name:")
        self.assertEqual(len(merged[0].glyph_rects), len(merged[0].text))

    def test_three_stream_lines_group_into_three_lines(self):
        stream = (
            b"BT /F1 10 Tf 14 TL 72 700 Td (one) Tj T* (two) Tj T* (three) Tj ET"
        )
        lines = group_spans_into_lines(self._spans(stream))
        self.assertEqual([line[0].text for line in lines], ["one", "two", "three"])

    def test_reading_order_over_a_two_column_stream(self):
        stream = (
            b"BT /F1 10 Tf 14 TL 320 700 Td (right one) Tj T* (right two) Tj ET "
            b"BT /F1 10 Tf 14 TL 72 700 Td (left one) Tj T* (left two) Tj ET"
        )
        spans = self._spans(stream)
        self.assertEqual(len(detect_columns(spans, LETTER)), 2)
        order = [s.text for s in reading_order(spans, LETTER)]
        self.assertEqual(order, ["left one", "left two", "right one", "right two"])

    def test_invisible_ocr_layer_is_kept_out_of_a_merged_visible_run(self):
        stream = (
            b"BT /F1 10 Tf 72 700 Td (Total) Tj ET "
            b"BT /F1 10 Tf 3 Tr 72 700 Td (Total) Tj ET"
        )
        spans = self._spans(stream)
        self.assertEqual(len(spans), 2)
        merged = merge_adjacent_spans(spans)
        self.assertEqual(sorted(s.text for s in merged), ["Total", "Total"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
