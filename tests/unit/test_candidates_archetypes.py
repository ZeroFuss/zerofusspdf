"""Unit tests for the eleven form archetype detectors.

Every test hand-builds the geometry a real page would present -- text spans plus vector
primitives -- and asserts the exact candidates that come back: how many, where (to half a
point), of what type and with which label.  The awkward cases have their own tests: two
fields sharing one rule, an underlined heading, a page-wide separator, a radio group, a
nine-cell comb, a table with a header row, and a detector that blows up.
"""

from __future__ import annotations

import unittest
from typing import List, Optional, Sequence

from zfp.candidates import (
    DEFAULT_DETECTORS,
    BlankRegionDetector,
    BoxFieldDetector,
    CandidateContext,
    CheckboxDetector,
    ColonRunDetector,
    CombFieldDetector,
    DateBoxDetector,
    FreeTextAreaDetector,
    RadioGroupDetector,
    SignatureLineDetector,
    TableCellDetector,
    UnderlineFieldDetector,
    build_context,
    clean_label,
    export_value_for,
    generate_candidates,
)
from zfp.candidates.archetypes import (
    FIELD_HEIGHT_FONT_RATIO,
    looks_like_date,
    looks_like_signature,
    type_from_label,
)
from zfp.core.config import DetectionConfig, ZfpConfig
from zfp.core.geometry import PageGeometry, Point, Rect
from zfp.core.types import FieldCandidate, FieldType, RasterWord, TextSpan, VectorPrimitive

PAGE_W = 612.0
PAGE_H = 792.0
BODY = 10.0
#: Helvetica ascent/descent as a fraction of the em, close enough for a test fixture.
ASCENT = 0.72
DESCENT = 0.21


def geometry(rotation: int = 0) -> PageGeometry:
    """A US Letter page."""
    box = Rect(0.0, 0.0, PAGE_W, PAGE_H)
    return PageGeometry(index=0, media_box=box, crop_box=box, rotation=rotation)


def span(
    text: str,
    x: float,
    baseline: float,
    width: Optional[float] = None,
    size: float = BODY,
    bold: bool = False,
    source: str = "native",
    confidence: float = 1.0,
) -> TextSpan:
    """A text span laid out the way a content stream reports one: box around a baseline."""
    advance = width if width is not None else 0.5 * size * len(text)
    rect = Rect(x, baseline - DESCENT * size, x + advance, baseline + ASCENT * size)
    return TextSpan(
        text=text,
        rect=rect,
        page=0,
        font_name="Helvetica-Bold" if bold else "Helvetica",
        font_size=size,
        source=source,
        confidence=confidence,
        baseline=baseline,
    )


def hline(x0: float, x1: float, y: float, width: float = 0.6) -> VectorPrimitive:
    """A stroked horizontal rule, with the endpoints a real content stream carries."""
    return VectorPrimitive(
        kind="line",
        rect=Rect(min(x0, x1), y, max(x0, x1), y),
        page=0,
        stroke_width=width,
        points=[Point(x0, y), Point(x1, y)],
    )


def vline(y0: float, y1: float, x: float, width: float = 0.6) -> VectorPrimitive:
    """A stroked vertical rule."""
    return VectorPrimitive(
        kind="line",
        rect=Rect(x, min(y0, y1), x, max(y0, y1)),
        page=0,
        stroke_width=width,
        points=[Point(x, y0), Point(x, y1)],
    )


def box(x0: float, y0: float, x1: float, y1: float, width: float = 0.6) -> VectorPrimitive:
    """A stroked rectangle."""
    return VectorPrimitive(
        kind="rect", rect=Rect(x0, y0, x1, y1), page=0, stroke_width=width, points=[]
    )


def circle(x0: float, y0: float, size: float, width: float = 0.6) -> VectorPrimitive:
    """A stroked circle, reported by its bounding box."""
    return VectorPrimitive(
        kind="circle",
        rect=Rect(x0, y0, x0 + size, y0 + size),
        page=0,
        stroke_width=width,
        points=[],
    )


def context(
    spans: Sequence[TextSpan] = (),
    prims: Sequence[VectorPrimitive] = (),
    words: Sequence[RasterWord] = (),
    widgets: Sequence[Rect] = (),
) -> CandidateContext:
    """Build a real context (vision layer included) for one hand-made page."""
    return build_context(
        0, geometry(), list(spans), list(prims), list(words), ZfpConfig.default(), list(widgets)
    )


class RectAssertions(unittest.TestCase):
    """Shared assertion helpers."""

    def assert_rect(self, actual: Rect, expected: Sequence[float], tol: float = 0.5) -> None:
        """Assert a rectangle matches ``[x0, y0, x1, y1]`` to within ``tol`` points."""
        for name, got, want in zip("xyXY", actual.as_list(), expected):
            self.assertAlmostEqual(
                got,
                want,
                delta=tol,
                msg="%s: %r != expected %r" % (name, actual.as_list(), list(expected)),
            )

    def labels(self, candidates: Sequence[FieldCandidate]) -> List[Optional[str]]:
        """The visible labels of ``candidates``, in order."""
        return [c.visible_label for c in candidates]


# ====================================================================== 1. underlines
class UnderlineFieldDetectorTests(RectAssertions):
    """``Label: ______``."""

    def test_single_underline_rect_type_and_label(self) -> None:
        ctx = context(
            [span("First Name:", 54.0, 702.0, width=56.0)],
            [hline(115.0, 300.0, 699.0)],
        )
        found = UnderlineFieldDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        # y0 = rule + underline_gap_pt; height = max(field_height_pt, 1.0 * 10) = 12
        self.assert_rect(found[0].rect, [115.0, 701.0, 300.0, 713.0])
        self.assertIs(found[0].field_type, FieldType.TEXT)
        self.assertEqual(found[0].visible_label, "First Name")
        self.assertAlmostEqual(found[0].confidence.geometry, 0.99, places=6)
        self.assertIn("vector_line", found[0].sources)
        self.assertIn("label_link", found[0].sources)
        self.assertIsNone(found[0].canonical_key)

    def test_two_fields_sharing_one_rule_are_split_and_labelled(self) -> None:
        # One rule runs the whole row; both labels are printed on top of it.
        spans = [
            span("First Name:", 54.0, 700.0, width=56.0),
            span("Last Name:", 300.0, 700.0, width=55.0),
        ]
        ctx = context(spans, [hline(54.0, 550.0, 699.0)])
        found = sorted(UnderlineFieldDetector().detect(ctx), key=lambda c: c.rect.x0)
        self.assertEqual(len(found), 2)
        self.assert_rect(found[0].rect, [110.0, 701.0, 300.0, 713.0])
        self.assert_rect(found[1].rect, [355.0, 701.0, 550.0, 713.0])
        self.assertEqual(self.labels(found), ["First Name", "Last Name"])
        for candidate in found:
            self.assertIs(candidate.field_type, FieldType.TEXT)

    def test_two_separate_rules_on_one_row(self) -> None:
        spans = [
            span("First Name:", 54.0, 702.0, width=56.0),
            span("Last Name:", 300.0, 702.0, width=55.0),
        ]
        prims = [hline(115.0, 280.0, 699.0), hline(360.0, 550.0, 699.0)]
        found = sorted(UnderlineFieldDetector().detect(context(spans, prims)), key=lambda c: c.rect.x0)
        self.assertEqual(len(found), 2)
        self.assert_rect(found[0].rect, [115.0, 701.0, 280.0, 713.0])
        self.assert_rect(found[1].rect, [360.0, 701.0, 550.0, 713.0])
        self.assertEqual(self.labels(found), ["First Name", "Last Name"])

    def test_rule_with_text_sitting_on_it_yields_nothing(self) -> None:
        ctx = context(
            [span("Employment History", 54.0, 702.0, width=136.0, size=12.0, bold=True)],
            [hline(54.0, 190.0, 699.0, width=0.8)],
        )
        self.assertEqual(UnderlineFieldDetector().detect(ctx), [])
        self.assertEqual(generate_candidates(ctx), [])

    def test_full_width_separator_yields_nothing(self) -> None:
        ctx = context(
            [span("Section A", 54.0, 722.0, width=48.0)],
            [hline(20.0, 592.0, 700.0, width=0.8)],
        )
        self.assertEqual(UnderlineFieldDetector().detect(ctx), [])

    def test_rule_spanning_the_whole_text_column_is_a_separator(self) -> None:
        spans = [span("Personal Details", 54.0, 722.0, width=90.0), span("Notes", 480.0, 722.0, width=28.0)]
        ctx = context(spans, [hline(54.0, 508.0, 700.0)])
        self.assertEqual(UnderlineFieldDetector().detect(ctx), [])

    def test_rule_shorter_than_the_minimum_is_ignored(self) -> None:
        ctx = context([span("No:", 54.0, 702.0, width=16.0)], [hline(80.0, 92.0, 699.0)])
        self.assertEqual(UnderlineFieldDetector().detect(ctx), [])

    def test_field_height_follows_a_larger_body_font(self) -> None:
        ctx = context(
            [span("Name:", 54.0, 702.0, width=40.0, size=16.0)],
            [hline(110.0, 320.0, 699.0)],
        )
        found = UnderlineFieldDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        # The font term only takes over above DetectionConfig.field_height_pt (12pt):
        # a 16pt face gets a 16pt field, a 10pt face keeps the configured 12.
        self.assertAlmostEqual(found[0].rect.height, FIELD_HEIGHT_FONT_RATIO * 16.0, delta=0.01)
        self.assertGreater(found[0].rect.height, DetectionConfig().field_height_pt)

    def test_max_chars_estimated_from_the_rule_width(self) -> None:
        ctx = context([span("Name:", 54.0, 702.0, width=40.0)], [hline(110.0, 350.0, 699.0)])
        found = UnderlineFieldDetector().detect(ctx)
        estimate = found[0].constraints.max_chars_estimate
        self.assertIsNotNone(estimate)
        self.assertTrue(40 <= estimate <= 60, estimate)

    def test_label_below_the_rule_is_found(self) -> None:
        # The signature-block idiom: the caption sits under the rule.
        ctx = context([span("Printed Name", 54.0, 674.0, width=62.0)], [hline(54.0, 290.0, 688.0)])
        found = UnderlineFieldDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].visible_label, "Printed Name")

    def test_existing_widget_suppresses_the_candidate(self) -> None:
        spans = [span("First Name:", 54.0, 702.0, width=56.0)]
        prims = [hline(115.0, 300.0, 699.0)]
        ctx = context(spans, prims, widgets=[Rect(115.0, 701.0, 300.0, 714.5)])
        self.assertEqual(UnderlineFieldDetector().detect(ctx), [])


# ============================================================================ 2. boxes
class BoxFieldDetectorTests(RectAssertions):
    """``Label [__________]``."""

    def test_box_is_inset_by_its_stroke_width(self) -> None:
        ctx = context(
            [span("Email:", 54.0, 697.0, width=32.0)],
            [box(120.0, 690.0, 320.0, 708.0, width=1.0)],
        )
        found = BoxFieldDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        self.assert_rect(found[0].rect, [121.0, 691.0, 319.0, 707.0])
        self.assertIs(found[0].field_type, FieldType.TEXT)
        self.assertEqual(found[0].visible_label, "Email")
        self.assertFalse(found[0].constraints.multiline)
        self.assertIn("vector_rect", found[0].sources)

    def test_tall_box_is_multiline(self) -> None:
        ctx = context(
            [span("Comments:", 54.0, 700.0, width=50.0)],
            [box(120.0, 650.0, 520.0, 710.0, width=1.0)],
        )
        found = BoxFieldDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].field_type, FieldType.MULTILINE_TEXT)
        self.assertTrue(found[0].constraints.multiline)

    def test_label_above_the_box(self) -> None:
        ctx = context(
            [span("Street Address:", 120.0, 716.0, width=72.0)],
            [box(120.0, 692.0, 400.0, 710.0, width=1.0)],
        )
        found = BoxFieldDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].visible_label, "Street Address")

    def test_checkbox_sized_box_is_not_a_text_field(self) -> None:
        ctx = context([span("Yes", 136.0, 602.0, width=16.0)], [box(122.0, 599.0, 132.0, 609.0)])
        self.assertEqual(BoxFieldDetector().detect(ctx), [])

    def test_table_frame_is_not_a_field(self) -> None:
        prims = _table_primitives()
        ctx = context(_table_header_spans(), prims)
        for candidate in BoxFieldDetector().detect(ctx):
            self.assertLess(candidate.rect.height, 40.0, candidate.rect)


# ======================================================================== 3. checkboxes
class CheckboxDetectorTests(RectAssertions):
    """``[ ] Yes   [ ] No``."""

    def test_two_checkboxes_with_labels_to_the_right(self) -> None:
        spans = [
            span("Subscribe?", 54.0, 602.0, width=56.0),
            span("Yes", 136.0, 602.0, width=16.0),
            span("No", 196.0, 602.0, width=12.0),
        ]
        prims = [box(122.0, 599.0, 132.0, 609.0), box(182.0, 599.0, 192.0, 609.0)]
        found = sorted(CheckboxDetector().detect(context(spans, prims)), key=lambda c: c.rect.x0)
        self.assertEqual(len(found), 2)
        self.assert_rect(found[0].rect, [122.0, 599.0, 132.0, 609.0])
        self.assert_rect(found[1].rect, [182.0, 599.0, 192.0, 609.0])
        for candidate in found:
            self.assertIs(candidate.field_type, FieldType.CHECKBOX)
        # Both boxes belong to one stem, so the stem names them and the words beside
        # them are their export values.
        self.assertEqual(self.labels(found), ["Subscribe?", "Subscribe?"])
        self.assertEqual([c.export_value for c in found], ["Yes", "No"])

    def test_checkbox_glyph_in_the_text_layer(self) -> None:
        spans = [span("☐", 122.0, 601.0, width=10.0), span("Agree", 136.0, 601.0, width=26.0)]
        found = CheckboxDetector().detect(context(spans, []))
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].field_type, FieldType.CHECKBOX)
        self.assertEqual(found[0].export_value, "Agree")

    def test_comb_cells_are_not_checkboxes(self) -> None:
        prims = [box(90.0 + i * 17.5, 496.0, 105.0 + i * 17.5, 512.0) for i in range(9)]
        ctx = context([span("SSN:", 54.0, 502.0, width=24.0)], prims)
        self.assertEqual(CheckboxDetector().detect(ctx), [])

    def test_label_falls_back_to_the_left(self) -> None:
        spans = [span("Confirmed", 54.0, 602.0, width=48.0)]
        ctx = context(spans, [box(122.0, 599.0, 132.0, 609.0)])
        found = CheckboxDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].visible_label, "Confirmed")


# ====================================================================== 4. radio groups
class RadioGroupDetectorTests(RectAssertions):
    """``Status: ( ) Single ( ) Married ( ) Other``."""

    def _context(self) -> CandidateContext:
        spans = [
            span("Status:", 54.0, 602.0, width=36.0),
            span("Single", 116.0, 602.0, width=29.0),
            span("Married", 186.0, 602.0, width=34.0),
            span("Other", 256.0, 602.0, width=29.0),
        ]
        prims = [circle(102.0, 599.0, 10.0), circle(172.0, 599.0, 10.0), circle(242.0, 599.0, 10.0)]
        return context(spans, prims)

    def test_three_circles_share_one_group_and_three_export_values(self) -> None:
        found = sorted(RadioGroupDetector().detect(self._context()), key=lambda c: c.rect.x0)
        self.assertEqual(len(found), 3)
        self.assertEqual(len({c.group_id for c in found}), 1)
        self.assertIsNotNone(found[0].group_id)
        self.assertEqual([c.export_value for c in found], ["Single", "Married", "Other"])
        self.assertEqual(len({c.export_value for c in found}), 3)
        for candidate in found:
            self.assertIs(candidate.field_type, FieldType.RADIO)
            self.assertEqual(candidate.visible_label, "Status")
            self.assertIsNone(candidate.canonical_key)
        self.assert_rect(found[0].rect, [102.0, 599.0, 112.0, 609.0])

    def test_radio_options_are_not_also_plain_checkboxes(self) -> None:
        self.assertEqual(CheckboxDetector().detect(self._context()), [])

    def test_a_lone_circle_is_not_a_group(self) -> None:
        spans = [span("Status:", 54.0, 602.0, width=36.0), span("Single", 116.0, 602.0, width=29.0)]
        ctx = context(spans, [circle(102.0, 599.0, 10.0)])
        self.assertEqual(RadioGroupDetector().detect(ctx), [])

    def test_a_run_without_a_stem_label_is_not_a_group(self) -> None:
        prims = [circle(102.0, 599.0, 10.0), circle(172.0, 599.0, 10.0)]
        self.assertEqual(RadioGroupDetector().detect(context([], prims)), [])


# ============================================================================= 5. combs
class CombFieldDetectorTests(RectAssertions):
    """``[ ][ ][ ][ ][ ][ ][ ][ ][ ]``."""

    def test_nine_cells_make_one_candidate(self) -> None:
        prims = [box(90.0 + i * 17.5, 496.0, 105.0 + i * 17.5, 512.0) for i in range(9)]
        ctx = context([span("SSN:", 54.0, 502.0, width=24.0)], prims)
        found = CombFieldDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].field_type, FieldType.COMB)
        self.assertEqual(found[0].constraints.comb_cells, 9)
        self.assertEqual(found[0].constraints.max_chars_estimate, 9)
        self.assertEqual(found[0].visible_label, "SSN")
        # union of the cells, inset by the stroke width
        self.assert_rect(found[0].rect, [90.6, 496.6, 244.4, 511.4])

    def test_divided_box_is_also_a_comb(self) -> None:
        prims = [box(90.0, 496.0, 225.0, 512.0)]
        prims += [vline(496.0, 512.0, 90.0 + i * 15.0) for i in range(1, 9)]
        ctx = context([span("Account:", 54.0, 502.0, width=40.0)], prims)
        found = CombFieldDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].constraints.comb_cells, 9)
        self.assert_rect(found[0].rect, [90.6, 496.6, 224.4, 511.4])

    def test_two_cells_are_not_a_comb(self) -> None:
        prims = [box(90.0, 496.0, 105.0, 512.0), box(107.5, 496.0, 122.5, 512.0)]
        ctx = context([span("ID:", 54.0, 502.0, width=16.0)], prims)
        self.assertEqual(CombFieldDetector().detect(ctx), [])


# ========================================================================= 6. date boxes
class DateBoxDetectorTests(RectAssertions):
    """``[  ] / [  ] / [    ]`` and a rule captioned ``MM/DD/YYYY``."""

    def test_three_groups_split_by_slashes(self) -> None:
        cells = [130.0, 146.0, 170.0, 186.0, 210.0, 226.0, 242.0, 258.0]
        prims = [box(x, 500.0, x + 14.0, 516.0) for x in cells]
        spans = [
            span("Date of Birth:", 54.0, 504.0, width=62.0),
            span("/", 163.0, 504.0, width=4.0),
            span("/", 203.0, 504.0, width=4.0),
        ]
        found = DateBoxDetector().detect(context(spans, prims))
        grouped = [c for c in found if c.constraints.comb_cells == 8]
        self.assertEqual(len(grouped), 1)
        self.assertIs(grouped[0].field_type, FieldType.DATE)
        self.assertEqual(grouped[0].constraints.format_hint, "MM/DD/YYYY")
        self.assert_rect(grouped[0].rect, [130.0, 500.0, 272.0, 516.0])
        self.assertEqual(grouped[0].visible_label, "Date of Birth")

    def test_rule_with_a_printed_date_placeholder(self) -> None:
        spans = [
            span("Date:", 54.0, 662.0, width=26.0),
            span("MM/DD/YYYY", 200.0, 645.0, width=54.0, size=8.0),
        ]
        found = DateBoxDetector().detect(context(spans, [hline(200.0, 400.0, 660.0)]))
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].field_type, FieldType.DATE)
        self.assertEqual(found[0].constraints.format_hint, "MM/DD/YYYY")
        self.assertGreater(found[0].confidence.semantic_type, 0.5)

    def test_plain_rule_is_not_a_date(self) -> None:
        ctx = context([span("City:", 54.0, 662.0, width=24.0)], [hline(90.0, 300.0, 660.0)])
        self.assertEqual(DateBoxDetector().detect(ctx), [])


# ========================================================================= 7. signatures
class SignatureLineDetectorTests(RectAssertions):
    """A rule captioned "Signature" -- and the "Date" rule beside it that is not one."""

    def _context(self) -> CandidateContext:
        spans = [span("Signature", 54.0, 674.0, width=44.0), span("Date", 320.0, 674.0, width=22.0)]
        prims = [hline(54.0, 290.0, 688.0), hline(320.0, 556.0, 688.0)]
        return context(spans, prims)

    def test_caption_below_the_rule(self) -> None:
        found = SignatureLineDetector().detect(self._context())
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].field_type, FieldType.SIGNATURE)
        self.assertEqual(found[0].visible_label, "Signature")
        self.assert_rect(found[0].rect, [54.0, 690.0, 290.0, 702.0])

    def test_adjacent_date_rule_stays_a_date(self) -> None:
        found = UnderlineFieldDetector().detect(self._context())
        by_x = sorted(found, key=lambda c: c.rect.x0)
        self.assertEqual(len(by_x), 2)
        self.assertIs(by_x[0].field_type, FieldType.SIGNATURE)
        self.assertIs(by_x[1].field_type, FieldType.DATE)
        self.assertEqual(by_x[1].visible_label, "Date")

    def test_caption_to_the_left(self) -> None:
        ctx = context([span("Initials:", 54.0, 662.0, width=36.0)], [hline(100.0, 200.0, 660.0)])
        found = SignatureLineDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].field_type, FieldType.SIGNATURE)

    def test_design_is_not_a_signature(self) -> None:
        ctx = context([span("Design Notes:", 54.0, 662.0, width=64.0)], [hline(130.0, 300.0, 660.0)])
        self.assertEqual(SignatureLineDetector().detect(ctx), [])


# ========================================================================= 8. table cells
def _table_primitives() -> List[VectorPrimitive]:
    """A three-column grid with a header row and two data rows."""
    left, right, top, row_h = 54.0, 558.0, 600.0, 18.0
    col_w = (right - left) / 3.0
    prims = [hline(left, right, top - i * row_h) for i in range(4)]
    prims += [vline(top - 3 * row_h, top, left + j * col_w) for j in range(4)]
    return prims


def _table_header_spans() -> List[TextSpan]:
    """The captions of the header row."""
    return [
        span("Employer", 58.0, 590.0, width=42.0, size=9.0),
        span("Role", 226.0, 590.0, width=22.0, size=9.0),
        span("Years", 394.0, 590.0, width=26.0, size=9.0),
    ]


class TableCellDetectorTests(RectAssertions):
    """Only the empty data cells become fields."""

    def test_header_row_is_skipped_and_data_cells_are_labelled(self) -> None:
        ctx = context(_table_header_spans(), _table_primitives())
        found = TableCellDetector().detect(ctx)
        self.assertEqual(len(found), 6)
        for candidate in found:
            self.assertIs(candidate.field_type, FieldType.TEXT)
            self.assertLess(candidate.rect.y1, 583.0, "a header cell became a field")
            self.assertIn("table_cell", candidate.sources)
        self.assertEqual(
            sorted({c.visible_label for c in found}), ["Employer", "Role", "Years"]
        )
        top_left = min(found, key=lambda c: (-c.rect.y1, c.rect.x0))
        self.assert_rect(top_left.rect, [54.6, 564.6, 221.4, 581.4])

    def test_cells_that_already_hold_text_are_skipped(self) -> None:
        spans = _table_header_spans() + [span("Acme Corp", 58.0, 570.0, width=48.0, size=9.0)]
        found = TableCellDetector().detect(context(spans, _table_primitives()))
        self.assertEqual(len(found), 5)

    def test_row_label_becomes_parent_context(self) -> None:
        spans = _table_header_spans() + [span("2019", 58.0, 570.0, width=20.0, size=9.0)]
        found = TableCellDetector().detect(context(spans, _table_primitives()))
        same_row = [c for c in found if abs(c.rect.y1 - 581.4) < 1.0]
        self.assertEqual(len(same_row), 2)
        for candidate in same_row:
            self.assertEqual(candidate.parent_context, ["2019"])

    def test_a_lone_grid_line_is_not_a_table(self) -> None:
        ctx = context([], [hline(54.0, 558.0, 600.0)])
        self.assertEqual(TableCellDetector().detect(ctx), [])


# ====================================================================== 9/10. whitespace
def borderless_context(
    regions: Sequence[Rect], spans: Sequence[TextSpan]
) -> CandidateContext:
    """A context whose blank regions are stated outright, independent of the vision layer."""
    return CandidateContext(
        page=0,
        geometry=geometry(),
        spans=list(spans),
        config=ZfpConfig.default(),
        blank_regions=list(regions),
        content_extent=Rect(54.0, 0.0, 558.0, PAGE_H),
    )


class BlankRegionDetectorTests(RectAssertions):
    """Borderless forms: a label and the whitespace next to it."""

    def test_blank_below_a_label(self) -> None:
        label = span("Employer Name", 54.0, 700.0, width=68.0)
        region = Rect(56.0, 682.0, 300.0, 697.0)
        found = BlankRegionDetector().detect(borderless_context([region], [label]))
        self.assertEqual(len(found), 1)
        self.assert_rect(found[0].rect, [56.0, 682.0, 300.0, 697.0])
        self.assertIs(found[0].field_type, FieldType.TEXT)
        self.assertEqual(found[0].visible_label, "Employer Name")
        self.assertTrue(0.5 <= found[0].confidence.geometry <= 0.75)
        self.assertIn("blank_region", found[0].sources)

    def test_blank_right_of_a_label_scores_higher(self) -> None:
        label = span("City", 54.0, 700.0, width=22.0)
        region = Rect(90.0, 696.0, 300.0, 710.0)
        found = BlankRegionDetector().detect(borderless_context([region], [label]))
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].confidence.geometry, 0.65, places=6)

    def test_unlabelled_whitespace_is_ignored(self) -> None:
        region = Rect(56.0, 400.0, 300.0, 415.0)
        self.assertEqual(BlankRegionDetector().detect(borderless_context([region], [])), [])

    def test_whitespace_already_claimed_by_a_rule_is_ignored(self) -> None:
        spans = [span("First Name:", 54.0, 702.0, width=56.0)]
        prims = [hline(115.0, 300.0, 699.0)]
        ctx = context(spans, prims)
        ctx.blank_regions = [Rect(115.0, 701.0, 300.0, 714.5)]
        ctx.cache.clear()
        self.assertEqual(BlankRegionDetector().detect(ctx), [])

    def test_region_outside_the_text_column_is_ignored(self) -> None:
        label = span("City", 54.0, 700.0, width=22.0)
        region = Rect(0.0, 696.0, 612.0, 710.0)
        self.assertEqual(BlankRegionDetector().detect(borderless_context([region], [label])), [])


class FreeTextAreaDetectorTests(RectAssertions):
    """Large empty blocks: "describe below"."""

    def test_large_block_under_a_prompt(self) -> None:
        label = span("Describe your responsibilities", 54.0, 700.0, width=150.0)
        region = Rect(54.0, 600.0, 500.0, 694.0)
        found = FreeTextAreaDetector().detect(borderless_context([region], [label]))
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].field_type, FieldType.MULTILINE_TEXT)
        self.assertTrue(found[0].constraints.multiline)
        self.assertEqual(found[0].visible_label, "Describe your responsibilities")
        self.assert_rect(found[0].rect, [54.0, 600.0, 500.0, 694.0])

    def test_narrow_block_is_not_a_free_text_area(self) -> None:
        label = span("Notes", 54.0, 700.0, width=28.0)
        region = Rect(54.0, 600.0, 180.0, 694.0)
        self.assertEqual(FreeTextAreaDetector().detect(borderless_context([region], [label])), [])

    def test_page_scale_void_is_not_a_field(self) -> None:
        label = span("Notes", 54.0, 700.0, width=28.0)
        region = Rect(54.0, 60.0, 558.0, 694.0)
        self.assertEqual(FreeTextAreaDetector().detect(borderless_context([region], [label])), [])


# ========================================================================= 11. colon runs
class ColonRunDetectorTests(RectAssertions):
    """The pure-text case: no vector geometry at all."""

    def test_underscore_run_after_a_label(self) -> None:
        text = "Name: ______________________"
        found = ColonRunDetector().detect(context([span(text, 54.0, 702.0, width=186.0)], []))
        self.assertEqual(len(found), 1)
        self.assertIs(found[0].field_type, FieldType.TEXT)
        self.assertEqual(found[0].visible_label, "Name")
        self.assertAlmostEqual(found[0].rect.x1, 240.0, delta=0.5)
        self.assertAlmostEqual(found[0].rect.y0, 702.0, delta=0.5)
        self.assertGreater(found[0].rect.width, 100.0)
        self.assertAlmostEqual(found[0].confidence.geometry, 0.70, places=6)

    def test_leader_dots(self) -> None:
        text = "Address ........................"
        found = ColonRunDetector().detect(context([span(text, 54.0, 682.0, width=196.0)], []))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].visible_label, "Address")

    def test_bare_colon_runs_to_the_next_span(self) -> None:
        # Three tightly stacked rows: the first two have no room underneath, so their
        # colons open an entry line to the right.
        spans = [
            span("City:", 54.0, 662.0, width=24.0),
            span("Notes", 400.0, 662.0, width=28.0),
            span("Country:", 54.0, 650.0, width=38.0),
            span("Region", 400.0, 650.0, width=32.0),
            span("Postcode:", 54.0, 638.0, width=44.0),
        ]
        found = sorted(ColonRunDetector().detect(context(spans, [])), key=lambda c: -c.rect.y1)
        self.assertEqual(self.labels(found), ["City", "Country"])
        self.assertAlmostEqual(found[0].rect.x1, 398.0, delta=0.5)
        self.assertAlmostEqual(found[0].confidence.geometry, 0.55, places=6)

    def test_a_label_with_writing_space_underneath_is_not_a_colon_field(self) -> None:
        # The borderless idiom: the entry area is under the label, not beside it.
        spans = [span("City:", 54.0, 662.0, width=24.0), span("Notes", 400.0, 662.0, width=28.0)]
        ctx = context(spans, [])
        self.assertTrue(ctx.label_blanks)
        self.assertEqual(ColonRunDetector().detect(ctx), [])

    def test_a_colon_over_a_vector_rule_is_left_to_the_rule_detector(self) -> None:
        spans = [span("City:", 54.0, 662.0, width=24.0)]
        found = ColonRunDetector().detect(context(spans, [hline(90.0, 300.0, 659.0)]))
        self.assertEqual(found, [])

    def test_ocr_words_drive_the_same_detector(self) -> None:
        words = [
            RasterWord(text="Name:", rect=Rect(54.0, 700.0, 90.0, 712.0), confidence=0.9, page=0),
            RasterWord(
                text="_____________", rect=Rect(96.0, 700.0, 260.0, 712.0), confidence=0.9, page=0
            ),
        ]
        ctx = build_context(0, geometry(), [], [], words, ZfpConfig.default())
        self.assertEqual(len(ctx.text_spans), 2)
        found = ColonRunDetector().detect(ctx)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].visible_label, "Name")
        self.assertIn("ocr_text", found[0].sources)


# ======================================================================= orchestration
class FailingDetector:
    """A detector that always raises, to prove one failure cannot sink the run."""

    name = "explode"

    def detect(self, ctx: CandidateContext) -> List[FieldCandidate]:
        """Always raise."""
        raise RuntimeError("boom")


class GenerateCandidatesTests(RectAssertions):
    """``generate_candidates`` -- ordering, tagging and fault tolerance."""

    def _page(self) -> CandidateContext:
        spans = [
            span("First Name:", 54.0, 702.0, width=56.0),
            span("Last Name:", 300.0, 702.0, width=55.0),
            span("Subscribe?", 54.0, 602.0, width=56.0),
            span("Yes", 136.0, 602.0, width=16.0),
        ]
        prims = [
            hline(115.0, 280.0, 699.0),
            hline(360.0, 550.0, 699.0),
            box(122.0, 599.0, 132.0, 609.0),
        ]
        return context(spans, prims)

    def test_a_failing_detector_does_not_lose_the_others(self) -> None:
        ctx = self._page()
        detectors = [FailingDetector(), UnderlineFieldDetector(), FailingDetector(), CheckboxDetector()]
        found = generate_candidates(ctx, detectors)
        self.assertEqual(len(found), 3)
        self.assertEqual(
            sorted(c.field_type.value for c in found), ["checkbox", "text", "text"]
        )

    def test_default_detectors_are_the_eleven_archetypes(self) -> None:
        self.assertEqual(len(DEFAULT_DETECTORS), 11)
        names = [d.name for d in DEFAULT_DETECTORS]
        self.assertEqual(len(set(names)), 11)
        self.assertEqual(names[0], "underline")
        self.assertEqual(names[-1], "colon_run")

    def test_reading_order_and_order_index(self) -> None:
        found = generate_candidates(self._page())
        self.assertTrue(found)
        keys = [(-c.rect.y1, c.rect.x0) for c in found]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual([c.order for c in found], list(range(len(found))))

    def test_evidence_is_tagged_with_the_detector(self) -> None:
        found = generate_candidates(self._page(), [UnderlineFieldDetector()])
        self.assertTrue(found)
        for candidate in found:
            self.assertTrue(candidate.evidence)
            for item in candidate.evidence:
                self.assertEqual(item.source_agent, "underline")

    def test_no_detector_sets_a_canonical_key(self) -> None:
        for candidate in generate_candidates(self._page()):
            self.assertIsNone(candidate.canonical_key)

    def test_results_are_deterministic(self) -> None:
        first = generate_candidates(self._page())
        second = generate_candidates(self._page())
        self.assertEqual(
            [(c.id, c.rect.as_list(), c.field_type) for c in first],
            [(c.id, c.rect.as_list(), c.field_type) for c in second],
        )

    def test_candidates_carry_stable_ids(self) -> None:
        found = generate_candidates(self._page())
        self.assertEqual(len({c.id for c in found}), len({c.rect.as_list()[0] for c in found}) or 1)
        for candidate in found:
            self.assertTrue(candidate.id.startswith("fc_"))


# ============================================================================= context
class CandidateContextTests(RectAssertions):
    """The shared, precomputed page context."""

    def test_invisible_spans_are_excluded_from_text_spans(self) -> None:
        visible = span("Name:", 54.0, 702.0, width=26.0)
        hidden = span("Name:", 54.0, 702.0, width=26.0, confidence=0.0)
        blank = span("   ", 200.0, 702.0, width=12.0)
        ctx = context([visible, hidden, blank], [])
        self.assertEqual(len(ctx.all_spans), 3)
        self.assertEqual(ctx.text_spans, [visible])

    def test_spans_near_is_a_working_spatial_index(self) -> None:
        spans = [span("A", 54.0, 700.0, width=10.0), span("B", 400.0, 700.0, width=10.0)]
        ctx = context(spans, [])
        near = ctx.spans_near(Rect(50.0, 695.0, 120.0, 715.0))
        self.assertEqual([s.text for s in near], ["A"])
        self.assertEqual(len(ctx.spans_near(Rect(0.0, 0.0, PAGE_W, PAGE_H))), 2)

    def test_median_font_size_and_stroke_width(self) -> None:
        spans = [span("A", 54.0, 700.0, width=10.0, size=9.0), span("B", 100.0, 700.0, width=10.0, size=11.0)]
        ctx = context(spans, [hline(54.0, 300.0, 690.0, width=0.8)])
        self.assertAlmostEqual(ctx.median_font_size, 10.0, places=6)
        self.assertAlmostEqual(ctx.median_stroke_width, 0.8, places=6)

    def test_derived_structures_are_shared_not_recomputed(self) -> None:
        ctx = context(_table_header_spans(), _table_primitives())
        self.assertGreaterEqual(len(ctx.table_cells), 9)
        self.assertEqual(ctx.page, 0)
        self.assertAlmostEqual(ctx.page_width, PAGE_W, places=6)
        first = ctx.h_rules
        self.assertIs(first, ctx.h_rules)

    def test_clamp_keeps_candidates_on_the_page(self) -> None:
        ctx = context([span("Name:", 54.0, 40.0, width=26.0)], [hline(90.0, 300.0, -4.0)])
        for candidate in generate_candidates(ctx):
            self.assertGreaterEqual(candidate.rect.x0, 0.0)
            self.assertLessEqual(candidate.rect.y1, PAGE_H)


class DegradedBackendTests(RectAssertions):
    """The detectors must survive a missing or broken vision layer."""

    def test_a_broken_vision_backend_degrades_instead_of_crashing(self) -> None:
        import zfp.candidates.context as context_module

        class Broken:
            """A vision module whose every function raises."""

            def __getattr__(self, name: str):
                def boom(*args: object, **kwargs: object) -> None:
                    raise RuntimeError("vision is down")

                return boom

        saved = (context_module.vision_primitives, context_module.vision_blanks)
        context_module.vision_primitives = Broken()
        context_module.vision_blanks = Broken()
        try:
            ctx = context(
                [span("First Name:", 54.0, 702.0, width=56.0)], [hline(115.0, 300.0, 699.0)]
            )
            found = UnderlineFieldDetector().detect(ctx)
        finally:
            context_module.vision_primitives, context_module.vision_blanks = saved
        self.assertEqual(len(found), 1)
        self.assert_rect(found[0].rect, [115.0, 701.0, 300.0, 713.0])
        self.assertEqual(found[0].visible_label, "First Name")

    def test_a_missing_vision_backend_degrades_instead_of_crashing(self) -> None:
        import zfp.candidates.context as context_module

        saved = (context_module.vision_primitives, context_module.vision_blanks)
        context_module.vision_primitives = None
        context_module.vision_blanks = None
        try:
            ctx = context(
                [span("Email:", 54.0, 697.0, width=32.0)],
                [box(120.0, 690.0, 320.0, 708.0, width=1.0)],
            )
            found = BoxFieldDetector().detect(ctx)
        finally:
            context_module.vision_primitives, context_module.vision_blanks = saved
        self.assertEqual(len(found), 1)
        self.assert_rect(found[0].rect, [121.0, 691.0, 319.0, 707.0])

    def test_an_empty_page_produces_nothing(self) -> None:
        ctx = context([], [])
        self.assertEqual(generate_candidates(ctx), [])
        self.assertEqual(ctx.text_spans, [])
        self.assertEqual(ctx.blank_regions, [])


# ============================================================================= helpers
class LabelHelperTests(unittest.TestCase):
    """The small text helpers the detectors share."""

    def test_clean_label(self) -> None:
        self.assertEqual(clean_label("  First Name:  "), "First Name")
        self.assertEqual(clean_label("Signature ____"), "Signature")
        self.assertEqual(clean_label("• Notes ......"), "Notes")
        self.assertEqual(clean_label(None), "")

    def test_export_value_for(self) -> None:
        self.assertEqual(export_value_for("Yes"), "Yes")
        self.assertEqual(export_value_for("n"), "No")
        self.assertEqual(export_value_for("Married Filing Jointly"), "Married_Filing_Jointly")
        self.assertEqual(export_value_for(""), "On")

    def test_signature_and_date_word_matching(self) -> None:
        self.assertTrue(looks_like_signature("Signature of Applicant"))
        self.assertTrue(looks_like_signature("Initials"))
        self.assertFalse(looks_like_signature("Design Review"))
        self.assertTrue(looks_like_date("Date of Birth"))
        self.assertFalse(looks_like_date("Candidate Name"))

    def test_type_from_label(self) -> None:
        self.assertIs(type_from_label("Signature"), FieldType.SIGNATURE)
        self.assertIs(type_from_label("Start Date"), FieldType.DATE)
        self.assertIs(type_from_label("City"), FieldType.TEXT)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
