"""Unit tests for :mod:`zfp.synth` -- the generator, its painter and its layouts.

The synthetic corpus is the yardstick every detector is scored against, so these tests
are deliberately paranoid about the two properties that make it a yardstick at all: the
PDFs are real and reopen through the project's own parser, and the ground-truth
rectangles are exact, disjoint, on-page and ontology-backed.
"""

from __future__ import annotations

import json
import os
import random
import re
import tempfile
import unittest

from tests.fixtures.factory import build, cache_size, clear_cache, corpus
from zfp.core.errors import ValidationError
from zfp.core.geometry import Rect
from zfp.core.serde import dumps, loads
from zfp.core.types import FieldType
from zfp.ontology import CANONICAL_KEYS, lookup
from zfp.pdfio.document import Document
from zfp.pdfio.fonts import font_ascent, font_descent, text_width
from zfp.pdfio.objects import PdfDict, PdfName, PdfStream
from zfp.synth import (
    KINDS,
    PAGE_KINDS,
    ContentBuilder,
    GroundTruthField,
    SyntheticForm,
    SynthOptions,
    attach_page_content,
    build_style,
    draw_page,
    font_dictionary,
    generate,
    generate_corpus,
    spec_for,
    specs_for,
)
from zfp.synth.layouts import (
    CHECKBOX_GROUPS,
    COMB_FIELDS,
    DET,
    LABEL_GAP,
    MAX_FIELDS_PER_PAGE,
    MIN_FIELDS_PER_PAGE,
    REPEAT_LABELS,
    RULE_DROP,
    SECTIONS,
    SIGNATURE_LABELS,
    TABLE_COLUMNS,
    Canvas,
    Style,
    group_checkbox,
    row_boxed,
    row_underline,
)

SEEDS = (0, 1, 2)


def all_forms(seeds=SEEDS):
    """Every kind at every seed, from the shared fixture cache."""
    return [build(kind, seed) for kind in KINDS for seed in seeds]


# --------------------------------------------------------------------------------------
# content.py
# --------------------------------------------------------------------------------------


class ContentBuilderTest(unittest.TestCase):
    def test_starts_empty(self):
        builder = ContentBuilder()
        self.assertTrue(builder.is_empty())
        self.assertEqual(builder.build(), b"")
        self.assertEqual(builder.fonts_used, {})

    def test_every_method_chains(self):
        builder = ContentBuilder()
        result = (
            builder.save()
            .gray(0.5)
            .setlinewidth(1.0)
            .text(10, 20, "hi")
            .line(0, 0, 10, 0)
            .rect(1, 2, 3, 4)
            .circle(5, 5, 2)
            .dotted_line(0, 1, 9, 1)
            .restore()
        )
        self.assertIs(result, builder)
        self.assertFalse(builder.is_empty())

    def test_text_emits_a_text_object_and_records_the_font(self):
        builder = ContentBuilder()
        builder.text(72.0, 700.0, "First Name:", font_res="F3", size=11.0, base_font="Arial")
        out = builder.build()
        self.assertIn(b"BT /F3 11 Tf 1 0 0 1 72 700 Tm (First Name:) Tj ET", out)
        # Arial resolves onto the base-14 face it substitutes for.
        self.assertEqual(builder.fonts_used, {"F3": "Helvetica"})

    def test_text_escapes_parentheses_and_high_bytes(self):
        builder = ContentBuilder()
        builder.text(0, 0, "a(b)c\\d")
        self.assertIn(rb"(a\(b\)c\\d) Tj", builder.build())

    def test_empty_text_draws_nothing(self):
        builder = ContentBuilder()
        builder.text(0, 0, "")
        self.assertTrue(builder.is_empty())

    def test_numbers_never_use_scientific_notation(self):
        builder = ContentBuilder()
        builder.line(0.0000001, 1e7, 2.5, 3.25, width=0.25)
        out = builder.build()
        self.assertNotIn(b"e", out.replace(b"re", b"").replace(b"line", b""))
        self.assertIn(b"2.5 3.25 l", out)

    def test_line_is_stroked_inside_its_own_graphics_state(self):
        out = ContentBuilder().line(1, 2, 3, 4, width=0.7).build()
        self.assertEqual(out, b"q 0.7 w 1 2 m 3 4 l S Q\n")

    def test_rect_stroke_fill_and_both(self):
        stroked = ContentBuilder().rect(1, 2, 3, 4, width=0.6).build()
        self.assertIn(b"1 2 3 4 re S", stroked)
        filled = ContentBuilder().rect(1, 2, 3, 4, width=0.6, fill=0.9).build()
        self.assertIn(b"0.9 g", filled)
        self.assertIn(b"re B", filled)
        fill_only = ContentBuilder().rect(1, 2, 3, 4, width=0.0, fill=0.0).build()
        self.assertIn(b"re f", fill_only)

    def test_circle_is_four_bezier_arcs(self):
        out = ContentBuilder().circle(100, 100, 10).build()
        self.assertEqual(out.count(b" c"), 4)
        self.assertIn(b"110 100 m", out)
        self.assertIn(b"h S", out)

    def test_zero_radius_circle_draws_nothing(self):
        self.assertTrue(ContentBuilder().circle(1, 1, 0).is_empty())

    def test_dotted_line_sets_a_dash_pattern(self):
        out = ContentBuilder().dotted_line(0, 0, 10, 0, dash=(2, 3)).build()
        self.assertIn(b"[2 3] 0 d", out)

    def test_gray_sets_fill_and_stroke(self):
        self.assertIn(b"0.4 g 0.4 G", ContentBuilder().gray(0.4).build())
        self.assertIn(b"1 g 1 G", ContentBuilder().gray(7.0).build())     # clamped

    def test_width_of_matches_the_font_metrics(self):
        builder = ContentBuilder()
        self.assertAlmostEqual(
            builder.width_of("Employer", "Helvetica", 10.0),
            text_width("Employer", "Helvetica", 10.0),
            places=9,
        )


class AttachPageContentTest(unittest.TestCase):
    def _drawn(self):
        builder = ContentBuilder()
        builder.text(72, 700, "Street Address:", font_res="F1", size=10, base_font="Times-Roman")
        builder.line(160, 697, 400, 697, 0.6)
        return builder

    def test_round_trips_through_a_real_save_and_reopen(self):
        builder = self._drawn()
        doc = Document.from_pages_blank(1)
        attach_page_content(doc, 0, builder.build(), builder.fonts_used)
        reopened = Document.open(doc.to_bytes(incremental=False))
        content = reopened.page(0).content_bytes()
        self.assertIn(b"(Street Address:) Tj", content)
        self.assertIn(b"160 697 m", content)

    def test_builds_the_font_resources(self):
        builder = self._drawn()
        doc = Document.from_pages_blank(1)
        attach_page_content(doc, 0, builder.build(), builder.fonts_used)
        reopened = Document.open(doc.to_bytes(incremental=False))
        resources = reopened.page(0).resources()
        fonts = reopened.resolve(resources.get("Font"))
        self.assertIsInstance(fonts, PdfDict)
        font = reopened.resolve(fonts.get("F1"))
        self.assertEqual(font.get_name("BaseFont"), "Times-Roman")
        self.assertEqual(font.get_name("Subtype"), "Type1")
        self.assertEqual(font.get_name("Encoding"), "WinAnsiEncoding")

    def test_reuses_the_existing_content_object_number(self):
        doc = Document.from_pages_blank(1)
        before = doc.page(0).dict.get("Contents")
        ref = attach_page_content(doc, 0, b"q Q\n", {})
        self.assertEqual(ref.num, before.num)

    def test_compressed_streams_decode_transparently(self):
        builder = self._drawn()
        doc = Document.from_pages_blank(1)
        attach_page_content(doc, 0, builder.build(), builder.fonts_used, compress=True)
        data = doc.to_bytes(incremental=False)
        reopened = Document.open(data)
        stream = reopened.resolve(reopened.page(0).dict.get("Contents"))
        self.assertIsInstance(stream, PdfStream)
        self.assertEqual(stream.dict.get_name("Filter"), "FlateDecode")
        self.assertIn(b"(Street Address:) Tj", reopened.page(0).content_bytes())

    def test_accepts_pairs_as_well_as_a_mapping(self):
        doc = Document.from_pages_blank(1)
        attach_page_content(doc, 0, b"q Q\n", [("F1", "Courier")])
        fonts = doc.resolve(doc.page(0).resources().get("Font"))
        self.assertEqual(doc.resolve(fonts.get("F1")).get_name("BaseFont"), "Courier")

    def test_rejects_a_malformed_font_map(self):
        doc = Document.from_pages_blank(1)
        with self.assertRaises(ValidationError):
            attach_page_content(doc, 0, b"", ["F1"])
        with self.assertRaises(ValidationError):
            attach_page_content(doc, 0, b"", {"": "Helvetica"})

    def test_symbolic_fonts_keep_their_builtin_encoding(self):
        self.assertNotIn("Encoding", font_dictionary("ZapfDingbats"))
        self.assertEqual(font_dictionary("Symbol").get("BaseFont"), PdfName("Symbol"))


# --------------------------------------------------------------------------------------
# layouts.py
# --------------------------------------------------------------------------------------


class VocabularyTest(unittest.TestCase):
    """Every label the layouts can draw must resolve to a canonical key."""

    def _labels(self):
        labels = []
        for _, section in SECTIONS:
            labels.extend(section)
        labels.extend(stem for stem, _ in CHECKBOX_GROUPS)
        labels.extend(label for label, _ in COMB_FIELDS)
        labels.extend(TABLE_COLUMNS)
        labels.extend(SIGNATURE_LABELS)
        labels.extend(REPEAT_LABELS)
        return labels

    def test_every_drawable_label_resolves(self):
        unresolved = [label for label in self._labels() if lookup(label) is None]
        self.assertEqual(unresolved, [])

    def test_specs_for_drops_unknown_labels(self):
        specs = specs_for(["First Name", "Zzzz Not A Real Label", "City"])
        self.assertEqual([s.label for s in specs], ["First Name", "City"])

    def test_spec_carries_the_ontology_field_type(self):
        self.assertEqual(spec_for("Email Address").field_type, FieldType.EMAIL)
        self.assertEqual(spec_for("Date of Birth").field_type, FieldType.DATE)
        self.assertIsNone(spec_for("Zzzz Not A Real Label"))

    def test_section_names_are_unique(self):
        names = [name for name, _ in SECTIONS]
        self.assertEqual(len(names), len(set(names)))


class ExactRectangleTest(unittest.TestCase):
    """The ground truth is arithmetic on the drawing numbers, not a measurement."""

    def _canvas(self, **overrides):
        style = Style(
            base_font="Helvetica",
            bold_font="Helvetica-Bold",
            font_size=10.0,
            line_width=0.6,
            row_gap=10.0,
            margin=54.0,
            page_width=612.0,
            page_height=792.0,
            **overrides,
        )
        return Canvas(style, random.Random(0), 0)

    def test_underline_rect_sits_above_the_rule(self):
        cv = self._canvas()
        spec = spec_for("First Name")
        self.assertTrue(row_underline(cv, [spec]))
        baseline = 792.0 - 54.0 - 10.0
        rule_y = baseline - RULE_DROP
        x0 = 54.0 + text_width("First Name:", "Helvetica", 10.0) + LABEL_GAP
        expected = Rect(
            x0,
            rule_y + DET.underline_gap_pt,
            558.0,
            rule_y + DET.underline_gap_pt + DET.field_height_pt,
        )
        mark = cv.marks[0]
        for got, want in zip(mark.rect.as_list(), expected.as_list()):
            self.assertAlmostEqual(got, want, places=3)
        self.assertEqual(mark.canonical_key, "person.name.first")
        self.assertIn(b"725 m", cv.b.build())

    def test_boxed_rect_is_the_box_deflated_by_the_stroke(self):
        cv = self._canvas(box_label="above")
        spec = spec_for("City")
        self.assertTrue(row_boxed(cv, [spec], True))
        box_h = max(DET.field_height_pt + 4.0, 15.0)
        top = 792.0 - 54.0 - 10.0 - 4.0
        expected = Rect(54.0, top - box_h, 558.0, top).inflated(-0.6)
        for got, want in zip(cv.marks[0].rect.as_list(), expected.as_list()):
            self.assertAlmostEqual(got, want, places=3)

    def test_checkbox_rect_is_the_glyph_box(self):
        cv = self._canvas(checkbox_side=10.0)
        stem = spec_for("Marital Status")
        self.assertTrue(group_checkbox(cv, stem, ("Single", "Married"), False))
        self.assertEqual(len(cv.marks), 2)
        for mark in cv.marks:
            self.assertEqual(mark.field_type, FieldType.CHECKBOX)
            self.assertAlmostEqual(mark.rect.width, 10.0, places=6)
            self.assertAlmostEqual(mark.rect.height, 10.0, places=6)
            self.assertEqual(mark.canonical_key, "person.marital_status")
        self.assertEqual([m.expected_value for m in cv.marks], ["Single", "Married"])

    def test_radio_groups_draw_circles(self):
        cv = self._canvas()
        group_checkbox(cv, spec_for("Gender"), ("Male", "Female"), True)
        self.assertIn(b" c", cv.b.build())
        self.assertTrue(all(m.field_type == FieldType.RADIO for m in cv.marks))

    def test_borderless_fields_leave_no_ink(self):
        rng = random.Random(11)
        style = build_style(rng)
        drawing = draw_page("borderless", rng, style, 0, title="Loan Application")
        content = drawing.content
        self.assertNotIn(b" re ", content)      # no boxes anywhere on the page
        self.assertGreaterEqual(len(drawing.marks), MIN_FIELDS_PER_PAGE)

    def test_unknown_page_kind_is_rejected(self):
        rng = random.Random(0)
        with self.assertRaises(ValidationError):
            draw_page("nope", rng, build_style(rng), 0)


class PageTemplateTest(unittest.TestCase):
    def test_every_template_produces_a_realistic_field_count(self):
        for kind in PAGE_KINDS:
            for seed in range(4):
                rng = random.Random(seed)
                drawing = draw_page(kind, rng, build_style(rng), 0, title="Patient Intake Form")
                with self.subTest(kind=kind, seed=seed):
                    self.assertGreaterEqual(len(drawing.marks), MIN_FIELDS_PER_PAGE)
                    self.assertLessEqual(len(drawing.marks), MAX_FIELDS_PER_PAGE)
                    self.assertTrue(drawing.fonts)

    def test_headings_are_suppressed_on_request(self):
        headings = [name for name, _ in SECTIONS]
        for seed in range(3):
            rng = random.Random(seed)
            drawing = draw_page(
                "underline", rng, build_style(rng), 0, title="Loan Application",
                include_sections=False,
            )
            for heading in headings:
                self.assertNotIn(heading.encode("ascii"), drawing.content)


# --------------------------------------------------------------------------------------
# generator.py -- the documents themselves
# --------------------------------------------------------------------------------------


class DocumentIntegrityTest(unittest.TestCase):
    def test_every_kind_reopens_with_the_expected_page_count(self):
        for form in all_forms():
            with self.subTest(kind=form.kind, seed=form.seed):
                self.assertTrue(form.pdf_bytes.startswith(b"%PDF-"))
                doc = Document.open(form.pdf_bytes)
                self.assertEqual(doc.page_count, form.pages)
                self.assertFalse(doc.file.is_encrypted)
                self.assertFalse(doc.has_xfa())
                self.assertEqual(doc.existing_fields(), [])

    def test_multipage_documents_have_several_pages(self):
        form = build("multipage", 0)
        self.assertGreaterEqual(form.pages, 3)
        self.assertEqual(Document.open(form.pdf_bytes).page_count, form.pages)

    def test_explicit_page_count_is_honoured(self):
        form = build("underline", 0, pages=4)
        self.assertEqual(form.pages, 4)
        self.assertEqual(Document.open(form.pdf_bytes).page_count, 4)
        self.assertEqual({f.page for f in form.fields}, {0, 1, 2, 3})

    def test_every_page_carries_fields(self):
        for form in all_forms():
            for page in range(form.pages):
                with self.subTest(kind=form.kind, seed=form.seed, page=page):
                    count = len(form.fields_on_page(page))
                    self.assertGreaterEqual(count, MIN_FIELDS_PER_PAGE)
                    self.assertLessEqual(count, MAX_FIELDS_PER_PAGE)

    def test_field_names_are_unique(self):
        for form in all_forms():
            names = [f.name for f in form.fields]
            with self.subTest(kind=form.kind, seed=form.seed):
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(all(name and " " not in name for name in names))


class GroundTruthGeometryTest(unittest.TestCase):
    def test_rects_are_on_the_page_and_non_degenerate(self):
        for form in all_forms():
            doc = Document.open(form.pdf_bytes)
            for page in range(form.pages):
                crop = doc.page(page).geometry.crop_box
                for gt in form.fields_on_page(page):
                    with self.subTest(kind=form.kind, seed=form.seed, field=gt.name):
                        self.assertTrue(crop.contains_rect(gt.rect), gt.rect.as_list())
                        self.assertGreater(gt.rect.width, 1.0)
                        self.assertGreater(gt.rect.height, 1.0)
                        self.assertLessEqual(gt.rect.x0, gt.rect.x1)
                        self.assertLessEqual(gt.rect.y0, gt.rect.y1)

    def test_rects_on_a_page_do_not_overlap(self):
        for form in all_forms():
            for page in range(form.pages):
                fields = form.fields_on_page(page)
                for i in range(len(fields)):
                    for j in range(i + 1, len(fields)):
                        iou = fields[i].rect.iou(fields[j].rect)
                        with self.subTest(kind=form.kind, a=fields[i].name, b=fields[j].name):
                            self.assertLess(iou, 0.05)

    def test_text_fields_are_wide_enough_to_be_detectable(self):
        small = (FieldType.CHECKBOX, FieldType.RADIO)
        for form in all_forms():
            for gt in form.fields:
                if gt.field_type in small:
                    self.assertGreaterEqual(gt.rect.width, DET.checkbox_min_pt)
                    self.assertLessEqual(gt.rect.width, DET.checkbox_max_pt)
                else:
                    self.assertGreaterEqual(gt.rect.width, DET.min_line_length_pt)

    def test_fields_are_in_reading_order(self):
        for form in all_forms():
            for page in range(form.pages):
                tops = [round(f.rect.y1, 3) for f in form.fields_on_page(page)]
                self.assertEqual(tops, sorted(tops, reverse=True))


class OntologyTruthTest(unittest.TestCase):
    def test_every_field_has_a_real_canonical_key(self):
        for form in all_forms():
            for gt in form.fields:
                with self.subTest(kind=form.kind, field=gt.name):
                    self.assertIn(gt.canonical_key, CANONICAL_KEYS)

    def test_the_label_still_resolves_to_that_key(self):
        for form in all_forms():
            for gt in form.fields:
                self.assertEqual(lookup(gt.label), gt.canonical_key)

    def test_field_types_are_real_enum_members(self):
        for form in all_forms():
            for gt in form.fields:
                self.assertIsInstance(gt.field_type, FieldType)
                self.assertTrue(gt.field_type.pdf_kind)

    def test_checkbox_kind_yields_button_fields(self):
        form = build("checkbox", 0)
        kinds = {gt.field_type for gt in form.fields}
        self.assertTrue(kinds & {FieldType.CHECKBOX, FieldType.RADIO})
        for gt in form.fields:
            if gt.field_type in (FieldType.CHECKBOX, FieldType.RADIO):
                self.assertEqual(gt.field_type.pdf_kind, "Btn")
                self.assertTrue(gt.expected_value)

    def test_comb_kind_yields_comb_fields(self):
        form = build("comb", 0)
        self.assertTrue(all(gt.field_type == FieldType.COMB for gt in form.fields))

    def test_signature_kind_yields_a_signature_field(self):
        form = build("signature", 0)
        self.assertIn(FieldType.SIGNATURE, {gt.field_type for gt in form.fields})


class ContentRoundTripTest(unittest.TestCase):
    def test_drawn_labels_survive_the_parser(self):
        for form in all_forms():
            doc = Document.open(form.pdf_bytes)
            for page in range(form.pages):
                content = doc.page(page).content_bytes()
                for gt in form.fields_on_page(page):
                    with self.subTest(kind=form.kind, field=gt.name):
                        self.assertIn(gt.label.encode("ascii"), content)

    def test_underline_rules_are_really_stroked(self):
        form = build("underline", 1)
        content = Document.open(form.pdf_bytes).page(0).content_bytes()
        self.assertGreaterEqual(content.count(b" l S Q"), len(form.fields))

    def test_boxed_forms_stroke_rectangles(self):
        form = build("boxed", 1)
        content = Document.open(form.pdf_bytes).page(0).content_bytes()
        self.assertGreaterEqual(content.count(b" re "), len(form.fields))


_TEXT_OP = re.compile(
    rb"BT /(\w+) ([\d.]+) Tf 1 0 0 1 ([-\d.]+) ([-\d.]+) Tm \((.*?)\) Tj ET", re.S
)
_LINE_OP = re.compile(rb"q ([\d.]+) w ([-\d.]+) ([-\d.]+) m ([-\d.]+) ([-\d.]+) l S Q")


def _text_spans(doc, page_index):
    """Re-read the drawn text back out of a page as ``(text, rect)`` pairs."""
    page = doc.page(page_index)
    fonts = doc.resolve(page.resources().get("Font")) or {}
    base = {name: doc.resolve(ref).get_name("BaseFont") for name, ref in fonts.items()}
    spans = []
    for match in _TEXT_OP.finditer(page.content_bytes()):
        resource, size, x, y, payload = match.groups()
        size, x, y = float(size), float(x), float(y)
        text = (
            payload.replace(rb"\(", b"(").replace(rb"\)", b")").decode("latin-1")
        )
        font = base[resource.decode("ascii")]
        spans.append(
            (
                text,
                Rect(
                    x,
                    y + font_descent(font) * size / 1000.0,
                    x + text_width(text, font, size),
                    y + font_ascent(font) * size / 1000.0,
                ),
            )
        )
    return spans


class BlankFieldTest(unittest.TestCase):
    """A field a detector is meant to find must contain no ink of its own."""

    def test_no_drawn_text_intrudes_into_a_field(self):
        for form in all_forms(seeds=(0, 1)):
            doc = Document.open(form.pdf_bytes)
            for page in range(form.pages):
                spans = _text_spans(doc, page)
                self.assertTrue(spans)
                for gt in form.fields_on_page(page):
                    for text, rect in spans:
                        overlap = gt.rect.intersection(rect)
                        area = 0.0 if overlap is None else overlap.area
                        with self.subTest(kind=form.kind, field=gt.name, text=text):
                            self.assertLess(area, 0.5)

    def test_every_underline_field_sits_exactly_on_its_rule(self):
        for seed in SEEDS:
            form = build("underline", seed)
            doc = Document.open(form.pdf_bytes)
            for page in range(form.pages):
                rules = set()
                for match in _LINE_OP.finditer(doc.page(page).content_bytes()):
                    _, x0, y0, x1, y1 = (float(v) for v in match.groups())
                    if abs(y0 - y1) < 1e-9:
                        rules.add((round(x0, 3), round(x1, 3), round(y0, 3)))
                for gt in form.fields_on_page(page):
                    expected = (
                        round(gt.rect.x0, 3),
                        round(gt.rect.x1, 3),
                        round(gt.rect.y0 - DET.underline_gap_pt, 3),
                    )
                    with self.subTest(seed=seed, field=gt.name):
                        self.assertIn(expected, rules)
                        self.assertAlmostEqual(gt.rect.height, DET.field_height_pt, places=6)


class DeterminismTest(unittest.TestCase):
    def test_the_same_seed_gives_identical_bytes(self):
        for kind in KINDS:
            first = generate(SynthOptions(kind=kind, seed=17))
            second = generate(SynthOptions(kind=kind, seed=17))
            with self.subTest(kind=kind):
                self.assertEqual(first.pdf_bytes, second.pdf_bytes)
                self.assertEqual(first.truth_dict(), second.truth_dict())

    def test_different_seeds_give_different_bytes(self):
        for kind in KINDS:
            seen = {}
            for seed in range(6):
                data = build(kind, seed).pdf_bytes
                with self.subTest(kind=kind, seed=seed):
                    self.assertNotIn(data, seen)
                seen[data] = seed

    def test_the_global_random_module_is_never_used(self):
        random.seed(1234)
        first = generate(SynthOptions(kind="mixed", seed=3)).pdf_bytes
        random.seed(999999)
        [random.random() for _ in range(50)]
        second = generate(SynthOptions(kind="mixed", seed=3)).pdf_bytes
        self.assertEqual(first, second)

    def test_generation_order_does_not_matter(self):
        forward = [generate(SynthOptions(kind="mixed", seed=s)).pdf_bytes for s in (1, 2, 3)]
        backward = [generate(SynthOptions(kind="mixed", seed=s)).pdf_bytes for s in (3, 2, 1)]
        self.assertEqual(forward, list(reversed(backward)))


class RotationTest(unittest.TestCase):
    def test_rotate_is_written_on_every_page(self):
        for rotation in (0, 90, 180, 270):
            form = build("mixed", 5, rotation=rotation)
            doc = Document.open(form.pdf_bytes)
            for page in range(form.pages):
                with self.subTest(rotation=rotation, page=page):
                    self.assertEqual(doc.page(page).geometry.rotation, rotation)

    def test_ground_truth_stays_in_unrotated_user_space(self):
        upright = build("mixed", 5, rotation=0)
        for rotation in (90, 180, 270):
            rotated = build("mixed", 5, rotation=rotation)
            with self.subTest(rotation=rotation):
                self.assertEqual(
                    [f.as_dict() for f in upright.fields],
                    [f.as_dict() for f in rotated.fields],
                )
                self.assertNotEqual(upright.pdf_bytes, rotated.pdf_bytes)

    def test_negative_rotation_is_normalized(self):
        form = build("underline", 0, rotation=-90)
        self.assertEqual(Document.open(form.pdf_bytes).page(0).geometry.rotation, 270)


class RepeatedFieldTest(unittest.TestCase):
    def test_multipage_repeats_labels_across_pages(self):
        for seed in range(4):
            form = build("multipage", seed)
            pages_by_label = {}
            for gt in form.fields:
                pages_by_label.setdefault(gt.label, set()).add(gt.page)
            repeated = {k: v for k, v in pages_by_label.items() if len(v) >= 2}
            with self.subTest(seed=seed):
                self.assertTrue(repeated, "no label appeared on two pages")
                self.assertGreaterEqual(len(repeated), len(REPEAT_LABELS))

    def test_repeated_labels_keep_one_canonical_key(self):
        form = build("multipage", 1)
        by_label = {}
        for gt in form.fields:
            by_label.setdefault(gt.label, set()).add(gt.canonical_key)
        for label, keys in by_label.items():
            self.assertEqual(len(keys), 1, label)


class OptionsTest(unittest.TestCase):
    def test_defaults_match_the_contract(self):
        options = SynthOptions()
        self.assertEqual(options.kind, "underline")
        self.assertEqual(options.pages, 1)
        self.assertEqual(options.seed, 0)
        self.assertEqual(options.font, "Helvetica")
        self.assertEqual(options.font_size, 10.0)
        self.assertEqual(options.line_width, 0.6)
        self.assertTrue(options.include_sections)
        self.assertEqual(options.rotation, 0)
        self.assertEqual(options.locale, "en_US")

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValidationError):
            generate(SynthOptions(kind="spirals"))

    def test_bad_page_count_is_rejected(self):
        with self.assertRaises(ValidationError):
            generate(SynthOptions(pages=0))

    def test_bad_rotation_is_rejected(self):
        with self.assertRaises(ValidationError):
            generate(SynthOptions(rotation=45))

    def test_tiny_pages_are_rejected(self):
        with self.assertRaises(ValidationError):
            generate(SynthOptions(page_width=100.0))

    def test_a_pinned_font_is_honoured_everywhere(self):
        form = build("underline", 4, font="Times-Roman")
        doc = Document.open(form.pdf_bytes)
        fonts = doc.resolve(doc.page(0).resources().get("Font"))
        used = {doc.resolve(ref).get_name("BaseFont") for ref in fonts.values()}
        self.assertTrue(used)
        self.assertTrue(all(name.startswith("Times") for name in used), used)

    def test_a_pinned_size_and_stroke_reach_the_content_stream(self):
        form = build("underline", 4, font_size=8.0, line_width=0.9)
        content = Document.open(form.pdf_bytes).page(0).content_bytes()
        self.assertIn(b" 8 Tf", content)
        self.assertIn(b"0.9 w", content)

    def test_custom_page_size_is_used(self):
        form = build("underline", 0, page_width=595.0, page_height=842.0)
        geometry = Document.open(form.pdf_bytes).page(0).geometry
        self.assertAlmostEqual(geometry.width, 595.0, places=6)
        self.assertAlmostEqual(geometry.height, 842.0, places=6)


class TruthDictTest(unittest.TestCase):
    def test_shape_and_json_round_trip(self):
        form = build("mixed", 2)
        truth = form.truth_dict()
        self.assertEqual(set(truth), {"kind", "seed", "fields"})
        self.assertEqual(truth["kind"], "mixed")
        self.assertEqual(truth["seed"], 2)
        self.assertEqual(len(truth["fields"]), len(form.fields))
        decoded = json.loads(json.dumps(truth))
        self.assertEqual(decoded, truth)

    def test_field_entries_carry_everything_a_scorer_needs(self):
        entry = build("underline", 0).truth_dict()["fields"][0]
        self.assertEqual(
            set(entry),
            {"name", "canonical_key", "field_type", "page", "rect", "label", "expected_value"},
        )
        self.assertEqual(len(entry["rect"]), 4)
        self.assertIsInstance(entry["field_type"], str)

    def test_serde_round_trip(self):
        form = build("comb", 0)
        restored = loads(dumps(form.truth_dict()))
        self.assertEqual(restored, form.truth_dict())

    def test_expected_values_are_present_and_short(self):
        form = build("underline", 0)
        filled = [gt for gt in form.fields if gt.expected_value]
        self.assertTrue(filled)
        for gt in filled:
            self.assertLess(len(gt.expected_value), 80)


class SaveTest(unittest.TestCase):
    def test_save_writes_a_reopenable_file(self):
        form = build("boxed", 0)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "form.pdf")
            form.save(path)
            self.assertEqual(os.path.getsize(path), len(form.pdf_bytes))
            self.assertEqual(Document.open(path).page_count, form.pages)

    def test_document_helper_reopens_the_bytes(self):
        form = build("underline", 0)
        self.assertEqual(form.document().page_count, form.pages)

    def test_labels_helper(self):
        form = build("underline", 0)
        self.assertEqual(form.labels(), [gt.label for gt in form.fields])


class CorpusTest(unittest.TestCase):
    def test_generate_corpus_cycles_kinds_and_advances_the_seed(self):
        forms = generate_corpus(6, seed=100, kinds=["underline", "boxed"])
        self.assertEqual([f.kind for f in forms], ["underline", "boxed"] * 3)
        self.assertEqual([f.seed for f in forms], list(range(100, 106)))
        self.assertEqual(len({f.pdf_bytes for f in forms}), 6)

    def test_generate_corpus_defaults_to_every_kind(self):
        forms = generate_corpus(len(KINDS))
        self.assertEqual([f.kind for f in forms], list(KINDS))

    def test_generate_corpus_is_deterministic(self):
        first = [f.pdf_bytes for f in generate_corpus(3, seed=5)]
        second = [f.pdf_bytes for f in generate_corpus(3, seed=5)]
        self.assertEqual(first, second)

    def test_generate_corpus_rejects_bad_arguments(self):
        self.assertEqual(generate_corpus(0), [])
        with self.assertRaises(ValidationError):
            generate_corpus(-1)
        with self.assertRaises(ValidationError):
            generate_corpus(1, kinds=[])
        with self.assertRaises(ValidationError):
            generate_corpus(1, kinds=["origami"])


class DataclassTest(unittest.TestCase):
    def test_ground_truth_field_is_constructible_by_hand(self):
        gt = GroundTruthField(
            name="person_name_first",
            canonical_key="person.name.first",
            field_type=FieldType.TEXT,
            page=0,
            rect=Rect(10, 20, 110, 32),
            label="First Name",
        )
        self.assertEqual(gt.expected_value, "")
        self.assertEqual(gt.as_dict()["rect"], [10.0, 20.0, 110.0, 32.0])

    def test_synthetic_form_is_constructible_by_hand(self):
        form = SyntheticForm(pdf_bytes=b"%PDF-1.7\n", fields=[], seed=3, kind="boxed")
        self.assertEqual(form.truth_dict(), {"kind": "boxed", "seed": 3, "fields": []})
        self.assertEqual(form.pages, 1)


class FactoryTest(unittest.TestCase):
    def test_build_is_memoized(self):
        clear_cache()
        first = build("underline", 42)
        self.assertEqual(cache_size(), 1)
        self.assertIs(build("underline", 42), first)
        self.assertEqual(cache_size(), 1)
        self.assertIsNot(build("underline", 43), first)

    def test_keyword_arguments_take_part_in_the_key(self):
        clear_cache()
        plain = build("underline", 7)
        rotated = build("underline", 7, rotation=90)
        self.assertIsNot(plain, rotated)
        self.assertIs(build("underline", 7, rotation=90), rotated)

    def test_corpus_returns_consecutive_seeds(self):
        forms = corpus("boxed", 4, seed=20)
        self.assertEqual([f.seed for f in forms], [20, 21, 22, 23])
        self.assertTrue(all(f.kind == "boxed" for f in forms))

    def test_corpus_of_zero_is_empty(self):
        self.assertEqual(corpus("boxed", 0), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
