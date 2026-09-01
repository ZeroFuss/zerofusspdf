"""Unit tests for :mod:`zfp.core.types`."""

from __future__ import annotations

import dataclasses
import math
import unittest

from zfp.core.config import ScoringWeights
from zfp.core.geometry import PageGeometry, Point, Rect
from zfp.core.types import (
    CONFIDENCE_WEIGHTS,
    EVIDENCE_BUCKETS,
    SCORE_KEYS,
    Confidence,
    DocumentClass,
    DocumentProfile,
    Evidence,
    EvidenceKind,
    FieldCandidate,
    FieldConstraints,
    FieldSpec,
    FieldType,
    FilledValue,
    FillReport,
    FormSchema,
    PageMode,
    PageProfile,
    RasterWord,
    TextSpan,
    VectorPrimitive,
)

GEOM = PageGeometry(0, Rect(0, 0, 612, 792), Rect(0, 0, 612, 792), 0)


class EnumTest(unittest.TestCase):
    def test_field_type_values_are_strings(self) -> None:
        self.assertEqual(FieldType.TEXT, "text")
        self.assertEqual(FieldType("multiline_text"), FieldType.MULTILINE_TEXT)
        self.assertEqual(len(FieldType), 15)

    def test_pdf_kind_mapping(self) -> None:
        expected = {
            FieldType.TEXT: "Tx",
            FieldType.MULTILINE_TEXT: "Tx",
            FieldType.DATE: "Tx",
            FieldType.NUMBER: "Tx",
            FieldType.CURRENCY: "Tx",
            FieldType.EMAIL: "Tx",
            FieldType.PHONE: "Tx",
            FieldType.COMB: "Tx",
            FieldType.UNKNOWN: "Tx",
            FieldType.CHECKBOX: "Btn",
            FieldType.RADIO: "Btn",
            FieldType.BUTTON: "Btn",
            FieldType.CHOICE: "Ch",
            FieldType.LISTBOX: "Ch",
            FieldType.SIGNATURE: "Sig",
        }
        self.assertEqual(set(expected), set(FieldType))
        for field_type, kind in expected.items():
            self.assertEqual(field_type.pdf_kind, kind, field_type.value)

    def test_other_enums(self) -> None:
        self.assertEqual(PageMode.SCANNED_FORM, "scanned_form")
        self.assertEqual(DocumentClass.EXISTING_ACROFORM, "existing_acroform")
        self.assertEqual(EvidenceKind.CHECKBOX_GLYPH, "checkbox_glyph")
        self.assertEqual(len(EvidenceKind), 15)


class ConfidenceTest(unittest.TestCase):
    def test_default_is_zero(self) -> None:
        self.assertEqual(Confidence().overall(), 0.0)

    def test_single_measured_axis_returns_that_axis(self) -> None:
        self.assertAlmostEqual(Confidence(geometry=0.75).overall(), 0.75)
        self.assertAlmostEqual(Confidence(autofill_value=0.4).overall(), 0.4)

    def test_weighted_geometric_mean_formula(self) -> None:
        c = Confidence(geometry=0.9, label_link=0.8, semantic_type=0.7, autofill_value=0.6)
        num = (
            CONFIDENCE_WEIGHTS["geometry"] * math.log(0.9)
            + CONFIDENCE_WEIGHTS["label_link"] * math.log(0.8)
            + CONFIDENCE_WEIGHTS["semantic_type"] * math.log(0.7)
            + CONFIDENCE_WEIGHTS["autofill_value"] * math.log(0.6)
        )
        self.assertAlmostEqual(c.overall(), math.exp(num / sum(CONFIDENCE_WEIGHTS.values())))

    def test_zero_axis_does_not_annihilate_the_score(self) -> None:
        c = Confidence(geometry=0.95, label_link=0.9, semantic_type=0.0, autofill_value=0.0)
        self.assertGreater(c.overall(), 0.9)

    def test_all_ones_is_one(self) -> None:
        self.assertAlmostEqual(Confidence(1.0, 1.0, 1.0, 1.0).overall(), 1.0)

    def test_geometric_mean_punishes_a_weak_axis(self) -> None:
        balanced = Confidence(0.7, 0.7, 0.7, 0.7).overall()
        lopsided = Confidence(1.0, 1.0, 1.0, 0.05).overall()
        self.assertLess(lopsided, balanced)

    def test_values_are_clamped(self) -> None:
        self.assertAlmostEqual(Confidence(geometry=5.0).overall(), 1.0)
        self.assertEqual(Confidence(geometry=-1.0).overall(), 0.0)

    def test_as_dict_round_trip(self) -> None:
        c = Confidence(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(
            c.as_dict(),
            {"geometry": 0.1, "label_link": 0.2, "semantic_type": 0.3, "autofill_value": 0.4},
        )
        self.assertEqual(Confidence.from_dict(c.as_dict()), c)


class FieldConstraintsTest(unittest.TestCase):
    def test_defaults_are_independent(self) -> None:
        a, b = FieldConstraints(), FieldConstraints()
        a.choices.append("x")
        self.assertEqual(b.choices, [])

    def test_as_dict(self) -> None:
        fc = FieldConstraints(max_chars_estimate=48, required=True, choices=["a", "b"])
        d = fc.as_dict()
        self.assertEqual(d["max_chars_estimate"], 48)
        self.assertTrue(d["required"])
        self.assertEqual(d["choices"], ["a", "b"])
        self.assertEqual(FieldConstraints.from_dict(d), fc)


class TextSpanTest(unittest.TestCase):
    def test_is_blank(self) -> None:
        self.assertTrue(TextSpan("", Rect(0, 0, 1, 1), 0).is_blank())
        self.assertTrue(TextSpan("   \t\n", Rect(0, 0, 1, 1), 0).is_blank())
        self.assertFalse(TextSpan(" x ", Rect(0, 0, 1, 1), 0).is_blank())

    def test_normalized_text(self) -> None:
        cases = {
            "Applicant's Name*:": "applicants name",
            "  FIRST   NAME  ": "first name",
            "E-Mail": "e mail",
            "First/Last": "first last",
            "Date_of_Birth": "date of birth",
            "ZIP (5-digit)": "zip 5 digit",
            "": "",
        }
        for raw, want in cases.items():
            self.assertEqual(TextSpan(raw, Rect(0, 0, 1, 1), 0).normalized_text(), want, raw)

    def test_defaults_are_independent(self) -> None:
        a = TextSpan("a", Rect(0, 0, 1, 1), 0)
        b = TextSpan("b", Rect(0, 0, 1, 1), 0)
        a.glyph_rects.append(Rect(0, 0, 1, 1))
        self.assertEqual(b.glyph_rects, [])

    def test_as_dict(self) -> None:
        span = TextSpan("hi", Rect(1, 2, 3, 4), 2, font_size=9.5, source="ocr", confidence=0.8)
        d = span.as_dict()
        self.assertEqual(d["rect"], [1, 2, 3, 4])
        self.assertEqual(d["source"], "ocr")
        self.assertEqual(d["page"], 2)


class VectorPrimitiveTest(unittest.TestCase):
    def test_orientation(self) -> None:
        cases = [
            (Rect(0, 100, 200, 100.6), "horizontal"),
            (Rect(0, 100, 200, 100), "horizontal"),
            (Rect(50, 0, 50.5, 300), "vertical"),
            (Rect(50, 0, 50, 300), "vertical"),
            (Rect(0, 0, 20, 20), "other"),
            (Rect(0, 0, 30, 20), "other"),
            (Rect(0, 0, 0, 0), "other"),
        ]
        for rect, want in cases:
            prim = VectorPrimitive(kind="line", rect=rect, page=0)
            self.assertEqual(prim.orientation(), want, str(rect))

    def test_defaults_are_independent(self) -> None:
        a = VectorPrimitive("line", Rect(0, 0, 1, 1), 0)
        b = VectorPrimitive("line", Rect(0, 0, 1, 1), 0)
        a.points.append(Point(0, 0))
        self.assertEqual(b.points, [])

    def test_as_dict(self) -> None:
        prim = VectorPrimitive("line", Rect(0, 10, 100, 10), 1, stroke_width=0.6)
        d = prim.as_dict()
        self.assertEqual(d["orientation"], "horizontal")
        self.assertEqual(d["stroke_width"], 0.6)


class RasterWordTest(unittest.TestCase):
    def test_as_dict(self) -> None:
        w = RasterWord("Name", Rect(1, 2, 3, 4), 0.92, 0, alternatives=[("Nome", 0.3)])
        d = w.as_dict()
        self.assertEqual(d["text"], "Name")
        self.assertEqual(d["alternatives"], [["Nome", 0.3]])

    def test_defaults_are_independent(self) -> None:
        a = RasterWord("a", Rect(0, 0, 1, 1), 1.0, 0)
        b = RasterWord("b", Rect(0, 0, 1, 1), 1.0, 0)
        a.alternatives.append(("x", 0.1))
        self.assertEqual(b.alternatives, [])
        self.assertEqual(a.line_id, -1)
        self.assertEqual(a.block_id, -1)


class FieldCandidateTest(unittest.TestCase):
    def make(self) -> FieldCandidate:
        return FieldCandidate(id="fc_1", page=0, rect=Rect(10, 20, 110, 34))

    def test_defaults(self) -> None:
        c = self.make()
        self.assertEqual(c.field_type, FieldType.UNKNOWN)
        self.assertEqual(c.sources, [])
        self.assertEqual(c.evidence, [])
        self.assertEqual(c.confidence, Confidence())
        self.assertEqual(c.constraints, FieldConstraints())

    def test_mutable_defaults_are_independent(self) -> None:
        a, b = self.make(), self.make()
        a.sources.append("vector_line")
        a.parent_context.append("Applicant")
        a.confidence.geometry = 0.9
        a.constraints.choices.append("x")
        self.assertEqual(b.sources, [])
        self.assertEqual(b.parent_context, [])
        self.assertEqual(b.confidence.geometry, 0.0)
        self.assertEqual(b.constraints.choices, [])

    def test_add_evidence_tracks_sources_without_duplicates(self) -> None:
        c = self.make()
        c.add_evidence(Evidence(EvidenceKind.VECTOR_LINE, 0.9))
        c.add_evidence(Evidence(EvidenceKind.VECTOR_LINE, 0.4))
        c.add_evidence(Evidence(EvidenceKind.NATIVE_TEXT, 0.8))
        self.assertEqual(c.sources, ["vector_line", "native_text"])
        self.assertEqual(len(c.evidence), 3)

    def test_evidence_scores_bucket_mapping(self) -> None:
        expected_buckets = {
            EvidenceKind.VECTOR_LINE: "geometric_evidence",
            EvidenceKind.VECTOR_RECT: "geometric_evidence",
            EvidenceKind.VECTOR_CIRCLE: "geometric_evidence",
            EvidenceKind.EXISTING_WIDGET: "geometric_evidence",
            EvidenceKind.CHECKBOX_GLYPH: "geometric_evidence",
            EvidenceKind.TABLE_CELL: "geometric_evidence",
            EvidenceKind.COMB_CELL: "geometric_evidence",
            EvidenceKind.BLANK_REGION: "blank_region_evidence",
            EvidenceKind.LABEL_LINK: "nearby_label_evidence",
            EvidenceKind.LAYOUT: "layout_consistency",
            EvidenceKind.REPEAT: "repeated_pattern_evidence",
            EvidenceKind.PATTERN: "semantic_type_confidence",
            EvidenceKind.NATIVE_TEXT: "semantic_type_confidence",
            EvidenceKind.OCR_TEXT: "semantic_type_confidence",
            EvidenceKind.MODEL: "model_consensus",
        }
        self.assertEqual(EVIDENCE_BUCKETS, expected_buckets)
        for kind, bucket in expected_buckets.items():
            c = self.make()
            c.add_evidence(Evidence(kind, 0.5))
            scores = c.evidence_scores()
            self.assertEqual(scores[bucket], 0.5, kind.value)
            self.assertEqual(sum(scores.values()), 0.5, kind.value)

    def test_evidence_scores_takes_the_maximum_per_bucket(self) -> None:
        c = self.make()
        c.add_evidence(Evidence(EvidenceKind.VECTOR_LINE, 0.4))
        c.add_evidence(Evidence(EvidenceKind.TABLE_CELL, 0.91))
        c.add_evidence(Evidence(EvidenceKind.VECTOR_RECT, 0.6))
        c.add_evidence(Evidence(EvidenceKind.PATTERN, 0.3))
        c.add_evidence(Evidence(EvidenceKind.OCR_TEXT, 0.75))
        scores = c.evidence_scores()
        self.assertAlmostEqual(scores["geometric_evidence"], 0.91)
        self.assertAlmostEqual(scores["semantic_type_confidence"], 0.75)
        self.assertEqual(scores["model_consensus"], 0.0)

    def test_evidence_scores_keys_match_scoring_weights(self) -> None:
        weight_names = tuple(f.name for f in dataclasses.fields(ScoringWeights))
        self.assertEqual(SCORE_KEYS, weight_names)
        self.assertEqual(set(self.make().evidence_scores()), set(weight_names))

    def test_evidence_scores_feed_scoring_weights(self) -> None:
        c = self.make()
        c.add_evidence(Evidence(EvidenceKind.VECTOR_LINE, 1.0))
        c.add_evidence(Evidence(EvidenceKind.LABEL_LINK, 1.0))
        self.assertAlmostEqual(ScoringWeights().score(c.evidence_scores()), 0.45)

    def test_as_dict_round_trip(self) -> None:
        c = self.make()
        c.field_type = FieldType.DATE
        c.visible_label = "Date of Birth"
        c.canonical_key = "person.dob"
        c.parent_context = ["Applicant"]
        c.confidence = Confidence(0.99, 0.98, 0.97, 0.0)
        c.constraints = FieldConstraints(max_chars_estimate=10, format_hint="MM/DD/YYYY")
        c.add_evidence(Evidence(EvidenceKind.VECTOR_LINE, 0.9, "underline", "VectorAgent",
                                Rect(10, 19, 110, 20)))
        c.group_id = "g1"
        c.export_value = "On"
        c.order = 7
        d = c.as_dict()
        self.assertEqual(d["field_type"], "date")
        self.assertEqual(d["rect"], [10, 20, 110, 34])
        self.assertEqual(d["evidence"][0]["kind"], "vector_line")
        self.assertEqual(FieldCandidate.from_dict(d), c)


class ProfileTest(unittest.TestCase):
    def test_page_profile_as_dict(self) -> None:
        p = PageProfile(index=3, geometry=GEOM, mode=PageMode.SCANNED_FORM, has_raster=True)
        d = p.as_dict()
        self.assertEqual(d["mode"], "scanned_form")
        self.assertEqual(d["geometry"]["crop_box"], [0, 0, 612, 792])
        self.assertEqual(d["geometry"]["rotation"], 0)

    def test_document_profile_page_views(self) -> None:
        pages = [
            PageProfile(0, GEOM, PageMode.NATIVE_DOCUMENT, has_native_text=True),
            PageProfile(1, GEOM, PageMode.SCANNED_FORM, has_raster=True),
            PageProfile(2, GEOM, PageMode.HYBRID, has_native_text=True, has_raster=True),
        ]
        prof = DocumentProfile(document_id="doc", page_count=3, pages=pages)
        self.assertEqual(prof.native_text_pages, [0, 2])
        self.assertEqual(prof.raster_pages, [1, 2])
        self.assertEqual(prof.doc_class, DocumentClass.NON_FORM)
        d = prof.as_dict()
        self.assertEqual(d["doc_class"], "non_form")
        self.assertEqual(len(d["pages"]), 3)
        self.assertTrue(d["can_modify"])

    def test_document_profile_warnings_are_independent(self) -> None:
        a = DocumentProfile("a", 0, [])
        b = DocumentProfile("b", 0, [])
        a.warnings.append("boom")
        self.assertEqual(b.warnings, [])


class FieldSpecTest(unittest.TestCase):
    def make(self) -> FieldSpec:
        return FieldSpec(
            name="applicant.name",
            field_type=FieldType.TEXT,
            page=0,
            rect=Rect(72, 700, 300, 714),
        )

    def test_defaults(self) -> None:
        spec = self.make()
        self.assertEqual(spec.font_name, "Helv")
        self.assertEqual(spec.font_size, 0.0)
        self.assertEqual(spec.text_color, (0.0, 0.0, 0.0))
        self.assertIsNone(spec.border_color)
        self.assertEqual(spec.choices, [])
        self.assertEqual(spec.extra_widgets, [])
        self.assertEqual(spec.pdf_kind, "Tx")

    def test_widgets(self) -> None:
        spec = self.make()
        spec.extra_widgets.append((2, Rect(72, 100, 300, 114)))
        self.assertEqual(
            spec.widgets(), [(0, Rect(72, 700, 300, 714)), (2, Rect(72, 100, 300, 114))]
        )

    def test_mutable_defaults_are_independent(self) -> None:
        a, b = self.make(), self.make()
        a.choices.append("Yes")
        a.extra_widgets.append((1, Rect(0, 0, 1, 1)))
        self.assertEqual(b.choices, [])
        self.assertEqual(b.extra_widgets, [])

    def test_as_dict_round_trip(self) -> None:
        spec = self.make()
        spec.field_type = FieldType.CHOICE
        spec.choices = ["A", "B"]
        spec.border_color = (0.2, 0.2, 0.2)
        spec.extra_widgets = [(1, Rect(10, 20, 30, 40))]
        spec.max_length = 32
        d = spec.as_dict()
        self.assertEqual(d["field_type"], "choice")
        self.assertEqual(d["border_color"], [0.2, 0.2, 0.2])
        self.assertEqual(d["extra_widgets"], [[1, [10, 20, 30, 40]]])
        self.assertEqual(FieldSpec.from_dict(d), spec)


class FormSchemaTest(unittest.TestCase):
    def make(self) -> FormSchema:
        return FormSchema(
            document_id="doc1",
            fields=[
                FieldSpec("a", FieldType.TEXT, 0, Rect(0, 0, 10, 10)),
                FieldSpec("b", FieldType.CHECKBOX, 1, Rect(0, 0, 10, 10)),
                FieldSpec(
                    "c",
                    FieldType.TEXT,
                    0,
                    Rect(0, 0, 10, 10),
                    extra_widgets=[(1, Rect(5, 5, 15, 15))],
                ),
            ],
        )

    def test_by_name(self) -> None:
        schema = self.make()
        self.assertIsNotNone(schema.by_name("b"))
        self.assertEqual(schema.by_name("b").field_type, FieldType.CHECKBOX)  # type: ignore[union-attr]
        self.assertIsNone(schema.by_name("missing"))

    def test_by_page_includes_extra_widgets(self) -> None:
        schema = self.make()
        self.assertEqual([f.name for f in schema.by_page(0)], ["a", "c"])
        self.assertEqual([f.name for f in schema.by_page(1)], ["b", "c"])
        self.assertEqual(schema.by_page(9), [])

    def test_round_trip(self) -> None:
        schema = self.make()
        candidate = FieldCandidate(id="fc_1", page=0, rect=Rect(1, 2, 3, 4))
        candidate.add_evidence(Evidence(EvidenceKind.BLANK_REGION, 0.6))
        schema.source_candidates.append(candidate)
        self.assertEqual(FormSchema.from_dict(schema.as_dict()), schema)

    def test_defaults_are_independent(self) -> None:
        a, b = FormSchema("x"), FormSchema("y")
        a.fields.append(FieldSpec("f", FieldType.TEXT, 0, Rect(0, 0, 1, 1)))
        self.assertEqual(b.fields, [])


class FillReportTest(unittest.TestCase):
    def test_filled_value_defaults(self) -> None:
        v = FilledValue("a", "person.name", "Ada", 0.99)
        self.assertEqual(v.status, "filled")
        self.assertEqual(v.provenance, {})
        self.assertEqual(v.reason_codes, [])

    def test_recount(self) -> None:
        report = FillReport(document_id="d")
        report.values = [
            FilledValue("a", None, "x", 0.99),
            FilledValue("b", None, None, 0.0, status="unavailable"),
            FilledValue("c", None, None, 0.4, status="low_confidence"),
        ]
        report.recount()
        self.assertEqual(report.filled_count, 1)
        self.assertEqual(report.unresolved_count, 2)

    def test_as_dict(self) -> None:
        report = FillReport(document_id="d")
        report.values.append(
            FilledValue("a", "person.name", "Ada", 0.99, provenance={"source": "vault"})
        )
        report.recount()
        d = report.as_dict()
        self.assertEqual(d["filled_count"], 1)
        self.assertEqual(d["values"][0]["provenance"], {"source": "vault"})

    def test_defaults_are_independent(self) -> None:
        a, b = FillReport("a"), FillReport("b")
        a.values.append(FilledValue("x", None, None, 0.0))
        self.assertEqual(b.values, [])


class EvidenceTest(unittest.TestCase):
    def test_frozen_and_hashable_fields(self) -> None:
        e = Evidence(EvidenceKind.PATTERN, 0.95, detail="ssn", source_agent="RulesMember")
        with self.assertRaises(Exception):
            e.score = 0.1  # type: ignore[misc]
        self.assertEqual(e.as_dict()["kind"], "pattern")
        self.assertIsNone(e.as_dict()["rect"])
        self.assertEqual(Evidence.from_dict(e.as_dict()), e)

    def test_round_trip_with_rect(self) -> None:
        e = Evidence(EvidenceKind.VECTOR_RECT, 0.5, rect=Rect(1, 2, 3, 4))
        self.assertEqual(Evidence.from_dict(e.as_dict()), e)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
