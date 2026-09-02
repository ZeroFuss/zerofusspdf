"""Unit tests for :mod:`zfp.qa.verify`."""

from __future__ import annotations

import unittest

from zfp.acroform.writer import AcroFormWriter
from zfp.core.geometry import Rect
from zfp.core.types import FieldSpec, FieldType, FormSchema
from zfp.pdfio.document import Document
from zfp.qa.verify import (
    check_field_names_unique, check_fields_roundtrip, check_in_page_bounds,
    check_integrity, check_no_overlap, check_prefix_preserved, verify_document,
)


class PrefixPreservedTests(unittest.TestCase):
    def test_passes_on_a_true_prefix(self):
        findings = check_prefix_preserved(b"abc", b"abcdef")
        self.assertTrue(all(f.severity != "error" for f in findings))

    def test_fails_with_correct_offset(self):
        findings = check_prefix_preserved(b"abcdef", b"abcXef")
        self.assertTrue(any(f.severity == "error" for f in findings))
        self.assertIn("offset 3", findings[0].message)


class NoOverlapTests(unittest.TestCase):
    def test_flags_50_percent_overlap_passes_at_5_percent(self):
        a = FieldSpec(name="a", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 100, 100))
        b = FieldSpec(name="b", field_type=FieldType.TEXT, page=0, rect=Rect(50, 0, 150, 100))
        schema = FormSchema(document_id="d", fields=[a, b])
        findings = check_no_overlap(schema)
        self.assertTrue(any(f.code == "FIELD_OVERLAP" for f in findings))

        c = FieldSpec(name="c", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 100, 100))
        d = FieldSpec(name="d", field_type=FieldType.TEXT, page=0, rect=Rect(190, 0, 290, 100))
        schema2 = FormSchema(document_id="d", fields=[c, d])
        findings2 = check_no_overlap(schema2)
        self.assertFalse(any(f.code == "FIELD_OVERLAP" for f in findings2))


class InPageBoundsTests(unittest.TestCase):
    def test_flags_rect_off_page(self):
        doc = Document.from_pages_blank(1, width=612, height=792)
        spec = FieldSpec(name="a", field_type=FieldType.TEXT, page=0,
                         rect=Rect(500, 700, 900, 720))
        schema = FormSchema(document_id="d", fields=[spec])
        findings = check_in_page_bounds(doc, schema)
        self.assertTrue(any(f.code == "FIELD_OUT_OF_BOUNDS" for f in findings))


class IntegrityTests(unittest.TestCase):
    def test_valid_pdf_has_no_error_findings(self):
        doc = Document.from_pages_blank(1)
        data = doc.to_bytes(incremental=False)
        findings = check_integrity(data)
        self.assertFalse(any(f.severity == "error" for f in findings))


class FieldsRoundtripTests(unittest.TestCase):
    def test_detects_missing_changed_value_and_missing_appearance(self):
        doc = Document.from_pages_blank(1)
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="a", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 100, 14),
                     value="x"),
        ])
        AcroFormWriter(doc).write(schema)
        produced = doc.to_bytes(incremental=False)

        ghost_schema = FormSchema(document_id="d", fields=list(schema.fields) + [
            FieldSpec(name="ghost", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 1, 1)),
        ])
        findings = check_fields_roundtrip(produced, ghost_schema)
        self.assertTrue(any(f.code == "FIELD_MISSING" for f in findings))

        wrong_value_schema = FormSchema(document_id="d", fields=[
            FieldSpec(name="a", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 100, 14),
                     value="different"),
        ])
        findings2 = check_fields_roundtrip(produced, wrong_value_schema)
        self.assertTrue(any(f.code == "FIELD_VALUE_MISMATCH" for f in findings2))


class DuplicateNamesTests(unittest.TestCase):
    def test_flags_duplicate_names(self):
        schema = FormSchema(document_id="d", fields=[
            FieldSpec(name="a", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 1, 1)),
            FieldSpec(name="a", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 1, 1)),
        ])
        findings = check_field_names_unique(schema)
        self.assertTrue(any(f.code == "DUPLICATE_FIELD_NAME" for f in findings))


class VerifyDocumentTests(unittest.TestCase):
    def test_clean_write_passes(self):
        doc = Document.from_pages_blank(1)
        original = doc.to_bytes(incremental=False)
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="a", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 100, 14),
                     value="x"),
        ])
        AcroFormWriter(doc).write(schema)
        produced = doc.to_bytes(incremental=True)
        report = verify_document(original, produced, schema)
        self.assertTrue(report.passed, report.as_dict())


if __name__ == "__main__":
    unittest.main()
