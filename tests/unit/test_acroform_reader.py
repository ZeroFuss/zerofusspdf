"""Unit tests for :mod:`zfp.acroform.reader`."""

from __future__ import annotations

import unittest

from zfp.acroform.reader import (
    export_csv, export_fdf, export_json, export_xml, field_tree,
    import_json, read_fields, read_values, roundtrip_check,
)
from zfp.acroform.writer import AcroFormWriter
from zfp.core.geometry import Rect
from zfp.core.types import FieldSpec, FieldType, FormSchema
from zfp.pdfio.document import Document


def _written_doc():
    doc = Document.from_pages_blank(1)
    schema = FormSchema(document_id=doc.document_id, fields=[
        FieldSpec(name="first_name", field_type=FieldType.TEXT, page=0,
                 rect=Rect(0, 0, 100, 14), value="Jane"),
        FieldSpec(name="last_name", field_type=FieldType.TEXT, page=0,
                 rect=Rect(0, 20, 100, 34), value="Public"),
    ])
    AcroFormWriter(doc).write(schema)
    data = doc.to_bytes(incremental=False)
    return Document.open(data), schema


class ReadFieldsTests(unittest.TestCase):
    def test_read_fields_matches_written_schema(self):
        doc, _schema = _written_doc()
        names = {f.name for f in read_fields(doc)}
        self.assertEqual(names, {"first_name", "last_name"})

    def test_read_values(self):
        doc, _schema = _written_doc()
        values = read_values(doc)
        self.assertEqual(values["first_name"], "Jane")
        self.assertEqual(values["last_name"], "Public")


class FieldTreeTests(unittest.TestCase):
    def test_tree_carries_widgets_and_object_numbers(self):
        doc, _schema = _written_doc()
        tree = field_tree(doc)
        self.assertIn("first_name", tree)
        entry = tree["first_name"]
        self.assertIsNotNone(entry["num"])
        self.assertEqual(len(entry["widgets"]), 1)


class ExportFormatTests(unittest.TestCase):
    def test_export_json(self):
        doc, _schema = _written_doc()
        data = export_json(doc)
        self.assertEqual(data["first_name"], "Jane")

    def test_export_xml_contains_field_names_and_values(self):
        doc, _schema = _written_doc()
        xml = export_xml(doc)
        self.assertIn("first_name", xml)
        self.assertIn("Jane", xml)

    def test_export_csv_has_header_and_rows(self):
        doc, _schema = _written_doc()
        csv_text = export_csv(doc)
        lines = csv_text.strip().splitlines()
        self.assertEqual(lines[0], "name,value")
        self.assertEqual(len(lines), 3)

    def test_export_fdf_starts_with_fdf_header(self):
        doc, _schema = _written_doc()
        fdf = export_fdf(doc)
        self.assertTrue(fdf.startswith(b"%FDF"))
        self.assertTrue(fdf.rstrip().endswith(b"%%EOF"))


class ImportJsonTests(unittest.TestCase):
    def test_maps_canonical_key_to_field_name(self):
        doc, _schema = _written_doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="first_name", field_type=FieldType.TEXT, page=0,
                     rect=Rect(0, 0, 1, 1), canonical_key="person.name.first"),
        ])
        resolved = import_json(doc, {"person.name.first": "Bob"}, schema)
        self.assertEqual(resolved["first_name"], "Bob")

    def test_plain_field_name_passes_through(self):
        doc, _schema = _written_doc()
        resolved = import_json(doc, {"first_name": "Bob"})
        self.assertEqual(resolved["first_name"], "Bob")


class RoundtripCheckTests(unittest.TestCase):
    def test_flags_a_missing_field(self):
        doc, schema = _written_doc()
        extra = FormSchema(document_id=doc.document_id, fields=list(schema.fields) + [
            FieldSpec(name="ghost", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 1, 1)),
        ])
        issues = roundtrip_check(doc, extra)
        self.assertTrue(any("ghost" in i for i in issues))


if __name__ == "__main__":
    unittest.main()
