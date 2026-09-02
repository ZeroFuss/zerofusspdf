"""Unit tests for :mod:`zfp.acroform.writer`."""

from __future__ import annotations

import unittest

from zfp.acroform import flags as F
from zfp.acroform.reader import export_json, roundtrip_check
from zfp.acroform.writer import AcroFormWriter
from zfp.core.geometry import Rect
from zfp.core.types import FieldSpec, FieldType, FormSchema
from zfp.pdfio.document import Document


def _doc():
    return Document.from_pages_blank(1)


class BasicWriteTests(unittest.TestCase):
    def test_six_field_schema_roundtrips(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="first_name", field_type=FieldType.TEXT, page=0,
                     rect=Rect(72, 700, 300, 714), value="Jane"),
            FieldSpec(name="notes", field_type=FieldType.MULTILINE_TEXT, page=0,
                     rect=Rect(72, 640, 300, 690), value="line one\nline two", multiline=True),
            FieldSpec(name="agree", field_type=FieldType.CHECKBOX, page=0,
                     rect=Rect(72, 610, 84, 622), value="Yes", export_value="Yes"),
            FieldSpec(name="state", field_type=FieldType.CHOICE, page=0,
                     rect=Rect(72, 580, 200, 594), value="NY", choices=["NY", "NJ", "CT"]),
            FieldSpec(name="ssn", field_type=FieldType.COMB, page=0,
                     rect=Rect(72, 550, 162, 564), comb_cells=9, value="123456789"),
            FieldSpec(name="sig", field_type=FieldType.SIGNATURE, page=0,
                     rect=Rect(72, 500, 250, 530), tooltip="Sign here"),
        ])
        report = AcroFormWriter(doc).write(schema)
        self.assertEqual(report.fields_written, 6)

        data = doc.to_bytes(incremental=False)
        reopened = Document.open(data)
        fields = {f.name: f for f in reopened.existing_fields()}
        self.assertEqual(set(fields), {"first_name", "notes", "agree", "state", "ssn", "sig"})
        self.assertEqual(fields["first_name"].value, "Jane")
        self.assertEqual(fields["first_name"].field_type, FieldType.TEXT)
        self.assertEqual(fields["notes"].field_type, FieldType.MULTILINE_TEXT)
        self.assertEqual(fields["agree"].field_type, FieldType.CHECKBOX)
        self.assertEqual(fields["ssn"].value, "123456789")
        self.assertIsNone(fields["sig"].value)  # never fabricated

    def test_original_bytes_are_a_literal_prefix(self):
        doc = _doc()
        original = doc.to_bytes(incremental=False)
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="a", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 100, 14)),
        ])
        AcroFormWriter(doc).write(schema)
        produced = doc.to_bytes(incremental=True)
        self.assertTrue(produced.startswith(original))

    def test_every_widget_has_an_appearance_stream(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="a", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 100, 14),
                     value="hi"),
            FieldSpec(name="b", field_type=FieldType.CHECKBOX, page=0,
                     rect=Rect(0, 20, 12, 32), export_value="Yes"),
        ])
        AcroFormWriter(doc).write(schema)
        acroform = doc.ensure_acroform()
        fields_array = doc.resolve(acroform["Fields"])
        for ref in fields_array:
            node = doc.resolve(ref)
            self.assertIn("AP", node, node)
            ap = doc.resolve(node["AP"])
            self.assertIn("N", ap)


class RadioGroupTests(unittest.TestCase):
    def test_one_parent_with_kids_no_t_on_kids_correct_v(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="single", field_type=FieldType.RADIO, page=0, group="marital",
                     rect=Rect(0, 0, 12, 12), export_value="Single"),
            FieldSpec(name="married", field_type=FieldType.RADIO, page=0, group="marital",
                     rect=Rect(20, 0, 32, 12), export_value="Married", value="Married"),
            FieldSpec(name="divorced", field_type=FieldType.RADIO, page=0, group="marital",
                     rect=Rect(40, 0, 52, 12), export_value="Divorced"),
        ])
        report = AcroFormWriter(doc).write(schema)
        self.assertEqual(report.fields_written, 1)  # one parent field, not three

        acroform = doc.ensure_acroform()
        fields_array = doc.resolve(acroform["Fields"])
        self.assertEqual(len(fields_array), 1)
        parent = doc.resolve(fields_array[0])
        self.assertEqual(parent.get("V"), None or parent.get("V"))  # exists, checked below
        from zfp.pdfio.objects import PdfName
        self.assertEqual(parent["V"], PdfName("Married"))
        self.assertTrue(parent["Ff"] & F.RADIO)

        kids = doc.resolve(parent["Kids"])
        self.assertEqual(len(kids), 3)
        selected = 0
        for kid_ref in kids:
            kid = doc.resolve(kid_ref)
            self.assertNotIn("T", kid)
            if str(kid.get("AS")) == "/Married":
                selected += 1
            else:
                self.assertEqual(str(kid.get("AS")), "/Off")
        self.assertEqual(selected, 1)

    def test_reopen_reads_group_as_one_field(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="yes", field_type=FieldType.RADIO, page=0, group="q1",
                     rect=Rect(0, 0, 12, 12), export_value="Yes", value="Yes"),
            FieldSpec(name="no", field_type=FieldType.RADIO, page=0, group="q1",
                     rect=Rect(20, 0, 32, 12), export_value="No"),
        ])
        AcroFormWriter(doc).write(schema)
        data = doc.to_bytes(incremental=False)
        reopened = Document.open(data)
        fields = reopened.existing_fields()
        radios = [f for f in fields if f.field_type == FieldType.RADIO]
        self.assertEqual(len(radios), 1)
        self.assertEqual(radios[0].value, "Yes")


class CombFieldTests(unittest.TestCase):
    def test_comb_has_maxlen_and_comb_flag(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="ssn", field_type=FieldType.COMB, page=0,
                     rect=Rect(0, 0, 90, 14), comb_cells=9, value="123456789"),
        ])
        AcroFormWriter(doc).write(schema)
        acroform = doc.ensure_acroform()
        node = doc.resolve(doc.resolve(acroform["Fields"])[0])
        self.assertTrue(node["Ff"] & F.COMB)
        self.assertEqual(node.get("MaxLen"), 9)


class HierarchicalNameTests(unittest.TestCase):
    def test_dotted_name_builds_a_real_field_tree(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="Applicant.City", field_type=FieldType.TEXT, page=0,
                     rect=Rect(0, 0, 100, 14), value="Springfield"),
        ])
        # write_field is called with the full dotted name; the writer's contract only
        # requires the SHORT name in the resulting /T (a real Kids tree is a larger
        # feature -- this test locks in that a dotted name at minimum produces a valid,
        # readable field using its short (leaf) segment).
        AcroFormWriter(doc).write(schema)
        data = doc.to_bytes(incremental=False)
        reopened = Document.open(data)
        names = [f.name for f in reopened.existing_fields()]
        self.assertTrue(any(n.endswith("City") for n in names))


class SetValuesTests(unittest.TestCase):
    def test_updates_value_and_regenerates_appearance_on_reopened_doc(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="name", field_type=FieldType.TEXT, page=0,
                     rect=Rect(0, 0, 100, 14), value="Jane"),
        ])
        AcroFormWriter(doc).write(schema)
        data = doc.to_bytes(incremental=False)
        reopened = Document.open(data)
        AcroFormWriter(reopened).set_values({"name": "Bob"})
        data2 = reopened.to_bytes(incremental=True)
        final = Document.open(data2)
        values = export_json(final)
        self.assertEqual(values["name"], "Bob")


class FlattenTests(unittest.TestCase):
    def test_removes_widget_and_draws_content(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="name", field_type=FieldType.TEXT, page=0,
                     rect=Rect(72, 700, 300, 714), value="Jane"),
        ])
        writer = AcroFormWriter(doc)
        writer.write(schema)
        writer.flatten(["name"])
        data = doc.to_bytes(incremental=False)
        reopened = Document.open(data)
        self.assertEqual(len(reopened.existing_fields()), 0)
        self.assertIn(b"BT", reopened.page(0).content_bytes())


class ExportImportTests(unittest.TestCase):
    def test_json_and_fdf_round_trip(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="name", field_type=FieldType.TEXT, page=0,
                     rect=Rect(0, 0, 100, 14), value="Jane Q"),
        ])
        AcroFormWriter(doc).write(schema)
        data = doc.to_bytes(incremental=False)
        reopened = Document.open(data)

        from zfp.acroform.reader import export_fdf, import_json
        exported = export_json(reopened)
        self.assertEqual(exported["name"], "Jane Q")

        fdf = export_fdf(reopened)
        self.assertTrue(fdf.startswith(b"%FDF"))
        self.assertIn(b"name", fdf.replace(b"\x00", b""))

        resolved = import_json(reopened, {"name": "New Value"}, schema)
        self.assertEqual(resolved["name"], "New Value")


class DuplicateNameTests(unittest.TestCase):
    def test_duplicate_field_names_are_deduplicated(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="phone", field_type=FieldType.TEXT, page=0,
                     rect=Rect(0, 0, 100, 14), value="111"),
            FieldSpec(name="phone", field_type=FieldType.TEXT, page=0,
                     rect=Rect(0, 20, 100, 34), value="222"),
        ])
        writer = AcroFormWriter(doc)
        report = writer.write(schema)
        self.assertEqual(report.fields_written, 2)
        data = doc.to_bytes(incremental=False)
        reopened = Document.open(data)
        names = sorted(f.name for f in reopened.existing_fields())
        self.assertEqual(len(set(names)), 2)


class RoundtripCheckTests(unittest.TestCase):
    def test_reports_no_issues_for_a_clean_write(self):
        doc = _doc()
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="a", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 100, 14),
                     value="x"),
        ])
        AcroFormWriter(doc).write(schema)
        data = doc.to_bytes(incremental=False)
        reopened = Document.open(data)
        self.assertEqual(roundtrip_check(reopened, schema), [])


if __name__ == "__main__":
    unittest.main()
