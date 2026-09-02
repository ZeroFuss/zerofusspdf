"""Unit tests for :mod:`zfp.qa.renderdiff`."""

from __future__ import annotations

import unittest

from zfp.acroform.writer import AcroFormWriter
from zfp.core.geometry import Rect
from zfp.core.types import FieldSpec, FieldType, FormSchema
from zfp.pdfio.document import Document
from zfp.qa.renderdiff import structural_diff


class StructuralDiffTests(unittest.TestCase):
    def test_zero_changed_pages_for_annotation_only_update(self):
        doc = Document.from_pages_blank(1)
        original = doc.to_bytes(incremental=False)
        schema = FormSchema(document_id=doc.document_id, fields=[
            FieldSpec(name="a", field_type=FieldType.TEXT, page=0, rect=Rect(0, 0, 100, 14)),
        ])
        AcroFormWriter(doc).write(schema)
        produced = doc.to_bytes(incremental=True)
        report = structural_diff(original, produced)
        self.assertEqual(report.metrics["pages_changed"], [])
        self.assertTrue(report.passed)

    def test_nonzero_when_content_stream_replaced(self):
        doc = Document.from_pages_blank(1)
        original = doc.to_bytes(incremental=False)

        from zfp.pdfio.filters import encode_flate
        from zfp.pdfio.objects import PdfDict, PdfName, PdfStream

        page = doc.page(0)
        content = b"BT /F1 12 Tf 100 700 Td (Changed) Tj ET"
        encoded = encode_flate(content)
        stream = PdfStream(PdfDict({"Filter": PdfName("FlateDecode"), "Length": len(encoded)}),
                           encoded)
        ref = doc.writer.add_object(stream)
        page.dict["Contents"] = ref
        page.touch()
        produced = doc.to_bytes(incremental=True)

        report = structural_diff(original, produced)
        self.assertEqual(report.metrics["pages_changed"], [0])


if __name__ == "__main__":
    unittest.main()
