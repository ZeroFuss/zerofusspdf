"""Unit tests for :mod:`zfp.appearance.streams`."""

from __future__ import annotations

import unittest

from zfp.appearance import streams
from zfp.core.geometry import Rect
from zfp.core.types import FieldSpec, FieldType
from zfp.pdfio import fonts
from zfp.pdfio.document import Document
from zfp.pdfio.objects import PdfDict, PdfRef


def _spec(field_type=FieldType.TEXT, rect=None, **kw):
    rect = rect or Rect(400.0, 600.0, 500.0, 614.0)
    return FieldSpec(name="f", field_type=field_type, page=0, rect=rect, **kw)


def _resources():
    return PdfDict({"Font": PdfDict()})


class TextAppearanceTests(unittest.TestCase):
    def test_contains_bt_et_and_escaped_value(self):
        spec = _spec(font_size=10.0)
        content = streams.text_appearance(spec, "Jane (Q) Public", _resources())
        self.assertIn(b"BT", content)
        self.assertIn(b"ET", content)
        self.assertIn(b"Jane \\(Q\\) Public", content)

    def test_coordinates_stay_inside_the_bbox_not_page_space(self):
        spec = _spec(rect=Rect(400.0, 600.0, 500.0, 614.0), font_size=10.0)
        content = streams.text_appearance(spec, "X", _resources())
        # A Td line should show small local coordinates, never anything near 600.
        for line in content.split(b"\n"):
            if line.endswith(b"Td"):
                parts = line.split()
                x, y = float(parts[0]), float(parts[1])
                self.assertLess(abs(x), 100.0)
                self.assertLess(abs(y), 14.0)

    def test_clip_present_when_value_overflows_fixed_size(self):
        spec = _spec(rect=Rect(0, 0, 30, 14), font_size=12.0)
        content = streams.text_appearance(spec, "way too long to fit here", _resources())
        self.assertIn(b"W n", content)

    def test_empty_value_is_still_a_valid_marked_content_block(self):
        spec = _spec()
        content = streams.text_appearance(spec, "", _resources())
        self.assertIn(b"/Tx BMC", content)
        self.assertIn(b"EMC", content)


class CombAppearanceTests(unittest.TestCase):
    def test_places_exactly_n_show_operators(self):
        spec = _spec(field_type=FieldType.COMB, rect=Rect(0, 0, 90, 14), comb_cells=9)
        content = streams.comb_appearance(spec, "123456789")
        self.assertEqual(content.count(b" Tj"), 9)


class CheckboxRadioAppearanceTests(unittest.TestCase):
    def test_on_and_off_differ_and_on_has_drawing_ops(self):
        spec = _spec(field_type=FieldType.CHECKBOX, rect=Rect(0, 0, 12, 12))
        off = streams.checkbox_appearance(spec, on=False)
        on = streams.checkbox_appearance(spec, on=True)
        self.assertNotEqual(off, on)
        self.assertIn(b"S", on)

    def test_radio_on_contains_a_filled_path(self):
        spec = _spec(field_type=FieldType.RADIO, rect=Rect(0, 0, 12, 12))
        on = streams.radio_appearance(spec, on=True)
        self.assertIn(b"\nf\n", on)


class BuildXObjectTests(unittest.TestCase):
    def test_bbox_matches_rect_size_and_resources_carry_font(self):
        doc = Document.from_pages_blank(1)
        short, ref = fonts.ensure_standard_font(doc, "Helvetica")
        resources = PdfDict({"Font": PdfDict({short: ref})})
        rect = Rect(0, 0, 123.0, 45.0)
        spec = _spec(rect=rect)
        xref = streams.build_xobject(doc, spec, b"q Q\n", resources)
        self.assertIsInstance(xref, PdfRef)
        stream = doc.resolve(xref)
        bbox = [float(v) for v in stream.dict["BBox"]]
        self.assertAlmostEqual(bbox[2], rect.width, places=2)
        self.assertAlmostEqual(bbox[3], rect.height, places=2)
        res = doc.resolve(stream.dict["Resources"])
        self.assertIn("Font", res)


class AppearanceForDispatchTests(unittest.TestCase):
    def test_text_field_returns_one_ref(self):
        doc = Document.from_pages_blank(1)
        spec = _spec()
        result = streams.appearance_for(doc, spec, "Jane")
        self.assertIsInstance(result, PdfRef)

    def test_checkbox_returns_state_mapping(self):
        doc = Document.from_pages_blank(1)
        spec = _spec(field_type=FieldType.CHECKBOX, rect=Rect(0, 0, 12, 12),
                     export_value="Yes")
        result = streams.appearance_for(doc, spec, None)
        self.assertIn("Off", result)
        self.assertIn("Yes", result)
        self.assertIsInstance(result["Off"], PdfRef)

    def test_sanitize_export_value_strips_unsafe_characters(self):
        self.assertEqual(streams.sanitize_export_value("Yes, please!"),
                         "Yes__please_" .rstrip("_"))
        self.assertTrue(streams.sanitize_export_value(""))


class EscapingRoundTripTests(unittest.TestCase):
    def test_parens_and_backslash_round_trip(self):
        spec = _spec(font_size=10.0)
        content = streams.text_appearance(spec, "(a) \\ b", _resources())
        self.assertIn(b"\\(a\\)", content)
        self.assertIn(b"\\\\", content)


if __name__ == "__main__":
    unittest.main()
