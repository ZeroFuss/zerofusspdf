"""Unit tests for :mod:`zfp.preflight.classifier`."""

from __future__ import annotations

import json
import unittest

from zfp.core.serde import dumps
from zfp.core.types import DocumentClass, DocumentProfile, PageMode, PageProfile
from zfp.core.geometry import PageGeometry, Rect
from zfp.pdfio.document import Document, Page
from zfp.pdfio.objects import PdfArray, PdfDict, PdfName, PdfRef, PdfStream, PdfString
from zfp.pdfio.writer import build_document
from zfp.preflight.classifier import (
    FORM_MARKER_PATTERNS,
    MIN_FORM_CHECKBOXES,
    classify_page,
    describe,
    profile_document,
    profile_to_dict,
    route,
    scan_content,
)

PROSE = b"The quick brown fox jumps over the lazy dog while the sun sets."
FORM_TEXT = b"Full Name:  Signature:  Date of birth:"


# --------------------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------------------
def blank(pages: int = 1, width: float = 612.0, height: float = 792.0) -> Document:
    """A brand new document of empty pages."""
    return Document.from_pages_blank(pages, width, height)


def set_content(doc: Document, index: int, payload: bytes) -> None:
    """Replace a page's content stream through the writer overlay."""
    page = doc.page(index)
    ref = page.dict.get("Contents")
    assert isinstance(ref, PdfRef), "from_pages_blank should give every page a /Contents ref"
    doc.writer.set_object(ref.num, PdfStream(PdfDict({"Length": len(payload)}), payload))


def text_ops(text: bytes) -> bytes:
    """A minimal ``BT ... Tj ... ET`` block showing ``text``."""
    return b"BT /F1 11 Tf 72 700 Td (" + text + b") Tj ET\n"


def rules(count: int, *, y0: float = 640.0) -> bytes:
    """``count`` thin, page-wide horizontal rules drawn as filled rectangles."""
    out = []
    for index in range(count):
        out.append(b"72 %.1f 400 0.8 re f\n" % (y0 - 20.0 * index))
    return b"".join(out)


def add_image(
    doc: Document,
    index: int,
    *,
    name: str = "Im0",
    width: int = 2550,
    height: int = 3300,
) -> PdfRef:
    """Attach an image XObject to a page's resources."""
    stream = PdfStream(
        PdfDict(
            {
                "Type": PdfName("XObject"),
                "Subtype": PdfName("Image"),
                "Width": width,
                "Height": height,
                "ColorSpace": PdfName("DeviceGray"),
                "BitsPerComponent": 8,
                "Filter": PdfName("DCTDecode"),
                "Length": 3,
            }
        ),
        b"jpg",
    )
    ref = doc.writer.add_object(stream)
    page = doc.page(index)
    resources = page.resources()
    xobjects = resources.get("XObject")
    if not isinstance(xobjects, PdfDict):
        xobjects = PdfDict()
        resources["XObject"] = xobjects
    xobjects[name] = ref
    page.dict["Resources"] = resources
    page.touch()
    return ref


def add_form_xobject_with_image(doc: Document, index: int, *, name: str = "Fm0") -> PdfRef:
    """Attach a form XObject whose own resources hold an image."""
    image = PdfStream(
        PdfDict(
            {
                "Type": PdfName("XObject"),
                "Subtype": PdfName("Image"),
                "Width": 600,
                "Height": 400,
                "Length": 3,
            }
        ),
        b"jpg",
    )
    image_ref = doc.writer.add_object(image)
    form = PdfStream(
        PdfDict(
            {
                "Type": PdfName("XObject"),
                "Subtype": PdfName("Form"),
                "BBox": PdfArray([0, 0, 300, 200]),
                "Resources": PdfDict({"XObject": PdfDict({"Inner": image_ref})}),
                "Length": 0,
            }
        ),
        b"",
    )
    form_ref = doc.writer.add_object(form)
    page = doc.page(index)
    resources = page.resources()
    resources["XObject"] = PdfDict({name: form_ref})
    page.dict["Resources"] = resources
    page.touch()
    return form_ref


def add_widget(doc: Document, index: int, name: str = "field_1") -> PdfRef:
    """Attach a widget annotation to a page."""
    annot = PdfDict(
        {
            "Type": PdfName("Annot"),
            "Subtype": PdfName("Widget"),
            "FT": PdfName("Tx"),
            "T": PdfString.from_text(name),
            "Rect": PdfArray([100, 100, 300, 118]),
        }
    )
    ref = doc.writer.add_object(annot)
    doc.page(index).add_annotation(ref)
    return ref


def profile_of(doc: Document, index: int = 0) -> PageProfile:
    return classify_page(doc, index)


def make_profile(modes, **kwargs) -> DocumentProfile:
    """A DocumentProfile with one page per entry of ``modes``."""
    box = Rect(0.0, 0.0, 612.0, 792.0)
    pages = [
        PageProfile(
            index=i,
            geometry=PageGeometry(index=i, media_box=box, crop_box=box),
            mode=mode,
            has_widgets=mode is PageMode.INTERACTIVE_FORM,
            has_native_text=mode
            in (
                PageMode.NATIVE_DOCUMENT,
                PageMode.FLAT_NATIVE_FORM,
                PageMode.HYBRID,
            ),
            has_raster=mode
            in (PageMode.SCANNED_FORM, PageMode.SCANNED_DOCUMENT, PageMode.HYBRID),
        )
        for i, mode in enumerate(modes)
    ]
    return DocumentProfile(
        document_id="doc_test", page_count=len(pages), pages=pages, **kwargs
    )


# --------------------------------------------------------------------------------------
# The content scan
# --------------------------------------------------------------------------------------
class ScanContentTests(unittest.TestCase):
    def test_empty_stream(self):
        scan = scan_content(b"")
        self.assertEqual(scan.char_count, 0)
        self.assertFalse(scan.has_visible_text)
        self.assertEqual(scan.vector_op_count, 0)

    def test_tj_and_tj_array_and_quote_operators(self):
        payload = (
            b"BT (Hello) Tj [(wor) -250 (ld)] TJ (again) ' 1 2 (spaced) \" ET"
        )
        scan = scan_content(payload)
        expected = len("Hello") + len("world") + len("again") + len("spaced")
        self.assertEqual(scan.char_count, expected)
        self.assertTrue(scan.has_visible_text)
        self.assertIn("world", scan.text)

    def test_hex_strings_are_counted(self):
        scan = scan_content(b"BT <48656C6C6F> Tj ET")
        self.assertEqual(scan.char_count, 5)
        self.assertEqual(scan.text, "Hello")

    def test_whitespace_only_text_is_not_visible(self):
        scan = scan_content(b"BT (        ) Tj ET")
        self.assertEqual(scan.char_count, 8)
        self.assertFalse(scan.has_visible_text)

    def test_every_path_operator_is_counted(self):
        payload = b"0 0 m 10 0 l 1 1 2 2 3 3 c 1 1 2 2 v 1 1 2 2 y S s f F f* B B* b b* 0 0 5 5 re"
        scan = scan_content(payload)
        # m l c v y  +  S s f F f* B B* b b*  +  re
        self.assertEqual(scan.vector_op_count, 5 + 9 + 1)

    def test_non_path_operators_are_ignored(self):
        scan = scan_content(b"q 1 0 0 1 0 0 cm /GS0 gs 0.5 w /P0 SCN Q W n")
        self.assertEqual(scan.vector_op_count, 0)
        self.assertEqual(scan.placements, [(1.0, 1.0)])

    def test_inline_image_binary_is_not_lexed(self):
        # The binary payload deliberately contains bytes that look like operators.
        binary = b"(Tj re re re re S BT ET" + bytes(range(0, 32))
        payload = b"q BI /W 8 /H 8 /BPC 8 /CS /G ID " + binary + b" EI Q\n72 700 400 1 re f"
        scan = scan_content(payload)
        self.assertEqual(scan.inline_image_count, 1)
        self.assertEqual(scan.inline_images, [(8.0, 8.0)])
        self.assertEqual(scan.char_count, 0)
        # Only the real `re f` after EI counted, not the bytes inside the image.
        self.assertEqual(scan.vector_op_count, 2)

    def test_unterminated_inline_image_does_not_hang(self):
        scan = scan_content(b"BI /W 4 /H 4 ID \x00\x01\x02")
        self.assertEqual(scan.inline_image_count, 1)
        self.assertEqual(scan.vector_op_count, 0)

    def test_horizontal_rules_from_rects_and_stroked_lines(self):
        payload = rules(2) + b"72 600 m 472 600 l S\n72 580 m 472 583 l S\n"
        scan = scan_content(payload)
        # two thin rects + one horizontal stroked line; the sloped one does not count.
        self.assertEqual(scan.horizontal_rules, 3)

    def test_consecutive_show_operators_do_not_fuse_words(self):
        scan = scan_content(b"BT (Yes) Tj (No) Tj ET")
        self.assertEqual(scan.char_count, 5)
        self.assertEqual(scan.text, "Yes No")

    def test_tj_array_elements_are_not_separated(self):
        scan = scan_content(b"BT [(Sig) (nature)] TJ ET")
        self.assertEqual(scan.text, "Signature")

    def test_checkbox_sized_squares_are_counted(self):
        payload = b"".join(
            b"%d 600 12 12 re S " % (100 + 40 * index) for index in range(4)
        )
        scan = scan_content(payload)
        self.assertEqual(scan.checkbox_glyphs, 4)
        self.assertEqual(scan.horizontal_rules, 0)

    def test_oblong_and_oversized_rects_are_not_checkboxes(self):
        scan = scan_content(b"10 10 12 30 re S 10 60 200 200 re S 10 10 2 2 re S")
        self.assertEqual(scan.checkbox_glyphs, 0)

    def test_unstroked_lines_are_not_rules(self):
        scan = scan_content(b"72 600 m 472 600 l n")
        self.assertEqual(scan.horizontal_rules, 0)

    def test_short_or_thick_rects_are_not_rules(self):
        scan = scan_content(b"10 10 5 0.5 re f 10 20 400 40 re f")
        self.assertEqual(scan.horizontal_rules, 0)

    def test_truncation_flag(self):
        scan = scan_content(b"q Q " * 100, max_bytes=10)
        self.assertTrue(scan.truncated)

    def test_malformed_stream_is_survivable(self):
        scan = scan_content(b"<< /A ] ) >> ((( 3 4 re f")
        self.assertGreaterEqual(scan.vector_op_count, 0)


# --------------------------------------------------------------------------------------
# Page classification
# --------------------------------------------------------------------------------------
class ClassifyPageTests(unittest.TestCase):
    def test_blank_page_is_empty(self):
        page = profile_of(blank())
        self.assertIs(page.mode, PageMode.EMPTY)
        self.assertFalse(page.has_native_text)
        self.assertFalse(page.has_vector)
        self.assertFalse(page.has_raster)
        self.assertEqual(page.char_count, 0)
        self.assertEqual(page.image_area_ratio, 0.0)

    def test_widgets_win_outright(self):
        doc = blank()
        set_content(doc, 0, text_ops(PROSE) + rules(6))
        add_widget(doc, 0)
        page = profile_of(doc)
        self.assertTrue(page.has_widgets)
        self.assertIs(page.mode, PageMode.INTERACTIVE_FORM)

    def test_non_widget_annotation_is_not_a_widget(self):
        doc = blank()
        annot = PdfDict({"Type": PdfName("Annot"), "Subtype": PdfName("Link")})
        doc.page(0).add_annotation(doc.writer.add_object(annot))
        self.assertFalse(profile_of(doc).has_widgets)

    def test_text_with_rules_is_a_flat_native_form(self):
        doc = blank()
        set_content(doc, 0, text_ops(b"Applicant") + rules(4))
        page = profile_of(doc)
        self.assertIs(page.mode, PageMode.FLAT_NATIVE_FORM)
        self.assertTrue(page.has_native_text)
        self.assertTrue(page.has_vector)
        self.assertEqual(page.vector_op_count, 8)

    def test_text_with_marker_words_alone_is_a_flat_native_form(self):
        doc = blank()
        set_content(doc, 0, text_ops(FORM_TEXT))
        page = profile_of(doc)
        self.assertIs(page.mode, PageMode.FLAT_NATIVE_FORM)
        self.assertFalse(page.has_vector)

    def test_checkbox_squares_alone_are_a_form_cue(self):
        doc = blank()
        boxes = b"".join(
            b"%d 600 12 12 re S " % (100 + 40 * index)
            for index in range(MIN_FORM_CHECKBOXES)
        )
        set_content(doc, 0, text_ops(PROSE) + boxes)
        page = profile_of(doc)
        self.assertIs(page.mode, PageMode.FLAT_NATIVE_FORM)

    def test_yes_no_options_split_across_show_operators_are_markers(self):
        doc = blank()
        set_content(
            doc,
            0,
            b"BT /F1 11 Tf 72 700 Td (Citizenship) Tj (Yes) Tj (No) Tj ET",
        )
        page = profile_of(doc)
        self.assertIs(page.mode, PageMode.FLAT_NATIVE_FORM)

    def test_prose_without_cues_is_a_native_document(self):
        doc = blank()
        set_content(doc, 0, text_ops(PROSE))
        page = profile_of(doc)
        self.assertIs(page.mode, PageMode.NATIVE_DOCUMENT)
        self.assertTrue(page.has_native_text)

    def test_a_single_short_word_is_not_native_text(self):
        doc = blank()
        set_content(doc, 0, text_ops(b"Hi"))
        page = profile_of(doc)
        self.assertEqual(page.char_count, 2)
        self.assertFalse(page.has_native_text)
        self.assertIs(page.mode, PageMode.EMPTY)

    def test_full_page_image_without_text_is_a_scanned_form(self):
        doc = blank()
        add_image(doc, 0)
        set_content(doc, 0, b"q 612 0 0 792 0 0 cm /Im0 Do Q")
        page = profile_of(doc)
        self.assertTrue(page.has_raster)
        self.assertFalse(page.has_native_text)
        self.assertEqual(page.image_area_ratio, 1.0)
        self.assertIs(page.mode, PageMode.SCANNED_FORM)

    def test_full_page_image_is_recognised_without_a_placement_matrix(self):
        doc = blank()
        add_image(doc, 0)  # 2550x3300 == US Letter at 300 dpi
        set_content(doc, 0, b"/Im0 Do")
        page = profile_of(doc)
        self.assertEqual(page.image_area_ratio, 1.0)
        self.assertIs(page.mode, PageMode.SCANNED_FORM)

    def test_scan_with_an_ocr_text_layer_is_a_scanned_document(self):
        doc = blank()
        add_image(doc, 0)
        set_content(doc, 0, b"q 612 0 0 792 0 0 cm /Im0 Do Q\n" + text_ops(PROSE))
        page = profile_of(doc)
        self.assertTrue(page.has_raster)
        self.assertTrue(page.has_native_text)
        self.assertIs(page.mode, PageMode.SCANNED_DOCUMENT)

    def test_scan_with_text_and_form_cues_is_hybrid(self):
        doc = blank()
        add_image(doc, 0)
        set_content(doc, 0, b"q 612 0 0 792 0 0 cm /Im0 Do Q\n" + text_ops(FORM_TEXT))
        page = profile_of(doc)
        self.assertIs(page.mode, PageMode.HYBRID)

    def test_small_logo_does_not_make_a_page_a_scan(self):
        doc = blank()
        add_image(doc, 0, width=100, height=50)
        set_content(doc, 0, b"q 100 0 0 50 40 700 cm /Im0 Do Q\n" + text_ops(PROSE))
        page = profile_of(doc)
        self.assertTrue(page.has_raster)
        self.assertLess(page.image_area_ratio, 0.05)
        self.assertIs(page.mode, PageMode.NATIVE_DOCUMENT)

    def test_image_area_ratio_is_clamped(self):
        doc = blank()
        add_image(doc, 0, width=9000, height=12000)
        set_content(doc, 0, b"/Im0 Do /Im0 Do /Im0 Do")
        page = profile_of(doc)
        self.assertEqual(page.image_area_ratio, 1.0)

    def test_inline_image_page_is_raster(self):
        doc = blank()
        set_content(
            doc,
            0,
            b"q 612 0 0 792 0 0 cm BI /W 2550 /H 3300 /BPC 8 /CS /G ID \x00\x01\x02 EI Q",
        )
        page = profile_of(doc)
        self.assertTrue(page.has_raster)
        self.assertEqual(page.image_area_ratio, 1.0)
        self.assertIs(page.mode, PageMode.SCANNED_FORM)

    def test_image_nested_in_a_form_xobject_is_found(self):
        doc = blank()
        add_form_xobject_with_image(doc, 0)
        set_content(doc, 0, b"q 1 0 0 1 0 0 cm /Fm0 Do Q")
        page = profile_of(doc)
        self.assertTrue(page.has_raster)

    def test_vector_only_page_is_a_native_document(self):
        doc = blank()
        set_content(doc, 0, rules(5))
        page = profile_of(doc)
        self.assertTrue(page.has_vector)
        self.assertFalse(page.has_native_text)
        self.assertIs(page.mode, PageMode.NATIVE_DOCUMENT)

    def test_geometry_is_carried_through(self):
        doc = blank(1, 200.0, 400.0)
        page = doc.page(0)
        page.dict["Rotate"] = 90
        page.touch()
        profile = profile_of(doc)
        self.assertEqual(profile.geometry.rotation, 90)
        self.assertEqual(profile.geometry.width, 200.0)
        self.assertEqual(profile.index, 0)

    def test_out_of_range_index_raises_a_zfp_error(self):
        from zfp.core.errors import ValidationError

        with self.assertRaises(ValidationError):
            classify_page(blank(), 5)


# --------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------
class RouteTests(unittest.TestCase):
    def test_encrypted_and_locked_outranks_everything(self):
        profile = make_profile(
            [PageMode.INTERACTIVE_FORM],
            encrypted=True,
            can_modify=False,
            signed=True,
            acroform=True,
            dynamic_xfa=True,
        )
        self.assertIs(route(profile), DocumentClass.ENCRYPTED)

    def test_encrypted_but_modifiable_is_not_encrypted_class(self):
        profile = make_profile([PageMode.FLAT_NATIVE_FORM], encrypted=True, can_modify=True)
        self.assertIs(route(profile), DocumentClass.FLAT_NATIVE_FORM)

    def test_signed_outranks_xfa_and_forms(self):
        profile = make_profile(
            [PageMode.INTERACTIVE_FORM], signed=True, acroform=True, dynamic_xfa=True
        )
        self.assertIs(route(profile), DocumentClass.SIGNED)

    def test_dynamic_xfa_outranks_acroform(self):
        profile = make_profile(
            [PageMode.INTERACTIVE_FORM], acroform=True, xfa=True, dynamic_xfa=True
        )
        self.assertIs(route(profile), DocumentClass.XFA)

    def test_static_xfa_routes_as_existing_acroform(self):
        profile = make_profile([PageMode.NATIVE_DOCUMENT], acroform=True, xfa=True)
        self.assertIs(route(profile), DocumentClass.EXISTING_ACROFORM)

    def test_empty_acroform_without_widgets_is_not_an_existing_form(self):
        # A left-over empty /AcroForm must not send the document down the
        # read-the-existing-fields path, where it would find nothing.
        profile = make_profile([PageMode.FLAT_NATIVE_FORM], acroform=True)
        self.assertIs(route(profile), DocumentClass.FLAT_NATIVE_FORM)

    def test_widget_page_is_an_existing_acroform(self):
        profile = make_profile([PageMode.INTERACTIVE_FORM, PageMode.NATIVE_DOCUMENT])
        self.assertIs(route(profile), DocumentClass.EXISTING_ACROFORM)

    def test_scanned_majority(self):
        profile = make_profile(
            [PageMode.SCANNED_FORM, PageMode.SCANNED_FORM, PageMode.SCANNED_DOCUMENT]
        )
        self.assertIs(route(profile), DocumentClass.SCANNED_FORM)

    def test_flat_native_majority(self):
        profile = make_profile([PageMode.FLAT_NATIVE_FORM, PageMode.NATIVE_DOCUMENT])
        self.assertIs(route(profile), DocumentClass.FLAT_NATIVE_FORM)

    def test_mixed_native_and_raster_is_hybrid(self):
        profile = make_profile([PageMode.FLAT_NATIVE_FORM, PageMode.SCANNED_FORM])
        self.assertIs(route(profile), DocumentClass.HYBRID)

    def test_hybrid_pages_are_hybrid(self):
        profile = make_profile([PageMode.HYBRID, PageMode.NATIVE_DOCUMENT])
        self.assertIs(route(profile), DocumentClass.HYBRID)

    def test_no_form_pages_is_non_form(self):
        profile = make_profile(
            [PageMode.NATIVE_DOCUMENT, PageMode.EMPTY, PageMode.SCANNED_DOCUMENT]
        )
        self.assertIs(route(profile), DocumentClass.NON_FORM)

    def test_empty_document_is_non_form(self):
        self.assertIs(route(make_profile([])), DocumentClass.NON_FORM)


# --------------------------------------------------------------------------------------
# Whole-document profiling
# --------------------------------------------------------------------------------------
def acroform_pdf(*, xfa: bool = False, needs_rendering: bool = False, fields: bool = True) -> bytes:
    """A one-page document with an AcroForm, optionally carrying an XFA packet."""
    catalog = PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2), "AcroForm": PdfRef(5)})
    if needs_rendering:
        catalog["NeedsRendering"] = True
    page = PdfDict(
        {
            "Type": PdfName("Page"),
            "Parent": PdfRef(2),
            "MediaBox": PdfArray([0, 0, 612, 792]),
            "Resources": PdfDict(),
            "Contents": PdfRef(4),
        }
    )
    acroform = PdfDict({"Fields": PdfArray(), "DA": PdfString(b"/Helv 0 Tf 0 g")})
    objects = {
        1: catalog,
        2: PdfDict({"Type": PdfName("Pages"), "Kids": PdfArray([PdfRef(3)]), "Count": 1}),
        3: page,
        4: PdfStream(PdfDict({"Length": 0}), b""),
        5: acroform,
    }
    if fields:
        page["Annots"] = PdfArray([PdfRef(6)])
        acroform["Fields"] = PdfArray([PdfRef(6)])
        objects[6] = PdfDict(
            {
                "Type": PdfName("Annot"),
                "Subtype": PdfName("Widget"),
                "FT": PdfName("Tx"),
                "T": PdfString.from_text("name"),
                "Rect": PdfArray([72, 700, 300, 718]),
                "P": PdfRef(3),
            }
        )
    if xfa:
        packet = b"<config><present><pdf><interactive>1</interactive></pdf></present></config>"
        if needs_rendering:
            packet = (
                b"<config><present><pdf><dynamicRender>required"
                b"</dynamicRender></pdf></present></config>"
            )
        objects[7] = PdfStream(PdfDict({"Length": len(packet)}), packet)
        acroform["XFA"] = PdfArray([PdfString.from_text("config"), PdfRef(7)])
    return build_document(objects, PdfRef(1))


def broken_content_pdf() -> bytes:
    """A one-page document whose ``/Contents`` cannot be decoded."""
    objects = {
        1: PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2)}),
        2: PdfDict({"Type": PdfName("Pages"), "Kids": PdfArray([PdfRef(3)]), "Count": 1}),
        3: PdfDict(
            {
                "Type": PdfName("Page"),
                "Parent": PdfRef(2),
                "MediaBox": PdfArray([0, 0, 612, 792]),
                "Contents": PdfRef(4),
            }
        ),
        4: PdfStream(
            PdfDict({"Filter": PdfName("FlateDecode"), "Length": 9}), b"not-flate"
        ),
    }
    return build_document(objects, PdfRef(1))


def metadata_pdf() -> bytes:
    """A tagged one-page document with an ``/Info /Producer``."""
    objects = {
        1: PdfDict(
            {
                "Type": PdfName("Catalog"),
                "Pages": PdfRef(2),
                "MarkInfo": PdfDict({"Marked": True}),
            }
        ),
        2: PdfDict({"Type": PdfName("Pages"), "Kids": PdfArray([PdfRef(3)]), "Count": 1}),
        3: PdfDict(
            {
                "Type": PdfName("Page"),
                "Parent": PdfRef(2),
                "MediaBox": PdfArray([0, 0, 612, 792]),
            }
        ),
        4: PdfDict({"Producer": PdfString.from_text("ZFP Test Harness 1.0")}),
    }
    return build_document(objects, PdfRef(1), info=PdfRef(4))


class ProfileDocumentTests(unittest.TestCase):
    def test_blank_document(self):
        profile = profile_document(blank(3))
        self.assertEqual(profile.page_count, 3)
        self.assertEqual(len(profile.pages), 3)
        self.assertIs(profile.doc_class, DocumentClass.NON_FORM)
        self.assertFalse(profile.encrypted)
        self.assertTrue(profile.can_modify)
        self.assertFalse(profile.signed)
        self.assertFalse(profile.acroform)
        self.assertEqual(profile.version, "1.7")
        self.assertEqual(profile.native_text_pages, [])
        self.assertEqual(profile.raster_pages, [])

    def test_mixed_document_reports_page_lists(self):
        doc = blank(3)
        set_content(doc, 0, text_ops(FORM_TEXT) + rules(4))
        add_image(doc, 1)
        set_content(doc, 1, b"q 612 0 0 792 0 0 cm /Im0 Do Q")
        set_content(doc, 2, text_ops(PROSE))
        profile = profile_document(doc)
        self.assertEqual([p.mode for p in profile.pages][0], PageMode.FLAT_NATIVE_FORM)
        self.assertEqual(profile.native_text_pages, [0, 2])
        self.assertEqual(profile.raster_pages, [1])
        self.assertIs(profile.doc_class, DocumentClass.HYBRID)

    def test_acroform_document_routes_to_existing_acroform(self):
        profile = profile_document(Document.open(acroform_pdf()))
        self.assertTrue(profile.acroform)
        self.assertFalse(profile.xfa)
        self.assertTrue(profile.pages[0].has_widgets)
        self.assertIs(profile.pages[0].mode, PageMode.INTERACTIVE_FORM)
        self.assertIs(profile.doc_class, DocumentClass.EXISTING_ACROFORM)

    def test_static_xfa_is_detected_but_not_dynamic(self):
        profile = profile_document(Document.open(acroform_pdf(xfa=True)))
        self.assertTrue(profile.xfa)
        self.assertFalse(profile.dynamic_xfa)
        self.assertIs(profile.doc_class, DocumentClass.EXISTING_ACROFORM)
        self.assertIn("static XFA: treated as its AcroForm shadow", profile.warnings)

    def test_dynamic_xfa_routes_to_the_compatibility_layer(self):
        profile = profile_document(
            Document.open(acroform_pdf(xfa=True, needs_rendering=True))
        )
        self.assertTrue(profile.xfa)
        self.assertTrue(profile.dynamic_xfa)
        self.assertIs(profile.doc_class, DocumentClass.XFA)
        self.assertIn("dynamic XFA: routed to compatibility layer", profile.warnings)

    def test_acroform_without_widgets_still_routes_by_field_count(self):
        # An AcroForm whose only field is not reachable from any page's /Annots.
        objects = {
            1: PdfDict(
                {"Type": PdfName("Catalog"), "Pages": PdfRef(2), "AcroForm": PdfRef(5)}
            ),
            2: PdfDict({"Type": PdfName("Pages"), "Kids": PdfArray([PdfRef(3)]), "Count": 1}),
            3: PdfDict(
                {
                    "Type": PdfName("Page"),
                    "Parent": PdfRef(2),
                    "MediaBox": PdfArray([0, 0, 612, 792]),
                }
            ),
            5: PdfDict({"Fields": PdfArray([PdfRef(6)])}),
            6: PdfDict(
                {
                    "FT": PdfName("Tx"),
                    "T": PdfString.from_text("orphan"),
                    "Rect": PdfArray([72, 700, 300, 718]),
                }
            ),
        }
        profile = profile_document(Document.open(build_document(objects, PdfRef(1))))
        self.assertFalse(any(p.has_widgets for p in profile.pages))
        self.assertIs(profile.doc_class, DocumentClass.EXISTING_ACROFORM)
        # route() sees only the profile, which carries no field count, so on its own it
        # cannot know about a field no page points at.  This divergence is documented on
        # route(); profile.doc_class is the authoritative answer.
        self.assertIs(route(profile), DocumentClass.NON_FORM)
        self.assertTrue(
            any("no page carries a widget annotation" in w for w in profile.warnings)
        )

    def test_metadata_producer_and_tagging(self):
        profile = profile_document(Document.open(metadata_pdf()))
        self.assertEqual(profile.producer, "ZFP Test Harness 1.0")
        self.assertTrue(profile.tagged)

    def test_corrupt_content_stream_does_not_raise(self):
        profile = profile_document(Document.open(broken_content_pdf()))
        self.assertEqual(profile.page_count, 1)
        self.assertIs(profile.pages[0].mode, PageMode.EMPTY)
        self.assertEqual(profile.pages[0].char_count, 0)

    def test_unparseable_page_is_reported_not_raised(self):
        doc = blank(2)

        class ExplodingPage(Page):
            def content_bytes(self):  # type: ignore[override]
                from zfp.core.errors import PdfParseError

                raise PdfParseError("simulated object damage")

        original = doc.page(1)
        doc._pages[1] = ExplodingPage(doc, 1, original.dict, original.ref)

        profile = profile_document(doc)
        self.assertEqual(len(profile.pages), 2)
        self.assertIs(profile.pages[1].mode, PageMode.EMPTY)
        self.assertEqual(profile.pages[1].index, 1)
        self.assertTrue(
            any(w.startswith("page 1 unparseable:") for w in profile.warnings),
            profile.warnings,
        )

    def test_profiling_is_deterministic(self):
        doc = blank(2)
        set_content(doc, 0, text_ops(FORM_TEXT) + rules(3))
        first = profile_to_dict(profile_document(doc))
        second = profile_to_dict(profile_document(doc))
        self.assertEqual(first, second)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
class ReportingTests(unittest.TestCase):
    def test_profile_to_dict_is_json_serialisable(self):
        doc = blank(1)
        set_content(doc, 0, text_ops(FORM_TEXT))
        payload = profile_to_dict(profile_document(doc))
        self.assertEqual(payload["doc_class"], "flat_native_form")
        self.assertEqual(payload["pages"][0]["mode"], "flat_native_form")
        json.loads(dumps(payload))

    def test_describe_reports_the_essentials(self):
        doc = blank(2)
        set_content(doc, 0, text_ops(FORM_TEXT) + rules(4))
        add_image(doc, 1)
        set_content(doc, 1, b"q 612 0 0 792 0 0 cm /Im0 Do Q")
        profile = profile_document(doc)
        text = describe(profile)
        self.assertIn(profile.document_id, text)
        self.assertIn("class      hybrid", text)
        self.assertIn("flat_native_form", text)
        self.assertIn("scanned_form", text)
        self.assertIn("native text pages  [0]", text)
        self.assertIn("raster pages       [1]", text)
        self.assertEqual(text, describe(profile))

    def test_describe_lists_warnings(self):
        profile = profile_document(
            Document.open(acroform_pdf(xfa=True, needs_rendering=True))
        )
        text = describe(profile)
        self.assertIn("warnings", text)
        self.assertIn("dynamic XFA", text)

    def test_marker_table_is_unique_and_lowercase(self):
        labels = [label for label, _ in FORM_MARKER_PATTERNS]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(labels, [label.lower() for label in labels])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
