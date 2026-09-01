"""Unit tests for :mod:`zfp.pdfio.document`."""

from __future__ import annotations

import os
import tempfile
import unittest

from zfp.core.errors import PdfWriteError, ValidationError
from zfp.core.types import FieldType
from zfp.pdfio.document import Document, Page
from zfp.pdfio.objects import PdfArray, PdfDict, PdfName, PdfRef, PdfStream, PdfString
from zfp.pdfio.writer import build_document

FF_RADIO = 1 << 15
FF_MULTILINE = 1 << 12
FF_COMB = 1 << 24
FF_COMBO = 1 << 17
FF_REQUIRED = 1 << 1
FF_READ_ONLY = 1 << 0


def _stream(payload: bytes = b"") -> PdfStream:
    return PdfStream(PdfDict({"Length": len(payload)}), payload)


def inheritance_pdf(
    *,
    media=(0, 0, 400, 500),
    rotate=None,
    crop=None,
    on_page=False,
) -> bytes:
    """A one-page file whose geometry lives on the ``/Pages`` node, not the page.

    With ``on_page`` the attributes are written on the page instead, which is the
    control case for the inheritance tests.
    """
    page = PdfDict({"Type": PdfName("Page"), "Parent": PdfRef(2), "Resources": PdfDict()})
    pages = PdfDict({"Type": PdfName("Pages"), "Kids": PdfArray([PdfRef(3)]), "Count": 1})
    target = page if on_page else pages
    target["MediaBox"] = PdfArray(list(media))
    if rotate is not None:
        target["Rotate"] = rotate
    if crop is not None:
        target["CropBox"] = PdfArray(list(crop))
    objects = {
        1: PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2)}),
        2: pages,
        3: page,
    }
    return build_document(objects, PdfRef(1))


def acroform_pdf() -> bytes:
    """A one-page form: a radio group with two widgets, a text field, a nested field."""
    objects = {}
    objects[1] = PdfDict(
        {"Type": PdfName("Catalog"), "Pages": PdfRef(2), "AcroForm": PdfRef(10)}
    )
    objects[2] = PdfDict(
        {
            "Type": PdfName("Pages"),
            "Kids": PdfArray([PdfRef(3)]),
            "Count": 1,
            "MediaBox": PdfArray([0, 0, 612, 792]),
        }
    )
    objects[3] = PdfDict(
        {
            "Type": PdfName("Page"),
            "Parent": PdfRef(2),
            "MediaBox": PdfArray([0, 0, 612, 792]),
            "Resources": PdfDict(),
            "Annots": PdfArray([PdfRef(6), PdfRef(7), PdfRef(8), PdfRef(12)]),
        }
    )
    # -- radio group: parent field 5 owning widget kids 6 and 7 ----------------------
    objects[5] = PdfDict(
        {
            "T": PdfString.from_text("gender"),
            "FT": PdfName("Btn"),
            "Ff": FF_RADIO,
            "V": PdfName("F"),
            "Kids": PdfArray([PdfRef(6), PdfRef(7)]),
        }
    )
    objects[6] = PdfDict(
        {
            "Type": PdfName("Annot"),
            "Subtype": PdfName("Widget"),
            "Parent": PdfRef(5),
            "Rect": PdfArray([100, 700, 112, 712]),
            "P": PdfRef(3),
            "AS": PdfName("Off"),
            "AP": PdfDict({"N": PdfDict({"M": PdfRef(20), "Off": PdfRef(21)})}),
        }
    )
    objects[7] = PdfDict(
        {
            "Type": PdfName("Annot"),
            "Subtype": PdfName("Widget"),
            "Parent": PdfRef(5),
            "Rect": PdfArray([200, 712, 212, 700]),  # deliberately un-normalized
            "P": PdfRef(3),
            "AS": PdfName("F"),
            "AP": PdfDict({"N": PdfDict({"F": PdfRef(22), "Off": PdfRef(23)})}),
        }
    )
    objects[20] = _stream(b"")
    objects[21] = _stream(b"")
    objects[22] = _stream(b"")
    objects[23] = _stream(b"")
    # -- merged text field + widget ---------------------------------------------------
    objects[8] = PdfDict(
        {
            "Type": PdfName("Annot"),
            "Subtype": PdfName("Widget"),
            "FT": PdfName("Tx"),
            "T": PdfString.from_text("full_name"),
            "TU": PdfString.from_text("Legal name"),
            "V": PdfString.from_text("Ada Lovelace"),
            "DV": PdfString.from_text(""),
            "MaxLen": 40,
            "Q": 1,
            "Ff": FF_REQUIRED,
            "DA": PdfString(b"/Helv 11 Tf 0 g"),
            "Rect": PdfArray([72, 640, 300, 660]),
            "P": PdfRef(3),
        }
    )
    # -- nested parent 9 -> kid 12 (qualified name "applicant.email") -----------------
    objects[9] = PdfDict(
        {
            "T": PdfString.from_text("applicant"),
            "FT": PdfName("Tx"),
            "DA": PdfString(b"/TiRo 9 Tf 0 g"),
            "Kids": PdfArray([PdfRef(12)]),
        }
    )
    objects[12] = PdfDict(
        {
            "Type": PdfName("Annot"),
            "Subtype": PdfName("Widget"),
            "T": PdfString.from_text("email"),
            "Rect": PdfArray([72, 600, 300, 620]),
            "P": PdfRef(3),
        }
    )
    objects[10] = PdfDict(
        {
            "Fields": PdfArray([PdfRef(5), PdfRef(8), PdfRef(9)]),
            "DA": PdfString(b"/Helv 0 Tf 0 g"),
            "DR": PdfDict({"Font": PdfDict()}),
        }
    )
    return build_document(objects, PdfRef(1))


class BlankDocumentTest(unittest.TestCase):
    def test_builds_a_parseable_file(self):
        doc = Document.from_pages_blank(3)
        self.assertEqual(doc.page_count, 3)
        self.assertTrue(doc.source_bytes.startswith(b"%PDF-"))
        self.assertTrue(doc.document_id.startswith("doc_"))

    def test_geometry_defaults_to_us_letter(self):
        geometry = Document.from_pages_blank(1).page(0).geometry
        self.assertEqual(geometry.media_box.as_list(), [0.0, 0.0, 612.0, 792.0])
        self.assertEqual(geometry.crop_box.as_list(), [0.0, 0.0, 612.0, 792.0])
        self.assertEqual(geometry.rotation, 0)

    def test_custom_page_size(self):
        page = Document.from_pages_blank(1, width=200.0, height=100.0).page(0)
        self.assertEqual(page.geometry.media_box.as_list(), [0.0, 0.0, 200.0, 100.0])

    def test_pages_are_indexed_and_carry_refs(self):
        doc = Document.from_pages_blank(2)
        self.assertEqual([p.index for p in doc.pages], [0, 1])
        for page in doc.pages:
            self.assertIsInstance(page.ref, PdfRef)
        self.assertNotEqual(doc.pages[0].ref, doc.pages[1].ref)

    def test_deterministic_document_id(self):
        self.assertEqual(
            Document.from_pages_blank(2).document_id,
            Document.from_pages_blank(2).document_id,
        )

    def test_content_bytes_is_empty_but_present(self):
        self.assertEqual(Document.from_pages_blank(1).page(0).content_bytes(), b"")

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValidationError):
            Document.from_pages_blank(0)
        with self.assertRaises(ValidationError):
            Document.from_pages_blank(1, width=0)

    def test_page_index_bounds(self):
        doc = Document.from_pages_blank(2)
        self.assertIs(doc.page(-1), doc.pages[1])
        with self.assertRaises(ValidationError):
            doc.page(2)


class OpenSourcesTest(unittest.TestCase):
    def test_open_from_bytes_and_from_path(self):
        data = Document.from_pages_blank(1).source_bytes
        from_bytes = Document.open(data)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blank.pdf")
            with open(path, "wb") as handle:
                handle.write(data)
            from_path = Document.open(path)
            self.assertEqual(from_path.path, path)
        self.assertEqual(from_bytes.document_id, from_path.document_id)
        self.assertEqual(from_bytes.page_count, from_path.page_count)

    def test_password_is_accepted_and_recorded(self):
        data = Document.from_pages_blank(1).source_bytes
        doc = Document.open(data, password="secret")
        self.assertEqual(doc.password, "secret")
        # An unencrypted document ignores the password rather than failing.
        self.assertEqual(doc.page_count, 1)
        self.assertFalse(doc.file.is_encrypted)

    def test_resolver_protocol(self):
        doc = Document.from_pages_blank(1)
        catalog = doc.resolve(doc.catalog_ref())
        self.assertEqual(catalog.get_name("Type"), "Catalog")

    def test_resolve_handles_dangling_and_cyclic_references(self):
        doc = Document.from_pages_blank(1)
        self.assertFalse(doc.resolve(PdfRef(9999)))
        doc.writer.set_object(500, PdfRef(501))
        doc.writer.set_object(501, PdfRef(500))
        self.assertFalse(doc.resolve(PdfRef(500)))


class InheritanceTest(unittest.TestCase):
    def test_media_box_inherited_from_the_pages_node(self):
        page = Document.open(inheritance_pdf(media=(0, 0, 400, 500))).page(0)
        self.assertNotIn("MediaBox", page.dict)
        self.assertEqual(page.inherited("MediaBox"), [0, 0, 400, 500])
        self.assertEqual(page.geometry.media_box.as_list(), [0.0, 0.0, 400.0, 500.0])

    def test_rotate_inherited_from_the_pages_node(self):
        page = Document.open(inheritance_pdf(rotate=90)).page(0)
        self.assertEqual(page.geometry.rotation, 90)
        self.assertEqual(page.geometry.display_size, (500.0, 400.0))

    def test_rotation_normalization(self):
        for raw, expected in ((-90, 270), (450, 90), (720, 0), (-450, 270), (180, 180)):
            page = Document.open(inheritance_pdf(rotate=raw)).page(0)
            self.assertEqual(page.geometry.rotation, expected, "rotate=%r" % raw)

    def test_page_value_overrides_the_inherited_one(self):
        data = inheritance_pdf(media=(0, 0, 400, 500), on_page=True)
        page = Document.open(data).page(0)
        self.assertEqual(page.geometry.media_box.as_list(), [0.0, 0.0, 400.0, 500.0])

    def test_resources_inherited(self):
        doc = Document.open(inheritance_pdf())
        self.assertIsInstance(doc.page(0).resources(), PdfDict)

    def test_missing_key_returns_none(self):
        self.assertIsNone(Document.open(inheritance_pdf()).page(0).inherited("Nope"))


class CropBoxTest(unittest.TestCase):
    def test_crop_box_smaller_than_media_box_is_kept(self):
        data = inheritance_pdf(media=(0, 0, 400, 500), crop=(20, 30, 380, 470))
        geometry = Document.open(data).page(0).geometry
        self.assertEqual(geometry.media_box.as_list(), [0.0, 0.0, 400.0, 500.0])
        self.assertEqual(geometry.crop_box.as_list(), [20.0, 30.0, 380.0, 470.0])
        self.assertEqual(geometry.width, 360.0)
        self.assertEqual(geometry.height, 440.0)

    def test_crop_box_is_intersected_with_the_media_box(self):
        data = inheritance_pdf(media=(0, 0, 400, 500), crop=(-50, -50, 900, 900))
        geometry = Document.open(data).page(0).geometry
        self.assertEqual(geometry.crop_box.as_list(), [0.0, 0.0, 400.0, 500.0])

    def test_disjoint_crop_box_falls_back_to_the_media_box(self):
        data = inheritance_pdf(media=(0, 0, 400, 500), crop=(900, 900, 1000, 1000))
        geometry = Document.open(data).page(0).geometry
        self.assertEqual(geometry.crop_box.as_list(), [0.0, 0.0, 400.0, 500.0])

    def test_crop_box_defaults_to_the_media_box(self):
        geometry = Document.open(inheritance_pdf(media=(0, 0, 400, 500))).page(0).geometry
        self.assertEqual(geometry.crop_box.as_list(), geometry.media_box.as_list())


class AnnotationTest(unittest.TestCase):
    def test_add_annotation_survives_an_incremental_save(self):
        doc = Document.from_pages_blank(1)
        original = doc.source_bytes
        annot = doc.writer.add_object(
            PdfDict(
                {
                    "Type": PdfName("Annot"),
                    "Subtype": PdfName("Square"),
                    "Rect": PdfArray([10, 20, 110, 60]),
                    "T": PdfString.from_text("probe"),
                }
            )
        )
        doc.page(0).add_annotation(annot)
        out = doc.to_bytes(incremental=True)

        # The visual substrate is untouched: the original bytes are an exact prefix.
        self.assertEqual(out[: len(original)], original)
        self.assertGreater(len(out), len(original))

        reopened = Document.open(out)
        annots = reopened.page(0).annotations()
        self.assertEqual(len(annots), 1)
        self.assertEqual(annots[0].get_name("Subtype"), "Square")
        self.assertEqual(annots[0].get("Rect"), [10, 20, 110, 60])

    def test_add_annotation_is_visible_before_saving(self):
        doc = Document.from_pages_blank(1)
        ref = doc.writer.add_object(PdfDict({"Subtype": PdfName("Widget")}))
        doc.page(0).add_annotation(ref)
        self.assertEqual(len(doc.page(0).annotations()), 1)
        self.assertEqual(doc.page(0).annotation_refs(), [ref])

    def test_appending_to_an_existing_indirect_annots_array(self):
        objects = {
            1: PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2)}),
            2: PdfDict(
                {
                    "Type": PdfName("Pages"),
                    "Kids": PdfArray([PdfRef(3)]),
                    "Count": 1,
                    "MediaBox": PdfArray([0, 0, 612, 792]),
                }
            ),
            3: PdfDict({"Type": PdfName("Page"), "Parent": PdfRef(2), "Annots": PdfRef(4)}),
            4: PdfArray([PdfRef(5)]),
            5: PdfDict({"Type": PdfName("Annot"), "Subtype": PdfName("Link")}),
        }
        doc = Document.open(build_document(objects, PdfRef(1)))
        ref = doc.writer.add_object(PdfDict({"Subtype": PdfName("Widget")}))
        doc.page(0).add_annotation(ref)
        # Object 4 (the array) changed; the page dictionary did not need to.
        self.assertIn(4, doc.writer.updates)
        self.assertNotIn(3, doc.writer.updates)

        reopened = Document.open(doc.to_bytes())
        subtypes = [a.get_name("Subtype") for a in reopened.page(0).annotations()]
        self.assertEqual(subtypes, ["Link", "Widget"])

    def test_add_annotation_rejects_a_non_reference(self):
        doc = Document.from_pages_blank(1)
        with self.assertRaises(PdfWriteError):
            doc.page(0).add_annotation(PdfDict())

    def test_add_annotation_needs_a_page_object_number(self):
        doc = Document.from_pages_blank(1)
        orphan = Page(doc, 0, PdfDict(), ref=None)
        with self.assertRaises(PdfWriteError):
            orphan.add_annotation(PdfRef(99))


class SaveTest(unittest.TestCase):
    def test_save_incremental_and_full(self):
        doc = Document.from_pages_blank(1)
        original = doc.source_bytes
        ref = doc.writer.add_object(PdfDict({"Subtype": PdfName("Widget")}))
        doc.page(0).add_annotation(ref)
        with tempfile.TemporaryDirectory() as tmp:
            incremental_path = os.path.join(tmp, "inc.pdf")
            full_path = os.path.join(tmp, "full.pdf")
            doc.save(incremental_path, incremental=True)
            doc.save(full_path, incremental=False)
            with open(incremental_path, "rb") as handle:
                incremental = handle.read()
            with open(full_path, "rb") as handle:
                full = handle.read()
        self.assertEqual(incremental[: len(original)], original)
        self.assertNotEqual(full[: len(original)], original)
        for data in (incremental, full):
            self.assertEqual(len(Document.open(data).page(0).annotations()), 1)

    def test_unmodified_document_round_trips_byte_identically(self):
        doc = Document.from_pages_blank(1)
        self.assertEqual(doc.to_bytes(incremental=True), doc.source_bytes)


class AcroFormTest(unittest.TestCase):
    def test_no_acroform_on_a_blank_document(self):
        doc = Document.from_pages_blank(1)
        self.assertIsNone(doc.acroform())
        self.assertFalse(doc.has_xfa())
        self.assertFalse(doc.is_signed())

    def test_ensure_acroform_creates_the_expected_shape(self):
        doc = Document.from_pages_blank(1)
        acroform = doc.ensure_acroform()
        self.assertEqual(list(acroform.get("Fields")), [])
        self.assertEqual(acroform.get("DA").text(), "/Helv 0 Tf 0 g")
        self.assertIsInstance(acroform.get("DR").get("Font"), PdfDict)
        self.assertIsInstance(doc.acroform_ref, PdfRef)

    def test_ensure_acroform_is_idempotent(self):
        doc = Document.from_pages_blank(1)
        first = doc.ensure_acroform()
        staged = dict(doc.writer.updates)
        second = doc.ensure_acroform()
        self.assertIs(first, second)
        self.assertEqual(set(doc.writer.updates), set(staged))

    def test_ensure_acroform_survives_a_save(self):
        doc = Document.from_pages_blank(1)
        doc.ensure_acroform()
        reopened = Document.open(doc.to_bytes())
        acroform = reopened.acroform()
        self.assertIsNotNone(acroform)
        self.assertEqual(acroform.get("DA").text(), "/Helv 0 Tf 0 g")
        self.assertEqual(list(acroform.get("Fields")), [])

    def test_ensure_acroform_promotes_a_direct_dictionary(self):
        objects = {
            1: PdfDict(
                {
                    "Type": PdfName("Catalog"),
                    "Pages": PdfRef(2),
                    "AcroForm": PdfDict({"Fields": PdfArray()}),
                }
            ),
            2: PdfDict(
                {
                    "Type": PdfName("Pages"),
                    "Kids": PdfArray([PdfRef(3)]),
                    "Count": 1,
                    "MediaBox": PdfArray([0, 0, 612, 792]),
                }
            ),
            3: PdfDict({"Type": PdfName("Page"), "Parent": PdfRef(2)}),
        }
        doc = Document.open(build_document(objects, PdfRef(1)))
        acroform = doc.ensure_acroform()
        acroform["Fields"] = PdfArray([PdfRef(1)])
        self.assertIsInstance(doc.acroform_ref, PdfRef)
        reopened = Document.open(doc.to_bytes())
        self.assertEqual(list(reopened.acroform().get("Fields")), [PdfRef(1)])

    def test_acroform_is_found_on_a_real_form(self):
        doc = Document.open(acroform_pdf())
        acroform = doc.acroform()
        self.assertIsNotNone(acroform)
        self.assertEqual(len(acroform.get("Fields")), 3)


class ExistingFieldsTest(unittest.TestCase):
    def setUp(self):
        self.doc = Document.open(acroform_pdf())
        self.fields = {spec.name: spec for spec in self.doc.existing_fields()}

    def test_field_names_and_count(self):
        self.assertEqual(
            sorted(self.fields), ["applicant.email", "full_name", "gender"]
        )

    def test_radio_group_inherits_type_and_flags_from_the_parent(self):
        radio = self.fields["gender"]
        self.assertIs(radio.field_type, FieldType.RADIO)
        self.assertEqual(radio.group, "gender")
        self.assertEqual(radio.value, "F")

    def test_radio_widgets_are_split_between_rect_and_extra_widgets(self):
        radio = self.fields["gender"]
        self.assertEqual(radio.page, 0)
        self.assertEqual(radio.rect.as_list(), [100.0, 700.0, 112.0, 712.0])
        self.assertEqual(len(radio.extra_widgets), 1)
        page, rect = radio.extra_widgets[0]
        self.assertEqual(page, 0)
        # The source rectangle was written y1 < y0; it comes back normalized.
        self.assertEqual(rect.as_list(), [200.0, 700.0, 212.0, 712.0])

    def test_radio_export_value_comes_from_the_appearance_states(self):
        self.assertEqual(self.fields["gender"].export_value, "M")

    def test_text_field_attributes(self):
        text = self.fields["full_name"]
        self.assertIs(text.field_type, FieldType.TEXT)
        self.assertEqual(text.value, "Ada Lovelace")
        self.assertEqual(text.tooltip, "Legal name")
        self.assertEqual(text.max_length, 40)
        self.assertEqual(text.alignment, 1)
        self.assertTrue(text.required)
        self.assertFalse(text.read_only)
        self.assertEqual(text.rect.as_list(), [72.0, 640.0, 300.0, 660.0])
        self.assertIsNone(text.export_value)

    def test_default_appearance_is_parsed(self):
        self.assertEqual(self.fields["full_name"].font_name, "Helv")
        self.assertEqual(self.fields["full_name"].font_size, 11.0)

    def test_nested_field_name_is_fully_qualified_and_inherits_da(self):
        email = self.fields["applicant.email"]
        self.assertIs(email.field_type, FieldType.TEXT)
        self.assertEqual(email.font_name, "TiRo")
        self.assertEqual(email.font_size, 9.0)
        self.assertEqual(email.page, 0)
        self.assertEqual(email.rect.as_list(), [72.0, 600.0, 300.0, 620.0])

    def test_empty_document_has_no_fields(self):
        self.assertEqual(Document.from_pages_blank(1).existing_fields(), [])

    def test_fields_survive_a_full_rewrite(self):
        rewritten = Document.open(self.doc.to_bytes(incremental=False))
        names = sorted(spec.name for spec in rewritten.existing_fields())
        self.assertEqual(names, ["applicant.email", "full_name", "gender"])


class FieldTypeMappingTest(unittest.TestCase):
    def _spec(self, ft: str, flags: int = 0, **extra):
        field = PdfDict(
            {
                "T": PdfString.from_text("f"),
                "FT": PdfName(ft),
                "Ff": flags,
                "Rect": PdfArray([0, 0, 10, 10]),
                "Subtype": PdfName("Widget"),
                "P": PdfRef(3),
            }
        )
        field.update(extra)
        objects = {
            1: PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2), "AcroForm": PdfRef(4)}),
            2: PdfDict(
                {
                    "Type": PdfName("Pages"),
                    "Kids": PdfArray([PdfRef(3)]),
                    "Count": 1,
                    "MediaBox": PdfArray([0, 0, 612, 792]),
                }
            ),
            3: PdfDict({"Type": PdfName("Page"), "Parent": PdfRef(2), "Annots": PdfArray([PdfRef(5)])}),
            4: PdfDict({"Fields": PdfArray([PdfRef(5)])}),
            5: field,
        }
        doc = Document.open(build_document(objects, PdfRef(1)))
        return doc.existing_fields()[0]

    def test_button_variants(self):
        self.assertIs(self._spec("Btn").field_type, FieldType.CHECKBOX)
        self.assertIs(self._spec("Btn", FF_RADIO).field_type, FieldType.RADIO)
        self.assertIs(self._spec("Btn", 1 << 16).field_type, FieldType.BUTTON)

    def test_choice_variants(self):
        self.assertIs(self._spec("Ch").field_type, FieldType.LISTBOX)
        self.assertIs(self._spec("Ch", FF_COMBO).field_type, FieldType.CHOICE)

    def test_text_variants(self):
        self.assertIs(self._spec("Tx").field_type, FieldType.TEXT)
        self.assertIs(self._spec("Tx", FF_MULTILINE).field_type, FieldType.MULTILINE_TEXT)
        comb = self._spec("Tx", FF_COMB, MaxLen=9)
        self.assertIs(comb.field_type, FieldType.COMB)
        self.assertEqual(comb.comb_cells, 9)

    def test_signature_and_unknown(self):
        self.assertIs(self._spec("Sig").field_type, FieldType.SIGNATURE)
        self.assertIs(self._spec("Zz").field_type, FieldType.UNKNOWN)

    def test_read_only_flag(self):
        self.assertTrue(self._spec("Tx", FF_READ_ONLY).read_only)

    def test_choices_from_opt(self):
        opt = PdfArray(
            [
                PdfString.from_text("Alpha"),
                PdfArray([PdfString.from_text("b"), PdfString.from_text("Beta")]),
            ]
        )
        spec = self._spec("Ch", FF_COMBO, Opt=opt)
        self.assertEqual(spec.choices, ["Alpha", "Beta"])


class SignatureAndXfaTest(unittest.TestCase):
    def _doc(self, *, catalog_extra=None, acroform_extra=None, field=None) -> Document:
        objects = {
            1: PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2), "AcroForm": PdfRef(4)}),
            2: PdfDict(
                {
                    "Type": PdfName("Pages"),
                    "Kids": PdfArray([PdfRef(3)]),
                    "Count": 1,
                    "MediaBox": PdfArray([0, 0, 612, 792]),
                }
            ),
            3: PdfDict({"Type": PdfName("Page"), "Parent": PdfRef(2)}),
            4: PdfDict({"Fields": PdfArray()}),
        }
        if field is not None:
            objects[5] = field
            objects[4]["Fields"] = PdfArray([PdfRef(5)])
        objects[1].update(catalog_extra or {})
        objects[4].update(acroform_extra or {})
        return Document.open(build_document(objects, PdfRef(1)))

    def test_unsigned(self):
        self.assertFalse(self._doc().is_signed())

    def test_signature_field_without_a_value_is_not_signed(self):
        field = PdfDict({"T": PdfString.from_text("sig"), "FT": PdfName("Sig")})
        self.assertFalse(self._doc(field=field).is_signed())

    def test_signature_field_with_a_value_is_signed(self):
        field = PdfDict(
            {
                "T": PdfString.from_text("sig"),
                "FT": PdfName("Sig"),
                "V": PdfDict({"Type": PdfName("Sig")}),
            }
        )
        self.assertTrue(self._doc(field=field).is_signed())

    def test_docmdp_permission_counts_as_signed(self):
        extra = {"Perms": PdfDict({"DocMDP": PdfDict({"Type": PdfName("Sig")})})}
        self.assertTrue(self._doc(catalog_extra=extra).is_signed())

    def test_signature_widget_on_a_page_counts_as_signed(self):
        objects = {
            1: PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2)}),
            2: PdfDict(
                {
                    "Type": PdfName("Pages"),
                    "Kids": PdfArray([PdfRef(3)]),
                    "Count": 1,
                    "MediaBox": PdfArray([0, 0, 612, 792]),
                }
            ),
            3: PdfDict({"Type": PdfName("Page"), "Parent": PdfRef(2), "Annots": PdfArray([PdfRef(4)])}),
            4: PdfDict(
                {
                    "Type": PdfName("Annot"),
                    "Subtype": PdfName("Widget"),
                    "FT": PdfName("Sig"),
                    "V": PdfDict({"Type": PdfName("Sig")}),
                }
            ),
        }
        self.assertTrue(Document.open(build_document(objects, PdfRef(1))).is_signed())

    def test_has_xfa(self):
        self.assertFalse(self._doc().has_xfa())
        doc = self._doc(acroform_extra={"XFA": PdfArray()})
        self.assertTrue(doc.has_xfa())

    def test_static_xfa_is_not_dynamic(self):
        packet = _stream(b"<config><present><pdf><dynamicRender>forbidden</dynamicRender>")
        objects_doc = self._doc_with_xfa(packet)
        self.assertTrue(objects_doc.has_xfa())
        self.assertFalse(objects_doc.xfa_is_dynamic())

    def test_dynamic_xfa_is_detected(self):
        packet = _stream(b"<config><present><pdf><dynamicRender>required</dynamicRender>")
        self.assertTrue(self._doc_with_xfa(packet).xfa_is_dynamic())

    def test_needs_rendering_flag_is_enough(self):
        doc = self._doc(
            catalog_extra={"NeedsRendering": True},
            acroform_extra={"XFA": PdfArray()},
        )
        self.assertTrue(doc.xfa_is_dynamic())

    def test_no_xfa_is_never_dynamic(self):
        self.assertFalse(self._doc().xfa_is_dynamic())

    def _doc_with_xfa(self, packet: PdfStream) -> Document:
        objects = {
            1: PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2), "AcroForm": PdfRef(4)}),
            2: PdfDict(
                {
                    "Type": PdfName("Pages"),
                    "Kids": PdfArray([PdfRef(3)]),
                    "Count": 1,
                    "MediaBox": PdfArray([0, 0, 612, 792]),
                }
            ),
            3: PdfDict({"Type": PdfName("Page"), "Parent": PdfRef(2)}),
            4: PdfDict(
                {
                    "Fields": PdfArray(),
                    "XFA": PdfArray([PdfString.from_text("config"), PdfRef(5)]),
                }
            ),
            5: packet,
        }
        return Document.open(build_document(objects, PdfRef(1)))


class DefaultAppearanceParsingTest(unittest.TestCase):
    def test_variants(self):
        parse = Document.parse_default_appearance
        self.assertEqual(parse(PdfString(b"/Helv 12 Tf 0 g")), ("Helv", 12.0))
        self.assertEqual(parse(PdfString(b"0 g /TiRo 9.5 Tf")), ("TiRo", 9.5))
        self.assertEqual(parse(PdfString(b"/F#231 0 Tf")), ("F#1", 0.0))
        self.assertEqual(parse(PdfString(b"1 0 0 RG")), ("Helv", 0.0))
        self.assertEqual(parse(None), ("Helv", 0.0))

    def test_last_font_operator_wins(self):
        parse = Document.parse_default_appearance
        self.assertEqual(parse(PdfString(b"/Helv 8 Tf /Cour 14 Tf")), ("Cour", 14.0))


class ContentBytesTest(unittest.TestCase):
    def test_array_of_streams_is_joined(self):
        objects = {
            1: PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2)}),
            2: PdfDict(
                {
                    "Type": PdfName("Pages"),
                    "Kids": PdfArray([PdfRef(3)]),
                    "Count": 1,
                    "MediaBox": PdfArray([0, 0, 612, 792]),
                }
            ),
            3: PdfDict(
                {
                    "Type": PdfName("Page"),
                    "Parent": PdfRef(2),
                    "Contents": PdfArray([PdfRef(4), PdfRef(5)]),
                }
            ),
            4: _stream(b"q 1 0 0 1 0 0 cm"),
            5: _stream(b"Q"),
        }
        page = Document.open(build_document(objects, PdfRef(1))).page(0)
        self.assertEqual(page.content_bytes(), b"q 1 0 0 1 0 0 cm\nQ")

    def test_missing_contents_is_empty(self):
        page = Document.open(inheritance_pdf()).page(0)
        self.assertEqual(page.content_bytes(), b"")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
