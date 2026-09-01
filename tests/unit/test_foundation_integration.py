"""Cross-module integration tests for the ZFP foundation layer.

Every other module in ``tests/unit`` exercises one specialist's module in isolation.
This file exercises the *seams*: the places where two independently written modules have
to agree about a helper name, a return shape or who is responsible for staging an edit.

The behaviours pinned here are the ones the rest of ZFP is built on:

a. :meth:`Document.from_pages_blank` -> :meth:`~Document.to_bytes` -> :meth:`Document.open`
   survives a full round trip with the right page count and geometry.
b. ``ensure_acroform()`` + ``add_annotation()`` + an *incremental* write reload intact --
   including on a document whose ``/AcroForm`` already existed (a second editing pass).
c. A corrupted ``startxref`` still loads, via the brute-force rebuild.
d. ``fonts.text_width`` / ``fonts.fit_font_size`` agree with each other and with ``Rect``.
e. ChaCha20-Poly1305 round trips and rejects tampering.
f. The ontology resolves a messy human label to a canonical key.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from zfp.core.errors import VaultError
from zfp.core.geometry import Rect
from zfp.core.types import FieldType
from zfp.pdfio import fonts
from zfp.pdfio.document import Document
from zfp.pdfio.objects import PdfArray, PdfDict, PdfName, PdfRef, PdfString
from zfp.pdfio.parser import PdfFile
from zfp.vault.cipher import chacha20_poly1305_decrypt, chacha20_poly1305_encrypt

import zfp.ontology as ontology

LETTER_W = 612.0
LETTER_H = 792.0


def _widget(name: str, rect: PdfArray) -> PdfDict:
    """A minimal but complete text-field widget annotation."""
    return PdfDict(
        {
            "Type": PdfName("Annot"),
            "Subtype": PdfName("Widget"),
            "Rect": rect,
            "FT": PdfName("Tx"),
            "T": PdfString.from_text(name),
            "F": 4,
        }
    )


def _add_field(doc: Document, page_index: int, name: str, rect: PdfArray) -> PdfRef:
    """Attach a widget to a page *and* register it in the AcroForm, the way a real
    field writer has to: the annotation is a new indirect object, the page's ``/Annots``
    gains a reference to it, and the form's ``/Fields`` gains the same reference."""
    acroform = doc.ensure_acroform()
    ref = doc.writer.add_object(_widget(name, rect))
    doc.page(page_index).add_annotation(ref)
    fields = doc.resolve(acroform["Fields"])
    fields.append(ref)
    return ref


# ======================================================================================
# (a) blank document round trip
# ======================================================================================


class BlankDocumentRoundTripTests(unittest.TestCase):
    """``from_pages_blank`` -> ``to_bytes`` -> ``open`` keeps count and geometry."""

    def test_two_blank_pages_reload_with_correct_geometry(self):
        doc = Document.from_pages_blank(2)
        self.assertEqual(doc.page_count, 2)

        data = doc.to_bytes()
        self.assertTrue(data.startswith(b"%PDF-"))

        reopened = Document.open(data)
        self.assertEqual(reopened.page_count, 2)
        self.assertEqual(len(reopened.pages), 2)

        for index, page in enumerate(reopened.pages):
            geometry = page.geometry
            self.assertEqual(geometry.index, index)
            self.assertEqual(geometry.width, LETTER_W)
            self.assertEqual(geometry.height, LETTER_H)
            self.assertEqual(geometry.rotation, 0)
            self.assertEqual(geometry.media_box, Rect(0.0, 0.0, LETTER_W, LETTER_H))
            self.assertEqual(geometry.crop_box, geometry.media_box)
            self.assertEqual(geometry.display_size, (LETTER_W, LETTER_H))

    def test_a_custom_page_size_survives_the_round_trip(self):
        doc = Document.open(Document.from_pages_blank(1, 200.0, 400.0).to_bytes())
        self.assertEqual(doc.pages[0].geometry.media_box, Rect(0.0, 0.0, 200.0, 400.0))

    def test_round_trip_through_a_file_on_disk(self):
        doc = Document.from_pages_blank(2)
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "blank.pdf")
        doc.save(path, incremental=False)

        self.assertEqual(Document.open(path).page_count, 2)
        self.assertEqual(PdfFile.open(path).page_dicts().__len__(), 2)

    def test_document_id_is_deterministic_and_well_formed(self):
        data = Document.from_pages_blank(1).to_bytes()
        first, second = Document.open(data), Document.open(data)
        self.assertEqual(first.document_id, second.document_id)
        # "doc_<hex>", not "doc__<hex>": stable_id supplies the separator itself.
        self.assertTrue(first.document_id.startswith("doc_"))
        self.assertFalse(first.document_id.startswith("doc__"))


# ======================================================================================
# (b) AcroForm + annotation across an incremental write
# ======================================================================================


class IncrementalFormRoundTripTests(unittest.TestCase):
    """A field added to a blank document survives an incremental save."""

    def test_annotation_and_acroform_survive_an_incremental_write(self):
        doc = Document.from_pages_blank(2)
        rect = PdfArray([72.0, 700.0, 272.0, 720.0])
        _add_field(doc, 0, "given_name", rect)

        data = doc.to_bytes(incremental=True)
        reopened = Document.open(data)

        self.assertEqual(reopened.page_count, 2)

        acroform = reopened.acroform()
        self.assertIsNotNone(acroform)
        self.assertEqual(len(acroform.get("Fields")), 1)

        annots = reopened.pages[0].annotations()
        self.assertEqual(len(annots), 1)
        self.assertEqual(annots[0].get_name("Subtype"), "Widget")
        self.assertEqual(annots[0]["T"].text(), "given_name")
        # The untouched page stays untouched.
        self.assertEqual(reopened.pages[1].annotations(), [])

        fields = reopened.existing_fields()
        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].name, "given_name")
        self.assertEqual(fields[0].field_type, FieldType.TEXT)
        self.assertEqual(fields[0].page, 0)
        self.assertEqual(fields[0].rect, Rect(72.0, 700.0, 272.0, 720.0))

    def test_incremental_write_preserves_the_original_bytes_exactly(self):
        original = Document.from_pages_blank(1).to_bytes(incremental=False)
        doc = Document.open(original)
        _add_field(doc, 0, "f", PdfArray([0.0, 0.0, 10.0, 10.0]))
        updated = doc.to_bytes(incremental=True)

        self.assertTrue(updated.startswith(original))
        self.assertGreater(len(updated), len(original))

    def test_a_second_editing_pass_still_registers_its_field(self):
        """Regression: ``ensure_acroform`` must stage an *already indirect* form.

        The first pass creates the /AcroForm, so the writer knows its object number.
        The second pass only resolves it -- and unless ``ensure_acroform`` re-stages
        that object, appending to /Fields mutates a dictionary the writer never
        serializes, so the field silently vanishes on reload.
        """
        first = Document.from_pages_blank(1)
        _add_field(first, 0, "one", PdfArray([0.0, 0.0, 10.0, 10.0]))
        pass_one = first.to_bytes(incremental=True)

        second = Document.open(pass_one)
        _add_field(second, 0, "two", PdfArray([20.0, 20.0, 30.0, 30.0]))
        pass_two = second.to_bytes(incremental=True)

        self.assertTrue(pass_two.startswith(pass_one))

        reopened = Document.open(pass_two)
        self.assertEqual(len(reopened.acroform().get("Fields")), 2)
        self.assertEqual(len(reopened.pages[0].annotations()), 2)
        self.assertEqual(
            sorted(f.name for f in reopened.existing_fields()), ["one", "two"]
        )

    def test_direct_page_edits_persist_once_the_page_is_touched(self):
        """``page.dict`` is handed out live; ``touch()`` is what stages it."""
        doc = Document.open(Document.from_pages_blank(1).to_bytes(incremental=False))
        page = doc.pages[0]
        self.assertEqual(page.geometry.rotation, 0)  # also primes the geometry cache

        page.dict["Rotate"] = 90
        page.touch()

        # The cache was dropped, so the live object already reads back correctly ...
        self.assertEqual(page.geometry.rotation, 90)
        self.assertEqual(page.geometry.display_size, (LETTER_H, LETTER_W))

        # ... and the change is in the bytes.
        reopened = Document.open(doc.to_bytes(incremental=True))
        self.assertEqual(reopened.pages[0].geometry.rotation, 90)
        self.assertEqual(reopened.pages[0].geometry.display_size, (LETTER_H, LETTER_W))

    def test_a_font_registered_through_fonts_reaches_the_saved_form(self):
        """The fonts module writes into a Document supplied by document.py."""
        doc = Document.open(Document.from_pages_blank(1).to_bytes(incremental=False))
        doc.ensure_acroform()

        name, ref = fonts.ensure_standard_font(doc, "Helvetica")
        self.assertIsInstance(name, str)
        self.assertTrue(name)
        self.assertIsInstance(ref, PdfRef)
        # Idempotent: asking twice reuses the same font object.
        self.assertEqual((name, ref), fonts.ensure_standard_font(doc, "Helvetica"))

        reopened = Document.open(doc.to_bytes(incremental=True))
        resources = reopened.resolve(reopened.acroform()["DR"])
        font_dict = reopened.resolve(resources["Font"])
        self.assertIn(name, font_dict)

        font = reopened.resolve(font_dict[name])
        self.assertEqual(font.get_name("Type"), "Font")
        self.assertEqual(font.get_name("Subtype"), "Type1")
        self.assertEqual(font.get_name("BaseFont"), "Helvetica")


# ======================================================================================
# (c) recovery from a corrupted cross-reference
# ======================================================================================


class BrokenXrefRecoveryTests(unittest.TestCase):
    """A file whose ``startxref`` points nowhere still loads, by rebuilding."""

    @staticmethod
    def _corrupt_startxref(data: bytes) -> bytes:
        index = data.rfind(b"startxref")
        assert index != -1, "fixture has no startxref"
        return data[:index] + b"startxref\n999999999\n%%EOF\n"

    def test_pdffile_load_rebuilds_when_startxref_is_wrong(self):
        good = Document.from_pages_blank(2).to_bytes(incremental=False)
        pdf = PdfFile.load(self._corrupt_startxref(good))

        self.assertTrue(pdf.rebuilt)
        self.assertEqual(len(pdf.page_dicts()), 2)
        self.assertIsInstance(pdf.catalog, PdfDict)
        self.assertEqual(pdf.catalog.get_name("Type"), "Catalog")

    def test_document_open_recovers_the_same_pages_and_geometry(self):
        good = Document.from_pages_blank(2).to_bytes(incremental=False)
        recovered = Document.open(self._corrupt_startxref(good))

        self.assertTrue(recovered.file.rebuilt)
        self.assertEqual(recovered.page_count, 2)
        for page in recovered.pages:
            self.assertEqual(page.geometry.width, LETTER_W)
            self.assertEqual(page.geometry.height, LETTER_H)

    def test_a_rebuilt_document_is_still_editable(self):
        good = Document.from_pages_blank(1).to_bytes(incremental=False)
        recovered = Document.open(self._corrupt_startxref(good))
        _add_field(recovered, 0, "after_rebuild", PdfArray([1.0, 2.0, 3.0, 4.0]))

        reopened = Document.open(recovered.to_bytes(incremental=True))
        self.assertEqual(reopened.page_count, 1)
        self.assertEqual(
            [f.name for f in reopened.existing_fields()], ["after_rebuild"]
        )

    def test_a_truncated_trailer_is_also_recoverable(self):
        good = Document.from_pages_blank(2).to_bytes(incremental=False)
        truncated = good[: good.rfind(b"xref")]
        recovered = Document.open(truncated)

        self.assertTrue(recovered.file.rebuilt)
        self.assertEqual(recovered.page_count, 2)


# ======================================================================================
# (d) font metrics
# ======================================================================================


class FontMetricsIntegrationTests(unittest.TestCase):
    """``text_width`` and ``fit_font_size`` have to agree with each other."""

    def test_text_width_is_a_sane_positive_float(self):
        width = fonts.text_width("Hello", "Helvetica", 12.0)
        self.assertIsInstance(width, float)
        self.assertGreater(width, 0.0)
        # Five glyphs at 12pt cannot be narrower than a thin pen stroke nor wider
        # than five full ems.
        self.assertGreater(width, 12.0)
        self.assertLess(width, 5 * 12.0)

    def test_width_scales_linearly_with_size(self):
        at_12 = fonts.text_width("Hello", "Helvetica", 12.0)
        at_24 = fonts.text_width("Hello", "Helvetica", 24.0)
        self.assertAlmostEqual(at_24, at_12 * 2.0, places=6)

    def test_width_grows_with_the_string(self):
        self.assertLess(
            fonts.text_width("Hello", "Helvetica", 12.0),
            fonts.text_width("Hello there", "Helvetica", 12.0),
        )

    def test_empty_text_has_no_width(self):
        self.assertEqual(fonts.text_width("", "Helvetica", 12.0), 0.0)

    def test_fit_font_size_shrinks_a_long_string_in_a_small_rect(self):
        rect = Rect(0.0, 0.0, 100.0, 14.0)
        text = "x" * 60
        size = fonts.fit_font_size(text, "Helvetica", rect)

        # It actually shrank, and stayed inside the documented bounds.
        self.assertLess(size, 12.0)
        self.assertGreaterEqual(size, 4.0)
        self.assertIsInstance(size, float)

        # A short string in the same box is allowed to be much bigger -- though in a
        # 14pt-tall box the leading, not the width, is what caps it.
        short = fonts.fit_font_size("Hi", "Helvetica", rect)
        self.assertGreater(short, size)
        self.assertAlmostEqual(short, round((14.0 - 4.0) / fonts.LEADING_FACTOR, 2), places=2)

        # Give it room in both axes and nothing is shrunk at all.
        self.assertEqual(fonts.fit_font_size("Hi", "Helvetica", Rect(0, 0, 100, 40)), 12.0)

    def test_the_chosen_size_really_fits_when_a_fit_is_possible(self):
        rect = Rect(0.0, 0.0, 200.0, 20.0)
        text = "Jane Q. Public, 1234 Long Winding Street"
        size = fonts.fit_font_size(text, "Helvetica", rect)

        self.assertLess(size, 12.0)
        self.assertLessEqual(
            fonts.text_width(text, "Helvetica", size), rect.width - 2 * 2.0 + 1e-6
        )
        # One notch larger would overflow, so the answer is the largest that fits.
        self.assertGreater(
            fonts.text_width(text, "Helvetica", size + 0.5), rect.width - 2 * 2.0
        )

    def test_height_alone_can_constrain_the_size(self):
        # A very wide but very short box: leading, not advance width, is the limit.
        self.assertLess(fonts.fit_font_size("Hi", "Helvetica", Rect(0, 0, 10_000, 10)), 12.0)

    def test_wrapped_lines_measure_back_within_the_width(self):
        text = "The quick brown fox jumps over the lazy dog near the riverbank"
        width = 120.0
        lines = fonts.wrap_text(text, "Helvetica", 10.0, width)

        self.assertGreater(len(lines), 1)
        for line in lines:
            # Trailing whitespace is deliberately preserved by wrap_text and renders
            # as nothing, so measure the visible part.
            self.assertLessEqual(
                fonts.text_width(line.rstrip(), "Helvetica", 10.0), width + 1e-6, line
            )
        self.assertEqual(" ".join(lines).split(), text.split())


# ======================================================================================
# (e) ChaCha20-Poly1305
# ======================================================================================


class ChaChaRoundTripTests(unittest.TestCase):
    KEY = bytes(range(32))
    NONCE = bytes(range(12))

    def test_encrypt_decrypt_round_trip(self):
        plaintext = b"the vault holds the user's real data"
        aad = b"zfp-vault-v1"

        ciphertext = chacha20_poly1305_encrypt(self.KEY, self.NONCE, plaintext, aad)
        self.assertIsInstance(ciphertext, bytes)
        self.assertNotEqual(ciphertext, plaintext)
        # Ciphertext carries a 16 byte Poly1305 tag.
        self.assertEqual(len(ciphertext), len(plaintext) + 16)

        self.assertEqual(
            chacha20_poly1305_decrypt(self.KEY, self.NONCE, ciphertext, aad), plaintext
        )

    def test_round_trip_without_aad_and_on_empty_input(self):
        for plaintext in (b"", b"x", b"y" * 1000):
            ciphertext = chacha20_poly1305_encrypt(self.KEY, self.NONCE, plaintext)
            self.assertEqual(
                chacha20_poly1305_decrypt(self.KEY, self.NONCE, ciphertext), plaintext
            )

    def test_a_tampered_ciphertext_is_rejected(self):
        ciphertext = bytearray(
            chacha20_poly1305_encrypt(self.KEY, self.NONCE, b"secret", b"aad")
        )
        ciphertext[0] ^= 0x01
        with self.assertRaises(VaultError):
            chacha20_poly1305_decrypt(self.KEY, self.NONCE, bytes(ciphertext), b"aad")

    def test_the_wrong_aad_is_rejected(self):
        ciphertext = chacha20_poly1305_encrypt(self.KEY, self.NONCE, b"secret", b"aad")
        with self.assertRaises(VaultError):
            chacha20_poly1305_decrypt(self.KEY, self.NONCE, ciphertext, b"other")

    def test_the_wrong_key_is_rejected(self):
        ciphertext = chacha20_poly1305_encrypt(self.KEY, self.NONCE, b"secret", b"aad")
        with self.assertRaises(VaultError):
            chacha20_poly1305_decrypt(bytes(32), self.NONCE, ciphertext, b"aad")


# ======================================================================================
# (f) ontology
# ======================================================================================


class OntologyLookupTests(unittest.TestCase):
    POSTAL = "person.address.postal_code"

    def test_a_messy_label_resolves_to_the_canonical_key(self):
        self.assertEqual(ontology.lookup("ZIP / Postal"), self.POSTAL)

    def test_the_many_spellings_of_a_postal_code_agree(self):
        for label in (
            "ZIP",
            "Zip Code",
            "zipcode",
            "ZIP/Postal",
            "ZIP / Postal",
            "Postal Code",
            "  postal   code  ",
        ):
            self.assertEqual(ontology.lookup(label), self.POSTAL, label)

    def test_the_resolved_key_is_a_declared_canonical_key(self):
        self.assertIn(self.POSTAL, ontology.CANONICAL_KEYS)
        self.assertEqual(ontology.get(self.POSTAL).key, self.POSTAL)

    def test_an_unknown_label_resolves_to_nothing(self):
        self.assertIsNone(ontology.lookup("wholly unrelated gibberish xyzzy"))

    def test_every_alias_target_and_pattern_hint_is_a_real_key(self):
        """The three ontology tables must not drift apart."""
        keys = set(ontology.CANONICAL_KEYS)
        self.assertEqual(sorted(set(ontology.ALIAS_INDEX.values()) - keys), [])
        hints = {r.canonical_hint for r in ontology.PATTERNS if r.canonical_hint}
        self.assertEqual(sorted(hints - keys), [])

    def test_a_placeholder_is_recognized_without_a_model(self):
        rule = ontology.match_placeholder("##-#######")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.canonical_hint, "company.tax_id.ein")


# ======================================================================================
# Encrypted documents: parser + crypt + vault.cipher in one path
# ======================================================================================


class EncryptedDocumentIntegrationTests(unittest.TestCase):
    """``Document.open`` transparently decrypts, using the vault's AES primitives."""

    def test_an_encrypted_document_opens_and_decodes_its_content(self):
        from tests.unit.test_pdfio_crypt import CONTENT, encrypted_pdf

        data, _ = encrypted_pdf(CONTENT, b"Title")
        doc = Document.open(data)

        self.assertTrue(doc.file.is_encrypted)
        self.assertTrue(doc.file.is_authenticated)
        self.assertEqual(doc.page_count, 1)
        self.assertEqual(doc.pages[0].content_bytes(), CONTENT)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
