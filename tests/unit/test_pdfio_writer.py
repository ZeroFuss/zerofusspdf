"""Unit tests for :mod:`zfp.pdfio.writer`."""

from __future__ import annotations

import dataclasses
import re
import struct
import unittest

from zfp.core.errors import PdfWriteError
from zfp.pdfio.filters import encode_flate
from zfp.pdfio.objects import (
    PdfArray,
    PdfDict,
    PdfName,
    PdfNull,
    PdfRef,
    PdfStream,
    PdfString,
)
from zfp.pdfio.parser import PdfFile
from zfp.pdfio.writer import (
    BINARY_COMMENT,
    PdfWriter,
    build_document,
    format_number,
    serialize_object,
)


def minimal_objects(page_count: int = 1) -> dict:
    """Return a tiny but complete object table: catalog, page tree, pages."""
    objects = {}
    kids = PdfArray()
    for i in range(page_count):
        num = 3 + i
        objects[num] = PdfDict(
            {
                "Type": PdfName("Page"),
                "Parent": PdfRef(2),
                "MediaBox": PdfArray([0, 0, 612, 792]),
                "Resources": PdfDict(),
            }
        )
        kids.append(PdfRef(num))
    objects[2] = PdfDict(
        {"Type": PdfName("Pages"), "Kids": kids, "Count": page_count}
    )
    objects[1] = PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2)})
    return objects


def minimal_pdf(page_count: int = 1) -> bytes:
    return build_document(minimal_objects(page_count), PdfRef(1))


def compressed_pdf() -> bytes:
    """A PDF 1.5 file whose catalog/page tree live in an object stream.

    Built by hand because ``build_document`` only emits plain objects; this is the
    input that proves a full rewrite flattens compressed revisions.
    """
    inner = {
        1: PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2)}),
        2: PdfDict({"Type": PdfName("Pages"), "Kids": PdfArray([PdfRef(3)]), "Count": 1}),
        3: PdfDict(
            {
                "Type": PdfName("Page"),
                "Parent": PdfRef(2),
                "MediaBox": PdfArray([0, 0, 612, 792]),
            }
        ),
    }
    bodies, pairs, cursor = [], [], 0
    for num in sorted(inner):
        body = serialize_object(inner[num])
        pairs.append(b"%d %d" % (num, cursor))
        bodies.append(body)
        cursor += len(body) + 1
    header = b" ".join(pairs) + b"\n"
    objstm = PdfStream(
        PdfDict(
            {
                "Type": PdfName("ObjStm"),
                "N": len(inner),
                "First": len(header),
                "Filter": PdfName("FlateDecode"),
            }
        ),
        encode_flate(header + b" ".join(bodies)),
    )

    out = bytearray(b"%PDF-1.5\n" + BINARY_COMMENT + b"\n")
    objstm_offset = len(out)
    out += b"4 0 obj\n" + serialize_object(objstm) + b"\nendobj\n"
    xref_offset = len(out)

    rows = bytearray()
    rows += bytes([0]) + struct.pack(">I", 0) + struct.pack(">H", 65535)
    for index in range(len(inner)):
        rows += bytes([2]) + struct.pack(">I", 4) + struct.pack(">H", index)
    rows += bytes([1]) + struct.pack(">I", objstm_offset) + struct.pack(">H", 0)
    rows += bytes([1]) + struct.pack(">I", xref_offset) + struct.pack(">H", 0)
    xref_stream = PdfStream(
        PdfDict(
            {
                "Type": PdfName("XRef"),
                "Size": 6,
                "W": PdfArray([1, 4, 2]),
                "Root": PdfRef(1),
                "Filter": PdfName("FlateDecode"),
            }
        ),
        encode_flate(bytes(rows)),
    )
    out += b"5 0 obj\n" + serialize_object(xref_stream) + b"\nendobj\n"
    out += b"startxref\n%d\n" % xref_offset + b"%%EOF\n"
    return bytes(out)


class _BrokenFile:
    """A PdfFile stand-in whose object 4 always fails to resolve."""

    version = "1.7"

    def __init__(self, data: bytes, trailer: PdfDict) -> None:
        self.data = data
        self.trailer = trailer

    def object_numbers(self):
        return [1, 2, 3, 4]

    def get_object(self, num, gen=0):
        if num == 4:
            raise RuntimeError("object 4 is corrupt")
        return PdfDict({"N": num})


class FormatNumberTest(unittest.TestCase):
    def test_integers_are_plain(self):
        self.assertEqual(format_number(0), b"0")
        self.assertEqual(format_number(612), b"612")
        self.assertEqual(format_number(-42), b"-42")

    def test_floats_strip_trailing_zeros(self):
        self.assertEqual(format_number(1.5), b"1.5")
        self.assertEqual(format_number(10.0), b"10")
        self.assertEqual(format_number(100.0), b"100")
        self.assertEqual(format_number(0.125), b"0.125")

    def test_six_decimal_cap(self):
        self.assertEqual(format_number(1.0 / 3.0), b"0.333333")

    def test_negative_zero_is_normalized(self):
        self.assertEqual(format_number(-0.0), b"0")
        self.assertEqual(format_number(-1e-9), b"0")

    def test_never_scientific_notation(self):
        for value in (1e-7, 1e20, 2.5e-9, -3.75e18):
            self.assertNotIn(b"e", format_number(value).lower())

    def test_non_finite_degrades_to_zero(self):
        self.assertEqual(format_number(float("nan")), b"0")
        self.assertEqual(format_number(float("inf")), b"0")
        self.assertEqual(format_number(float("-inf")), b"0")


class SerializeTest(unittest.TestCase):
    def test_null_and_booleans(self):
        self.assertEqual(serialize_object(None), b"null")
        self.assertEqual(serialize_object(PdfNull.NULL), b"null")
        self.assertEqual(serialize_object(True), b"true")
        self.assertEqual(serialize_object(False), b"false")

    def test_bool_is_not_an_integer(self):
        # ``isinstance(True, int)`` is True; the serializer must not emit ``1``.
        self.assertEqual(serialize_object(PdfArray([True, 1])), b"[true 1]")

    def test_name_escaping(self):
        self.assertEqual(serialize_object(PdfName("Type")), b"/Type")
        self.assertEqual(serialize_object(PdfName("A B")), b"/A#20B")
        self.assertEqual(serialize_object(PdfName("a#b")), b"/a#23b")

    def test_strings_use_their_own_serializer(self):
        self.assertEqual(serialize_object(PdfString(b"hi")), b"(hi)")
        self.assertEqual(serialize_object(PdfString(b"\x01", hexform=True)), b"<01>")
        self.assertEqual(serialize_object(PdfString(b"a(b)")), b"(a\\(b\\))")

    def test_plain_str_becomes_a_string_not_a_name(self):
        self.assertEqual(serialize_object("/Helv 0 Tf 0 g"), b"(/Helv 0 Tf 0 g)")

    def test_reference(self):
        self.assertEqual(serialize_object(PdfRef(12, 3)), b"12 3 R")

    def test_array_and_dict(self):
        self.assertEqual(serialize_object(PdfArray([1, 2.5, PdfName("X")])), b"[1 2.5 /X]")
        self.assertEqual(serialize_object(PdfDict()), b"<<>>")
        self.assertEqual(
            serialize_object(PdfDict({"Type": PdfName("Page"), "N": 3})),
            b"<</Type /Page /N 3>>",
        )

    def test_nested_containers(self):
        obj = PdfDict({"K": PdfArray([PdfDict({"A": 1}), PdfRef(9)])})
        self.assertEqual(serialize_object(obj), b"<</K [<</A 1>> 9 0 R]>>")

    def test_stream_length_is_corrected(self):
        stream = PdfStream(PdfDict({"Length": 999, "Filter": PdfName("FlateDecode")}), b"abcd")
        out = serialize_object(stream)
        self.assertIn(b"/Length 4", out)
        self.assertNotIn(b"999", out)
        self.assertIn(b"\nstream\nabcd\nendstream", out)

    def test_unserializable_value_raises(self):
        with self.assertRaises(PdfWriteError):
            serialize_object(object())


class BuildDocumentTest(unittest.TestCase):
    def test_header_and_binary_comment(self):
        data = minimal_pdf()
        self.assertTrue(data.startswith(b"%PDF-1.7\n"))
        self.assertIn(BINARY_COMMENT, data[:32])

    def test_free_head_entry_and_trailer(self):
        data = minimal_pdf()
        self.assertIn(b"\n0000000000 65535 f \n", data)
        self.assertIn(b"/Size 4", data)
        self.assertIn(b"/Root 1 0 R", data)
        self.assertTrue(data.rstrip().endswith(b"%%EOF"))

    def test_startxref_points_at_the_xref_keyword(self):
        data = minimal_pdf()
        offset = int(re.search(rb"startxref\s+(\d+)", data).group(1))
        self.assertEqual(data[offset : offset + 4], b"xref")

    def test_deterministic(self):
        self.assertEqual(minimal_pdf(2), minimal_pdf(2))

    def test_requires_indirect_root(self):
        with self.assertRaises(PdfWriteError):
            build_document({1: PdfDict()}, PdfDict())  # type: ignore[arg-type]

    def test_gaps_produce_a_chained_free_list(self):
        objects = minimal_objects()
        objects[7] = PdfDict({"Type": PdfName("Whatever")})
        data = build_document(objects, PdfRef(1))
        pdf = PdfFile.load(data)
        self.assertEqual(pdf.resolve(PdfRef(7)).get_name("Type"), "Whatever")

    def test_parses_back(self):
        pdf = PdfFile.load(minimal_pdf(2))
        self.assertEqual(pdf.catalog.get_name("Type"), "Catalog")
        pages = pdf.resolve(pdf.catalog.get("Pages"))
        self.assertEqual(pages.get("Count"), 2)


class WriterBasicsTest(unittest.TestCase):
    def setUp(self):
        self.pdf = PdfFile.load(minimal_pdf())
        self.writer = PdfWriter(self.pdf)

    def test_next_number_is_past_the_highest_existing(self):
        # objects 1..3 exist, so the first allocation is 4.
        self.assertEqual(self.writer.allocate(), 4)
        self.assertEqual(self.writer.allocate(), 5)

    def test_add_object_returns_a_reference(self):
        ref = self.writer.add_object(PdfDict({"A": 1}))
        self.assertIsInstance(ref, PdfRef)
        self.assertEqual(self.writer.updates[ref.num].get("A"), 1)

    def test_set_object_advances_the_counter(self):
        self.writer.set_object(50, PdfDict())
        self.assertEqual(self.writer.allocate(), 51)

    def test_set_object_rejects_zero(self):
        with self.assertRaises(PdfWriteError):
            self.writer.set_object(0, PdfDict())

    def test_no_changes_returns_the_original_bytes(self):
        self.assertEqual(self.writer.write_incremental(), self.pdf.data)

    def test_update_trailer_forces_an_entry(self):
        self.writer.update_trailer("/Custom", 7)
        self.assertIn(b"/Custom 7", self.writer.write_incremental())


class IncrementalUpdateTest(unittest.TestCase):
    def setUp(self):
        self.original = minimal_pdf()
        self.pdf = PdfFile.load(self.original)
        self.writer = PdfWriter(self.pdf)

    def test_original_bytes_are_preserved_exactly(self):
        self.writer.add_object(PdfDict({"Hello": PdfName("World")}))
        out = self.writer.write_incremental()
        self.assertGreater(len(out), len(self.original))
        self.assertEqual(out[: len(self.original)], self.original)

    def test_appended_revision_shape(self):
        self.writer.add_object(PdfDict({"Hello": PdfName("World")}))
        out = self.writer.write_incremental()
        tail = out[len(self.original) :]
        self.assertIn(b"4 0 obj", tail)
        self.assertIn(b"endobj", tail)
        self.assertIn(b"xref", tail)
        self.assertIn(b"trailer", tail)
        self.assertTrue(tail.rstrip().endswith(b"%%EOF"))

    def test_prev_points_at_the_original_startxref(self):
        previous = int(re.search(rb"startxref\s+(\d+)", self.original).group(1))
        self.writer.add_object(PdfDict({"A": 1}))
        out = self.writer.write_incremental()
        tail = out[len(self.original) :]
        self.assertIn(b"/Prev %d" % previous, tail)

    def test_id_keeps_the_first_half_and_is_deterministic(self):
        first = self.pdf.trailer.get("ID")[0]
        self.writer.add_object(PdfDict({"A": 1}))
        out_a = self.writer.write_incremental()
        out_b = self.writer.write_incremental()
        self.assertEqual(out_a, out_b)

        reloaded = PdfFile.load(out_a)
        ids = reloaded.trailer.get("ID")
        self.assertEqual(len(ids), 2)
        self.assertEqual(ids[0], first)
        self.assertNotEqual(ids[0], ids[1])

    def test_one_xref_subsection_per_contiguous_run(self):
        self.writer.set_object(2, PdfDict({"Type": PdfName("Pages"), "Count": 0}))
        self.writer.set_object(3, PdfDict({"Type": PdfName("Page")}))
        self.writer.set_object(9, PdfDict({"Type": PdfName("Extra")}))
        tail = self.writer.write_incremental()[len(self.original) :]
        body = tail.split(b"\nxref\n", 1)[1].split(b"trailer", 1)[0]
        headers = re.findall(rb"^(\d+) (\d+)$", body, re.MULTILINE)
        self.assertEqual(headers, [(b"2", b"2"), (b"9", b"1")])

    def test_update_is_visible_after_reload(self):
        self.writer.set_object(1, PdfDict({"Type": PdfName("Catalog"), "Pages": PdfRef(2), "Marker": 42}))
        reloaded = PdfFile.load(self.writer.write_incremental())
        self.assertEqual(reloaded.catalog.get("Marker"), 42)

    def test_stacked_revisions_keep_every_prefix(self):
        self.writer.add_object(PdfDict({"Round": 1}))
        first = self.writer.write_incremental()
        second_writer = PdfWriter(PdfFile.load(first))
        second_writer.add_object(PdfDict({"Round": 2}))
        second = second_writer.write_incremental()
        self.assertEqual(second[: len(first)], first)
        self.assertEqual(second[: len(self.original)], self.original)
        reloaded = PdfFile.load(second)
        self.assertEqual(reloaded.resolve(PdfRef(4)).get("Round"), 1)
        self.assertEqual(reloaded.resolve(PdfRef(5)).get("Round"), 2)


class FullRewriteTest(unittest.TestCase):
    def setUp(self):
        self.original = minimal_pdf(2)
        self.pdf = PdfFile.load(self.original)
        self.writer = PdfWriter(self.pdf)

    def test_header_and_free_head_entry(self):
        out = self.writer.write_full()
        self.assertTrue(out.startswith(b"%PDF-"))
        self.assertIn(BINARY_COMMENT, out[:32])
        self.assertIn(b"\n0000000000 65535 f \n", out)
        self.assertNotIn(b"/Prev", out)

    def test_single_xref_subsection_starting_at_zero(self):
        out = self.writer.write_full()
        body = out.rsplit(b"\nxref\n", 1)[1].split(b"trailer", 1)[0]
        headers = re.findall(rb"^(\d+) (\d+)$", body, re.MULTILINE)
        self.assertEqual(len(headers), 1)
        self.assertEqual(headers[0][0], b"0")

    def test_object_numbers_are_preserved(self):
        reloaded = PdfFile.load(self.writer.write_full())
        self.assertEqual(reloaded.resolve(PdfRef(2)).get("Count"), 2)
        self.assertEqual(reloaded.resolve(PdfRef(3)).get_name("Type"), "Page")

    def test_updates_win_over_source_objects(self):
        self.writer.set_object(3, PdfDict({"Type": PdfName("Page"), "Marker": 5}))
        reloaded = PdfFile.load(self.writer.write_full())
        self.assertEqual(reloaded.resolve(PdfRef(3)).get("Marker"), 5)

    def test_unresolvable_objects_become_null(self):
        writer = PdfWriter(_BrokenFile(self.original, self.pdf.trailer))
        out = writer.write_full()
        self.assertIn(b"4 0 obj\nnull\nendobj", out)

    def test_deterministic(self):
        self.assertEqual(self.writer.write_full(), self.writer.write_full())


class RoundTripTest(unittest.TestCase):
    """write_full(load(write_incremental(...))) keeps every object value."""

    def test_incremental_then_full_preserves_values(self):
        original = minimal_pdf(2)
        writer = PdfWriter(PdfFile.load(original))
        payload = PdfDict(
            {
                "Kind": PdfName("Probe"),
                "Text": PdfString.from_text("Ada Lovelace"),
                "Numbers": PdfArray([1, -2, 3.5, 0.125]),
                "Flag": True,
                "Nothing": PdfNull.NULL,
                "Link": PdfRef(1),
            }
        )
        ref = writer.add_object(payload)
        writer.set_object(3, PdfDict({"Type": PdfName("Page"), "Parent": PdfRef(2), "Probe": ref}))
        incremental = writer.write_incremental()
        self.assertEqual(incremental[: len(original)], original)

        middle = PdfFile.load(incremental)
        full = PdfWriter(middle).write_full()
        final = PdfFile.load(full)

        probe = final.resolve(PdfRef(ref.num))
        self.assertEqual(probe.get_name("Kind"), "Probe")
        self.assertEqual(probe.get("Text").text(), "Ada Lovelace")
        self.assertEqual(list(probe.get("Numbers")), [1, -2, 3.5, 0.125])
        self.assertIs(probe.get("Flag"), True)
        self.assertEqual(probe.get("Link"), PdfRef(1))
        self.assertEqual(final.resolve(PdfRef(3)).get("Probe"), ref)
        self.assertEqual(final.resolve(PdfRef(2)).get("Count"), 2)

    def test_streams_survive_the_round_trip(self):
        objects = minimal_objects()
        objects[3]["Contents"] = PdfRef(4)
        objects[4] = PdfStream(PdfDict({"Length": 0}), b"BT /F1 12 Tf ET")
        original = build_document(objects, PdfRef(1))

        writer = PdfWriter(PdfFile.load(original))
        writer.set_object(4, PdfStream(PdfDict(), b"1 0 0 1 10 10 cm"))
        incremental = writer.write_incremental()
        self.assertEqual(incremental[: len(original)], original)

        final = PdfFile.load(PdfWriter(PdfFile.load(incremental)).write_full())
        stream = final.resolve(PdfRef(4))
        self.assertIsInstance(stream, PdfStream)
        self.assertEqual(stream.raw, b"1 0 0 1 10 10 cm")
        self.assertEqual(stream.dict.get("Length"), len(b"1 0 0 1 10 10 cm"))


class GenerationTest(unittest.TestCase):
    """Object generations survive both writers, so embedded refs keep matching."""

    @staticmethod
    def _with_generation(pdf, num: int, gen: int):
        """Restamp one xref entry's generation (``XrefEntry`` is frozen)."""
        pdf.xref[num] = dataclasses.replace(pdf.xref[num], gen=gen)
        return pdf

    def test_new_objects_use_generation_zero(self):
        writer = PdfWriter(PdfFile.load(minimal_pdf()))
        writer.add_object(PdfDict({"A": 1}))
        self.assertIn(b"4 0 obj", writer.write_incremental())

    def test_existing_generation_is_reused_on_update(self):
        pdf = self._with_generation(PdfFile.load(minimal_pdf()), 3, 4)
        writer = PdfWriter(pdf)
        writer.set_object(3, PdfDict({"Type": PdfName("Page")}))
        out = writer.write_incremental()
        tail = out[len(pdf.data) :]
        self.assertIn(b"3 4 obj", tail)
        self.assertIn(b"00004 n", tail)

    def test_full_rewrite_preserves_generations(self):
        pdf = self._with_generation(PdfFile.load(minimal_pdf()), 3, 4)
        out = PdfWriter(pdf).write_full()
        self.assertIn(b"3 4 obj", out)
        self.assertIn(b"00004 n", out)


class CompressedSourceTest(unittest.TestCase):
    """A full rewrite flattens object streams and drops the dead containers."""

    def setUp(self):
        self.pdf = PdfFile.load(compressed_pdf())

    def test_source_really_uses_an_object_stream(self):
        self.assertEqual(self.pdf.object_numbers(), [1, 2, 3, 4, 5])
        self.assertEqual(self.pdf.catalog.get_name("Type"), "Catalog")

    def test_rewrite_drops_objstm_and_xref_streams(self):
        out = PdfWriter(self.pdf).write_full()
        self.assertNotIn(b"/ObjStm", out)
        self.assertNotIn(b"/XRef", out)

    def test_flattened_objects_survive(self):
        reloaded = PdfFile.load(PdfWriter(self.pdf).write_full())
        self.assertEqual(reloaded.object_numbers(), [1, 2, 3])
        self.assertEqual(reloaded.catalog.get_name("Type"), "Catalog")
        self.assertEqual(len(reloaded.page_dicts()), 1)
        self.assertEqual(
            list(reloaded.resolve(PdfRef(3)).get("MediaBox")), [0, 0, 612, 792]
        )

    def test_incremental_update_over_a_compressed_source(self):
        original = self.pdf.data
        writer = PdfWriter(self.pdf)
        writer.add_object(PdfDict({"Added": True}))
        out = writer.write_incremental()
        self.assertEqual(out[: len(original)], original)
        reloaded = PdfFile.load(out)
        self.assertIs(reloaded.resolve(PdfRef(6)).get("Added"), True)
        self.assertEqual(reloaded.catalog.get_name("Type"), "Catalog")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
