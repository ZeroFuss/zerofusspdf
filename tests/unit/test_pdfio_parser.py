"""Unit tests for :mod:`zfp.pdfio.parser`.

Every fixture is a real PDF assembled byte by byte inside the test file, so the suite
stays offline, deterministic and free of binary blobs in git.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import zlib

from zfp.core.errors import PdfParseError
from zfp.pdfio.objects import (
    PdfDict,
    PdfName,
    PdfNull,
    PdfRef,
    PdfStream,
    PdfString,
)
from zfp.pdfio.parser import ObjectParser, PdfFile, XrefEntry

HEADER = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
CONTENT = b"BT /F1 12 Tf 72 720 Td (Hello) Tj ET"


# --------------------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------------------


def stream_object(body: bytes, extra: bytes = b"") -> bytes:
    """A well-formed stream object body with a correct ``/Length``."""
    return b"<< /Length %d%s >>\nstream\n%s\nendstream" % (len(body), extra, body)


def classic_pdf(
    objects,
    root: int = 1,
    extra_trailer: bytes = b"",
    free=(),
    startxref: int | None = None,
    prefix: bytes = b"",
):
    """Assemble a PDF with a classic cross-reference table.

    ``objects`` is a list of object bodies, numbered from 1.  ``free`` names object
    numbers whose xref entry is written as free even though the object exists (used to
    build hybrid-reference fixtures).
    """
    out = bytearray(prefix + HEADER)
    offsets = {}
    for number, body in enumerate(objects, start=1):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    table_offset = len(out)
    count = len(objects) + 1
    out += b"xref\n0 %d\n" % count
    out += b"0000000000 65535 f \n"
    for number in range(1, count):
        if number in free:
            out += b"0000000000 65535 f \n"
        else:
            out += b"%010d 00000 n \n" % offsets[number]
    out += b"trailer\n<< /Size %d /Root %d 0 R" % (count, root) + extra_trailer + b" >>\n"
    out += b"startxref\n%d\n%%%%EOF\n" % (
        table_offset if startxref is None else startxref
    )
    return bytes(out), offsets


ONE_PAGE_OBJECTS = [
    b"<< /Type /Catalog /Pages 2 0 R >>",
    b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
    stream_object(CONTENT),
]


def one_page_pdf(**kwargs):
    return classic_pdf(list(ONE_PAGE_OBJECTS), **kwargs)


def xref_stream_row(kind: int, field2: int, field3: int) -> bytes:
    """One ``/W [1 4 2]`` cross-reference stream row."""
    return bytes([kind]) + field2.to_bytes(4, "big") + field3.to_bytes(2, "big")


def png_up_encode(rows, width: int) -> bytes:
    """Encode fixed-width rows with the PNG ``Up`` predictor (tag 2)."""
    out = bytearray()
    previous = bytes(width)
    for row in rows:
        out += b"\x02" + bytes((row[i] - previous[i]) & 0xFF for i in range(width))
        previous = row
    return bytes(out)


def objstm_pdf(predictor: bool = False) -> bytes:
    """A PDF whose page tree lives in an object stream indexed by a cross-reference stream."""
    out = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}

    offsets[4] = len(out)
    out += b"4 0 obj\n" + stream_object(CONTENT) + b"\nendobj\n"

    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R >>",
    ]
    payload = bytearray()
    pairs = []
    for number, body in zip((1, 2, 3), bodies):
        pairs.append((number, len(payload)))
        payload += body + b" "
    index = b" ".join(b"%d %d" % pair for pair in pairs) + b"\n"
    compressed = zlib.compress(bytes(index) + bytes(payload))
    offsets[5] = len(out)
    out += b"5 0 obj\n<< /Type /ObjStm /N 3 /First %d /Length %d /Filter /FlateDecode >>\nstream\n" % (
        len(index),
        len(compressed),
    )
    out += compressed + b"\nendstream\nendobj\n"

    offsets[6] = len(out)
    rows = [
        xref_stream_row(0, 0, 65535),
        xref_stream_row(2, 5, 0),
        xref_stream_row(2, 5, 1),
        xref_stream_row(2, 5, 2),
        xref_stream_row(1, offsets[4], 0),
        xref_stream_row(1, offsets[5], 0),
        xref_stream_row(1, offsets[6], 0),
    ]
    parms = b""
    raw = b"".join(rows)
    if predictor:
        raw = png_up_encode(rows, 7)
        parms = b" /DecodeParms << /Predictor 12 /Columns 7 >>"
    body = zlib.compress(raw)
    out += b"6 0 obj\n<< /Type /XRef /Size 7 /W [1 4 2] /Root 1 0 R /Filter /FlateDecode%s /Length %d >>\nstream\n" % (
        parms,
        len(body),
    )
    out += body + b"\nendstream\nendobj\n"
    out += b"startxref\n%d\n%%%%EOF\n" % offsets[6]
    return bytes(out)


# --------------------------------------------------------------------------------------
# ObjectParser
# --------------------------------------------------------------------------------------


class ObjectParserTests(unittest.TestCase):
    def parse(self, data: bytes, resolver=None):
        return ObjectParser(data, 0, resolver=resolver).parse_object()

    def test_scalars(self):
        self.assertEqual(self.parse(b"42"), 42)
        self.assertAlmostEqual(self.parse(b"-1.5"), -1.5)
        self.assertIs(self.parse(b"true"), True)
        self.assertIs(self.parse(b"false"), False)
        self.assertEqual(self.parse(b"null"), PdfNull.NULL)
        self.assertEqual(self.parse(b"/Type"), PdfName("Type"))

    def test_strings(self):
        literal = self.parse(b"(hello \\(world\\))")
        self.assertIsInstance(literal, PdfString)
        self.assertEqual(literal.raw, b"hello (world)")
        self.assertFalse(literal.hexform)
        hexed = self.parse(b"<48656C6C6F>")
        self.assertEqual(hexed.raw, b"Hello")
        self.assertTrue(hexed.hexform)

    def test_reference_and_number_disambiguation(self):
        self.assertEqual(self.parse(b"12 0 R"), PdfRef(12, 0))
        array = self.parse(b"[1 2 3]")
        self.assertEqual(list(array), [1, 2, 3])
        mixed = self.parse(b"[1 0 R 7 8]")
        self.assertEqual(list(mixed), [PdfRef(1, 0), 7, 8])

    def test_nested_containers(self):
        obj = self.parse(b"<< /A [1 2 << /B (x) >>] /C /D /E 9 0 R >>")
        self.assertIsInstance(obj, PdfDict)
        self.assertEqual(obj.get_name("C"), "D")
        self.assertEqual(obj["E"], PdfRef(9, 0))
        inner = obj["A"][2]
        self.assertEqual(inner["B"].raw, b"x")

    def test_dictionary_survives_junk_between_entries(self):
        obj = self.parse(b"<< /A 1 ) /B 2 >>")
        self.assertEqual(obj["A"], 1)
        self.assertEqual(obj["B"], 2)

    def test_unterminated_array_stops_at_endobj(self):
        obj = self.parse(b"[1 2 3 endobj")
        self.assertEqual(list(obj), [1, 2, 3])

    def test_stream_with_correct_length(self):
        obj = self.parse(b"<< /Length 5 >>\nstream\nABCDE\nendstream\nendobj")
        self.assertIsInstance(obj, PdfStream)
        self.assertEqual(obj.raw, b"ABCDE")

    def test_stream_with_crlf_after_keyword(self):
        obj = self.parse(b"<< /Length 5 >>\r\nstream\r\nABCDE\r\nendstream")
        self.assertEqual(obj.raw, b"ABCDE")

    def test_stream_with_wrong_length_scans_for_endstream(self):
        obj = self.parse(b"<< /Length 2 >>\nstream\nABCDEFGH\nendstream\nendobj")
        self.assertEqual(obj.raw, b"ABCDEFGH")

    def test_stream_with_oversized_length_scans_for_endstream(self):
        obj = self.parse(b"<< /Length 900 >>\nstream\nABCDEFGH\nendstream\nendobj")
        self.assertEqual(obj.raw, b"ABCDEFGH")

    def test_stream_with_indirect_length_uses_resolver(self):
        class Resolver:
            def resolve(self, obj):
                return 5 if obj == PdfRef(9, 0) else obj

        obj = self.parse(
            b"<< /Length 9 0 R >>\nstream\nABCDE\nendstream\nendobj", Resolver()
        )
        self.assertEqual(obj.raw, b"ABCDE")

    def test_stream_with_indirect_length_and_no_resolver(self):
        obj = self.parse(b"<< /Length 9 0 R >>\nstream\nABCDE\nendstream\nendobj")
        self.assertEqual(obj.raw, b"ABCDE")

    def test_parse_indirect_object(self):
        parser = ObjectParser(b"7 0 obj\n<< /A 1 >>\nendobj\n")
        num, gen, obj = parser.parse_indirect_object()
        self.assertEqual((num, gen), (7, 0))
        self.assertEqual(obj["A"], 1)

    def test_parse_indirect_object_rejects_non_header(self):
        with self.assertRaises(PdfParseError):
            ObjectParser(b"<< /A 1 >>").parse_indirect_object()


# --------------------------------------------------------------------------------------
# Classic cross-reference tables
# --------------------------------------------------------------------------------------


class ClassicXrefTests(unittest.TestCase):
    def test_loads_one_page_document(self):
        data, _ = one_page_pdf()
        pdf = PdfFile.load(data)
        self.assertEqual(pdf.version, "1.7")
        self.assertFalse(pdf.rebuilt)
        self.assertEqual(pdf.object_numbers(), [1, 2, 3, 4])
        self.assertEqual(pdf.startxref, data.rfind(b"xref\n0 "))
        self.assertEqual(pdf.catalog.get_name("Type"), "Catalog")
        self.assertEqual(len(pdf.page_dicts()), 1)
        self.assertEqual(pdf.page_refs(), [PdfRef(3, 0)])
        self.assertEqual(pdf.warnings, [])

    def test_content_stream_round_trips(self):
        data, _ = one_page_pdf()
        pdf = PdfFile.load(data)
        page = pdf.page_dicts()[0]
        stream = pdf.resolve(page["Contents"])
        self.assertIsInstance(stream, PdfStream)
        self.assertEqual(stream.decoded(pdf), CONTENT)

    def test_free_entry_resolves_to_null(self):
        data, _ = one_page_pdf(free=(4,))
        pdf = PdfFile.load(data)
        self.assertEqual(pdf.object_numbers(), [1, 2, 3])
        self.assertEqual(pdf.get_object(4), PdfNull.NULL)

    def test_get_object_is_cached(self):
        data, _ = one_page_pdf()
        pdf = PdfFile.load(data)
        self.assertIs(pdf.get_object(3), pdf.get_object(3))

    def test_junk_before_header_shifts_offsets(self):
        junk = b"GARBAGE-FROM-A-BROKEN-DOWNLOAD\n" * 3
        data, _ = one_page_pdf(prefix=junk)
        pdf = PdfFile.load(data)
        self.assertEqual(pdf.header_offset, len(junk))
        self.assertFalse(pdf.rebuilt)
        self.assertEqual(len(pdf.page_dicts()), 1)

    def test_open_from_path(self):
        data, _ = one_page_pdf()
        handle, path = tempfile.mkstemp(suffix=".pdf")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(data)
            pdf = PdfFile.open(path)
            self.assertEqual(len(pdf.page_dicts()), 1)
        finally:
            os.unlink(path)

    def test_incremental_update_prefers_the_newer_revision(self):
        data, offsets = one_page_pdf()
        out = bytearray(data)
        new_offset = len(out)
        out += b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 300] >>\nendobj\n"
        table = len(out)
        out += b"xref\n0 1\n0000000000 65535 f \n3 1\n%010d 00000 n \n" % new_offset
        out += b"trailer\n<< /Size 5 /Root 1 0 R /Prev %d >>\nstartxref\n%d\n%%%%EOF\n" % (
            data.rfind(b"xref\n0 "),
            table,
        )
        pdf = PdfFile.load(bytes(out))
        self.assertFalse(pdf.rebuilt)
        page = pdf.page_dicts()[0]
        self.assertEqual(list(page["MediaBox"]), [0, 0, 200, 300])
        # The untouched objects still come from the original revision.
        self.assertEqual(pdf.catalog.get_name("Type"), "Catalog")


# --------------------------------------------------------------------------------------
# Cross-reference streams, object streams, hybrid files
# --------------------------------------------------------------------------------------


class XrefStreamTests(unittest.TestCase):
    def test_object_stream_document(self):
        pdf = PdfFile.load(objstm_pdf())
        self.assertFalse(pdf.rebuilt)
        self.assertEqual(pdf.object_numbers(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(pdf.catalog.get_name("Type"), "Catalog")
        self.assertEqual(pdf.page_refs(), [PdfRef(3, 0)])
        page = pdf.page_dicts()[0]
        self.assertEqual(list(page["MediaBox"]), [0, 0, 595, 842])
        self.assertEqual(pdf.resolve(page["Contents"]).decoded(pdf), CONTENT)

    def test_object_stream_document_with_png_predictor(self):
        pdf = PdfFile.load(objstm_pdf(predictor=True))
        self.assertFalse(pdf.rebuilt)
        self.assertEqual(len(pdf.page_dicts()), 1)
        self.assertEqual(pdf.xref[3].kind, 2)
        self.assertEqual(pdf.xref[3].stream_num, 5)

    def test_compressed_objects_are_cached(self):
        pdf = PdfFile.load(objstm_pdf())
        self.assertIs(pdf.get_object(2), pdf.get_object(2))

    def test_zero_width_type_field_defaults_to_type_one(self):
        # /W [0 3 0] means "no type field": every entry is an in-use offset.
        out = bytearray(b"%PDF-1.5\n")
        offsets = {}
        bodies = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 11 22] >>",
        ]
        for number, body in enumerate(bodies, start=1):
            offsets[number] = len(out)
            out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
        offsets[4] = len(out)
        rows = b"".join(
            offset.to_bytes(3, "big") for offset in (0, offsets[1], offsets[2], offsets[3], offsets[4])
        )
        payload = zlib.compress(rows)
        out += b"4 0 obj\n<< /Type /XRef /Size 5 /W [0 3 0] /Root 1 0 R /Filter /FlateDecode /Length %d >>\nstream\n" % len(payload)
        out += payload + b"\nendstream\nendobj\n"
        out += b"startxref\n%d\n%%%%EOF\n" % offsets[4]
        pdf = PdfFile.load(bytes(out))
        self.assertFalse(pdf.rebuilt)
        self.assertEqual(pdf.xref[2].kind, 1)
        self.assertEqual(list(pdf.page_dicts()[0]["MediaBox"]), [0, 0, 11, 22])

    def test_object_stream_index_mismatch_falls_back_to_a_search(self):
        data = objstm_pdf()
        pdf = PdfFile.load(data)
        # Claim object 3 sits at slot 0 of the object stream; it is really at slot 2.
        pdf.xref[3] = XrefEntry(3, 2, stream_num=5, stream_index=0)
        pdf._cache.clear()
        page = pdf.get_object(3)
        self.assertEqual(page.get_name("Type"), "Page")

    def test_object_stream_with_an_unknown_member_yields_null(self):
        pdf = PdfFile.load(objstm_pdf())
        pdf.xref[42] = XrefEntry(42, 2, stream_num=5, stream_index=0)
        self.assertEqual(pdf.get_object(42), PdfNull.NULL)

    def test_hybrid_reference_file(self):
        # The classic table hides object 3 by marking it free; the /XRefStm holds the
        # real entry.  A reader that understands cross-reference streams must find it.
        objects = list(ONE_PAGE_OBJECTS)
        objects.append(b"<< /Placeholder true >>")
        _, offsets = classic_pdf(objects, free=(3, 5), extra_trailer=b" /XRefStm 0000000000")
        payload = zlib.compress(xref_stream_row(1, offsets[3], 0))
        objects[4] = (
            b"<< /Type /XRef /Size 6 /W [1 4 2] /Index [3 1] /Root 1 0 R "
            b"/Filter /FlateDecode /Length %d >>\nstream\n" % len(payload)
        ) + payload + b"\nendstream"
        data, offsets = classic_pdf(
            objects, free=(3, 5), extra_trailer=b" /XRefStm 0000000000"
        )
        data = data.replace(b"/XRefStm 0000000000", b"/XRefStm %010d" % offsets[5], 1)
        pdf = PdfFile.load(data)
        self.assertFalse(pdf.rebuilt)
        self.assertEqual(pdf.xref[3].kind, 1)
        self.assertEqual(len(pdf.page_dicts()), 1)
        self.assertEqual(pdf.page_refs(), [PdfRef(3, 0)])


# --------------------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------------------


class RebuildTests(unittest.TestCase):
    def test_corrupt_startxref_triggers_rebuild(self):
        data, _ = one_page_pdf(startxref=999999)
        pdf = PdfFile.load(data)
        self.assertTrue(pdf.rebuilt)
        # The bogus offset is never advertised: an incremental writer would chain
        # /Prev onto it.
        self.assertEqual(pdf.startxref, -1)
        self.assertTrue(pdf.warnings)
        self.assertEqual(pdf.catalog.get_name("Type"), "Catalog")
        self.assertEqual(len(pdf.page_dicts()), 1)
        self.assertEqual(pdf.page_refs(), [PdfRef(3, 0)])
        self.assertEqual(pdf.resolve(pdf.page_dicts()[0]["Contents"]).decoded(pdf), CONTENT)

    def test_missing_startxref_triggers_rebuild(self):
        data, _ = one_page_pdf()
        data = data[: data.rfind(b"startxref")]
        pdf = PdfFile.load(data)
        self.assertTrue(pdf.rebuilt)
        self.assertEqual(len(pdf.page_dicts()), 1)

    def test_shifted_object_offsets_trigger_rebuild(self):
        # A long padded content stream puts every real object far away from the bogus
        # offsets, so the small-window recovery cannot save the table and the
        # brute-force scan has to run.
        padding = b"% padding line\n" * 400
        objects = list(ONE_PAGE_OBJECTS)
        objects[3] = stream_object(padding)
        data, offsets = classic_pdf(objects)
        decoy = offsets[4] + 2000
        table = data.rfind(b"xref\n0 ")
        head = data[:table]
        tail = data[table:]
        for number in sorted(offsets):
            tail = tail.replace(
                b"%010d 00000 n " % offsets[number], b"%010d 00000 n " % decoy, 1
            )
        pdf = PdfFile.load(head + tail)
        self.assertTrue(pdf.rebuilt)
        self.assertEqual(len(pdf.page_dicts()), 1)
        self.assertEqual(pdf.resolve(pdf.page_dicts()[0]["Contents"]).raw, padding)

    def test_rebuild_finds_catalog_without_a_trailer(self):
        data, _ = one_page_pdf()
        data = data[: data.rfind(b"trailer")]
        pdf = PdfFile.load(data)
        self.assertTrue(pdf.rebuilt)
        self.assertEqual(pdf.catalog.get_name("Type"), "Catalog")
        self.assertEqual(len(pdf.page_dicts()), 1)

    def test_rebuild_keeps_the_newest_object_definition(self):
        data, _ = one_page_pdf()
        out = bytearray(data)
        out += b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 20] >>\nendobj\n"
        out += b"startxref\n999999\n%%EOF\n"
        pdf = PdfFile.load(bytes(out))
        self.assertTrue(pdf.rebuilt)
        self.assertEqual(list(pdf.page_dicts()[0]["MediaBox"]), [0, 0, 10, 20])

    def test_explicit_rebuild_is_idempotent(self):
        data, _ = one_page_pdf()
        pdf = PdfFile.load(data)
        pdf.rebuild_xref()
        pdf.rebuild_xref()
        self.assertTrue(pdf.rebuilt)
        self.assertEqual(pdf.object_numbers(), [1, 2, 3, 4])
        self.assertEqual(len(pdf.page_dicts()), 1)

    def test_not_a_pdf_raises(self):
        with self.assertRaises(PdfParseError):
            PdfFile.load(b"this is not a pdf at all")
        with self.assertRaises(PdfParseError):
            PdfFile.load(b"")

    def test_headerless_file_with_objects_still_loads(self):
        data, _ = one_page_pdf()
        data = data[len(HEADER) :]
        pdf = PdfFile.load(data)
        self.assertEqual(len(pdf.page_dicts()), 1)


# --------------------------------------------------------------------------------------
# Graph traversal
# --------------------------------------------------------------------------------------


class TraversalTests(unittest.TestCase):
    def test_reference_cycle_resolves_to_null(self):
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] /X 5 0 R >>",
            b"<< /Empty true >>",
            b"6 0 R",
            b"5 0 R",
        ]
        data, _ = classic_pdf(objects)
        pdf = PdfFile.load(data)
        self.assertEqual(pdf.resolve(PdfRef(5, 0)), PdfNull.NULL)

    def test_page_tree_cycle_is_survivable(self):
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 2 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ]
        data, _ = classic_pdf(objects)
        pdf = PdfFile.load(data)
        self.assertEqual(len(pdf.page_dicts()), 1)

    def test_missing_count_is_tolerated(self):
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 4 0 R] >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 2 2] >>",
        ]
        data, _ = classic_pdf(objects)
        pdf = PdfFile.load(data)
        self.assertEqual(len(pdf.page_dicts()), 2)
        self.assertEqual(pdf.page_refs(), [PdfRef(3, 0), PdfRef(4, 0)])

    def test_nested_page_tree_is_depth_first(self):
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R 6 0 R] /Count 3 >>",
            b"<< /Type /Pages /Parent 2 0 R /Kids [4 0 R 5 0 R] /Count 2 >>",
            b"<< /Type /Page /Parent 3 0 R /MediaBox [0 0 1 1] >>",
            b"<< /Type /Page /Parent 3 0 R /MediaBox [0 0 2 2] >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 3 3] >>",
        ]
        data, _ = classic_pdf(objects)
        pdf = PdfFile.load(data)
        self.assertEqual(pdf.page_refs(), [PdfRef(4, 0), PdfRef(5, 0), PdfRef(6, 0)])
        widths = [page["MediaBox"][2] for page in pdf.page_dicts()]
        self.assertEqual(widths, [1, 2, 3])

    def test_inline_page_yields_null_reference(self):
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [<< /Type /Page /MediaBox [0 0 9 9] >>] /Count 1 >>",
        ]
        data, _ = classic_pdf(objects)
        pdf = PdfFile.load(data)
        self.assertEqual(len(pdf.page_dicts()), 1)
        self.assertEqual(pdf.page_refs(), [PdfNull.NULL])

    def test_dangling_root_falls_back_to_a_catalog_scan(self):
        data, _ = one_page_pdf(root=99)
        pdf = PdfFile.load(data)
        self.assertEqual(pdf.catalog.get_name("Type"), "Catalog")
        self.assertEqual(len(pdf.page_dicts()), 1)

    def test_broken_page_tree_falls_back_to_a_page_scan(self):
        objects = [
            b"<< /Type /Catalog /Pages 9 0 R >>",
            b"<< /Type /Pages /Kids [] /Count 0 >>",
            b"<< /Type /Page /MediaBox [0 0 4 4] >>",
        ]
        data, _ = classic_pdf(objects)
        pdf = PdfFile.load(data)
        self.assertEqual(pdf.page_refs(), [PdfRef(3, 0)])

    def test_xref_entry_reports_use(self):
        self.assertTrue(XrefEntry(1, 1, offset=10).in_use)
        self.assertTrue(XrefEntry(2, 2, stream_num=5).in_use)
        self.assertFalse(XrefEntry(3, 0).in_use)

    def test_repr_is_informative(self):
        data, _ = one_page_pdf()
        pdf = PdfFile.load(data)
        self.assertIn("PdfFile(", repr(pdf))
        self.assertIn("1.7", repr(pdf))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
