"""Unit tests for :mod:`zfp.raster.render` and :mod:`zfp.raster.image`.

Every fixture here is synthesized in-process: hand-built image XObjects, a hand-encoded
CCITT Group 4 stream and a hand-encoded baseline JPEG.  Nothing touches the network, the
clock, or a checked-in binary.
"""

from __future__ import annotations

import math
import os
import unittest

from zfp.core.errors import UnsupportedFeatureError, ValidationError
from zfp.core.geometry import Rect
from zfp.pdfio.document import Document
from zfp.pdfio.filters import encode_flate
from zfp.pdfio.objects import PdfArray, PdfDict, PdfName, PdfStream, PdfString
from zfp.raster import image as image_module
from zfp.raster.image import ccitt_decode, decode_image_xobject, decode_jpeg_gray
from zfp.raster.render import (
    AGPL_ENV_VAR,
    BACKEND_EMBEDDED,
    BACKEND_PYMUPDF,
    RenderedPage,
    available_backends,
    embedded_page_images,
    parse_pgm,
    render_available,
    render_page,
)

# ======================================================================================
# Builders
# ======================================================================================


def make_page(gray, width, height, scale=1.0, backend="test"):
    return RenderedPage(
        page=0, width=width, height=height, scale=scale, gray=bytes(gray), backend=backend
    )


def image_stream(width, height, data, **entries):
    """Build an image XObject whose samples are Flate compressed."""
    d = PdfDict(
        {
            "Type": PdfName("XObject"),
            "Subtype": PdfName("Image"),
            "Width": width,
            "Height": height,
            "Filter": PdfName("FlateDecode"),
        }
    )
    d.update(entries)
    return PdfStream(d, encode_flate(bytes(data)))


def raw_image_stream(width, height, data, **entries):
    """Build an image XObject with a still-encoded payload (no Flate wrapper)."""
    d = PdfDict(
        {
            "Type": PdfName("XObject"),
            "Subtype": PdfName("Image"),
            "Width": width,
            "Height": height,
        }
    )
    d.update(entries)
    return PdfStream(d, bytes(data))


def document_with_image(stream, content, page_width=160.0, page_height=80.0, rotate=None):
    """A one-page document painting ``stream`` as ``/Im0`` with ``content``."""
    doc = Document.from_pages_blank(1, page_width, page_height)
    page = doc.page(0)
    image_ref = doc.writer.add_object(stream)
    content_ref = doc.writer.add_object(PdfStream(PdfDict({}), content))
    page.dict["Resources"] = PdfDict({"XObject": PdfDict({"Im0": image_ref})})
    page.dict["Contents"] = content_ref
    if rotate is not None:
        page.dict["Rotate"] = rotate
    page.touch()
    return doc


def bilevel_rows(rows):
    """Pack a list of 0/1 rows (1 = set bit) into MSB-first bytes."""
    out = bytearray()
    for row in rows:
        packed = bytearray((len(row) + 7) // 8)
        for x, value in enumerate(row):
            if value:
                packed[x >> 3] |= 0x80 >> (x & 7)
        out += packed
    return bytes(out)


# ======================================================================================
# RenderedPage
# ======================================================================================


class RenderedPageTests(unittest.TestCase):
    def test_size_must_match_the_buffer(self):
        with self.assertRaises(ValidationError):
            make_page(b"\x00" * 5, 3, 3)

    def test_pixel_and_row_clamp(self):
        page = make_page(bytes(range(12)), 4, 3)
        self.assertEqual(page.pixel(0, 0), 0)
        self.assertEqual(page.pixel(3, 2), 11)
        self.assertEqual(page.pixel(-5, -5), 0)
        self.assertEqual(page.pixel(99, 99), 11)
        self.assertEqual(page.row(1), bytes([4, 5, 6, 7]))
        self.assertEqual(page.row(50), bytes([8, 9, 10, 11]))

    def test_crop(self):
        page = make_page(bytes(range(12)), 4, 3)
        cropped = page.crop(Rect(1, 1, 3, 3))
        self.assertEqual((cropped.width, cropped.height), (2, 2))
        self.assertEqual(cropped.gray, bytes([5, 6, 9, 10]))
        self.assertEqual(cropped.scale, page.scale)
        self.assertEqual(cropped.backend, page.backend)

    def test_crop_outside_the_page_is_a_white_pixel(self):
        page = make_page(bytes(range(12)), 4, 3)
        cropped = page.crop(Rect(50, 50, 60, 60))
        self.assertEqual((cropped.width, cropped.height), (1, 1))
        self.assertEqual(cropped.gray, b"\xff")

    def test_pgm_round_trip(self):
        page = make_page(bytes(range(256)) * 2, 32, 16, scale=4.1666)
        data = page.to_pgm()
        self.assertTrue(data.startswith(b"P5\n32 16\n255\n"))
        back = RenderedPage.from_pgm(data, page=page.page, scale=page.scale)
        self.assertEqual(back.width, page.width)
        self.assertEqual(back.height, page.height)
        self.assertEqual(back.gray, page.gray)
        self.assertEqual(parse_pgm(data), (32, 16, page.gray))

    def test_pgm_header_with_comments(self):
        data = b"P5\n# made by a test\n2 2\n255\n\x00\x40\x80\xff"
        self.assertEqual(parse_pgm(data), (2, 2, b"\x00\x40\x80\xff"))

    def test_pgm_rejects_non_pgm(self):
        with self.assertRaises(ValidationError):
            parse_pgm(b"P6\n2 2\n255\n")
        with self.assertRaises(ValidationError):
            parse_pgm(b"P5\n2 2\n65535\n")


# ======================================================================================
# Backend discovery
# ======================================================================================


class BackendTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get(AGPL_ENV_VAR)
        os.environ.pop(AGPL_ENV_VAR, None)

    def tearDown(self):
        os.environ.pop(AGPL_ENV_VAR, None)
        if self._saved is not None:
            os.environ[AGPL_ENV_VAR] = self._saved

    def test_embedded_is_always_available_and_last(self):
        backends = available_backends()
        self.assertIn(BACKEND_EMBEDDED, backends)
        self.assertEqual(backends[-1], BACKEND_EMBEDDED)

    def test_agpl_backend_is_opt_in(self):
        from zfp.raster import render as render_module

        original = render_module._pymupdf_module
        render_module._pymupdf_module = lambda: object()
        try:
            self.assertNotIn(BACKEND_PYMUPDF, available_backends())
            os.environ[AGPL_ENV_VAR] = "1"
            self.assertIn(BACKEND_PYMUPDF, available_backends())
            os.environ[AGPL_ENV_VAR] = "no"
            self.assertNotIn(BACKEND_PYMUPDF, available_backends())
        finally:
            render_module._pymupdf_module = original

    def test_the_agpl_label_names_its_licence(self):
        self.assertIn("agpl", BACKEND_PYMUPDF)


# ======================================================================================
# Embedded rendering
# ======================================================================================


class EmbeddedRenderTests(unittest.TestCase):
    def _half_black_document(self, **kwargs):
        rows = [[0] * 8 + [1] * 8 for _ in range(8)]
        stream = image_stream(
            16, 8, bilevel_rows(rows),
            BitsPerComponent=1, ColorSpace=PdfName("DeviceGray"),
        )
        return document_with_image(stream, b"q 160 0 0 80 0 0 cm /Im0 Do Q", **kwargs)

    def test_full_page_image_renders(self):
        doc = self._half_black_document()
        page = render_page(doc, 0, dpi=72)
        self.assertEqual(page.backend, BACKEND_EMBEDDED)
        self.assertEqual((page.width, page.height), (160, 80))
        self.assertAlmostEqual(page.scale, 1.0)
        self.assertEqual(page.pixel(10, 10), 0)
        self.assertEqual(page.pixel(150, 10), 255)
        self.assertEqual(page.row(0).count(0), 80)

    def test_dpi_scales_the_raster(self):
        doc = self._half_black_document()
        page = render_page(doc, 0, dpi=144)
        self.assertEqual((page.width, page.height), (320, 160))
        self.assertAlmostEqual(page.scale, 2.0)
        self.assertEqual(page.pixel(20, 20), 0)
        self.assertEqual(page.pixel(300, 20), 255)

    def test_page_rotation_is_honoured(self):
        doc = self._half_black_document(rotate=90)
        page = render_page(doc, 0, dpi=72)
        self.assertEqual((page.width, page.height), (80, 160))
        # Rotating the page 90 degrees clockwise puts the black half on top.
        self.assertEqual(page.pixel(40, 10), 0)
        self.assertEqual(page.pixel(40, 150), 255)

    def test_partial_placement_from_the_cm_operator(self):
        rows = [[1] * 8 for _ in range(8)]
        stream = image_stream(
            8, 8, bilevel_rows(rows),
            BitsPerComponent=1, ColorSpace=PdfName("DeviceGray"),
        )
        # Paint the (white) image over the top-left quarter only.
        doc = document_with_image(stream, b"q 80 0 0 40 0 40 cm /Im0 Do Q")
        placements = embedded_page_images(doc, 0)
        self.assertEqual(len(placements), 1)
        rect = placements[0][0]
        self.assertAlmostEqual(rect.x0, 0.0)
        self.assertAlmostEqual(rect.y0, 40.0)
        self.assertAlmostEqual(rect.x1, 80.0)
        self.assertAlmostEqual(rect.y1, 80.0)

    def test_placement_falls_back_to_the_whole_page(self):
        rows = [[1] * 8 for _ in range(8)]
        stream = image_stream(
            8, 8, bilevel_rows(rows),
            BitsPerComponent=1, ColorSpace=PdfName("DeviceGray"),
        )
        # No Do in the content stream at all: the sole image still covers the page.
        doc = document_with_image(stream, b"q Q")
        placements = embedded_page_images(doc, 0)
        self.assertEqual(len(placements), 1)
        self.assertEqual(placements[0][0].as_list(), [0.0, 0.0, 160.0, 80.0])

    def test_embedded_page_images_reports_the_image_codec(self):
        stream = raw_image_stream(
            8, 8, b"\x00" * 8,
            BitsPerComponent=1,
            ColorSpace=PdfName("DeviceGray"),
            Filter=PdfName("CCITTFaxDecode"),
            DecodeParms=PdfDict({"Columns": 8, "K": -1}),
        )
        doc = document_with_image(stream, b"q 160 0 0 80 0 0 cm /Im0 Do Q")
        rect, data, codec = embedded_page_images(doc, 0)[0]
        self.assertEqual(codec, "CCITTFaxDecode")
        self.assertEqual(data, b"\x00" * 8)
        self.assertEqual(rect.as_list(), [0.0, 0.0, 160.0, 80.0])

    def test_flate_image_reports_no_residual_codec(self):
        doc = self._half_black_document()
        _rect, data, codec = embedded_page_images(doc, 0)[0]
        self.assertEqual(codec, "")
        self.assertEqual(len(data), 16)  # 16x8 at one bit per pixel

    def test_downsampling_averages_instead_of_dropping_ink(self):
        # A one-pixel-wide black line every four pixels; at quarter scale a nearest
        # neighbour sampler would either lose it or double it, an average keeps it gray.
        rows = []
        for _ in range(32):
            rows.append([0 if (x % 4) == 0 else 1 for x in range(32)])
        stream = image_stream(
            32, 32, bilevel_rows(rows),
            BitsPerComponent=1, ColorSpace=PdfName("DeviceGray"),
        )
        doc = document_with_image(stream, b"q 8 0 0 8 0 0 cm /Im0 Do Q", 8.0, 8.0)
        page = render_page(doc, 0, dpi=72)
        self.assertEqual((page.width, page.height), (8, 8))
        self.assertTrue(all(0 < value < 255 for value in page.gray))

    def test_no_image_and_no_renderer_raises(self):
        doc = Document.from_pages_blank(1)
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            render_page(doc, 0, backend=BACKEND_EMBEDDED)
        self.assertIn("zerofusspdf[render]", str(ctx.exception))

    def test_unknown_backend_is_refused(self):
        doc = Document.from_pages_blank(1)
        with self.assertRaises(UnsupportedFeatureError):
            render_page(doc, 0, backend="imagination")

    def test_index_and_dpi_validation(self):
        doc = Document.from_pages_blank(1)
        with self.assertRaises(ValidationError):
            render_page(doc, 5)
        with self.assertRaises(ValidationError):
            render_page(doc, 0, dpi=0)

    def test_render_available(self):
        doc = self._half_black_document()
        self.assertTrue(render_available(doc, 0))
        self.assertFalse(render_available(doc, 3))
        blank = Document.from_pages_blank(1)
        self.assertEqual(render_available(blank, 0), len(available_backends()) > 1)

    def test_image_mask_paints_only_its_ink(self):
        rows = [[1] * 4 + [0] * 4 for _ in range(4)]
        stream = image_stream(8, 4, bilevel_rows(rows), ImageMask=True)
        doc = document_with_image(stream, b"q 8 0 0 4 0 0 cm /Im0 Do Q", 8.0, 4.0)
        page = render_page(doc, 0, dpi=72)
        # Sample 0 paints (black); sample 1 leaves the paper alone.
        self.assertEqual(page.pixel(0, 0), 255)
        self.assertEqual(page.pixel(6, 0), 0)


# ======================================================================================
# Raw sample decoding
# ======================================================================================


class RawImageDecodeTests(unittest.TestCase):
    def test_one_bit_flate_gray(self):
        rows = [[0, 0, 1, 1, 0, 1, 0, 1], [1, 1, 1, 1, 0, 0, 0, 0]]
        stream = image_stream(
            8, 2, bilevel_rows(rows),
            BitsPerComponent=1, ColorSpace=PdfName("DeviceGray"),
        )
        decoded = decode_image_xobject(stream)
        self.assertTrue(decoded.supported)
        self.assertEqual(decoded.kind, "gray")
        self.assertEqual((decoded.width, decoded.height), (8, 2))
        self.assertEqual(
            list(decoded.gray),
            [0, 0, 255, 255, 0, 255, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0],
        )

    def test_decode_array_inverts_a_gray_image(self):
        rows = [[0, 1, 0, 1, 0, 1, 0, 1]]
        stream = image_stream(
            8, 1, bilevel_rows(rows),
            BitsPerComponent=1,
            ColorSpace=PdfName("DeviceGray"),
            Decode=PdfArray([1, 0]),
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(list(decoded.gray), [255, 0, 255, 0, 255, 0, 255, 0])

    def test_eight_bit_gray_is_passed_through(self):
        data = bytes([0, 32, 64, 128, 200, 255])
        stream = image_stream(
            3, 2, data, BitsPerComponent=8, ColorSpace=PdfName("DeviceGray")
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(decoded.gray, data)

    def test_four_bit_gray_scales_to_full_range(self):
        # Two pixels per byte: 0x0F -> 0 then 15.
        stream = image_stream(
            2, 1, bytes([0x0F]), BitsPerComponent=4, ColorSpace=PdfName("DeviceGray")
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(list(decoded.gray), [0, 255])

    def test_rgb_uses_luminance(self):
        data = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255])
        stream = image_stream(
            4, 1, data, BitsPerComponent=8, ColorSpace=PdfName("DeviceRGB")
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(decoded.kind, "rgb")
        expected = [
            round(0.299 * 255), round(0.587 * 255), round(0.114 * 255), 255,
        ]
        for got, want in zip(decoded.gray, expected):
            self.assertLessEqual(abs(got - want), 1)

    def test_cmyk_uses_the_ink_model(self):
        # pure cyan, pure black, no ink
        data = bytes([255, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 0])
        stream = image_stream(
            3, 1, data, BitsPerComponent=8, ColorSpace=PdfName("DeviceCMYK")
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(decoded.kind, "cmyk")
        self.assertEqual(decoded.gray[1], 0)
        self.assertEqual(decoded.gray[2], 255)
        self.assertLess(decoded.gray[0], 200)

    def test_indexed_palette(self):
        palette = PdfString(bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]), False)
        colorspace = PdfArray([PdfName("Indexed"), PdfName("DeviceRGB"), 3, palette])
        stream = image_stream(
            4, 1, bytes([0, 1, 2, 3]), BitsPerComponent=8, ColorSpace=colorspace
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(decoded.kind, "indexed")
        self.assertEqual(decoded.gray[3], 255)
        self.assertLess(decoded.gray[0], 100)     # red
        self.assertGreater(decoded.gray[1], 130)  # green is the bright primary
        self.assertLess(decoded.gray[2], 60)      # blue

    def test_indexed_two_bit_samples(self):
        palette = PdfString(bytes([0, 85, 170, 255]), False)
        colorspace = PdfArray([PdfName("Indexed"), PdfName("DeviceGray"), 3, palette])
        # 0b00_01_10_11 -> indices 0,1,2,3
        stream = image_stream(
            4, 1, bytes([0b00011011]), BitsPerComponent=2, ColorSpace=colorspace
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(list(decoded.gray), [0, 85, 170, 255])

    def test_image_mask_decode_flips_the_painted_bit(self):
        rows = [[0, 1, 0, 1, 1, 1, 1, 1]]
        plain = image_stream(8, 1, bilevel_rows(rows), ImageMask=True)
        self.assertEqual(list(decode_image_xobject(plain).gray)[:2], [0, 255])
        flipped = image_stream(
            8, 1, bilevel_rows(rows), ImageMask=True, Decode=PdfArray([1, 0])
        )
        self.assertEqual(list(decode_image_xobject(flipped).gray)[:2], [255, 0])

    def test_truncated_stream_degrades_to_white(self):
        stream = image_stream(
            4, 4, b"\x00\x00", BitsPerComponent=8, ColorSpace=PdfName("DeviceGray")
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(len(decoded.gray), 16)
        self.assertEqual(decoded.gray[-1], 255)

    def test_other_filter_chains_reach_the_same_samples(self):
        from zfp.pdfio.filters import encode_ascii85, encode_lzw, encode_runlength

        samples = bytes([0, 40, 80, 120, 160, 200, 240, 255])
        cases = (
            (PdfName("RunLengthDecode"), encode_runlength(samples)),
            (PdfName("LZWDecode"), encode_lzw(samples)),
            (
                PdfArray([PdfName("ASCII85Decode"), PdfName("FlateDecode")]),
                encode_ascii85(encode_flate(samples)),
            ),
        )
        for filters, payload in cases:
            stream = raw_image_stream(
                8, 1, payload,
                BitsPerComponent=8,
                ColorSpace=PdfName("DeviceGray"),
                Filter=filters,
            )
            self.assertEqual(decode_image_xobject(stream).gray, samples)

    def test_unsupported_codecs_report_themselves(self):
        for codec in ("JPXDecode", "JBIG2Decode"):
            stream = raw_image_stream(
                4, 4, b"\x00" * 8,
                BitsPerComponent=8,
                ColorSpace=PdfName("DeviceGray"),
                Filter=PdfName(codec),
            )
            decoded = decode_image_xobject(stream)
            self.assertFalse(decoded.supported)
            self.assertEqual(decoded.kind, "unsupported")
            self.assertEqual(len(decoded.gray), 16)
            self.assertIn(codec, decoded.detail)

    def test_zero_sized_and_absurd_images(self):
        self.assertEqual(decode_image_xobject(None).kind, "empty")
        tiny = raw_image_stream(0, 5, b"")
        self.assertEqual(decode_image_xobject(tiny).kind, "empty")
        huge = raw_image_stream(200000, 200000, b"")
        self.assertFalse(decode_image_xobject(huge).supported)


# ======================================================================================
# CCITT
# ======================================================================================

_MODE_BITS = {0: "1", 1: "011", 2: "000011", 3: "0000011",
              -1: "010", -2: "000010", -3: "0000010"}


class CcittHelper:
    """A minimal Group 3/4 encoder, used to exercise the decoder."""

    def __init__(self):
        self.white = dict((run, bits) for bits, run in image_module._WHITE_CODES)
        self.black = dict((run, bits) for bits, run in image_module._BLACK_CODES)
        self.extended = dict((run, bits) for bits, run in image_module._EXT_CODES)

    def run(self, length, white):
        table = self.white if white else self.black
        out = ""
        while length >= 64:
            makeup = min(2560, (length // 64) * 64)
            while makeup > 64 and makeup not in table and makeup not in self.extended:
                makeup -= 64
            out += table.get(makeup) or self.extended[makeup]
            length -= makeup
        return out + table[length]

    @staticmethod
    def changes(row):
        out = []
        previous = 0
        for x, value in enumerate(row):
            if value != previous:
                out.append(x)
                previous = value
        return out

    def encode_g4(self, rows, columns):
        bits = []
        reference = []
        for row in rows:
            bits.append(self._encode_g4_row(row, reference, columns))
            reference = self.changes(row)
        return self.pack("".join(bits))

    def _encode_g4_row(self, row, reference, columns):
        bits = []
        current = self.changes(row)
        a0 = -1
        colour = 0
        while a0 < columns:
            a1 = columns
            for index, position in enumerate(current):
                if position > a0 and (index & 1) == colour:
                    a1 = position
                    break
            a2 = columns
            for position in current:
                if position > a1:
                    a2 = position
                    break
            n = len(reference)
            i = 0
            while i < n and reference[i] <= a0:
                i += 1
            if (i & 1) != colour:
                i += 1
            b1 = reference[i] if i < n else columns
            b2 = reference[i + 1] if i + 1 < n else columns
            if b2 < a1:
                bits.append("0001")
                a0 = b2
            elif abs(a1 - b1) <= 3:
                bits.append(_MODE_BITS[a1 - b1])
                a0 = a1
                colour ^= 1
            else:
                start = 0 if a0 < 0 else a0
                bits.append("001")
                bits.append(self.run(a1 - start, colour == 0))
                bits.append(self.run(a2 - a1, colour != 0))
                a0 = a2
        return "".join(bits)

    def encode_g4_byte_aligned(self, rows, columns):
        """Group 4 with every row starting on a byte boundary (EncodedByteAlign)."""
        chunks = []
        reference = []
        for row in rows:
            bits = self._encode_g4_row(row, reference, columns)
            reference = self.changes(row)
            chunks.append(bits + "0" * ((8 - len(bits) % 8) % 8))
        return self.pack("".join(chunks))

    def encode_g3_mixed(self, rows, columns):
        """Group 3 K>0: an EOL and a 1D/2D flag bit in front of every one-dimensional row."""
        bits = []
        for row in rows:
            bits.append("000000000001")
            bits.append("1")
            bits.append(self._encode_1d_row(row, columns))
        return self.pack("".join(bits))

    def encode_g3_1d(self, rows, columns):
        return self.pack("".join(self._encode_1d_row(row, columns) for row in rows))

    def _encode_1d_row(self, row, columns):
        bits = []
        position = 0
        colour = 0
        while position < columns:
            length = 0
            while position + length < columns and row[position + length] == colour:
                length += 1
            bits.append(self.run(length, colour == 0))
            position += length
            colour ^= 1
        return "".join(bits)

    @staticmethod
    def pack(bits):
        bits += "0" * ((8 - len(bits) % 8) % 8)
        return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))

    @staticmethod
    def unpack(packed, columns, rows):
        stride = (columns + 7) // 8
        out = []
        for y in range(rows):
            chunk = packed[y * stride : (y + 1) * stride]
            out.append(
                [0 if (chunk[x >> 3] >> (7 - (x & 7))) & 1 else 1 for x in range(columns)]
            )
        return out


def sample_rows(columns, height, seed=7):
    """Deterministic pseudo-random horizontal bars, seeded, never wall-clock."""
    import random

    rng = random.Random(seed)
    rows = []
    for _ in range(height):
        row = [0] * columns
        for _ in range(rng.randint(0, 5)):
            start = rng.randrange(columns)
            stop = min(columns, start + rng.randint(1, 40))
            for x in range(start, stop):
                row[x] = 1
        rows.append(row)
    return rows


class CcittTests(unittest.TestCase):
    def test_code_tables_are_prefix_free(self):
        for name, table in (
            ("white", image_module._WHITE_CODES + image_module._EXT_CODES),
            ("black", image_module._BLACK_CODES + image_module._EXT_CODES),
            ("mode", image_module._MODE_CODES),
        ):
            codes = [bits for bits, _value in table]
            self.assertEqual(len(codes), len(set(codes)), "%s has duplicate codes" % name)
            ordered = sorted(codes, key=len)
            for i, short in enumerate(ordered):
                for long in ordered[i + 1 :]:
                    self.assertFalse(
                        long.startswith(short) and long != short,
                        "%s: %r is a prefix of %r" % (name, short, long),
                    )

    def test_run_lengths_are_complete(self):
        whites = set(run for _bits, run in image_module._WHITE_CODES)
        blacks = set(run for _bits, run in image_module._BLACK_CODES)
        for run in range(64):
            self.assertIn(run, whites)
            self.assertIn(run, blacks)
        for run in range(64, 1729, 64):
            self.assertIn(run, whites)
            self.assertIn(run, blacks)

    def test_hand_encoded_group4_rows(self):
        # V0 alone: the coding line matches the imaginary white reference line.
        self.assertEqual(ccitt_decode(b"\x80", columns=8, rows=1, k=-1), (b"\xff", 1))
        # Horizontal mode, white run 0 then black run 8: 001 00110101 000101
        self.assertEqual(
            ccitt_decode(b"\x26\xa2\x80", columns=8, rows=1, k=-1), (b"\x00", 1)
        )

    def test_hand_encoded_group3_rows(self):
        self.assertEqual(ccitt_decode(b"\x98", columns=8, rows=1, k=0), (b"\xff", 1))
        helper = CcittHelper()
        black = helper.pack(helper.run(0, True) + helper.run(8, False))
        self.assertEqual(ccitt_decode(black, columns=8, rows=1, k=0), (b"\x00", 1))

    def test_black_is_1_flips_the_output_convention(self):
        self.assertEqual(
            ccitt_decode(b"\x80", columns=8, rows=1, k=-1, black_is_1=True), (b"\x00", 1)
        )

    def test_group4_round_trip(self):
        columns, height = 64, 24
        rows = sample_rows(columns, height)
        helper = CcittHelper()
        encoded = helper.encode_g4(rows, columns)
        packed, produced = ccitt_decode(encoded, columns=columns, rows=height, k=-1)
        self.assertEqual(produced, height)
        self.assertEqual(helper.unpack(packed, columns, produced), rows)

    def test_group3_1d_round_trip(self):
        columns, height = 96, 12
        rows = sample_rows(columns, height, seed=11)
        helper = CcittHelper()
        encoded = helper.encode_g3_1d(rows, columns)
        packed, produced = ccitt_decode(encoded, columns=columns, rows=height, k=0)
        self.assertEqual(produced, height)
        self.assertEqual(helper.unpack(packed, columns, produced), rows)

    def test_wide_runs_use_make_up_codes(self):
        columns = 1728
        rows = [[0] * columns, [1] * columns, [0] * 900 + [1] * 828]
        helper = CcittHelper()
        encoded = helper.encode_g4(rows, columns)
        packed, produced = ccitt_decode(encoded, columns=columns, rows=3, k=-1)
        self.assertEqual(produced, 3)
        self.assertEqual(helper.unpack(packed, columns, produced), rows)

    def test_encoded_byte_align(self):
        columns, height = 48, 10
        rows = sample_rows(columns, height, seed=5)
        helper = CcittHelper()
        encoded = helper.encode_g4_byte_aligned(rows, columns)
        packed, produced = ccitt_decode(
            encoded, columns=columns, rows=height, k=-1, byte_align=True
        )
        self.assertEqual(produced, height)
        self.assertEqual(helper.unpack(packed, columns, produced), rows)

    def test_group3_mixed_k_uses_the_row_flag(self):
        columns, height = 64, 8
        rows = sample_rows(columns, height, seed=9)
        helper = CcittHelper()
        encoded = helper.encode_g3_mixed(rows, columns)
        packed, produced = ccitt_decode(encoded, columns=columns, rows=height, k=4)
        self.assertEqual(produced, height)
        self.assertEqual(helper.unpack(packed, columns, produced), rows)

    def test_malformed_input_terminates(self):
        for payload in (b"", b"\x00" * 64, b"\x55" * 200, bytes(range(256))):
            packed, produced = ccitt_decode(payload, columns=128, rows=0, k=-1)
            self.assertEqual(len(packed), produced * 16)
            self.assertLess(produced, 70000)

    def test_ccitt_image_xobject(self):
        columns, height = 32, 6
        rows = sample_rows(columns, height, seed=3)
        helper = CcittHelper()
        stream = raw_image_stream(
            columns, height, helper.encode_g4(rows, columns),
            BitsPerComponent=1,
            ColorSpace=PdfName("DeviceGray"),
            Filter=PdfName("CCITTFaxDecode"),
            DecodeParms=PdfDict({"K": -1, "Columns": columns, "Rows": height}),
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(decoded.kind, "ccitt")
        self.assertTrue(decoded.supported)
        for y in range(height):
            expected = [0 if value else 255 for value in rows[y]]
            self.assertEqual(list(decoded.row(y)), expected)

    def test_columns_wider_than_the_image_are_cropped(self):
        columns, width, height = 16, 12, 4
        rows = [[1 if x < 6 else 0 for x in range(columns)] for _ in range(height)]
        helper = CcittHelper()
        stream = raw_image_stream(
            width, height, helper.encode_g4(rows, columns),
            BitsPerComponent=1,
            ColorSpace=PdfName("DeviceGray"),
            Filter=PdfName("CCITTFaxDecode"),
            DecodeParms=PdfDict({"K": -1, "Columns": columns, "Rows": height}),
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual((decoded.width, decoded.height), (width, height))
        self.assertEqual(list(decoded.row(0)), [0] * 6 + [255] * 6)

    def test_ccitt_rows_shortfall_is_padded_white(self):
        stream = raw_image_stream(
            8, 4, b"\x80",  # one all-white row of data for a four-row image
            BitsPerComponent=1,
            ColorSpace=PdfName("DeviceGray"),
            Filter=PdfName("CCITTFaxDecode"),
            DecodeParms=PdfDict({"K": -1, "Columns": 8}),
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(len(decoded.gray), 32)
        self.assertEqual(decoded.gray.count(255), 32)


# ======================================================================================
# JPEG
# ======================================================================================

_DC_COUNTS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
_DC_SYMBOLS = list(range(12))
_AC_COUNTS = [0] * 7 + [255] + [0] * 8
_AC_SYMBOLS = list(range(255))


def canonical_codes(counts, symbols):
    codes = {}
    code = 0
    index = 0
    for length in range(1, 17):
        for _ in range(counts[length - 1]):
            codes[symbols[index]] = (code, length)
            index += 1
            code += 1
        code <<= 1
    return codes


class _BitWriter:
    def __init__(self):
        self.bits = []

    def put(self, value, count):
        for shift in range(count - 1, -1, -1):
            self.bits.append((value >> shift) & 1)

    def bytes(self):
        padded = self.bits + [1] * ((8 - len(self.bits) % 8) % 8)
        out = bytearray()
        for i in range(0, len(padded), 8):
            value = 0
            for bit in padded[i : i + 8]:
                value = (value << 1) | bit
            out.append(value)
            if value == 0xFF:
                out.append(0x00)  # byte stuffing
        return bytes(out)


def _category(value):
    magnitude = abs(value)
    size = 0
    while magnitude:
        size += 1
        magnitude >>= 1
    return size


def _segment(marker, payload):
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def encode_baseline_jpeg(blocks, width, height):
    """Encode 8x8 coefficient blocks (natural order) as a grayscale baseline JPEG."""
    zigzag = image_module._ZIGZAG
    dc_codes = canonical_codes(_DC_COUNTS, _DC_SYMBOLS)
    ac_codes = canonical_codes(_AC_COUNTS, _AC_SYMBOLS)
    writer = _BitWriter()
    predictor = 0
    for block in blocks:
        ordered = [block[zigzag[i]] for i in range(64)]
        diff = ordered[0] - predictor
        predictor = ordered[0]
        size = _category(diff)
        code, length = dc_codes[size]
        writer.put(code, length)
        if size:
            writer.put(diff if diff > 0 else diff + (1 << size) - 1, size)
        run = 0
        k = 1
        while k < 64:
            value = ordered[k]
            if value == 0:
                run += 1
                k += 1
                continue
            while run > 15:
                code, length = ac_codes[0xF0]
                writer.put(code, length)
                run -= 16
            size = _category(value)
            code, length = ac_codes[(run << 4) | size]
            writer.put(code, length)
            writer.put(value if value > 0 else value + (1 << size) - 1, size)
            run = 0
            k += 1
        if run or all(v == 0 for v in ordered[1:]):
            code, length = ac_codes[0x00]
            writer.put(code, length)
    out = bytearray(b"\xff\xd8")
    out += _segment(0xDB, bytes([0x00]) + bytes([1] * 64))
    out += _segment(
        0xC0,
        bytes([8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big")
        + bytes([1, 1, 0x11, 0]),
    )
    out += _segment(0xC4, bytes([0x00]) + bytes(_DC_COUNTS) + bytes(_DC_SYMBOLS))
    out += _segment(0xC4, bytes([0x10]) + bytes(_AC_COUNTS) + bytes(_AC_SYMBOLS))
    out += _segment(0xDA, bytes([1, 1, 0x00, 0x00, 0x3F, 0x00]))
    out += writer.bytes()
    out += b"\xff\xd9"
    return bytes(out)


def encode_jpeg_with_restarts(blocks, width, height, interval=1):
    """Same encoder, but with a restart marker (and a DC predictor reset) per interval."""
    zigzag = image_module._ZIGZAG
    dc_codes = canonical_codes(_DC_COUNTS, _DC_SYMBOLS)
    ac_codes = canonical_codes(_AC_COUNTS, _AC_SYMBOLS)
    entropy = bytearray()
    restart = 0
    for index, block in enumerate(blocks):
        if index and index % interval == 0:
            entropy += bytes([0xFF, 0xD0 + (restart & 7)])
            restart += 1
        writer = _BitWriter()
        ordered = [block[zigzag[i]] for i in range(64)]
        size = _category(ordered[0])
        code, length = dc_codes[size]
        writer.put(code, length)
        if size:
            value = ordered[0]
            writer.put(value if value > 0 else value + (1 << size) - 1, size)
        code, length = ac_codes[0x00]
        writer.put(code, length)
        entropy += writer.bytes()
    out = bytearray(b"\xff\xd8")
    out += _segment(0xDB, bytes([0x00]) + bytes([1] * 64))
    out += _segment(0xDD, interval.to_bytes(2, "big"))
    out += _segment(
        0xC0,
        bytes([8]) + height.to_bytes(2, "big") + width.to_bytes(2, "big")
        + bytes([1, 1, 0x11, 0]),
    )
    out += _segment(0xC4, bytes([0x00]) + bytes(_DC_COUNTS) + bytes(_DC_SYMBOLS))
    out += _segment(0xC4, bytes([0x10]) + bytes(_AC_COUNTS) + bytes(_AC_SYMBOLS))
    out += _segment(0xDA, bytes([1, 1, 0x00, 0x00, 0x3F, 0x00]))
    out += entropy
    out += b"\xff\xd9"
    return bytes(out)


def reference_idct(coefficients):
    """The textbook 2D inverse DCT, written out longhand as an independent check."""
    out = []
    for y in range(8):
        for x in range(8):
            total = 0.0
            for v in range(8):
                for u in range(8):
                    cu = 1.0 / math.sqrt(2.0) if u == 0 else 1.0
                    cv = 1.0 / math.sqrt(2.0) if v == 0 else 1.0
                    total += (
                        (cu * cv / 4.0)
                        * coefficients[v * 8 + u]
                        * math.cos((2 * x + 1) * u * math.pi / 16.0)
                        * math.cos((2 * y + 1) * v * math.pi / 16.0)
                    )
            out.append(max(0, min(255, int(total + 128.5))))
    return out


class JpegTests(unittest.TestCase):
    def test_canonical_codes_match_the_published_table(self):
        codes = canonical_codes(_DC_COUNTS, _DC_SYMBOLS)
        self.assertEqual(codes[0], (0b00, 2))
        self.assertEqual(codes[1], (0b010, 3))
        self.assertEqual(codes[5], (0b110, 3))
        self.assertEqual(codes[6], (0b1110, 4))
        self.assertEqual(codes[7], (0b11110, 5))

    def test_zigzag_is_a_permutation(self):
        self.assertEqual(sorted(image_module._ZIGZAG), list(range(64)))

    def test_flat_blocks_decode_to_their_dc_level(self):
        levels = [0, 64, 128, 255]
        blocks = []
        for level in levels:
            block = [0] * 64
            block[0] = (level - 128) * 8
            blocks.append(block)
        width, height, gray = decode_jpeg_gray(encode_baseline_jpeg(blocks, 16, 16))
        self.assertEqual((width, height), (16, 16))
        self.assertEqual(gray[0], levels[0])
        self.assertEqual(gray[8], levels[1])
        self.assertEqual(gray[16 * 8], levels[2])
        self.assertEqual(gray[16 * 8 + 8], levels[3])

    def test_ac_coefficients_match_a_reference_idct(self):
        block = [0] * 64
        block[0] = 40
        block[1] = -60
        block[8] = 25
        block[9] = 12
        block[63] = 7
        _w, _h, gray = decode_jpeg_gray(encode_baseline_jpeg([block], 8, 8))
        self.assertEqual(list(gray), reference_idct(block))

    def test_jpeg_image_xobject(self):
        block = [0] * 64
        block[0] = -400
        data = encode_baseline_jpeg([block], 8, 8)
        stream = raw_image_stream(
            8, 8, data,
            BitsPerComponent=8,
            ColorSpace=PdfName("DeviceGray"),
            Filter=PdfName("DCTDecode"),
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual(decoded.kind, "jpeg")
        self.assertTrue(decoded.supported)
        self.assertEqual(len(decoded.gray), 64)
        self.assertEqual(decoded.gray[0], 78)

    def test_restart_markers_reset_the_dc_predictor(self):
        levels = [32, 96, 160, 224]
        blocks = []
        for level in levels:
            block = [0] * 64
            block[0] = (level - 128) * 8  # absolute, because each block restarts
            blocks.append(block)
        width, height, gray = decode_jpeg_gray(
            encode_jpeg_with_restarts(blocks, 16, 16, interval=1)
        )
        self.assertEqual((width, height), (16, 16))
        self.assertEqual(gray[0], levels[0])
        self.assertEqual(gray[8], levels[1])
        self.assertEqual(gray[16 * 8], levels[2])
        self.assertEqual(gray[16 * 8 + 8], levels[3])

    def test_declared_size_wins_over_the_jpeg_header(self):
        block = [0] * 64
        block[0] = -800
        stream = raw_image_stream(
            16, 4, encode_baseline_jpeg([block], 8, 8),
            BitsPerComponent=8,
            ColorSpace=PdfName("DeviceGray"),
            Filter=PdfName("DCTDecode"),
        )
        decoded = decode_image_xobject(stream)
        self.assertEqual((decoded.width, decoded.height), (16, 4))
        self.assertEqual(len(decoded.gray), 64)
        self.assertEqual(set(decoded.gray), {28})

    def test_progressive_jpeg_degrades_to_mid_gray(self):
        block = [0] * 64
        block[0] = 0
        data = bytearray(encode_baseline_jpeg([block], 8, 8))
        data[data.index(b"\xff\xc0") + 1] = 0xC2  # pretend it is progressive
        stream = raw_image_stream(
            8, 8, bytes(data),
            BitsPerComponent=8,
            ColorSpace=PdfName("DeviceGray"),
            Filter=PdfName("DCTDecode"),
        )
        decoded = decode_image_xobject(stream)
        self.assertFalse(decoded.supported)
        self.assertEqual(decoded.kind, "unsupported")
        self.assertEqual(set(decoded.gray), {128})
        self.assertIn("progressive", decoded.detail)

    def test_garbage_is_not_fatal(self):
        stream = raw_image_stream(
            4, 4, b"not a jpeg at all",
            BitsPerComponent=8,
            ColorSpace=PdfName("DeviceGray"),
            Filter=PdfName("DCTDecode"),
        )
        decoded = decode_image_xobject(stream)
        self.assertFalse(decoded.supported)
        self.assertEqual(len(decoded.gray), 16)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
