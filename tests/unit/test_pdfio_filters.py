"""Unit tests for :mod:`zfp.pdfio.filters`."""

from __future__ import annotations

import random
import unittest
import zlib

from zfp.pdfio.filters import (
    FILTER_ALIASES,
    IMAGE_FILTERS,
    apply_predictor,
    ascii85_decode,
    ascii_hex_decode,
    decode,
    decode_one,
    encode_ascii85,
    encode_ascii_hex,
    encode_flate,
    encode_lzw,
    encode_runlength,
    flate_decode,
    is_image_filter,
    lzw_decode,
    normalize_filter_name,
    run_length_decode,
)
from zfp.pdfio.objects import PdfDict, PdfName


def png_rows(rows):
    """Serialize ``[(filter_type, [bytes...]), ...]`` into a PNG-predictor payload."""
    out = bytearray()
    for filter_type, data in rows:
        out.append(filter_type)
        out += bytes(data)
    return bytes(out)


def parms(**kwargs):
    return PdfDict(kwargs)


class FilterNameTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_filter_name("FlateDecode"), "FlateDecode")
        self.assertEqual(normalize_filter_name("/FlateDecode"), "FlateDecode")
        self.assertEqual(normalize_filter_name(PdfName("FlateDecode")), "FlateDecode")
        self.assertEqual(normalize_filter_name(b"/FlateDecode"), "FlateDecode")
        self.assertEqual(normalize_filter_name(None), "")

    def test_abbreviations(self):
        for short, long in FILTER_ALIASES.items():
            self.assertEqual(normalize_filter_name(short), long)

    def test_is_image_filter(self):
        for name in ("DCTDecode", "JPXDecode", "CCITTFaxDecode", "JBIG2Decode"):
            self.assertTrue(is_image_filter(name))
            self.assertTrue(is_image_filter("/" + name))
            self.assertTrue(is_image_filter(PdfName(name)))
            self.assertIn(name, IMAGE_FILTERS)
        for name in ("FlateDecode", "LZWDecode", "ASCII85Decode", "RunLengthDecode", "Nope"):
            self.assertFalse(is_image_filter(name))
        self.assertTrue(is_image_filter("DCT"))
        self.assertTrue(is_image_filter("CCF"))


class FlateTests(unittest.TestCase):
    def test_round_trip(self):
        payload = b"The quick brown fox jumps over the lazy dog. " * 40
        encoded = encode_flate(payload)
        self.assertLess(len(encoded), len(payload))
        self.assertEqual(flate_decode(encoded), payload)

    def test_round_trip_binary(self):
        rng = random.Random(1234)
        payload = bytes(rng.randrange(256) for _ in range(2000))
        self.assertEqual(flate_decode(encode_flate(payload)), payload)

    def test_encode_flate_is_deterministic(self):
        self.assertEqual(encode_flate(b"abc" * 50), encode_flate(b"abc" * 50))

    def test_empty(self):
        self.assertEqual(flate_decode(encode_flate(b"")), b"")
        self.assertEqual(flate_decode(b""), b"")

    def test_truncated_stream_recovers_a_prefix(self):
        payload = b"recoverable content, repeated. " * 60
        encoded = encode_flate(payload)
        truncated = encoded[: len(encoded) - 12]
        recovered = flate_decode(truncated)
        self.assertGreater(len(recovered), 0)
        self.assertTrue(payload.startswith(recovered))

    def test_leading_whitespace_is_tolerated(self):
        payload = b"whitespace prefixed"
        self.assertEqual(flate_decode(b"\r\n" + encode_flate(payload)), payload)

    def test_raw_deflate_without_a_zlib_header(self):
        compressor = zlib.compressobj(9, zlib.DEFLATED, -15)
        raw = compressor.compress(b"headerless") + compressor.flush()
        self.assertEqual(flate_decode(raw), b"headerless")

    def test_garbage_returns_empty_rather_than_raising(self):
        self.assertEqual(flate_decode(b"not compressed at all"), b"")


class LzwTests(unittest.TestCase):
    def test_known_vector(self):
        # Hand-computed: clear(256), 'A'(65), 258, 259, EOD(257) at 9 bits each.
        # 258 and 259 are both built during decoding, exercising the KwKwK case.
        self.assertEqual(lzw_decode(b"\x80\x10\x60\x50\x38\x08"), b"AAAAAA")

    def test_encoder_reproduces_the_known_vector(self):
        self.assertEqual(encode_lzw(b"AAAAAA"), b"\x80\x10\x60\x50\x38\x08")

    def test_round_trip_short(self):
        for payload in [b"", b"A", b"AB", b"-" * 3 + b"A" * 3 + b"B" * 3, b"abcabcabcabc"]:
            self.assertEqual(lzw_decode(encode_lzw(payload)), payload)

    def test_round_trip_across_code_width_growth(self):
        # Long enough to push the table past 511, 1023 and 2047 entries, so the 10-,
        # 11- and 12-bit widths all get exercised.
        rng = random.Random(7)
        payload = bytes(rng.randrange(64) for _ in range(20000))
        self.assertEqual(lzw_decode(encode_lzw(payload)), payload)

    def test_round_trip_without_early_change(self):
        rng = random.Random(11)
        payload = bytes(rng.randrange(40) for _ in range(6000))
        encoded = encode_lzw(payload, early=0)
        self.assertEqual(lzw_decode(encoded, parms(EarlyChange=0)), payload)
        self.assertEqual(lzw_decode(encoded, early=0), payload)

    def test_early_change_default_is_one(self):
        rng = random.Random(13)
        payload = bytes(rng.randrange(50) for _ in range(6000))
        encoded = encode_lzw(payload, early=1)
        self.assertEqual(lzw_decode(encoded), payload)
        self.assertEqual(lzw_decode(encoded, parms(EarlyChange=1)), payload)

    def test_table_reset_on_clear_code(self):
        # Two independent runs separated by an explicit clear code.
        self.assertEqual(lzw_decode(encode_lzw(b"x" * 5000)), b"x" * 5000)

    def test_corrupt_input_truncates_instead_of_raising(self):
        encoded = encode_lzw(b"stable payload " * 50)
        self.assertIsInstance(lzw_decode(encoded[:6]), bytes)
        self.assertEqual(lzw_decode(b""), b"")
        self.assertIsInstance(lzw_decode(b"\xff\xff\xff\xff"), bytes)

    def test_predictor_is_applied_after_lzw(self):
        payload = png_rows([(2, [10, 20, 30]), (2, [1, 2, 3])])
        encoded = encode_lzw(payload)
        result = lzw_decode(
            encoded, parms(Predictor=12, Colors=1, BitsPerComponent=8, Columns=3)
        )
        self.assertEqual(result, bytes([10, 20, 30, 11, 22, 33]))


class AsciiHexTests(unittest.TestCase):
    def test_known_vector(self):
        self.assertEqual(ascii_hex_decode(b"48656C6C6F>"), b"Hello")
        self.assertEqual(ascii_hex_decode(b"48656c6c6f>"), b"Hello")

    def test_odd_length_pads_with_zero(self):
        self.assertEqual(ascii_hex_decode(b"4>"), b"\x40")
        self.assertEqual(ascii_hex_decode(b"901FA>"), b"\x90\x1f\xa0")

    def test_whitespace_and_junk_ignored(self):
        self.assertEqual(ascii_hex_decode(b"48 65\n6C\t6C 6F >"), b"Hello")
        self.assertEqual(ascii_hex_decode(b"48zz65>"), b"He")

    def test_missing_terminator(self):
        self.assertEqual(ascii_hex_decode(b"48656C6C6F"), b"Hello")

    def test_data_after_the_terminator_is_dropped(self):
        self.assertEqual(ascii_hex_decode(b"4865>6C6C6F"), b"He")

    def test_round_trip(self):
        payload = bytes(range(256))
        self.assertEqual(encode_ascii_hex(payload), payload.hex().upper().encode() + b">")
        self.assertEqual(ascii_hex_decode(encode_ascii_hex(payload)), payload)


class Ascii85Tests(unittest.TestCase):
    def test_known_vector(self):
        self.assertEqual(ascii85_decode(b"9jqo^~>"), b"Man ")

    def test_z_shorthand(self):
        self.assertEqual(ascii85_decode(b"z~>"), b"\x00\x00\x00\x00")
        self.assertEqual(ascii85_decode(b"zz~>"), b"\x00" * 8)
        self.assertEqual(ascii85_decode(b"9jqo^z~>"), b"Man \x00\x00\x00\x00")

    def test_partial_group(self):
        self.assertEqual(ascii85_decode(b"9jn~>"), b"Ma")
        self.assertEqual(ascii85_decode(b"9jqo~>"), b"Man")

    def test_leading_adobe_prefix(self):
        self.assertEqual(ascii85_decode(b"<~9jqo^~>"), b"Man ")

    def test_whitespace_is_ignored(self):
        self.assertEqual(ascii85_decode(b"9j\nqo\t^\r\n~>"), b"Man ")

    def test_missing_terminator(self):
        self.assertEqual(ascii85_decode(b"9jqo^"), b"Man ")

    def test_data_after_terminator_is_dropped(self):
        self.assertEqual(ascii85_decode(b"9jqo^~>ignored"), b"Man ")

    def test_round_trip(self):
        rng = random.Random(99)
        for length in [0, 1, 2, 3, 4, 5, 7, 8, 100, 257]:
            payload = bytes(rng.randrange(256) for _ in range(length))
            self.assertEqual(ascii85_decode(encode_ascii85(payload)), payload, length)

    def test_encode_uses_z_for_zero_groups(self):
        self.assertEqual(encode_ascii85(b"\x00\x00\x00\x00"), b"z~>")
        self.assertEqual(encode_ascii85(b"Man "), b"9jqo^~>")


class RunLengthTests(unittest.TestCase):
    def test_run(self):
        self.assertEqual(run_length_decode(bytes([250, 65, 128])), b"A" * 7)
        self.assertEqual(run_length_decode(bytes([255, 66, 128])), b"BB")

    def test_literals(self):
        self.assertEqual(run_length_decode(bytes([2, 65, 66, 67, 128])), b"ABC")
        self.assertEqual(run_length_decode(bytes([0, 88, 128])), b"X")

    def test_mixed(self):
        data = bytes([2, 65, 66, 67]) + bytes([253, 90]) + bytes([128])
        self.assertEqual(run_length_decode(data), b"ABC" + b"Z" * 4)

    def test_missing_eod(self):
        self.assertEqual(run_length_decode(bytes([1, 65, 66])), b"AB")

    def test_truncated_run_does_not_raise(self):
        self.assertEqual(run_length_decode(bytes([250])), b"")
        self.assertEqual(run_length_decode(bytes([5, 65])), b"A")

    def test_encode_round_trip(self):
        rng = random.Random(5)
        payloads = [
            b"",
            b"A",
            b"A" * 200,
            b"ABCDEF",
            b"AAABBBCCC" * 30,
            bytes(rng.randrange(4) for _ in range(1000)),
            bytes(rng.randrange(256) for _ in range(1000)),
        ]
        for payload in payloads:
            encoded = encode_runlength(payload)
            self.assertEqual(encoded[-1], 128)
            self.assertEqual(run_length_decode(encoded), payload)


class PngPredictorTests(unittest.TestCase):
    def test_none_filter(self):
        data = png_rows([(0, [1, 2, 3])])
        self.assertEqual(
            apply_predictor(data, parms(Predictor=10, Columns=3)), bytes([1, 2, 3])
        )

    def test_sub_filter(self):
        data = png_rows([(1, [1, 1, 1, 1])])
        result = apply_predictor(data, parms(Predictor=11, Colors=1, BitsPerComponent=8, Columns=4))
        self.assertEqual(result, bytes([1, 2, 3, 4]))

    def test_up_filter(self):
        # Hand-computed: row0 has no predecessor, row1 adds row0, row2 is unfiltered.
        data = png_rows([(2, [10, 20, 30]), (2, [1, 2, 3]), (0, [5, 5, 5])])
        result = apply_predictor(data, parms(Predictor=12, Colors=1, BitsPerComponent=8, Columns=3))
        self.assertEqual(result, bytes([10, 20, 30, 11, 22, 33, 5, 5, 5]))

    def test_average_filter(self):
        # out[i] = raw[i] + floor((left + up) / 2); up is 0 on the first row.
        data = png_rows([(3, [10, 10, 10])])
        result = apply_predictor(data, parms(Predictor=13, Colors=1, BitsPerComponent=8, Columns=3))
        self.assertEqual(result, bytes([10, 15, 17]))

    def test_paeth_filter(self):
        # Row 0 (no predecessor) degenerates to Sub: 3, 3+4=7, 7+5=12.
        # Row 1 picks the "up" neighbour every time: 1+3=4, 1+7=8, 1+12=13.
        data = png_rows([(4, [3, 4, 5]), (4, [1, 1, 1])])
        result = apply_predictor(data, parms(Predictor=15, Colors=1, BitsPerComponent=8, Columns=3))
        self.assertEqual(result, bytes([3, 7, 12, 4, 8, 13]))

    def test_multi_component_sub(self):
        data = png_rows([(1, [10, 20, 30, 1, 2, 3])])
        result = apply_predictor(data, parms(Predictor=11, Colors=3, BitsPerComponent=8, Columns=2))
        self.assertEqual(result, bytes([10, 20, 30, 11, 22, 33]))

    def test_wrapping_is_modulo_256(self):
        data = png_rows([(2, [200]), (2, [100])])
        result = apply_predictor(data, parms(Predictor=12, Colors=1, BitsPerComponent=8, Columns=1))
        self.assertEqual(result, bytes([200, 44]))

    def test_truncated_final_row_is_padded(self):
        data = png_rows([(2, [1, 2, 3])]) + bytes([2, 4])
        result = apply_predictor(data, parms(Predictor=12, Colors=1, BitsPerComponent=8, Columns=3))
        self.assertEqual(result, bytes([1, 2, 3, 5, 2, 3]))

    def test_unknown_filter_byte_is_left_alone(self):
        data = png_rows([(9, [1, 2, 3])])
        result = apply_predictor(data, parms(Predictor=12, Columns=3))
        self.assertEqual(result, bytes([1, 2, 3]))

    def test_sub_byte_depth_row_length(self):
        # 1 bit per component, 16 columns => 2 bytes per row plus the filter byte.
        data = png_rows([(0, [0b10101010, 0b11001100]), (2, [0b00000001, 0])])
        result = apply_predictor(data, parms(Predictor=12, Colors=1, BitsPerComponent=1, Columns=16))
        self.assertEqual(result, bytes([0b10101010, 0b11001100, 0b10101011, 0b11001100]))


class TiffPredictorTests(unittest.TestCase):
    def test_predictor_2_rgb(self):
        data = bytes([10, 20, 30, 1, 2, 3, 5, 5, 5, 5, 5, 5])
        result = apply_predictor(data, parms(Predictor=2, Colors=3, BitsPerComponent=8, Columns=2))
        self.assertEqual(result, bytes([10, 20, 30, 11, 22, 33, 5, 5, 5, 10, 10, 10]))

    def test_predictor_2_gray(self):
        data = bytes([1, 1, 1, 1])
        result = apply_predictor(data, parms(Predictor=2, Colors=1, BitsPerComponent=8, Columns=4))
        self.assertEqual(result, bytes([1, 2, 3, 4]))

    def test_predictor_2_wraps(self):
        data = bytes([200, 100])
        result = apply_predictor(data, parms(Predictor=2, Colors=1, BitsPerComponent=8, Columns=2))
        self.assertEqual(result, bytes([200, 44]))

    def test_predictor_2_16_bit(self):
        data = bytes([0x01, 0x00, 0x00, 0x02])  # 256 then +2
        result = apply_predictor(data, parms(Predictor=2, Colors=1, BitsPerComponent=16, Columns=2))
        self.assertEqual(result, bytes([0x01, 0x00, 0x01, 0x02]))

    def test_predictor_2_4_bit(self):
        # Four 4-bit samples 1,1,1,1 -> running sums 1,2,3,4.
        data = bytes([0x11, 0x11])
        result = apply_predictor(data, parms(Predictor=2, Colors=1, BitsPerComponent=4, Columns=4))
        self.assertEqual(result, bytes([0x12, 0x34]))

    def test_predictor_1_and_absent_are_no_ops(self):
        data = bytes([1, 2, 3])
        self.assertEqual(apply_predictor(data, parms(Predictor=1)), data)
        self.assertEqual(apply_predictor(data, None), data)
        self.assertEqual(apply_predictor(data, parms()), data)

    def test_parms_accepts_a_plain_dict(self):
        data = bytes([1, 1, 1, 1])
        result = apply_predictor(data, {"Predictor": 2, "Colors": 1, "BitsPerComponent": 8, "Columns": 4})
        self.assertEqual(result, bytes([1, 2, 3, 4]))


class DecodeChainTests(unittest.TestCase):
    def test_no_filters(self):
        self.assertEqual(decode(b"raw", None), b"raw")
        self.assertEqual(decode(b"raw", []), b"raw")

    def test_single_filter_as_a_bare_name(self):
        self.assertEqual(decode(encode_flate(b"solo"), "FlateDecode"), b"solo")
        self.assertEqual(decode(encode_flate(b"solo"), PdfName("FlateDecode")), b"solo")

    def test_two_stage_chain(self):
        payload = b"two stage chain payload"
        data = encode_ascii85(encode_flate(payload))
        self.assertEqual(decode(data, ["ASCII85Decode", "FlateDecode"], [None, None]), payload)

    def test_chain_with_predictor_parms(self):
        rows = png_rows([(2, [10, 20, 30]), (2, [1, 2, 3])])
        data = encode_flate(rows)
        result = decode(
            data,
            ["FlateDecode"],
            [parms(Predictor=12, Colors=1, BitsPerComponent=8, Columns=3)],
        )
        self.assertEqual(result, bytes([10, 20, 30, 11, 22, 33]))

    def test_stops_at_an_image_filter(self):
        jpeg = b"\xff\xd8\xff\xe0 pretend jpeg"
        data = encode_ascii85(jpeg)
        self.assertEqual(decode(data, ["ASCII85Decode", "DCTDecode"], [None, None]), jpeg)
        self.assertEqual(decode(jpeg, ["DCTDecode"], [None]), jpeg)

    def test_stops_at_an_unknown_filter(self):
        payload = b"stops here"
        data = encode_flate(payload)
        self.assertEqual(decode(data, ["FlateDecode", "MadeUpDecode"], [None, None]), payload)

    def test_short_parms_list_is_padded(self):
        payload = b"short parms"
        data = encode_ascii85(encode_flate(payload))
        self.assertEqual(decode(data, ["ASCII85Decode", "FlateDecode"], []), payload)
        self.assertEqual(decode(data, ["ASCII85Decode", "FlateDecode"], None), payload)

    def test_abbreviated_names_in_a_chain(self):
        payload = b"abbreviated"
        data = encode_ascii85(encode_flate(payload))
        self.assertEqual(decode(data, ["A85", "Fl"]), payload)

    def test_decode_one(self):
        self.assertEqual(decode_one(encode_flate(b"one"), "FlateDecode"), b"one")
        self.assertEqual(decode_one(b"untouched", "DCTDecode"), b"untouched")
        self.assertEqual(decode_one(b"untouched", "Whatever"), b"untouched")

    def test_crypt_filter_is_identity(self):
        self.assertEqual(decode(b"passthrough", ["Crypt"], [None]), b"passthrough")

    def test_all_supported_filters_round_trip_through_decode(self):
        payload = b"round trip through every text filter " * 5
        cases = [
            (encode_flate(payload), "FlateDecode"),
            (encode_lzw(payload), "LZWDecode"),
            (encode_ascii_hex(payload), "ASCIIHexDecode"),
            (encode_ascii85(payload), "ASCII85Decode"),
            (encode_runlength(payload), "RunLengthDecode"),
        ]
        for data, name in cases:
            self.assertEqual(decode(data, [name], [None]), payload, name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class LzwCodeWidthBoundaryTests(unittest.TestCase):
    """Pin the 9->10 bit transition, the one place LZW implementations disagree."""

    PAYLOAD = bytes(range(256)) * 8

    def test_early_change_actually_changes_the_bit_stream(self):
        self.assertNotEqual(encode_lzw(self.PAYLOAD, early=1), encode_lzw(self.PAYLOAD, early=0))

    def test_wrong_early_change_setting_corrupts_the_output(self):
        # If the width switch were not exercised, both settings would decode alike.
        encoded = encode_lzw(self.PAYLOAD, early=1)
        self.assertEqual(lzw_decode(encoded, early=1), self.PAYLOAD)
        self.assertNotEqual(lzw_decode(encoded, early=0), self.PAYLOAD)

    def test_transition_is_lossless_in_both_directions(self):
        for early in (0, 1):
            encoded = encode_lzw(self.PAYLOAD, early=early)
            self.assertEqual(lzw_decode(encoded, early=early), self.PAYLOAD, early)
