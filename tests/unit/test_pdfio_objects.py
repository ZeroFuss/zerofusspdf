"""Unit tests for :mod:`zfp.pdfio.objects`."""

from __future__ import annotations

import unittest
import zlib

from zfp.pdfio.objects import (
    PdfArray,
    PdfDict,
    PdfName,
    PdfNull,
    PdfRef,
    PdfStream,
    PdfString,
    as_list,
    make_array,
    make_dict,
    pdfdoc_decode,
    pdfdoc_encode,
    resolve_with,
)


class DictResolver:
    """A minimal ``Resolver`` implementation backed by a mapping of (num, gen)."""

    def __init__(self, objects):
        self.objects = objects

    def resolve(self, obj):
        seen = 0
        while isinstance(obj, PdfRef) and seen < 32:
            obj = self.objects.get((obj.num, obj.gen))
            seen += 1
        return obj


class PdfNullTests(unittest.TestCase):
    def test_singleton(self):
        self.assertIs(PdfNull(), PdfNull.NULL)
        self.assertIs(PdfNull(), PdfNull())

    def test_falsy_and_none_equality(self):
        self.assertFalse(bool(PdfNull.NULL))
        self.assertEqual(PdfNull.NULL, None)
        self.assertEqual(None, PdfNull.NULL)
        self.assertNotEqual(PdfNull.NULL, 0)
        self.assertEqual(str(PdfNull.NULL), "null")


class PdfNameTests(unittest.TestCase):
    def test_stores_without_slash(self):
        name = PdfName("Type")
        self.assertEqual(name.value, "Type")
        self.assertEqual(str(name), "/Type")

    def test_frozen_and_hashable(self):
        name = PdfName("Type")
        self.assertEqual(hash(name), hash(PdfName("Type")))
        self.assertEqual({PdfName("A"): 1}[PdfName("A")], 1)
        with self.assertRaises(AttributeError):  # dataclasses.FrozenInstanceError
            name.value = "Other"

    def test_decode_hash_escapes(self):
        self.assertEqual(PdfName.decode(b"/A#20B").value, "A B")
        self.assertEqual(PdfName.decode("/A#20B").value, "A B")
        self.assertEqual(PdfName.decode(b"A#20B").value, "A B")
        self.assertEqual(PdfName.decode(b"/Lime#20Green").value, "Lime Green")
        self.assertEqual(PdfName.decode(b"/paired#28#29parentheses").value, "paired()parentheses")
        self.assertEqual(PdfName.decode(b"/A#42").value, "AB")

    def test_decode_is_lenient_about_bad_escapes(self):
        self.assertEqual(PdfName.decode(b"/A#ZZ").value, "A#ZZ")
        self.assertEqual(PdfName.decode(b"/trailing#").value, "trailing#")

    def test_decode_passes_through_a_name(self):
        name = PdfName("Type")
        self.assertIs(PdfName.decode(name), name)

    def test_encoded_escapes_specials(self):
        self.assertEqual(PdfName("A B").encoded, b"/A#20B")
        self.assertEqual(PdfName("a/b").encoded, b"/a#2Fb")
        self.assertEqual(PdfName("a#b").encoded, b"/a#23b")
        self.assertEqual(PdfName("(x)").encoded, b"/#28x#29")
        self.assertEqual(PdfName("Type").encoded, b"/Type")

    def test_encoded_round_trip(self):
        for value in ["Type", "A B", "a/b", "a#b", "Ünïcode", "%weird[]{}"]:
            self.assertEqual(PdfName.decode(PdfName(value).encoded).value, value)

    def test_utf8_names(self):
        self.assertEqual(PdfName("é").encoded, b"/#C3#A9")
        self.assertEqual(PdfName.decode(b"/#C3#A9").value, "é")


class PdfRefTests(unittest.TestCase):
    def test_repr_and_encoding(self):
        ref = PdfRef(12, 0)
        self.assertEqual(str(ref), "12 0 R")
        self.assertEqual(ref.encoded, b"12 0 R")
        self.assertEqual(ref.as_tuple(), (12, 0))

    def test_equality_and_hash(self):
        self.assertEqual(PdfRef(3), PdfRef(3, 0))
        self.assertEqual(len({PdfRef(3, 0), PdfRef(3)}), 1)
        self.assertNotEqual(PdfRef(3, 0), PdfRef(3, 1))


class PdfStringTests(unittest.TestCase):
    def test_ascii_text(self):
        self.assertEqual(PdfString(b"Hello").text(), "Hello")

    def test_utf16be_bom(self):
        raw = b"\xfe\xff\x00H\x00i"
        self.assertEqual(PdfString(raw).text(), "Hi")

    def test_utf16be_odd_length_is_tolerated(self):
        self.assertEqual(PdfString(b"\xfe\xff\x00H\x00i\x00").text(), "Hi")

    def test_pdfdocencoding_differences(self):
        # 0x18..0x1f and 0x80..0x9f differ from Latin-1, as does 0xa0 (Euro).
        self.assertEqual(PdfString(bytes([0x18])).text(), "˘")
        self.assertEqual(PdfString(bytes([0x1F])).text(), "˜")
        self.assertEqual(PdfString(bytes([0x80])).text(), "•")
        self.assertEqual(PdfString(bytes([0x92])).text(), "™")
        self.assertEqual(PdfString(bytes([0x9E])).text(), "ž")
        self.assertEqual(PdfString(bytes([0xA0])).text(), "€")
        # Everything else agrees with Latin-1.
        self.assertEqual(PdfString(bytes([0xE9])).text(), "é")

    def test_pdfdoc_helpers(self):
        self.assertEqual(pdfdoc_decode(b"\x80"), "•")
        self.assertEqual(pdfdoc_encode("•"), b"\x80")
        self.assertIsNone(pdfdoc_encode("中"))

    def test_from_text_prefers_ascii(self):
        value = PdfString.from_text("Jane Doe")
        self.assertEqual(value.raw, b"Jane Doe")
        self.assertFalse(value.hexform)

    def test_from_text_falls_back_to_utf16(self):
        value = PdfString.from_text("naïve")
        self.assertTrue(value.raw.startswith(b"\xfe\xff"))
        self.assertEqual(value.text(), "naïve")

    def test_from_text_round_trip(self):
        for text in ["", "plain", "naïve", "日本語", "mixed é 123"]:
            self.assertEqual(PdfString.from_text(text).text(), text)

    def test_serialize_literal_escapes(self):
        self.assertEqual(PdfString(b"a(b)c").serialize(), rb"(a\(b\)c)")
        self.assertEqual(PdfString(b"back\\slash").serialize(), rb"(back\\slash)")
        self.assertEqual(
            PdfString(b"\r\n\t\b\f").serialize(), b"(\\r\\n\\t\\b\\f)"
        )
        self.assertEqual(PdfString(b"\x01").serialize(), b"(\\001)")
        self.assertEqual(PdfString(b"plain").serialize(), b"(plain)")

    def test_serialize_hexform(self):
        self.assertEqual(PdfString(b"Hello", hexform=True).serialize(), b"<48656C6C6F>")
        self.assertEqual(PdfString(b"", hexform=True).serialize(), b"<>")

    def test_equality_ignores_hexform(self):
        self.assertEqual(PdfString(b"x"), PdfString(b"x", hexform=True))
        self.assertEqual(PdfString(b"x"), b"x")
        self.assertEqual(len({PdfString(b"x"), PdfString(b"x", hexform=True)}), 1)

    def test_len_and_bytes(self):
        value = PdfString(b"abc")
        self.assertEqual(len(value), 3)
        self.assertEqual(bytes(value), b"abc")


class PdfArrayTests(unittest.TestCase):
    def test_is_a_list(self):
        array = PdfArray([1, 2, 3])
        self.assertIsInstance(array, list)
        self.assertEqual(array[1], 2)
        array.append(4)
        self.assertEqual(len(array), 4)

    def test_resolved(self):
        resolver = DictResolver({(1, 0): 42})
        array = PdfArray([PdfRef(1), 7])
        self.assertEqual(array.resolved(resolver), [42, 7])


class PdfDictTests(unittest.TestCase):
    def test_key_coercion(self):
        d = PdfDict()
        d[PdfName("Type")] = PdfName("Page")
        self.assertIn("Type", d)
        self.assertIn(PdfName("Type"), d)
        self.assertIn("/Type", d)
        self.assertEqual(d[PdfName("Type")], PdfName("Page"))
        self.assertEqual(d["/Type"], PdfName("Page"))
        self.assertEqual(list(d.keys()), ["Type"])

    def test_constructor_and_update_coerce(self):
        d = PdfDict({PdfName("A"): 1, "/B": 2})
        self.assertEqual(sorted(d.keys()), ["A", "B"])
        d.update({PdfName("C"): 3})
        self.assertEqual(d.get(PdfName("C")), 3)
        d.update([("/D", 4)])
        self.assertEqual(d["D"], 4)

    def test_pop_setdefault_delete(self):
        d = PdfDict({"A": 1})
        self.assertEqual(d.setdefault(PdfName("B"), 2), 2)
        self.assertEqual(d.pop("/B"), 2)
        self.assertEqual(d.pop("/B", "missing"), "missing")
        del d[PdfName("A")]
        self.assertEqual(len(d), 0)

    def test_get_name(self):
        d = PdfDict({"Type": PdfName("Page"), "S": "/GoTo", "N": 3})
        self.assertEqual(d.get_name("Type"), "Page")
        self.assertEqual(d.get_name("S"), "GoTo")
        self.assertIsNone(d.get_name("N"))
        self.assertEqual(d.get_name("Missing", "fallback"), "fallback")

    def test_get_numeric(self):
        d = PdfDict({"I": 5, "F": 1.5, "S": "2.5", "B": True, "X": PdfName("no")})
        self.assertEqual(d.get_int("I"), 5)
        self.assertEqual(d.get_int("F"), 1)
        self.assertEqual(d.get_int("S"), 2)
        self.assertEqual(d.get_int("B"), 1)
        self.assertIsNone(d.get_int("X"))
        self.assertEqual(d.get_int("Missing", 7), 7)
        self.assertEqual(d.get_float("I"), 5.0)
        self.assertEqual(d.get_float("F"), 1.5)
        self.assertEqual(d.get_float("S"), 2.5)
        self.assertEqual(d.get_float("Missing", 0.0), 0.0)

    def test_get_container_readers(self):
        inner = PdfDict({"K": 1})
        d = PdfDict({"D": inner, "A": [1, 2], "S": PdfStream(PdfDict({"L": 1}), b"")})
        self.assertIs(d.get_dict("D"), inner)
        self.assertEqual(d.get_array("A"), [1, 2])
        self.assertIsInstance(d.get_array("A"), PdfArray)
        self.assertEqual(d.get_dict("S"), PdfDict({"L": 1}))
        self.assertIsNone(d.get_dict("Missing"))
        self.assertIsNone(d.get_array("D"))

    def test_readers_follow_a_resolver(self):
        resolver = DictResolver({(1, 0): PdfName("Page"), (2, 0): PdfArray([1, 2])})
        d = PdfDict({"Type": PdfRef(1), "Kids": PdfRef(2)})
        self.assertIsNone(d.get_name("Type"))
        self.assertEqual(d.get_name("Type", None, resolver), "Page")
        self.assertEqual(d.get_array("Kids", None, resolver), [1, 2])

    def test_null_reads_as_default(self):
        d = PdfDict({"X": PdfNull.NULL})
        self.assertIsNone(d.get_name("X"))
        self.assertEqual(d.get_int("X", 9), 9)

    def test_get_text_and_bool(self):
        d = PdfDict({"T": PdfString.from_text("naïve"), "B": False})
        self.assertEqual(d.get_text("T"), "naïve")
        self.assertIs(d.get_bool("B"), False)
        self.assertIsNone(d.get_bool("T"))

    def test_copy_returns_pdfdict(self):
        d = PdfDict({"A": 1})
        self.assertIsInstance(d.copy(), PdfDict)


class PdfStreamTests(unittest.TestCase):
    def test_no_filter_returns_raw(self):
        stream = PdfStream(PdfDict({"Length": 5}), b"plain")
        self.assertEqual(stream.decoded(), b"plain")
        self.assertEqual(stream.raw, b"plain")

    def test_flate_decoding(self):
        payload = b"BT /F1 12 Tf (hi) Tj ET" * 4
        raw = zlib.compress(payload)
        stream = PdfStream(PdfDict({"Filter": PdfName("FlateDecode")}), raw)
        self.assertEqual(stream.decoded(), payload)

    def test_decoded_is_cached(self):
        payload = b"cached"
        stream = PdfStream(PdfDict({"Filter": PdfName("FlateDecode")}), zlib.compress(payload))
        first = stream.decoded()
        # Poke the private buffer: a cached result must not be recomputed.
        stream._raw = b"garbage"
        self.assertEqual(stream.decoded(), first)
        # Assigning through the public setter clears the cache.
        stream.raw = zlib.compress(b"fresh")
        self.assertEqual(stream.decoded(), b"fresh")

    def test_filter_chain(self):
        from zfp.pdfio.filters import encode_ascii_hex

        payload = b"chained payload"
        raw = encode_ascii_hex(zlib.compress(payload))
        stream = PdfStream(
            PdfDict({"Filter": PdfArray([PdfName("ASCIIHexDecode"), PdfName("FlateDecode")])}),
            raw,
        )
        self.assertEqual(stream.decoded(), payload)

    def test_image_filter_returns_raw(self):
        stream = PdfStream(PdfDict({"Filter": PdfName("DCTDecode")}), b"\xff\xd8\xff\xe0jpeg")
        self.assertEqual(stream.decoded(), b"\xff\xd8\xff\xe0jpeg")
        self.assertTrue(stream.is_image())

    def test_flate_then_image_filter_stops_at_the_codec(self):
        inner = b"\xff\xd8jpegbytes"
        stream = PdfStream(
            PdfDict({"Filter": PdfArray([PdfName("FlateDecode"), PdfName("DCTDecode")])}),
            zlib.compress(inner),
        )
        self.assertEqual(stream.decoded(), inner)

    def test_filter_names_and_parms(self):
        stream = PdfStream(
            PdfDict(
                {
                    "Filter": PdfArray([PdfName("FlateDecode")]),
                    "DecodeParms": PdfDict({"Predictor": 12, "Columns": 3}),
                }
            ),
            b"",
        )
        self.assertEqual(stream.filter_names(), ["FlateDecode"])
        parms = stream.decode_parms()
        self.assertEqual(len(parms), 1)
        self.assertEqual(parms[0].get("Predictor"), 12)

    def test_predictor_applied_through_the_stream(self):
        rows = bytes([2, 10, 20, 30]) + bytes([2, 1, 2, 3])
        stream = PdfStream(
            PdfDict(
                {
                    "Filter": PdfName("FlateDecode"),
                    "DecodeParms": PdfDict(
                        {"Predictor": 12, "Colors": 1, "BitsPerComponent": 8, "Columns": 3}
                    ),
                }
            ),
            zlib.compress(rows),
        )
        self.assertEqual(stream.decoded(), bytes([10, 20, 30, 11, 22, 33]))

    def test_filter_via_a_reference(self):
        resolver = DictResolver({(5, 0): PdfName("FlateDecode")})
        stream = PdfStream(PdfDict({"Filter": PdfRef(5)}), zlib.compress(b"ref"))
        self.assertEqual(stream.decoded(resolver), b"ref")

    def test_len_and_repr(self):
        stream = PdfStream(PdfDict(), b"1234")
        self.assertEqual(len(stream), 4)
        self.assertIn("PdfStream", repr(stream))


class HelperTests(unittest.TestCase):
    def test_resolve_with_callable_and_none(self):
        self.assertEqual(resolve_with(3, None), 3)
        self.assertEqual(resolve_with(PdfRef(1), lambda o: 99), 99)
        self.assertEqual(resolve_with(PdfRef(1), DictResolver({(1, 0): 5})), 5)

    def test_as_list(self):
        self.assertEqual(as_list(None), [])
        self.assertEqual(as_list(PdfNull.NULL), [])
        self.assertEqual(as_list(3), [3])
        self.assertEqual(as_list(PdfArray([1, 2])), [1, 2])

    def test_make_helpers(self):
        d = make_dict({"A": 1}, Type=PdfName("Page"))
        self.assertEqual(d.get_name("Type"), "Page")
        self.assertEqual(d["A"], 1)
        self.assertIsInstance(make_array([1]), PdfArray)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
