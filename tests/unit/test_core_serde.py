"""Unit tests for :mod:`zfp.core.serde` and :mod:`zfp.core.ids`."""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import subprocess
import sys
import unittest
from enum import Enum
from typing import Any, Dict, List

from zfp.core import ids, serde
from zfp.core.geometry import Matrix, PageGeometry, Point, Rect
from zfp.core.types import (
    Confidence,
    Evidence,
    EvidenceKind,
    FieldCandidate,
    FieldSpec,
    FieldType,
    FormSchema,
)


class Color(str, Enum):
    RED = "red"


class Level(Enum):
    HIGH = 3


@dataclasses.dataclass
class Nested:
    rect: Rect
    tags: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Outer:
    name: str
    child: Nested
    color: Color
    blob: bytes


class HasAsDict:
    def as_dict(self) -> Dict[str, Any]:
        return {"kind": "custom", "rect": Rect(0, 0, 1, 2)}


class ToJsonableTest(unittest.TestCase):
    def test_scalars_pass_through(self) -> None:
        for value in (None, True, False, 0, -5, 1.25, "text", ""):
            self.assertEqual(serde.to_jsonable(value), value)

    def test_geometry_becomes_flat_lists(self) -> None:
        self.assertEqual(serde.to_jsonable(Rect(1, 2, 3, 4)), [1, 2, 3, 4])
        self.assertEqual(serde.to_jsonable(Point(1.5, -2)), [1.5, -2])
        self.assertEqual(serde.to_jsonable(Matrix()), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def test_enums_become_their_value(self) -> None:
        self.assertEqual(serde.to_jsonable(Color.RED), "red")
        self.assertEqual(serde.to_jsonable(Level.HIGH), 3)
        self.assertEqual(serde.to_jsonable(FieldType.SIGNATURE), "signature")

    def test_bytes_become_a_marked_base64_payload(self) -> None:
        out = serde.to_jsonable(b"\x00\xff\xfe binary")
        self.assertEqual(list(out), [serde.BYTES_MARKER])
        self.assertEqual(base64.b64decode(out[serde.BYTES_MARKER]), b"\x00\xff\xfe binary")
        self.assertEqual(serde.to_jsonable(bytearray(b"ab")), serde.to_jsonable(b"ab"))

    def test_dataclasses_recurse(self) -> None:
        out = serde.to_jsonable(
            Outer(name="o", child=Nested(Rect(0, 0, 5, 5), ["a"]), color=Color.RED, blob=b"z")
        )
        self.assertEqual(out["name"], "o")
        self.assertEqual(out["child"], {"rect": [0, 0, 5, 5], "tags": ["a"]})
        self.assertEqual(out["color"], "red")
        self.assertEqual(base64.b64decode(out["blob"][serde.BYTES_MARKER]), b"z")

    def test_as_dict_is_preferred_and_recursed(self) -> None:
        self.assertEqual(serde.to_jsonable(HasAsDict()), {"kind": "custom", "rect": [0, 0, 1, 2]})

    def test_sets_are_sorted_for_determinism(self) -> None:
        self.assertEqual(serde.to_jsonable({3, 1, 2}), [1, 2, 3])
        self.assertEqual(serde.to_jsonable(frozenset({"b", "a"})), ["a", "b"])

    def test_tuples_become_lists(self) -> None:
        self.assertEqual(serde.to_jsonable((1, (2, 3))), [1, [2, 3]])

    def test_mapping_keys_become_strings(self) -> None:
        self.assertEqual(serde.to_jsonable({1: Rect(0, 0, 1, 1)}), {"1": [0, 0, 1, 1]})

    def test_generators_become_lists(self) -> None:
        self.assertEqual(serde.to_jsonable(iter([1, 2])), [1, 2])

    def test_unknown_objects_fall_back_to_str(self) -> None:
        self.assertEqual(serde.to_jsonable(object), str(object))

    def test_page_geometry_round_trips_through_dataclass_walk(self) -> None:
        geom = PageGeometry(2, Rect(0, 0, 612, 792), Rect(10, 10, 602, 782), 90)
        out = serde.to_jsonable(geom)
        self.assertEqual(out["index"], 2)
        self.assertEqual(out["crop_box"], [10, 10, 602, 782])
        self.assertEqual(out["rotation"], 90)


class DumpsLoadsTest(unittest.TestCase):
    def test_dumps_is_valid_json_with_sorted_keys(self) -> None:
        text = serde.dumps({"b": 1, "a": Rect(0, 0, 1, 1)}, indent=None)
        self.assertEqual(text, '{"a": [0, 0, 1, 1], "b": 1}')
        self.assertEqual(json.loads(text)["b"], 1)

    def test_dumps_indent(self) -> None:
        self.assertIn("\n", serde.dumps({"a": 1}))
        self.assertNotIn("\n", serde.dumps({"a": 1}, indent=None))

    def test_dumps_is_deterministic(self) -> None:
        payload = {"z": [1, 2], "a": {"k": Rect(1, 2, 3, 4)}, "s": {3, 1}}
        self.assertEqual(serde.dumps(payload), serde.dumps(payload))

    def test_bytes_round_trip(self) -> None:
        restored = serde.loads(serde.dumps({"blob": b"\x00\x01\xff"}))
        self.assertEqual(restored["blob"], b"\x00\x01\xff")

    def test_loads_leaves_plain_dicts_alone(self) -> None:
        self.assertEqual(serde.loads('{"a": 1}'), {"a": 1})

    def test_loads_ignores_a_malformed_bytes_marker(self) -> None:
        out = serde.loads('{"%s": "not base64 !!"}' % serde.BYTES_MARKER)
        self.assertIsInstance(out, dict)


class RegisterDecoderTest(unittest.TestCase):
    def test_registered_type_is_reconstructed(self) -> None:
        @dataclasses.dataclass
        class Widget:
            label: str

            def as_dict(self) -> Dict[str, Any]:
                return {serde.TYPE_MARKER: "Widget", "label": self.label}

        serde.register_decoder(Widget, lambda d: Widget(label=d["label"]))
        self.assertIn("Widget", serde.decoders())
        restored = serde.loads(serde.dumps(Widget("hello")))
        self.assertEqual(restored, Widget("hello"))

    def test_unregistered_marker_stays_a_dict(self) -> None:
        out = serde.loads('{"%s": "NotRegisteredAnywhere", "x": 1}' % serde.TYPE_MARKER)
        self.assertEqual(out["x"], 1)

    def test_decoder_must_be_callable(self) -> None:
        with self.assertRaises(TypeError):
            serde.register_decoder(Rect, "not callable")  # type: ignore[arg-type]

    def test_form_schema_decoder_is_registered(self) -> None:
        self.assertIn("FormSchema", serde.decoders())
        schema = FormSchema("doc", [FieldSpec("a", FieldType.TEXT, 0, Rect(0, 0, 1, 1))])
        payload = dict(schema.as_dict())
        payload[serde.TYPE_MARKER] = "FormSchema"
        self.assertEqual(serde.loads(json.dumps(payload)), schema)


class CanonicalReprTest(unittest.TestCase):
    def test_scalars(self) -> None:
        self.assertEqual(ids.canonical_repr(None), "None")
        self.assertEqual(ids.canonical_repr(True), "true")
        self.assertEqual(ids.canonical_repr(7), "7")
        self.assertEqual(ids.canonical_repr("a"), "s:a")
        self.assertEqual(ids.canonical_repr(b"\x01"), "b:01")

    def test_floats_are_rounded_and_signed_zero_normalized(self) -> None:
        self.assertEqual(ids.canonical_repr(-0.0), ids.canonical_repr(0.0))
        self.assertEqual(ids.canonical_repr(1.0000000001), ids.canonical_repr(1.0))
        self.assertNotEqual(ids.canonical_repr(1.001), ids.canonical_repr(1.0))
        self.assertEqual(ids.canonical_repr(float("nan")), "nan")

    def test_mapping_order_does_not_matter(self) -> None:
        self.assertEqual(ids.canonical_repr({"a": 1, "b": 2}), ids.canonical_repr({"b": 2, "a": 1}))

    def test_geometry_and_enums(self) -> None:
        self.assertIn("Rect(", ids.canonical_repr(Rect(0, 0, 1, 1)))
        self.assertIn("FieldType", ids.canonical_repr(FieldType.TEXT))

    def test_addresses_are_stripped(self) -> None:
        rendered = ids.canonical_repr(object())
        self.assertIn("0xX", rendered)
        self.assertEqual(rendered, ids.canonical_repr(object()))


class StableIdTest(unittest.TestCase):
    def test_deterministic_within_a_process(self) -> None:
        a = ids.stable_id("page", 1, Rect(1, 2, 3, 4))
        b = ids.stable_id("page", 1, Rect(1, 2, 3, 4))
        self.assertEqual(a, b)

    def test_default_length_and_hex(self) -> None:
        value = ids.stable_id("x")
        self.assertEqual(len(value), ids.DEFAULT_LENGTH)
        int(value, 16)

    def test_length_argument(self) -> None:
        self.assertEqual(len(ids.stable_id("x", length=8)), 8)
        self.assertEqual(len(ids.stable_id("x", length=64)), 64)
        self.assertEqual(len(ids.stable_id("x", length=1000)), 64)
        with self.assertRaises(ValueError):
            ids.stable_id("x", length=0)

    def test_prefix(self) -> None:
        bare = ids.stable_id("x")
        self.assertEqual(ids.stable_id("x", prefix="fc"), "fc_" + bare)

    def test_different_inputs_differ(self) -> None:
        self.assertNotEqual(ids.stable_id("a"), ids.stable_id("b"))
        self.assertNotEqual(ids.stable_id("a", "b"), ids.stable_id("ab"))
        self.assertNotEqual(ids.stable_id(1), ids.stable_id("1"))

    def test_identical_across_processes_and_hash_seeds(self) -> None:
        script = (
            "from zfp.core.ids import stable_id\n"
            "from zfp.core.geometry import Rect\n"
            "print(stable_id('page', 3, Rect(1.5, 2.5, 3.5, 4.5), {'b': 2, 'a': 1}, prefix='fc'))\n"
        )
        expected = ids.stable_id(
            "page", 3, Rect(1.5, 2.5, 3.5, 4.5), {"b": 2, "a": 1}, prefix="fc"
        )
        for seed in ("0", "1", "12345"):
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = seed
            env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
            out = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.strip(), expected, "PYTHONHASHSEED=%s" % seed)


class CandidateIdTest(unittest.TestCase):
    def test_deterministic(self) -> None:
        self.assertEqual(
            ids.candidate_id(0, Rect(1, 2, 3, 4), "underline"),
            ids.candidate_id(0, Rect(1, 2, 3, 4), "underline"),
        )

    def test_prefixed(self) -> None:
        self.assertTrue(ids.candidate_id(0, Rect(0, 0, 1, 1), "box").startswith("fc_"))

    def test_sub_millipoint_jitter_collapses(self) -> None:
        self.assertEqual(
            ids.candidate_id(0, Rect(1.0, 2.0, 3.0, 4.0), "underline"),
            ids.candidate_id(0, Rect(1.0000004, 2.0, 3.0, 4.0), "underline"),
        )

    def test_inverted_rect_normalizes_to_the_same_id(self) -> None:
        self.assertEqual(
            ids.candidate_id(0, Rect(3, 4, 1, 2), "underline"),
            ids.candidate_id(0, Rect(1, 2, 3, 4), "underline"),
        )

    def test_page_and_kind_matter(self) -> None:
        base = ids.candidate_id(0, Rect(1, 2, 3, 4), "underline")
        self.assertNotEqual(base, ids.candidate_id(1, Rect(1, 2, 3, 4), "underline"))
        self.assertNotEqual(base, ids.candidate_id(0, Rect(1, 2, 3, 4), "box"))


class EngineTypeSerializationTest(unittest.TestCase):
    def test_field_candidate_survives_dumps_loads(self) -> None:
        candidate = FieldCandidate(
            id=ids.candidate_id(0, Rect(10, 20, 110, 34), "underline"),
            page=0,
            rect=Rect(10, 20, 110, 34),
            field_type=FieldType.EMAIL,
            confidence=Confidence(0.9, 0.8, 0.7, 0.0),
        )
        candidate.add_evidence(Evidence(EvidenceKind.VECTOR_LINE, 0.95, rect=Rect(10, 19, 110, 20)))
        restored = FieldCandidate.from_dict(serde.loads(serde.dumps(candidate)))
        self.assertEqual(restored, candidate)

    def test_form_schema_survives_dumps_loads(self) -> None:
        schema = FormSchema(
            document_id="doc",
            fields=[
                FieldSpec("a", FieldType.TEXT, 0, Rect(0, 0, 10, 10), max_length=20),
                FieldSpec("b", FieldType.CHECKBOX, 1, Rect(0, 0, 10, 10), export_value="Yes"),
            ],
        )
        restored = FormSchema.from_dict(serde.loads(serde.dumps(schema)))
        self.assertEqual(restored, schema)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
