"""The council protocol: questions, votes, verdicts, schemas.

Covers ``zfp.council.base`` and ``zfp.council.questions`` -- the closed-question layer
that makes the council auditable: a stable question id, an enum the answer cannot
escape, and a JSON-Schema validator that is 60 deterministic lines instead of a
third-party dependency.
"""

from __future__ import annotations

import unittest

from zfp.core.config import PrivacyConfig
from zfp.core.errors import CouncilError
from zfp.core.geometry import Rect
from zfp.core.types import FieldCandidate, FieldConstraints, FieldType
from zfp.council import base as B
from zfp.council import questions as Q


def candidate(
    label: str = "Tax ID",
    *,
    key: str = None,
    section=("Employer",),
    ftype: FieldType = FieldType.TEXT,
    rect: Rect = Rect(100.0, 700.0, 280.0, 712.0),
    constraints: FieldConstraints = None,
) -> FieldCandidate:
    """Build a candidate the way the detection stage would hand one over."""
    return FieldCandidate(
        id="fc_test_%s" % (label or "blank").replace(" ", "_").lower(),
        page=0,
        rect=rect,
        field_type=ftype,
        visible_label=label,
        canonical_key=key,
        parent_context=list(section),
        constraints=constraints or FieldConstraints(),
    )


class TestQuestion(unittest.TestCase):
    def test_id_is_stable_across_construction_order(self) -> None:
        a = B.Question.build("canonical_key", "p", {"type": "object"}, {"a": 1, "b": 2})
        b = B.Question.build("canonical_key", "p", {"type": "object"}, {"b": 2, "a": 1})
        self.assertEqual(a.id, b.id)

    def test_id_changes_with_context(self) -> None:
        a = B.Question.build("canonical_key", "p", {}, {"label": "City"})
        b = B.Question.build("canonical_key", "p", {}, {"label": "State"})
        self.assertNotEqual(a.id, b.id)

    def test_id_changes_with_kind_and_prompt(self) -> None:
        base = B.Question.build("canonical_key", "p", {}, {"label": "City"})
        self.assertNotEqual(base.id, B.Question.build("field_type", "p", {}, {"label": "City"}).id)
        self.assertNotEqual(base.id, B.Question.build("canonical_key", "q", {}, {"label": "City"}).id)

    def test_derived_carries_context_and_new_id(self) -> None:
        q = B.Question.build("canonical_key", "p", {}, {"label": "City"}, ["a"])
        d = q.derived(round=2, leading_answer="a")
        self.assertEqual(d.context["round"], 2)
        self.assertEqual(d.context["label"], "City")
        self.assertEqual(d.options, q.options)
        self.assertNotEqual(d.id, q.id)
        self.assertNotIn("round", q.context)

    def test_options_are_normalized_to_a_tuple(self) -> None:
        q = B.Question.build("canonical_key", "p", {}, {}, ["b", "a"])
        self.assertEqual(q.options, ("b", "a"))

    def test_hashable_despite_dict_fields(self) -> None:
        q = B.Question.build("canonical_key", "p", {"type": "object"}, {"label": "City"})
        self.assertEqual(hash(q), hash(q))
        self.assertIn(q, {q})

    def test_rejects_empty_kind(self) -> None:
        with self.assertRaises(CouncilError):
            B.Question("q_1", "  ", "p", {}, {})

    def test_enum_and_allows(self) -> None:
        q = Q.canonical_key_question(candidate(), ["person.name.first"])
        self.assertIn("unknown", q.enum())
        self.assertTrue(q.allows("person.name.first"))
        self.assertFalse(q.allows("person.name.last"))

    def test_as_dict_round_trip_shape(self) -> None:
        q = B.Question.build("canonical_key", "p", {"type": "object"}, {"a": 1}, ["x"])
        d = q.as_dict()
        self.assertEqual(d["id"], q.id)
        self.assertEqual(d["options"], ["x"])


class TestAnswerSignature(unittest.TestCase):
    def test_missing_and_empty_collapse_to_unknown(self) -> None:
        self.assertEqual(B.answer_signature(None, "canonical_key"), B.UNKNOWN)
        self.assertEqual(B.answer_signature({}, "canonical_key"), B.UNKNOWN)
        self.assertEqual(B.answer_signature({"canonical_key": "  "}, "canonical_key"), B.UNKNOWN)

    def test_choice_sets_are_order_insensitive(self) -> None:
        left = B.answer_signature({"choices": ["b", "a"]}, "choice_set")
        right = B.answer_signature({"choices": ["a", "b", "a"]}, "choice_set")
        self.assertEqual(left, right)

    def test_confidence_is_not_part_of_the_signature(self) -> None:
        left = B.answer_signature({"canonical_key": "person.email", "confidence": 0.1}, "canonical_key")
        right = B.answer_signature({"canonical_key": "person.email", "confidence": 0.9}, "canonical_key")
        self.assertEqual(left, right)

    def test_answer_field_per_kind(self) -> None:
        self.assertEqual(B.answer_field("canonical_key"), "canonical_key")
        self.assertEqual(B.answer_field("field_type"), "field_type")
        self.assertEqual(B.answer_field("choice_set"), "choices")
        self.assertEqual(B.answer_field("ambiguity"), "winner")
        self.assertEqual(B.answer_field("something_new"), "answer")


class TestVoteAndVerdict(unittest.TestCase):
    def test_confidence_is_clamped(self) -> None:
        self.assertEqual(B.Vote("m", {}, 5.0).confidence, 1.0)
        self.assertEqual(B.Vote("m", {}, -1.0).confidence, 0.0)

    def test_failed_vote_carries_error_and_zero_confidence(self) -> None:
        vote = B.Vote.failed("m", "boom")
        self.assertEqual(vote.confidence, 0.0)
        self.assertEqual(vote.error, "boom")
        self.assertTrue(vote.is_unknown("canonical_key"))

    def test_verdict_as_dict_is_json_ready(self) -> None:
        vote = B.Vote("m", {"canonical_key": "person.email", "confidence": 0.9}, 0.9)
        verdict = B.Verdict(
            question_id="q_1",
            answer={"canonical_key": "person.email"},
            confidence=0.9,
            consensus=1.0,
            votes=[vote],
            dissent=[],
            agreed_by=["m"],
        )
        d = verdict.as_dict()
        self.assertTrue(verdict.decided)
        self.assertEqual(d["agreed_by"], ["m"])
        self.assertEqual(d["votes"][0]["member"], "m")
        self.assertEqual(verdict.value("canonical_key"), "person.email")

    def test_undecided_verdict_reports_no_value(self) -> None:
        verdict = B.Verdict("q_1", None, 0.0, 0.0, [], [])
        self.assertFalse(verdict.decided)
        self.assertIsNone(verdict.value("canonical_key"))


class TestBaseMember(unittest.TestCase):
    def test_internal_failure_becomes_a_vote(self) -> None:
        class Broken(B.BaseMember):
            name = "broken"

            def _vote(self, q):
                raise RuntimeError("kaboom")

        vote = Broken().vote(B.Question.build("canonical_key", "p", {}, {}))
        self.assertEqual(vote.confidence, 0.0)
        self.assertIn("kaboom", vote.error or "")

    def test_policy_error_is_never_swallowed(self) -> None:
        from zfp.core.errors import PolicyError

        class Refuses(B.BaseMember):
            name = "refuses"

            def _vote(self, q):
                raise PolicyError("egress disabled")

        with self.assertRaises(PolicyError):
            Refuses().vote(B.Question.build("canonical_key", "p", {}, {}))

    def test_none_becomes_an_unknown_vote(self) -> None:
        class Silent(B.BaseMember):
            name = "silent"

            def _vote(self, q):
                return None

        vote = Silent().vote(B.Question.build("canonical_key", "p", {}, {}))
        self.assertTrue(vote.is_unknown("canonical_key"))

    def test_protocol_is_satisfied_by_the_local_members(self) -> None:
        from zfp.council.members import local_members

        for member in local_members():
            self.assertIsInstance(member, B.CouncilMember)


class TestQuestionBuilders(unittest.TestCase):
    def test_canonical_key_question_shape(self) -> None:
        q = Q.canonical_key_question(candidate(), ["company.tax_id.ein", "company.tax_id.vat"])
        self.assertEqual(q.kind, "canonical_key")
        self.assertEqual(q.schema["required"], ["canonical_key", "confidence"])
        self.assertFalse(q.schema["additionalProperties"])
        self.assertEqual(
            q.schema["properties"]["canonical_key"]["enum"],
            ["company.tax_id.ein", "company.tax_id.vat", "unknown"],
        )
        self.assertEqual(q.schema["properties"]["confidence"]["minimum"], 0)
        self.assertEqual(q.schema["properties"]["confidence"]["maximum"], 1)

    def test_derived_options_put_the_right_key_first(self) -> None:
        options = Q.derive_options(candidate("Tax ID"))
        self.assertEqual(options[0], "company.tax_id.ein")
        self.assertLessEqual(len(options), Q.MAX_OPTIONS)
        self.assertNotIn("unknown", options)

    def test_derived_options_complete_a_sibling_namespace(self) -> None:
        blank = candidate("", section=())
        options = Q.derive_options(
            blank,
            {"neighbour_keys": ["person.address.city", "person.address.region"]},
        )
        self.assertIn("person.address.postal_code", options)

    def test_field_type_question_enumerates_field_types(self) -> None:
        q = Q.field_type_question(candidate())
        enum = q.schema["properties"]["field_type"]["enum"]
        self.assertIn(FieldType.DATE.value, enum)
        self.assertIn("unknown", enum)
        self.assertEqual(len([e for e in enum if e == "unknown"]), 1)

    def test_choice_set_question_is_closed_over_the_captions(self) -> None:
        q = Q.choice_set_question(candidate("Marital Status", ftype=FieldType.RADIO), ["Single", "Married"])
        items = q.schema["properties"]["choices"]["items"]
        self.assertEqual(items["enum"], ["Single", "Married", "unknown"])
        self.assertEqual(q.schema["required"], ["choices", "confidence"])

    def test_ambiguity_question_lists_the_competitors(self) -> None:
        q = Q.ambiguity_question(
            candidate("City"), ["billing.address.city", "shipping.address.city"]
        )
        self.assertEqual(
            q.schema["properties"]["winner"]["enum"],
            ["billing.address.city", "shipping.address.city", "unknown"],
        )
        self.assertIn("billing.address.city = Billing City", q.context["competing_labels"])

    def test_context_is_redacted_before_it_reaches_the_question(self) -> None:
        q = Q.canonical_key_question(
            candidate(), [], {"placeholder": "12-3456789", "person.ssn": "123-45-6789"}
        )
        self.assertEqual(q.context["placeholder"], "##-#######")
        self.assertNotIn("person.ssn", q.context)
        self.assertNotIn("123-45-6789", q.prompt)

    def test_prompt_never_carries_an_unredacted_value(self) -> None:
        q = Q.canonical_key_question(candidate(), [], {"placeholder": "12-3456789"})
        self.assertIn("##-#######", q.prompt)
        self.assertNotIn("12-3456789", q.prompt)

    def test_geometry_is_never_asked_about(self) -> None:
        for question in (
            Q.canonical_key_question(candidate(), ["person.email"]),
            Q.field_type_question(candidate()),
            Q.ambiguity_question(candidate(), ["person.email"]),
        ):
            self.assertNotIn("rect", question.schema.get("properties", {}))
            self.assertNotIn("where", question.prompt.lower())

    def test_privacy_budget_is_honoured(self) -> None:
        tight = PrivacyConfig(max_context_chars=20)
        q = Q.canonical_key_question(
            candidate("Tax ID"), ["company.tax_id.ein"], {"document_title": "x" * 500}, privacy=tight
        )
        from zfp.council.redaction import context_char_count

        # ``options`` is appended after redaction: it is ontology vocabulary, not content.
        budgeted = {k: v for k, v in q.context.items() if k != "options"}
        self.assertLessEqual(context_char_count(budgeted), 20)


class TestValidateAnswer(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = Q.canonical_key_schema(["person.email", "person.phone.mobile"])

    def test_accepts_a_well_formed_answer(self) -> None:
        self.assertTrue(
            Q.validate_answer(
                {"canonical_key": "person.email", "confidence": 0.75, "reason_codes": ["a"]},
                self.schema,
            )
        )

    def test_accepts_the_unknown_escape_hatch(self) -> None:
        self.assertTrue(Q.validate_answer({"canonical_key": "unknown", "confidence": 0.0}, self.schema))

    def test_rejects_an_out_of_enum_key(self) -> None:
        self.assertFalse(
            Q.validate_answer({"canonical_key": "person.name.first", "confidence": 0.9}, self.schema)
        )

    def test_rejects_a_missing_required_field(self) -> None:
        self.assertFalse(Q.validate_answer({"canonical_key": "person.email"}, self.schema))

    def test_rejects_an_extra_property(self) -> None:
        self.assertFalse(
            Q.validate_answer(
                {"canonical_key": "person.email", "confidence": 0.9, "rect": [0, 0, 1, 1]},
                self.schema,
            )
        )

    def test_rejects_out_of_range_confidence(self) -> None:
        self.assertFalse(
            Q.validate_answer({"canonical_key": "person.email", "confidence": 1.5}, self.schema)
        )
        self.assertFalse(
            Q.validate_answer({"canonical_key": "person.email", "confidence": -0.1}, self.schema)
        )

    def test_rejects_a_wrong_type(self) -> None:
        self.assertFalse(
            Q.validate_answer({"canonical_key": "person.email", "confidence": "high"}, self.schema)
        )
        self.assertFalse(Q.validate_answer(["person.email"], self.schema))
        self.assertFalse(Q.validate_answer(None, self.schema))

    def test_booleans_are_not_numbers(self) -> None:
        self.assertFalse(
            Q.validate_answer({"canonical_key": "person.email", "confidence": True}, self.schema)
        )

    def test_array_items_are_validated(self) -> None:
        schema = Q.choice_set_schema(["Yes", "No"])
        self.assertTrue(Q.validate_answer({"choices": ["Yes"], "confidence": 0.5}, schema))
        self.assertTrue(Q.validate_answer({"choices": [], "confidence": 0.5}, schema))
        self.assertFalse(Q.validate_answer({"choices": ["Maybe"], "confidence": 0.5}, schema))
        self.assertFalse(Q.validate_answer({"choices": "Yes", "confidence": 0.5}, schema))

    def test_unknown_keywords_are_ignored(self) -> None:
        schema = {"type": "object", "title": "doc", "properties": {"a": {"type": "string"}}}
        self.assertTrue(Q.validate_answer({"a": "x"}, schema))

    def test_nested_objects_and_bounds(self) -> None:
        schema = {
            "type": "object",
            "required": ["inner"],
            "additionalProperties": False,
            "properties": {
                "inner": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 3}},
                }
            },
        }
        self.assertTrue(Q.validate_answer({"inner": {"n": 2}}, schema))
        self.assertFalse(Q.validate_answer({"inner": {"n": 9}}, schema))
        self.assertFalse(Q.validate_answer({"inner": {"m": 2}}, schema))


if __name__ == "__main__":
    unittest.main()
