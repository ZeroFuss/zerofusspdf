"""The council and its analyst.

The analyst is what makes a council more useful than a majority vote: it reports
consensus, dissent, contradictions and blind spots, and it refuses to answer at all
rather than guess.  These tests pin that behaviour, including the two properties the
rest of ZFP depends on -- determinism, and "never invent a value".
"""

from __future__ import annotations

import os
import unittest

from zfp.core.config import CouncilConfig, PrivacyConfig, ZfpConfig
from zfp.core.errors import CouncilError
from zfp.core.geometry import Rect
from zfp.core.types import FieldCandidate, FieldConstraints, FieldType
from zfp.council import questions as Q
from zfp.council.base import UNKNOWN, BaseMember, Question, Vote
from zfp.council.council import Council, build_default_council
from zfp.council.members import OpenRouterMember, local_members


def candidate(label: str = "Tax ID", *, section=("Employer",), ftype=FieldType.TEXT) -> FieldCandidate:
    """Build a candidate the way the detection stage would hand one over."""
    return FieldCandidate(
        id="fc_council_test",
        page=0,
        rect=Rect(100.0, 700.0, 280.0, 712.0),
        field_type=ftype,
        visible_label=label,
        parent_context=list(section),
        constraints=FieldConstraints(),
    )


class StubMember(BaseMember):
    """A member with a scripted answer, for exercising the analyst directly."""

    def __init__(self, name, answer, confidence, *, follows_leader=False, fails=False):
        self.name = name
        self.answer = answer
        self.confidence = confidence
        self.follows_leader = follows_leader
        self.fails = fails

    def _vote(self, q: Question) -> Vote:
        if self.fails:
            raise RuntimeError("stub failure")
        answer = self.answer
        if self.follows_leader and int(q.context.get("round", 1) or 1) >= 2:
            answer = q.context.get("leading_answer") or answer
        if answer is None:
            return self._unknown(q, "scripted_unknown")
        return self._answer(q, answer, self.confidence, ["scripted"])


def stub_question(options=("a", "b", "c")) -> Question:
    """A synthetic canonical-key question over an arbitrary closed ballot."""
    schema = {
        "type": "object",
        "required": ["canonical_key", "confidence"],
        "additionalProperties": False,
        "properties": {
            "canonical_key": {"enum": list(options) + [UNKNOWN]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
        },
    }
    return Question.build("canonical_key", "stub?", schema, {"label": "stub"}, list(options))


def council_of(members, **overrides) -> Council:
    """Build a council with a tweakable config over the given members."""
    config = CouncilConfig(**overrides)
    return Council(members, config, PrivacyConfig())


class TestDeliberateOnRealMembers(unittest.TestCase):
    def setUp(self) -> None:
        self.council = build_default_council(ZfpConfig.default())

    def test_a_pattern_backed_question_reaches_a_high_confidence_verdict(self) -> None:
        question = Q.canonical_key_question(
            candidate("Tax ID"),
            ["company.tax_id.ein", "company.tax_id.vat"],
            {"document_title": "Vendor Onboarding", "placeholder": "12-3456789"},
        )
        verdict = self.council.deliberate(question)
        self.assertEqual(verdict.answer["canonical_key"], "company.tax_id.ein")
        self.assertGreaterEqual(verdict.confidence, 0.80)
        self.assertEqual(verdict.consensus, 1.0)
        self.assertIn("rules", verdict.agreed_by)
        self.assertFalse(verdict.escalated)
        self.assertTrue(Q.validate_answer(verdict.answer, question.schema))

    def test_the_verdict_answer_always_satisfies_the_question_schema(self) -> None:
        for question in (
            Q.canonical_key_question(candidate("Email Address"), ["person.email"]),
            Q.field_type_question(candidate("Date of Birth")),
            Q.ambiguity_question(
                candidate("City", section=("Billing Information",)),
                ["billing.address.city", "shipping.address.city"],
            ),
        ):
            verdict = self.council.deliberate(question)
            if verdict.answer is not None:
                self.assertTrue(Q.validate_answer(verdict.answer, question.schema))

    def test_a_question_nobody_can_answer_returns_no_answer(self) -> None:
        question = Q.canonical_key_question(
            candidate("Reason for requesting this permit", section=()),
            ["vehicle.vin", "insurance.policy_number"],
        )
        verdict = self.council.deliberate(question)
        self.assertIsNone(verdict.answer)
        self.assertTrue(verdict.blind_spots)
        self.assertTrue(verdict.escalated)

    def test_sibling_consensus_resolves_an_unlabelled_blank(self) -> None:
        question = Q.canonical_key_question(
            candidate("", section=()),
            [],
            {"neighbour_keys": ["person.address.city", "person.address.region"]},
        )
        verdict = self.council.deliberate(question)
        self.assertIn("person.address.postal_code", question.options)
        self.assertIn("sibling", [v.member for v in verdict.votes])

    def test_deliberation_is_deterministic(self) -> None:
        question = Q.canonical_key_question(
            candidate("Tax ID"), [], {"placeholder": "12-3456789"}
        )
        first = self.council.deliberate(question)
        second = self.council.deliberate(question)
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_a_fresh_council_reaches_the_same_verdict(self) -> None:
        question = Q.canonical_key_question(candidate("Tax ID"), [], {"placeholder": "12-3456789"})
        first = build_default_council(ZfpConfig.default()).deliberate(question)
        second = build_default_council(ZfpConfig.default()).deliberate(question)
        self.assertEqual(first.as_dict(), second.as_dict())


class TestAnalyst(unittest.TestCase):
    def test_three_agreeing_and_one_dissenting(self) -> None:
        members = [
            StubMember("m1", "a", 0.9),
            StubMember("m2", "a", 0.9),
            StubMember("m3", "a", 0.9),
            StubMember("m4", "b", 0.9),
        ]
        verdict = council_of(members, escalate_below_confidence=0.0).deliberate(stub_question())
        self.assertAlmostEqual(verdict.consensus, 0.75, places=6)
        self.assertEqual(len(verdict.dissent), 1)
        self.assertEqual(verdict.dissent[0].member, "m4")
        self.assertEqual(verdict.agreed_by, ["m1", "m2", "m3"])
        self.assertEqual(verdict.answer["canonical_key"], "a")
        self.assertAlmostEqual(verdict.confidence, 0.9 * 0.75, places=6)

    def test_every_member_unknown_means_no_answer(self) -> None:
        members = [StubMember("m%d" % i, None, 0.0) for i in range(1, 5)]
        verdict = council_of(members).deliberate(stub_question())
        self.assertIsNone(verdict.answer)
        self.assertEqual(verdict.consensus, 0.0)
        self.assertIn("every member returned unknown", verdict.blind_spots)
        self.assertIn("no member proposed a canonical key", verdict.blind_spots)
        self.assertTrue(verdict.escalated)

    def test_confidence_weights_the_vote(self) -> None:
        members = [StubMember("m1", "a", 0.9), StubMember("m2", "b", 0.1)]
        verdict = council_of(members, escalate_below_confidence=0.0, max_rounds=1).deliberate(
            stub_question()
        )
        self.assertEqual(verdict.answer["canonical_key"], "a")
        self.assertAlmostEqual(verdict.consensus, 0.9, places=6)

    def test_contradictions_name_the_disagreeing_pairs(self) -> None:
        members = [
            StubMember("m1", "a", 0.95),
            StubMember("m2", "b", 0.90),
            StubMember("m3", "a", 0.20),
        ]
        verdict = council_of(members, escalate_below_confidence=0.0, max_rounds=1).deliberate(
            stub_question()
        )
        self.assertEqual(len(verdict.contradictions), 1)
        self.assertIn("m1=a@0.95", verdict.contradictions[0])
        self.assertIn("m2=b@0.90", verdict.contradictions[0])

    def test_a_failed_member_is_a_blind_spot_not_a_crash(self) -> None:
        members = [StubMember("m1", "a", 0.9), StubMember("m2", None, 0.0, fails=True)]
        verdict = council_of(members, escalate_below_confidence=0.0, max_rounds=1, quorum=2).deliberate(
            stub_question()
        )
        self.assertEqual(verdict.answer["canonical_key"], "a")
        self.assertIn("no vote from failed member(s): m2", verdict.blind_spots)

    def test_quorum_shortfall_is_reported(self) -> None:
        verdict = council_of([StubMember("m1", "a", 0.9)], escalate_below_confidence=0.0, max_rounds=1).deliberate(
            stub_question()
        )
        self.assertIn("quorum not met: 1 of 3 members voted", verdict.blind_spots)

    def test_low_confidence_produces_no_answer_rather_than_a_guess(self) -> None:
        members = [StubMember("m1", "a", 0.5), StubMember("m2", "a", 0.5), StubMember("m3", "a", 0.5)]
        verdict = council_of(members, escalate_below_confidence=0.80, max_rounds=1).deliberate(
            stub_question()
        )
        self.assertIsNone(verdict.answer)
        self.assertTrue(verdict.escalated)
        self.assertEqual(verdict.agreed_by, ["m1", "m2", "m3"])
        self.assertTrue(any("escalation threshold" in b for b in verdict.blind_spots))

    def test_the_unknown_group_can_never_win(self) -> None:
        members = [
            StubMember("m1", None, 0.0),
            StubMember("m2", None, 0.0),
            StubMember("m3", "a", 0.9),
        ]
        verdict = council_of(members, escalate_below_confidence=0.0, max_rounds=1).deliberate(
            stub_question()
        )
        self.assertEqual(verdict.answer["canonical_key"], "a")
        self.assertEqual(verdict.consensus, 1.0)

    def test_ties_break_deterministically_on_confidence_then_name(self) -> None:
        members = [StubMember("zeta", "b", 0.5), StubMember("alpha", "a", 0.5)]
        verdict = council_of(members, escalate_below_confidence=0.0, max_rounds=1).deliberate(
            stub_question()
        )
        self.assertEqual(verdict.answer["canonical_key"], "a")
        self.assertEqual([v.member for v in verdict.votes], ["alpha", "zeta"])

    def test_a_local_only_council_reports_that_blind_spot_when_undecided(self) -> None:
        verdict = council_of([StubMember("m1", None, 0.0)]).deliberate(stub_question())
        self.assertIn("no external member participated (local-only council)", verdict.blind_spots)


class TestRounds(unittest.TestCase):
    def test_a_split_council_runs_a_second_round(self) -> None:
        members = [
            StubMember("m1", "a", 0.9, follows_leader=True),
            StubMember("m2", "b", 0.9, follows_leader=True),
            StubMember("m3", "c", 0.9, follows_leader=True),
        ]
        verdict = council_of(members, max_rounds=2, escalate_below_confidence=0.0).deliberate(
            stub_question()
        )
        self.assertTrue(verdict.escalated)
        self.assertEqual(verdict.consensus, 1.0)
        self.assertEqual(verdict.answer["canonical_key"], "a")
        self.assertTrue(any("round 2 was required" in b for b in verdict.blind_spots))

    def test_the_verdict_keeps_the_original_question_id(self) -> None:
        question = stub_question()
        members = [
            StubMember("m1", "a", 0.9, follows_leader=True),
            StubMember("m2", "b", 0.9, follows_leader=True),
        ]
        verdict = council_of(members, max_rounds=2, escalate_below_confidence=0.0).deliberate(question)
        self.assertEqual(verdict.question_id, question.id)

    def test_a_second_round_that_does_not_help_keeps_the_first(self) -> None:
        members = [StubMember("m1", "a", 0.9), StubMember("m2", "b", 0.9)]
        verdict = council_of(members, max_rounds=2, escalate_below_confidence=0.0).deliberate(
            stub_question()
        )
        self.assertAlmostEqual(verdict.consensus, 0.5, places=6)
        self.assertTrue(verdict.escalated)

    def test_a_single_round_council_never_derives_a_question(self) -> None:
        members = [StubMember("m1", "a", 0.9), StubMember("m2", "b", 0.9)]
        verdict = council_of(members, max_rounds=1, escalate_below_confidence=0.0).deliberate(
            stub_question()
        )
        self.assertFalse(any("round 2" in b for b in verdict.blind_spots))

    def test_rounds_stay_deterministic(self) -> None:
        def build():
            return council_of(
                [
                    StubMember("m1", "a", 0.9, follows_leader=True),
                    StubMember("m2", "b", 0.9, follows_leader=True),
                    StubMember("m3", "c", 0.9, follows_leader=True),
                ],
                max_rounds=2,
                escalate_below_confidence=0.0,
            )

        self.assertEqual(
            build().deliberate(stub_question()).as_dict(),
            build().deliberate(stub_question()).as_dict(),
        )


class TestDeliberateMany(unittest.TestCase):
    def setUp(self) -> None:
        self.council = build_default_council(ZfpConfig.default())
        self.questions = [
            Q.canonical_key_question(candidate("Tax ID"), [], {"placeholder": "12-3456789"}),
            Q.canonical_key_question(candidate("Email Address", section=()), ["person.email"]),
            Q.canonical_key_question(candidate("First Name", section=()), ["person.name.first"]),
            Q.field_type_question(candidate("Date of Birth", section=())),
        ]

    def test_results_are_returned_in_the_input_order(self) -> None:
        verdicts = self.council.deliberate_many(self.questions, max_workers=4)
        self.assertEqual([v.question_id for v in verdicts], [q.id for q in self.questions])

    def test_parallel_matches_serial(self) -> None:
        parallel = [v.as_dict() for v in self.council.deliberate_many(self.questions, max_workers=4)]
        serial = [self.council.deliberate(q).as_dict() for q in self.questions]
        self.assertEqual(parallel, serial)

    def test_an_empty_batch_is_an_empty_list(self) -> None:
        self.assertEqual(self.council.deliberate_many([]), [])

    def test_a_single_worker_still_works(self) -> None:
        verdicts = self.council.deliberate_many(self.questions, max_workers=1)
        self.assertEqual(len(verdicts), len(self.questions))


class TestRoster(unittest.TestCase):
    def _clear_key(self) -> None:
        previous = os.environ.pop("OPENROUTER_API_KEY", None)
        if previous is not None:
            self.addCleanup(os.environ.__setitem__, "OPENROUTER_API_KEY", previous)

    def _set_key(self, value: str) -> None:
        previous = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = value
        if previous is None:
            self.addCleanup(os.environ.pop, "OPENROUTER_API_KEY", None)
        else:
            self.addCleanup(os.environ.__setitem__, "OPENROUTER_API_KEY", previous)

    def test_the_default_council_is_four_local_members(self) -> None:
        self._clear_key()
        council = build_default_council(ZfpConfig.default())
        self.assertEqual(
            [m.name for m in council.members], ["heuristic", "ontology", "rules", "sibling"]
        )

    def test_a_key_alone_does_not_add_the_remote_member(self) -> None:
        self._set_key("sk-test")
        council = build_default_council(ZfpConfig.default())
        self.assertNotIn("openrouter", [m.name for m in council.members])

    def test_permission_plus_a_key_adds_the_remote_member(self) -> None:
        self._set_key("sk-test")
        config = ZfpConfig.default()
        config.privacy.allow_external_inference = True
        council = build_default_council(config)
        self.assertEqual(
            [m.name for m in council.members],
            ["heuristic", "ontology", "openrouter", "rules", "sibling"],
        )
        self.assertTrue(council.available_members()[2].available())

    def test_permission_without_a_key_stays_local(self) -> None:
        self._clear_key()
        config = ZfpConfig.default()
        config.privacy.allow_external_inference = True
        self.assertNotIn(
            "openrouter", [m.name for m in build_default_council(config).members]
        )

    def test_a_remote_member_is_skipped_when_the_council_forbids_egress(self) -> None:
        member = OpenRouterMember(
            api_key="sk-test", privacy=PrivacyConfig(allow_external_inference=True)
        )
        council = Council(local_members() + [member], CouncilConfig(), PrivacyConfig())
        self.assertNotIn("openrouter", [m.name for m in council.available_members()])

    def test_members_are_sorted_on_entry(self) -> None:
        council = Council(
            [StubMember("zeta", "a", 0.5), StubMember("alpha", "a", 0.5)],
            CouncilConfig(),
            PrivacyConfig(),
        )
        self.assertEqual([m.name for m in council.members], ["alpha", "zeta"])

    def test_an_empty_council_is_a_construction_error(self) -> None:
        with self.assertRaises(CouncilError):
            Council([], CouncilConfig(), PrivacyConfig())

    def test_an_unavailable_roster_still_returns_a_verdict(self) -> None:
        class Absent(BaseMember):
            name = "absent"

            def available(self) -> bool:
                return False

            def _vote(self, q):
                raise AssertionError("must not be asked")

        verdict = Council([Absent()], CouncilConfig(), PrivacyConfig()).deliberate(stub_question())
        self.assertIsNone(verdict.answer)
        self.assertIn("no council member was available", verdict.blind_spots)

    def test_a_member_with_a_broken_availability_check_is_skipped(self) -> None:
        class Flaky(BaseMember):
            name = "flaky"

            def available(self) -> bool:
                raise RuntimeError("no idea")

            def _vote(self, q):
                raise AssertionError("must not be asked")

        council = Council([Flaky(), StubMember("m1", "a", 0.9)], CouncilConfig(), PrivacyConfig())
        self.assertEqual([m.name for m in council.available_members()], ["m1"])


if __name__ == "__main__":
    unittest.main()
