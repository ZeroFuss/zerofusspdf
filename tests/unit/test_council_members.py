"""The council members, local and remote.

The four local members are pure Python and always available; the remote member is off
unless egress is explicitly permitted.  No test in this file touches the network: the
one function in ZFP that can open a socket, ``zfp.council.openrouter._transport``, is
replaced with a recorder, and several tests assert it was never called at all.
"""

from __future__ import annotations

import json
import unittest

from zfp.core.config import PrivacyConfig
from zfp.core.errors import CouncilError, PolicyError
from zfp.core.geometry import Rect
from zfp.core.types import FieldCandidate, FieldConstraints, FieldType
from zfp.council import members as M
from zfp.council import openrouter as OR
from zfp.council import questions as Q
from zfp.council.base import UNKNOWN, Question


def candidate(
    label: str = "Tax ID",
    *,
    section=("Employer",),
    ftype: FieldType = FieldType.TEXT,
    key: str = None,
    constraints: FieldConstraints = None,
) -> FieldCandidate:
    """Build a candidate the way the detection stage would hand one over."""
    return FieldCandidate(
        id="fc_member_test",
        page=0,
        rect=Rect(100.0, 700.0, 280.0, 712.0),
        field_type=ftype,
        visible_label=label,
        canonical_key=key,
        parent_context=list(section),
        constraints=constraints or FieldConstraints(),
    )


class TestRulesMember(unittest.TestCase):
    def setUp(self) -> None:
        self.member = M.RulesMember()

    def test_is_always_available(self) -> None:
        self.assertTrue(self.member.available())

    def test_placeholder_pattern_produces_a_high_confidence_vote(self) -> None:
        q = Q.canonical_key_question(
            candidate("Tax ID"), ["company.tax_id.ein", "company.tax_id.vat"], {"placeholder": "12-3456789"}
        )
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["canonical_key"], "company.tax_id.ein")
        self.assertGreaterEqual(vote.confidence, 0.95)
        self.assertIn("placeholder_pattern:ein_placeholder", vote.reason_codes)
        self.assertIn("pattern_label_agree", vote.reason_codes)

    def test_pattern_alone_still_votes(self) -> None:
        q = Q.canonical_key_question(
            candidate("", section=()), ["company.tax_id.ein"], {"placeholder": "##-#######"}
        )
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["canonical_key"], "company.tax_id.ein")
        self.assertGreater(vote.confidence, 0.9)

    def test_exact_alias_alone_votes_a_little_lower(self) -> None:
        q = Q.canonical_key_question(candidate("Email Address", section=()), ["person.email"])
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["canonical_key"], "person.email")
        self.assertIn("alias_exact", vote.reason_codes)
        self.assertLess(vote.confidence, 0.95)

    def test_a_key_off_the_ballot_is_reported_not_voted(self) -> None:
        q = Q.canonical_key_question(candidate("Email Address", section=()), ["person.phone.mobile"])
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["canonical_key"], UNKNOWN)
        self.assertIn("resolved_key_not_on_ballot", vote.reason_codes)

    def test_no_signal_is_an_explicit_unknown(self) -> None:
        q = Q.canonical_key_question(candidate("", section=()), ["person.email"])
        vote = self.member.vote(q)
        self.assertTrue(vote.is_unknown("canonical_key"))
        self.assertEqual(vote.confidence, 0.0)
        self.assertIsNone(vote.error)

    def test_field_type_comes_from_the_pattern(self) -> None:
        q = Q.field_type_question(
            candidate("Date", section=(), ftype=FieldType.UNKNOWN), {"placeholder": "MM/DD/YYYY"}
        )
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["field_type"], FieldType.DATE.value)

    def test_choice_set_uses_declared_choices(self) -> None:
        cand = candidate(
            "Marital Status",
            ftype=FieldType.RADIO,
            constraints=FieldConstraints(choices=["Single", "Married"]),
        )
        q = Q.choice_set_question(cand, ["Single", "Married", "Divorced"])
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["choices"], ["Single", "Married"])
        self.assertIn("declared_choices", vote.reason_codes)

    def test_choice_set_falls_back_to_nearby_captions_for_selectable_controls(self) -> None:
        q = Q.choice_set_question(candidate("Status", ftype=FieldType.CHECKBOX), ["Yes", "No"])
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["choices"], ["Yes", "No"])
        self.assertIn("control_is_selectable", vote.reason_codes)


class TestHeuristicMember(unittest.TestCase):
    def setUp(self) -> None:
        self.member = M.HeuristicMember()

    def test_section_namespace_breaks_a_billing_shipping_tie(self) -> None:
        q = Q.ambiguity_question(
            candidate("City", section=("Billing Information",)),
            ["billing.address.city", "shipping.address.city"],
        )
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["winner"], "billing.address.city")
        self.assertIn("section_namespace", vote.reason_codes)

    def test_the_same_label_under_ship_to_goes_the_other_way(self) -> None:
        q = Q.ambiguity_question(
            candidate("City", section=("Shipping Address",)),
            ["billing.address.city", "shipping.address.city"],
        )
        self.assertEqual(self.member.vote(q).answer["winner"], "shipping.address.city")

    def test_penalizes_an_option_whose_type_contradicts_the_geometry(self) -> None:
        q = Q.canonical_key_question(
            candidate("", section=(), ftype=FieldType.DATE),
            ["person.date_of_birth", "person.signature"],
        )
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["canonical_key"], "person.date_of_birth")
        self.assertIn("field_type_agrees", vote.reason_codes)

    def test_neighbour_namespace_is_worth_something(self) -> None:
        q = Q.canonical_key_question(
            candidate("", section=()),
            ["shipping.address.postal_code", "person.address.postal_code"],
            {"neighbour_keys": ["shipping.address.city", "shipping.address.region"]},
        )
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["canonical_key"], "shipping.address.postal_code")

    def test_abstains_on_a_choice_set_question(self) -> None:
        q = Q.choice_set_question(candidate("Status", ftype=FieldType.RADIO), ["Yes", "No"])
        self.assertTrue(self.member.vote(q).is_unknown("choice_set"))


class TestOntologyFuzzyMember(unittest.TestCase):
    def setUp(self) -> None:
        self.member = M.OntologyFuzzyMember()

    def test_survives_ocr_damage(self) -> None:
        q = Q.canonical_key_question(
            candidate("Frst Narne", section=()), ["person.name.first", "person.name.last"]
        )
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["canonical_key"], "person.name.first")
        self.assertGreater(vote.confidence, 0.55)
        self.assertLess(vote.confidence, 1.0)

    def test_confidence_is_the_similarity_ratio(self) -> None:
        q = Q.canonical_key_question(candidate("Email", section=()), ["person.email"])
        self.assertEqual(self.member.vote(q).confidence, 1.0)

    def test_no_label_means_no_vote(self) -> None:
        q = Q.canonical_key_question(candidate("", section=()), ["person.email"])
        vote = self.member.vote(q)
        self.assertTrue(vote.is_unknown("canonical_key"))
        self.assertIn("no_label", vote.reason_codes)

    def test_unrelated_label_stays_below_the_floor(self) -> None:
        q = Q.canonical_key_question(
            candidate("Reason for requesting this permit", section=()), ["person.email"]
        )
        self.assertTrue(self.member.vote(q).is_unknown("canonical_key"))


class TestSiblingConsensusMember(unittest.TestCase):
    def setUp(self) -> None:
        self.member = M.SiblingConsensusMember()

    def test_completes_the_neighbours_namespace(self) -> None:
        q = Q.canonical_key_question(
            candidate("", section=()),
            ["person.address.postal_code", "company.tax_id.ein"],
            {"neighbour_keys": ["person.address.city", "person.address.region"]},
        )
        vote = self.member.vote(q)
        self.assertEqual(vote.answer["canonical_key"], "person.address.postal_code")
        self.assertIn("namespace_completion:person.address", vote.reason_codes)
        self.assertGreater(vote.confidence, 0.6)

    def test_prefers_an_unclaimed_slot(self) -> None:
        q = Q.canonical_key_question(
            candidate("", section=()),
            ["person.address.city", "person.address.postal_code"],
            {"neighbour_keys": ["person.address.city", "person.address.region"]},
        )
        self.assertEqual(self.member.vote(q).answer["canonical_key"], "person.address.postal_code")

    def test_without_neighbours_it_abstains(self) -> None:
        q = Q.canonical_key_question(candidate("Tax ID"), ["company.tax_id.ein"])
        vote = self.member.vote(q)
        self.assertTrue(vote.is_unknown("canonical_key"))
        self.assertIn("no_resolved_neighbours", vote.reason_codes)

    def test_a_single_neighbour_is_capped(self) -> None:
        q = Q.canonical_key_question(
            candidate("", section=()),
            ["person.address.postal_code"],
            {"neighbour_keys": ["person.address.city"]},
        )
        vote = self.member.vote(q)
        self.assertLessEqual(vote.confidence, 0.55)
        self.assertIn("single_neighbour", vote.reason_codes)


class TestLocalModelMember(unittest.TestCase):
    def setUp(self) -> None:
        self.question = Q.canonical_key_question(
            candidate("Email Address", section=()), ["person.email", "person.phone.mobile"]
        )

    def test_disabled_without_a_classifier(self) -> None:
        member = M.LocalModelMember(module_name="zfp_no_such_local_classifier")
        self.assertFalse(member.available())
        self.assertIn("no_local_classifier", member.vote(self.question).reason_codes)

    def test_a_supplied_classifier_votes(self) -> None:
        member = M.LocalModelMember(
            lambda question: {"canonical_key": "person.email", "confidence": 0.77}
        )
        self.assertTrue(member.available())
        vote = member.vote(self.question)
        self.assertEqual(vote.answer["canonical_key"], "person.email")
        self.assertEqual(vote.confidence, 0.77)
        self.assertIn("local_classifier", vote.reason_codes)

    def test_the_classifier_sees_only_the_question(self) -> None:
        seen = {}

        def classifier(question):
            seen.update(question)
            return None

        member = M.LocalModelMember(classifier)
        vote = member.vote(self.question)
        self.assertEqual(seen["id"], self.question.id)
        self.assertEqual(seen["context"], self.question.context)
        self.assertIn("classifier_declined", vote.reason_codes)

    def test_an_invalid_reply_is_an_error_not_an_exception(self) -> None:
        member = M.LocalModelMember(lambda question: {"canonical_key": "person.name.first"})
        vote = member.vote(self.question)
        self.assertEqual(vote.confidence, 0.0)
        self.assertIn("schema", vote.error or "")

    def test_an_off_ballot_answer_is_refused(self) -> None:
        loose = Question.build(
            "canonical_key",
            "which key?",
            {
                "type": "object",
                "required": ["canonical_key", "confidence"],
                "additionalProperties": False,
                "properties": {
                    "canonical_key": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            {"label": "Email"},
            ["person.email"],
        )
        member = M.LocalModelMember(
            lambda question: {"canonical_key": "person.name.first", "confidence": 0.9}
        )
        vote = member.vote(loose)
        self.assertEqual(vote.confidence, 0.0)
        self.assertIn("off the ballot", vote.error or "")

    def test_a_crashing_classifier_is_a_failed_vote(self) -> None:
        def boom(question):
            raise RuntimeError("model exploded")

        vote = M.LocalModelMember(boom).vote(self.question)
        self.assertEqual(vote.confidence, 0.0)
        self.assertIn("model exploded", vote.error or "")

    def test_it_is_not_part_of_the_default_local_roster(self) -> None:
        self.assertNotIn("local_model", [m.name for m in M.local_members()])


class _Recorder:
    """Stand-in transport that records calls instead of opening a socket."""

    def __init__(self, payload=None, error=None):
        self.calls = []
        self.payload = payload
        self.error = error

    def __call__(self, url, data, headers, timeout):
        self.calls.append({"url": url, "data": data, "headers": dict(headers), "timeout": timeout})
        if self.error is not None:
            raise self.error
        return json.dumps(self.payload).encode("utf-8")


def _reply(answer) -> dict:
    """A minimal chat-completions envelope carrying ``answer`` as JSON content."""
    return {"choices": [{"message": {"role": "assistant", "content": json.dumps(answer)}}]}


class _TransportPatch(unittest.TestCase):
    """Base class that swaps the transport out for the duration of a test."""

    def install(self, recorder: _Recorder) -> _Recorder:
        original = OR._transport
        OR._transport = recorder
        self.addCleanup(lambda: setattr(OR, "_transport", original))
        return recorder


class TestChatJson(_TransportPatch):
    def setUp(self) -> None:
        self.schema = Q.canonical_key_schema(["person.email"])
        self.messages = [{"role": "user", "content": "hello"}]

    def test_refuses_before_touching_the_network_when_egress_is_off(self) -> None:
        recorder = self.install(_Recorder(_reply({"canonical_key": "person.email", "confidence": 1.0})))
        with self.assertRaises(PolicyError):
            OR.chat_json(
                "openrouter/auto",
                self.messages,
                self.schema,
                api_key="sk-test",
                privacy=PrivacyConfig(),
            )
        self.assertEqual(recorder.calls, [])

    def test_refuses_without_an_api_key(self) -> None:
        recorder = self.install(_Recorder(_reply({})))
        with self.assertRaises(PolicyError):
            OR.chat_json("openrouter/auto", self.messages, self.schema, api_key="")
        self.assertEqual(recorder.calls, [])

    def test_refuses_plaintext_egress(self) -> None:
        recorder = self.install(_Recorder(_reply({})))
        with self.assertRaises(PolicyError):
            OR.chat_json(
                "openrouter/auto",
                self.messages,
                self.schema,
                api_key="sk-test",
                base_url="http://openrouter.ai/api/v1",
            )
        self.assertEqual(recorder.calls, [])

    def test_refuses_to_drop_zero_data_retention(self) -> None:
        recorder = self.install(_Recorder(_reply({})))
        privacy = PrivacyConfig(allow_external_inference=True, require_zero_data_retention=True)
        with self.assertRaises(PolicyError):
            OR.chat_json(
                "openrouter/auto",
                self.messages,
                self.schema,
                api_key="sk-test",
                zdr=False,
                privacy=privacy,
            )
        self.assertEqual(recorder.calls, [])

    def test_refuses_a_provider_off_the_allowlist(self) -> None:
        recorder = self.install(_Recorder(_reply({})))
        privacy = PrivacyConfig(allow_external_inference=True, provider_allowlist=["trusted"])
        with self.assertRaises(PolicyError):
            OR.chat_json(
                "openrouter/auto",
                self.messages,
                self.schema,
                api_key="sk-test",
                allow_providers=["random"],
                privacy=privacy,
            )
        self.assertEqual(recorder.calls, [])

    def test_sends_strict_structured_output_and_zdr_routing(self) -> None:
        answer = {"canonical_key": "person.email", "confidence": 0.9}
        recorder = self.install(_Recorder(_reply(answer)))
        privacy = PrivacyConfig(allow_external_inference=True, provider_allowlist=["trusted"])
        out = OR.chat_json(
            "openrouter/auto",
            self.messages,
            self.schema,
            api_key="sk-test",
            privacy=privacy,
        )
        self.assertEqual(out, answer)
        self.assertEqual(len(recorder.calls), 1)
        call = recorder.calls[0]
        self.assertEqual(call["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer sk-test")
        self.assertIn("HTTP-Referer", call["headers"])
        self.assertIn("X-Title", call["headers"])
        body = json.loads(call["data"].decode("utf-8"))
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["response_format"]["json_schema"]["schema"], self.schema)
        self.assertTrue(body["provider"]["zdr"])
        self.assertEqual(body["provider"]["data_collection"], "deny")
        self.assertEqual(body["provider"]["only"], ["trusted"])
        self.assertEqual(body["temperature"], 0)

    def test_request_bytes_are_deterministic(self) -> None:
        answer = {"canonical_key": "person.email", "confidence": 0.9}
        recorder = self.install(_Recorder(_reply(answer)))
        privacy = PrivacyConfig(allow_external_inference=True)
        for _ in range(2):
            OR.chat_json("m", self.messages, self.schema, api_key="k", privacy=privacy)
        self.assertEqual(recorder.calls[0]["data"], recorder.calls[1]["data"])

    def test_malformed_replies_raise_council_errors(self) -> None:
        privacy = PrivacyConfig(allow_external_inference=True)
        for payload in ({"choices": []}, {"choices": [{"message": {"content": "not json"}}]}):
            self.install(_Recorder(payload))
            with self.assertRaises(CouncilError):
                OR.chat_json("m", self.messages, self.schema, api_key="k", privacy=privacy)

    def test_transport_failure_becomes_a_council_error(self) -> None:
        self.install(_Recorder(error=OSError("connection reset")))
        privacy = PrivacyConfig(allow_external_inference=True)
        with self.assertRaises(CouncilError):
            OR.chat_json("m", self.messages, self.schema, api_key="k", privacy=privacy)


class TestOpenRouterMember(_TransportPatch):
    def setUp(self) -> None:
        self.question = Q.canonical_key_question(
            candidate("Tax ID"), ["company.tax_id.ein", "company.tax_id.vat"]
        )

    def test_unavailable_by_default(self) -> None:
        self.assertFalse(M.OpenRouterMember(api_key="sk-test").available())
        self.assertFalse(
            M.OpenRouterMember(
                api_key="", privacy=PrivacyConfig(allow_external_inference=True)
            ).available()
        )

    def test_available_only_with_a_key_and_permission(self) -> None:
        member = M.OpenRouterMember(
            api_key="sk-test", privacy=PrivacyConfig(allow_external_inference=True)
        )
        self.assertTrue(member.available())
        self.assertTrue(member.remote)

    def test_vote_raises_policy_error_with_egress_off(self) -> None:
        recorder = self.install(_Recorder(_reply({})))
        member = M.OpenRouterMember(api_key="sk-test")
        with self.assertRaises(PolicyError):
            member.vote(self.question)
        self.assertEqual(recorder.calls, [])

    def test_a_valid_reply_becomes_a_vote(self) -> None:
        answer = {"canonical_key": "company.tax_id.ein", "confidence": 0.93, "reason_codes": ["fmt"]}
        self.install(_Recorder(_reply(answer)))
        member = M.OpenRouterMember(
            "test/model", api_key="sk-test", privacy=PrivacyConfig(allow_external_inference=True)
        )
        vote = member.vote(self.question)
        self.assertEqual(vote.answer["canonical_key"], "company.tax_id.ein")
        self.assertEqual(vote.confidence, 0.93)
        self.assertIn("remote_model:test/model", vote.reason_codes)
        self.assertIsNone(vote.error)

    def test_an_off_ballot_answer_is_refused_not_raised(self) -> None:
        self.install(_Recorder(_reply({"canonical_key": "person.email", "confidence": 0.99})))
        member = M.OpenRouterMember(
            api_key="sk-test", privacy=PrivacyConfig(allow_external_inference=True)
        )
        vote = member.vote(self.question)
        self.assertEqual(vote.confidence, 0.0)
        self.assertIsNotNone(vote.error)

    def test_a_schema_violation_is_an_error_not_an_exception(self) -> None:
        self.install(_Recorder(_reply({"canonical_key": "company.tax_id.ein"})))
        member = M.OpenRouterMember(
            api_key="sk-test", privacy=PrivacyConfig(allow_external_inference=True)
        )
        vote = member.vote(self.question)
        self.assertEqual(vote.confidence, 0.0)
        self.assertIn("schema", vote.error or "")

    def test_a_transport_failure_is_an_error_not_an_exception(self) -> None:
        self.install(_Recorder(error=OSError("connection reset")))
        member = M.OpenRouterMember(
            api_key="sk-test", privacy=PrivacyConfig(allow_external_inference=True)
        )
        vote = member.vote(self.question)
        self.assertEqual(vote.confidence, 0.0)
        self.assertIsNotNone(vote.error)

    def test_an_unknown_answer_is_an_unknown_vote(self) -> None:
        self.install(_Recorder(_reply({"canonical_key": "unknown", "confidence": 0.0})))
        member = M.OpenRouterMember(
            api_key="sk-test", privacy=PrivacyConfig(allow_external_inference=True)
        )
        self.assertTrue(member.vote(self.question).is_unknown("canonical_key"))

    def test_a_secret_in_the_context_stops_the_request(self) -> None:
        recorder = self.install(_Recorder(_reply({})))
        privacy = PrivacyConfig(allow_external_inference=True)
        member = M.OpenRouterMember(api_key="sk-test", privacy=privacy)
        leaky = Question.build(
            "canonical_key",
            "which key?",
            Q.canonical_key_schema(["person.ssn"]),
            {"label": "SSN", "nearby": ["123-45-6789"]},
            ["person.ssn"],
        )
        with self.assertRaises(PolicyError):
            member.vote(leaky)
        self.assertEqual(recorder.calls, [])

    def test_messages_are_deterministic_and_carry_only_the_context(self) -> None:
        member = M.OpenRouterMember(api_key="sk-test")
        first = member.messages_for(self.question)
        self.assertEqual(first, member.messages_for(self.question))
        payload = json.loads(first[1]["content"])
        self.assertEqual(payload["context"], self.question.context)
        self.assertIn(UNKNOWN, payload["options"])


class TestLocalMembers(unittest.TestCase):
    def test_four_members_in_name_order(self) -> None:
        names = [m.name for m in M.local_members()]
        self.assertEqual(names, ["heuristic", "ontology", "rules", "sibling"])

    def test_none_of_them_is_remote(self) -> None:
        self.assertFalse(any(m.remote for m in M.local_members()))

    def test_all_of_them_answer_every_question_kind(self) -> None:
        cand = candidate("Tax ID")
        questions = [
            Q.canonical_key_question(cand, ["company.tax_id.ein"]),
            Q.field_type_question(cand),
            Q.choice_set_question(cand, ["Yes", "No"]),
            Q.ambiguity_question(cand, ["company.tax_id.ein", "company.tax_id.vat"]),
        ]
        for member in M.local_members():
            for question in questions:
                vote = member.vote(question)
                self.assertIsNone(vote.error, "%s failed on %s" % (member.name, question.kind))
                self.assertTrue(Q.validate_answer(vote.answer, question.schema))


if __name__ == "__main__":
    unittest.main()
