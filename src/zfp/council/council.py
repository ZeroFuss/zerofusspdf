"""The council and its analyst.

A council convenes **only** for a field the deterministic stages could not settle, and
it answers exactly one closed question.  What comes back is not a label but a
:class:`~zfp.council.base.Verdict`: the winning answer, how much of the council's weight
stood behind it, who dissented, which high-confidence votes contradict each other, and
what *no* member addressed at all.

The analyst's arithmetic, in full:

* every vote's **weight** is its confidence, so an uncertain member dilutes rather than
  decides;
* votes are grouped by their normalized answer, and the ``"unknown"`` group can never
  win -- unanimous ignorance is not a verdict;
* ``consensus`` = winning group weight / total weight;
* ``confidence`` = the winning group's mean confidence scaled by ``consensus``, so three
  confident members who agree beat one confident member the others contradict.

Below ``CouncilConfig.escalate_below_confidence`` the council returns ``answer=None``.
That is the whole point of the design: "zero touch" must never mean "silently
hallucinate", and a field ZFP cannot resolve is reported as unresolved, not guessed.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.config import CouncilConfig, PrivacyConfig, ZfpConfig
from ..core.errors import CouncilError
from ..core.logging import get_logger
from .base import UNKNOWN, Question, Verdict, Vote, answer_field, answer_signature
from .members import (
    HeuristicMember,
    OntologyFuzzyMember,
    OpenRouterMember,
    RulesMember,
    SiblingConsensusMember,
)
from .openrouter import DEFAULT_MODEL
from .questions import validate_answer

__all__ = ["Council", "build_default_council"]

_LOG = get_logger(__name__)

#: What it means when nobody answered a question of this kind.
_KIND_BLIND_SPOT: Dict[str, str] = {
    "canonical_key": "no member proposed a canonical key",
    "field_type": "no member proposed a field type",
    "choice_set": "no member proposed a choice set",
    "ambiguity": "no member chose between the competing keys",
}


def _vote_sort_key(vote: Vote) -> Tuple[float, str]:
    """The one deterministic vote ordering: ``(-confidence, member_name)``."""
    return (-vote.confidence, vote.member)


class Council:
    """A quorum of independent members voting on one structured question."""

    def __init__(
        self,
        members: Sequence[Any],
        config: CouncilConfig,
        privacy: PrivacyConfig,
        logger: Any = None,
    ) -> None:
        """Assemble a council.

        Args:
            members: The members.  Sorted by name on entry, so the order they were
                passed in cannot change a verdict.
            config: Quorum, agreement threshold, escalation floor, round budget.
            privacy: Egress policy.  A remote member is skipped outright -- not asked
                and not counted -- when this forbids external inference.
            logger: Optional logger; defaults to the module logger.
        """
        if not members:
            raise CouncilError("a council needs at least one member")
        self.members = sorted(members, key=lambda m: str(getattr(m, "name", "")))
        self.config = config
        self.privacy = privacy
        self.log = logger if logger is not None else _LOG

    # -- roster ------------------------------------------------------------------
    def _permitted(self, member: Any) -> bool:
        """False for a remote member when the privacy policy forbids egress."""
        return bool(self.privacy.allow_external_inference) or not bool(
            getattr(member, "remote", False)
        )

    def available_members(self) -> List[Any]:
        """The members that can vote right now, in deterministic name order.

        A member whose ``available()`` itself misbehaves is treated as unavailable
        rather than being allowed to take the council down.
        """
        out: List[Any] = []
        for member in self.members:
            if not self._permitted(member):
                continue
            try:
                if member.available():
                    out.append(member)
            except Exception as exc:  # noqa: BLE001 - availability must never raise
                self.log.debug(
                    "council member %s failed its availability check: %s",
                    getattr(member, "name", "?"),
                    exc,
                )
        return out

    def _collect(self, q: Question, members: Sequence[Any]) -> List[Vote]:
        """Collect one vote per member, in deterministic order."""
        votes: List[Vote] = []
        for member in members:
            vote = member.vote(q)
            if vote is None:
                continue
            votes.append(vote)
        return votes

    # -- deliberation ------------------------------------------------------------
    def deliberate(self, q: Question) -> Verdict:
        """Run one question past the council and return the analyst's verdict.

        When the first round's consensus falls below ``config.agreement_threshold`` and
        ``config.max_rounds > 1``, a second round runs on a *derived* question carrying
        the disagreement summary as context.  Local members may use it (as a tie-break
        toward the leading answer, never as a new source of truth), and because the
        summary is deterministic the second round is reproducible too.  The better of
        the two rounds is kept, and the verdict always reports the original question id.
        """
        members = self.available_members()
        if not members:
            return Verdict(
                question_id=q.id,
                answer=None,
                confidence=0.0,
                consensus=0.0,
                votes=[],
                dissent=[],
                blind_spots=[
                    "no council member was available",
                    _KIND_BLIND_SPOT.get(q.kind, "no member answered"),
                ],
                escalated=True,
            )

        verdict = self.analyst(self._collect(q, members), q)
        escalated = False

        if (
            verdict.consensus < self.config.agreement_threshold
            and self.config.max_rounds > 1
            and int(q.context.get("round", 1) or 1) < 2
        ):
            escalated = True
            second = q.derived(**self._round2_context(verdict, q))
            rerun = self.analyst(self._collect(second, members), second)
            rerun.question_id = q.id
            rerun.blind_spots = sorted(
                set(rerun.blind_spots)
                | {"round 2 was required (derived question %s)" % second.id}
            )
            if rerun.consensus >= verdict.consensus:
                verdict = rerun
            else:
                verdict.blind_spots = sorted(
                    set(verdict.blind_spots)
                    | {"round 2 did not improve consensus (question %s)" % second.id}
                )

        if verdict.answer is not None and verdict.confidence < self.config.escalate_below_confidence:
            verdict.answer = None
            escalated = True
            verdict.blind_spots = sorted(
                set(verdict.blind_spots)
                | {
                    "confidence %.3f is below the escalation threshold %.3f; no answer "
                    "was invented" % (verdict.confidence, self.config.escalate_below_confidence)
                }
            )

        if verdict.answer is None and not any(
            getattr(m, "remote", False) for m in members
        ):
            verdict.blind_spots = sorted(
                set(verdict.blind_spots) | {"no external member participated (local-only council)"}
            )

        verdict.escalated = verdict.escalated or escalated
        self.log.debug(
            "council verdict for %s: answer=%r consensus=%.3f confidence=%.3f",
            q.id,
            None if verdict.answer is None else verdict.answer.get(answer_field(q.kind)),
            verdict.consensus,
            verdict.confidence,
        )
        return verdict

    def _round2_context(self, verdict: Verdict, q: Question) -> Dict[str, Any]:
        """Build the deterministic disagreement summary shown in round two."""
        tally: List[str] = []
        for vote in verdict.votes:
            tally.append(
                "%s=%s@%.2f" % (vote.member, vote.signature(q.kind), vote.confidence)
            )
        leader = ""
        if verdict.answer is not None:
            value = verdict.answer.get(answer_field(q.kind))
            leader = answer_signature({answer_field(q.kind): value}, q.kind)
        elif verdict.votes:
            decisive = [v for v in verdict.votes if not v.is_unknown(q.kind)]
            if decisive:
                leader = sorted(decisive, key=_vote_sort_key)[0].signature(q.kind)
        return {
            "round": 2,
            "leading_answer": leader,
            "disagreement": "round 1 reached %.2f consensus; votes: %s"
            % (verdict.consensus, ", ".join(tally)),
        }

    def deliberate_many(self, qs: List[Question], max_workers: int = 4) -> List[Verdict]:
        """Deliberate a batch of questions, returning verdicts in the input order.

        Members are stateless, so the questions are genuinely independent; the results
        are re-sorted into the caller's order before returning, which is what keeps a
        parallel run byte-identical to a serial one.
        """
        questions = list(qs)
        if not questions:
            return []
        workers = max(1, min(int(max_workers), len(questions)))
        if workers == 1:
            return [self.deliberate(q) for q in questions]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(index, pool.submit(self.deliberate, q)) for index, q in enumerate(questions)]
            results: List[Tuple[int, Verdict]] = [(index, f.result()) for index, f in futures]
        results.sort(key=lambda item: item[0])
        return [verdict for _, verdict in results]

    # -- analysis ----------------------------------------------------------------
    def analyst(self, votes: List[Vote], q: Question) -> Verdict:
        """Turn a round of votes into a verdict.

        Consensus, dissent, contradictions and blind spots are all computed here; this
        is the pass that makes a council more useful than a majority vote, because it
        reports what the council *failed* to consider as explicitly as what it decided.
        """
        ordered = sorted(votes, key=_vote_sort_key)
        groups: Dict[str, List[Vote]] = {}
        for vote in ordered:
            groups.setdefault(vote.signature(q.kind), []).append(vote)

        total_weight = sum(v.confidence for v in ordered)
        decisive = {sig: group for sig, group in groups.items() if sig != UNKNOWN}

        blind_spots = self._blind_spots(ordered, decisive, q)
        contradictions = self._contradictions(ordered, q)

        if not decisive or total_weight <= 0.0:
            return Verdict(
                question_id=q.id,
                answer=None,
                confidence=0.0,
                consensus=0.0,
                votes=ordered,
                dissent=[],
                blind_spots=blind_spots,
                contradictions=contradictions,
                escalated=True,
                agreed_by=[],
            )

        ranked = sorted(
            decisive.items(),
            key=lambda item: (
                -sum(v.confidence for v in item[1]),
                -max(v.confidence for v in item[1]),
                item[0],
            ),
        )
        signature, winners = ranked[0]
        win_weight = sum(v.confidence for v in winners)
        consensus = win_weight / total_weight if total_weight > 0 else 0.0
        mean_confidence = win_weight / float(len(winners))
        confidence = mean_confidence * consensus

        winner_names = {v.member for v in winners}
        dissent = [v for v in ordered if v.member not in winner_names]
        answer = self._compose_answer(winners, q, confidence)
        if answer is not None and not validate_answer(answer, q.schema):
            blind_spots = sorted(
                set(blind_spots) | {"the winning answer did not satisfy the question schema"}
            )
            answer = None

        return Verdict(
            question_id=q.id,
            answer=answer,
            confidence=confidence,
            consensus=consensus,
            votes=ordered,
            dissent=dissent,
            blind_spots=blind_spots,
            contradictions=contradictions,
            escalated=False,
            agreed_by=sorted(winner_names),
        )

    def _compose_answer(
        self, winners: Sequence[Vote], q: Question, confidence: float
    ) -> Optional[Dict[str, Any]]:
        """Build the verdict's answer object from the winning group.

        Only properties the schema declares are emitted, the confidence is the
        *verdict's* confidence rather than any single member's, and the reason codes are
        the union of the group's -- so the answer explains itself.
        """
        best = sorted(winners, key=_vote_sort_key)[0]
        field = answer_field(q.kind)
        value = best.answer.get(field)
        if isinstance(value, (list, tuple)):
            value = sorted({str(v) for v in value})
        if value is None:
            return None

        allowed = set((q.schema.get("properties") or {}).keys()) or {
            field,
            "confidence",
            "reason_codes",
        }
        answer: Dict[str, Any] = {}
        if field in allowed:
            answer[field] = value
        if "confidence" in allowed:
            answer["confidence"] = round(min(1.0, max(0.0, confidence)), 6)
        if "reason_codes" in allowed:
            codes = sorted({str(c) for v in winners for c in v.reason_codes})
            if codes:
                answer["reason_codes"] = codes
        return answer

    def _blind_spots(
        self, votes: Sequence[Vote], decisive: Mapping[str, Sequence[Vote]], q: Question
    ) -> List[str]:
        """Describe what no member addressed."""
        found: List[str] = []
        if not decisive:
            found.append("every member returned unknown")
            found.append(_KIND_BLIND_SPOT.get(q.kind, "no member answered the question"))
        if votes and all(v.error for v in votes):
            found.append("every member failed internally")
        else:
            failed = sorted(v.member for v in votes if v.error)
            if failed:
                found.append("no vote from failed member(s): %s" % ", ".join(failed))
        if votes and not any(v.reason_codes for v in votes):
            found.append("no member gave a reason code")
        if len(votes) < self.config.quorum:
            found.append(
                "quorum not met: %d of %d members voted" % (len(votes), self.config.quorum)
            )
        if q.kind in ("canonical_key", "ambiguity") and q.options:
            unaddressed = [o for o in q.options if o not in decisive]
            if len(unaddressed) == len(q.options) and decisive:
                found.append("no member voted for any option on the ballot")
        return sorted(set(found))

    def _contradictions(self, votes: Sequence[Vote], q: Question) -> List[str]:
        """Describe pairs of high-confidence votes that disagree."""
        threshold = self.config.agreement_threshold
        strong = [
            v
            for v in sorted(votes, key=lambda v: v.member)
            if v.confidence >= threshold and not v.is_unknown(q.kind)
        ]
        out: List[str] = []
        for index, left in enumerate(strong):
            for right in strong[index + 1 :]:
                if left.signature(q.kind) == right.signature(q.kind):
                    continue
                out.append(
                    "%s=%s@%.2f vs %s=%s@%.2f"
                    % (
                        left.member,
                        left.signature(q.kind),
                        left.confidence,
                        right.member,
                        right.signature(q.kind),
                        right.confidence,
                    )
                )
        return out


def build_default_council(config: ZfpConfig) -> Council:
    """Build the council ZFP uses by default.

    The four local members are always present -- they need no credential, no network and
    no optional dependency, which is what makes ``CouncilConfig.providers``
    (``rules``, ``heuristic``, ``ontology``) satisfiable on a bare stdlib install.
    ``SiblingConsensusMember`` joins them as the fourth local opinion.

    :class:`~zfp.council.members.OpenRouterMember` is added **only** when
    ``config.privacy.allow_external_inference`` is true *and* ``OPENROUTER_API_KEY`` is
    in the environment.  Either one missing means a purely local council, which is the
    default posture.  The model id can be pinned with ``ZFP_COUNCIL_MODEL``.
    """
    members: List[Any] = [
        HeuristicMember(),
        OntologyFuzzyMember(),
        RulesMember(),
        SiblingConsensusMember(),
    ]
    privacy = config.privacy
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if privacy.allow_external_inference and api_key:
        members.append(
            OpenRouterMember(
                os.environ.get("ZFP_COUNCIL_MODEL", DEFAULT_MODEL),
                api_key=api_key,
                privacy=privacy,
            )
        )
    return Council(members, config.council, privacy)
