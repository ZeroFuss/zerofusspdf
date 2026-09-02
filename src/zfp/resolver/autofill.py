"""The autofill resolver: field + vault + ontology -> a validated, provenance-tracked value.

"ZFP should not ask a generative model to invent answers." Every value placed into a
field is traced back through a deterministic resolution cascade to something the vault
actually holds (or can honestly derive); a field the vault cannot answer is reported as
unavailable, never guessed.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from ..core.config import ZfpConfig
from ..core.types import FieldCandidate, FieldSpec, FieldType, FillReport, FilledValue, FormSchema
from ..ontology import get as ontology_get
from ..ontology import lookup as ontology_lookup
from . import normalizers, validators
from .policy import FillPolicy, SigningPolicy, check_fill_allowed


class AutofillResolver:
    """Resolves every field in a schema against a vault, normalizes, validates, and
    records exactly why each value was (or was not) written."""

    def __init__(self, vault: Any, config: ZfpConfig, council: Optional[Any] = None,
                *, fill_policy: Optional[FillPolicy] = None,
                signing_policy: Optional[SigningPolicy] = None) -> None:
        self.vault = vault
        self.config = config
        self.council = council
        self.fill_policy = fill_policy or FillPolicy()
        self.signing_policy = signing_policy

    # -- single field ---------------------------------------------------------------------

    def resolve_field(self, spec: FieldSpec,
                      candidate: Optional[FieldCandidate] = None) -> FilledValue:
        autofill_cfg = self.config.autofill
        parent_context = tuple(getattr(candidate, "parent_context", ()) or ())
        reason_codes: List[str] = []

        if spec.field_type is FieldType.SIGNATURE:
            # A signature field is a policy boundary before it is anything else: ZFP
            # never fabricates or auto-applies a signature, so this is checked ahead of
            # vault lookup rather than only when a value happens to be available.
            allowed, reason = check_fill_allowed(
                spec, spec.canonical_key, None, self.fill_policy,
                signing_policy=self.signing_policy)
            if not allowed:
                return FilledValue(field_name=spec.name, canonical_key=spec.canonical_key,
                                   value=None, confidence=0.0, status="policy_blocked",
                                   reason_codes=[reason])

        if autofill_cfg.mode == "off":
            return FilledValue(field_name=spec.name, canonical_key=spec.canonical_key,
                               value=None, confidence=0.0, status="unavailable",
                               reason_codes=["autofill_disabled"])

        key = spec.canonical_key
        entry = None
        step = None

        if key:
            entry = self.vault.resolve(key, parent_context)
            step = "canonical_key" if entry is not None else None

        if entry is None and not key and getattr(candidate, "visible_label", None):
            guessed = ontology_lookup(candidate.visible_label)
            if guessed:
                key = guessed
                entry = self.vault.resolve(key, parent_context)
                step = "alias_from_label" if entry is not None else None

        if entry is None and key:
            # Explicit alias-of-key pass: a namespace child the vault might hold under a
            # slightly different key shape (handled by ProfileVault.resolve's own
            # derivation cascade already, but recorded here for provenance clarity).
            entry = self.vault.resolve(key, parent_context)
            if entry is not None:
                step = step or "derived"

        if entry is None:
            escalated = self._maybe_escalate(spec, candidate, key)
            if escalated is not None:
                return escalated
            return FilledValue(field_name=spec.name, canonical_key=key, value=None,
                               confidence=0.0, status="unavailable",
                               reason_codes=["no_vault_entry"] + ([f"tried:{key}"] if key else []))

        allowed, reason = check_fill_allowed(
            spec, key, entry, self.fill_policy, signing_policy=self.signing_policy)
        if not allowed:
            return FilledValue(field_name=spec.name, canonical_key=key, value=None,
                               confidence=0.0, status="policy_blocked", reason_codes=[reason])

        key_spec = ontology_get(key) if key else None
        raw_value = entry.value
        normalized = normalizers.normalize(raw_value, key_spec) if key_spec else raw_value

        outcome = validators.validate(normalized, key_spec) if key_spec else \
            validators.ValidationOutcome(True)
        if outcome.normalized:
            normalized = outcome.normalized
        if not outcome.ok:
            return FilledValue(field_name=spec.name, canonical_key=key, value=None,
                               confidence=0.0, status="invalid",
                               reason_codes=["validation_failed", outcome.message])

        constraint_outcome = validators.validate_against_constraints(normalized, spec)
        if not constraint_outcome.ok:
            return FilledValue(field_name=spec.name, canonical_key=key, value=None,
                               confidence=entry.confidence, status="invalid",
                               reason_codes=["constraint_failed", constraint_outcome.message])

        threshold = (autofill_cfg.min_fill_confidence if autofill_cfg.mode == "conservative"
                    else autofill_cfg.min_completion_confidence)
        confidence = entry.confidence

        provenance = {
            "source": entry.source, "verified_at": entry.verified_at,
            "confidence": entry.confidence, "vault_key": key, "resolution_step": step,
        }

        if confidence < threshold:
            if autofill_cfg.mode == "conservative":
                return FilledValue(
                    field_name=spec.name, canonical_key=key, value=None,
                    confidence=confidence, status="low_confidence",
                    reason_codes=["below_conservative_threshold"],
                    provenance={**provenance, "withheld_value_key": key},
                )
            # completion mode: still below threshold -> low_confidence, but caller may
            # choose to accept it; we do not silently upgrade past the configured floor.
            return FilledValue(field_name=spec.name, canonical_key=key, value=None,
                               confidence=confidence, status="low_confidence",
                               reason_codes=["below_completion_threshold"],
                               provenance=provenance)

        return FilledValue(field_name=spec.name, canonical_key=key, value=normalized,
                           confidence=confidence, status="filled",
                           reason_codes=reason_codes or [step or "resolved"],
                           provenance=provenance)

    def _maybe_escalate(self, spec: FieldSpec, candidate: Optional[FieldCandidate],
                        key: Optional[str]) -> Optional[FilledValue]:
        if self.council is None or not getattr(self.config.council, "enabled", False):
            return None
        # The council resolves MEANING (which canonical key), not values it does not
        # have -- so escalation here can only ever improve `key`, never manufacture a
        # value. Left as an extension point; the deterministic cascade above is
        # authoritative for value resolution.
        return None

    # -- whole schema ---------------------------------------------------------------------

    def resolve_schema(self, schema: FormSchema,
                       candidates: Optional[Sequence[FieldCandidate]] = None) -> FillReport:
        by_name = {}
        if candidates:
            for c in candidates:
                by_name.setdefault(getattr(c, "canonical_key", None), []).append(c)

        candidate_by_field = {}
        if candidates:
            cand_list = list(candidates)
            for spec, cand in zip(schema.fields, cand_list):
                candidate_by_field[spec.name] = cand

        report = FillReport(document_id=schema.document_id)
        for spec in schema.fields:
            cand = candidate_by_field.get(spec.name)
            report.values.append(self.resolve_field(spec, cand))
        report.recount()

        if self.config.autofill.propagate_repeats:
            self._propagate_repeats(schema, report)

        return report

    def _propagate_repeats(self, schema: FormSchema, report: FillReport) -> None:
        try:
            from ..semantics.repeats import check_consistency, find_repeated_fields, propagate
        except Exception:  # noqa: BLE001 - semantics module not present -> no-op
            return

        candidates = list(getattr(schema, "source_candidates", []) or [])
        if not candidates:
            return
        groups = find_repeated_fields(candidates)
        if not groups:
            return
        try:
            report.values = propagate(groups, report.values)
        except Exception:  # noqa: BLE001 - defensive: propagation must never crash a run
            return
        report.recount()
        try:
            inconsistencies = check_consistency(groups, report.values)
        except Exception:  # noqa: BLE001
            inconsistencies = []
        if inconsistencies:
            report.values.append(FilledValue(
                field_name="__repeat_consistency__", canonical_key=None, value=None,
                confidence=0.0, status="unavailable", reason_codes=list(inconsistencies)))

    def summary_by_status(self, report: FillReport) -> Mapping[str, int]:
        out: dict = {}
        for v in report.values:
            out[v.status] = out.get(v.status, 0) + 1
        return out


__all__ = ["AutofillResolver"]
