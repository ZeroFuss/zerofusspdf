# Privacy architecture

ZFP is **local-first**. Every stage of the pipeline — preflight, perception, geometry,
semantics, resolution, writing, verification — runs on the machine holding the document. The
only stage that can ever leave the machine is *semantic escalation*, and it is off by default.

## The privacy cascade

```
local rules
    ↓  unresolved
local alias / fuzzy ontology
    ↓  unresolved
local layout + section heuristics
    ↓  unresolved
local vision / OCR
    ↓  unresolved semantics only
external model  ── requires: allow_external_inference = True
                             provider on the allow-list
                             zero-data-retention routing
                             redacted, truncated context
```

`PrivacyConfig.allow_external_inference` defaults to **False**. With that default, the
`OpenRouterMember` refuses to vote and raises `PolicyError` if invoked directly; the council
still reaches a verdict from its always-local members.

## What is sent, when egress is permitted

Only what a semantic question needs:

- the visible label, the section/parent context, the document title
- the *shape* of the blank (rect dimensions, comb cell count, placeholder pattern)
- the candidate canonical keys under consideration

And never:

- the document itself, or a page image, or a page's full text
- any value from the profile vault
- any value already filled into another field

`zfp.council.redaction.redact_context` enforces this: it strips values, truncates to
`max_context_chars`, and — when `redact_values_in_prompts` is set — replaces digit runs with
`#` so that a placeholder's *shape* survives while its content does not.

## Vault

The profile vault is the system of record for identity data. It is:

- **encrypted at rest** with ChaCha20-Poly1305 under a scrypt-derived key (pure Python,
  no third-party crypto dependency), when a password is supplied;
- **provenance-bearing** — every entry records where it came from, when it was verified, and
  with what confidence, so a fill can always be explained;
- **sensitivity-tagged** — `normal` / `pii` / `secret`. Secret-class values (SSN, card
  number, CVV, account number) are never included in any council context under any
  configuration, and are redacted from logs and traces.

## Autonomy boundaries

"Zero touch" must never mean "silently hallucinate".

| Mode | Behaviour |
|---|---|
| `conservative` (default) | Fills only values clearing the confidence and validation thresholds. Everything else is reported as **data unavailable** — with no manual box placement required from the user. |
| `completion` | Uses model consensus and best-evidence resolution to fill every resolvable field, attaching confidence and provenance internally. |
| `off` | Detection and field creation only; no values written. |

A field asking for a fact the system does not possess — "Reason for requesting this permit" —
is categorically different from a field it can resolve. ZFP finds the field either way, and
declines to invent the answer.

## Signatures

A signature field is a **policy boundary**, not a text field. ZFP will identify signature,
initial, date and title fields and prepopulate authorized identity metadata around them, but
applying a legally meaningful signature requires an explicit `SigningPolicy`. Without one,
signature fields are created and left with status `policy_blocked`.

## Documents ZFP will not quietly modify

- **Encrypted**: honoured, not bypassed. Removing protection requires the owner password,
  and that is a credential-gated operation, never a detection problem.
- **Signed**: the signed revision is preserved. Adding fields would invalidate it, so ZFP
  refuses unless an authorized workflow deliberately creates a new revision.
