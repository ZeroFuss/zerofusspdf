# The ZFP Agent Mesh

ZFP's processing model is a **deployment**, not a function call. A run deploys an
orchestrator, which deploys stages of specialized agents in parallel, which deploy
sub-agents per page shard, whose competing proposals are reconciled by facilitators and,
where they still disagree, settled by a council.

Every one of those roles is deterministic Python in `zfp.agents`. Only the *council* may
consult a model, and only for semantics — never for geometry.

## Roles

| Role | Class | What it does |
|---|---|---|
| **Orchestrator** | `zfp.agents.orchestrator.Orchestrator` | Owns the stage plan, the blackboard, the budget and the trace. Decides which stages run for this document class, deploys each stage, and produces the `RunReport`. |
| **Facilitator** | `zfp.agents.facilitator.Facilitator` | Sits between agents that produce *competing* answers about the same thing. Collects `Proposal`s, reconciles them into one value plus a conflict record, and flags what must escalate. |
| **Council** | `zfp.council.council.Council` | A quorum of independent members that vote on one structured question with a JSON schema. Produces a `Verdict` carrying consensus, dissent, contradictions and blind spots. |
| **Specialist** | `zfp.agents.specialists.*` | A single-purpose worker: preflight, native text, vector geometry, OCR, shapes, candidates, sections, labels, types, canonicalization, vault, validation, writing, appearance, integrity, diff, metrics. |
| **Sub-agent** | `zfp.agents.subagents.*` | A child deployed by a specialist to cover a slice of the work — a page shard, or a cropped region that came back low-confidence. |
| **Verifier** | integrity / render-diff / metrics agents | Adversarial: their job is to fail the run, not to bless it. |
| **Scribe** | `ReportAgent` | Renders the trace, the conflicts and the metrics into the run report. |

## Deployment plan

The default stage plan. Stages marked ∥ deploy their agents in parallel; stages marked ⇊
additionally fan out one sub-agent per page shard.

```
                          ORCHESTRATOR
                                │
  security-gate                 │  encrypted? signed? XFA? → policy decision, may stop here
  preflight            ⇊ ∥      │  per-page classification → DocumentProfile
  route                         │  DocumentClass → which downstream stages exist at all
                                │
        ┌───────────────────────┴────────────────────────┐
        ▼ native pages                                   ▼ raster pages
  native-extract  ⇊ ∥                             raster-ocr  ⇊ ∥
   ├ NativeTextAgent      (spans, glyph boxes)     ├ RasterRenderAgent
   └ VectorGeometryAgent  (lines, rects, circles)  ├ ScanPreprocessAgent  (deskew/denoise)
                                                   └ OcrAgent             (cascade + suspects)
        └───────────────────────┬────────────────────────┘
                                ▼
  shape-detect         ⇊ ∥   ShapeDetectionAgent · BlankRegionAgent
  candidate-generate   ⇊ ∥   eleven archetype detectors, deployed concurrently
                                ▼
  geometry-fuse                 GeometryFacilitator ◄── competing rects from vector/OCR/CV
                                ▼
  section-detect       ∥        SectionAgent
  label-link           ∥        LabelLinkAgent · PatternRuleAgent
  type-infer           ∥        TypeInferenceAgent
  canonicalize                  CanonicalizeAgent  → SemanticFacilitator
                                ▼
  ambiguity-council             AmbiguityTriageAgent → Council.deliberate_many()
                                ▼
  repeat-reconcile              RepeatConsistencyAgent (one inference, N instances agree)
  schema-build                  SchemaBuilderAgent → FormSchema
                                ▼
  vault-resolve        ∥        VaultResolveAgent → ValueFacilitator
  validate             ∥        NormalizeValidateAgent
                                ▼
  acroform-write                AcroFormWriteAgent  (incremental update)
  appearance                    AppearanceAgent
                                ▼
  integrity-verify     ∥        IntegrityAgent   ─┐
  render-diff          ∥        RenderDiffAgent   ├─ verifiers, adversarial
  metrics              ∥        MetricsAgent     ─┘
  report                        ReportAgent
```

`zfp agents` prints this tree for the current configuration, and `RunReport.trace` prints
what *actually* deployed for a given document, with per-agent timing and evidence counts.

## The blackboard

Agents never call each other. They read and write a thread-safe `Blackboard` under the keys
declared in `zfp.agents.keys`, and each agent declares `requires` / `produces`. That makes
the dependency graph inspectable, makes stages trivially parallel, and makes it impossible
for an agent to smuggle state past the facilitators.

Page-scoped values use `page_key(base, page)` so shards never collide.

## Facilitation, concretely

Three agents can each claim a rectangle for the same blank:

```
VectorGeometryAgent   rect=[601.0, 512.4, 1127.0, 526.4]  conf .99  evidence=vector_line
OcrAgent              rect=[598.2, 511.8, 1130.4, 527.1]  conf .71  evidence=ocr_text
ShapeDetectionAgent   rect=[600.5, 512.1, 1128.2, 526.9]  conf .84  evidence=blank_region
```

`GeometryFacilitator` does not average them. It **snaps to the strongest geometric
primitive** — the vector line, which is exact — records the other two as corroborating
evidence (raising `confidence.geometry`), logs the disagreement in the `ConflictLog`, and
only escalates when no primitive dominates. Averaging would produce a rectangle that matches
nothing on the page; snapping produces one that matches the ink.

`SemanticFacilitator` and `ValueFacilitator` play the same role for meaning and for values.

## The council

A council convenes **only** for a field the deterministic stages could not settle. Members
vote on one `Question` with a strict JSON schema; there is no free prose anywhere in the
protocol.

```
Candidate: [____________]     left label "Tax ID"   section "Employer"
                              document "Vendor Onboarding"

geometry engine   type=text, conf .998          ← already settled, not up for a vote
RulesMember       company.tax_id.ein   .995     (nearby placeholder ##-#######)
HeuristicMember   company.tax_id.ein   .94      (section=Employer)
OntologyFuzzy     company.tax_id.vat   .61
OpenRouterMember  company.tax_id.ein   .97      (only if egress is permitted)

analyst → consensus 0.75, dissent [vat], verdict company.tax_id.ein @ .993
```

Members that are always available are pure Python: ontology rules, layout heuristics, fuzzy
alias matching. The external model member is **off by default**, refuses to run when
`PrivacyConfig.allow_external_inference` is false, sends only redacted structural context
under `max_context_chars`, requires zero-data-retention routing and a provider allow-list,
and returns strict JSON-schema output or nothing.

Ties break deterministically on `(-confidence, member_name)`, so a run is reproducible.

## Sub-agents and sharding

`spawn_page_subagents` splits a document into shards of `page_shard_size` pages and deploys
one sub-agent per shard, per stage. A 400-page document is a scheduling problem, not an
unsupported case. `RegionSubAgent` re-examines a single low-confidence rectangle at higher
resolution rather than re-processing the page.

## Determinism

- Parallel results are re-sorted by `(page, task_id, agent_name)` before returning.
- `Trace` events are numbered by a monotonic counter, not a clock.
- No unseeded randomness anywhere in the mesh.

Two runs of the same document with the same config produce byte-identical output and an
identical trace. That is a requirement, not a nicety: it is what makes the QA metrics and the
render diff meaningful.
