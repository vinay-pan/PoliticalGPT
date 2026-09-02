# Dev Log

Lab notebook for the eventual paper — decisions, rejected alternatives, corpus/data
stats, surprising or negative results, appended at every phase/decision checkpoint
(DEVL-02). Append-only: entries are never edited or deleted after being committed,
only added to.

Started: 2026-09-01

## Preregistered Analysis Plan

**Registered:** 2026-09-01, before any generation exists — no model has been loaded
and no `.generate()` call has occurred anywhere in this project as of this entry.
This preregistration is recorded first specifically so that no later decision in
this document can be read as having been informed by output it wasn't.

### Measurement axes

- **A1 — NeuS calibrated arousal.** Scored against the base-model neutral reference
  (the strongest available metric per the 2026-08-26 research pass: human-validated
  on machine-generated text, ρ=0.636 vs. human judgment).
- **A2 — BASIL lexical/informational bias span schema.** Target entity, polarity,
  aim, in-quote — scored via LLM-jury + adjudication, reusing BASIL's schema, not
  its own weak classifier (MEAS-01/02/03).
- **A3 — Sap et al. 2017 power/agency connotation frames.** Toward the target
  politician (MEAS-01/02/03).

A5 / Media Frames Corpus (MEAS-V2-03) is **kept viable, not activated**, for v1. D-01
deliberately keeps a policy-area slot in the mad-libs grammar (now the committed
`prompts/grammar.yaml`, with 6 real policy areas — healthcare, immigration, trade,
climate_energy, education, criminal_justice) specifically so this axis remains
reachable if a v2 trigger warrants it, rather than foreclosing it by omitting
policy content from the prompt design.

### λ grid

Coherence-sweep grid first (Phase 4/5, to bound the usable λ range before any
headline result is computed). Headline-result grid = {0, 1} (Phase 5-6, per
GEN-01/MEAS-08) — the two pure-adapter endpoints. The full dose-response grid
across intermediate λ values is fixed once the coherence sweep has bounded the
usable range; it is not fixed now because fixing it before that sweep exists would
mean guessing at values the coherence data hasn't justified yet.

### Prompt families

The `prompts/grammar.yaml` cross-product — politician × event_type × setting ×
policy_area × outcome — as committed in Plan 01-02: **5 politicians × 6 event
types × 5 settings × 6 policy areas × 6 outcomes = 5,400 total prompt cells**
(verified via `grammar.enumerate_all()` against the committed file). Prompt cells
are split into a pilot set (touched freely during pipeline development and
debugging) and a held-out confirmatory set (touched once, for reported numbers) —
the exact split mechanism and confirmatory-set size are Phase 5 decisions, made
once the pipeline is otherwise working, not now.

### Positive-result definition

**D-05:** A positive result per axis, before multiplicity correction, is: the
aggregate/bootstrapped confidence interval for that axis's divergence statistic
excludes zero. No separate minimum-effect-size threshold is preregistered —
CI-excludes-zero is the sole bar. This is consistent with MEAS-07's aggregate-only,
CI-based reporting requirement, and avoids preregistering an arbitrary effect-size
cutoff this project has no prior basis to set.

### Multiplicity handling

**D-06:** Multiple-comparison correction across the 3 axes (A1 NeuS, A2 BASIL, A3
connotation frames) uses Benjamini-Hochberg FDR
(`statsmodels.stats.multitest.multipletests(method='fdr_bh')`), **not Bonferroni**.
The three axes are correlated — all of them measure some facet of "framing" on the
same underlying generated pairs — so a family-wise-error correction designed for
independent tests (Bonferroni) would be needlessly conservative; FDR is the
better-justified choice for correlated tests.

### Sample size

**D-07:** The preregistered plan commits now to a *method* for sample size, not a
fixed N: a power-style analysis (via `statsmodels.stats.power`, or a
simulation-based approach if the divergence statistic's sampling distribution
isn't analytically tractable — concrete tool choice deferred, Claude's discretion)
runs once Phase 5 pilot-sweep variance exists, and that analysis fixes the final N
for the headline result. No arbitrary N (e.g. "~200 pairs") is locked in now,
because Phase 1 has no variance estimate to base one on.

### What is NOT preregistered

Named explicitly here so this preregistration cannot later be read as either
incomplete or as hiding researcher-degrees-of-freedom:

- **Exact wording of grammar categories** — Claude's discretion (CONTEXT.md).
- **Power-analysis tool choice** — deferred to Phase 5, once pilot-sweep variance
  exists to choose against (D-07).
- **Whether A5/MFC gets activated as an actual scored axis** — kept viable by D-01,
  not committed to for v1.

## Decision Checkpoint: Base-model and dependency pins

**2026-09-01.** `configs/base_model.yaml` pins `meta-llama/Llama-3.1-8B-Instruct` at
resolved commit SHA `0e9e39f249a16976918f6564b8830bc894c89659` — a real, live-resolved
value (via `HfApi().model_info(...).sha`), not a placeholder or the mutable branch
name `main`. Both outlet adapters (Phase 4) and the dial loader (Phase 2/5) read
this single file rather than each hardcoding their own copy, so an asymmetric pin
between the two adapters can't silently happen.

`pyproject.toml` pins `transformers==5.15.1` and `trl==1.10.0` — the versions
STACK.md's research pass verified by actual installation and execution — **rather
than** the current PyPI latest (5.16.1/1.12.0) at time of writing, since "pin what
was tested" beats "pin what's newest" for a project whose numerics (the dial's
interpolation math) depend on stable library behavior.

The CUDA-specific `torch` build/index binding is deliberately deferred to Phase 4
**rather than** baking a `pytorch-cu124`-style index into Phase 1's
`pyproject.toml` now: the current dev machine is macOS/arm64 with no CUDA GPU, and
the target RunPod image's actual CUDA version isn't chosen yet, so binding to a
specific CUDA index now would be a guess, not a decision.

## Decision Checkpoint: Mad-libs grammar design

**2026-09-01.** `prompts/grammar.yaml`'s 6th event-type category is
`legislative_vote` ("{politician} pushes for a vote on {policy_area} legislation")
— chosen because it is distinct from `policy_announcement` in centering a formal
vote/procedural outcome **rather than** a unilateral announcement, giving the
6-category set genuine variety instead of two near-duplicate announcement-shaped
categories.

Now that the actual `policy_areas` list exists (healthcare, immigration, trade,
climate_energy, education, criminal_justice — 6 entries, real policy content, not
generic filler), D-01's premise holds concretely: the deferred A5/Media Frames
Corpus axis remains a viable v2 activation target, since the grammar already
carries the policy-area dimension MFC's frame schema would need. No per-(politician,
event_type) restriction map was added anywhere in the file — every politician is
usable with every event type, per D-04.

## Decision Checkpoint: Prior-art sweep outcome

**2026-09-01.** `RELATED_WORK.md`'s Media Frames Corpus forward-citation sweep hit
a persistent Semantic Scholar HTTP 429 across 4 attempts (initial + 10s/30s/60s
backoff) and remains **blocked, not completed** — documented explicitly with
attempt count, timestamps, and status, rather than silently treated as a completed
"no collision" result. The resulting standing novelty verdict — no single paper
found doing the full combination of outlet-conditioned LoRA adapters, runtime
interpolation, counterfactual invented-event prompts, and literature-grounded
framing measurement — carries a stated confidence of **MEDIUM-HIGH, not HIGH**.

This is explicitly **not** being treated as a final/resolved verdict on the
strength of a single research pass (per 01-RESEARCH.md Pitfall 1 and this
project's own research-rigor constraint). What changed today versus the
2026-08-26 abstract-level pass: 2 full paper reads instead of abstract-only
(arXiv:2606.08792, arXiv:2503.02080), and 5-of-5 D-10 anchor citation trees
checked (up from 4-of-5) — but the MFC anchor's own citation tree is still
unswept via the Graph API, and D-10's broader topic queries were not
exhaustively triaged. The open item (retry MFC sweep with a registered API key
or longer cooldown before any external write-up) is carried forward in
`RELATED_WORK.md` itself, not just here.

## Decision Checkpoint: Correction — arXiv:2503.02080 full read

**2026-09-02.** The "Prior-art sweep outcome" entry above, logged 2026-09-01,
claimed "2 full paper reads" for the prior-art sweep. Independent verification
(this project's phase-verifier, run against the actual committed files rather
than trusting the plan summaries) found that claim was an overclaim: the
2026-09-01 read of arXiv:2503.02080 was actually abstract-plus-metadata only —
its PDF fetch had failed that day — and the shortfall was masked at the summary
level in both `RELATED_WORK.md` and the entry above, exactly the failure mode
01-RESEARCH.md's own Pitfall 1 warns against (treating a single-pass claim as
settled without verification).

The PDF was confirmed reachable today (`curl -sI` returned HTTP 200 — **not** a
persistent block like the Media Frames Corpus 429) and was read in full via
direct download **rather than** a tool-mediated fetch, surfacing genuine
methodological detail not previously captured: specific attention-head indices
(e.g. layer 15/head 18 in Llama-2-7b-chat), the exact cross-validated
correlation values (ρ up to 0.861 for lawmaker ideology, up to 0.798 for
outlet-slant transfer), the inference-time steering equation, and a
layer-localization robustness finding (steering effective in layers <22,
ineffective in layers ≥22). `RELATED_WORK.md`'s entry for this paper has been
rewritten with this detail; **"2 full paper reads" is now an accurate claim**,
not an overclaim, and the standing novelty verdict's MEDIUM-HIGH confidence now
rests solely on the still-unresolved MFC sweep gap, not on any residual
abstract-only read.

Per this file's own append-only convention, the original 2026-09-01 entry above
is left unedited; this entry is the correction, not a silent amendment.

## Decision Checkpoint: Release policy

**2026-09-01.** **D-08:** This project stays private/research-only for now — no
commitment to public release of code, corpus metadata, or adapters at this stage.
This is a revisitable decision, not a permanent one; it will be revisited once
results exist and a venue (if any) is targeted, **rather than** decided once and
never reopened. Stated here as an explicit assumption, not a silent default, per
D-08's own instruction.

**D-09:** Trained adapter weights may live on a private, authenticated cloud GPU
training service (e.g. RunPod) during and after training — **not** restricted to
local-machine-only storage. Nothing gets pushed to any public host (no public
GitHub push of adapter weights, no public Hugging Face Hub upload) until the
release-policy decision in D-08 is revisited.
