# ST-OMR Training — Polyphonic V2 Current Architecture and Roadmap

Updated: 2026-09-14

This document separates implemented capability from measured quality and preserves the dependency order toward a defensible Polyphonic V2 model.

## Current chain

Protected `main` baseline:

`79c2631682ebdb3b2be30c146a191e3ef8183ad0`

```text
TR-POLY-02 evaluation taxonomy / benchmark identity       ✅
TR-POLY-03 external dataset + license registry            ✅
TR-POLY-04 deterministic external benchmark harness       ✅
TR-POLY-05 Polyphonic Representation V2                   ✅
TR-POLY-06 V2 parser/tokenizer/lossless roundtrip         ✅
TR-POLY-07 research model registry                         ✅
TR-POLY-08 tiny 2D Transformer                            ✅ RESEARCH
TR-POLY-08A bounded trainer                               ✅ SMOKE ONLY (≤2 steps)
TR-POLY-08B exact checkpoint persistence/reload            ✅
TR-POLY-08C exact Stage 6 V1→V2 execution                 ✅
TR-POLY-09A native explicit V2 materialization            ✅
TR-POLY-09B1 free-running greedy inference                ✅ MERGED
TR-POLY-09B2 deterministic metric/adaptor layer           ✅ MERGED
TR-POLY-09B3 deterministic VALIDATION aggregation         ✅ IMPLEMENTED / MERGE GATE
smoke-checkpoint B1→B2→B3 sanity baseline                 🔄 OPTIONAL
quality-training regime                                   🔒 REQUIRED
first quality-trained candidate                           🔒
quality VALIDATION benchmark                              🔒
P09C evidence-driven refinement                           🔒
missing metric admission                                  🔒 5 surfaces
P09D final candidate freeze                               🔒
Stage 9 sealed TEST decision                              🔒
Stage 10 ScoreMosaic shadow integration                   🔒
```

## Measurement architecture

```text
native V2 VALIDATION image + canonical target
        ↓
exact verified checkpoint
        ↓
B1 free-running greedy prediction
        ↓
strict V2 parse OR explicit invalid/abstain
        ↓
B2 deterministic per-sample metrics
        ↓
B3 deterministic candidate aggregation
        ├─ overall
        ├─ 1 / 2 / 3 / 4+ voice
        └─ robustness buckets
```

B1 removes teacher forcing. B2 provides per-sample metric evidence. B3 makes those reports deterministic and inspectable without hiding failure cases.

## B3 aggregation contract

B3 accepts only B2 reports that are:

- VALIDATION-only;
- checkpoint-bound;
- from one exact benchmark identity;
- from one exact candidate identity;
- from the same versioned B2 metric/alignment/relation contract.

It rejects mixed benchmark identities, mixed candidate identities, duplicate sample IDs, unbound checkpoints and unexpected strata/buckets.

The first aggregation policy is `sample-macro-mean-v1`. Every available metric is averaged equally over samples and reports mean/min/max plus coverage counts. A later pooled/micro policy must be separately versioned.

## Failure visibility

Free-running invalid/abstain outputs remain in the B3 population because B2 emits explicit sample reports for them.

```text
parse-invalid sample
        ≠ dropped sample
        = visible failed sample in aggregate evidence
```

This prevents survivorship bias.

## Voice and robustness strata

Required voice coverage:

- `1_voice`
- `2_voice`
- `3_voice`
- `4_plus_voice`

Missing required voice strata are explicit and keep the common-comparison gate closed.

Observed robustness buckets remain separate:

- `clean`
- `scan`
- `phone`
- `blur`
- `perspective`
- `low_contrast`

## Current metric coverage

### Available numeric metrics — 11

- `parse_success`
- `ter`
- `normalized_edit_distance`
- `exact_sequence_accuracy`
- `pitch_accuracy`
- `duration_accuracy`
- `onset_accuracy`
- `voice_accuracy`
- `staff_accuracy`
- `accidental_note_f1`
- `note_staff_f1`

### Unsupported frozen metrics — 5

- `musicxml_validity`
- `tedn`
- `notehead_stem_f1`
- `beam_relation_f1`
- `tie_relation_f1`

B3 propagates these as unsupported with explicit reasons and no numeric proxy.

## Critical training-readiness boundary

The current Polyphonic 2D trainer is intentionally limited by `MAX_POLY_2D_SMOKE_STEPS = 2`. This means the training/checkpoint chain is an infrastructure proof, not a converged training program.

The existing smoke checkpoint can still be valuable for one purpose: verifying that the complete image → inference → metric → aggregation path works end to end.

It cannot support claims such as:

- expected architecture accuracy;
- meaningful voice-separation quality;
- production readiness;
- comparison against mature OMR systems.

Before model-quality measurement, a separately versioned quality-training regime must be implemented with:

- TRAIN-only parameter updates;
- meaningful bounded step/epoch budget;
- deterministic batch/order policy;
- checkpoint cadence and exact checkpoint selection;
- read-only VALIDATION selection/evaluation;
- early-stop or fixed-budget policy defined before execution;
- resume/restart provenance;
- no TEST access;
- exact training-recipe fingerprint.

Only after that regime produces a frozen candidate checkpoint should B1→B2→B3 numbers be interpreted as model-quality evidence.

## Common comparison gate

B3 validates comparability but does not rank candidates.

A report becomes `common_comparison_ready` only when:

1. candidate evidence is checkpoint-bound;
2. all required voice strata are represented;
3. every frozen TR-POLY-02 metric is numerically admitted.

Multiple candidates additionally require identical benchmark identity and exact VALIDATION sample IDs.

Today this gate remains closed because five metrics are unsupported. Partial diagnostic VALIDATION evidence remains useful for P09C once a quality-trained candidate exists.

## Recommended execution order

```text
B3 exact-head CI + merge
        ↓
optional current smoke checkpoint sanity run
        ↓
quality-training contract + implementation
        ↓
train first native-V2 candidate on TRAIN
        ↓
freeze exact checkpoint + recipe identity
        ↓
quality VALIDATION B1→B2→B3
        ↓
P09C evidence-driven refinement
        ↓
missing metric admission / complete comparison surface
        ↓
P09D candidate + evaluation recipe freeze
        ↓
Stage 9 one-shot sealed TEST decision
        ↓
Stage 10 independent ScoreMosaic shadow gate
```

## P09C decision policy

Use measured quality evidence rather than model-size intuition:

```text
high parse failure       → decoder/search diagnosis
high TER, valid parses   → sequence modeling/search diagnosis
weak pitch               → visual/pitch evidence diagnosis
weak duration/onset      → rhythm modeling/data diagnosis
weak 2+/3+/4+ voice      → polyphonic separation/data diagnosis
weak note_staff_f1       → staff/cross-staff diagnosis
strong clean, weak scan  → robustness/domain coverage diagnosis
```

Only demonstrated failure families should justify model, data or search expansion.

## Missing-metric admission

Before a complete TR-POLY-02 winner/promotion claim, separately implement and review:

1. deterministic V2→MusicXML export plus independent validation;
2. exact notehead-stem relation representation/adapter if retained as required;
3. explicit cross-event beam relation representation/adapter;
4. explicit cross-event tie relation representation/adapter;
5. exact TEDn algorithm/dependency/license/version surface.

## Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN is the only parameter-updating split.
- VALIDATION is read-only evidence.
- Smoke checkpoint evidence cannot be relabeled as quality-trained evidence.
- Unsupported metrics cannot receive proxy values under frozen IDs.
- Invalid/abstain outputs remain visible in denominators.
- Different benchmark/candidate identities cannot be mixed.
- External data requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- Benchmark evidence never grants automatic production authority.

## Immediate next action

Finish TR-POLY-09B3 exact-head CI/merge. Then establish the quality-training contract and execution path before treating VALIDATION metrics as model-quality evidence. A smoke-checkpoint B1→B2→B3 run is optional and may be used only as a pipeline sanity baseline. TEST remains sealed.
