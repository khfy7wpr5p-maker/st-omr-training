# TR-POLY-09B3 — Common VALIDATION Aggregation

Status: implementation package; sealed TEST and production promotion remain closed.

## Purpose

TR-POLY-09B3 aggregates the deterministic per-sample metric evidence produced by TR-POLY-09B2 for one exact checkpoint-bound candidate on one exact VALIDATION benchmark identity.

The package answers a limited question:

> On this frozen VALIDATION identity, where is this exact candidate strong or weak across voice complexity and admitted robustness buckets?

It does not run inference itself, open dataset files, access TEST, retrain a model, choose a winner, or grant ScoreMosaic/production authority.

## Input boundary

B3 accepts only `PolyV2SampleMetricReport` values that are already:

- `split=validation`;
- bound to one exact `benchmark_identity_sha256`;
- bound to one exact `candidate_identity_sha256`;
- produced from checkpoint-bound inference evidence;
- produced by the frozen B2 metric/adaptor, event-alignment and relation-metric versions.

The aggregator rejects:

- empty input;
- TRAIN or TEST evidence;
- unbound checkpoints;
- mixed benchmark identities;
- mixed candidate/checkpoint identities;
- duplicate sample IDs;
- mismatched B2 metric/adaptor versions;
- unexpected voice strata or robustness buckets.

## Aggregation policy

The first aggregation policy is frozen as:

`sample-macro-mean-v1`

For every numerically available metric, B3 reports:

- sample count;
- available count;
- unsupported count;
- mean;
- minimum;
- maximum.

No hidden weighting by family size, note count, token count or score length is introduced in this first common report.

This policy is intentionally simple and auditable. A future micro/pooled metric may be added only under a separate versioned policy.

## Invalid / abstain handling

Invalid or abstaining free-running model outputs remain in the sample population because B2 already records them rather than dropping them.

Therefore B3 does not create a survivorship-biased report.

Example:

```text
100 VALIDATION samples
12 strict-parse failures
88 parse-valid predictions

parse_success = 88 / 100 = 0.88
```

The 12 failures are not removed before semantic averages. Their B2-available semantic/relation metrics remain zero according to the B2 contract.

## Report slices

B3 always creates one overall VALIDATION slice and creates observed slices for the frozen voice strata:

- `1_voice`
- `2_voice`
- `3_voice`
- `4_plus_voice`

Any missing required voice stratum is recorded explicitly in `missing_voice_strata`.

Observed robustness buckets are reported separately from the frozen set:

- `clean`
- `scan`
- `phone`
- `blur`
- `perspective`
- `low_contrast`

An aggregate metric must never be the only evidence shown when per-stratum evidence exists.

## Metric availability

B3 preserves B2 availability exactly. It does not turn unsupported metrics into zeros or inferred values.

Current B2 numeric surface:

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

Current unsupported surface:

- `musicxml_validity`
- `tedn`
- `notehead_stem_f1`
- `beam_relation_f1`
- `tie_relation_f1`

For an unsupported metric the aggregate contains no numeric mean/min/max and preserves the explicit unsupported reason.

## Coverage and comparison gate

B3 exposes separate readiness checks:

- checkpoint-bound candidate identity;
- full required-metric coverage;
- required voice-stratum coverage.

`common_comparison_ready` is true only when all three are satisfied.

With the current B2 metric surface, this gate remains false because five frozen TR-POLY-02 metrics are still unsupported. This is deliberate: partial diagnostic evidence is useful, but it is not silently upgraded into a complete winner/promotion claim.

## Candidate comparison boundary

`validate_common_candidate_reports(...)` can validate multiple B3 candidate reports only when:

- every candidate passes the common-comparison gate;
- benchmark identity is exactly identical;
- VALIDATION sample IDs are exactly identical;
- candidate identities are distinct.

The helper validates comparability only. It does not rank candidates or select a winner.

## Deterministic evidence identity

The B3 report fingerprint binds:

- benchmark identity;
- candidate identity;
- exact sorted sample IDs;
- exact B2 sample-report fingerprints;
- overall slice;
- voice-stratum slices;
- robustness-bucket slices;
- missing voice strata;
- unsupported metric IDs;
- aggregation version/policy;
- B2 adapter/alignment/relation versions.

Input order does not change the report fingerprint because sample reports are sorted by sample ID before aggregation.

## Regression coverage

Tests cover:

- overall + voice + robustness aggregation;
- macro mean behavior;
- invalid-output denominator preservation;
- unsupported metric preservation;
- comparison-gate closure while metrics are unsupported;
- missing voice-stratum visibility;
- mixed candidate rejection;
- mixed benchmark rejection;
- duplicate sample rejection;
- unbound checkpoint rejection;
- input-order-independent report fingerprint.

## Safety / non-claims

TR-POLY-09B3 does not:

- open TEST;
- load TEST artifacts;
- train or mutate a model;
- change the checkpoint;
- fabricate unsupported metrics;
- hide invalid predictions;
- rank candidates;
- claim a benchmark winner;
- modify ScoreMosaic;
- grant production authority.

## Exit condition

A green B3 merge means the repository can deterministically summarize one exact checkpoint-bound candidate's existing B2 VALIDATION evidence by overall, voice-complexity and robustness slices.

It does **not** mean real benchmark numbers have already been produced. Real numeric evidence still requires execution of the frozen checkpoint against the admitted native V2 VALIDATION artifacts outside ordinary Git content.

## Next execution gate

After B3 is green on protected `main`:

1. freeze the exact VALIDATION manifest/build identity and exact checkpoint artifact;
2. run B1 free-running inference on every admitted VALIDATION sample;
3. produce one B2 report per sample;
4. aggregate them through B3;
5. inspect pitch/duration/onset/voice/staff and robustness strata;
6. use only that evidence to choose P09C refinement work;
7. keep TEST sealed.

The five unsupported metric surfaces remain separate admission work before any complete TR-POLY-02 comparison/promotion claim.
