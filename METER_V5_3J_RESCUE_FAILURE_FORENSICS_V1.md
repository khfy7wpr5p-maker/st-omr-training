# Meter V5-3J — Rescue Failure Forensics V1

## Purpose

V5-3J is a TRAIN-only, read-only forensic stage opened because V5-3I returned `HOLD`.

Bound V5-3I evidence:

- V5-3I implementation HEAD: `88c7acc551fa2b00b1f877f6a839704d58825adb`
- V5-3I module blob: `abb5f1ae4c42b0c5f3ae26b80f2a467f47582197`
- V5-3I report SHA-256: `448b807086bc9ee66d090fdf173ce54e3c5e2a133e60cf6ae0a791aed2717434`
- decision: `HOLD`

The observed HOLD witness is fixed:

| specialist | V5 TRAIN F1 | V5 FP | V5 FN | V5 frozen-correct regressions | historical TRAIN frozen-correct regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2-AI | 1.0 | 0 | 0 | 0 | 5307 |
| 3-AI | 0.0 | 0 | 90 | 0 | 15775 |

## Question

V5-3J answers only:

> Why did the fixed V5-3G rescue artifacts fail the V5-3I TRAIN acceptance gate?

It does not choose a repair recipe.

## Forensic measurements

For 2-AI and 3-AI separately, and only on V5 adaptation TRAIN plus historical TRAIN, V5-3J records:

- exact frozen-negative eligible positive/negative counts;
- rescue probabilities at the unchanged rescue threshold `0.50`;
- correction counts for frozen false negatives;
- regression counts for frozen true negatives;
- fixed probability quantiles (`p01` through `p99`);
- mean and median positive-minus-negative score gaps;
- positive-over-negative pairwise rank fraction inside each TRAIN domain;
- V5-positive-over-historical-negative cross-domain rank fraction;
- deterministic failure signatures describing V5 recovery versus historical-TN collapse.

These are descriptive measurements only. No threshold is selected from the diagnostics.

## Safety boundary

V5-3J performs no training, backward pass, optimizer step, checkpoint write, rescue artifact write, threshold tuning, threshold sweep, hyperparameter sweep, architecture change, automatic second configuration, runtime wiring, resolver wiring, or production promotion.

The following surfaces remain closed:

- Historical Validation
- immutable First-30
- V5 reserve
- V5 validation
- FINAL_HOLDOUT

4-AI is not loaded. Original frozen specialists and rescue artifacts are fingerprinted before and after forensics and must remain bit-identical.

## Failure signatures

The diagnostic may emit one of these descriptive signatures:

- `V5_RECOVERED_HISTORICAL_TN_COLLAPSE`
- `V5_POSITIVE_NOT_RECOVERED_HISTORICAL_TN_COLLAPSE`
- `PARTIAL_V5_RECOVERY_HISTORICAL_TN_COLLAPSE`
- `V5_RECOVERY_FAILURE_WITHOUT_HISTORICAL_TN_COLLAPSE`
- `NO_FAILURE_SIGNATURE`

A signature is evidence, not authorization.

## Next gate

V5-3J cannot authorize retraining. After a completed forensic report, a separate stage may preregister one repair hypothesis only if the evidence supports it. Any later training execution requires separate authorization and a new TRAIN acceptance gate before Historical Validation can become eligible.
