# Meter Real-Domain Retention v3 Contract

## Purpose

This is a **shadow-only, fail-closed** follow-up to the completed 20-epoch Meter real-domain adaptation V2 run. It does not weaken any acceptance gate, open sealed TEST, connect runtime/Resolver, replace D11, or authorize production promotion.

The V2 run proved two things simultaneously:

1. the frozen-D11 + small glyph-adapter path can learn the real Teacher Gold domain;
2. the current constant learning-rate trajectory does not reach a checkpoint that satisfies real-domain accuracy and synthetic-retention gates at the same epoch.

V3 changes exactly one training mechanism: the learning-rate schedule. All data, model surfaces, loss weights, replay cardinalities, split identities, D11 identity, and acceptance gates remain frozen to V2.

## Accepted V2 evidence

Authoritative completed shadow run:

- run directory: `meter-real-domain-background-v3-run`;
- D11 checkpoint SHA-256: `cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3`;
- Teacher Gold: 72 records, family-disjoint train/validation, TEST unopened;
- D10 validation: 1,224 Meter records, TEST unopened;
- epochs: 20;
- optimizer steps: 600;
- result: `HOLD_NO_ACCEPTED_CANDIDATE`;
- no candidate checkpoint emitted.

The V2 synthetic baseline was:

- macro F1: `0.9089455720000736`;
- positive localization F1@2px: `0.7105245746065033`.

The best real epoch was epoch 13:

- real accuracy: `16/18 = 0.8888888888888888`;
- real macro F1: `0.8375`;
- real recall: `none=9/9`, `2/4=2/3`, `3/4=3/3`, `4/4=2/3`;
- the remaining `2/4` and `4/4` errors were both predicted as `3/4`;
- synthetic macro F1: `0.8739820582836907`;
- synthetic localization F1@2px: `0.6649182194961449`.

V2 also demonstrated that synthetic retention itself is reachable with the unchanged model/replay path. Epoch 10 satisfied both synthetic-retention limits, and epoch 20 again satisfied both synthetic-retention limits, while the real-domain score had already moved away from its epoch-13 maximum. Therefore V3 treats the observed failure as a joint optimization/stability problem, not as evidence that replay is incapable of retaining D10.

## Single-variable intervention

The V2 learning rate is constant at `1000` micro-units (`0.001`) for all 20 epochs.

V3 freezes the following deterministic midpoint schedule:

| Epoch | Learning rate |
| --- | ---: |
| 1–10 | `1000` micro-units = `0.001` |
| 11–20 | `250` micro-units = `0.00025` |

The schedule is epoch-indexed only. It must not inspect validation metrics, gate reasons, class recalls, losses, or TEST. There is no adaptive thresholding, validation-triggered decay, or retry loop.

The midpoint is frozen before the V3 run. The V2 result is used only to justify opening this new shadow experiment; it is not permitted to change the schedule during execution.

## Everything else remains frozen

V3 must preserve V2 exactly for:

- exact D11 checkpoint and fully frozen D11 parameters;
- glyph-adapter-only trainable surface;
- Teacher Gold admission, family split, labels, and derivative bytes;
- D10 authoritative manifest/artifact binding;
- 128 synthetic replay records per class per epoch;
- real balanced repeat factor;
- batch size 32;
- AdamW and weight decay;
- presence/digit/bbox/distillation/bbox-anchor objectives and weights;
- real augmentation bounds;
- deterministic CPU execution and seed;
- 20 total epochs;
- all acceptance gates.

## Acceptance gates — unchanged

A V3 candidate is accepted only if one epoch simultaneously satisfies all V2 gates:

- real validation macro F1 >= `0.900`;
- real validation accuracy >= `0.900`;
- real `none` recall >= `8/9`;
- real `2/4`, `3/4`, and `4/4` recall each exactly `3/3`;
- D10 validation macro-F1 drop <= `0.020` from the frozen D11 baseline;
- D10 positive localization F1@2px drop <= `0.030` from the frozen D11 baseline.

No gate may be relaxed because the validation set is small.

## Diagnostic output

V3 may add read-only explainability diagnostics, but they have no authority over training or acceptance. At minimum the final report should preserve:

- all epoch-level real/synthetic metrics and confusion matrices;
- exact gate reasons per epoch;
- the first epoch that passes both synthetic-retention gates, if any;
- the best real epoch;
- whether a joint accepted epoch exists;
- per-class real recall for the best real epoch.

A later diagnostic extension may expose per-record real-validation probabilities and predicted class identities, but it must not use those records to tune thresholds or labels within the same experiment.

## Fail-closed boundaries

The following remain false regardless of V3 outcome:

- sealed TEST access;
- runtime connection;
- Deterministic Resolver connection;
- production promotion;
- automatic D11 replacement;
- online learning;
- threshold tuning;
- silent fallback to a rejected epoch.

If no epoch passes every unchanged gate, the result remains `HOLD_NO_ACCEPTED_CANDIDATE` and no candidate checkpoint is emitted.

## Implementation gate

This contract freezes the V3 schedule before runtime integration. The scheduler helper and its deterministic tests may merge into the V3 branch only after CI passes. The actual adaptation loop must then be wired in a separate small commit and re-run in Colab as a new output directory; existing V2 run artifacts are immutable evidence and must not be overwritten.
