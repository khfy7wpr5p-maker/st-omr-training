# Meter Real-Domain Adaptation V3-A2 Contract

Status: **SHADOW-ONLY / DEVELOPMENT EVIDENCE**

## Purpose

V3-A2 is a single-variable follow-up to V3-A1. V3-A1 proved that freezing bbox exactly and strengthening D10 source retention removes the localization regression and keeps synthetic macro-F1 inside the allowed retention gate, but the held-out 18-record real validation surface still ended at 16/18 with two positive-class errors.

V3-A2 changes only one training mechanism: a fixed pairwise margin over the three positive Meter classes on **REAL TRAIN positive records only**.

## Frozen inherited boundary

- exact D11 checkpoint remains fully frozen;
- exact D11 bbox output is returned unchanged;
- no trainable bbox head exists;
- the V3-A1 classification adapter architecture is unchanged;
- Teacher Gold split remains TRAIN=54 / VALIDATION=18;
- D10 replay remains deterministic, 128 TRAIN records per class;
- D10 VALIDATION remains 1,224 Meter records;
- V3-A1 logit distillation remains unchanged;
- V3-A1 adapter residual-zero retention remains unchanged;
- augmentation, optimizer, learning rate, seed, batch size, and 20-epoch cardinality remain unchanged;
- sparse transport may stage only the exact 512 replay images plus all 1,224 Meter VALIDATION images; it does not change selection or evaluation;
- TEST remains sealed;
- runtime, Resolver, checkpoint replacement, and production promotion remain closed.

## Single new variable

For each REAL TRAIN positive record with true class `y in {2/4, 3/4, 4/4}`:

`L_margin = relu(2.0 - (z_y - max(z_other_positive_classes)))`

Frozen values:

- margin = `2.0` logits;
- margin loss weight = `1.0`;
- the `none` logit is excluded from this margin term;
- no margin term is computed on D10 synthetic records;
- no margin term is computed on Teacher Gold VALIDATION records;
- no validation `record_id`, `family_id`, or observed error pair may be hard-coded into the objective.

The complete V3-A2 objective is the V3-A1 objective plus this one margin term.

## Acceptance gate

Unchanged from V3-A1/V2 strict gate:

- real macro-F1 >= 0.900;
- real accuracy >= 0.900, which on 18 records requires at least 17/18;
- none recall >= 8/9;
- each positive class recall = 3/3;
- D10 synthetic macro-F1 drop <= 0.020 from the frozen D11 baseline;
- D10 positive localization F1@2px drop <= 0.030;
- because bbox is exact frozen D11 output, any observed localization change is a hard implementation/provenance failure.

## Candidate policy

A candidate checkpoint may be emitted only if one epoch passes all gates. Any emitted checkpoint is still shadow-only. A passing result does not authorize TEST access, runtime wiring, Resolver wiring, merge, checkpoint replacement, or production promotion.

## Interpretation

This experiment is allowed to answer only whether a general REAL-TRAIN positive-class separation term can improve the held-out real decision boundary while preserving the V3-A1 source-retention and exact-bbox guarantees. It must not be interpreted as an independent generalization study because the hypothesis was chosen after development validation analysis.
