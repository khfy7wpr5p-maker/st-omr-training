# Meter V5-2K — Parameter-Gradient Balance Audit V1

## Purpose

V5-2K tests whether the domain-normalized objective proposed after V5-2I/V5-2J is also balanced in actual model-parameter gradient space, not only in logit-space first-order pressure.

The audit is diagnostic only. It reads exact frozen 2-AI/3-AI checkpoints and the already-approved TRAIN surfaces, computes gradients, and performs **zero optimizer steps**.

## Exact surfaces

- V5: exactly the existing 540 `adaptation_train` slots from V5-2B.
- Historical source: exactly the existing 26,964 M4A TRAIN records.
- First-30 V5 diagnostic seeds: excluded from gradients.
- 900 reserved V5 TRAIN: closed.
- V5 VALIDATION: closed.
- FINAL_HOLDOUT: locked.
- 4-AI: frozen and not opened by this audit.

Historical pixels must replay the already-frozen M4A/D10 crop/preprocess path. No new crop, BBox, slot, or spatial heuristic is permitted.

## Objective audited

For each specialist independently:

`L(lambda) = mean(V5_BCE_pos_weight_1) + lambda * mean(HISTORICAL_BCE_pos_weight_1)`

The audit computes the exact full-surface mean parameter gradients at the frozen checkpoint:

- `g_v5 = grad(mean(V5_BCE_w1))`
- `g_source = grad(mean(HISTORICAL_BCE_w1))`

No model parameter is changed.

## Required measurements

For 2-AI and 3-AI separately:

- V5/source gradient L2 norms;
- dot product and cosine similarity;
- whether the domain gradients conflict (`dot < 0`);
- non-negative coefficient minimizing parameter-gradient norm:
  `lambda* = max(0, -dot(g_v5, g_source) / ||g_source||^2)`;
- combined-gradient norms at the V5-2J references:
  - 2-AI logit zero-crossing;
  - V5-2J logit minimax reference;
  - 3-AI logit zero-crossing;
- the same diagnostics split into `features` and `head` parameter groups;
- a common cross-specialist reference coefficient minimizing the maximum normalized combined-gradient norm across 2-AI and 3-AI. This is **reference-only** and does not select a training setting.

## Safety

V5-2K authorizes gradient calculation only. It does not authorize:

- `optimizer.step()`;
- checkpoint writes;
- training epochs;
- threshold tuning;
- a selected domain coefficient;
- a second repair pilot;
- new BBoxes/crops/spatial rules;
- reserve V5 TRAIN;
- V5 VALIDATION;
- FINAL_HOLDOUT;
- 4-AI mutation;
- Resolver wiring;
- production promotion.

The report must state `training=False`, `optimizer_steps=0`, `checkpoint_write=False`, and `domain_weight_selected=False`.
