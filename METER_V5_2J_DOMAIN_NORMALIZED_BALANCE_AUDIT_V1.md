# Meter V5-2J — Domain-Normalized Loss Balance Audit V1

## Purpose

V5-2I proved that one raw mixed BCE over 540 V5 slots plus 6,480 historical replay rows is not an adequate repair objective: the 12:1 mixture remained historical-retention HOLD and almost completely suppressed V5 positive learning.

V5-2J is inference-free and gradient-free. It does not train. It uses only existing V5-2F signed-pressure JSON evidence plus the completed V5-2I JSON evidence to analyze an explicit two-domain objective:

`L_total = mean(L_V5) + lambda_source * mean(L_historical)`

Both domain means use diagnostic `pos_weight=1.0`. `lambda_source` is a loss coefficient, not a replay sample ratio.

## Evidence boundary

The audit requires:

- V5-2F replay-balance report;
- V5-2I training report showing the exact 12:1 pilot;
- V5-2I final report with overall `HOLD`, historical retention `HOLD`, and V5 diagnostic `NOT_RUN`;
- the exact observed V5-2I candidate SHA-256 values.

## Analysis

For each specialist, V5-2F already provides the signed logit-pressure means at `pos_weight=1.0`:

`signed_mean = negative_pressure_mean - positive_pressure_mean`

For the domain-normalized objective, the zero crossing is:

`lambda_source = -v5_signed_mean / historical_signed_mean`

The cross-specialist interval between the 2-AI and 3-AI zero crossings is reported as an evidence interval only.

V5-2J also computes one **minimax reference coefficient**: the shared `lambda_source` inside that interval that minimizes the maximum absolute signed residual across 2-AI and 3-AI at the frozen checkpoint. This is a mathematical reference, not a selected training setting.

The audit also evaluates the previous raw 12:1 mixture as an equivalent first-order domain coefficient of `lambda_source=12` and reports whether it lies beyond both zero crossings.

## Safety boundary

V5-2J performs no:

- training;
- backward pass;
- optimizer step;
- checkpoint read/write;
- image read;
- threshold tuning;
- BBox/crop/spatial derivation;
- reserve V5 TRAIN access;
- V5 validation access;
- FINAL_HOLDOUT access;
- 4-AI mutation;
- Resolver wiring;
- production promotion.

No domain weight, replay ratio, learning rate, epoch count, or repair training recipe is automatically selected or authorized by V5-2J.
