# Meter Real-Domain Adaptation V3-A1 Contract

## Purpose

V3-A1 is a shadow-only ablation derived from the V2 evidence. It isolates Meter classification from localization so the two remaining failure modes can be separated safely:

- positive Meter classes (`2/4` and `4/4`) being pulled into `3/4` on the 18-record real validation surface;
- D10 source forgetting despite a fully frozen D11 base.

V3-A1 is not a promotion candidate by itself and does not change production behavior.

## Frozen inputs and closed boundaries

- Base D11 checkpoint SHA-256 remains `cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3`.
- Teacher Gold remains exactly 54 TRAIN + 18 family-disjoint VALIDATION records.
- D10 TRAIN/VALIDATION identity remains unchanged.
- TEST is not enumerated or opened.
- Runtime, Resolver, checkpoint replacement, automatic learning, and production promotion remain disabled.
- V2 evidence and files remain unchanged on their parent branch.

## Trainable surface

The D11 model is fully frozen. V3-A1 trains only a bounded Meter glyph classification adapter:

```text
exact frozen D11
    + bounded Meter glyph encoder
    + presence head
    + upper-digit head (2 / 3 / 4)
    -> classification residual only
```

There is no trainable bbox head in V3-A1. The candidate bbox output is exactly the frozen D11 bbox output for every input.

## Objective

Real Teacher Gold records drive only classification objectives:

- four-class Meter cross entropy;
- presence loss (`none` vs visible Meter);
- positive upper-digit loss (`2`, `3`, `4`).

D10 replay is used only for source retention:

- logit distillation to the frozen D11 logits;
- hard residual-zero penalty that pushes the adapter classification residual toward zero on D10 records.

V3-A1 intentionally does not add positive-class margin loss. If `2/4 -> 3/4` or `4/4 -> 3/4` remains after source retention is fixed, margin/separation is reserved for V3-A2 so only one causal change is tested at a time.

## Acceptance gate

The same bounded shadow gate remains frozen:

| Gate | Requirement |
| --- | ---: |
| Real VALIDATION macro F1 | >= 0.900 |
| Real VALIDATION accuracy | >= 0.900 (therefore at least 17/18) |
| Real `none` recall | >= 8/9 |
| Real `2/4` recall | 3/3 |
| Real `3/4` recall | 3/3 |
| Real `4/4` recall | 3/3 |
| D10 macro-F1 drop | <= 0.020 |
| D10 localization-F1 drop | <= 0.030 |

Because V3-A1 always returns the frozen D11 bbox, any synthetic localization regression is a hard implementation/provenance failure, not an acceptable training trade-off.

## Decision rule

- If real classification passes and D10 retention passes: V3-A1 is a successful ablation and the next step is larger real shadow validation before any promotion discussion.
- If D10 retention passes but `2/4` or `4/4` still collapses toward `3/4`: proceed to V3-A2 positive-class separation.
- If D10 still regresses: do not widen the model; first reduce adapter capacity or strengthen residual-zero retention.
- If real classification does not improve: stop training changes and inspect ROI/feature evidence for the two failing positive classes.
