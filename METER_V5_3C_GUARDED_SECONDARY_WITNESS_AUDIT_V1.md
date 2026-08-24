# Meter V5-3C — Guarded Secondary Witness Audit v1

## Status

Preregistered diagnostic-only numerical reformulation.  This stage does not
authorize a candidate checkpoint, retention, validation, or production use.

## Bound evidence

- V5-3A source implementation: `cdc6683a556c16b00e7b154fca8e89ba5dd848b7`
- V5-3A source harness: `c2d5f1652adac52387e33b9d2f33078f864f980b`
- V5-3A report SHA256:
  `a483173353b9e425a4a3eb8d177376c15a7c5fa1d13c62689356c04b3fffd92e`
- HOLD recovery envelope SHA256:
  `6514983e886c9ba41398f2a0c1888d3088455ab612cd6ad91614bcd8d7db4d40`

The exact V5-3A 3-AI secondary witness exceeded the unchanged primary L1 cap
by `1.2146432482040836e-7`.  The fixed witness tolerance was `1e-7`; therefore
the only failing amount beyond tolerance was `2.146432482040836e-8`, or about
`5.17e-10` of the cap.  Every classification, decision-margin, parameter-bound,
functional-identity, transition, and V5 TRAIN F1 check passed.

## One fixed reformulation

V5-3C reuses the exact V5-3A primary L1 optimum and external acceptance cap.
It does not rerun the primary LP.

The secondary LP is otherwise unchanged, except for both of these fixed
numerical measures:

1. the solver-facing L1 cap is tightened by `5 * 1e-7 = 5e-7`;
2. that single cap row is divided by the tightened cap so its RHS is `1.0`.

The external acceptance cap remains `primary optimum + 1e-6`.  Independent
verification still uses the original `1e-7` witness tolerance.  Both the
tightened internal cap and unchanged external cap must pass.

## Frozen contract

- SciPy `1.18.0`, `highs-ds`, presolve off;
- primal and dual solver tolerances remain `1e-9`;
- V5 margin remains `0.25` plus the existing solver buffer;
- historical retained-margin policy is unchanged;
- primary L1 and secondary Linf objectives are unchanged;
- backbone, head bias, thresholds, 4-AI, BBox, crop geometry, and spatial
  behavior remain frozen;
- no solver sweep, fallback solver, alternate configuration, tolerance change,
  threshold search, bias search, autograd, backward, or optimizer step.

## Fail-closed output

The stage writes only
`v5_3c_guarded_secondary_witness_audit_v1.json`.  Weight values are neither
emitted nor persisted.  No checkpoint is written even when both witnesses
pass.  Historical retention, First-30, V5 VAL, and FINAL_HOLDOUT remain closed.

Any solver-status failure, residual violation, float32-copy margin failure,
input-hash mismatch, or output collision produces HOLD or a hard failure.
