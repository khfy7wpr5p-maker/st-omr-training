# Meter Real-Domain Adaptation V2 Contract

## Decision

V1 is retained as evidence but is not a promotion candidate. Its strongest
epoch improved real validation from `0.1667` to `0.6595` macro F1 and from
`0.5000` to `0.7778` accuracy, while D10 validation macro F1 fell from
`0.9089` to `0.8269`. The V1 shared-head update therefore failed both the real
quality target and the synthetic-retention boundary.

V2 changes the adaptation mechanism, not the accepted data boundary:

```text
exact audited D11 Meter model (fully frozen)
    + bounded 56:248 x 8:184 Meter search-strip adapter
    + presence decision (none / visible Meter)
    + upper-digit decision (2 / 3 / 4)
    + bounded bbox residual
    + D10 logit and bbox distillation
    -> shadow-only Meter candidate
```

## Frozen inputs and closed boundaries

- The exact D11 checkpoint SHA-256 remains
  `cd2d6192411371628518f4a8327cb0169910425494fa4a82082cd268d85254f3`.
- The accepted 72-record Teacher Gold bundle remains 54 TRAIN and 18
  family-disjoint VALIDATION records.
- D10 TRAIN/VALIDATION identity and the strict 44,260-file local cache remain
  unchanged. A complete V1 cache is reused; no second Drive copy is allowed.
- TEST is neither enumerated nor opened.
- Runtime, Resolver, frozen-checkpoint replacement, automatic learning, and
  production promotion remain false.

## Trainable surface

Every D11 parameter is frozen and verified by state hash before and after the
run. Only the new glyph adapter may update. The adapter reads a bounded Meter
search strip rather than pooling the entire measure ROI. It preserves horizontal
spatial evidence because approved `4/4` examples include right-shifted glyphs
whose mapped right edge reaches approximately x=244.

The final class logits are the frozen D11 logits plus a hierarchical residual:

- `none`: negative presence evidence;
- `2/4`, `3/4`, `4/4`: positive presence evidence plus centered 2/3/4 digit
  evidence.

The denominator is deterministically fixed to `4` because the frozen Meter
class contract contains only `2/4`, `3/4`, and `4/4`.

## Forgetting control

Each epoch contains a class-balanced real surface and 128 deterministic D10
TRAIN replay records per class. D10 samples receive both their accepted class
target and distillation anchors to the original D11 logits and boxes. This
prevents the real adapter from replacing the accepted synthetic decision
surface.

## Acceptance

A checkpoint is emitted only if one epoch satisfies all gates:

| Gate | Requirement |
| --- | ---: |
| Real VALIDATION macro F1 | at least 0.900 |
| Real VALIDATION accuracy | at least 0.900 (at least 17/18) |
| Real `none` recall | at least 8/9 |
| Each positive-class recall | exactly 3/3 |
| D10 macro-F1 drop | at most 0.020 |
| D10 localization-F1 drop | at most 0.030 |

The 18-record pilot can prove only this bounded shadow gate. It cannot establish
production-level generalization. A larger family-disjoint real evaluation set
is required before runtime connection or promotion.

## Execution and recovery

The V3 Colab notebook launches the V2 runner in a detached background process,
writes heartbeat/progress to Drive, exposes `epoch ?/20` and `batch ?/total`,
and writes a safe resume state after each complete epoch. Relaunch reuses the
existing D10 local cache and resumes only when repository, configuration, data,
base checkpoint, and baseline metrics match exactly.
